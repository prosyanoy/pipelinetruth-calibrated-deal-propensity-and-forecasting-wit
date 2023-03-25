"""Quarterly revenue forecast with prediction intervals whose coverage is measured in a backtest.

Forecast for quarter [T, E] made at T:  sum_i amount_i * P(win_i) * P(close in quarter | win_i, age_i).
The win probability is the calibrated model score; the timing term comes from the empirical cycle
lengths of *matured* won deals of the same segment (created long enough ago that all have closed),
which avoids right-censoring bias. Deal outcomes are treated as independent Bernoulli draws
(Monte Carlo) to get a base interval, which is then widened by a factor learned only from the
standardised errors of *earlier* quarters, because independence and model error both make the
naive interval too narrow. Cells with little history say so (``k_source``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import BacktestResult, quarter_bounds
from .models import ModelBundle

N_SIMS = 4000


def build_pipeline(feat: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    """Deals that are open at T, DQ-clean and already past their snapshot (so features are known)."""
    m = (~feat["dq_error"]) & (feat["snapshot_date"] <= T) & (feat["closed_date"].isna() | (feat["closed_date"] >= T))
    return feat[m & (feat["created_date"] < T)].copy()


def timing_model(feat: pd.DataFrame, T: pd.Timestamp, max_cycle: int = 180):
    """Empirical cycle lengths (days) of matured won deals, by segment, as known at T."""
    won = feat[(feat["won"] == 1) & (~feat["dq_error"]) & (feat["closed_date"] < T)
               & (feat["created_date"] <= T - pd.Timedelta(days=max_cycle))]
    cyc = (won["closed_date"] - won["created_date"]).dt.days
    by_seg = {s: np.sort(cyc[won["segment"] == s].to_numpy()) for s in won["segment"].unique()}
    return by_seg, np.sort(cyc.to_numpy())


def p_close_in_quarter(age: np.ndarray, segment: np.ndarray, days_left: int, timing, min_n: int = 15) -> np.ndarray:
    by_seg, pooled = timing
    out = np.ones(len(age))
    for s in np.unique(segment):
        arr = by_seg.get(s, np.array([]))
        if len(arr) < min_n:
            arr = pooled
        if len(arr) < min_n:
            continue
        m = segment == s
        a = age[m]
        denom = len(arr) - np.searchsorted(arr, a, side="left")           # cycle >= age
        numer = np.searchsorted(arr, a + days_left, side="right") - np.searchsorted(arr, a, side="left")
        out[m] = np.where(denom >= min_n, numer / np.maximum(denom, 1), 1.0)
    return out


def simulate_interval(amount: np.ndarray, pi: np.ndarray, level: float, rng: np.random.Generator):
    """Monte Carlo sum of independent amount*Bernoulli(pi). Returns mean, sd, lo, hi."""
    if len(amount) == 0:
        return 0.0, 0.0, 0.0, 0.0
    draws = (rng.random((N_SIMS, len(amount))) < pi) @ amount
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return float((amount * pi).sum()), float(draws.std()), float(lo), float(hi)


def widening_factor(z_hist: list[float], level: float, min_hist: int = 3) -> tuple[float, str]:
    """k >= 1 such that the level-quantile of past |z| would have been covered."""
    if len(z_hist) < min_hist:
        return 1.0, f"uncalibrated (only {len(z_hist)} prior points)"
    k = float(np.quantile(np.abs(z_hist), level) / norm.ppf((1 + level) / 2))
    return max(1.0, k), f"learned from {len(z_hist)} prior points"


def _cell_forecast(df: pd.DataFrame, level: float, k: float, rng) -> dict:
    mean, sd, lo, hi = simulate_interval(df["amount"].to_numpy(), df["pi"].to_numpy(), level, rng)
    return {"forecast": mean, "sd": sd, "lo": max(0.0, mean - k * (mean - lo)), "hi": mean + k * (hi - mean),
            "pipeline_amount": float(df["amount"].sum()), "n_deals": int(len(df))}


def score_pipeline(feat: pd.DataFrame, bundle: ModelBundle, calibration: str, T: pd.Timestamp, E: pd.Timestamp) -> pd.DataFrame:
    pipe = build_pipeline(feat, T)
    if pipe.empty:
        return pipe.assign(p_win=[], p_close=[], pi=[])
    pipe["p_win"] = bundle.predict_proba(pipe, calibration)
    age = (T - pipe["created_date"]).dt.days.to_numpy()
    pipe["p_close"] = p_close_in_quarter(age, pipe["segment"].to_numpy(), (E - T).days + 1, timing_model(feat, T))
    pipe["pi"] = pipe["p_win"] * pipe["p_close"]
    return pipe


def run_forecast_backtest(feat: pd.DataFrame, res: BacktestResult, level: float = 0.8, seed: int = 7) -> pd.DataFrame:
    """Forecast each backtest quarter at its first day and compare with what was actually booked."""
    rng = np.random.default_rng(seed)
    rows = []
    hist_total: list[float] = []
    hist_seg: list[float] = []
    prior_pipe_amt = prior_real = 0.0
    for i, f in enumerate(res.folds):
        q = pd.Period(f)
        T, E = quarter_bounds(q)
        bundle = res.bundles[(f, res.chosen_model)]
        pipe = score_pipeline(feat, bundle, res.chosen_calibration, T, E)
        pipe["won_in_q"] = ((pipe["won"] == 1) & (pipe["closed_date"] >= T) & (pipe["closed_date"] <= E)).astype(float)
        all_won = feat[(~feat["dq_error"]) & (feat["won"] == 1) & (feat["closed_date"] >= T) & (feat["closed_date"] <= E)]
        booked_total = float(all_won["amount"].sum())
        cells = [("ALL", pipe)] + [(s, pipe[pipe["segment"] == s]) for s in sorted(feat["segment"].unique())]
        new_z_total, new_z_seg = [], []
        for name, d in cells:
            hist = hist_total if name == "ALL" else hist_seg
            k, k_src = widening_factor(hist, level)
            r = _cell_forecast(d, level, k, rng)
            r["realized"] = float((d["amount"] * d["won_in_q"]).sum())
            z = (r["realized"] - r["forecast"]) / r["sd"] if r["sd"] > 0 else 0.0
            ratio = prior_real / prior_pipe_amt if prior_pipe_amt > 0 else bundle.base_rate
            baseline = r["pipeline_amount"] * (ratio if name == "ALL" else bundle.base_rate)
            rows.append({"quarter": f, "cell": name, **r, "z": z, "k": k, "k_source": k_src,
                         "covered": bool(r["lo"] <= r["realized"] <= r["hi"]),
                         "baseline_forecast": baseline if name == "ALL" else np.nan,
                         "warmup": i < res.n_warmup,
                         "booked_total_in_quarter": booked_total if name == "ALL" else np.nan,
                         "unforecast_share": 1 - r["realized"] / booked_total if name == "ALL" and booked_total > 0 else np.nan})
            (new_z_total if name == "ALL" else new_z_seg).append(z)
            if name == "ALL":
                prior_pipe_amt += r["pipeline_amount"]
                prior_real += r["realized"]
        # realized values of this quarter become known only after it ends -> usable from the next fold on
        hist_total += new_z_total
        hist_seg += new_z_seg
    out = pd.DataFrame(rows)
    out.attrs["level"] = level
    out.attrs["final_k"] = {"ALL": widening_factor(hist_total, level), "segment": widening_factor(hist_seg, level)}
    return out


def coverage_summary(fc: pd.DataFrame, level: float) -> dict:
    rep = fc[~fc["warmup"]]
    out = {"nominal": level}
    for label, d in (("total", rep[rep["cell"] == "ALL"]), ("segment", rep[rep["cell"] != "ALL"])):
        n, c = len(d), int(d["covered"].sum())
        # Wilson interval for the coverage estimate itself: with a handful of quarters it is very wide
        z = 1.96
        ph = c / n if n else float("nan")
        if n:
            den = 1 + z * z / n
            ctr = (ph + z * z / (2 * n)) / den
            half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
            wl, wh = ctr - half, ctr + half
        else:
            wl = wh = float("nan")
        out[label] = {"n": n, "covered": c, "coverage": ph, "wilson_lo": float(wl), "wilson_hi": float(wh)}
    tot = rep[rep["cell"] == "ALL"]
    out["mape_model"] = float((abs(tot["forecast"] - tot["realized"]) / tot["realized"]).mean())
    out["mape_baseline"] = float((abs(tot["baseline_forecast"] - tot["realized"]) / tot["realized"]).mean())
    out["bias_pct"] = float(((tot["forecast"] - tot["realized"]) / tot["realized"]).mean())
    out["mean_unforecast_share"] = float(tot["unforecast_share"].mean())
    return out


def forecast_next_quarter(feat: pd.DataFrame, bundle: ModelBundle, calibration: str, as_of: pd.Timestamp,
                          fc_backtest: pd.DataFrame, level: float = 0.8, seed: int = 11) -> pd.DataFrame:
    """Forecast the quarter after ``as_of`` using the widening factors learned over the whole backtest."""
    rng = np.random.default_rng(seed)
    T = (as_of + pd.Timedelta(days=1)).normalize()
    q = pd.Period(T, freq="Q")
    _, E = quarter_bounds(q)
    pipe = score_pipeline(feat, bundle, calibration, T, E)
    ks = fc_backtest.attrs["final_k"]
    rows = []
    for name, d in [("ALL", pipe)] + [(s, pipe[pipe["segment"] == s]) for s in sorted(feat["segment"].unique())]:
        k, k_src = ks["ALL"] if name == "ALL" else ks["segment"]
        rows.append({"quarter": str(q), "cell": name, **_cell_forecast(d, level, k, rng), "k": k, "k_source": k_src})
    return pd.DataFrame(rows)


def plot_forecast_backtest(fc: pd.DataFrame, path, level: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = fc[(fc["cell"] == "ALL")].reset_index(drop=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=110)
    ax.errorbar(x, d["forecast"] / 1e6, yerr=[(d["forecast"] - d["lo"]) / 1e6, (d["hi"] - d["forecast"]) / 1e6],
                fmt="o", capsize=4, label=f"forecast + {int(level * 100)}% interval")
    ax.plot(x, d["realized"] / 1e6, "D", color="#d1495b", label="realized")
    ax.set_xticks(x)
    ax.set_xticklabels(d["quarter"], rotation=30, fontsize=8)
    ax.set_ylabel("booked revenue from open pipeline ($M)")
    n_warm = int(d["warmup"].sum())
    scored = d[~d["warmup"]]
    ax.set_title(f"Forecast vs realized (day-1 forecasts); {int(level * 100)}% interval covered "
                 f"{int(scored['covered'].sum())}/{len(scored)} scored quarters", fontsize=10)
    if n_warm:
        ax.axvspan(-0.5, n_warm - 0.5, color="#ddd", alpha=0.5, label="warm-up (not scored)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
