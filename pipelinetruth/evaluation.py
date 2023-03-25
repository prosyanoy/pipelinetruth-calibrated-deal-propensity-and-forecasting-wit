"""Segment-level performance and the data-driven 'known failure modes' / 'do not use when' sections."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import metrics

MIN_SEGMENT_N = 150
MAX_GAP = 0.05
MIN_AUC = 0.60


def segment_report(preds: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, d in preds.groupby(by, observed=True):
        m = metrics(d["y"], d["p"])
        gap = m["mean_pred"] - m["base_rate"]
        flags = []
        if m["n"] < MIN_SEGMENT_N:
            flags.append("low_n")
        if abs(gap) > MAX_GAP:
            flags.append("miscalibrated")
        if m["auc"] < MIN_AUC:
            flags.append("weak_ranking")
        rows.append({by: key, **m, "calib_gap": gap, "flags": ",".join(flags)})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def add_amount_band(preds: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 10_000, 50_000, 150_000, np.inf]
    labels = ["<10k", "10-50k", "50-150k", ">150k"]
    return preds.assign(amount_band=pd.cut(preds["amount"], bins=bins, labels=labels))


def caution_segments(seg_report: pd.DataFrame, col: str = "segment") -> set:
    return {r[col] for _, r in seg_report.iterrows() if r["flags"]}


def failure_modes(bt, seg_reports: dict, cov: dict, drift_status: str, drift_top: pd.DataFrame, dq: dict,
                  preds: pd.DataFrame) -> list[dict]:
    """Each entry is evidence-backed from this run; static design limits are appended last."""
    out = []
    m = bt.metrics
    chosen = m[(m["model"] == bt.chosen_model) & (m["calibration"] == bt.chosen_calibration)].set_index("fold")
    sel = [f for f in bt.folds[bt.n_warmup:bt.n_warmup + bt.n_selection]]
    hold = bt.holdout_folds
    if hold:
        b_sel, b_hold = chosen.loc[sel, "brier"].mean(), chosen.loc[hold, "brier"].mean()
        base_sel = m[(m["model"] == "base_rate") & m["fold"].isin(sel)]["brier"].mean()
        base_hold = m[(m["model"] == "base_rate") & m["fold"].isin(hold)]["brier"].mean()
        s_sel, s_hold = 1 - b_sel / base_sel, 1 - b_hold / base_hold
        e_sel, e_hold = chosen.loc[sel, "ece"].mean(), chosen.loc[hold, "ece"].mean()
        if e_hold > e_sel * 1.2 or s_hold < s_sel - 0.03:
            out.append({"mode": "Performance degrades on the most recent quarters",
                        "evidence": f"Brier skill {s_sel:.3f} on selection folds vs {s_hold:.3f} on the latest {len(hold)} folds; "
                                    f"ECE {e_sel:.3f} -> {e_hold:.3f}. Recent data contains a regime change the training window barely saw."})
    skill = float(1 - chosen.loc[bt.reported_folds, "brier"].mean() / m[(m["model"] == "base_rate") & m["fold"].isin(bt.reported_folds)]["brier"].mean())
    auc = float(chosen.loc[bt.reported_folds, "auc"].mean())
    out.append({"mode": "Modest discrimination",
                "evidence": f"Mean out-of-time AUC {auc:.3f}, Brier skill vs base rate {skill:.3f}. Scores help rank deals but most outcomes "
                            "are not predictable from day-14 CRM data; do not read single probabilities as near-certain."})
    for name, rep in seg_reports.items():
        key = rep.columns[0]
        for _, r in rep[rep["flags"] != ""].iterrows():
            out.append({"mode": f"{key} = {r[key]}: {r['flags'].replace(',', ', ')}",
                        "evidence": f"n={int(r['n'])}, observed win rate {r['base_rate']:.3f} vs mean predicted {r['mean_pred']:.3f}, "
                                    f"AUC {r['auc']:.3f}, ECE {r['ece']:.3f}"})
    t = cov["total"]
    if t["n"] and t["coverage"] < cov["nominal"]:
        out.append({"mode": "Forecast intervals under-cover",
                    "evidence": f"Nominal {cov['nominal']:.0%} intervals covered {t['covered']}/{t['n']} quarters at the total level "
                                f"({t['coverage']:.0%}, 95% Wilson CI {t['wilson_lo']:.0%}-{t['wilson_hi']:.0%}) and "
                                f"{cov['segment']['covered']}/{cov['segment']['n']} segment cells. Few quarters means this estimate is itself very uncertain."})
    out.append({"mode": "Forecast ignores deals not yet in the pipeline",
                "evidence": f"On average {cov['mean_unforecast_share']:.0%} of the revenue booked in a quarter came from deals that were "
                            "not scoreable on day 1 of that quarter. The forecast is a pipeline-yield forecast, not a total-bookings forecast."})
    if drift_status in ("warn", "alert"):
        top = drift_top.head(3)
        out.append({"mode": f"Input drift ({drift_status})",
                    "evidence": "Largest PSI: " + ", ".join(f"{r.feature}={r.psi:.2f}" for r in top.itertuples())})
    if dq["n_quarantined"]:
        out.append({"mode": "Records quarantined by data-quality errors",
                    "evidence": f"{dq['n_quarantined']} of {dq['n_opps']} opportunities ({dq['n_quarantined'] / dq['n_opps']:.1%}) fail an "
                                "error-severity rule and are excluded from training, scoring and forecasts; they receive no score."})
    w = preds.assign(has_warn=preds["dq_warnings"].fillna(0) > 0)
    if w["has_warn"].nunique() == 2:
        a = w[w["has_warn"]]
        b = w[~w["has_warn"]]
        out.append({"mode": "Warning-flagged records",
                    "evidence": f"Deals with DQ warnings (n={len(a)}): Brier {metrics(a['y'], a['p'])['brier']:.3f} vs "
                                f"{metrics(b['y'], b['p'])['brier']:.3f} for clean deals (n={len(b)})."})
    out += [
        {"mode": "Scores freeze the day-14 snapshot",
         "evidence": "Features are as of created_date + snapshot days. A deal that changed materially later is not re-scored with new information."},
        {"mode": "Right-censored recent training labels",
         "evidence": "Training only uses deals whose outcome was known at the cutoff; the newest cohort over-represents fast-closing deals."},
        {"mode": "Synthetic data", "evidence": "Results illustrate the method on a simulated CRM; absolute numbers will not transfer to real data."},
    ]
    return out


def do_not_use(cfg, bundle, seg_reports: dict, drift_status: str, cov: dict) -> list[str]:
    items = [
        f"The deal is younger than {cfg.snapshot_days} days: no snapshot exists, so no score is produced.",
        "The record fails a data-quality error rule (missing or non-positive amount, close date before creation, unknown account): it is rejected, not scored.",
        "A feature lies outside the training range or a category is unseen (flagged `out_of_range` / `unseen_category`): the model is extrapolating.",
    ]
    lo, hi = bundle.ranges.get("log_amount", (None, None))
    if hi:
        items.append(f"Deal amount is above ~${np.exp(hi):,.0f} or below ~${np.exp(lo):,.0f} (0.5th-99.5th percentile of training deals).")
    seg = seg_reports.get("segment")
    if seg is not None:
        for _, r in seg[seg["flags"] != ""].iterrows():
            items.append(f"Segment **{r['segment']}** ({r['flags'].replace(',', ', ')}, n={int(r['n'])} out-of-time): treat scores as directional only.")
    if drift_status in ("warn", "alert"):
        items.append(f"Input drift status is **{drift_status}**: re-check calibration on fresh outcomes before relying on probabilities.")
    t = cov["total"]
    if t["n"] and t["coverage"] < cov["nominal"]:
        items.append("You need a guaranteed range: the quarterly forecast interval has under-covered in backtests; use it as a rough guide only.")
    items += [
        "You need to forecast total bookings: the forecast excludes deals created in the last snapshot window and in-quarter new pipeline.",
        "Making an individual-level decision (compensation, performance management): probabilities are about deals and calibrated in aggregate only.",
        "The sales process, pricing, CRM logging practice or lead sources change materially: the model has no knowledge of the new regime.",
    ]
    return items
