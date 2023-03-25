"""Auto-generated model card (Markdown), rebuilt from scratch on every run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .drift import DRIFT_EXEMPT


def md_table(df: pd.DataFrame, floatfmt: dict | None = None, default: str = "{:.3f}") -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.astype(object).iterrows():  # object dtype keeps ints from being upcast to float
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)) and not isinstance(v, bool):
                cells.append("" if np.isnan(v) else floatfmt.get(c, default).format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _money(v: float) -> str:
    return f"${v:,.0f}"


def render(ctx: dict) -> str:
    bt, cfg = ctx["bt"], ctx["cfg"]
    m = bt.metrics
    rep = m[m["fold"].isin(bt.reported_folds)]
    summary = (rep.groupby(["model", "calibration"])[["brier", "brier_skill", "ece", "log_loss", "auc"]].mean()
               .reset_index().sort_values("brier"))
    fold_tbl = m[(m["model"] == bt.chosen_model) & (m["calibration"] == bt.chosen_calibration)][
        ["fold", "n", "base_rate", "mean_pred", "brier", "brier_skill", "ece", "auc"]].copy()
    fold_tbl["role"] = ["warm-up" if f in bt.folds[:bt.n_warmup] else "selection" if f in bt.folds[bt.n_warmup:bt.n_warmup + bt.n_selection]
                        else "held out from selection" for f in fold_tbl["fold"]]
    fc = ctx["fc_bt"]
    fct = fc[fc["cell"] == "ALL"][["quarter", "forecast", "lo", "hi", "realized", "covered", "k", "baseline_forecast", "warmup"]].copy()
    for c in ("forecast", "lo", "hi", "realized", "baseline_forecast"):
        fct[c] = fct[c].map(_money)
    fct["covered"] = fct["covered"].map({True: "yes", False: "NO"})
    fct["role"] = fct["warmup"].map({True: "warm-up (not scored)", False: "scored"})
    fct = fct.drop(columns="warmup")
    cov = ctx["cov"]
    lvl = f"{cov['nominal']:.0%}"
    nxt = ctx["next_fc"].copy()
    nxt_tbl = pd.DataFrame({"cell": nxt["cell"], "open deals": nxt["n_deals"], "pipeline $": nxt["pipeline_amount"].map(_money),
                            "expected $": nxt["forecast"].map(_money),
                            f"{lvl} interval": [f"{_money(a)} - {_money(b)}" for a, b in zip(nxt["lo"], nxt["hi"])],
                            "interval widening k": nxt["k"].map("{:.2f}".format)})
    dq = ctx["dq"]
    dq_rules = dq["rules"].copy()
    dq_rules["pct_records"] = dq_rules["pct_records"].map("{:.2f}%".format)
    drift = ctx["drift"].head(8)
    exempt = ", ".join(f"`{k}` ({v})" for k, v in DRIFT_EXEMPT.items())
    seg_blocks = []
    for name, rep_ in ctx["seg_reports"].items():
        t = rep_[[rep_.columns[0], "n", "base_rate", "mean_pred", "calib_gap", "brier", "ece", "auc", "flags"]]
        seg_blocks.append(f"**By {rep_.columns[0].replace('_', ' ')}**\n\n" + md_table(t, {"calib_gap": "{:+.3f}"}))
    fm = "\n".join(f"- **{f['mode']}** - {f['evidence']}" for f in ctx["failure_modes"])
    dnu = "\n".join(f"- {x}" for x in ctx["do_not_use"])
    sel = bt.selection_table.copy()
    t_cov, s_cov = cov["total"], cov["segment"]
    return f"""# Model card - PipelineTruth win-probability and revenue forecast

*Generated {ctx['generated']} - run `{ctx['run_id']}` - model version `{ctx['version']}`*

## Summary
- **Task:** probability that an open B2B deal is won, scored {cfg.snapshot_days} days after creation, plus a quarterly revenue forecast from the open pipeline.
- **Chosen model:** `{bt.chosen_model}` with `{bt.chosen_calibration}` calibration (chosen by mean Brier on the {bt.n_selection} selection folds only).
- **Data:** synthetic CRM, {dq['n_opps']:,} opportunities created {cfg.start_date.date()} to {cfg.as_of.date()}; trained on {ctx['n_train']:,} deals with outcomes known by {cfg.as_of.date()}.
- **Out-of-time performance ({len(bt.reported_folds)} scored quarterly folds):** Brier {ctx['headline']['brier']:.3f} (base-rate Brier {ctx['headline']['base_brier']:.3f}, skill {ctx['headline']['skill']:.3f}), ECE {ctx['headline']['ece']:.3f}, AUC {ctx['headline']['auc']:.3f}.
- **Drift status:** **{ctx['drift_status'].upper()}**.
- **Forecast interval coverage:** nominal {lvl}, measured {t_cov['covered']}/{t_cov['n']} quarters (total) and {s_cov['covered']}/{s_cov['n']} segment-quarters.

## Do not use this model when...
{dnu}

## Known failure modes
{fm}

## Evaluation protocol
Rolling-origin backtest, one fold per quarter ({bt.folds[0]} to {bt.folds[-1]}). For a fold starting at T the model is trained only on deals whose outcome was known before T and tested on deals snapshotted inside the quarter. Within the training window the *latest* 25% (by time) fits the calibrator and the earlier 75% fits the model. The first {bt.n_warmup} folds only seed forecast-error history; the next {bt.n_selection} choose the model; the remaining folds were not used for any choice.

### Model x calibration (mean over scored folds)
{md_table(summary)}

Selection-fold ranking used to choose the model:

{md_table(sel)}

### Chosen configuration by fold
{md_table(fold_tbl)}

### Calibration
![Reliability diagram](reliability.png)

Reliability bins (chosen configuration, scored folds pooled):

{md_table(ctx['reliability'])}

## Performance by segment (out-of-time, chosen configuration)
Flags: `low_n` (n < 150), `miscalibrated` (|mean predicted - observed| > 0.05), `weak_ranking` (AUC < 0.60).

{chr(10).join(seg_blocks)}

## Quarterly revenue forecast
Forecast made on day 1 of each quarter from the open, scoreable pipeline: sum of amount x P(win) x P(close in quarter). Intervals are Monte Carlo with a widening factor `k` learned from earlier quarters' errors only.

![Forecast backtest](forecast_backtest.png)

{md_table(fct)}

- {lvl} interval coverage, total: **{t_cov['covered']}/{t_cov['n']} ({t_cov['coverage']:.0%})**, 95% Wilson CI for the true coverage {t_cov['wilson_lo']:.0%}-{t_cov['wilson_hi']:.0%}.
- {lvl} interval coverage, segment cells: **{s_cov['covered']}/{s_cov['n']} ({s_cov['coverage']:.0%})**, CI {s_cov['wilson_lo']:.0%}-{s_cov['wilson_hi']:.0%}.
- MAPE of model forecast {cov['mape_model']:.1%} vs {cov['mape_baseline']:.1%} for the baseline (pipeline amount x trailing realized/pipeline ratio). Mean signed bias {cov['bias_pct']:+.1%}.
- On average {cov['mean_unforecast_share']:.0%} of booked revenue came from deals outside the day-1 scoreable pipeline.

### Forecast for the next quarter ({ctx['next_fc']['quarter'].iloc[0]})
{md_table(nxt_tbl)}

## Data quality
{dq['n_quarantined']} of {dq['n_opps']} opportunities ({dq['n_quarantined'] / dq['n_opps']:.1%}) are quarantined by error-severity rules.

{md_table(dq_rules)}

Table-level checks:

{md_table(dq['table_checks'], default="{:,.0f}")}

## Drift status: {ctx['drift_status'].upper()}
Reference = training features; current = {ctx['n_drift_current']} deals snapshotted in the last 90 days before {cfg.as_of.date()}. PSI < 0.10 ok, 0.10-0.25 warn, > 0.25 alert. Not monitored with PSI because they drift by construction: {exempt}.

{md_table(drift[['feature', 'kind', 'psi', 'status']])}

## Intended use
Ranking and triaging open deals for rep attention, and giving finance a pipeline-yield estimate with an honest uncertainty range. Every score ships with up to three reason codes (additive log-odds contributions) and trust flags.
"""
