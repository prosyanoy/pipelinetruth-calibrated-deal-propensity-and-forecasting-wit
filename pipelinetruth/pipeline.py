"""End-to-end run: generate -> SQL transforms + DQ -> backtest -> calibrate -> forecast -> drift -> model card."""
from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .backtest import chosen_predictions, labeled, run_backtest
from .calibration import metrics, plot_reliability, reliability_table
from .config import Config
from .drift import DRIFT_EXEMPT, drift_report, overall_status
from .evaluation import add_amount_band, caution_segments, do_not_use, failure_modes, segment_report
from .forecast import coverage_summary, forecast_next_quarter, plot_forecast_backtest, run_forecast_backtest
from .models import CATEGORICAL, NUMERIC, fit_bundle, prepare_X
from .modelcard import md_table, render
from .scoring import score_frame
from .synth import generate, write_raw
from .warehouse import build_warehouse, dq_report, load_features

log = logging.getLogger("pipelinetruth")


def backtest_folds(cfg: Config) -> list[pd.Period]:
    first = pd.Period(cfg.start_date + pd.Timedelta(days=365), freq="Q")
    last = pd.Period(cfg.as_of - pd.Timedelta(days=cfg.max_cycle_days), freq="Q")
    if last.end_time.normalize() + pd.Timedelta(days=cfg.max_cycle_days) > cfg.as_of:
        last -= 1  # a quarter is only a fair test if every deal in it has had time to resolve
    return list(pd.period_range(first, last, freq="Q"))


def build_priorities(feat: pd.DataFrame, bundle, cfg: Config, caution: set, drift_status: str) -> tuple[pd.DataFrame, dict]:
    """Open, DQ-clean deals past their snapshot, ranked by expected value, each with reason codes."""
    open_ = feat[feat["status"] == "open"]
    too_new = open_[open_["snapshot_date"] > cfg.as_of]
    rejected = open_[open_["dq_error"]]
    cand = open_[(~open_["dq_error"]) & (open_["snapshot_date"] <= cfg.as_of)].reset_index(drop=True)
    sc = score_frame(bundle, cand, caution_segments=caution, drift_status=drift_status)
    out = pd.concat([cand[["opp_id", "account_id", "segment", "stage", "amount", "created_date", "dq_warning_rules"]], sc], axis=1)
    out["age_days"] = (cfg.as_of - out["created_date"]).dt.days
    out["expected_value"] = out["p_win"] * out["amount"]
    out["reason_codes"] = out["reasons"].map(lambda rs: "; ".join(r["text"] for r in rs))
    out["trust_notes"] = out["trust_flags"].map(lambda f: ", ".join(f))
    out = out.sort_values("expected_value", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    cols = ["rank", "opp_id", "account_id", "segment", "stage", "amount", "age_days", "p_win", "expected_value",
            "reason_codes", "trust", "trust_notes"]
    return out[cols], {"scored": len(out), "too_new": len(too_new), "rejected_dq": len(rejected)}


def priorities_markdown(pr: pd.DataFrame, stats: dict, top: int = 25) -> str:
    t = pr.head(top)[["rank", "opp_id", "segment", "stage", "amount", "p_win", "expected_value", "reason_codes", "trust"]].copy()
    t["amount"] = t["amount"].map("${:,.0f}".format)
    t["expected_value"] = t["expected_value"].map("${:,.0f}".format)
    t["p_win"] = t["p_win"].map("{:.0%}".format)
    return (f"# Prioritized open deals (top {top})\n\nScored {stats['scored']} deals; {stats['too_new']} too new for a snapshot; "
            f"{stats['rejected_dq']} rejected by data-quality errors. Ranked by expected value (calibrated P(win) x amount). "
            f"`trust = caution` means a trust flag applies (see priorities.csv).\n\n{md_table(t)}\n")


def run(cfg: Config | None = None, regenerate: bool = True, use_mlflow: bool = True) -> dict:
    cfg = cfg or Config()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    if regenerate or not (cfg.raw_dir / "opportunities.csv").exists():
        log.info("generating synthetic CRM (n=%d, seed=%d)", cfg.n_opps, cfg.seed)
        write_raw(generate(cfg), cfg.raw_dir)
    log.info("running SQL transforms and data-quality checks")
    con = build_warehouse(cfg)
    feat = load_features(con)
    dq = dq_report(con)
    con.close()

    log.info("time-based backtest")
    bt = run_backtest(feat, backtest_folds(cfg), seed=cfg.seed)
    preds = chosen_predictions(bt, feat)
    preds = add_amount_band(preds)
    seg_reports = {"segment": segment_report(preds, "segment"), "lead_source": segment_report(preds.fillna({"lead_source": "Unknown"}), "lead_source"),
                   "amount_band": segment_report(preds, "amount_band")}
    caution = caution_segments(seg_reports["segment"])

    log.info("forecast backtest")
    fc_bt = run_forecast_backtest(feat, bt, cfg.interval_level)
    cov = coverage_summary(fc_bt, cfg.interval_level)

    log.info("fitting final model on all outcomes known at %s", cfg.as_of.date())
    T = cfg.as_of + pd.Timedelta(days=1)
    lab = labeled(feat)
    train = lab[(lab["closed_date"] < T) & (lab["snapshot_date"] < T)]
    bundle = fit_bundle(train, kind=bt.chosen_model, calibration=bt.chosen_calibration, seed=cfg.seed)
    bundle.version = f"{bt.chosen_model}-{bt.chosen_calibration}-{cfg.as_of:%Y%m%d}"
    next_fc = forecast_next_quarter(feat, bundle, bt.chosen_calibration, cfg.as_of, fc_bt, cfg.interval_level)

    log.info("drift checks")
    window = feat[(~feat["dq_error"]) & (feat["snapshot_date"] > cfg.as_of - pd.Timedelta(days=90)) & (feat["snapshot_date"] <= cfg.as_of)]
    cur_X = prepare_X(window)
    drift = drift_report(bundle.reference, cur_X, [c for c in NUMERIC if c not in DRIFT_EXEMPT],
                         [c for c in CATEGORICAL if c not in DRIFT_EXEMPT],
                         bundle.predict_proba(bundle.reference), bundle.predict_proba(window))
    drift_status = overall_status(drift)

    pr, pr_stats = build_priorities(feat, bundle, cfg, caution, drift_status)

    log.info("writing artifacts")
    ad = cfg.artifact_dir
    rel_curves = {"chosen (calibrated)": reliability_table(preds["y"], preds["p"], 10)}
    raw_preds = bt.preds[(bt.preds["model"] == bt.chosen_model) & (bt.preds["calibration"] == "none") & bt.preds["fold"].isin(bt.reported_folds)]
    rel_curves["chosen (uncalibrated)"] = reliability_table(raw_preds["y"], raw_preds["p"], 10)
    plot_reliability(rel_curves, ad / "reliability.png")
    plot_forecast_backtest(fc_bt, ad / "forecast_backtest.png", cfg.interval_level)

    ch = bt.metrics[(bt.metrics["model"] == bt.chosen_model) & (bt.metrics["calibration"] == bt.chosen_calibration) & bt.metrics["fold"].isin(bt.reported_folds)]
    base = bt.metrics[(bt.metrics["model"] == "base_rate") & bt.metrics["fold"].isin(bt.reported_folds)]
    headline = {"brier": float(ch["brier"].mean()), "base_brier": float(base["brier"].mean()), "ece": float(ch["ece"].mean()),
                "auc": float(ch["auc"].mean())}
    headline["skill"] = 1 - headline["brier"] / headline["base_brier"]
    rel_tbl = reliability_table(preds["y"], preds["p"], 10).rename(columns={"frac_pos": "observed_win_rate"})
    run_id = uuid.uuid4().hex[:12]
    ctx = {"bt": bt, "cfg": cfg, "fc_bt": fc_bt, "cov": cov, "next_fc": next_fc, "dq": dq, "drift": drift, "drift_status": drift_status,
           "n_drift_current": len(window), "seg_reports": seg_reports, "headline": headline, "reliability": rel_tbl,
           "n_train": len(train), "run_id": run_id, "version": bundle.version,
           "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "failure_modes": failure_modes(bt, seg_reports, cov, drift_status, drift, dq, preds),
           "do_not_use": do_not_use(cfg, bundle, seg_reports, drift_status, cov)}
    card = render(ctx)
    (ad / "model_card.md").write_text(card)
    pr.to_csv(ad / "priorities.csv", index=False)
    (ad / "priorities.md").write_text(priorities_markdown(pr, pr_stats))
    bt.metrics.to_csv(ad / "backtest_metrics.csv", index=False)
    fc_bt.to_csv(ad / "forecast_backtest.csv", index=False)
    next_fc.to_csv(ad / "forecast_next_quarter.csv", index=False)
    joblib.dump(bundle, cfg.model_dir / "bundle.joblib")
    meta = {"version": bundle.version, "run_id": run_id, "package_version": __version__, "chosen_model": bt.chosen_model,
            "chosen_calibration": bt.chosen_calibration, "trained_through": str(bundle.trained_through.date()),
            "snapshot_days": cfg.snapshot_days, "drift_status": drift_status, "caution_segments": sorted(caution),
            "headline": headline, "interval_coverage": {k: cov[k] for k in ("nominal", "total", "segment")},
            "generated": ctx["generated"]}
    (cfg.model_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if use_mlflow:
        try:
            _log_mlflow(cfg, meta, ctx, ad)
        except Exception as exc:  # tracking is observability, it must not break the pipeline
            log.warning("MLflow logging failed: %s", exc)
    return {"meta": meta, "priorities": pr, "backtest": bt, "forecast_backtest": fc_bt, "coverage": cov, "next_forecast": next_fc,
            "drift": drift, "dq": dq, "model_card": card}


def _log_mlflow(cfg: Config, meta: dict, ctx: dict, ad) -> None:
    import mlflow
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment("pipelinetruth")
    with mlflow.start_run(run_name=meta["version"]):
        mlflow.log_params({"model": meta["chosen_model"], "calibration": meta["chosen_calibration"], "seed": cfg.seed,
                           "n_opps": cfg.n_opps, "snapshot_days": cfg.snapshot_days, "interval_level": cfg.interval_level,
                           "trained_through": meta["trained_through"]})
        h, cov = ctx["headline"], ctx["cov"]
        mlflow.log_metrics({"oot_brier": h["brier"], "oot_brier_skill": h["skill"], "oot_ece": h["ece"], "oot_auc": h["auc"],
                            "interval_coverage_total": cov["total"]["coverage"], "interval_coverage_segment": cov["segment"]["coverage"],
                            "forecast_mape": cov["mape_model"], "forecast_mape_baseline": cov["mape_baseline"],
                            "n_quarantined": ctx["dq"]["n_quarantined"], "max_psi": float(ctx["drift"]["psi"].max())})
        mlflow.set_tag("drift_status", meta["drift_status"])
        for f in ("model_card.md", "reliability.png", "forecast_backtest.png", "priorities.csv", "backtest_metrics.csv"):
            mlflow.log_artifact(str(ad / f))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the full PipelineTruth pipeline")
    ap.add_argument("--n-opps", type=int, help="number of synthetic opportunities (default: PT_N_OPPS or 7000)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--reuse-data", action="store_true", help="skip regeneration if raw CSVs exist")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--demo", action="store_true", help="after the run, print the top deals and open the model card")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    base = Config()
    cfg = Config(n_opps=a.n_opps or base.n_opps, seed=a.seed if a.seed is not None else base.seed)
    out = run(cfg, regenerate=not a.reuse_data, use_mlflow=not a.no_mlflow)
    m = out["meta"]
    print(f"\nmodel={m['chosen_model']}+{m['chosen_calibration']} drift={m['drift_status']} "
          f"brier={m['headline']['brier']:.3f} skill={m['headline']['skill']:.3f}")
    print(f"artifacts: {cfg.artifact_dir}/model_card.md, priorities.md, priorities.csv")
    if a.demo:
        _demo(cfg, out)


def _demo(cfg: Config, out: dict) -> None:
    """Print the headline numbers and top deals, then try to open the model card (HTML) in a browser."""
    cov = out["coverage"]["total"]
    print(f"\nForecast interval coverage: nominal {out['coverage']['nominal']:.0%}, measured {cov['covered']}/{cov['n']} quarters")
    print("\nTop 10 deals:\n" + out["priorities"].head(10)[["rank", "opp_id", "segment", "amount", "p_win", "expected_value", "reason_codes", "trust"]]
          .to_string(index=False, max_colwidth=60, formatters={"amount": "${:,.0f}".format, "p_win": "{:.0%}".format, "expected_value": "${:,.0f}".format}))
    card = (cfg.artifact_dir / "model_card.md").resolve()
    html = card.with_suffix(".html")
    import base64
    import html as h
    body = h.escape(out["model_card"])
    for img in ("reliability.png", "forecast_backtest.png"):
        data = base64.b64encode((cfg.artifact_dir / img).read_bytes()).decode()
        body = body.replace(f"![{'Reliability diagram' if img.startswith('rel') else 'Forecast backtest'}]({img})",
                            f'</pre><img src="data:image/png;base64,{data}" width="520"><pre>')
    html.write_text(f"<html><body><pre style='white-space:pre-wrap;font-family:monospace'>{body}</pre></body></html>")
    print(f"\nModel card: {card}\nHTML view:  {html}")
    try:
        import webbrowser
        webbrowser.open(html.as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()
