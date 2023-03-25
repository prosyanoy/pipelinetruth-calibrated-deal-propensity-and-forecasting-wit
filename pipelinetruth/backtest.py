"""Rolling-origin, time-based backtesting. Random splits are never used.

For a test quarter starting at T:
  * train = deals whose *outcome was already known* at T (closed_date < T), i.e. what we would really have had;
  * test  = deals snapshotted inside the quarter.
Every (model, calibrator) pair is scored on the same folds, so model choice is a measured decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import brier_skill, metrics
from .models import MODEL_KINDS, ModelBundle, fit_bundle

CALIBRATION_KINDS = ("none", "platt", "isotonic")


def quarter_bounds(q: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    return q.start_time.normalize(), q.end_time.normalize()


def labeled(feat: pd.DataFrame) -> pd.DataFrame:
    return feat[(~feat["dq_error"]) & feat["won"].notna()]


def train_test_split_at(feat: pd.DataFrame, q: pd.Period) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end = quarter_bounds(q)
    lab = labeled(feat)
    train = lab[(lab["closed_date"] < start) & (lab["snapshot_date"] < start)]
    test = lab[(lab["snapshot_date"] >= start) & (lab["snapshot_date"] <= end)]
    return train, test


@dataclass
class BacktestResult:
    folds: list
    n_warmup: int
    n_selection: int
    metrics: pd.DataFrame            # one row per fold x model x calibration (+ baseline)
    preds: pd.DataFrame              # long out-of-time predictions
    bundles: dict = field(default_factory=dict)  # (fold, kind) -> ModelBundle
    chosen_model: str = ""
    chosen_calibration: str = ""
    selection_table: pd.DataFrame | None = None

    @property
    def reported_folds(self) -> list:
        return self.folds[self.n_warmup:]

    @property
    def holdout_folds(self) -> list:
        return self.folds[self.n_warmup + self.n_selection:]


def run_backtest(feat: pd.DataFrame, folds: list[pd.Period], n_warmup: int = 2, n_selection: int = 4,
                 kinds=MODEL_KINDS, seed: int = 42) -> BacktestResult:
    """Fit/score all (model, calibration) pairs per fold, then choose using only the selection folds.

    The first ``n_warmup`` folds only exist to build forecast-residual history; the following ``n_selection``
    folds choose the model/calibrator; the remaining folds are held out from that choice.
    """
    rows, pred_rows, bundles = [], [], {}
    for q in folds:
        train, test = train_test_split_at(feat, q)
        if len(train) < 200 or len(test) < 30:
            raise ValueError(f"fold {q}: not enough data (train={len(train)}, test={len(test)})")
        y = test["won"].astype(int).to_numpy()
        base = metrics(y, np.full(len(y), train["won"].mean()))
        rows.append({"fold": str(q), "model": "base_rate", "calibration": "-", **base,
                     "brier_skill": 0.0, "n_train": len(train)})
        for kind in kinds:
            b = fit_bundle(train, kind=kind, seed=seed)
            bundles[(str(q), kind)] = b
            raw = b.raw_proba(test)
            for cal in CALIBRATION_KINDS:
                p = b.calibrators[cal].predict(raw)
                rows.append({"fold": str(q), "model": kind, "calibration": cal, **metrics(y, p),
                             "brier_skill": brier_skill(y, p, train["won"].mean()), "n_train": len(train)})
                pred_rows.append(pd.DataFrame({"fold": str(q), "model": kind, "calibration": cal,
                                               "opp_id": test["opp_id"].to_numpy(), "y": y, "p": p}))
    m = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    sel_folds = [str(q) for q in folds[n_warmup:n_warmup + n_selection]]
    sel = (m[(m["fold"].isin(sel_folds)) & (m["model"] != "base_rate")]
           .groupby(["model", "calibration"])[["brier", "ece", "auc", "log_loss"]].mean().sort_values("brier"))
    best = sel.index[0]
    return BacktestResult(folds=[str(q) for q in folds], n_warmup=n_warmup, n_selection=n_selection, metrics=m,
                          preds=preds, bundles=bundles, chosen_model=best[0], chosen_calibration=best[1],
                          selection_table=sel.reset_index())


def chosen_predictions(res: BacktestResult, feat: pd.DataFrame, reported_only: bool = True) -> pd.DataFrame:
    p = res.preds[(res.preds["model"] == res.chosen_model) & (res.preds["calibration"] == res.chosen_calibration)]
    if reported_only:
        p = p[p["fold"].isin(res.reported_folds)]
    cols = ["opp_id", "segment", "lead_source", "region", "amount", "created_date", "snapshot_date", "dq_warnings"]
    return p.merge(feat[cols], on="opp_id", how="left")
