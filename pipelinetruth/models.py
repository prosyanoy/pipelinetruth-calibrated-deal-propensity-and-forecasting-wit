"""Win-probability models: logistic-regression baseline vs LightGBM, with time-ordered calibration."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .calibration import CALIBRATORS, make_calibrator

NUMERIC = ["log_amount", "n_activities_snap", "n_meetings_snap", "n_contacts_snap", "days_since_last_activity",
           "stage_ord", "expected_cycle_days", "close_date_stale", "account_prior_deals", "account_prior_wins",
           "owner_prior_deals", "owner_prior_win_rate", "days_to_quarter_end"]
CATEGORICAL = ["segment", "region", "industry", "lead_source", "created_qtr"]
FEATURES = NUMERIC + CATEGORICAL
MODEL_KINDS = ("logreg", "lgbm")


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs; missing columns become NaN so partial API payloads still score."""
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    for c in CATEGORICAL:
        X[c] = df[c].astype("object").where(df[c].notna(), "Unknown").astype(str) if c in df else "Unknown"
    return X


def _preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        # scaler goes after the imputer; indicator columns are scaled too (harmless for a linear model)
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def _estimator(kind: str, seed: int):
    if kind == "logreg":
        return LogisticRegression(C=0.5, max_iter=3000)
    if kind == "lgbm":
        return LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                              random_state=seed, verbose=-1, n_jobs=1)
    raise ValueError(f"unknown model kind {kind!r}")


@dataclass
class ModelBundle:
    kind: str
    preprocessor: ColumnTransformer
    estimator: object
    calibrators: dict
    calibration: str = "isotonic"
    trained_through: pd.Timestamp | None = None
    n_fit: int = 0
    n_cal: int = 0
    base_rate: float = 0.0
    feature_sources: list = field(default_factory=list)
    # reference data used by drift checks, the support check and typical-value reason text
    reference: pd.DataFrame | None = None
    ranges: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    medians: dict = field(default_factory=dict)
    version: str = "dev"

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.preprocessor.transform(prepare_X(df)))[:, 1]

    def predict_proba(self, df: pd.DataFrame, calibration: str | None = None) -> np.ndarray:
        return self.calibrators[calibration or self.calibration].predict(self.raw_proba(df))

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additive log-odds contribution of each source feature (SHAP for LightGBM, coef*x for logistic)."""
        Z = self.preprocessor.transform(prepare_X(df))
        if self.kind == "lgbm":
            contrib = self.estimator.booster_.predict(Z, pred_contrib=True)[:, :-1]
        else:
            contrib = Z * self.estimator.coef_[0]
        out = pd.DataFrame(0.0, index=df.index, columns=FEATURES)
        for j, src in enumerate(self.feature_sources):
            out[src] += contrib[:, j]
        return out


def _feature_sources(pre: ColumnTransformer) -> list[str]:
    srcs = []
    for name in pre.get_feature_names_out():
        base = name.split("__", 1)[1]
        base = base.replace("missingindicator_", "")
        if base in NUMERIC:
            srcs.append(base)
        else:
            srcs.append(next(c for c in CATEGORICAL if base.startswith(c + "_")))
    return srcs


def fit_bundle(train: pd.DataFrame, kind: str = "lgbm", calibration: str = "isotonic", seed: int = 42,
               cal_frac: float = 0.25) -> ModelBundle:
    """Fit on the earlier part of ``train`` and fit calibrators on its *later* part (time-ordered, never random).

    ``train`` must already be restricted to rows whose outcome was known at the training cutoff.
    """
    train = train.sort_values("snapshot_date").reset_index(drop=True)
    n_cal = max(int(len(train) * cal_frac), 1)
    fit_df, cal_df = train.iloc[:-n_cal], train.iloc[-n_cal:]
    pre = _preprocessor(scale=(kind == "logreg"))
    est = _estimator(kind, seed)
    Xf = pre.fit_transform(prepare_X(fit_df))
    est.fit(Xf, fit_df["won"].astype(int))
    p_cal = est.predict_proba(pre.transform(prepare_X(cal_df)))[:, 1]
    cals = {k: make_calibrator(k).fit(p_cal, cal_df["won"].astype(int).to_numpy()) for k in CALIBRATORS}
    X = prepare_X(train)
    return ModelBundle(
        kind=kind, preprocessor=pre, estimator=est, calibrators=cals, calibration=calibration,
        trained_through=train["snapshot_date"].max(), n_fit=len(fit_df), n_cal=len(cal_df),
        base_rate=float(train["won"].mean()), feature_sources=_feature_sources(pre),
        reference=X.sample(min(len(X), 3000), random_state=seed).reset_index(drop=True),
        ranges={c: (float(X[c].quantile(0.005)), float(X[c].quantile(0.995))) for c in NUMERIC if X[c].notna().any()},
        categories={c: sorted(X[c].unique().tolist()) for c in CATEGORICAL},
        medians={c: float(X[c].median()) for c in NUMERIC if X[c].notna().any()},
    )
