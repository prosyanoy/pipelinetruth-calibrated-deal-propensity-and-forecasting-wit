"""Scoring with reason codes and trust flags, shared by the batch priorities table and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import CATEGORICAL, NUMERIC, ModelBundle, prepare_X

LABELS = {
    "log_amount": "Deal size", "n_activities_snap": "Activity volume", "n_meetings_snap": "Meetings held",
    "n_contacts_snap": "Contacts engaged", "days_since_last_activity": "Days since last touch",
    "stage_ord": "Pipeline stage", "expected_cycle_days": "Expected sales cycle (days)",
    "close_date_stale": "Stale CRM close date", "account_prior_deals": "Account's prior deals",
    "account_prior_wins": "Account's prior wins", "owner_prior_deals": "Owner's closed-deal history",
    "owner_prior_win_rate": "Owner's historical win rate", "days_to_quarter_end": "Days to quarter end",
    "segment": "Segment", "region": "Region", "industry": "Industry", "lead_source": "Lead source",
    "created_qtr": "Creation quarter",
}


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def reason_codes(contrib_row: pd.Series, raw_row: pd.Series, top: int = 3, min_abs: float = 0.05) -> list[dict]:
    """Top additive log-odds drivers. Direction is relative to the average deal the model was trained on."""
    items = contrib_row.reindex(contrib_row.abs().sort_values(ascending=False).index)
    out = []
    for feat, c in items.items():
        if abs(c) < min_abs or len(out) >= top:
            continue
        v = raw_row.get("amount") if feat == "log_amount" else raw_row.get(feat)
        out.append({"feature": feat, "label": LABELS[feat], "value": _fmt(v), "direction": "up" if c > 0 else "down",
                    "log_odds": round(float(c), 3),
                    "text": f"{LABELS[feat]} = {_fmt(v)} {'raises' if c > 0 else 'lowers'} win odds"})
    return out


def trust_flags(bundle: ModelBundle, X: pd.DataFrame, dq_rules: pd.Series | None = None,
                caution_segments: set | None = None, drift_status: str = "ok") -> list[list[str]]:
    """Per-row list of reasons the score deserves less trust (empty list = no known concern)."""
    flags: list[list[str]] = [[] for _ in range(len(X))]
    for c in NUMERIC:
        if c in bundle.ranges:
            lo, hi = bundle.ranges[c]
            bad = X[c].notna() & ((X[c] < lo) | (X[c] > hi))
            for i in np.flatnonzero(bad.to_numpy()):
                flags[i].append(f"out_of_range:{c}")
    for c in CATEGORICAL:
        known = set(bundle.categories.get(c, []))
        for i in np.flatnonzero((~X[c].isin(known | {"Unknown"})).to_numpy()):
            flags[i].append(f"unseen_category:{c}")
    if dq_rules is not None:
        for i, rules in enumerate(dq_rules.fillna("").to_numpy()):
            flags[i].extend(f"dq:{r}" for r in str(rules).split(",") if r)
    if caution_segments:
        for i, s in enumerate(X["segment"].to_numpy()):
            if s in caution_segments:
                flags[i].append(f"segment_low_confidence:{s}")
    if drift_status == "alert":
        for f in flags:
            f.append("model_drift_alert")
    return flags


def score_frame(bundle: ModelBundle, df: pd.DataFrame, top_reasons: int = 3, caution_segments: set | None = None,
                drift_status: str = "ok", dq_rules_col: str | None = "dq_warning_rules") -> pd.DataFrame:
    """Calibrated probability + reason codes + trust flags for each row of ``df``."""
    df = df.reset_index(drop=True)
    X = prepare_X(df)
    p = bundle.predict_proba(df)
    contrib = bundle.contributions(df)
    dq = df[dq_rules_col] if dq_rules_col and dq_rules_col in df else None
    flags = trust_flags(bundle, X, dq, caution_segments, drift_status)
    reasons = [reason_codes(contrib.iloc[i], df.iloc[i], top_reasons) for i in range(len(df))]
    return pd.DataFrame({"p_win": p, "reasons": reasons, "trust_flags": flags,
                         "trust": ["caution" if f else "ok" for f in flags]})
