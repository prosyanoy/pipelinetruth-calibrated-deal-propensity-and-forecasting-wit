"""Hand-rolled drift checks: Population Stability Index for numeric and categorical features and scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_WARN, PSI_ALERT = 0.10, 0.25
# Features whose distribution moves by construction, so PSI against the training window is a false alarm:
# cumulative history counts/rates grow as the CRM ages, and creation quarter is seasonal.
DRIFT_EXEMPT = {"account_prior_deals": "cumulative", "account_prior_wins": "cumulative", "owner_prior_deals": "cumulative",
                "owner_prior_win_rate": "history-derived", "created_qtr": "seasonal"}


def psi_status(v: float) -> str:
    return "alert" if v >= PSI_ALERT else "warn" if v >= PSI_WARN else "ok"


def _psi_from_counts(ref: np.ndarray, cur: np.ndarray, eps: float = 1e-4) -> float:
    r = np.maximum(ref / max(ref.sum(), 1), eps)
    c = np.maximum(cur / max(cur.sum(), 1), eps)
    return float(((c - r) * np.log(c / r)).sum())


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Quantile bins from the reference; missing values form their own bucket."""
    ref, cur = pd.to_numeric(ref, errors="coerce"), pd.to_numeric(cur, errors="coerce")
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1))) if ref.notna().any() else np.array([0.0, 1.0])
    if len(edges) < 3:  # near-constant feature: bucket on equality instead
        rc = np.array([(ref == ref.median()).sum(), (ref != ref.median()).sum() - ref.isna().sum(), ref.isna().sum()], float)
        cc = np.array([(cur == ref.median()).sum(), (cur != ref.median()).sum() - cur.isna().sum(), cur.isna().sum()], float)
        return _psi_from_counts(rc, cc)
    edges[0], edges[-1] = -np.inf, np.inf
    rc = np.append(np.histogram(ref.dropna(), edges)[0], ref.isna().sum()).astype(float)
    cc = np.append(np.histogram(cur.dropna(), edges)[0], cur.isna().sum()).astype(float)
    return _psi_from_counts(rc, cc)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    rc = ref.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(float)
    cc = cur.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(float)
    return _psi_from_counts(rc, cc)


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, numeric: list[str], categorical: list[str],
                 ref_scores=None, cur_scores=None, min_rows: int = 50) -> pd.DataFrame:
    rows = []
    for c in numeric:
        rows.append({"feature": c, "kind": "numeric", "psi": psi_numeric(ref[c], cur[c])})
    for c in categorical:
        rows.append({"feature": c, "kind": "categorical", "psi": psi_categorical(ref[c], cur[c])})
    if ref_scores is not None and cur_scores is not None:
        rows.append({"feature": "model_score", "kind": "score",
                     "psi": psi_numeric(pd.Series(ref_scores), pd.Series(cur_scores))})
    out = pd.DataFrame(rows)
    out["status"] = out["psi"].map(psi_status)
    out["n_current"] = len(cur)
    if len(cur) < min_rows:
        out["status"] = "insufficient_data"
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def overall_status(report: pd.DataFrame) -> str:
    s = set(report["status"])
    if "insufficient_data" in s:
        return "unknown"
    return "alert" if "alert" in s else "warn" if "warn" in s else "ok"
