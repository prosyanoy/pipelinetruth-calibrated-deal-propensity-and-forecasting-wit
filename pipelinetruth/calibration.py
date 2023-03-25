"""Calibration: Platt / isotonic calibrators, proper scoring metrics and reliability diagrams."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class IdentityCalibrator:
    name = "none"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


class PlattCalibrator:
    """Logistic regression on the logit of the raw probability (sigmoid calibration)."""
    name = "platt"

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Monotone, non-parametric calibration. Needs more data than Platt but fixes any shape of miscalibration."""
    name = "isotonic"

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(np.asarray(p, float), y)
        return self

    def predict(self, p):
        return self.iso_.predict(np.asarray(p, float))


CALIBRATORS = {"none": IdentityCalibrator, "platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


def make_calibrator(kind: str):
    return CALIBRATORS[kind]()


def reliability_table(y, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Per-bin mean prediction vs observed frequency."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    rows = []
    for b in np.unique(ids):
        m = ids == b
        rows.append({"bin": int(b), "n": int(m.sum()), "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = reliability_table(y, p, n_bins, strategy="uniform")
    return float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["n"].sum())


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    out = {"n": int(len(y)), "base_rate": float(y.mean()), "mean_pred": float(p.mean()),
           "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p)}
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1])) if len(np.unique(y)) > 1 else float("nan")
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return out


def brier_skill(y, p, reference_rate: float) -> float:
    """1 - Brier / Brier(constant reference_rate). > 0 means better than always predicting the base rate."""
    y = np.asarray(y, float)
    ref = float(np.mean((y - reference_rate) ** 2))
    return float(1 - brier_score_loss(y, np.clip(p, EPS, 1 - EPS)) / ref) if ref > 0 else float("nan")


def plot_reliability(curves: dict[str, pd.DataFrame], path, title: str = "Reliability diagram (out-of-time)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect")
    for label, t in curves.items():
        ax.plot(t["mean_pred"], t["frac_pos"], marker="o", lw=1.6, ms=4, label=label)
    ax.set_xlabel("mean predicted win probability")
    ax.set_ylabel("observed win frequency")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
