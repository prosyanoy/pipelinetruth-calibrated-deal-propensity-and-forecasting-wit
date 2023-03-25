import numpy as np
import pandas as pd

from pipelinetruth.drift import drift_report, overall_status, psi_categorical, psi_numeric
from pipelinetruth.forecast import (build_pipeline, coverage_summary, p_close_in_quarter, simulate_interval,
                                    timing_model, widening_factor)


def test_simulated_interval_brackets_mean_and_matches_variance():
    rng = np.random.default_rng(0)
    amount = np.full(200, 1000.0)
    pi = np.full(200, 0.3)
    mean, sd, lo, hi = simulate_interval(amount, pi, 0.8, rng)
    assert mean == 60000.0 and lo < mean < hi
    assert abs(sd - 1000 * np.sqrt(200 * 0.3 * 0.7)) / sd < 0.05


def test_widening_factor_rules():
    k, src = widening_factor([0.1, -0.2], 0.8)
    assert k == 1.0 and "uncalibrated" in src
    k, _ = widening_factor([2.5, -3.0, 2.8, -2.2, 3.1], 0.8)
    assert k > 1.5
    k, _ = widening_factor([0.1, -0.1, 0.2, 0.05], 0.8)
    assert k == 1.0  # never narrows below the model-based interval


def test_p_close_in_quarter_is_a_probability_and_grows_with_window(feat):
    T = pd.Timestamp("2024-01-01")
    tm = timing_model(feat, T)
    age = np.array([20, 20, 60, 200])
    seg = np.array(["SMB", "Enterprise", "MidMarket", "SMB"])
    short = p_close_in_quarter(age, seg, 10, tm)
    long = p_close_in_quarter(age, seg, 90, tm)
    assert ((0 <= short) & (short <= 1)).all() and (long >= short).all()
    assert short[0] > short[1]  # SMB closes faster than Enterprise


def test_pipeline_only_contains_open_clean_deals(feat):
    T = pd.Timestamp("2024-04-01")
    p = build_pipeline(feat, T)
    assert not p["dq_error"].any()
    assert (p["snapshot_date"] <= T).all() and (p["closed_date"].isna() | (p["closed_date"] >= T)).all()


def test_coverage_summary_wilson_and_counts():
    rows = []
    for i, cov in enumerate([True, True, False, True, True, False]):
        rows.append({"quarter": f"q{i}", "cell": "ALL", "forecast": 100.0, "lo": 90, "hi": 110, "realized": 100.0, "covered": cov,
                     "warmup": False, "baseline_forecast": 90.0, "unforecast_share": 0.3})
    rows += [{"quarter": "q0", "cell": "SMB", "forecast": 1.0, "lo": 0, "hi": 2, "realized": 1.0, "covered": True,
              "warmup": False, "baseline_forecast": np.nan, "unforecast_share": np.nan}]
    cov = coverage_summary(pd.DataFrame(rows), 0.8)
    assert cov["total"]["covered"] == 4 and cov["total"]["n"] == 6
    assert cov["total"]["wilson_lo"] < 4 / 6 < cov["total"]["wilson_hi"]
    assert abs(cov["mape_baseline"] - 0.1) < 1e-9


def test_psi_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(0)
    a, b = pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, b) < 0.05
    assert psi_numeric(a, pd.Series(rng.normal(1.5, 1, 5000))) > 0.25
    c1 = pd.Series(["a"] * 800 + ["b"] * 200)
    assert psi_categorical(c1, pd.Series(["a"] * 200 + ["b"] * 800)) > 0.25
    assert psi_categorical(c1, c1) < 1e-9


def test_drift_report_status_and_insufficient_data():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000), "c": rng.choice(["a", "b"], 2000)})
    cur = pd.DataFrame({"x": rng.normal(2, 1, 500), "c": rng.choice(["a", "b"], 500)})
    r = drift_report(ref, cur, ["x"], ["c"])
    assert r.iloc[0]["feature"] == "x" and overall_status(r) == "alert"
    assert overall_status(drift_report(ref, cur.head(10), ["x"], ["c"])) == "unknown"
