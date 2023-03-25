import numpy as np

from pipelinetruth.calibration import (IsotonicCalibrator, PlattCalibrator, brier_skill, expected_calibration_error,
                                       metrics, reliability_table)


def _overconfident(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < true_p).astype(int)
    logit = np.log(true_p / (1 - true_p)) * 2.2  # overconfident scores
    return 1 / (1 + np.exp(-logit)), y


def test_calibrators_reduce_brier_and_ece_out_of_sample():
    p, y = _overconfident()
    p_tr, y_tr, p_te, y_te = p[:3000], y[:3000], p[3000:], y[3000:]
    base = metrics(y_te, p_te)
    for cal in (PlattCalibrator(), IsotonicCalibrator()):
        out = cal.fit(p_tr, y_tr).predict(p_te)
        m = metrics(y_te, out)
        assert m["brier"] < base["brier"]
        assert m["ece"] < base["ece"]
        assert m["ece"] < 0.04


def test_isotonic_is_monotone_and_bounded():
    p, y = _overconfident()
    out = IsotonicCalibrator().fit(p, y).predict(np.linspace(0, 1, 200))
    assert np.all(np.diff(out) >= -1e-12) and out.min() > 0 and out.max() < 1


def test_reliability_table_counts_and_perfect_calibration():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < p).astype(int)
    t = reliability_table(y, p, 10)
    assert t["n"].sum() == 20000
    assert (t["mean_pred"] - t["frac_pos"]).abs().max() < 0.05
    assert expected_calibration_error(y, p) < 0.02


def test_brier_skill_against_base_rate():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    assert brier_skill(y, np.full(8, y.mean()), y.mean()) == 0 or abs(brier_skill(y, np.full(8, y.mean()), y.mean())) < 1e-9
    assert brier_skill(y, y.astype(float) * 0.9 + 0.05, y.mean()) > 0.8
