import numpy as np
import pandas as pd
import pytest

from pipelinetruth.backtest import labeled, quarter_bounds, run_backtest, train_test_split_at
from pipelinetruth.models import FEATURES, fit_bundle, prepare_X


@pytest.mark.parametrize("q", ["2023Q2", "2023Q4", "2024Q2"])
def test_time_split_never_leaks(feat, q):
    q = pd.Period(q)
    start, end = quarter_bounds(q)
    train, test = train_test_split_at(feat, q)
    assert train["closed_date"].max() < start          # every training label was known before the test quarter
    assert train["snapshot_date"].max() < start
    assert test["snapshot_date"].min() >= start and test["snapshot_date"].max() <= end
    assert not set(train["opp_id"]) & set(test["opp_id"])
    assert not train["dq_error"].any() and not test["dq_error"].any()


def test_backtest_selects_using_selection_folds_only(feat):
    folds = list(pd.period_range("2023Q1", "2024Q2", freq="Q"))
    res = run_backtest(feat, folds, n_warmup=1, n_selection=3, kinds=("logreg", "lgbm"))
    assert res.holdout_folds == res.folds[4:]  # 6 folds: 1 warm-up, 3 selection, 2 held out
    m = res.metrics
    assert {"base_rate", "logreg", "lgbm"} == set(m["model"])
    sel = m[m["fold"].isin(res.folds[1:4]) & (m["model"] != "base_rate")].groupby(["model", "calibration"])["brier"].mean()
    assert (res.chosen_model, res.chosen_calibration) == sel.idxmin()
    assert res.preds["p"].between(0, 1).all()


@pytest.mark.parametrize("kind", ["logreg", "lgbm"])
def test_bundle_is_better_than_base_rate_and_contributions_add_up(feat, kind):
    lab = labeled(feat)
    train = lab[lab["closed_date"] < "2024-01-01"]
    test = lab[(lab["snapshot_date"] >= "2024-01-01") & (lab["snapshot_date"] < "2024-04-01")]
    b = fit_bundle(train, kind=kind)
    p = b.predict_proba(test)
    y = test["won"].astype(int).to_numpy()
    assert np.mean((p - y) ** 2) < np.mean((train["won"].mean() - y) ** 2)
    # calibrators are fit on a later time slice than the model
    assert b.n_cal > 0 and b.n_fit + b.n_cal == len(train)
    # additive explanation: contributions + bias reproduce the raw log-odds
    c = b.contributions(test.head(50))
    assert list(c.columns) == FEATURES
    raw = b.raw_proba(test.head(50))
    logit = np.log(raw / (1 - raw))
    bias = (b.estimator.intercept_[0] if kind == "logreg" else
            b.estimator.booster_.predict(b.preprocessor.transform(prepare_X(test.head(50))), pred_contrib=True)[:, -1])
    assert np.allclose(c.sum(axis=1).to_numpy() + bias, logit, atol=1e-6)


def test_bundle_scores_partial_payloads(feat):
    lab = labeled(feat)
    b = fit_bundle(lab[lab["closed_date"] < "2024-01-01"], kind="lgbm")
    p = b.predict_proba(pd.DataFrame([{"segment": "SMB"}]))
    assert 0 < p[0] < 1
