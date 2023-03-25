import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipelinetruth.api import create_app


def test_pipeline_artifacts_and_model_card(small_cfg, pipeline_out):
    ad = small_cfg.artifact_dir
    for f in ("model_card.md", "priorities.csv", "priorities.md", "reliability.png", "forecast_backtest.png", "model/bundle.joblib"):
        assert (ad / f).exists(), f
    card = (ad / "model_card.md").read_text()
    for section in ("## Do not use this model when...", "## Known failure modes", "## Performance by segment",
                    "## Quarterly revenue forecast", "## Drift status", "## Data quality", "### Calibration"):
        assert section in card
    assert pipeline_out["meta"]["chosen_model"] in ("logreg", "lgbm")
    assert 0 < pipeline_out["meta"]["headline"]["brier"] < 0.25


def test_priorities_are_ranked_with_reasons(pipeline_out):
    pr = pipeline_out["priorities"]
    assert len(pr) > 0
    assert pr["expected_value"].is_monotonic_decreasing and pr["rank"].tolist() == list(range(1, len(pr) + 1))
    assert (pr["reason_codes"].str.len() > 0).mean() > 0.9
    assert pr["p_win"].between(0, 1).all() and set(pr["trust"]) <= {"ok", "caution"}
    assert (pr["age_days"] >= 14).all()


def test_forecast_intervals_are_ordered_and_coverage_is_reported(pipeline_out):
    fc = pipeline_out["forecast_backtest"]
    assert (fc["lo"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["hi"]).all()
    cov = pipeline_out["coverage"]
    assert cov["total"]["n"] == int((~fc[fc["cell"] == "ALL"]["warmup"]).sum())
    nxt = pipeline_out["next_forecast"]
    assert nxt.iloc[0]["cell"] == "ALL" and nxt.iloc[0]["lo"] <= nxt.iloc[0]["forecast"] <= nxt.iloc[0]["hi"]


@pytest.fixture()
def client(small_cfg, pipeline_out):
    with TestClient(create_app(small_cfg)) as c:
        yield c


def test_api_health_and_model_card(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"]
    assert "Do not use this model when" in client.get("/model-card").text


GOOD = {"opp_id": "A", "amount": 25000, "segment": "MidMarket", "lead_source": "Referral", "region": "NA", "industry": "Software",
        "n_activities_snap": 12, "n_meetings_snap": 3, "n_contacts_snap": 3, "stage_ord": 3, "created_qtr": 2}


def test_api_score_returns_probability_reasons_and_rejections(client):
    body = {"deals": [GOOD,
                      {"opp_id": "B", "amount": None, "segment": "SMB"},
                      {"opp_id": "C", "amount": 9000, "age_days": 3},
                      dict(GOOD, opp_id="D", amount=5e8)]}
    r = client.post("/score", json=body)
    assert r.status_code == 200
    res = {x["opp_id"]: x for x in r.json()["results"]}
    assert 0 < res["A"]["p_win"] < 1 and 1 <= len(res["A"]["reason_codes"]) <= 3
    assert res["B"]["status"] == "rejected" and res["B"]["p_win"] is None and res["B"]["trust"] == "do_not_use"
    assert res["C"]["status"] == "rejected" and "too new" in res["C"]["trust_flags"][0]
    assert res["D"]["trust"] == "caution" and "out_of_range:log_amount" in res["D"]["trust_flags"]


def test_api_better_engagement_scores_higher(client):
    lo = dict(GOOD, opp_id="lo", n_activities_snap=1, n_meetings_snap=0, n_contacts_snap=1, stage_ord=1)
    hi = dict(GOOD, opp_id="hi", n_activities_snap=25, n_meetings_snap=8, n_contacts_snap=5, stage_ord=4)
    res = {x["opp_id"]: x for x in client.post("/score", json={"deals": [lo, hi]}).json()["results"]}
    assert res["hi"]["p_win"] > res["lo"]["p_win"]


def test_api_validation_and_priorities(client):
    assert client.post("/score", json={"deals": []}).status_code == 422
    assert client.post("/score", json={"deals": [{"amount": 1, "bogus": 1}]}).status_code == 422
    p = client.get("/priorities?top=5").json()
    assert len(p["deals"]) == 5 and p["deals"][0]["rank"] == 1


def test_api_without_model_returns_503(tmp_path):
    from pipelinetruth.config import Config
    with TestClient(create_app(Config(data_dir=tmp_path, artifact_dir=tmp_path))) as c:
        assert c.get("/health").json()["status"] == "no_model"
        assert c.post("/score", json={"deals": [{"amount": 1}]}).status_code == 503
