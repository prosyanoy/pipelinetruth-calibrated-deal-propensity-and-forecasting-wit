import numpy as np
import pandas as pd

from pipelinetruth.config import Config
from pipelinetruth.models import FEATURES
from pipelinetruth.synth import generate, write_raw
from pipelinetruth.warehouse import build_warehouse, dq_report, load_features


def test_generator_is_deterministic(small_cfg):
    a = generate(small_cfg, n_opps=300)["opportunities"]
    b = generate(small_cfg, n_opps=300)["opportunities"]
    pd.testing.assert_frame_equal(a, b)


def test_generator_injects_expected_mess(raw_tables):
    o = raw_tables["opportunities"]
    assert o["amount"].isna().any() and (o["amount"] <= 0).any()
    assert o["lead_source"].isna().any() and o["expected_close_date"].isna().any()
    assert o["opp_id"].duplicated().any()
    assert (o["closed_date"] < o["created_date"]).any()
    assert set(o["status"]) == {"won", "lost", "open"}
    acts = raw_tables["activities"]
    assert (~acts["opp_id"].isin(o["opp_id"])).any()  # orphans


def test_duplicate_accounts_share_a_canonical_key(warehouse):
    d = warehouse.execute("""SELECT account_key, count(*) c FROM stg_accounts GROUP BY 1 HAVING c > 1""").df()
    assert len(d) > 0
    names = warehouse.execute(f"SELECT account_name FROM stg_accounts WHERE account_key = '{d.account_key.iloc[0]}'").df()
    assert names["account_name"].nunique() == 2  # different spellings, one key


def test_dq_flags_bad_records_and_quarantines_errors(warehouse, feat):
    rep = dq_report(warehouse)
    rules = set(rep["rules"]["rule"])
    assert {"missing_amount", "nonpositive_amount", "closed_before_created", "stale_close_date", "duplicate_account"} <= rules
    assert rep["n_quarantined"] == int(feat["dq_error"].sum()) > 0
    bad = feat[feat["amount"].isna() | (feat["amount"] <= 0)]
    assert bad["dq_error"].all()
    assert (rep["table_checks"].set_index("check_name").loc["duplicate_opportunity_rows", "n_records"]) > 0


def test_one_feature_row_per_unique_opportunity(raw_tables, feat):
    assert feat["opp_id"].is_unique
    assert len(feat) == raw_tables["opportunities"]["opp_id"].nunique()


def test_features_are_point_in_time(small_cfg, raw_tables, feat, tmp_path):
    """Deleting every activity/stage event after the snapshot must not change any feature."""
    snap = feat.set_index("opp_id")["snapshot_date"]
    acts = raw_tables["activities"]
    keep = acts["opp_id"].map(snap).notna() & (pd.to_datetime(acts["activity_date"]) <= acts["opp_id"].map(snap))
    stages = raw_tables["stage_history"]
    keep_s = pd.to_datetime(stages["entered_date"]) <= stages["opp_id"].map(snap)
    assert (~keep).sum() > 1000  # the test is meaningful: lots of post-snapshot events exist
    tables = dict(raw_tables, activities=acts[keep | acts["opp_id"].map(snap).isna()], stage_history=stages[keep_s])
    write_raw(tables, tmp_path)
    con = build_warehouse(small_cfg, raw_dir=tmp_path, db_path=":memory:")
    f2 = load_features(con)
    cols = [c for c in FEATURES + ["log_amount", "n_activities_snap", "stage_ord"] if c in f2]
    pd.testing.assert_frame_equal(feat[["opp_id"] + cols].reset_index(drop=True), f2[["opp_id"] + cols].reset_index(drop=True))


def test_account_history_uses_only_earlier_outcomes(feat):
    """Recompute prior wins for deals with an account history in pandas and compare with the SQL result."""
    key = feat.set_index("opp_id")
    closed = feat[feat["status"].isin(["won", "lost"])]
    r = feat[feat["account_prior_deals"] > 0].iloc[0]
    same = closed[(closed["account_id"] == r["account_id"]) & (closed["opp_id"] != r["opp_id"]) & (closed["closed_date"] < r["snapshot_date"])]
    # SQL joins on the canonical account key (>= same-account_id matches), so it can only find more history
    assert r["account_prior_deals"] >= len(same) and r["account_prior_wins"] >= (same["won"] == 1).sum()
    assert (feat["account_prior_wins"] <= feat["account_prior_deals"]).all()
    assert len(key) == len(feat)


def test_labels_not_in_model_features():
    assert not {"won", "status", "closed_date"} & set(FEATURES)
