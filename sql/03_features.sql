-- Point-in-time feature table. One row per opportunity, evaluated at
-- snapshot_date = created_date + ${SNAPSHOT_DAYS}. Nothing after the snapshot may enter a feature:
-- activities, stages and "prior" outcomes are all filtered on dates <= snapshot_date.
-- The label comes from status/closed_date and is only selected in the final projection.

CREATE OR REPLACE TABLE feat_base AS
SELECT o.*, o.created_date + ${SNAPSHOT_DAYS} AS snapshot_date
FROM stg_opportunities o;

CREATE OR REPLACE TABLE feat_activity AS
SELECT b.opp_id,
       count(a.activity_id) AS n_activities,
       count(a.activity_id) FILTER (WHERE a.activity_type = 'meeting') AS n_meetings,
       count(DISTINCT a.contact_no) AS n_contacts,
       datediff('day', max(a.activity_date), any_value(b.snapshot_date)) AS days_since_last_activity
FROM feat_base b
LEFT JOIN stg_activities a
  ON a.opp_id = b.opp_id AND a.activity_date <= b.snapshot_date
GROUP BY b.opp_id;

CREATE OR REPLACE TABLE feat_stage AS
SELECT opp_id, stage, stage_ord FROM (
    SELECT b.opp_id, s.stage, s.stage_ord,
           row_number() OVER (PARTITION BY b.opp_id ORDER BY s.entered_date DESC, s.stage_ord DESC) AS rn
    FROM feat_base b JOIN stg_stage_history s
      ON s.opp_id = b.opp_id AND s.entered_date <= b.snapshot_date
) WHERE rn = 1;

-- prior outcomes of the same canonical account, known strictly before the snapshot
CREATE OR REPLACE TABLE feat_account_history AS
SELECT b.opp_id,
       count(c.opp_id) AS account_prior_deals,
       count(c.opp_id) FILTER (WHERE c.status = 'won') AS account_prior_wins
FROM feat_base b
LEFT JOIN feat_base c
  ON c.account_key = b.account_key AND c.opp_id <> b.opp_id
 AND c.status IN ('won', 'lost') AND c.closed_date < b.snapshot_date
GROUP BY b.opp_id;

CREATE OR REPLACE TABLE feat_owner_history AS
SELECT b.opp_id,
       count(c.opp_id) AS owner_prior_deals,
       CASE WHEN count(c.opp_id) >= 10
            THEN count(c.opp_id) FILTER (WHERE c.status = 'won') * 1.0 / count(c.opp_id) END AS owner_prior_win_rate
FROM feat_base b
LEFT JOIN feat_base c
  ON c.owner_id = b.owner_id
 AND c.status IN ('won', 'lost') AND c.closed_date < b.snapshot_date
GROUP BY b.opp_id;

CREATE OR REPLACE TABLE features AS
SELECT
    b.opp_id, b.account_id, b.owner_id, b.created_date, b.snapshot_date,
    b.segment, b.region, coalesce(b.industry, 'Unknown') AS industry,
    coalesce(b.lead_source, 'Unknown') AS lead_source,
    b.amount,
    CASE WHEN b.amount > 0 THEN ln(b.amount) END AS log_amount,
    act.n_activities AS n_activities_snap,
    act.n_meetings AS n_meetings_snap,
    act.n_contacts AS n_contacts_snap,
    act.days_since_last_activity,
    st.stage, st.stage_ord,
    datediff('day', b.created_date, b.expected_close_date) AS expected_cycle_days,
    CASE WHEN b.expected_close_date IS NOT NULL AND b.expected_close_date < b.snapshot_date THEN 1 ELSE 0 END AS close_date_stale,
    ah.account_prior_deals, ah.account_prior_wins,
    oh.owner_prior_deals, oh.owner_prior_win_rate,
    datediff('day', b.snapshot_date, CAST(date_trunc('quarter', b.snapshot_date) + INTERVAL 3 MONTH AS DATE)) AS days_to_quarter_end,
    quarter(b.created_date) AS created_qtr,
    d.dq_error, d.dq_warnings, d.dq_warning_rules,
    -- labels (never used as features)
    b.status, b.closed_date,
    CASE b.status WHEN 'won' THEN 1 WHEN 'lost' THEN 0 END AS won
FROM feat_base b
LEFT JOIN feat_activity act USING (opp_id)
LEFT JOIN feat_stage st USING (opp_id)
LEFT JOIN feat_account_history ah USING (opp_id)
LEFT JOIN feat_owner_history oh USING (opp_id)
LEFT JOIN dq_opp d USING (opp_id);
