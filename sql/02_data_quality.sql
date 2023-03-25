-- Data-quality checks. Every failing record lands in dq_flags (long format) with a severity:
--   error   -> record is quarantined (excluded from training, scoring and forecasting)
--   warning -> record is kept, but surfaced as a trust flag and counted in the model card
-- Parameters: ${SNAPSHOT_DAYS}

CREATE OR REPLACE TABLE dq_flags AS
WITH seg_median AS (
    SELECT segment, median(amount) AS med FROM stg_opportunities WHERE amount > 0 GROUP BY segment
),
acct_dupes AS (
    SELECT account_key FROM stg_accounts GROUP BY account_key HAVING count(*) > 1
)
SELECT opp_id, 'missing_amount' AS rule, 'error' AS severity
FROM stg_opportunities WHERE amount IS NULL
UNION ALL
SELECT opp_id, 'nonpositive_amount', 'error' FROM stg_opportunities WHERE amount <= 0
UNION ALL
SELECT opp_id, 'closed_before_created', 'error'
FROM stg_opportunities WHERE closed_date IS NOT NULL AND closed_date < created_date
UNION ALL
SELECT opp_id, 'closed_before_snapshot', 'error'
FROM stg_opportunities
WHERE closed_date IS NOT NULL AND closed_date >= created_date
  AND closed_date <= created_date + ${SNAPSHOT_DAYS}
UNION ALL
SELECT opp_id, 'unknown_account', 'error' FROM stg_opportunities WHERE account_key IS NULL
UNION ALL
SELECT o.opp_id, 'amount_outlier', 'warning'
FROM stg_opportunities o JOIN seg_median m USING (segment)
WHERE o.amount > 10 * m.med
UNION ALL
SELECT opp_id, 'expected_close_missing', 'warning' FROM stg_opportunities WHERE expected_close_date IS NULL
UNION ALL
-- stale: the CRM close date had already passed when the deal was snapshotted and nobody updated it
SELECT opp_id, 'stale_close_date', 'warning'
FROM stg_opportunities
WHERE expected_close_date IS NOT NULL AND expected_close_date < created_date + ${SNAPSHOT_DAYS}
UNION ALL
SELECT opp_id, 'missing_lead_source', 'warning' FROM stg_opportunities WHERE lead_source IS NULL
UNION ALL
SELECT opp_id, 'missing_industry', 'warning' FROM stg_opportunities WHERE industry IS NULL
UNION ALL
SELECT o.opp_id, 'duplicate_account', 'warning'
FROM stg_opportunities o JOIN acct_dupes d USING (account_key);

-- table-level checks (not tied to one opportunity)
CREATE OR REPLACE TABLE dq_table_checks AS
SELECT 'duplicate_opportunity_rows' AS check_name, 'warning' AS severity,
       (SELECT count(*) FROM stg_opportunities_raw WHERE dup_rank > 1) AS n_records
UNION ALL
SELECT 'orphan_activities', 'warning',
       (SELECT count(*) FROM stg_activities a ANTI JOIN stg_opportunities o USING (opp_id))
UNION ALL
SELECT 'duplicate_account_records', 'warning',
       (SELECT coalesce(sum(c - 1), 0) FROM (SELECT count(*) AS c FROM stg_accounts GROUP BY account_key HAVING count(*) > 1));

CREATE OR REPLACE VIEW dq_summary AS
SELECT rule, severity, count(DISTINCT opp_id) AS n_records,
       round(100.0 * count(DISTINCT opp_id) / (SELECT count(*) FROM stg_opportunities), 2) AS pct_records
FROM dq_flags GROUP BY rule, severity ORDER BY severity, n_records DESC;

CREATE OR REPLACE TABLE dq_opp AS
SELECT o.opp_id,
       coalesce(bool_or(f.severity = 'error'), false) AS dq_error,
       count(f.rule) FILTER (WHERE f.severity = 'warning') AS dq_warnings,
       string_agg(f.rule, ',') FILTER (WHERE f.severity = 'warning') AS dq_warning_rules
FROM stg_opportunities o LEFT JOIN dq_flags f USING (opp_id)
GROUP BY o.opp_id;
