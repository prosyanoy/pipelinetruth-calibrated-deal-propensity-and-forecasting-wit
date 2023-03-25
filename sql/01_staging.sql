-- Staging: load raw CSVs, dedupe, and derive a canonical account key.
-- Parameters (substituted by warehouse.py): ${RAW_DIR}

CREATE OR REPLACE TABLE stg_accounts AS
SELECT
    account_id,
    trim(account_name) AS account_name,
    industry,
    region,
    -- canonical key: lowercase, strip punctuation, drop legal-entity suffix
    trim(regexp_replace(
        regexp_replace(lower(account_name), '[^a-z0-9 ]', '', 'g'),
        '\s+(inc|incorporated|corp|corporation|llc|ltd|co|company)\s*$', '')) AS account_key
FROM read_csv_auto('${RAW_DIR}/accounts.csv', header = true);

-- exact duplicate opportunity rows are collapsed; the count is reported by the DQ layer
CREATE OR REPLACE TABLE stg_opportunities_raw AS
SELECT *, row_number() OVER (PARTITION BY opp_id ORDER BY created_date) AS dup_rank
FROM read_csv_auto('${RAW_DIR}/opportunities.csv', header = true, dateformat = '%Y-%m-%d');

CREATE OR REPLACE TABLE stg_opportunities AS
SELECT o.* EXCLUDE (dup_rank), a.account_key, a.industry, a.region
FROM stg_opportunities_raw o
LEFT JOIN stg_accounts a USING (account_id)
WHERE o.dup_rank = 1;

CREATE OR REPLACE TABLE stg_activities AS
SELECT activity_id, opp_id, CAST(activity_date AS DATE) AS activity_date, activity_type, contact_no
FROM read_csv_auto('${RAW_DIR}/activities.csv', header = true);

CREATE OR REPLACE TABLE stg_stage_history AS
SELECT opp_id, stage, CAST(entered_date AS DATE) AS entered_date,
       CASE stage WHEN 'Qualify' THEN 1 WHEN 'Develop' THEN 2 WHEN 'Propose' THEN 3 WHEN 'Negotiate' THEN 4 END AS stage_ord
FROM read_csv_auto('${RAW_DIR}/stage_history.csv', header = true);
