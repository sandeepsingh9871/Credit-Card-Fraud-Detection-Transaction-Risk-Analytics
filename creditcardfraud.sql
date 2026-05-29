USE fraud_detection;
 
--  Quick sanity check after loading   
SELECT COUNT(*)        AS total_rows   FROM transactions;
SELECT COUNT(*)        AS fraud_rows   FROM transactions WHERE Class = 1;
SELECT COUNT(*)        AS legit_rows   FROM transactions WHERE Class = 0;

-- QUERY 1 — Class Distribution & Key Amount Stats
-- Q: "How imbalanced is the dataset? Quantify it."
-- ============================================================
 
SELECT
    CASE Class WHEN 0 THEN 'Legitimate' ELSE 'Fraud' END   AS class_label,
    COUNT(*)                                                AS txn_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 4)      AS pct_of_total,
    ROUND(AVG(Amount), 2)                                   AS avg_amount,
    ROUND(MIN(Amount), 2)                                   AS min_amount,
    ROUND(MAX(Amount), 2)                                   AS max_amount,
    ROUND(STD(Amount),  2)                                  AS std_amount
FROM transactions
GROUP BY Class
ORDER BY Class;
 
--     Imbalance ratio                              
SELECT
    SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END)              AS legit_count,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)              AS fraud_count,
    ROUND(
        SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) /
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
    , 0)                                                    AS imbalance_ratio
FROM transactions;

-- QUERY 2 — Fraud Rate by Hour of Day
-- Q: "Which hours have the highest fraud rate?
--              Write a query to find out."
-- ============================================================
 
SELECT
    FLOOR(Time / 3600) MOD 24                               AS hour_of_day,
    COUNT(*)                                                AS total_txns,
    SUM(Class)                                              AS fraud_count,
    COUNT(*) - SUM(Class)                                   AS legit_count,
    ROUND(SUM(Class) * 100.0 / COUNT(*), 4)                 AS fraud_rate_pct,
    ROUND(AVG(Amount), 2)                                   AS avg_amount,
    ROUND(AVG(CASE WHEN Class = 1 THEN Amount END), 2)      AS avg_fraud_amount,
    ROUND(AVG(CASE WHEN Class = 0 THEN Amount END), 2)      AS avg_legit_amount
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC;
 
--     Top 5 highest fraud-rate hours      
SELECT
    FLOOR(Time / 3600) MOD 24                               AS hour_of_day,
    COUNT(*)                                                AS total_txns,
    SUM(Class)                                              AS fraud_count,
    ROUND(SUM(Class) * 100.0 / COUNT(*), 4)                 AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC
LIMIT 5;

-- QUERY 3 — Amount Bucket Analysis (CASE WHEN bucketing)
-- Q : "Bin transactions by amount and show
--              fraud concentration per bucket."
-- ============================================================
SELECT
    CASE
        WHEN Amount < 10                    THEN '1. Under $10'
        WHEN Amount BETWEEN 10   AND 49.99  THEN '2. $10 – $49'
        WHEN Amount BETWEEN 50   AND 99.99  THEN '3. $50 – $99'
        WHEN Amount BETWEEN 100  AND 199.99 THEN '4. $100 – $199'
        WHEN Amount BETWEEN 200  AND 499.99 THEN '5. $200 – $499'
        ELSE                                     '6. $500 +'
    END                                                     AS amount_bucket,
    COUNT(*)                                                AS total_txns,
    SUM(Class)                                              AS fraud_txns,
    COUNT(*) - SUM(Class)                                   AS legit_txns,
    ROUND(SUM(Class) * 100.0 / COUNT(*), 3)                 AS fraud_rate_pct,
    ROUND(AVG(Amount), 2)                                   AS avg_amount,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS total_fraud_amount
FROM transactions
GROUP BY amount_bucket
ORDER BY amount_bucket;

-- QUERY 4 — Transaction Velocity (Window Functions)
--  Q: "Approximate how many transactions happen
--              in short time windows and if velocity
--              correlates with fraud."
-- ============================================================
 
-- Step 1: tag each transaction with its minute bucket count
WITH velocity_tagged AS (
    SELECT
        Class,
        Amount,
        FLOOR(Time / 3600) MOD 24                           AS hour_of_day,
        FLOOR(Time / 60)                                    AS minute_bucket,
        COUNT(*) OVER (
            PARTITION BY FLOOR(Time / 60)
        )                                                   AS txns_in_same_minute
    FROM transactions
)
-- Step 2: bucket by velocity and compute fraud rate
SELECT
    CASE
        WHEN txns_in_same_minute = 1    THEN '1 txn  (low velocity)'
        WHEN txns_in_same_minute <= 3   THEN '2–3 txns'
        WHEN txns_in_same_minute <= 6   THEN '4–6 txns'
        WHEN txns_in_same_minute <= 10  THEN '7–10 txns'
        ELSE                                 '10+ txns (high velocity)'
    END                                                     AS velocity_bucket,
    COUNT(*)                                                AS total_txns,
    SUM(Class)                                              AS fraud_txns,
    ROUND(SUM(Class) * 100.0 / COUNT(*), 4)                 AS fraud_rate_pct,
    ROUND(AVG(Amount), 2)                                   AS avg_amount
FROM velocity_tagged
GROUP BY velocity_bucket
ORDER BY velocity_bucket;
 
--     Time gap between consecutive transactions                         
-- Shows how rapidly transactions occur — proxy for card testing
WITH ordered AS (
    SELECT
        Class,
        Amount,
        Time,
        LAG(Time) OVER (ORDER BY Time)                      AS prev_time
    FROM transactions
)
SELECT
    Class,
    CASE
        WHEN (Time - prev_time) < 1     THEN 'Under 1 sec'
        WHEN (Time - prev_time) < 10    THEN '1–10 secs'
        WHEN (Time - prev_time) < 60    THEN '10–60 secs'
        WHEN (Time - prev_time) < 300   THEN '1–5 mins'
        ELSE                                 '5 mins +'
    END                                                     AS time_gap_bucket,
    COUNT(*)                                                AS txn_count,
    ROUND(AVG(Amount), 2)                                   AS avg_amount
FROM ordered
WHERE prev_time IS NOT NULL
GROUP BY Class, time_gap_bucket
ORDER BY Class, time_gap_bucket;

-- QUERY 5 — Executive Summary (Single Query KPI View)
-- Q: "Give me one query that tells a risk officer
--              everything they need to know."
-- ============================================================
 
SELECT
    -- Volume
    COUNT(*)                                                AS total_transactions,
    SUM(Class)                                              AS total_fraud,
    COUNT(*) - SUM(Class)                                   AS total_legitimate,
 
    -- Rates
    ROUND(SUM(Class) * 100.0 / COUNT(*), 4)                 AS fraud_rate_pct,
 
    -- Imbalance
    ROUND(
        (COUNT(*) - SUM(Class)) / SUM(Class)
    , 0)                                                    AS imbalance_ratio,
 
    -- Fraud amounts
    ROUND(SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END), 2) AS total_fraud_value,
    ROUND(AVG(CASE WHEN Class=1 THEN Amount END),        2) AS avg_fraud_amount,
    ROUND(AVG(CASE WHEN Class=0 THEN Amount END),        2) AS avg_legit_amount,
    ROUND(MAX(CASE WHEN Class=1 THEN Amount END),        2) AS max_fraud_amount,
 
    -- Fraud % under $100 (card-testing pattern)
    ROUND(
        SUM(CASE WHEN Class=1 AND Amount < 100 THEN 1 ELSE 0 END)
        * 100.0 / SUM(Class)
    , 1)                                                    AS fraud_pct_under_100,
 
    -- Time span
    ROUND(MIN(Time) / 3600, 1)                              AS data_start_hr,
    ROUND(MAX(Time) / 3600, 1)                              AS data_end_hr,
    ROUND((MAX(Time) - MIN(Time)) / 3600, 1)                AS data_span_hrs
 
FROM transactions;

-- VIEWS — Reusable named queries (feed Phase 2 & Tableau)
-- ============================================================
 
--     View 1: fraud_hourly_summary         ─
DROP VIEW IF EXISTS fraud_hourly_summary;
CREATE VIEW fraud_hourly_summary AS
SELECT
    FLOOR(Time / 3600) MOD 24                               AS hour_of_day,
    COUNT(*)                                                AS total_txns,
    SUM(Class)                                              AS fraud_count,
    COUNT(*) - SUM(Class)                                   AS legit_count,
    ROUND(SUM(Class) * 100.0 / COUNT(*), 4)                 AS fraud_rate_pct,
    ROUND(AVG(Amount), 2)                                   AS avg_amount,
    ROUND(AVG(CASE WHEN Class=1 THEN Amount END), 2)        AS avg_fraud_amount
FROM transactions
GROUP BY hour_of_day;

--     View 2: fraud_amount_buckets         ─
DROP VIEW IF EXISTS fraud_amount_buckets;
CREATE VIEW fraud_amount_buckets AS
SELECT
    CASE
        WHEN Amount < 10                    THEN '1. Under $10'
        WHEN Amount BETWEEN 10   AND 49.99  THEN '2. $10 – $49'
        WHEN Amount BETWEEN 50   AND 99.99  THEN '3. $50 – $99'
        WHEN Amount BETWEEN 100  AND 199.99 THEN '4. $100 – $199'
        WHEN Amount BETWEEN 200  AND 499.99 THEN '5. $200 – $499'
        ELSE                                     '6. $500 +'
    END                                                     AS amount_bucket,
    COUNT(*)                                                AS total_txns,
    SUM(Class)                                              AS fraud_txns,
    ROUND(SUM(Class) * 100.0 / COUNT(*), 3)                 AS fraud_rate_pct
FROM transactions
GROUP BY amount_bucket;

--     View 3: clean_transactions (Phase 2 Python feed)             ─
DROP VIEW IF EXISTS clean_transactions;
CREATE VIEW clean_transactions AS
SELECT
    Time,
    FLOOR(Time / 3600) MOD 24                               AS hour_of_day,
    CASE
        WHEN FLOOR(Time / 3600) MOD 24 BETWEEN 0  AND 5  THEN 'Night'
        WHEN FLOOR(Time / 3600) MOD 24 BETWEEN 6  AND 11 THEN 'Morning'
        WHEN FLOOR(Time / 3600) MOD 24 BETWEEN 12 AND 17 THEN 'Afternoon'
        ELSE                                                    'Evening'
    END                                                     AS time_of_day,
    Amount,
    LOG(Amount + 1)                                         AS log_amount,
    V1,  V2,  V3,  V4,  V5,  V6,  V7,
    V8,  V9,  V10, V11, V12, V13, V14,
    V15, V16, V17, V18, V19, V20, V21,
    V22, V23, V24, V25, V26, V27, V28,
    Class
FROM transactions;


--     Verify all views                           ─
SELECT
    table_name   AS view_name,
    table_type
FROM information_schema.tables
WHERE table_schema = 'fraud_detection'
ORDER BY table_type, table_name;

-- Export 1: full clean dataset → for Python Phase 2
SET SESSION sql_select_limit = 300000;
SELECT * FROM clean_transactions;


-- Export 2: hourly summary → for Tableau dashboard
SELECT * FROM fraud_hourly_summary ORDER BY hour_of_day;

-- Export 3: amount buckets → for Tableau dashboard
SELECT * FROM fraud_amount_buckets ORDER BY amount_bucket;
