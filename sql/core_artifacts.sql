-- =============================================================================
-- Project: Credit Risk Analysis & Loan Default Prediction
-- Engine: PostgreSQL
-- File: sql/core_artifacts.sql
-- Purpose: Schema Initialization, Dynamic Feature Views, and Risk Profiling
-- =============================================================================

-- Initialize Analytics Schema
CREATE SCHEMA IF NOT EXISTS analytics;

-- -----------------------------------------------------------------------------
-- DYNAMIC FEATURE VIEW: Analytics Feature Engine
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW analytics.v_credit_risk_engineered AS 
SELECT 
    sk_id_curr,
    target,
    amt_credit,
    amt_income_total,
    ROUND((amt_credit::NUMERIC / NULLIF(amt_income_total::NUMERIC, 0)), 4) AS fe_loan_to_income_ratio,
    ROUND(ABS(days_birth::NUMERIC) / 365.25, 2) AS fe_age_years,
    ROUND(ABS((CASE WHEN days_employed = 365243 THEN NULL ELSE days_employed END)::NUMERIC) / 365.25, 2) AS fe_years_employed
FROM public.credit_risk_data;


-- -----------------------------------------------------------------------------
-- QUERY 1: Default Rate Breakdown by Occupation Type
-- -----------------------------------------------------------------------------
SELECT 
    COALESCE(occupation_type, 'Unknown / Not Disclosed') AS occupation_type,
    COUNT(*) AS total_applications,
    SUM(target) AS total_defaults,
    ROUND(AVG(target) * 100.0, 2) AS default_rate_pct
FROM public.credit_risk_data
GROUP BY occupation_type
HAVING COUNT(*) >= 100
ORDER BY default_rate_pct DESC;


-- -----------------------------------------------------------------------------
-- QUERY 2: Running Cumulative Default Rate Across Age Cohorts
-- -----------------------------------------------------------------------------
WITH age_cohorts AS (
    SELECT 
        WIDTH_BUCKET(fe_age_years, 20, 70, 10) AS bucket_id,
        MIN(fe_age_years) AS age_group_min,
        MAX(fe_age_years) AS age_group_max,
        COUNT(*) AS total_population,
        SUM(target) AS total_defaults,
        AVG(target) AS group_default_rate
    FROM analytics.v_credit_risk_engineered
    WHERE fe_age_years IS NOT NULL
    GROUP BY bucket_id
)
SELECT 
    CONCAT(ROUND(age_group_min, 0), ' - ', ROUND(age_group_max, 0), ' Yrs') AS age_cohort,
    total_population,
    total_defaults,
    ROUND(group_default_rate * 100.0, 2) AS cohort_default_rate_pct,
    ROUND(
        (SUM(total_defaults) OVER (ORDER BY bucket_id)::NUMERIC / 
         NULLIF(SUM(total_population) OVER (ORDER BY bucket_id), 0)) * 100.0, 
        2
    ) AS cumulative_running_default_rate_pct
FROM age_cohorts
ORDER BY bucket_id ASC;


-- -----------------------------------------------------------------------------
-- QUERY 3: Income Tier Risk Distribution
-- -----------------------------------------------------------------------------
WITH income_bucketed AS (
    SELECT 
        target,
        amt_income_total,
        CASE 
            WHEN amt_income_total < 100000 THEN '1. Low Income (<100k)'
            WHEN amt_income_total BETWEEN 100000 AND 199999.99 THEN '2. Medium Income (100k-200k)'
            WHEN amt_income_total BETWEEN 200000 AND 349999.99 THEN '3. High Income (200k-350k)'
            ELSE '4. Executive Income (>350k)'
        END AS income_band
    FROM public.credit_risk_data
)
SELECT 
    income_band,
    COUNT(*) AS total_applicants,
    SUM(target) AS total_defaults,
    ROUND(AVG(target) * 100.0, 2) AS default_rate_pct
FROM income_bucketed
GROUP BY income_band
ORDER BY income_band ASC;


-- -----------------------------------------------------------------------------
-- QUERY 4: Top 10 High-Risk Organization Types
-- -----------------------------------------------------------------------------
SELECT 
    organization_type,
    COUNT(*) AS total_applications,
    SUM(target) AS total_defaults,
    ROUND(AVG(target) * 100.0, 2) AS default_rate_pct
FROM public.credit_risk_data
WHERE organization_type NOT IN ('XNA', 'Other')
GROUP BY organization_type
HAVING COUNT(*) >= 250
ORDER BY default_rate_pct DESC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- QUERY 5: Financial Burden Analysis (Defaulters vs. Repaid)
-- -----------------------------------------------------------------------------
WITH risk_summary AS (
    SELECT 
        target,
        AVG(amt_credit) AS avg_credit_amount,
        AVG(fe_loan_to_income_ratio) AS avg_lti_ratio
    FROM analytics.v_credit_risk_engineered
    GROUP BY target
)
SELECT 
    CASE WHEN target = 1 THEN 'Defaulters' ELSE 'Repaid' END AS borrower_status,
    ROUND(avg_credit_amount::NUMERIC, 2) AS avg_loan_size,
    ROUND(avg_lti_ratio::NUMERIC, 2) AS avg_loan_to_income_ratio
FROM risk_summary;