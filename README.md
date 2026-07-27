# Credit Risk Portfolio Analytics & Default Prediction

[![Domain](https://img.shields.io/badge/Domain-Banking_%26_Financial_Services-blue)](#)
[![Tools](https://img.shields.io/badge/Tools-Python_%7C_SQL_%7C_Power_BI-orange)](#)
[![Model](https://img.shields.io/badge/Model-Logistic_Regression-green)](#)

## Executive Summary
Financial institutions face substantial capital loss from loan defaults among "thin-file" applicants who lack extensive credit bureau histories. Traditional credit scoring mechanisms often misgrade these unbanked borrowers. 

This project evaluates **307,511 loan applications** from the Home Credit dataset, implementing an end-to-end analytics and predictive scoring pipeline:
1. **Exploratory Data Analysis (Python):** Profiled demographic, financial, and external bureau risk drivers.
2. **Relational Analytics (PostgreSQL):** Built dynamic feature views and window-function cohort aggregations.
3. **Predictive Scoring (Scikit-Learn):** Trained a balanced Logistic Regression classifier prioritizing **Recall (63.50%)** to maximize unrecovered principal protection.
4. **Executive Reporting (Power BI):** Designed an interactive, two-page executive dashboard mapping portfolio risk drivers.

---

## Technical Architecture & File Governance

```text
Credit_Risk_Project/
├── assets/
│   ├── correlation_heatmap.png      # Feature Interaction Matrix (Pre-Modeling)
│   ├── dashboard_overview.png       # Power BI Page 1: Portfolio Risk Metrics
│   ├── dashboard_underwiring.png    # Power BI Page 2: Financial & Contract Exposure
│   ├── LR_coefficients.png          # Model Interpretability: Feature Risk Weights
│   └── ROC-Confusion_Matrix.png     # Predictive Scorecard & Class Separation
├── dashboards/
│   └── credit_risk_dashboard.pbix   # Primary Power BI Desktop Report File
├── data/                            # Local Staging Directory (Ignored in Git)
│   ├── processed/
│   │   └── cleaned_application_train.csv
│   └── raw/
│       └── application_train.csv
├── Notebooks/
│   └── credit_risk_analysis.ipynb   # Primary Python Exploratory & ML Pipeline
├── sql/
│   └── core_artifacts.sql           # PostgreSQL Schema, Views & Analytics Queries
├── .gitignore
├── README.md                        # Master Portfolio Documentation
└── requirements.txt                 # Reproducible Python Environment Manifest
```

---

## Key Business Insights & Risk Vectors

| Risk Vector | Empirical Finding | Underwriting Action Item |
| :--- | :--- | :--- |
| **Baseline Risk** | Overall portfolio default rate sits at **8.07%** (24,825 total defaults). | Benchmark for portfolio risk policy. |
| **Borrower Age** | Default risk declines monotonically from **12.5% (Ages 20–24)** to **<4.0% (Ages 65+)**. | Implement initial credit limit step-ups for borrowers under 30. |
| **Education Tier** | Lower Secondary applicants default at **~11.0%** vs. **1.8%** for Academic Degree holders. | Incorporate education tiering into risk weighting. |
| **Contract Type** | Cash Loans represent **90.5% of total volume** with an 8.35% default rate (vs. 5.48% for Revolving Loans). | Enforce strict Loan-to-Income caps on cash requests. |
| **Employment Status** | Applicants on Maternity Leave (**40.00%**) and Unemployed (**36.36%**) show acute risk spikes. | Mandate manual credit committee review for non-standard employment. |

---

## Machine Learning Scorecard & Portfolio Impact

Given the ~8.07% baseline default rate, raw accuracy is a misleading metric. The model was optimized with `class_weight='balanced'` to prioritize **Recall**:

* **Direct Capital Protection:** The model correctly catches **3,153 true defaults** out of 4,965 total defaulters in the holdout evaluation set. Proactively declining or restructuring these high-risk applications prevents millions in non-performing loan (NPL) write-offs.
* **Operational Trade-Off:** Optimizing for recall causes the model to flag **18,317 safe borrowers as potential risks** (False Positives). While this creates an opportunity cost in foregone interest margin, unrecovered principal losses are far more damaging to bank capital reserves than lost interest.

| Evaluation Metric | Score | Financial Rationale & Business Impact |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.70+** | Measures strong class-separation capability across varying decision thresholds. |
| **Recall (Default Class)** | **63.50%** | Primary risk metric: catches majority of defaulters to prevent capital write-offs. |
| **Precision** | **14.68%** | Expected low precision due to 8% baseline class imbalance and balanced weighting. |
| **Model Interpretability** | **High** | Logistic Regression coefficients provide clear Adverse Action notice drivers. |

> **Regulatory Compliance Note:** Logistic Regression was selected over black-box ensembles (e.g., XGBoost) to maintain 100% feature coefficient auditability for FCRA/Equal Credit Opportunity Act compliance when issuing Adverse Action notices to rejected applicants.