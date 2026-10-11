# Institutional Credit Risk Analytics & Default Prediction Engine

[![Domain](https://img.shields.io/badge/Domain-Banking_%26_Financial_Services_(BFSI)-003366?style=for-the-badge)](https://en.wikipedia.org/wiki/Credit_risk)
[![Governance](https://img.shields.io/badge/Governance-Basel_F--IRB_%7C_SR_11--7_%7C_ECOA_Reg_B-blueviolet?style=for-the-badge)](https://www.bis.org/bcbs/)
[![Stack](https://img.shields.io/badge/Stack-Python_3.12_%7C_XGBoost_%7C_LightGBM_%7C_SHAP-F7931E?style=for-the-badge)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-PostgreSQL_16-336791?style=for-the-badge)](https://www.postgresql.org/)
[![BI](https://img.shields.io/badge/BI-Power_BI_Desktop-F2C811?style=for-the-badge)](https://powerbi.microsoft.com/)

---

## Executive Summary & Business Impact

Retail financial institutions face severe capital write-offs when underwriting "thin-file" consumer applicants who lack comprehensive credit bureau histories[cite: 1, 2]. Traditional credit scorecards frequently misgrade these borrowers, relying on rigid linear thresholds that either cause non-performing loan (NPL) spikes or generate excessive, commercially unviable rejection volumes[cite: 1, 2].

This platform delivers an end-to-end, auditable **Credit Risk Engineering & Decision Platform** evaluating **307,511 loan applications** from the Home Credit Default Risk portfolio[cite: 1]. Moving beyond exploratory data analysis, this project establishes:

1. **Regulatory Baseline Replication:** Reconciled a frozen 5-feature balanced Logistic Regression baseline ($\text{ROC-AUC} = 0.7154$, $\text{Recall} = 63.73\%$, $\text{Precision} = 14.72\%$) strictly compliant with **ECOA Regulation B** non-discrimination standards[cite: 1, 2, 7].
2. **Multi-Model Benchmark:** Evaluated 5 model architectures across an isolated 5-fold Stratified Cross-Validation framework ($N = 246,008$), proving that **Gradient Boosted Decision Trees (XGBoost/LightGBM)** deliver a **+28.24% relative lift in Precision-Recall AUC** over traditional linear scorecards[cite: 7, 12].
3. **Probability Calibration & Basel Loss Optimization:** Corrected classifier probability distortion via **Isotonic Regression** (cutting Brier calibration error by **61.55%**)[cite: 7, 11]. Under the **Basel Foundation-IRB** loss framework ($LGD = 45\%$, net interest margin $= 9\%$, cost ratio $r = 5:1$), we mathematically derived an optimal cutoff ($t^* = 16.0\%$) that reduces expected portfolio credit loss by **$5,861.12 per applicant (25.62% net loss reduction)** over the historical baseline.
4. **Relational Feature Store:** Scaled the pipeline across **1.71M external bureau tradelines** and **1.67M previous application records**, demonstrating that multi-table behavioral history drives a **+4.13% incremental PR-AUC lift**[cite: 7, 23].
5. **Algorithmic Explainability & Compliance:** Extracted exact **SHAP TreeExplainer** feature attributions to generate automated, consumer-facing **Adverse Action Reason Codes** required under the **Fair Credit Reporting Act (FCRA § 615)**[cite: 7, 8].

---

## Technical Architecture & File Governance

The repository decouples data ingestion, feature transformations, model training, and evaluation harnesses to prevent data leakage[cite: 7]:

```text
Credit_Risk_Project/
├── assets/
│   ├── calibration_curve.png        # Reliability diagram: Isotonic calibration vs raw probabilities
│   ├── pr_roc_overlay.png           # 5-Fold OOF Precision-Recall & ROC comparative curves
│   ├── shap_bar.png                 # Global feature importance (Mean |SHAP|)
│   ├── shap_beeswarm.png            # Feature risk impact directionality beeswarm plot
│   └── shap_dependence.png          # Non-linear interaction plots (EXT_SOURCE_2 & Leverage)
├── dashboards/
│   └── credit_risk_dashboard.pbix   # Two-page executive Power BI underwriting suite
├── data/                            # Local staging directory (Ignored in Git)
│   ├── raw/
│   │   ├── application_train.csv    # 307,511 primary loan applications
│   │   ├── bureau.csv               # 1,716,428 external credit bureau tradelines
│   │   └── previous_application.csv # 1,670,214 historical internal loan decisions
│   └── processed/
├── Notebooks/
│   ├── credit_risk_analysis.ipynb   # Phase P0: Frozen regulatory v1 benchmark
│   └── 02_modeling_extension.ipynb  # Phase P1–P7: Champion-challenger ML & cost pipeline
├── results/                         # Production scorecards & audit artifacts (CSV / NPY)
│   ├── ablation_study.csv           # Multi-table relational feature store ablation table
│   ├── cv_metrics.csv               # 5-fold cross-validation multi-model scorecard
│   ├── imbalance_study.csv          # 5-run empirical sampling & reweighting benchmark
│   ├── rank_correlation.txt         # Spearman rank correlation summary metric
│   ├── shap_lr_rank_correlation.csv # Linear vs non-linear rank comparison
│   ├── shap_reason_codes.csv        # Statutory Adverse Action reason codes for sample cohorts
│   ├── test_metrics.csv             # 1,000 paired bootstrap 95% confidence intervals
│   ├── test_scored.csv              # Holdout test set with calibrated risk scores
│   └── threshold_table.csv          # Cost-sensitivity matrix across r in {2, 3, 5, 10, 20}
├── src/                             # Modular Python package
│   ├── __init__.py
│   ├── config.py                    # Paths, global seeds, regulatory columns, cost constants
│   ├── features.py                  # Feature engineering & dual Scikit-Learn transformers
│   ├── evaluate.py                  # 5-fold stratified CV engine & paired bootstrap harness
│   ├── costs.py                     # Isotonic calibration & Basel loss optimization
│   └── bureau_features.py           # Relational feature store aggregations (P6 & P7)
├── sql/
│   └── core_artifacts.sql           # PostgreSQL schema, analytical views, and cohort logic
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Model Selection & Cross-Validation Scorecard

Models were trained on the 80% partition ($N = 246,008$) under **5-Fold Stratified Cross-Validation**[cite: 7]. Because default prevalence is heavily imbalanced (**8.07% baseline rate**), **Precision-Recall AUC (PR-AUC)** was pre-committed as the primary decision metric, while ROC-AUC, Gini ($2 \times \text{AUC} - 1$), Kolmogorov-Smirnov (KS), and Brier Score served as secondary checks[cite: 1, 7].

$$\text{PR-AUC}_{\text{baseline}} = \text{Prevalence} = 0.0807$$

*Scorecard verified from `results/cv_metrics.csv`:*

| Model Architecture | Feature Space | PR-AUC (OOF) | ROC-AUC (OOF) | Gini Index | KS Stat | Brier Score | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Chance** | None | 0.0807 | 0.5000 | 0.0000 | 0.0000 | — | — |
| **LR (5-Feature Baseline)**[cite: 7] | Core 5 Features | 0.1925 ± 0.0056 | 0.7149 ± 0.0029 | 0.4298 | 0.3152 | 0.2140 | 1.1s |
| **Random Forest (300 Trees)**[cite: 7] | 108 Preprocessed | 0.2204 ± 0.0052 | 0.7444 ± 0.0020 | 0.4887 | 0.3632 | 0.1846 | 162.7s |
| **LR (Full Features, $L_2$)**[cite: 7] | 108 Preprocessed | 0.2257 ± 0.0073 | 0.7475 ± 0.0032 | 0.4950 | 0.3676 | 0.2031 | 43.9s |
| **LightGBM (500 Trees)**[cite: 7] | 108 Preprocessed | 0.2458 ± 0.0054 | 0.7621 ± 0.0033 | 0.5242 | 0.3940 | 0.1805 | 18.2s |
| **XGBoost (Champion)**[cite: 7] | 108 Preprocessed | **0.2468 ± 0.0049**[cite: 12] | **0.7628 ± 0.0028**[cite: 12] | **0.5256**[cite: 12] | **0.3951**[cite: 12] | **0.1804**[cite: 14] | 21.1s |

> **Champion Selection:** **XGBoost** earned Champion status by achieving the highest cross-validated PR-AUC (**0.2468**), generating a **+28.24% relative lift** over the 5-feature benchmark and a **+9.35% relative lift** over the full regularized Logistic Regression[cite: 12].

---

## Probability Calibration & Basel Cost Optimization

### 1. Isotonic Probability Calibration
Loss reweighting (`scale_pos_weight = 11.39`) improves ranking discrimination but distorts output probabilities, shifting the average predicted probability from 8.07% to ~45%[cite: 7]. Applying raw cutoff thresholds to these probabilities causes massive over-rejection of solvent borrowers[cite: 7].

Using an out-of-fold **Isotonic Regression** calibrator (`sklearn.isotonic.IsotonicRegression`), we mapped raw scores back to true empirical default posteriors[cite: 7]:
* **Raw Out-of-Fold Brier Score:** `0.1758`[cite: 11]
* **Calibrated Out-of-Fold Brier Score:** `0.0676` (**61.55% Calibration Error Reduction**)[cite: 11]
* **Reliability Alignment:** The calibrated probability curve tracks the ideal 45-degree diagonal across all probability deciles (`assets/calibration_curve.png`)[cite: 11].

### 2. Basel Loss Formulation & Cost Sensitivity Matrix
Under **Basel Foundation-IRB (F-IRB)** standards[cite: 7]:
* **False Negative (Default Approved):** $\text{Loss} = LGD \times \text{AMT\_CREDIT} = 0.45 \times \text{AMT\_CREDIT}$[cite: 7].
* **False Positive (Good Borrower Declined):** $\text{Loss} = \text{Margin} \times \text{AMT\_CREDIT} = 0.09 \times \text{AMT\_CREDIT}$[cite: 7].
* **Cost Ratio:** $r = \frac{0.45}{0.09} = 5.0$[cite: 7].
* **Analytic Bayes Optimal Threshold:** $t^* = \frac{1}{1 + r} = \frac{1}{1 + 5.0} \approx 0.1667$ (16.67%)[cite: 7].

An empirical grid search ($0.01 \le t \le 0.99$, step $0.005$) over calibrated probabilities identified the empirical cost-minimizing threshold at **$t^* = 0.1600$**, matching theoretical expectations within 67 basis points[cite: 10].

*Cost-sensitivity table across shifting cost ratios (`results/threshold_table.csv`):*[cite: 10]

| Cost Ratio ($r$)[cite: 10] | Risk Profile | Analytic $t^*$[cite: 10] | Optimal $t^*$[cite: 10] | Approval Rate[cite: 10] | Default Rate (Approved)[cite: 10] | Recall[cite: 10] | Precision[cite: 10] | Expected Cost / App[cite: 10] | % Cost vs Approve All[cite: 10] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2.0**[cite: 10] | Subprime / High Margin | 0.3333[cite: 10] | 0.330[cite: 10] | 97.91%[cite: 10] | 7.35%[cite: 10] | 10.80%[cite: 10] | 41.78%[cite: 10] | $19,682.37[cite: 10] | 97.19%[cite: 10] |
| **3.0**[cite: 10] | Near-Prime | 0.2500[cite: 10] | 0.250[cite: 10] | 96.26%[cite: 10] | 6.98%[cite: 10] | 16.77%[cite: 10] | 36.21%[cite: 10] | $19,058.68[cite: 10] | 94.11%[cite: 10] |
| **5.0**[cite: 10] | **Basel F-IRB Baseline**[cite: 7] | **0.1667**[cite: 10] | **0.160**[cite: 10] | **87.57%**[cite: 10] | **5.60%**[cite: 10] | **39.27%**[cite: 10] | **25.51%**[cite: 10] | **$17,338.76**[cite: 10] | **85.61%**[cite: 10] |
| **10.0**[cite: 10] | Prime Corporate | 0.0909[cite: 10] | 0.100[cite: 10] | 71.14%[cite: 10] | 4.01%[cite: 10] | 64.68%[cite: 10] | 18.09%[cite: 10] | $13,684.25[cite: 10] | 67.57%[cite: 10] |
| **20.0**[cite: 10] | Ultra-Conservative | 0.0476[cite: 10] | 0.045[cite: 10] | 46.34%[cite: 10] | 2.59%[cite: 10] | 85.13%[cite: 10] | 12.81%[cite: 10] | $9,496.80[cite: 10] | 46.89%[cite: 10] |

---

## Quarantined Holdout Test Results & Paired Bootstrap Inference

The holdout test partition ($N = 61,503$, 4,965 defaults) was evaluated with the decision threshold locked at $t^* = 0.1600$[cite: 1, 10]. To evaluate statistical significance, we ran **1,000 paired bootstrap iterations**, scoring both models on the *exact same resample* to control for sample covariance[cite: 7].

*Results verified from `results/test_metrics.csv`:*[cite: 9]

| Underwriting Policy Setup[cite: 9] | Test ROC-AUC (95% CI)[cite: 9] | Test PR-AUC (95% CI)[cite: 9] | Expected Cost / Applicant (95% CI)[cite: 9] | Net Loss Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Approve Everyone ($t=1.0$)**[cite: 9] | 0.5000 [0.5000, 0.5000][cite: 9] | 0.0807 [0.0786, 0.0829][cite: 9] | $20,298.78 [$19,628.72, $20,958.44][cite: 9] | Benchmark (0.0%) |
| **Baseline v1 LR ($t=0.50$)**[cite: 9] | 0.7153 [0.7076, 0.7230][cite: 9] | 0.1972 [0.1881, 0.2069][cite: 9] | $22,882.59 [$22,424.36, $23,340.82][cite: 9] | -12.73% (Cost Deficit) |
| **Champion XGBoost ($t=0.50$)**[cite: 9] | 0.7692 [0.7622, 0.7760][cite: 9] | 0.2584 [0.2469, 0.2702][cite: 9] | $19,246.00 [$18,795.34, $19,705.86][cite: 9] | +5.19% Savings |
| **Champion XGBoost ($t^*=0.1600$)**[cite: 9, 10] | **0.7691 [0.7620, 0.7759]**[cite: 9] | **0.2512 [0.2403, 0.2626]**[cite: 9] | **$17,013.17 [$16,478.45, $17,547.89]**[cite: 9] | **+25.62% Savings**[cite: 8] |

```text
========================================================================================
                      EXECUTIVE IMPACT & SIGNIFICANCE AUDIT
========================================================================================
• Point Estimate Savings per Loan App     : $5,861.12 (25.62% Net Loss Reduction)[cite: 8]
• Paired 95% Confidence Interval (Savings): [$5,402.24, $6,295.35] per applicant[cite: 8]
• Paired 95% CI for PR-AUC Lift           : [+0.0469, +0.0618][cite: 8]
• Statistical Significance Validation     : APPROVED (CI lower bound strictly > $0)[cite: 8]
========================================================================================
```

---

## Multi-Table Relational Feature Store (Phases P6 & P7)

To enrich the application-only file, we built an automated relational feature store across two auxiliary tables[cite: 7, 8]:
1. **External Bureau History (`bureau.csv` — 1,716,428 rows):** 11 attributes capturing multi-lender credit volume, active/closed counts, debt-to-credit exposure, and max overdue days[cite: 7, 20].
2. **Previous Internal Applications (`previous_application.csv` — 1,670,214 rows):** 9 attributes capturing historical approval/refusal counts, bank refusal rates, requested vs. granted amounts, and application recency[cite: 7, 22].

*Relational ablation results from `results/ablation_study.csv`:*[cite: 23]

| Relational Feature Store Layer[cite: 23] | Feature Count[cite: 23] | 5-Fold PR-AUC[cite: 23] | 5-Fold ROC-AUC[cite: 23] | Gini Index[cite: 23] | KS Separation[cite: 23] | Relative PR-AUC Lift[cite: 23] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Application-Only (Champion)**[cite: 23] | 108[cite: 23] | 0.246802[cite: 23] | 0.762824[cite: 23] | 0.525647[cite: 23] | 0.395070[cite: 23] | Benchmark (0.00%)[cite: 23] |
| **+ External Bureau Tradelines**[cite: 23] | 119[cite: 23] | 0.253641[cite: 23] | 0.767174[cite: 23] | 0.534349[cite: 23] | 0.398410[cite: 23] | +2.77%[cite: 23] |
| **+ Previous Internal History**[cite: 23] | 117[cite: 23] | 0.250579[cite: 23] | 0.764951[cite: 23] | 0.529903[cite: 23] | 0.397022[cite: 23] | +1.53%[cite: 23] |
| **+ Full Multi-Table Store (Both)**[cite: 23] | **128**[cite: 23] | **0.257005**[cite: 23] | **0.768770**[cite: 23] | **0.537540**[cite: 23] | **0.401399**[cite: 23] | **+4.13%**[cite: 23] |

---

## Model Governance & Explainability Audit

### 1. Interpretability vs. Discrimination Trade-Off
* **The Quantitative Lift:** XGBoost achieves **+5.43% higher PR-AUC** and **+5.38% higher ROC-AUC** than the linear baseline, reducing expected credit losses by **$5,861.12 per applicant** (95% CI: `[$5,402, $6,295]`)[cite: 8, 9].
* **Bridging the Black-Box Gap via SHAP:** Rather than sacrificing interpretability, we integrated **TreeExplainer** across 5,000 stratified holdout applicants to compute exact additive Shapley values[cite: 7]:
  $$\ln\left(\frac{p_i}{1 - p_i}\right) = \phi_0 + \sum_{j=1}^M \phi_{i,j}$$
* **Linear vs. Non-Linear Concordance:** An audit across 95 shared continuous attributes confirmed a **Spearman rank correlation of $\rho = 0.4772$ ($p = 1.0116 \times 10^{-6}$)** between linear model weights and mean absolute SHAP values[cite: 19]. Both model families agree on core risk drivers (`EXT_SOURCE_MEAN`, `CREDIT_TERM`, `EXT_SOURCE_3`, `YEARS_EMPLOYED`, `LOAN_TO_INCOME_RATIO`), proving the champion captures true structural portfolio signals[cite: 19].

### 2. Representative Adverse Action Notices (`results/shap_reason_codes.csv`)[cite: 15]

| Applicant Cohort[cite: 15] | Calibrated Risk[cite: 15] | Action[cite: 15] | Top Factor 1[cite: 15] | Top Factor 2[cite: 15] | Top Factor 3[cite: 15] | Plain-Language Adverse Action Disclosure[cite: 15] |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **High-Risk Case**[cite: 15] | 68.97%[cite: 15] | **Decline**[cite: 15] | `EXT_SOURCE_MEAN` ($\phi = +1.19$)[cite: 15] | `EXT_SOURCE_3` ($\phi = +0.66$)[cite: 15] | `EXT_SOURCE_2` ($\phi = +0.48$)[cite: 15] | Multi-agency external credit bureau composite rating falls significantly below prime standards[cite: 15]. |
| **Borderline Case**[cite: 15] | 15.70%[cite: 15] | **Approve**[cite: 15] | `EXT_SOURCE_MEAN` ($\phi = +0.90$)[cite: 15] | `CREDIT_TERM` ($\phi = +0.24$)[cite: 15] | `EXT_SOURCE_2` ($\phi = +0.11$)[cite: 15] | Approved under cost-optimal cutoff ($t^*=16.0\%$), but payment-to-credit term structure indicates an elevated amortization burden[cite: 10, 15]. |
| **Low-Risk Prime**[cite: 15] | 0.00%[cite: 15] | **Approve**[cite: 15] | `AMT_ANNUITY` ($\phi = +0.08$)[cite: 15] | `ORGANIZATION_TYPE` ($\phi = +0.07$)[cite: 15] | `AMT_GOODS_PRICE` ($\phi = +0.06$)[cite: 15] | Prime approval; minimal default risk contribution across all debt and demographic factors[cite: 15]. |

> **Illustrative Regulatory Compliance Disclaimer:**
> The Adverse Action reason codes generated here are model-driven Shapley approximations designed to reflect consumer disclosures under the **Fair Credit Reporting Act (FCRA § 615)** and **Equal Credit Opportunity Act (ECOA Reg B)**[cite: 7, 8]. Because Home Credit is an international research dataset rather than an active US regulated depository institution, these reason codes serve as an illustrative governance demonstration.

---

## Class Imbalance Strategy Benchmark (`results/imbalance_study.csv`)[cite: 14]

We empirically tested whether synthetic minority oversampling (SMOTE) improves rank discrimination once probabilities are calibrated and cost-optimized[cite: 7]. Across 5 benchmark runs:[cite: 7, 14]

| Imbalance Strategy[cite: 14] | Model Architecture[cite: 14] | 5-Fold PR-AUC[cite: 14] | 5-Fold ROC-AUC[cite: 14] | Raw Brier[cite: 14] | Optimal $t^*$[cite: 14] | Expected Cost / Applicant[cite: 14] | % Cost vs Approve All[cite: 14] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (Unweighted)**[cite: 14] | LR (full)[cite: 14] | 0.2276[cite: 14] | 0.7476[cite: 14] | 0.0685[cite: 14] | 0.170[cite: 14] | $17,755.61[cite: 14] | 87.67%[cite: 14] |
| **Class Weights (Balanced)**[cite: 14] | LR (full)[cite: 14] | 0.2257[cite: 14] | 0.7475[cite: 14] | 0.2031[cite: 14] | 0.170[cite: 14] | $17,805.17[cite: 14] | 87.92%[cite: 14] |
| **SMOTE (ratio=0.5)**[cite: 14] | LR (full)[cite: 14] | 0.2228[cite: 14] | 0.7419[cite: 14] | 0.1243[cite: 14] | 0.165[cite: 14] | $17,810.89[cite: 14] | 87.95%[cite: 14] |
| **None (Unweighted)**[cite: 14] | Champion (XGBoost)[cite: 14] | 0.2470[cite: 14] | 0.7645[cite: 14] | 0.0676[cite: 14] | 0.165[cite: 14] | $17,340.83[cite: 14] | 85.62%[cite: 14] |
| **Loss Weights (scale_pos_weight)**[cite: 14] | Champion (XGBoost)[cite: 14] | 0.2468[cite: 14] | 0.7628[cite: 14] | 0.1804[cite: 14] | 0.160[cite: 14] | $17,338.76[cite: 14] | 85.61%[cite: 14] |

1. **Ranking Invariance:** PR-AUC for unweighted Logistic Regression ($0.2276$) was slightly higher than both class-weighted ($0.2257$) and SMOTE-augmented ($0.2228$) models[cite: 14].
2. **Cost Convergence:** Post-calibration expected losses converged within **0.3%** across all three linear variants ($17,755.61 vs $17,810.89) and within **$2.07** on XGBoost ($17,340.83 vs $17,338.76)[cite: 14].
3. **Model Risk Takeaway:** Resampling serves as an implicit decision threshold shifter[cite: 7]. In high-dimensional credit data with binary flags and ordinal features, synthetic oversampling (SMOTE) offers no risk advantage over native loss weighting combined with post-hoc probability calibration[cite: 7].

---

## Instructions to Reproduce Pipeline

1. **Clone Repository & Set Up Virtual Environment:**
   ```bash
   git clone [https://github.com/your-username/Credit_Risk_Project.git](https://github.com/your-username/Credit_Risk_Project.git)
   cd Credit_Risk_Project
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Stage Raw Data Files:**
   Place `application_train.csv`, `bureau.csv`, and `previous_application.csv` in `data/raw/`.
3. **Execute End-to-End Quantitative Pipeline:**
   Launch Jupyter and run all cells in `Notebooks/02_modeling_extension.ipynb`:
   ```bash
   jupyter lab Notebooks/02_modeling_extension.ipynb
   ```
   All scorecards, threshold matrices, confidence intervals, and visualization assets will automatically populate into `results/` and `assets/`.
