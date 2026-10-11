# Credit Risk Analysis & Loan Default Prediction

An end-to-end credit-risk analytics project using the Home Credit Default Risk dataset. The project compares interpretable and tree-based classifiers, calibrates default probabilities, evaluates an illustrative cost-sensitive decision rule, and examines whether historical bureau and previous-application aggregates add predictive signal.

> **Result status:** The modeling extension has been rerun, and the regenerated result artifacts have passed the project's structural and consistency verification checks. The frozen v1 baseline was also rerun successfully. Current model metrics, decision comparisons, calibration diagnostics, and feature-ablation results are documented in the generated artifacts under results/. These results are based on a single dataset and a fixed train/test split; they do not establish fairness, regulatory compliance, causality, or production readiness.

## Business question

Can application information and historical relationship aggregates help rank applicants by default risk, and how does a decision threshold change the trade-off between approving a future defaulter and declining a non-defaulter?

The target is `TARGET` (1 = default, 0 = non-default). The main source is the Home Credit Default Risk application table. Supplementary tables are used only for applicant-level historical aggregates.

## Methods

1. **Data preparation:** deterministic feature engineering, schema checks, missing-value handling within model pipelines, and removal of the applicant identifier.
2. **Feature policy:** the default policy excludes `CODE_GENDER`, `NAME_FAMILY_STATUS`, `NAME_TYPE_SUITE`, `CNT_CHILDREN`, and `CNT_FAM_MEMBERS`. `AGE_YEARS` remains available to the model but is not mapped to an explanation. `NAME_INCOME_TYPE` remains and may act as a proxy for sensitive circumstances. This is an illustrative feature policy, **not a fair-lending test or a claim of legal compliance**.
3. **Model comparison:** five-fold stratified cross-validation on the training partition, using out-of-fold PR-AUC as the primary ranking metric. ROC-AUC, Gini, KS, Brier score, log-loss, and fold variability are secondary diagnostics. Hyperparameters are fixed; no hyperparameter search is claimed. For weighted boosted-tree CV, `scale_pos_weight` is recalculated from each fold's training labels only; the final refit uses the full training-partition class ratio.
4. **Probability calibration:** isotonic calibration is cross-fitted for training-side decisions. The final calibration mapping is fitted to the complete training OOF predictions and applied to the untouched test predictions. Reported test calibration metrics should be calculated on the holdout set, not inferred from the calibrator's training data.
5. **Decision-cost illustration:** the notebook assumes LGD = 0.45, an opportunity cost of declining a non-defaulter = 0.09, and therefore a cost ratio of 5. EAD is approximated by `AMT_CREDIT`. These are **illustrative economic assumptions**, not retail Basel F-IRB parameters. Costs are expressed in dataset currency units; no currency is asserted. The metric is called *expected decision cost* and should not be interpreted as a realized-loss estimate.
6. **Decision threshold:** the threshold is selected from cross-fitted calibrated training predictions. The holdout test set is not used to select the model or threshold. The raw weighted-score threshold of 0.50 is retained only as a reference, not as a calibrated policy.
7. **Explainability:** SHAP is used to inspect drivers of the selected tree model's raw output. These explanations are not calibrated-probability explanations and are **not legally validated adverse-action reason codes**. No regulatory compliance conclusion is made.
8. **Relational ablation:** bureau and previous-application history are aggregated per applicant without using `TARGET`. Their predictive lift must be checked fold by fold before it is described as robust.

## Repository layout

```text
Credit_Risk_Project/
├── Notebooks/
│   ├── credit_risk_analysis.ipynb       # Frozen v1 baseline
│   └── modeling_extension.ipynb         # Model comparison and decision analysis
├── src/
│   ├── config.py                        # Paths, seeds, feature policy, assumptions
│   ├── features.py                      # Application feature engineering/pipelines
│   ├── costs.py                         # Calibration and expected decision cost
│   ├── evaluate.py                      # Cross-validation and bootstrap evaluation
│   └── bureau_features.py               # Bureau/history aggregations
├── results/                             # Generated CSV/NPY tables and run manifest
├── scripts/
│   ├── verify_notebook.py               # Notebook syntax and stale-output checks
│   └── verify_results.py                # Result-integrity checks after a clean rerun
├── tests/test_core.py                   # Dataset-independent helper tests
├── .github/workflows/core-checks.yml    # CI on Python 3.12 and 3.14
├── assets/                              # Generated evaluation and SHAP figures
├── sql/                                 # PostgreSQL analytical artifacts
├── dashboards/                          # Power BI dashboard (if included locally)
├── requirements.txt
└── README.md
```

Raw data are not included. Place the official files in `data/raw/`:

- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`

The project ignores raw data files so they are not committed to Git.

## SQL and Power BI reporting

The Python workflow handles model development and generated evaluation artifacts. PostgreSQL and Power BI provide the separate analyst/stakeholder view:

- `sql/core_artifacts.sql` creates `analytics.v_credit_risk_engineered` and includes cohort-level default-rate queries, income tiers, organization-type comparisons, age-cohort analysis, and a defaulter/non-defaulter financial-burden summary.
- Before running the SQL file, import the cleaned application data into PostgreSQL as `public.credit_risk_data` using lower-case column identifiers. The age and loan-to-income queries read from the engineered view rather than assuming derived columns already exist in the base table.
- `dashboards/credit_risk_dashboard.pbix` is the Power BI artifact. The screenshots below are included as dashboard previews; refresh the underlying tables and re-check unit labels after regenerating the data/model outputs.

### Dashboard previews

![Credit risk dashboard overview](assets/dashboard_overview.png)

![Credit risk underwriting dashboard](assets/dashboard_underwiring.png)

## Business questions supported by the project

The current numeric tables below are generated from the verified result artifacts. Performance comparisons are specific to this dataset and fixed train/test split.

| Business question | Source of the current answer |
|---|---|
| How does default prevalence vary across borrower and income cohorts? | SQL queries in `sql/core_artifacts.sql` and the Power BI cohort visuals |
| How do model quality and calibration compare across candidates? | `results/cv_metrics.csv`, `results/cv_fold_metrics.csv`, and `results/test_calibration.csv` |
| How does the selected decision threshold change approval and risk outcomes? | `results/test_operating_points.csv` and `results/headline_costs.csv` |
| Do bureau and previous-application aggregates add predictive signal consistently across folds? | `results/ablation_study.csv` and `results/ablation_fold_diffs.csv` |
| How stable are tree-SHAP and logistic-regression feature rankings after reducing collinear feature triplets? | `results/shap_lr_rank_correlation.csv` and `results/shap_lr_rank_correlation_robust.csv` |

## Reproducing the analysis

Use a fresh Python environment compatible with the packages in `requirements.txt`.

### Windows PowerShell

```powershell
git clone https://github.com/godiyajatin-ops/Credit_Risk_Project.git
cd Credit_Risk_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
git clone https://github.com/godiyajatin-ops/Credit_Risk_Project.git
cd Credit_Risk_Project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

1. Put the three raw CSV files in `data/raw/`.
2. Open `Notebooks/credit_risk_analysis.ipynb` from the repository root in Jupyter and run all cells. Confirm the baseline reconciliation assertion passes; the core baseline model procedure is unchanged.
3. Open `Notebooks/modeling_extension.ipynb`, restart its kernel, and run all cells in order. This regenerates scorecards and figures under `results/` and `assets/`, including a run manifest.
4. Run the verification script from the repository root:

   ```bash
   python scripts/verify_results.py
   ```

The full notebook includes five-fold model fitting and SHAP computation, so runtime depends on hardware and installed library versions.

## Results and interpretation

> This section is generated from the current CSV files in `results/` by `scripts/render_readme_results.py`. Rerun the modeling extension and this script after any change to the data, split, model-selection rule, or evaluation code.

### Cross-validation scorecard

| Model | Input columns | PR-AUC (mean ± SD) | ROC-AUC (mean ± SD) | Gini | KS | Brier | Log-loss | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 104 | 0.2457 ± 0.0054 | 0.7621 ± 0.0024 | 0.5243 | 0.3938 | 0.1806 | 0.5375 | 37.87 |
| LightGBM | 104 | 0.2457 ± 0.0053 | 0.7617 ± 0.0021 | 0.5235 | 0.3921 | 0.1762 | 0.5251 | 30.01 |
| Logistic regression (full) | 222 | 0.2255 ± 0.0071 | 0.7473 ± 0.0029 | 0.4945 | 0.3690 | 0.2033 | 0.5927 | 38.88 |
| Random forest | 222 | 0.2212 ± 0.0059 | 0.7446 ± 0.0022 | 0.4892 | 0.3638 | 0.1840 | 0.5516 | 154.73 |
| Logistic regression (frozen v1) | 5 | 0.1925 ± 0.0056 | 0.7149 ± 0.0029 | 0.4298 | 0.3152 | 0.2140 | 0.6192 | 1.10 |

The selected champion in the recorded run is **xgboost**; the runner-up is **lightgbm**. The paired holdout confidence intervals include zero for both cost savings and PR-AUC lift; there is no conclusive holdout winner between XGBoost and LightGBM.  The paired PR-AUC differences favor the champion in 2 of 5 folds, which is not a consistent fold-level win. For the weighted boosted-tree models, CV Brier and log-loss are diagnostics on raw model scores; use the holdout calibration table below for calibrated-probability quality.

### Holdout decision costs and uncertainty

Point expected decision costs come from `headline_costs.csv`; the 95% bootstrap confidence intervals are from `test_metrics.csv`. Costs are in dataset currency units per applicant, under the stated illustrative assumptions.

| Policy | Threshold | Point expected cost / applicant | 95% bootstrap CI for cost | ROC-AUC (95% CI) | PR-AUC (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Approve everyone | 1.000 | 20,306.50 | 19,655.02–20,939.49 | 0.7678 (0.7612–0.7744) | 0.2536 (0.2423–0.2652) |
| Baseline v1 (threshold 0.50) | 0.500 | 22,877.79 | 22,435.29–23,339.13 | 0.7154 (0.7080–0.7225) | 0.1971 (0.1872–0.2068) |
| Baseline v1 (OOF-calibrated cost-optimal) | 0.175 | 18,424.42 | 17,882.50–18,965.14 | 0.7151 (0.7077–0.7222) | 0.1918 (0.1825–0.2012) |
| Champion xgboost (raw weighted score; reference only) | 0.500 | 19,469.71 | 19,015.03–19,910.56 | 0.7679 (0.7612–0.7745) | 0.2592 (0.2475–0.2710) |
| Champion xgboost (OOF-calibrated cost-optimal) | 0.165 | 17,037.71 | 16,517.88–17,533.80 | 0.7678 (0.7612–0.7744) | 0.2536 (0.2423–0.2652) |
| Runner-up lightgbm (OOF-calibrated cost-optimal) | 0.160 | 17,000.08 | 16,487.59–17,522.66 | 0.7672 (0.7604–0.7739) | 0.2510 (0.2393–0.2625) |

### Cost-saving comparisons and decomposition

| Comparison | Saving per applicant (dataset currency units) | Saving relative to reference policy |
| --- | --- | --- |
| Champion vs approve-all | 3,268.80 | 16.10% |
| Champion vs v1 at threshold 0.50 | 5,840.08 | 25.53% |
| Champion vs v1 at its own OOF-selected threshold | 1,386.71 | 7.53% |
| Threshold effect: v1 default to v1 tuned | 4,453.37 | 19.47% |
| Model effect: v1 tuned to champion | 1,386.71 | 6.06% |

The threshold effect compares the frozen v1 policy at 0.50 with v1's own threshold selected from cross-fitted training predictions. The model effect compares that tuned v1 policy with the selected champion. The cost ratio, LGD reference value, opportunity-cost assumption, and EAD proxy are illustrative, not Basel retail parameters.

### Holdout operating points

| Policy | Threshold | Approval rate | Default rate among approved | Recall | Precision | Expected cost / applicant |
| --- | --- | --- | --- | --- | --- | --- |
| Approve everyone | 1.000 | 100.00% | 8.07% | 0.00% | 0.00% | 20,306.50 |
| Baseline v1 (threshold 0.50) | 0.500 | 65.05% | 4.50% | 63.73% | 14.72% | 22,877.79 |
| Baseline v1 (OOF-calibrated cost-optimal) | 0.175 | 90.74% | 6.45% | 27.51% | 23.99% | 18,424.42 |
| Champion xgboost (raw weighted score; reference only) | 0.500 | 70.18% | 3.92% | 65.92% | 17.85% | 19,469.71 |
| Champion xgboost (OOF-calibrated cost-optimal) | 0.165 | 88.48% | 5.61% | 38.55% | 27.00% | 17,037.71 |
| Runner-up lightgbm (OOF-calibrated cost-optimal) | 0.160 | 86.43% | 5.36% | 42.58% | 25.32% | 17,000.08 |

The `Champion raw weighted score (reference only)` row is a diagnostic comparator, not a calibrated decision policy. Thresholds for the calibrated policies are selected using training-side cross-fitted predictions, not the holdout labels.

### Threshold sensitivity

| Cost ratio r | Analytic threshold | Selected training-side threshold | Approval rate | Default rate among approved | Recall | Precision | Expected cost / applicant | Cost as % of approve-all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0 | 0.3333 | 0.320 | 97.63% | 7.29% | 11.81% | 40.21% | 19,729.43 | 97.42% |
| 3.0 | 0.2500 | 0.255 | 95.80% | 6.90% | 18.11% | 34.84% | 19,059.85 | 94.11% |
| 5.0 | 0.1667 | 0.165 | 88.80% | 5.78% | 36.46% | 26.27% | 17,382.62 | 85.83% |
| 10.0 | 0.0909 | 0.090 | 66.94% | 3.73% | 69.04% | 16.86% | 13,780.79 | 68.05% |
| 20.0 | 0.0476 | 0.045 | 43.59% | 2.44% | 86.82% | 12.43% | 9,471.95 | 46.77% |

### Holdout calibration diagnostics

| Score type | Brier score | Mean predicted probability | Observed default prevalence | Unique scores | Exact zeros | Exact ones |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost: raw model score (not calibrated) | 0.1817 | 38.79% | 8.07% | 61412 | 0 | 0 |
| xgboost: isotonic-calibrated probability | 0.0671 | 8.18% | 8.07% | 126 | 0 | 0 |
| lightgbm: raw model score (not calibrated) | 0.1785 | 38.07% | 8.07% | 61503 | 0 | 0 |
| lightgbm: isotonic-calibrated probability | 0.0672 | 8.24% | 8.07% | 124 | 0 | 0 |

Calibrated probabilities use the configured floor/cap of 1e-4 and 1 − 1e-4. Calibration quality is evaluated on the holdout set; exact-zero and exact-one counts are shown as diagnostics.

### Relational feature-store ablation

| Feature set | Feature count | PR-AUC (mean ± SD) | ROC-AUC (mean ± SD) | Relative PR-AUC lift vs application-only | Folds with PR-AUC improvement |
| --- | --- | --- | --- | --- | --- |
| Application-only | 104 | 0.2457 ± 0.0054 | 0.7621 ± 0.0024 | 0.00% | Application-only baseline |
| + Bureau | 115 | 0.2534 ± 0.0061 | 0.7669 ± 0.0024 | 3.14% | 5/5 |
| + Previous | 113 | 0.2513 ± 0.0047 | 0.7663 ± 0.0016 | 2.27% | 5/5 |
| + Both (Full Store) | 124 | 0.2577 ± 0.0063 | 0.7704 ± 0.0023 | 4.88% | Application-only baseline |

The champion model used for the headline results is trained on application-only features. Bureau and previous-application feature sets are ablations, not the deployed/selected champion input. These results are predictive comparisons on this split, not causal evidence.

### Feature-policy sensitivity

| Policy variant | Feature count | PR-AUC | ROC-AUC | PR-AUC delta vs configured policy | ROC-AUC delta vs configured policy |
| --- | --- | --- | --- | --- | --- |
| Configured feature exclusions | 104 | 0.2457 | 0.7621 | 0.0000 | 0.0000 |
| Retain four additional sensitive/proxy features (CODE_GENDER still excluded) | 108 | 0.2468 | 0.7628 | 0.0011 | 0.0007 |

The configured exclusions are `CODE_GENDER`, `NAME_FAMILY_STATUS`, `NAME_TYPE_SUITE`, `CNT_CHILDREN`, and `CNT_FAM_MEMBERS`. `AGE_YEARS` remains available to the model but is excluded from illustrative risk-driver explanations. `NAME_INCOME_TYPE` remains and may act as a proxy. This is not a fairness test or legal-compliance conclusion.

### Thin-file segmentation

| Segment | Applicants | Default prevalence | ROC-AUC | PR-AUC | Approval rate | Default rate among approved | Expected cost / applicant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No linked bureau record | 8776 | 10.14% | 0.7438 | 0.2789 | 83.61% | 6.79% | 19,229.76 |
| Linked bureau record | 52727 | 7.73% | 0.7708 | 0.2484 | 89.28% | 5.42% | 16,672.86 |

These segment results are descriptive and based on the fixed holdout split; no subgroup fairness conclusion is implied.

### Paired holdout comparisons

| Comparison | Cost saving (mean, 95% CI) | PR-AUC lift (mean, 95% CI) |
| --- | --- | --- |
| Champion vs v1 at threshold 0.50 | 5,852.71 (5,434.09–6,297.23) | 0.0565 (0.0488–0.0641) |
| Champion vs approve-all | 3,262.41 (2,871.03–3,633.47) | 0.0000 (0.0000–0.0000) |
| Champion vs v1 own cost-optimal threshold | 1,380.78 (1,075.59–1,685.03) | 0.0618 (0.0541–0.0695) |
| Champion vs runner-up | -39.61 (-229.34–133.26) | 0.0026 (-0.0005–0.0055) |

The XGBoost-versus-LightGBM comparison is treated as **inconclusive** when paired intervals include zero. Do not manually relabel the champion; the `run_manifest.json` records the model selected from cross-validation.

### Explainability and ranking agreement

| Ranking comparison | Spearman correlation | Shared features |
| --- | --- | --- |
| All shared numeric features | 0.5004 | 93 |
| One representative per housing AVG/MODE/MEDI triplet | 0.6121 | 75 |

The correlations are descriptive. Collinearity, including correlated credit-amount and housing-area features, can produce different coefficient and SHAP rankings. SHAP values in this project explain the selected tree model's raw weighted score, not calibrated probability or legal adverse-action reasons.

**Illustrative top risk drivers (not legal notices)**

| Applicant cohort | Rank | Feature | SHAP attribution | Illustrative description |
| --- | --- | --- | --- | --- |
| High-Risk | 1 | EXT_SOURCE_MEAN | 1.1376 | Composite external risk-score feature contributes to estimated default risk |
| High-Risk | 2 | EXT_SOURCE_3 | 0.6067 | External risk score 3 contributes to the model's estimated default risk |
| High-Risk | 3 | EXT_SOURCE_2 | 0.4104 | External risk score 2 contributes to the model's estimated default risk |
| Borderline Near Threshold | 1 | CREDIT_TO_GOODS | 0.3884 | Loan amount exceeds the appraised market price of financed goods |
| Borderline Near Threshold | 2 | CREDIT_TERM | 0.2577 | Loan payment-to-credit term structure indicates elevated amortization burden |
| Borderline Near Threshold | 3 | DEF_60_CNT_SOCIAL_CIRCLE | 0.1890 | Adverse risk contribution from attribute: DEF_60_CNT_SOCIAL_CIRCLE |
| Low-Risk | 1 | DAYS_REGISTRATION | 0.0334 | Adverse risk contribution from attribute: DAYS_REGISTRATION |
| Low-Risk | 2 | BASEMENTAREA_MEDI | 0.0308 | Adverse risk contribution from attribute: BASEMENTAREA_MEDI |
| Low-Risk | 3 | FLAG_DOCUMENT_5 | 0.0299 | Adverse risk contribution from attribute: FLAG_DOCUMENT_5 |

### Class-imbalance sensitivity

| Strategy | Model | ROC-AUC | PR-AUC | Selected threshold | Expected cost / applicant | Cost as % of approve-all |
| --- | --- | --- | --- | --- | --- | --- |
| None (Unweighted) | LR (full) | 0.7473 | 0.2273 | 0.165 | 17,798.34 | 87.88% |
| None (Unweighted) | XGBoost (unweighted) | 0.7644 | 0.2476 | 0.160 | 17,328.26 | 85.56% |
| Class Weights (Balanced) | LR (full) | 0.7473 | 0.2255 | 0.170 | 17,789.80 | 87.84% |
| Loss Weights (scale_pos_weight) | Champion (xgboost) | 0.7621 | 0.2457 | 0.165 | 17,382.62 | 85.83% |
| SMOTE (ratio=0.5) | LR (full) | 0.7420 | 0.2228 | 0.160 | 17,870.23 | 88.24% |

SMOTE was tested with logistic regression only. A tree-plus-SMOTE comparison was not run, and SMOTE can create fractional one-hot values; treat that experiment as exploratory. The cost and ranking comparisons are specific to this dataset, split, and assumed cost function.

### Key business insights

| Insight | Result from generated artifacts | Interpretation / limit |
| --- | --- | --- |
| Model comparison | xgboost selected; lightgbm runner-up. | Paired cost and PR-AUC intervals include zero: True. |
| Cost-sensitive decision | Threshold 0.165; approval 88.48%; expected cost/applicant 17,037.71. | Threshold is selected from cross-fitted training predictions; cost assumptions are illustrative. |
| Relational features | Combined bureau + previous-application features: PR-AUC 0.2577, relative PR-AUC lift 4.88%. | Feature-store variants are ablations; they are not the selected champion's input. |
| Thin-file segment | No linked bureau record prevalence 10.14%; linked-bureau prevalence 10.14%. | Descriptive fixed-holdout comparison, not a fairness or causal finding. |
| Feature-policy sensitivity | Retaining four additional sensitive/proxy features changes PR-AUC by 0.00108 vs the configured exclusion policy. | Predictive lift alone does not justify re-adding excluded fields; this is not compliance testing. |

The SQL and Power BI layer supports cohort/default-rate exploration and reporting, but it is separate from the model-training pipeline. The Home Credit research dataset is not evidence of validated underwriting for a specific US lender or any other real lender.

### Result artifact index

- `cv_metrics.csv`, `cv_fold_metrics.csv`: cross-validation and fold-level model metrics.
- `headline_costs.csv`, `test_metrics.csv`, `test_paired_diffs.csv`, `test_operating_points.csv`, `threshold_table.csv`: holdout decision-cost points, confidence intervals, paired comparisons, operating points, and threshold sensitivity.
- `test_calibration.csv`, `test_scored.csv`: calibration diagnostics and locked holdout predictions.
- `ablation_study.csv`, `ablation_fold_diffs.csv`, `fair_lending_ablation.csv`, `thin_file_segments.csv`: relational and feature-policy ablations, fold deltas, and segment metrics.
- `imbalance_study.csv`: tested class-imbalance configurations.
- `shap_reason_codes.csv`, `shap_lr_rank_correlation.csv`, `shap_lr_rank_correlation_robust.csv`, `rank_correlation.txt`: illustrative risk drivers and ranking comparisons.
- `run_manifest.json`: selected champion/runner-up, data split, feature counts, assumptions, thresholds, and package versions.

## Limitations

- One dataset and one fixed train/test split are used; external and temporal validation are not included.
- `requirements.txt` - `requirements.txt` pins the listed direct and runtime dependencies to the versions used for the recorded Python 3.14.7 run. It does not pin every transitive dependency or include platform-specific hashes, so it is not a complete reproducible lockfile. The run manifest records selected package versions; exact bit-for-bit reproduction may still vary across operating systems and hardware.
- Hyperparameters are fixed rather than tuned.
- The cost function is a simplified decision-analysis scenario. It omits funding costs, capital requirements, recovery timing, pricing, exposure dynamics, and many operational constraints.
- Excluding selected attributes does not establish fairness. Dedicated subgroup metrics, proxy analysis, legal review, and governance validation have not been completed.
- SHAP explains the model output under the chosen explainer configuration; it does not, by itself, establish causality or satisfy adverse-action disclosure requirements.
- Better ranking metrics do not prove that the resulting underwriting policy is suitable for a real lender.

## License and data

Use the dataset under the terms provided by its original publisher. Confirm the licensing and redistribution terms for any external dataset, dashboard, or asset before redistributing the complete repository.
