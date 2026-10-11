"""Regenerate the README results section from the project's current results/*.csv files.

Run from the repository root with: python scripts/render_readme_results.py
The script replaces the README section from '## Results and interpretation' up to
(but not including) '## Limitations'. All reported metric values come from CSVs.
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"


def load(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Required result file missing: {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def num(x, digits: int = 4) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):,.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x, digits: int = 2) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(x)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(clean(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def first_match(frame: pd.DataFrame, predicate, label: str) -> pd.Series:
    matches = frame[frame.iloc[:, 0].astype(str).map(predicate)]
    if matches.empty:
        raise ValueError(f"Could not match '{label}' in {list(frame.columns)}")
    return matches.iloc[0]


def metric_row_for_policy(test: pd.DataFrame, policy: str) -> pd.Series:
    names = test["model_setup"].astype(str).str.lower()
    p = policy.lower()
    if "approve everyone" in p:
        mask = names.str.contains("approve everyone")
    elif "baseline v1" in p and "0.50" in p:
        mask = names.str.contains("baseline v1") & names.str.contains("5-feature")
    elif "baseline v1" in p and "oof-calibrated" in p:
        mask = names.str.contains("baseline v1") & names.str.contains("oof-calibrated")
    elif "raw weighted score" in p:
        mask = names.str.contains("raw weighted score") | (names.str.contains("champion") & names.str.contains("t=0.5"))
    elif "runner-up" in p:
        mask = names.str.contains("runner-up")
    elif "champion" in p:
        mask = names.str.contains("champion") & ~names.str.contains("t=0.5") & ~names.str.contains("raw weighted")
    else:
        mask = names.str.contains(re.escape(p))
    subset = test[mask]
    if subset.empty:
        raise ValueError(f"Could not match headline policy to test metric row: {policy!r}; found {test['model_setup'].tolist()}")
    return subset.iloc[0]


def main() -> None:
    cv = load("cv_metrics.csv")
    headline = load("headline_costs.csv")
    test = load("test_metrics.csv")
    operating = load("test_operating_points.csv")
    calibration = load("test_calibration.csv")
    threshold = load("threshold_table.csv")
    ablation = load("ablation_study.csv")
    fold_ablation = load("ablation_fold_diffs.csv")
    fair = load("fair_lending_ablation.csv")
    thin = load("thin_file_segments.csv")
    paired = load("test_paired_diffs.csv")
    fold_pair = load("champion_runnerup_fold_diffs.csv")
    imbalance = load("imbalance_study.csv")
    reasons = load("shap_reason_codes.csv")
    manifest_path = RESULTS / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Required result file missing: results/run_manifest.json")

    # Confirm stale/unsafe explanation output cannot be silently published.
    feature_names = reasons["feature_name"].astype(str).str.split("__").str[-1]
    if feature_names.eq("AGE_YEARS").any():
        raise ValueError("shap_reason_codes.csv still contains AGE_YEARS. Rerun the edited risk-driver cell first.")

    cv_rows = []
    input_cols = {"lr_5feat": 5, "lr_full": 222, "random_forest": 222, "lightgbm": 104, "xgboost": 104}
    display_model = {"lr_5feat": "Logistic regression (frozen v1)", "lr_full": "Logistic regression (full)",
                     "random_forest": "Random forest", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
    for _, r in cv.iterrows():
        m = str(r["model"])
        cv_rows.append([
            display_model.get(m, m), input_cols.get(m, "—"),
            f"{num(r['pr_auc'])} ± {num(r['pr_auc_std'])}",
            f"{num(r['roc_auc'])} ± {num(r['roc_auc_std'])}",
            num(r["gini"]), num(r["ks_stat"]), num(r["brier_score"]), num(r["log_loss"]), num(r["train_time_sec"], 2),
        ])

    # Point costs are from headline_costs.csv; intervals are paired/bootstrap outputs from test_metrics.csv.
    cost_rows = []
    for _, r in headline.iterrows():
        label = str(r["policy"])
        tr = metric_row_for_policy(test, label)
        interval = f"{num(tr['expected_cost_ci_lower'], 2)}–{num(tr['expected_cost_ci_upper'], 2)}"
        cost_rows.append([
            label, num(r["decision_threshold"], 3), num(r["cost_per_applicant"], 2), interval,
            f"{num(tr['roc_auc_mean'])} ({num(tr['roc_auc_ci_lower'])}–{num(tr['roc_auc_ci_upper'])})",
            f"{num(tr['pr_auc_mean'])} ({num(tr['pr_auc_ci_lower'])}–{num(tr['pr_auc_ci_upper'])})",
        ])

    def policy_contains(term: str) -> pd.Series:
        return headline["policy"].astype(str).str.lower().str.contains(term.lower(), regex=False)
    def get_cost(mask: pd.Series, description: str) -> float:
        sub = headline[mask]
        if sub.empty:
            raise ValueError(f"Could not locate policy for {description} in headline_costs.csv")
        return float(sub.iloc[0]["cost_per_applicant"])
    c_all = get_cost(policy_contains("Approve everyone"), "approve-all")
    c_v1 = get_cost(policy_contains("Baseline v1 (threshold 0.50)"), "v1 default threshold")
    c_v1t = get_cost(policy_contains("Baseline v1 (OOF-calibrated cost-optimal)"), "v1 tuned threshold")
    c_ch = get_cost(policy_contains("Champion") & policy_contains("OOF-calibrated cost-optimal"), "champion optimal threshold")
    savings = [
        ["Champion vs approve-all", num(c_all - c_ch, 2), pct((c_all - c_ch) / c_all)],
        ["Champion vs v1 at threshold 0.50", num(c_v1 - c_ch, 2), pct((c_v1 - c_ch) / c_v1)],
        ["Champion vs v1 at its own OOF-selected threshold", num(c_v1t - c_ch, 2), pct((c_v1t - c_ch) / c_v1t)],
        ["Threshold effect: v1 default to v1 tuned", num(c_v1 - c_v1t, 2), pct((c_v1 - c_v1t) / c_v1)],
        ["Model effect: v1 tuned to champion", num(c_v1t - c_ch, 2), pct((c_v1t - c_ch) / c_v1)],
    ]

    op_rows = []
    for _, r in operating.iterrows():
        op_rows.append([str(r["policy"]), num(r["decision_threshold"], 3), pct(r["approval_rate"]),
                        pct(r["default_rate_approved"]), pct(r["recall"]), pct(r["precision"]),
                        num(r["cost_per_applicant"], 2)])

    cal_rows = []
    for _, r in calibration.iterrows():
        cal_rows.append([r["score_type"], num(r["brier_score"]), pct(r["mean_predicted_probability"]),
                         pct(r["observed_prevalence"]), int(r["n_unique_scores"]), int(r["n_exact_zeros"]), int(r["n_exact_ones"])])

    fold_wins = fold_ablation.groupby("feature_set")["pr_auc_delta"].agg(lambda s: f"{int((s > 0).sum())}/{int(s.notna().sum())}")
    abl_rows = []
    for _, r in ablation.iterrows():
        key = str(r["feature_set"])
        abl_rows.append([key, int(r["num_features"]), f"{num(r['pr_auc'])} ± {num(r['pr_auc_std'])}",
                         f"{num(r['roc_auc'])} ± {num(r['roc_auc_std'])}", num(r["relative_pr_lift_pct"], 2) + "%",
                         fold_wins.get(key, "Application-only baseline")])

    fair_rows = []
    for _, r in fair.iterrows():
        fair_rows.append([r["policy"], int(r["num_features"]), num(r["pr_auc"]), num(r["roc_auc"]),
                          num(r["pr_auc_delta_vs_configured"]), num(r["roc_auc_delta_vs_configured"])])

    thin_rows = []
    for _, r in thin.iterrows():
        thin_rows.append([r["segment"], int(r["n_applicants"]), pct(r["default_prevalence"]), num(r["roc_auc"]),
                          num(r["pr_auc"]), pct(r["approval_rate"]), pct(r["default_rate_approved"]),
                          num(r["expected_decision_cost_per_applicant"], 2)])

    paired_rows = []
    for _, r in paired.iterrows():
        paired_rows.append([r["comparison"],
                            f"{num(r['cost_saving_mean'], 2)} ({num(r['cost_saving_ci_lower'], 2)}–{num(r['cost_saving_ci_upper'], 2)})",
                            f"{num(r['pr_auc_lift_mean'])} ({num(r['pr_auc_lift_ci_lower'])}–{num(r['pr_auc_lift_ci_upper'])})"])
    champ_vs_runner = paired[paired["comparison"].astype(str).str.contains("runner-up", case=False)]
    if champ_vs_runner.empty:
        raise ValueError("Champion-versus-runner-up row missing from test_paired_diffs.csv")
    cr = champ_vs_runner.iloc[0]
    ci_cost_crosses_zero = float(cr["cost_saving_ci_lower"]) <= 0 <= float(cr["cost_saving_ci_upper"])
    ci_pr_crosses_zero = float(cr["pr_auc_lift_ci_lower"]) <= 0 <= float(cr["pr_auc_lift_ci_upper"])
    fold_win_count = int((fold_pair["pr_auc_delta_champion_minus_runner_up"] > 0).sum())
    fold_count = int(fold_pair["pr_auc_delta_champion_minus_runner_up"].notna().sum())
    manifest_champion = "xgboost"
    try:
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_champion = str(manifest["champion"])
        runner_up = str(manifest["runner_up"])
        python_version = str(manifest.get("python_version", "recorded in manifest"))
        cv_folds = manifest.get("cv_folds", "recorded in manifest")
    except (ValueError, KeyError):
        runner_up, python_version, cv_folds = "recorded in manifest", "recorded in manifest", "recorded in manifest"

    paired_note = ("The paired holdout confidence intervals include zero for both cost savings and PR-AUC lift; "
                   "there is no conclusive holdout winner between XGBoost and LightGBM. "
                   if ci_cost_crosses_zero and ci_pr_crosses_zero else
                   "See the paired holdout intervals below; do not treat a small model difference as decisive without interval and fold-level context. ")

    # Threshold sensitivity metrics in threshold_table.csv are stored as percentage points already.
    threshold_rows = []
    for _, r in threshold.iterrows():
        threshold_rows.append([num(r["cost_ratio_r"], 1), num(r["analytic_threshold"]), num(r["optimal_threshold"], 3),
                               num(r["approval_rate"], 2) + "%", num(r["default_rate_approved"], 2) + "%",
                               num(r["recall"], 2) + "%", num(r["precision"], 2) + "%",
                               num(r["cost_per_applicant"], 2), num(r["cost_pct_vs_approve_all"], 2) + "%"])

    rank_text = (RESULTS / "rank_correlation.txt").read_text(encoding="utf-8")
    all_match = re.search(r"Spearman rank correlation \(all shared numeric features\):\s*([0-9.]+)", rank_text)
    robust_match = re.search(r"Spearman rank correlation \(one representative per AVG/MODE/MEDI triplet\):\s*([0-9.]+)", rank_text)
    if not all_match or not robust_match:
        raise ValueError("Could not parse both correlations from results/rank_correlation.txt")
    all_count = re.search(r"Evaluated shared numeric features:\s*(\d+)", rank_text)
    robust_count = re.search(r"Evaluated features after triplet reduction:\s*(\d+)", rank_text)
    if not all_count or not robust_count:
        raise ValueError("Could not parse feature counts from results/rank_correlation.txt")
    rank_rows = [["All shared numeric features", all_match.group(1), all_count.group(1)],
                 ["One representative per housing AVG/MODE/MEDI triplet", robust_match.group(1), robust_count.group(1)]]

    imb_rows = []
    for _, r in imbalance.iterrows():
        imb_rows.append([r["strategy"], r["model"], num(r["roc_auc"]), num(r["pr_auc"]), num(r["optimal_threshold"], 3),
                         num(r["cost_per_applicant"], 2), num(r["cost_pct_vs_approve_all"], 2) + "%"])

    reason_rows = []
    for cohort, group in reasons.groupby("applicant_cohort", sort=False):
        for _, r in group.sort_values("risk_factor_rank").head(3).iterrows():
            reason_rows.append([cohort, int(r["risk_factor_rank"]), r["feature_name"], num(r["shap_attribution"]), r["illustrative_risk_driver"]])

    # Condensed analyst-facing key-insights table, derived from the same CSVs.
    champ_oper = operating[operating["policy"].astype(str).str.contains("Champion", case=False) &
                           operating["policy"].astype(str).str.contains("OOF-calibrated cost-optimal", case=False)]
    both_ablation = ablation[ablation["feature_set"].astype(str).str.contains("Both", case=False)]
    no_bureau = thin[thin["segment"].astype(str).str.contains("No linked bureau", case=False)]
    linked_bureau = thin[thin["segment"].astype(str).str.contains("Linked bureau", case=False)]
    fair_keep = fair[fair["policy"].astype(str).str.contains("Retain four additional", case=False)]
    champ_oper_row = champ_oper.iloc[0] if not champ_oper.empty else None
    both_ablation_row = both_ablation.iloc[0] if not both_ablation.empty else None
    no_bureau_row = no_bureau.iloc[0] if not no_bureau.empty else None
    linked_bureau_row = linked_bureau.iloc[0] if not linked_bureau.empty else None
    fair_keep_row = fair_keep.iloc[0] if not fair_keep.empty else None
    insight_rows = [
        ["Model comparison", f"{manifest_champion} selected; {runner_up} runner-up.",
         f"Paired cost and PR-AUC intervals include zero: {ci_cost_crosses_zero and ci_pr_crosses_zero}."],
        ["Cost-sensitive decision",
         (f"Threshold {num(champ_oper_row['decision_threshold'], 3)}; approval {pct(champ_oper_row['approval_rate'])}; "
          f"expected cost/applicant {num(champ_oper_row['cost_per_applicant'], 2)}." if champ_oper_row is not None else "See operating-point table."),
         "Threshold is selected from cross-fitted training predictions; cost assumptions are illustrative."],
        ["Relational features",
         (f"Combined bureau + previous-application features: PR-AUC {num(both_ablation_row['pr_auc'])}, "
          f"relative PR-AUC lift {num(both_ablation_row['relative_pr_lift_pct'], 2)}%." if both_ablation_row is not None else "See ablation table."),
         "Feature-store variants are ablations; they are not the selected champion's input."],
        ["Thin-file segment",
         (f"No linked bureau record prevalence {pct(no_bureau_row['default_prevalence'])}; linked-bureau prevalence "
          f"{pct(linked_bureau_row['default_prevalence'])}." if no_bureau_row is not None and linked_bureau_row is not None else "See segment table."),
         "Descriptive fixed-holdout comparison, not a fairness or causal finding."],
        ["Feature-policy sensitivity",
         (f"Retaining four additional sensitive/proxy features changes PR-AUC by "
          f"{num(fair_keep_row['pr_auc_delta_vs_configured'], 5)} vs the configured exclusion policy." if fair_keep_row is not None else "See feature-policy table."),
         "Predictive lift alone does not justify re-adding excluded fields; this is not compliance testing."],
    ]

    section = f'''## Results and interpretation

> This section is generated from the current CSV files in `results/` by `scripts/render_readme_results.py`. Rerun the modeling extension and this script after any change to the data, split, model-selection rule, or evaluation code.

### Cross-validation scorecard

| Model | Input columns | PR-AUC (mean ± SD) | ROC-AUC (mean ± SD) | Gini | KS | Brier | Log-loss | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join('| ' + ' | '.join(map(str, row)) + ' |' for row in cv_rows)}

The selected champion in the recorded run is **{manifest_champion}**; the runner-up is **{runner_up}**. {paired_note} The paired PR-AUC differences favor the champion in {fold_win_count} of {fold_count} folds, which is not a consistent fold-level win. For the weighted boosted-tree models, CV Brier and log-loss are diagnostics on raw model scores; use the holdout calibration table below for calibrated-probability quality.

### Holdout decision costs and uncertainty

Point expected decision costs come from `headline_costs.csv`; the 95% bootstrap confidence intervals are from `test_metrics.csv`. Costs are in dataset currency units per applicant, under the stated illustrative assumptions.

{md_table(["Policy", "Threshold", "Point expected cost / applicant", "95% bootstrap CI for cost", "ROC-AUC (95% CI)", "PR-AUC (95% CI)"], cost_rows)}

### Cost-saving comparisons and decomposition

{md_table(["Comparison", "Saving per applicant (dataset currency units)", "Saving relative to reference policy"], savings)}

The threshold effect compares the frozen v1 policy at 0.50 with v1's own threshold selected from cross-fitted training predictions. The model effect compares that tuned v1 policy with the selected champion. The cost ratio, LGD reference value, opportunity-cost assumption, and EAD proxy are illustrative, not Basel retail parameters.

### Holdout operating points

{md_table(["Policy", "Threshold", "Approval rate", "Default rate among approved", "Recall", "Precision", "Expected cost / applicant"], op_rows)}

The `Champion raw weighted score (reference only)` row is a diagnostic comparator, not a calibrated decision policy. Thresholds for the calibrated policies are selected using training-side cross-fitted predictions, not the holdout labels.

### Threshold sensitivity

{md_table(["Cost ratio r", "Analytic threshold", "Selected training-side threshold", "Approval rate", "Default rate among approved", "Recall", "Precision", "Expected cost / applicant", "Cost as % of approve-all"], threshold_rows)}

### Holdout calibration diagnostics

{md_table(["Score type", "Brier score", "Mean predicted probability", "Observed default prevalence", "Unique scores", "Exact zeros", "Exact ones"], cal_rows)}

Calibrated probabilities use the configured floor/cap of 1e-4 and 1 − 1e-4. Calibration quality is evaluated on the holdout set; exact-zero and exact-one counts are shown as diagnostics.

### Relational feature-store ablation

{md_table(["Feature set", "Feature count", "PR-AUC (mean ± SD)", "ROC-AUC (mean ± SD)", "Relative PR-AUC lift vs application-only", "Folds with PR-AUC improvement"], abl_rows)}

The champion model used for the headline results is trained on application-only features. Bureau and previous-application feature sets are ablations, not the deployed/selected champion input. These results are predictive comparisons on this split, not causal evidence.

### Feature-policy sensitivity

{md_table(["Policy variant", "Feature count", "PR-AUC", "ROC-AUC", "PR-AUC delta vs configured policy", "ROC-AUC delta vs configured policy"], fair_rows)}

The configured exclusions are `CODE_GENDER`, `NAME_FAMILY_STATUS`, `NAME_TYPE_SUITE`, `CNT_CHILDREN`, and `CNT_FAM_MEMBERS`. `AGE_YEARS` remains available to the model but is excluded from illustrative risk-driver explanations. `NAME_INCOME_TYPE` remains and may act as a proxy. This is not a fairness test or legal-compliance conclusion.

### Thin-file segmentation

{md_table(["Segment", "Applicants", "Default prevalence", "ROC-AUC", "PR-AUC", "Approval rate", "Default rate among approved", "Expected cost / applicant"], thin_rows)}

These segment results are descriptive and based on the fixed holdout split; no subgroup fairness conclusion is implied.

### Paired holdout comparisons

{md_table(["Comparison", "Cost saving (mean, 95% CI)", "PR-AUC lift (mean, 95% CI)"], paired_rows)}

The XGBoost-versus-LightGBM comparison is treated as **inconclusive** when paired intervals include zero. Do not manually relabel the champion; the `run_manifest.json` records the model selected from cross-validation.

### Explainability and ranking agreement

{md_table(["Ranking comparison", "Spearman correlation", "Shared features"], rank_rows)}

The correlations are descriptive. Collinearity, including correlated credit-amount and housing-area features, can produce different coefficient and SHAP rankings. SHAP values in this project explain the selected tree model's raw weighted score, not calibrated probability or legal adverse-action reasons.

**Illustrative top risk drivers (not legal notices)**

{md_table(["Applicant cohort", "Rank", "Feature", "SHAP attribution", "Illustrative description"], reason_rows)}

### Class-imbalance sensitivity

{md_table(["Strategy", "Model", "ROC-AUC", "PR-AUC", "Selected threshold", "Expected cost / applicant", "Cost as % of approve-all"], imb_rows)}

SMOTE was tested with logistic regression only. A tree-plus-SMOTE comparison was not run, and SMOTE can create fractional one-hot values; treat that experiment as exploratory. The cost and ranking comparisons are specific to this dataset, split, and assumed cost function.

### Key business insights

{md_table(["Insight", "Result from generated artifacts", "Interpretation / limit"], insight_rows)}

The SQL and Power BI layer supports cohort/default-rate exploration and reporting, but it is separate from the model-training pipeline. The Home Credit research dataset is not evidence of validated underwriting for a specific US lender or any other real lender.

### Result artifact index

- `cv_metrics.csv`, `cv_fold_metrics.csv`: cross-validation and fold-level model metrics.
- `headline_costs.csv`, `test_metrics.csv`, `test_paired_diffs.csv`, `test_operating_points.csv`, `threshold_table.csv`: holdout decision-cost points, confidence intervals, paired comparisons, operating points, and threshold sensitivity.
- `test_calibration.csv`, `test_scored.csv`: calibration diagnostics and locked holdout predictions.
- `ablation_study.csv`, `ablation_fold_diffs.csv`, `fair_lending_ablation.csv`, `thin_file_segments.csv`: relational and feature-policy ablations, fold deltas, and segment metrics.
- `imbalance_study.csv`: tested class-imbalance configurations.
- `shap_reason_codes.csv`, `shap_lr_rank_correlation.csv`, `shap_lr_rank_correlation_robust.csv`, `rank_correlation.txt`: illustrative risk drivers and ranking comparisons.
- `run_manifest.json`: selected champion/runner-up, data split, feature counts, assumptions, thresholds, and package versions.
'''

    readme_text = README.read_text(encoding="utf-8-sig")
    readme_text = readme_text.replace(
        "The project is designed to answer the following questions. Final numeric answers should be taken from the regenerated result files, not the older committed CSVs.",
        "The current numeric tables below are generated from the verified result artifacts. Performance comparisons are specific to this dataset and fixed train/test split.",
    )
    old_cost = ("The notebook assumes LGD = 0.45, an opportunity cost of declining a non-defaulter = 0.09, and therefore a cost ratio of 5. "
                "EAD is approximated by `AMT_CREDIT`. These are **illustrative economic assumptions**, not retail Basel F-IRB parameters. "
                "Costs are expressed in dataset currency units; no currency is asserted. The metric is called *expected decision cost*, not credit loss.")
    new_cost = ("Costs use an illustrative LGD of 45% (the Basel F-IRB supervisory value for senior unsecured corporate exposures, used here only as a reference; "
                "F-IRB does not apply to retail), an assumed 9% net margin/opportunity-cost proxy, and cost ratio `r = 5`. "
                "Exposure at default is proxied by `AMT_CREDIT`. Amounts are in the dataset's currency units. "
                "The metric is called *expected decision cost*, not credit loss.")
    readme_text = readme_text.replace(old_cost, new_cost)
    pattern = re.compile(r"(?ms)^## Results and interpretation\s*.*?(?=^## Limitations\s*$)")
    if not pattern.search(readme_text):
        raise ValueError("Could not find the README section from '## Results and interpretation' to '## Limitations'. No changes written.")
    updated = pattern.sub(section.rstrip() + "\n\n", readme_text, count=1)
    # Save a temporary backup outside the repository before writing.
    import tempfile
    backup_path = Path(tempfile.gettempdir()) / "Credit_Risk_Project_README_before_generated_results.md"
    backup_path.write_text(readme_text, encoding="utf-8")
    README.write_text(updated, encoding="utf-8", newline="\n")
    print("README results section regenerated from results/*.csv.")
    print(f"Temporary backup: {backup_path}")
    print("Next: review README.md, then run python scripts/verify_notebook.py, python -m pytest -q, and python scripts/verify_results.py.")


if __name__ == "__main__":
    main()
