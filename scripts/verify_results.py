"""Verify that regenerated result artifacts are internally consistent.

Run from the repository root after executing Notebooks/modeling_extension.ipynb
from a clean kernel. This script does not train models or replace statistical review.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REQUIRED = [
    "run_manifest.json",
    "cv_metrics.csv",
    "cv_fold_metrics.csv",
    "threshold_table.csv",
    "test_metrics.csv",
    "test_calibration.csv",
    "test_paired_diffs.csv",
    "champion_runnerup_fold_diffs.csv",
    "test_operating_points.csv",
    "headline_costs.csv",
    "test_scored.csv",
    "imbalance_study.csv",
    "ablation_study.csv",
    "ablation_fold_diffs.csv",
    "fair_lending_ablation.csv",
    "shap_lr_rank_correlation.csv",
    "shap_lr_rank_correlation_robust.csv",
    "thin_file_segments.csv",
    "shap_reason_codes.csv",
    "rank_correlation.txt",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    missing = [name for name in REQUIRED if not (RESULTS / name).exists()]
    if missing:
        print("Missing expected result artifacts:", ", ".join(missing), file=sys.stderr)
        print("Run the updated modeling_extension.ipynb from a clean kernel first.", file=sys.stderr)
        return 2

    manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    cv = pd.read_csv(RESULTS / "cv_metrics.csv")
    folds = pd.read_csv(RESULTS / "cv_fold_metrics.csv")
    scored = pd.read_csv(RESULTS / "test_scored.csv")
    costs = pd.read_csv(RESULTS / "headline_costs.csv")
    paired = pd.read_csv(RESULTS / "test_paired_diffs.csv")
    threshold = pd.read_csv(RESULTS / "threshold_table.csv")
    ablation = pd.read_csv(RESULTS / "ablation_study.csv")
    ablation_folds = pd.read_csv(RESULTS / "ablation_fold_diffs.csv")
    policy = pd.read_csv(RESULTS / "fair_lending_ablation.csv")

    # G1: One champion follows the OOF PR-AUC ranking.
    if cv.empty or "model" not in cv or "pr_auc" not in cv:
        fail("cv_metrics.csv is empty or missing model/pr_auc.")
    cv = cv.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    if manifest.get("champion") != cv.loc[0, "model"]:
        fail("run_manifest champion does not match the highest-PR-AUC CV row.")
    if manifest.get("champion") not in {"xgboost", "lightgbm"}:
        fail("Champion must be selected from the XGBoost/LightGBM comparison.")
    if cv["model"].duplicated().any():
        fail("cv_metrics.csv contains duplicate model rows.")

    # G2: Fold metrics should cover each comparison model over five folds.
    if not {"model", "fold", "pr_auc", "roc_auc"}.issubset(folds.columns):
        fail("cv_fold_metrics.csv lacks required columns.")
    counts = folds.groupby("model")["fold"].nunique()
    if not (counts == 5).all():
        fail("Each CV model must have exactly five distinct validation folds.")
    feature_counts = manifest.get("feature_counts", {})
    expected_counts = {
        "application_only": 104,
        "application_plus_bureau": 115,
        "application_plus_previous": 113,
        "full_relational_store": 124,
    }
    if feature_counts != expected_counts:
        fail(f"Feature-count manifest mismatch; expected {expected_counts}, found {feature_counts}.")

    # G3: Holdout table matches manifest and holds valid model scores.
    if len(scored) != int(manifest.get("test_rows", -1)):
        fail("test_scored.csv row count does not match run_manifest.")
    for col in ["y_true", "amt_credit", "v1_prob", "champ_raw_prob", "champ_calib_prob", "champ_pred_optimal", "runner_up_raw_prob", "runner_up_calib_prob", "runner_up_pred_optimal"]:
        if col not in scored:
            fail(f"test_scored.csv missing required column: {col}")
    y = scored["y_true"].to_numpy(dtype=int)
    credit = scored["amt_credit"].to_numpy(dtype=float)
    pred = scored["champ_pred_optimal"].to_numpy(dtype=int)
    probs = scored["champ_calib_prob"].to_numpy(dtype=float)
    if not set(np.unique(y)).issubset({0, 1}):
        fail("Test target contains values other than 0/1.")
    if not np.isfinite(credit).all() or not np.isfinite(probs).all():
        fail("Test credit amounts or calibrated probabilities contain non-finite values.")
    if ((probs < 1e-4) | (probs > 1 - 1e-4)).any():
        fail("Calibrated probabilities violate the configured floor/cap.")

    # G4: Recompute the champion's point estimate directly from scored rows.
    assumptions = manifest.get("decision_cost_assumptions", {})
    lgd = float(assumptions.get("lgd_reference_assumption", 0.45))
    margin = float(assumptions.get("rejection_opportunity_cost_assumption", 0.09))
    fn = (y == 1) & (pred == 0)
    fp = (y == 0) & (pred == 1)
    recomputed_cost = float((lgd * credit[fn].sum() + margin * credit[fp].sum()) / len(y))
    champion_label = f"Champion {manifest['champion']} (OOF-calibrated cost-optimal)"
    selected_costs = costs.loc[costs["policy"] == champion_label, "cost_per_applicant"]
    if len(selected_costs) != 1:
        fail(f"headline_costs.csv must contain exactly one row for {champion_label!r}.")
    if not np.isclose(recomputed_cost, float(selected_costs.iloc[0]), atol=0.01, rtol=0):
        fail(f"Champion cost mismatch: scored={recomputed_cost:.4f}, headline={float(selected_costs.iloc[0]):.4f}.")

    # Calibration statistics must be generated from the same locked test scores.
    calibration = pd.read_csv(RESULTS / "test_calibration.csv")
    expected_calibration_types = {
        f"{manifest['champion']}: raw model score (not calibrated)",
        f"{manifest['champion']}: isotonic-calibrated probability",
        f"{manifest['runner_up']}: raw model score (not calibrated)",
        f"{manifest['runner_up']}: isotonic-calibrated probability",
    }
    if set(calibration["score_type"]) != expected_calibration_types:
        fail("test_calibration.csv has unexpected or missing score types.")
    calibrated_row = calibration.loc[
        calibration["score_type"] == f"{manifest['champion']}: isotonic-calibrated probability"
    ].iloc[0]
    if not np.isclose(float(calibrated_row["brier_score"]), np.mean((probs - y) ** 2), atol=1e-8, rtol=0):
        fail("Calibrated test Brier score does not match test_scored.csv.")
    if not np.isclose(float(calibrated_row["mean_predicted_probability"]), np.mean(probs), atol=1e-8, rtol=0):
        fail("Mean calibrated probability does not match test_scored.csv.")
    if not np.isclose(float(calibrated_row["observed_prevalence"]), np.mean(y), atol=1e-8, rtol=0):
        fail("Observed default prevalence does not match test_scored.csv.")

    # G5: Thresholds must be in range and results must include the expected comparisons.
    if not threshold["optimal_threshold"].between(0, 1).all():
        fail("threshold_table.csv contains an out-of-range threshold.")
    expected_comparisons = {
        "Champion vs v1 at threshold 0.50",
        "Champion vs approve-all",
        "Champion vs v1 own cost-optimal threshold",
        "Champion vs runner-up",
    }
    if not expected_comparisons.issubset(set(paired["comparison"])):
        fail("test_paired_diffs.csv is missing one or more required comparisons.")
    if not np.isfinite(paired.select_dtypes(include=[np.number]).to_numpy()).all():
        fail("test_paired_diffs.csv contains non-finite numeric entries.")

    # G6: Champion/runner-up comparison and relational ablations must be fold-aligned.
    model_pair = pd.read_csv(RESULTS / "champion_runnerup_fold_diffs.csv")
    if len(model_pair) != 5:
        fail("Champion-versus-runner-up fold comparison must contain five paired folds.")
    if set(model_pair["champion"]) != {manifest["champion"]} or set(model_pair["runner_up"]) != {manifest["runner_up"]}:
        fail("Champion-versus-runner-up folds disagree with run_manifest.")
    if not np.isfinite(model_pair.select_dtypes(include=[np.number]).to_numpy()).all():
        fail("Champion-versus-runner-up fold differences contain non-finite values.")
    if len(ablation) != 4 or len(ablation_folds) != 15:
        fail("Relational ablation must contain four scorecard rows and 15 paired fold rows.")
    if not {"policy", "pr_auc", "roc_auc", "pr_auc_delta_vs_configured"}.issubset(policy.columns):
        fail("fair_lending_ablation.csv lacks expected sensitivity columns.")
    if len(policy) != 2:
        fail("fair_lending_ablation.csv must contain configured and expanded feature policies.")
    rank_robust = pd.read_csv(RESULTS / "shap_lr_rank_correlation_robust.csv")
    if len(rank_robust) < 10 or rank_robust["feature"].duplicated().any():
        fail("Collinearity-robust rank table is too small or contains duplicate features.")

    # The thin-file segment report is expected to have both presence/absence cohorts.
    segments = pd.read_csv(RESULTS / "thin_file_segments.csv")
    if len(segments) != 2 or segments["segment"].nunique() != 2:
        fail("thin_file_segments.csv must report both bureau-history segments.")

    # Calibration sanity metric is diagnostic only, not a standalone acceptance criterion.
    prevalence = float(np.mean(y))
    mean_predicted = float(np.mean(probs))
    print("All structural and consistency gates passed.")
    print(f"Champion: {manifest['champion']}; runner-up: {manifest.get('runner_up')}")
    print(f"Holdout rows: {len(scored):,}; default prevalence: {prevalence:.4%}")
    print(f"Mean calibrated probability: {mean_predicted:.4%}")
    print(f"Recomputed champion expected decision cost: {recomputed_cost:,.2f} dataset currency units")
    if abs(mean_predicted - prevalence) > 0.01:
        print("WARNING: mean predicted probability differs from prevalence by more than 1 percentage point; review calibration.")
    print("These automated checks do not establish fairness, legal compliance, causal validity, or deployment readiness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
