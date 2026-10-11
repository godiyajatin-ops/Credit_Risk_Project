"""
Institutional Credit Risk Modeling Framework (v2.1)
Evaluation Harness: 5-Fold Stratified CV, OOF Generation, and Risk Metrics
"""

import time
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Callable
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

try:
    from src.config import RANDOM_SEED, CV_FOLDS, RESULTS_DIR, LGD_BASE, NET_MARGIN_RATE
except ImportError:
    from config import RANDOM_SEED, CV_FOLDS, RESULTS_DIR, LGD_BASE, NET_MARGIN_RATE


def calculate_risk_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculates comprehensive institutional credit scoring metrics:
    - ROC-AUC: Global class separation capability
    - PR-AUC: Average Precision (Primary metric under ~8% class imbalance)
    - Gini Coefficient: 2 * ROC-AUC - 1 (Standard banking discrimination metric)
    - KS Statistic: Maximum distance between default and non-default distributions
    - Brier Score: Mean squared probability error (Calibration measure)
    - Log-Loss: Cross-entropy error
    """
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    gini = 2.0 * auc - 1.0

    # Two-sample Kolmogorov-Smirnov separation
    prob_defaults = y_prob[y_true == 1]
    prob_non_defaults = y_prob[y_true == 0]
    ks_stat = stats.ks_2samp(prob_defaults, prob_non_defaults).statistic

    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)

    return {
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "gini": float(gini),
        "ks_stat": float(ks_stat),
        "brier_score": float(brier),
        "log_loss": float(ll),
    }


def cv_oof_evaluate(
    model_name: str,
    pipeline_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Optional[list] = None,
    save_oof: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]:
    """
    Executes a strict 5-Fold Stratified Cross-Validation on the training partition:
    - Instantiates a clean pipeline per fold to guarantee zero data leakage.
    - Captures out-of-fold (OOF) default probability predictions.
    - Saves out-of-fold array to results/oof_<model_name>.npy for Phase P3 calibration.
    - Returns OOF predictions, overall scorecard, and fold-level metrics table.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_predictions = np.zeros(len(y), dtype=np.float64)
    fold_metrics_list = []

    X_mat = X[feature_cols] if feature_cols is not None else X
    y_vec = y.to_numpy() if isinstance(y, pd.Series) else y

    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_vec), 1):
        if isinstance(X_mat, pd.DataFrame):
            X_tr, X_val = X_mat.iloc[train_idx], X_mat.iloc[val_idx]
        else:
            X_tr, X_val = X_mat[train_idx], X_mat[val_idx]

        y_tr, y_val = y_vec[train_idx], y_vec[val_idx]

        # Fresh estimator instance per fold
        estimator = pipeline_factory()
        estimator.fit(X_tr, y_tr)

        # Predict probability of default (Class 1)
        val_probs = estimator.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_probs

        fold_scores = calculate_risk_metrics(y_val, val_probs)
        fold_scores["fold"] = fold
        fold_metrics_list.append(fold_scores)

    elapsed_time = time.time() - start_time
    fold_df = pd.DataFrame(fold_metrics_list)

    # Compute overall metrics across full out-of-fold predictions
    overall_scores = calculate_risk_metrics(y_vec, oof_predictions)
    overall_scores["model"] = model_name
    overall_scores["train_time_sec"] = round(elapsed_time, 2)
    overall_scores["roc_auc_std"] = float(fold_df["roc_auc"].std())
    overall_scores["pr_auc_std"] = float(fold_df["pr_auc"].std())

    if save_oof:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        oof_path = RESULTS_DIR / f"oof_{model_name}.npy"
        np.save(oof_path, oof_predictions)

    return oof_predictions, overall_scores, fold_df


def bootstrap_test_evaluation(
    y_test: np.ndarray,
    amt_credit_test: np.ndarray,
    model_predictions: Dict[str, Tuple[np.ndarray, float]],
    n_bootstraps: int = 1000,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes 1,000 paired bootstrap iterations on the holdout test set to construct 95% Confidence
    Intervals for ROC-AUC, PR-AUC, and Expected Cost per applicant.

    Also computes paired difference CIs for:
    - Champion (Optimal t*) vs. Baseline v1 (t=0.5)
    - Champion (Optimal t*) vs. Runner-Up (t=0.5)
    """
    rng = np.random.default_rng(random_seed)
    n_samples = len(y_test)
    models = list(model_predictions.keys())

    # Storage matrices: [n_bootstraps, len(models)]
    boot_roc = {m: np.zeros(n_bootstraps) for m in models}
    boot_pr = {m: np.zeros(n_bootstraps) for m in models}
    boot_cost = {m: np.zeros(n_bootstraps) for m in models}

    b = 0
    while b < n_bootstraps:
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        y_b = y_test[boot_idx]
        credit_b = amt_credit_test[boot_idx]

        # Reject invalid resamples rather than leaving zero-valued CI entries.
        if len(np.unique(y_b)) < 2:
            continue

        for m, (probs, thresh) in model_predictions.items():
            probs_b = probs[boot_idx]
            boot_roc[m][b] = roc_auc_score(y_b, probs_b)
            boot_pr[m][b] = average_precision_score(y_b, probs_b)

            # Cost evaluation
            preds_b = (probs_b >= thresh).astype(int)
            fn_mask = (y_b == 1) & (preds_b == 0)
            fp_mask = (y_b == 0) & (preds_b == 1)
            cost_total = np.sum(LGD_BASE * credit_b[fn_mask]) + np.sum(NET_MARGIN_RATE * credit_b[fp_mask])
            boot_cost[m][b] = cost_total / n_samples
        b += 1

    # Aggregate 95% Confidence Intervals (Percentile Bootstrap: 2.5% to 97.5%)
    summary_list = []
    for m in models:
        summary_list.append({
            "model_setup": m,
            "roc_auc_mean": np.mean(boot_roc[m]),
            "roc_auc_ci_lower": np.percentile(boot_roc[m], 2.5),
            "roc_auc_ci_upper": np.percentile(boot_roc[m], 97.5),
            "pr_auc_mean": np.mean(boot_pr[m]),
            "pr_auc_ci_lower": np.percentile(boot_pr[m], 2.5),
            "pr_auc_ci_upper": np.percentile(boot_pr[m], 97.5),
            "expected_cost_mean": np.mean(boot_cost[m]),
            "expected_cost_ci_lower": np.percentile(boot_cost[m], 2.5),
            "expected_cost_ci_upper": np.percentile(boot_cost[m], 97.5),
        })

    df_ci_summary = pd.DataFrame(summary_list)

    # Paired Differences: Delta = (Baseline - Champion) -> Positive value means Champion saves money
    diff_records = []
    champ_key = "Champion (cost-optimal t*)"
    v1_key = "Baseline v1 (5-feature LR)"

    if champ_key in models and v1_key in models:
        cost_diff_v1 = boot_cost[v1_key] - boot_cost[champ_key]
        diff_records.append({
            "comparison": "Champion (t*) vs Baseline v1",
            "cost_saving_mean": np.mean(cost_diff_v1),
            "cost_saving_ci_lower": np.percentile(cost_diff_v1, 2.5),
            "cost_saving_ci_upper": np.percentile(cost_diff_v1, 97.5),
            "pr_auc_lift_mean": np.mean(boot_pr[champ_key] - boot_pr[v1_key]),
            "pr_auc_lift_ci_lower": np.percentile(boot_pr[champ_key] - boot_pr[v1_key], 2.5),
            "pr_auc_lift_ci_upper": np.percentile(boot_pr[champ_key] - boot_pr[v1_key], 97.5),
        })

    df_diff_summary = pd.DataFrame(diff_records)
    return df_ci_summary, df_diff_summary
