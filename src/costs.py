"""
Institutional Credit Risk Modeling Framework (v2.1)
Cost Optimization Engine: Isotonic Calibration, Basel Loss Functions, and Threshold Search
"""

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

try:
    from src.config import LGD_BASE, NET_MARGIN_RATE, COST_RATIO_DEFAULT, COST_SENSITIVITY_RATIOS
except ImportError:
    from config import LGD_BASE, NET_MARGIN_RATE, COST_RATIO_DEFAULT, COST_SENSITIVITY_RATIOS


def calibrate_probabilities_isotonic(
    y_train_true: np.ndarray,
    oof_train_probs: np.ndarray,
    test_raw_probs: np.ndarray,
) -> Tuple[IsotonicRegression, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Fits an Isotonic Regression model on Out-of-Fold training probabilities (out-of-sample by construction)
    and transforms both OOF training and test probabilities to restore true empirical posteriors.

    Returns:
    - iso_reg: Fitted IsotonicRegression transformer
    - oof_calib_probs: Calibrated out-of-fold training probabilities
    - test_calib_probs: Calibrated holdout test probabilities
    - brier_dict: Brier scores before and after calibration
    """
    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_reg.fit(oof_train_probs, y_train_true)

    oof_calib_probs = iso_reg.predict(oof_train_probs)
    test_calib_probs = iso_reg.predict(test_raw_probs)

    brier_before = float(brier_score_loss(y_train_true, oof_train_probs))
    brier_after = float(brier_score_loss(y_train_true, oof_calib_probs))

    brier_dict = {
        "brier_before": brier_before,
        "brier_after": brier_after,
        "brier_improvement_pct": ((brier_before - brier_after) / brier_before) * 100.0,
    }

    return iso_reg, oof_calib_probs, test_calib_probs, brier_dict


def compute_credit_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    amt_credit: np.ndarray,
    lgd: float = LGD_BASE,
    margin: float = NET_MARGIN_RATE,
) -> Dict[str, float]:
    """
    Computes total portfolio dollar cost, cost per applicant, and relative cost vs 'Approve Everyone'.

    Decision Rule:
    - Approve if y_prob < threshold (Predicted Non-Default, Class 0)
    - Decline if y_prob >= threshold (Predicted Default, Class 1)

    Asymmetric Loss:
    - False Negative (Approved Defaulter): Loss = LGD * AMT_CREDIT
    - False Positive (Declined Good Borrower): Loss = Margin * AMT_CREDIT
    """
    y_pred = (y_prob >= threshold).astype(int)
    n_applicants = len(y_true)

    # Cost vectors
    fn_mask = (y_true == 1) & (y_pred == 0)  # Approved default
    fp_mask = (y_true == 0) & (y_pred == 1)  # Rejected good borrower

    cost_fn = float(np.sum(lgd * amt_credit[fn_mask]))
    cost_fp = float(np.sum(margin * amt_credit[fp_mask]))
    total_cost = cost_fn + cost_fp

    # Baseline: Cost of Approving Everyone (Threshold = 1.0)
    approve_all_fn_mask = (y_true == 1)
    cost_approve_all = float(np.sum(lgd * amt_credit[approve_all_fn_mask]))

    cost_per_applicant = total_cost / n_applicants
    cost_pct_vs_approve_all = (total_cost / cost_approve_all) * 100.0 if cost_approve_all > 0 else 100.0

    approval_rate = float(np.mean(y_pred == 0))
    approved_mask = (y_pred == 0)
    default_rate_approved = float(np.mean(y_true[approved_mask])) if np.sum(approved_mask) > 0 else 0.0

    # Classification Metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum(fn_mask)
    fp = np.sum(fp_mask)

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "total_cost": total_cost,
        "cost_per_applicant": cost_per_applicant,
        "cost_pct_vs_approve_all": cost_pct_vs_approve_all,
        "cost_approve_all_total": cost_approve_all,
        "approval_rate": approval_rate,
        "default_rate_approved": default_rate_approved,
        "recall": recall,
        "precision": precision,
    }


def optimize_threshold_grid(
    y_true: np.ndarray,
    calib_probs: np.ndarray,
    amt_credit: np.ndarray,
    r_ratio: float = COST_RATIO_DEFAULT,
    grid_step: float = 0.005,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Performs grid search (0.01 to 0.99, step=0.005) over calibrated OOF probabilities to find the
    cost-minimizing empirical decision threshold. Compares against analytic t* = 1 / (1 + r).
    """
    thresholds = np.arange(0.01, 1.00, grid_step)
    lgd = LGD_BASE
    margin = lgd / r_ratio
    analytic_t_star = 1.0 / (1.0 + r_ratio)

    results = []
    for t in thresholds:
        cost_dict = compute_credit_cost(y_true, calib_probs, threshold=t, amt_credit=amt_credit, lgd=lgd, margin=margin)
        results.append(cost_dict)

    df_grid = pd.DataFrame(results)
    best_idx = df_grid["cost_per_applicant"].idxmin()
    empirical_t_star = float(df_grid.loc[best_idx, "threshold"])

    return empirical_t_star, analytic_t_star, df_grid


def generate_sensitivity_table(
    y_true: np.ndarray,
    calib_probs: np.ndarray,
    amt_credit: np.ndarray,
    r_values: List[float] = COST_SENSITIVITY_RATIOS,
) -> pd.DataFrame:
    """
    Computes optimal threshold, approval rate, recall, precision, and expected cost for
    r in {2.0, 3.0, 5.0, 10.0, 20.0}.
    """
    sensitivity_records = []
    for r in r_values:
        emp_t, ana_t, _ = optimize_threshold_grid(y_true, calib_probs, amt_credit, r_ratio=r)
        lgd = LGD_BASE
        margin = lgd / r
        perf = compute_credit_cost(y_true, calib_probs, threshold=emp_t, amt_credit=amt_credit, lgd=lgd, margin=margin)

        sensitivity_records.append({
            "cost_ratio_r": r,
            "analytic_threshold": round(ana_t, 4),
            "optimal_threshold": round(emp_t, 4),
            "approval_rate": round(perf["approval_rate"] * 100, 2),
            "default_rate_approved": round(perf["default_rate_approved"] * 100, 2),
            "recall": round(perf["recall"] * 100, 2),
            "precision": round(perf["precision"] * 100, 2),
            "cost_per_applicant": round(perf["cost_per_applicant"], 2),
            "cost_pct_vs_approve_all": round(perf["cost_pct_vs_approve_all"], 2),
        })

    return pd.DataFrame(sensitivity_records)
