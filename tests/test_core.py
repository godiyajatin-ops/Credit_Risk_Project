from __future__ import annotations

import numpy as np
import pandas as pd

from src.bureau_features import build_bureau_features, build_previous_app_features
from src.costs import calibrate_probabilities_isotonic, compute_credit_cost
from src.evaluate import bootstrap_test_evaluation
from src.features import build_features


def test_feature_policy_drops_configured_sensitive_and_proxy_columns():
    raw = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "CODE_GENDER": ["F", "M"],
            "NAME_FAMILY_STATUS": ["Single", "Married"],
            "NAME_TYPE_SUITE": ["Unaccompanied", "Family"],
            "CNT_CHILDREN": [0, 2],
            "CNT_FAM_MEMBERS": [1, 4],
            "DAYS_BIRTH": [-10000, -15000],
            "DAYS_EMPLOYED": [-1000, 365243],
            "AMT_CREDIT": [1000.0, 2000.0],
            "AMT_INCOME_TOTAL": [500.0, 1000.0],
        }
    )
    X, y = build_features(raw)
    assert y.tolist() == [0, 1]
    assert "SK_ID_CURR" not in X
    assert "CODE_GENDER" not in X
    for col in ["NAME_FAMILY_STATUS", "NAME_TYPE_SUITE", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]:
        assert col not in X
    assert {"AGE_YEARS", "YEARS_EMPLOYED", "EMPLOYED_ANOM", "LOAN_TO_INCOME_RATIO"}.issubset(X.columns)

    X_sensitivity, _ = build_features(raw, drop_fair_lending=False)
    assert "CODE_GENDER" not in X_sensitivity
    for col in ["NAME_FAMILY_STATUS", "NAME_TYPE_SUITE", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]:
        assert col in X_sensitivity


def test_cross_fitted_isotonic_calibration_returns_bounded_probabilities():
    rng = np.random.default_rng(7)
    y = np.tile(np.array([0, 0, 0, 1]), 50)
    raw = np.clip(0.08 + 0.65 * y + rng.normal(0, 0.15, len(y)), 0, 1)
    test_raw = np.array([0.0, 0.2, 0.8, 1.0])
    calibrator, calibrated_oof, calibrated_test, audit = calibrate_probabilities_isotonic(
        y_train_true=y,
        oof_train_probs=raw,
        test_raw_probs=test_raw,
        n_splits=5,
        random_seed=7,
    )
    assert calibrated_oof.shape == y.shape
    assert calibrated_test is not None
    assert np.isfinite(calibrated_oof).all()
    assert np.isfinite(calibrated_test).all()
    assert ((calibrated_oof >= 1e-4) & (calibrated_oof <= 1 - 1e-4)).all()
    assert ((calibrated_test >= 1e-4) & (calibrated_test <= 1 - 1e-4)).all()
    assert set(["brier_before", "brier_after", "brier_improvement_pct"]).issubset(audit)
    assert calibrator is not None


def test_expected_decision_cost_matches_hand_calculation():
    y = np.array([1, 0, 1, 0])
    prob = np.array([0.1, 0.9, 0.8, 0.1])
    amount = np.array([100.0, 200.0, 300.0, 400.0])
    result = compute_credit_cost(y, prob, threshold=0.5, amt_credit=amount)
    # One missed default: 0.45 * 100. One rejected non-defaulter: 0.09 * 200.
    assert np.isclose(result["total_cost"], 63.0)
    assert np.isclose(result["cost_per_applicant"], 15.75)
    assert np.isclose(result["cost_pct_vs_approve_all"], 35.0)
    assert np.isclose(result["approval_rate"], 0.5)
    assert np.isclose(result["recall"], 0.5)
    assert np.isclose(result["precision"], 0.5)


def test_bootstrap_rejects_one_class_resamples_and_returns_finite_intervals():
    rng = np.random.default_rng(13)
    y = np.array([0, 1] * 40, dtype=int)
    p_champ = np.clip(0.1 + 0.65 * y + rng.normal(0, 0.18, len(y)), 0.001, 0.999)
    p_base = np.clip(0.2 + 0.35 * y + rng.normal(0, 0.22, len(y)), 0.001, 0.999)
    amounts = rng.uniform(1000, 10000, len(y))
    models = {
        "Approve everyone": (np.zeros_like(p_champ), 0.5),
        "Baseline v1 (5-feature LR)": (p_base, 0.5),
        "Baseline v1 (OOF-calibrated cost-optimal t*)": (p_base, 0.45),
        "Champion (cost-optimal t*)": (p_champ, 0.3),
    }
    summary, diffs = bootstrap_test_evaluation(
        y_test=y,
        amt_credit_test=amounts,
        model_predictions=models,
        n_bootstraps=30,
        random_seed=13,
    )
    assert len(summary) == len(models)
    assert {
        "Champion vs v1 at threshold 0.50",
        "Champion vs approve-all",
        "Champion vs v1 own cost-optimal threshold",
    }.issubset(set(diffs["comparison"]))
    assert np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.isfinite(diffs.select_dtypes(include=[np.number]).to_numpy()).all()


def test_relational_aggregates_preserve_missing_debt_and_only_average_approved_amounts(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [11, 12, 21],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "DAYS_CREDIT": [-100.0, -300.0, -40.0],
            "CREDIT_DAY_OVERDUE": [0.0, 5.0, 0.0],
            "AMT_CREDIT_SUM": [1000.0, 2000.0, 0.0],
            "AMT_CREDIT_SUM_DEBT": [np.nan, np.nan, np.nan],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 50.0, 0.0],
        }
    ).to_csv(raw_dir / "bureau.csv", index=False)
    bureau = build_bureau_features(tmp_path).set_index("SK_ID_CURR")
    assert np.isnan(bureau.loc[1, "BUREAU_AMT_CREDIT_SUM_DEBT"])
    assert np.isnan(bureau.loc[1, "BUREAU_DEBT_CREDIT_RATIO"])
    assert np.isnan(bureau.loc[2, "BUREAU_DEBT_CREDIT_RATIO"])

    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "SK_ID_PREV": [101, 102, 103, 201],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Canceled", "Approved"],
            "AMT_APPLICATION": [100.0, 500.0, 300.0, 0.0],
            "AMT_CREDIT": [80.0, 0.0, 50.0, 25.0],
            "DAYS_DECISION": [-10.0, -20.0, -30.0, -5.0],
        }
    ).to_csv(raw_dir / "previous_application.csv", index=False)
    previous = build_previous_app_features(tmp_path).set_index("SK_ID_CURR")
    assert previous.loc[1, "PREV_APP_COUNT"] == 3
    assert previous.loc[1, "PREV_APPROVED_COUNT"] == 1
    assert previous.loc[1, "PREV_REFUSED_COUNT"] == 1
    assert np.isclose(previous.loc[1, "PREV_AMT_APPLICATION_MEAN"], 100.0)
    assert np.isclose(previous.loc[1, "PREV_AMT_CREDIT_MEAN"], 80.0)
    assert np.isclose(previous.loc[1, "PREV_CREDIT_APP_RATIO_MEAN"], 0.8)
    assert np.isnan(previous.loc[2, "PREV_CREDIT_APP_RATIO_MEAN"])
