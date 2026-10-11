"""
Credit Risk Modeling Project
Bureau and previous-application feature aggregation
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

try:
    from src.config import DATA_DIR
except ImportError:
    from config import DATA_DIR


def build_bureau_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Extracts, cleans, and aggregates historical credit bureau tradelines from bureau.csv (1.7M rows).

    Aggregates per SK_ID_CURR:
    - BUREAU_LOAN_COUNT: Total historical bureau tradelines
    - BUREAU_ACTIVE_LOAN_COUNT: Currently active loans
    - BUREAU_CLOSED_LOAN_COUNT: Formally settled/closed loans
    - BUREAU_DAYS_CREDIT_MEAN: Average loan age/vintage
    - BUREAU_DAYS_CREDIT_MAX: Most recent bureau credit (recency)
    - BUREAU_AMT_CREDIT_SUM: Total historical credit extended
    - BUREAU_AMT_CREDIT_SUM_DEBT: Total outstanding debt balance
    - BUREAU_DEBT_CREDIT_RATIO: Leverage exposure (Debt / Credit)
    - BUREAU_CREDIT_DAY_OVERDUE_MAX: Peak days past due across all tradelines
    - BUREAU_AMT_CREDIT_SUM_OVERDUE: Total currently delinquent balance
    - HAS_BUREAU: Flag indicating presence of external bureau history
    """
    bureau_path = data_dir / "raw" / "bureau.csv"
    if not bureau_path.exists():
        raise FileNotFoundError(f"bureau.csv not found at: {bureau_path}")

    use_cols = [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE",
    ]

    dtype_dict = {
        "SK_ID_CURR": "int32",
        "SK_ID_BUREAU": "int32",
        "CREDIT_ACTIVE": "category",
        "DAYS_CREDIT": "float32",
        "CREDIT_DAY_OVERDUE": "float32",
        "AMT_CREDIT_SUM": "float32",
        "AMT_CREDIT_SUM_DEBT": "float32",
        "AMT_CREDIT_SUM_OVERDUE": "float32",
    }

    print("Loading bureau.csv (1.7M records) with optimized dtypes...")
    df_bureau = pd.read_csv(bureau_path, usecols=use_cols, dtype=dtype_dict)

    # 1. Data Cleaning
    df_bureau = df_bureau.drop_duplicates()
    # Preserve unknown exposure/delinquency values as missing. Missing is not equivalent
    # to confirmed zero debt or zero days overdue; only semantic count features are zero-filled downstream.

    # 2. Status Indicators
    df_bureau["IS_ACTIVE"] = (df_bureau["CREDIT_ACTIVE"] == "Active").astype("int32")
    df_bureau["IS_CLOSED"] = (df_bureau["CREDIT_ACTIVE"] == "Closed").astype("int32")

    # 3. Aggregations per SK_ID_CURR
    print("Computing vectorized bureau aggregates per SK_ID_CURR...")
    agg_dict = {
        "SK_ID_BUREAU": "count",
        "IS_ACTIVE": "sum",
        "IS_CLOSED": "sum",
        "DAYS_CREDIT": ["mean", "max"],
        "AMT_CREDIT_SUM": lambda values: values.sum(min_count=1),
        "AMT_CREDIT_SUM_DEBT": lambda values: values.sum(min_count=1),
        "CREDIT_DAY_OVERDUE": "max",
        "AMT_CREDIT_SUM_OVERDUE": lambda values: values.sum(min_count=1),
    }

    bureau_agg = df_bureau.groupby("SK_ID_CURR").agg(agg_dict)
    bureau_agg.columns = [
        "BUREAU_LOAN_COUNT",
        "BUREAU_ACTIVE_LOAN_COUNT",
        "BUREAU_CLOSED_LOAN_COUNT",
        "BUREAU_DAYS_CREDIT_MEAN",
        "BUREAU_DAYS_CREDIT_MAX",
        "BUREAU_AMT_CREDIT_SUM",
        "BUREAU_AMT_CREDIT_SUM_DEBT",
        "BUREAU_CREDIT_DAY_OVERDUE_MAX",
        "BUREAU_AMT_CREDIT_SUM_OVERDUE",
    ]

    # 4. Financial Ratio Derivation
    credit_sum = bureau_agg["BUREAU_AMT_CREDIT_SUM"]
    debt_sum = bureau_agg["BUREAU_AMT_CREDIT_SUM_DEBT"]
    bureau_agg["BUREAU_DEBT_CREDIT_RATIO"] = (debt_sum / credit_sum.where(credit_sum > 0)).astype("float32")

    bureau_agg["HAS_BUREAU"] = 1
    bureau_agg.reset_index(inplace=True)

    print(f"Bureau features successfully built: {bureau_agg.shape[0]:,} unique clients.")
    return bureau_agg


def build_previous_app_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Extracts and aggregates internal loan origination history from previous_application.csv (1.67M rows).

    Aggregates per SK_ID_CURR:
    - PREV_APP_COUNT: Total prior applications with this institution
    - PREV_APPROVED_COUNT: Count of previously approved facilities
    - PREV_REFUSED_COUNT: Count of previously declined loan requests
    - PREV_REFUSAL_RATE: Historical bank rejection rate (Refused / Total)
    - PREV_AMT_APPLICATION_MEAN: Average amount requested on approved prior applications
    - PREV_AMT_CREDIT_MEAN: Average approved prior credit amount
    - PREV_CREDIT_APP_RATIO_MEAN: Approved credit-to-requested-amount ratio
    - PREV_DAYS_DECISION_MAX: Most recent prior application recency (days before current)
    - HAS_PREV: Flag indicating prior relationship history with the bank
    """
    prev_path = data_dir / "raw" / "previous_application.csv"
    if not prev_path.exists():
        raise FileNotFoundError(f"previous_application.csv not found at: {prev_path}")

    use_cols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "NAME_CONTRACT_STATUS",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "DAYS_DECISION",
    ]

    dtype_dict = {
        "SK_ID_CURR": "int32",
        "SK_ID_PREV": "int32",
        "NAME_CONTRACT_STATUS": "category",
        "AMT_APPLICATION": "float32",
        "AMT_CREDIT": "float32",
        "DAYS_DECISION": "float32",
    }

    print("Loading previous_application.csv (1.67M records) with optimized dtypes...")
    df_prev = pd.read_csv(prev_path, usecols=use_cols, dtype=dtype_dict)

    # 1. Retain missing amounts as missing; don't treat missing/refused amounts as zero.
    df_prev["IS_APPROVED"] = (df_prev["NAME_CONTRACT_STATUS"] == "Approved").astype("int32")
    df_prev["IS_REFUSED"] = (df_prev["NAME_CONTRACT_STATUS"] == "Refused").astype("int32")
    approved = df_prev["NAME_CONTRACT_STATUS"] == "Approved"
    df_prev["AMT_APPLICATION_APPROVED"] = df_prev["AMT_APPLICATION"].where(approved)
    df_prev["AMT_CREDIT_APPROVED"] = df_prev["AMT_CREDIT"].where(approved)
    valid_approved_application = approved & df_prev["AMT_APPLICATION"].gt(0)
    df_prev["CREDIT_APP_RATIO_APPROVED"] = (
        df_prev["AMT_CREDIT"].where(valid_approved_application)
        / df_prev["AMT_APPLICATION"].where(valid_approved_application)
    ).astype("float32")

    # 3. Aggregations per SK_ID_CURR
    print("Computing vectorized previous application aggregates per SK_ID_CURR...")
    agg_dict = {
        "SK_ID_PREV": "count",
        "IS_APPROVED": "sum",
        "IS_REFUSED": "sum",
        "AMT_APPLICATION_APPROVED": "mean",
        "AMT_CREDIT_APPROVED": "mean",
        "CREDIT_APP_RATIO_APPROVED": "mean",
        "DAYS_DECISION": "max",  # closest to current application (least negative)
    }

    prev_agg = df_prev.groupby("SK_ID_CURR").agg(agg_dict)
    prev_agg.columns = [
        "PREV_APP_COUNT",
        "PREV_APPROVED_COUNT",
        "PREV_REFUSED_COUNT",
        "PREV_AMT_APPLICATION_MEAN",
        "PREV_AMT_CREDIT_MEAN",
        "PREV_CREDIT_APP_RATIO_MEAN",
        "PREV_DAYS_DECISION_MAX",
    ]

    # Refusal rate
    prev_agg["PREV_REFUSAL_RATE"] = (
        prev_agg["PREV_REFUSED_COUNT"] / prev_agg["PREV_APP_COUNT"].where(prev_agg["PREV_APP_COUNT"] > 0)
    ).astype("float32")

    prev_agg["HAS_PREV"] = 1
    prev_agg.reset_index(inplace=True)

    print(f"Previous application features successfully built: {prev_agg.shape[0]:,} unique clients.")
    return prev_agg
