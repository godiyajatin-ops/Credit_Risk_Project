"""
Institutional Credit Risk Modeling Framework (v2.1)
Feature Engineering, Schema Governance, and Scikit-Learn Pipelines
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

try:
    from src.config import (
        TARGET_COL,
        ID_COL,
        PROTECTED_ATTRIBUTE,
        HIGH_NULL_COLS_60,
        EDUCATION_ORDINAL_MAP,
    )
except ImportError:
    from config import (
        TARGET_COL,
        ID_COL,
        PROTECTED_ATTRIBUTE,
        HIGH_NULL_COLS_60,
        EDUCATION_ORDINAL_MAP,
    )


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Executes production-grade feature transformations on the raw loan application schema.

    Transformations:
    1. Extracts target vector if present.
    2. Drops identifiers (SK_ID_CURR) and protected demographic attributes (CODE_GENDER).
    3. Prunes 17 high-missingness (>60% null) architectural features.
    4. Handles DAYS_EMPLOYED anomaly (365243 -> EMPLOYED_ANOM flag).
    5. Derives debt-burden ratios, loan terms, and composite bureau aggregates.
    6. Maps NAME_EDUCATION_TYPE into an explicit ordinal hierarchy.
    7. Sanitizes OCCUPATION_TYPE missing values to 'Unknown'.
    """
    data = df.copy()

    # 1. Target Extraction
    y = None
    if TARGET_COL in data.columns:
        y = data[TARGET_COL].copy()
        data = data.drop(columns=[TARGET_COL])

    # 2. Fair Lending (Reg B) and Identifier Pruning
    cols_to_drop = []
    if ID_COL in data.columns:
        cols_to_drop.append(ID_COL)
    if PROTECTED_ATTRIBUTE in data.columns:
        cols_to_drop.append(PROTECTED_ATTRIBUTE)

    # 3. High-Missingness Pruning (>60% Nulls)
    cols_to_drop.extend([col for col in HIGH_NULL_COLS_60 if col in data.columns])
    data = data.drop(columns=cols_to_drop, errors="ignore")

    # 4. DAYS_EMPLOYED Anomaly Resolution (365243 sentinel)
    if "DAYS_EMPLOYED" in data.columns:
        data["EMPLOYED_ANOM"] = (data["DAYS_EMPLOYED"] == 365243).astype(int)
        days_employed_clean = data["DAYS_EMPLOYED"].replace(365243, np.nan)
        data["YEARS_EMPLOYED"] = -days_employed_clean / 365.25
        data = data.drop(columns=["DAYS_EMPLOYED"])

    # 5. Core Demographic & Financial Ratio Derivations
    if "DAYS_BIRTH" in data.columns:
        data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25
        data = data.drop(columns=["DAYS_BIRTH"])

    if "AMT_CREDIT" in data.columns and "AMT_INCOME_TOTAL" in data.columns:
        data["LOAN_TO_INCOME_RATIO"] = data["AMT_CREDIT"] / data["AMT_INCOME_TOTAL"]

    if "AMT_ANNUITY" in data.columns and "AMT_INCOME_TOTAL" in data.columns:
        data["ANNUITY_TO_INCOME"] = data["AMT_ANNUITY"] / data["AMT_INCOME_TOTAL"]

    if "AMT_ANNUITY" in data.columns and "AMT_CREDIT" in data.columns:
        data["CREDIT_TERM"] = data["AMT_ANNUITY"] / data["AMT_CREDIT"]

    if "AMT_CREDIT" in data.columns and "AMT_GOODS_PRICE" in data.columns:
        data["CREDIT_TO_GOODS"] = data["AMT_CREDIT"] / data["AMT_GOODS_PRICE"]

    # Composite External Bureau Signal
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in data.columns]
    if ext_cols:
        data["EXT_SOURCE_MEAN"] = data[ext_cols].mean(axis=1)

    # 6. Categorical Sanitation & Ordinal Encoding
    if "OCCUPATION_TYPE" in data.columns:
        data["OCCUPATION_TYPE"] = data["OCCUPATION_TYPE"].fillna("Unknown")

    if "NAME_EDUCATION_TYPE" in data.columns:
        data["NAME_EDUCATION_TYPE"] = (
            data["NAME_EDUCATION_TYPE"]
            .map(EDUCATION_ORDINAL_MAP)
            .astype(float)
        )

    return data, y


def get_preprocessors(X: pd.DataFrame) -> Tuple[ColumnTransformer, ColumnTransformer]:
    """
    Constructs dual Scikit-Learn ColumnTransformers for Linear and Tree-based models.

    - lr_preprocessor: Median imputation + indicators + StandardScaler (numeric)
                       Constant imputation + OneHotEncoder(min_freq=0.01) (nominal)
    - tree_preprocessor: Passthrough numerics (preserves NaNs for native split routing)
                         OrdinalEncoder(unknown_value=-1) (nominal)
    """
    nominal_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Pipeline A: Logistic Regression Preprocessor
    lr_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    lr_nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
            ),
        ]
    )

    lr_preprocessor = ColumnTransformer(
        transformers=[
            ("num", lr_numeric_pipeline, numeric_cols),
            ("cat", lr_nominal_pipeline, nominal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Pipeline B: Tree-Based Estimator Preprocessor
    tree_nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", tree_nominal_pipeline, nominal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return lr_preprocessor, tree_preprocessor
