"""
Institutional Credit Risk Modeling Framework (v2.1)
Global Configuration, Architectural Constants, and Paths
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Base Directories & Relative Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "application_train.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
ASSETS_DIR = PROJECT_ROOT / "assets"
NOTEBOOKS_DIR = PROJECT_ROOT / "Notebooks"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Reproducibility & Partition Architecture
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
PROTECTED_ATTRIBUTE = "CODE_GENDER"

# -----------------------------------------------------------------------------
# Frozen v1 Benchmark Metadata (Phase P0 Locked)
# -----------------------------------------------------------------------------
V1_BASELINE_METRICS = {
    "roc_auc": 0.7154,
    "recall": 0.6373,
    "precision": 0.1472,
    "accuracy": 0.6821,
    "f1": 0.2392,
}

V1_CORE_FEATURES = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "LOAN_TO_INCOME_RATIO",
]

# -----------------------------------------------------------------------------
# Regulatory & Schema Governance Constants
# -----------------------------------------------------------------------------
# 17 columns with >60% missing values in application_train.csv
HIGH_NULL_COLS_60 = [
    "COMMONAREA_AVG",
    "COMMONAREA_MODE",
    "COMMONAREA_MEDI",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAPARTMENTS_MEDI",
    "FONDKAPREMONT_MODE",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAPARTMENTS_MODE",
    "LIVINGAPARTMENTS_MEDI",
    "FLOORSMIN_AVG",
    "FLOORSMIN_MODE",
    "FLOORSMIN_MEDI",
    "YEARS_BUILD_AVG",
    "YEARS_BUILD_MODE",
    "YEARS_BUILD_MEDI",
    "OWN_CAR_AGE",
]

EDUCATION_ORDINAL_MAP = {
    "Lower secondary": 0,
    "Secondary / secondary special": 1,
    "Incomplete higher": 2,
    "Higher education": 3,
    "Academic degree": 4,
}

# -----------------------------------------------------------------------------
# Basel Foundation-IRB Economic Parameters
# -----------------------------------------------------------------------------
LGD_BASE = 0.45          # Senior Unsecured Loss Given Default
NET_MARGIN_RATE = 0.09   # Opportunity cost of false rejection
COST_RATIO_DEFAULT = 5.0 # r = LGD / Margin = 0.45 / 0.09 = 5.0
COST_SENSITIVITY_RATIOS = [2.0, 3.0, 5.0, 10.0, 20.0]

