"""
Project-wide configuration and constants.

This module centralises all configuration values used across the
project — file paths, model hyper-parameters, feature engineering
weights, and training settings.
"""
import os
from typing import Dict, List

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR: str = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR: str = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")
REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
FIGURES_DIR: str = os.path.join(REPORTS_DIR, "figures")
DOCS_DIR: str = os.path.join(BASE_DIR, "docs")
TESTS_DIR: str = os.path.join(BASE_DIR, "tests")

DEFAULT_DATASET: str = os.path.join(DATA_RAW_DIR, "student_data.csv")

# ── Target Variables ───────────────────────────────────────────────
TARGET_COLS: List[str] = ["Graduation_Status", "Placement_Status"]
ID_COL: str = "Student_ID"

# ── Expected Raw Columns ──────────────────────────────────────────
EXPECTED_COLUMNS: List[str] = [
    "Student_ID", "Gender", "Age", "Attendance_Percentage", "CGPA",
    "Internal_Marks", "External_Marks", "Backlogs", "Study_Hours_Per_Week",
    "Internship_Experience", "Projects_Completed", "Communication_Skills_Score",
    "Aptitude_Test_Score", "Graduation_Status", "Placement_Status",
]

# ── Numerical & Categorical Feature Lists ─────────────────────────
NUMERICAL_FEATURES: List[str] = [
    "Age", "Attendance_Percentage", "CGPA", "Internal_Marks",
    "External_Marks", "Backlogs", "Study_Hours_Per_Week",
    "Internship_Experience", "Projects_Completed",
    "Communication_Skills_Score", "Aptitude_Test_Score",
]

CATEGORICAL_FEATURES: List[str] = ["Gender"]

# ── Feature Engineering Weights ────────────────────────────────────
ACADEMIC_INDEX_WEIGHTS: Dict[str, float] = {
    "cgpa": 0.4,
    "attendance": 0.3,
    "marks": 0.3,
}

EMPLOYABILITY_WEIGHTS: Dict[str, float] = {
    "communication": 0.30,
    "aptitude": 0.30,
    "projects": 0.20,
    "internship": 0.20,
}

RISK_FLAG_THRESHOLDS: Dict[str, float] = {
    "backlogs": 2,
    "attendance": 60,
}

# ── Outlier Handling ───────────────────────────────────────────────
OUTLIER_COLS: List[str] = [
    "Attendance_Percentage",
    "CGPA",
    "Internal_Marks",
    "External_Marks",
    "Study_Hours_Per_Week",
    "Communication_Skills_Score",
    "Aptitude_Test_Score",
]

# ── Model Hyper-parameter Grids (for tuning) ──────────────────────
PARAM_GRIDS: Dict[str, dict] = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    "Decision Tree": {
        "classifier__max_depth": [3, 5, 7, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5, 10],
        "classifier__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [5, 10, 15, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 5],
        "classifier__max_features": ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__reg_alpha": [0, 0.1, 1],
        "classifier__reg_lambda": [0.1, 1, 10],
    },
}

# ── Train / Validation / Test Split ──────────────────────────────
TEST_SIZE: float = 0.2
VALIDATION_SIZE: float = 0.1  # 10% of full data for validation
RANDOM_STATE: int = 42

# ── Cross-Validation ─────────────────────────────────────────────
N_FOLDS: int = 5

# ── Tuning ────────────────────────────────────────────────────────
TUNING_N_ITER: int = 50
TUNING_CV: int = 5

# ── Logging ───────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
