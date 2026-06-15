"""
Project-wide configuration and constants.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

DEFAULT_DATASET = os.path.join(DATA_RAW_DIR, "student_data.csv")

# ── Target Variables ───────────────────────────────────────────────
TARGET_COLS = ["Graduation_Status", "Placement_Status"]
ID_COL = "Student_ID"

# ── Feature Engineering Weights ────────────────────────────────────
ACADEMIC_INDEX_WEIGHTS = {"cgpa": 0.4, "attendance": 0.3, "marks": 0.3}
EMPLOYABILITY_WEIGHTS = {
    "communication": 0.30,
    "aptitude": 0.30,
    "projects": 0.20,
    "internship": 0.20,
}
RISK_FLAG_THRESHOLDS = {"backlogs": 2, "attendance": 60}

# ── Outlier Handling ───────────────────────────────────────────────
OUTLIER_COLS = [
    "Attendance_Percentage",
    "CGPA",
    "Internal_Marks",
    "External_Marks",
    "Study_Hours_Per_Week",
    "Communication_Skills_Score",
    "Aptitude_Test_Score",
]

# ── Model Hyper-parameter Grids (for tuning) ──────────────────────
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [0.1, 1, 10],
    },
}

# ── Train/Test Split ──────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Tuning ────────────────────────────────────────────────────────
TUNING_N_ITER = 50
TUNING_CV = 5
