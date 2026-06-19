"""
Data preprocessing: sklearn Pipeline with proper train/val/test methodology.

This module builds a complete preprocessing pipeline that avoids data
leakage by fitting only on training data and transforming validation
and test sets separately.

Pipeline stages
----------------
1. Feature engineering (custom transformer)
2. Missing-value imputation (median / mode)
3. Outlier capping (IQR method via custom transformer)
4. Categorical encoding (OneHotEncoder / OrdinalEncoder)
5. Feature scaling (StandardScaler)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import (
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERICAL_FEATURES,
    OUTLIER_COLS,
    RANDOM_STATE,
    TARGET_COLS,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.feature_engineering import FeatureEngineer
from src.utils import setup_logging

logger = setup_logging(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Custom Transformers
# ═══════════════════════════════════════════════════════════════════

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Cap outliers using the IQR method (1.5 × IQR).

    Learns the bounds from the *training* data during ``fit()`` and
    applies the same bounds during ``transform()`` to prevent leakage.

    Parameters
    ----------
    columns : list of str, optional
        Columns to cap.  Defaults to ``OUTLIER_COLS`` from config.
    factor : float
        IQR multiplier.  Default is ``1.5``.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        factor: float = 1.5,
    ) -> None:
        self.columns = columns or OUTLIER_COLS
        self.factor = factor
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OutlierCapper":
        """Learn IQR bounds from training data."""
        self.bounds_ = {}
        for col in self.columns:
            if col in X.columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.bounds_[col] = (
                    q1 - self.factor * iqr,
                    q3 + self.factor * iqr,
                )
        logger.debug("OutlierCapper fitted on %d columns.", len(self.bounds_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned bounds to cap outliers."""
        df = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drop specified columns from a DataFrame.

    Used in the pipeline to remove ID and target columns before
    model training.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnDropper":
        """No fitting required."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the specified columns."""
        cols_to_drop = [c for c in self.columns if c in X.columns]
        return X.drop(columns=cols_to_drop)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values while preserving DataFrame structure.

    Numerical columns: **median**
    Categorical columns: **most frequent**
    """

    def __init__(self) -> None:
        self.num_imputer_ = SimpleImputer(strategy="median")
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataFrameImputer":
        """Fit imputers on training data."""
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=["object"]).columns.tolist()

        if self.num_cols_:
            self.num_imputer_.fit(X[self.num_cols_])
        if self.cat_cols_:
            self.cat_imputer_.fit(X[self.cat_cols_])

        logger.debug(
            "DataFrameImputer fitted — %d numerical, %d categorical columns.",
            len(self.num_cols_), len(self.cat_cols_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation."""
        df = X.copy()

        if self.num_cols_:
            existing_num = [c for c in self.num_cols_ if c in df.columns]
            if existing_num:
                df[existing_num] = self.num_imputer_.transform(df[existing_num])

        if self.cat_cols_:
            existing_cat = [c for c in self.cat_cols_ if c in df.columns]
            if existing_cat:
                df[existing_cat] = self.cat_imputer_.transform(df[existing_cat])

        # Replace any remaining inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if df.isnull().values.any():
            df.fillna(0, inplace=True)

        return df


# ═══════════════════════════════════════════════════════════════════
#  Pipeline Builder
# ═══════════════════════════════════════════════════════════════════

def build_preprocessing_pipeline(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Pipeline:
    """
    Build the full sklearn preprocessing Pipeline.

    The pipeline processes data through:
    1. Column dropping (ID, targets)
    2. Feature engineering
    3. Missing value imputation
    4. Outlier capping
    5. Column transformation (scaling + encoding)

    Parameters
    ----------
    numerical_features : list of str, optional
        Override default numerical feature list.
    categorical_features : list of str, optional
        Override default categorical feature list.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted preprocessing pipeline.
    """
    num_feats = numerical_features or NUMERICAL_FEATURES
    cat_feats = categorical_features or CATEGORICAL_FEATURES

    # All numerical features after feature engineering
    engineered_numerical = num_feats + [
        "Total_Marks", "Marks_Ratio", "Academic_Index",
        "Employability_Score", "Risk_Flag",
    ]

    # Column transformer: scale numerics, encode categoricals
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                engineered_numerical,
            ),
            (
                "categorical",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                cat_feats,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("drop_columns", ColumnDropper(columns=[ID_COL] + TARGET_COLS)),
            ("feature_engineer", FeatureEngineer()),
            ("imputer", DataFrameImputer()),
            ("outlier_capper", OutlierCapper()),
            ("column_transformer", column_transformer),
        ]
    )

    logger.info("Preprocessing pipeline built successfully.")
    return pipeline


# ═══════════════════════════════════════════════════════════════════
#  Data Splitting
# ═══════════════════════════════════════════════════════════════════

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict:
    """
    Split data into train, validation, and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (must include the target column).
    target_col : str
        Name of the target column.
    test_size : float
        Fraction of data for the test set.
    val_size : float
        Fraction of data for the validation set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``X_train``, ``X_val``, ``X_test``,
        ``y_train``, ``y_val``, ``y_test``.
    """
    # Separate features and target
    drop_cols = [c for c in TARGET_COLS if c != target_col and c in df.columns]
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Second split: train vs validation
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        "Data split: train=%d, val=%d, test=%d (target='%s')",
        len(X_train), len(X_val), len(X_test), target_col,
    )

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_val": X_val.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


# ═══════════════════════════════════════════════════════════════════
#  Legacy Functions (backward compatibility for Streamlit pages)
# ═══════════════════════════════════════════════════════════════════

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy wrapper: impute missing values using median/mode."""
    return DataFrameImputer().fit_transform(df)


def handle_outliers(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Legacy wrapper: cap outliers using IQR."""
    capper = OutlierCapper(columns=cols)
    return capper.fit_transform(df)


def run_preprocessing_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Legacy wrapper: run the preprocessing pipeline.

    Returns
    -------
    df_clean : pd.DataFrame
    encoders : dict (empty — encoding is now handled inside the Pipeline)
    """
    df = handle_missing_values(df)
    df = handle_outliers(df)
    # Drop ID column
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    # Encode categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Legacy wrapper: standardise feature columns.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of str
    scaler : StandardScaler, optional
        If provided, uses ``transform``.  Otherwise ``fit_transform``.

    Returns
    -------
    df_scaled : pd.DataFrame
    scaler : StandardScaler
    """
    df_scaled = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    return df_scaled, scaler
