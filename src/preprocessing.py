"""
Data preprocessing: missing value imputation, outlier capping,
encoding, scaling, and train/test splitting.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config.settings import (
    OUTLIER_COLS,
    TARGET_COLS,
    ID_COL,
    TEST_SIZE,
    RANDOM_STATE,
)


# ── Missing Values ─────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numerical values with the **median** and
    categorical values with the **mode**.

    Returns a copy of the DataFrame.
    """
    df = df.copy()

    # Numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# ── Outlier Handling ───────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Cap outliers using the IQR method (1.5 × IQR).

    Parameters
    ----------
    df : pd.DataFrame
    cols : list, optional
        Columns to check.  Defaults to ``OUTLIER_COLS``.
    """
    df = df.copy()
    if cols is None:
        cols = OUTLIER_COLS

    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


# ── Encoding ───────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical features.

    * Binary → LabelEncoder
    * Multi-class → One-Hot Encoding (drop_first)

    Returns
    -------
    df : pd.DataFrame
    encoders : dict  – mapping {col_name: LabelEncoder} for binary cols
    """
    df = df.copy()

    # Drop ID column if present
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}

    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique == 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)

    return df, encoders


# ── Scaling ────────────────────────────────────────────────────────
def scale_features(
    df: pd.DataFrame, feature_cols: list, scaler: StandardScaler = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise feature columns (mean=0, std=1).

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list
    scaler : StandardScaler, optional
        If provided, uses ``transform`` (for inference). Otherwise ``fit_transform``.

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


# ── Train/Test Split ──────────────────────────────────────────────
def split_data(
    df: pd.DataFrame, target_col: str, feature_cols: list = None
) -> tuple:
    """
    Split into train/test sets with stratification.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in TARGET_COLS]

    X = df[feature_cols]
    y = df[target_col]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ── Full Pipeline ─────────────────────────────────────────────────
def run_preprocessing_pipeline(df: pd.DataFrame) -> tuple:
    """
    Run the complete preprocessing pipeline:
    missing values → outliers → encoding → return clean df + encoders.

    Returns
    -------
    df_clean : pd.DataFrame
    encoders : dict
    """
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df, encoders = encode_features(df)
    return df, encoders
