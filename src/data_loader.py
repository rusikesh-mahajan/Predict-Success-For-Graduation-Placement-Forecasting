"""
Data loading and validation utilities.
"""
import pandas as pd
import numpy as np
import os
from config.settings import DEFAULT_DATASET, ID_COL


def load_data(path: str = None) -> pd.DataFrame:
    """
    Load the student dataset from CSV.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file.  Falls back to the default location
        configured in ``config/settings.py``.

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = DEFAULT_DATASET

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run 'python data/generate_dataset.py' first, "
            "or place your CSV at that path."
        )

    df = pd.read_csv(path)
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Return a dictionary summarising the DataFrame.

    Keys
    ----
    shape, n_rows, n_cols, dtypes, numerical_cols, categorical_cols,
    missing_total, missing_by_col, n_duplicates, memory_mb
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Exclude ID col for duplicate check
    feature_cols = [c for c in df.columns if c != ID_COL]
    n_dups = df.duplicated(subset=feature_cols, keep="first").sum()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame(
        {"Count": missing, "Percentage": missing_pct}
    ).sort_values("Count", ascending=False)

    return {
        "shape": df.shape,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "dtypes": df.dtypes,
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "missing_total": int(missing.sum()),
        "missing_by_col": missing_df[missing_df["Count"] > 0],
        "n_duplicates": int(n_dups),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows (excluding Student_ID) and reset the index."""
    feature_cols = [c for c in df.columns if c != ID_COL]
    df = df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    return df
