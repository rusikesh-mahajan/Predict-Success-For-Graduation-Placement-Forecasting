"""
Data loading and validation utilities.

Provides functions to load the student dataset from CSV, validate
its structure, compute summary statistics, and handle duplicates.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import DEFAULT_DATASET, ID_COL, EXPECTED_COLUMNS
from src.utils import setup_logging

logger = setup_logging(__name__)


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the student dataset from a CSV file.

    Parameters
    ----------
    path : str, optional
        Absolute or relative path to the CSV file.  Falls back to
        ``config.settings.DEFAULT_DATASET`` when not provided.

    Returns
    -------
    pd.DataFrame
        The raw student dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the given path.
    ValueError
        If required columns are missing from the loaded data.
    """
    if path is None:
        path = DEFAULT_DATASET

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run 'python data/generate_dataset.py' first, "
            "or place your CSV at that path."
        )

    logger.info("Loading dataset from '%s'", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # Validate expected columns
    validate_columns(df)

    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Verify the DataFrame contains all expected columns.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Raises
    ------
    ValueError
        If any expected columns are missing.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns.tolist())}"
        )
    logger.debug("Column validation passed — all %d expected columns present.", len(EXPECTED_COLUMNS))


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a comprehensive summary of the DataFrame.

    Returns
    -------
    dict
        Keys: ``shape``, ``n_rows``, ``n_cols``, ``dtypes``,
        ``numerical_cols``, ``categorical_cols``, ``missing_total``,
        ``missing_by_col``, ``n_duplicates``, ``memory_mb``.
    """
    num_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols: List[str] = df.select_dtypes(include=["object"]).columns.tolist()

    # Exclude ID col for duplicate check
    feature_cols = [c for c in df.columns if c != ID_COL]
    n_dups: int = int(df.duplicated(subset=feature_cols, keep="first").sum())

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame(
        {"Count": missing, "Percentage": missing_pct}
    ).sort_values("Count", ascending=False)

    summary = {
        "shape": df.shape,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "dtypes": df.dtypes,
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "missing_total": int(missing.sum()),
        "missing_by_col": missing_df[missing_df["Count"] > 0],
        "n_duplicates": n_dups,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }

    logger.info(
        "Data summary: %d rows, %d cols, %d missing, %d duplicates",
        summary["n_rows"], summary["n_cols"],
        summary["missing_total"], summary["n_duplicates"],
    )

    return summary


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows (excluding ``Student_ID``) and reset the index.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        De-duplicated DataFrame with a reset integer index.
    """
    feature_cols = [c for c in df.columns if c != ID_COL]
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    after = len(df)
    logger.info("Removed %d duplicate rows (%d → %d)", before - after, before, after)
    return df
