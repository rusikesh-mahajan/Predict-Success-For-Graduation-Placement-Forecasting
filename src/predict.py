"""
Inference pipeline for new student data.

Prepares raw student records for prediction by applying the same
feature engineering and preprocessing used during training.
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import scale_features
from src.utils import setup_logging

logger = setup_logging(__name__)


def prepare_new_student(
    student_data: pd.DataFrame,
    feature_cols: List[str],
    scaler: Any,
) -> pd.DataFrame:
    """
    Apply feature engineering and scaling to new student records.

    Parameters
    ----------
    student_data : pd.DataFrame
        Raw input data (1 or more rows).
    feature_cols : list of str
        The exact list of columns the model expects.
    scaler : StandardScaler
        The fitted scaler from training.

    Returns
    -------
    pd.DataFrame
        Processed and scaled data ready for ``model.predict()``.
    """
    logger.info("Preparing %d student record(s) for prediction.", len(student_data))

    try:
        # 1. Feature Engineering
        df_eng = engineer_features(student_data)

        # 2. Ensure all expected columns are present
        for col in feature_cols:
            if col not in df_eng.columns:
                df_eng[col] = 0
                logger.debug("Added missing column '%s' (filled with 0).", col)

        # 3. Ensure correct column order
        df_eng = df_eng[feature_cols]

        # 4. Handle any NaN/Inf values
        df_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
        if df_eng.isnull().values.any():
            df_eng.fillna(0, inplace=True)

        # 5. Scaling
        df_scaled, _ = scale_features(df_eng, feature_cols, scaler=scaler)

        logger.info("Student data prepared successfully.")
        return df_scaled

    except Exception as e:
        logger.error("Error preparing student data: %s", e)
        raise


def predict_student(
    model: Any,
    student_data_scaled: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict outcome and probability for prepared student data.

    Parameters
    ----------
    model : estimator
        A fitted sklearn-compatible classifier.
    student_data_scaled : pd.DataFrame
        Preprocessed and scaled student data.

    Returns
    -------
    predictions : np.ndarray
        Binary predictions (0 or 1).
    probabilities : np.ndarray
        Probability of positive class (class 1).
    """
    try:
        pred = model.predict(student_data_scaled)
        prob = model.predict_proba(student_data_scaled)[:, 1]

        logger.info(
            "Prediction complete — %d records, positive rate: %.1f%%",
            len(pred), (pred.mean() * 100),
        )

        return pred, prob

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise


def validate_input(student_data: pd.DataFrame) -> List[str]:
    """
    Validate input student data and return a list of warnings.

    Parameters
    ----------
    student_data : pd.DataFrame
        Raw input data.

    Returns
    -------
    list of str
        Warning messages for any issues found.
    """
    warnings: List[str] = []

    # Check CGPA range
    if "CGPA" in student_data.columns:
        cgpa = student_data["CGPA"].iloc[0]
        if cgpa < 0 or cgpa > 10:
            warnings.append(f"CGPA ({cgpa}) is outside valid range [0, 10].")

    # Check attendance range
    if "Attendance_Percentage" in student_data.columns:
        att = student_data["Attendance_Percentage"].iloc[0]
        if att < 0 or att > 100:
            warnings.append(f"Attendance ({att}%) is outside valid range [0, 100].")

    # Check marks range
    for col in ["Internal_Marks", "External_Marks"]:
        if col in student_data.columns:
            val = student_data[col].iloc[0]
            if val < 0 or val > 100:
                warnings.append(f"{col} ({val}) is outside valid range [0, 100].")

    if warnings:
        for w in warnings:
            logger.warning("Input validation: %s", w)

    return warnings
