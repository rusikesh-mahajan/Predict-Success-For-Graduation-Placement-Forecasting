"""
Inference pipeline for new student data.
"""
import pandas as pd
from .feature_engineering import engineer_features
from .preprocessing import scale_features

def prepare_new_student(
    student_data: pd.DataFrame, feature_cols: list, scaler
) -> pd.DataFrame:
    """
    Apply feature engineering and scaling to new student records.

    Parameters
    ----------
    student_data : pd.DataFrame
        Raw input data (1 or more rows).
    feature_cols : list
        The exact list of columns the model expects.
    scaler : StandardScaler
        The fitted scaler from training.

    Returns
    -------
    pd.DataFrame
        Processed and scaled data ready for model.predict().
    """
    # 1. Feature Engineering
    df_eng = engineer_features(student_data)

    # Ensure all expected columns are present (e.g. one-hot encoded columns)
    # For this simplified app, we assume inputs are numerical or already encoded,
    # but in a full pipeline we'd apply the saved encoders here.
    # To prevent missing column errors, we add them filled with 0s if missing.
    for col in feature_cols:
        if col not in df_eng.columns:
            df_eng[col] = 0

    # Ensure correct column order
    df_eng = df_eng[feature_cols]

    # 2. Scaling
    df_scaled, _ = scale_features(df_eng, feature_cols, scaler=scaler)

    return df_scaled


def predict_student(model, student_data_scaled: pd.DataFrame) -> tuple:
    """
    Predict outcome and probability for the given students.

    Returns
    -------
    predictions : np.ndarray
    probabilities : np.ndarray (for class 1)
    """
    pred = model.predict(student_data_scaled)
    prob = model.predict_proba(student_data_scaled)[:, 1]
    return pred, prob
