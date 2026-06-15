"""
Feature engineering: create composite and derived features.
"""
import pandas as pd
from config.settings import (
    ACADEMIC_INDEX_WEIGHTS,
    EMPLOYABILITY_WEIGHTS,
    RISK_FLAG_THRESHOLDS,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the DataFrame.

    New columns
    -----------
    * ``Total_Marks``         – Internal + External marks
    * ``Marks_Ratio``         – Internal / Total (balance indicator)
    * ``Academic_Index``      – Weighted composite of academic metrics
    * ``Employability_Score`` – Weighted composite of job-readiness factors
    * ``Risk_Flag``           – Binary: backlogs > 2 AND attendance < 60

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain the base columns (post-imputation).

    Returns
    -------
    pd.DataFrame  (copy with new columns)
    """
    df = df.copy()

    # 1. Total Marks
    df["Total_Marks"] = df["Internal_Marks"] + df["External_Marks"]

    # 2. Marks Ratio  (epsilon to avoid division-by-zero)
    df["Marks_Ratio"] = df["Internal_Marks"] / (df["Total_Marks"] + 1e-8)

    # 3. Academic Index
    w = ACADEMIC_INDEX_WEIGHTS
    df["Academic_Index"] = (
        w["cgpa"] * (df["CGPA"] / 10) * 100
        + w["attendance"] * df["Attendance_Percentage"]
        + w["marks"] * (df["Total_Marks"] / 200) * 100
    )

    # 4. Employability Score
    w = EMPLOYABILITY_WEIGHTS
    df["Employability_Score"] = (
        w["communication"] * (df["Communication_Skills_Score"] / 10) * 100
        + w["aptitude"] * df["Aptitude_Test_Score"]
        + w["projects"] * (df["Projects_Completed"] / 5) * 100
        + w["internship"] * df["Internship_Experience"] * 100
    )

    # 5. Risk Flag
    t = RISK_FLAG_THRESHOLDS
    df["Risk_Flag"] = (
        (df["Backlogs"] > t["backlogs"])
        & (df["Attendance_Percentage"] < t["attendance"])
    ).astype(int)

    return df
