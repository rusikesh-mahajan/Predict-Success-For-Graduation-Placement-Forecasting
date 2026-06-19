"""
Feature engineering: create composite and derived features.

All transformations are implemented as a scikit-learn compatible
transformer so they can be included in a ``Pipeline`` without
data leakage.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.settings import (
    ACADEMIC_INDEX_WEIGHTS,
    EMPLOYABILITY_WEIGHTS,
    RISK_FLAG_THRESHOLDS,
)
from src.utils import setup_logging

logger = setup_logging(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that creates derived features.

    New columns added
    -----------------
    * ``Total_Marks``         — Internal + External marks
    * ``Marks_Ratio``         — Internal / Total (balance indicator)
    * ``Academic_Index``      — Weighted composite of academic metrics
    * ``Employability_Score`` — Weighted composite of job-readiness factors
    * ``Risk_Flag``           — Binary: backlogs > threshold AND attendance < threshold
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """No fitting required — returns self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain the base columns (post-imputation).

        Returns
        -------
        pd.DataFrame
            Copy of *X* with new columns appended.
        """
        df = X.copy()
        df = self._add_total_marks(df)
        df = self._add_marks_ratio(df)
        df = self._add_academic_index(df)
        df = self._add_employability_score(df)
        df = self._add_risk_flag(df)

        logger.debug("Feature engineering complete — %d new columns added.", 5)
        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return output feature names."""
        base = list(input_features) if input_features is not None else []
        return base + [
            "Total_Marks", "Marks_Ratio", "Academic_Index",
            "Employability_Score", "Risk_Flag",
        ]

    # ── Private helpers ────────────────────────────────────────────

    @staticmethod
    def _add_total_marks(df: pd.DataFrame) -> pd.DataFrame:
        df["Total_Marks"] = df["Internal_Marks"] + df["External_Marks"]
        return df

    @staticmethod
    def _add_marks_ratio(df: pd.DataFrame) -> pd.DataFrame:
        df["Marks_Ratio"] = df["Internal_Marks"] / (df["Total_Marks"] + 1e-8)
        return df

    @staticmethod
    def _add_academic_index(df: pd.DataFrame) -> pd.DataFrame:
        w = ACADEMIC_INDEX_WEIGHTS
        df["Academic_Index"] = (
            w["cgpa"] * (df["CGPA"] / 10) * 100
            + w["attendance"] * df["Attendance_Percentage"]
            + w["marks"] * (df["Total_Marks"] / 200) * 100
        )
        return df

    @staticmethod
    def _add_employability_score(df: pd.DataFrame) -> pd.DataFrame:
        w = EMPLOYABILITY_WEIGHTS
        df["Employability_Score"] = (
            w["communication"] * (df["Communication_Skills_Score"] / 10) * 100
            + w["aptitude"] * df["Aptitude_Test_Score"]
            + w["projects"] * (df["Projects_Completed"] / 5) * 100
            + w["internship"] * df["Internship_Experience"] * 100
        )
        return df

    @staticmethod
    def _add_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
        t = RISK_FLAG_THRESHOLDS
        df["Risk_Flag"] = (
            (df["Backlogs"] > t["backlogs"])
            & (df["Attendance_Percentage"] < t["attendance"])
        ).astype(int)
        return df


# ── Convenience function (backwards compatibility) ─────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Functional wrapper around :class:`FeatureEngineer`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially-processed student data.

    Returns
    -------
    pd.DataFrame
        Data with engineered features appended.
    """
    return FeatureEngineer().fit_transform(df)
