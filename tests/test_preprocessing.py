"""
Unit tests for the preprocessing module.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import (
    DataFrameImputer,
    OutlierCapper,
    ColumnDropper,
    handle_missing_values,
    handle_outliers,
    scale_features,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "num_col_1": [1.0, 2.0, np.nan, 4.0, 5.0, 100.0],
        "num_col_2": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0],
        "cat_col": ["A", "B", np.nan, "A", "B", "A"],
    })


class TestDataFrameImputer:
    """Tests for DataFrameImputer."""

    def test_imputes_numerical_nan(self, sample_df):
        """Should impute numerical NaN with median."""
        imputer = DataFrameImputer()
        result = imputer.fit_transform(sample_df)
        assert result["num_col_1"].isnull().sum() == 0
        assert result["num_col_2"].isnull().sum() == 0

    def test_imputes_categorical_nan(self, sample_df):
        """Should impute categorical NaN with mode."""
        imputer = DataFrameImputer()
        result = imputer.fit_transform(sample_df)
        assert result["cat_col"].isnull().sum() == 0

    def test_preserves_shape(self, sample_df):
        """Should not change DataFrame shape."""
        imputer = DataFrameImputer()
        result = imputer.fit_transform(sample_df)
        assert result.shape == sample_df.shape


class TestOutlierCapper:
    """Tests for OutlierCapper."""

    def test_caps_outliers(self, sample_df):
        """Values should be within IQR bounds after capping."""
        capper = OutlierCapper(columns=["num_col_1"])
        capper.fit(sample_df)
        result = capper.transform(sample_df)
        # 100.0 should be capped
        assert result["num_col_1"].max() < 100.0

    def test_bounds_stored(self, sample_df):
        """Bounds should be stored after fitting."""
        capper = OutlierCapper(columns=["num_col_1"])
        capper.fit(sample_df)
        assert "num_col_1" in capper.bounds_


class TestColumnDropper:
    """Tests for ColumnDropper."""

    def test_drops_columns(self, sample_df):
        """Should drop specified columns."""
        dropper = ColumnDropper(columns=["cat_col"])
        result = dropper.fit_transform(sample_df)
        assert "cat_col" not in result.columns

    def test_ignores_missing_columns(self, sample_df):
        """Should not error if column doesn't exist."""
        dropper = ColumnDropper(columns=["nonexistent"])
        result = dropper.fit_transform(sample_df)
        assert result.shape == sample_df.shape


class TestScaleFeatures:
    """Tests for scale_features."""

    def test_scales_features(self):
        """Scaled features should have ~0 mean and ~1 std."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        scaled, scaler = scale_features(df, ["a", "b"])
        assert abs(scaled["a"].mean()) < 1e-10
        assert abs(scaled["b"].mean()) < 1e-10

    def test_returns_scaler(self):
        """Should return a fitted scaler."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        _, scaler = scale_features(df, ["a"])
        assert scaler is not None
