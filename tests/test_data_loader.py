"""
Unit tests for the data loading and validation module.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on the path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import load_data, get_data_summary, remove_duplicates


class TestLoadData:
    """Tests for load_data()."""

    def test_load_data_returns_dataframe(self):
        """load_data should return a pd.DataFrame."""
        df = load_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_data_has_rows(self):
        """Loaded data should have rows."""
        df = load_data()
        assert len(df) > 0

    def test_load_data_has_expected_columns(self):
        """Loaded data should contain expected columns."""
        df = load_data()
        expected = ["Student_ID", "CGPA", "Graduation_Status", "Placement_Status"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_data_invalid_path(self):
        """Should raise FileNotFoundError for invalid path."""
        with pytest.raises(FileNotFoundError):
            load_data(path="nonexistent_file.csv")


class TestGetDataSummary:
    """Tests for get_data_summary()."""

    def test_summary_keys(self):
        """Summary should have all expected keys."""
        df = load_data()
        summary = get_data_summary(df)
        expected_keys = [
            "shape", "n_rows", "n_cols", "numerical_cols",
            "categorical_cols", "missing_total", "n_duplicates",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_summary_types(self):
        """Summary values should be correct types."""
        df = load_data()
        summary = get_data_summary(df)
        assert isinstance(summary["n_rows"], int)
        assert isinstance(summary["n_cols"], int)
        assert isinstance(summary["missing_total"], int)
        assert isinstance(summary["numerical_cols"], list)


class TestRemoveDuplicates:
    """Tests for remove_duplicates()."""

    def test_removes_duplicates(self):
        """Should remove duplicate rows."""
        df = load_data()
        original_len = len(df)
        df_clean = remove_duplicates(df)
        assert len(df_clean) <= original_len

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        df = load_data()
        result = remove_duplicates(df)
        assert isinstance(result, pd.DataFrame)
