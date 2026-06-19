"""
Unit tests for the model training module.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model_training import (
    get_models,
    train_evaluate_models,
    get_feature_importances,
    cross_validate_models,
)


@pytest.fixture
def classification_data():
    """Generate synthetic classification data for testing."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        random_state=42,
    )
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = pd.Series(y[:160]), pd.Series(y[160:])
    return X_train, X_test, y_train, y_test


class TestGetModels:
    """Tests for get_models()."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        models = get_models()
        assert isinstance(models, dict)

    def test_has_required_models(self):
        """Should include all required models."""
        models = get_models()
        required = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
        for name in required:
            assert name in models, f"Missing model: {name}"

    def test_at_least_four_models(self):
        """Should have at least 4 models."""
        models = get_models()
        assert len(models) >= 4


class TestTrainEvaluateModels:
    """Tests for train_evaluate_models()."""

    def test_returns_results_and_models(self, classification_data):
        """Should return (DataFrame, dict)."""
        X_train, X_test, y_train, y_test = classification_data
        results_df, trained_models = train_evaluate_models(
            X_train, X_test, y_train, y_test,
        )
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(trained_models, dict)

    def test_results_have_metrics(self, classification_data):
        """Results should contain expected metric columns."""
        X_train, X_test, y_train, y_test = classification_data
        results_df, _ = train_evaluate_models(
            X_train, X_test, y_train, y_test,
        )
        expected_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        for col in expected_cols:
            assert col in results_df.columns

    def test_metrics_in_valid_range(self, classification_data):
        """All metric values should be between 0 and 1."""
        X_train, X_test, y_train, y_test = classification_data
        results_df, _ = train_evaluate_models(
            X_train, X_test, y_train, y_test,
        )
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            assert (results_df[col] >= 0).all()
            assert (results_df[col] <= 1).all()

    def test_sorted_by_roc_auc(self, classification_data):
        """Results should be sorted by ROC-AUC descending."""
        X_train, X_test, y_train, y_test = classification_data
        results_df, _ = train_evaluate_models(
            X_train, X_test, y_train, y_test,
        )
        roc_values = results_df["ROC-AUC"].values
        assert all(roc_values[i] >= roc_values[i + 1] for i in range(len(roc_values) - 1))


class TestCrossValidateModels:
    """Tests for cross_validate_models()."""

    def test_returns_dataframe(self, classification_data):
        """Should return a DataFrame."""
        X_train, _, y_train, _ = classification_data
        cv_df = cross_validate_models(X_train, y_train, n_folds=3)
        assert isinstance(cv_df, pd.DataFrame)

    def test_has_mean_columns(self, classification_data):
        """Should have mean score columns."""
        X_train, _, y_train, _ = classification_data
        cv_df = cross_validate_models(X_train, y_train, n_folds=3)
        assert "accuracy_mean" in cv_df.columns
        assert "roc_auc_mean" in cv_df.columns


class TestGetFeatureImportances:
    """Tests for get_feature_importances()."""

    def test_returns_series(self, classification_data):
        """Should return a pandas Series."""
        X_train, X_test, y_train, y_test = classification_data
        _, trained_models = train_evaluate_models(
            X_train, X_test, y_train, y_test,
        )

        if "Random Forest" in trained_models:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            importances = get_feature_importances(
                trained_models["Random Forest"], feature_names,
            )
            assert isinstance(importances, pd.Series)
            assert len(importances) > 0
