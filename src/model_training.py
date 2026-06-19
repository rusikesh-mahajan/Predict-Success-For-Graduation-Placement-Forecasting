"""
Model training and evaluation utilities.

Supports training multiple classifiers, Stratified K-Fold
cross-validation, and holdout evaluation.  Returns structured
results for comparison.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config.settings import N_FOLDS, RANDOM_STATE
from src.utils import setup_logging

logger = setup_logging(__name__)

# ── Optional XGBoost ──────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE: bool = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed — skipping XGBClassifier.")


# ═══════════════════════════════════════════════════════════════════
#  Model Definitions
# ═══════════════════════════════════════════════════════════════════

def get_models() -> Dict[str, Any]:
    """
    Return an ordered dictionary of model name → untrained estimator.

    Models included:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - XGBoost (if installed)

    Returns
    -------
    dict
        ``{name: estimator}``
    """
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=10,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE, learning_rate=0.1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

    logger.info("Loaded %d model definitions.", len(models))
    return models


# ═══════════════════════════════════════════════════════════════════
#  Stratified K-Fold Cross-Validation
# ═══════════════════════════════════════════════════════════════════

def cross_validate_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    models: Optional[Dict[str, Any]] = None,
    n_folds: int = N_FOLDS,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Run Stratified K-Fold cross-validation on each model.

    Parameters
    ----------
    X_train : array-like
        Training features (already preprocessed).
    y_train : Series
        Training labels.
    models : dict, optional
        ``{name: estimator}``.  Defaults to :func:`get_models`.
    n_folds : int
        Number of CV folds.
    progress_callback : callable, optional
        ``callback(step, total, model_name)`` for progress bars.

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = metric mean ± std.
    """
    if models is None:
        models = get_models()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results: List[Dict[str, Any]] = []
    total = len(models)

    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(idx, total, f"CV: {name}")

        logger.info("Cross-validating %s (%d-fold)...", name, n_folds)

        try:
            scores = cross_validate(
                model, X_train, y_train,
                cv=skf,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False,
            )

            result = {"Model": name}
            for metric_name in scoring:
                key = f"test_{metric_name}"
                result[f"{metric_name}_mean"] = round(scores[key].mean(), 4)
                result[f"{metric_name}_std"] = round(scores[key].std(), 4)

            cv_results.append(result)

        except Exception as e:
            logger.error("Cross-validation failed for %s: %s", name, e)
            cv_results.append({"Model": name, "error": str(e)})

    if progress_callback:
        progress_callback(total, total, "CV Complete")

    cv_df = pd.DataFrame(cv_results).set_index("Model")
    cv_df = cv_df.sort_values("roc_auc_mean", ascending=False)

    logger.info("Cross-validation complete for %d models.", len(models))
    return cv_df


# ═══════════════════════════════════════════════════════════════════
#  Holdout Training & Evaluation
# ═══════════════════════════════════════════════════════════════════

def train_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    models: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train every model on the training set and evaluate on the test set.

    Parameters
    ----------
    X_train, X_test : array-like
        Preprocessed feature matrices.
    y_train, y_test : Series
        Labels.
    models : dict, optional
        ``{name: estimator}``.  Defaults to :func:`get_models`.
    progress_callback : callable, optional
        ``callback(step, total, model_name)`` for Streamlit progress bars.

    Returns
    -------
    results_df : pd.DataFrame
        Rows = models, columns = metrics (sorted by ROC-AUC desc).
    trained_models : dict
        ``{name: fitted_estimator}``
    """
    if models is None:
        models = get_models()

    # Guard against NaN / Inf in input arrays
    X_train = _sanitize_array(X_train)
    X_test = _sanitize_array(X_test)

    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}
    total = len(models)

    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(idx, total, name)

        logger.info("Training %s...", name)

        try:
            model.fit(X_train, y_train)
            trained_models[name] = model

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            }
            results.append(metrics)
            logger.info(
                "%s — Accuracy: %.4f, ROC-AUC: %.4f",
                name, metrics["Accuracy"], metrics["ROC-AUC"],
            )

        except Exception as e:
            logger.error("Training failed for %s: %s", name, e)

    if progress_callback:
        progress_callback(total, total, "Done")

    results_df = (
        pd.DataFrame(results)
        .set_index("Model")
        .sort_values("ROC-AUC", ascending=False)
    )

    logger.info("Best model: %s (ROC-AUC: %.4f)", results_df.index[0], results_df.iloc[0]["ROC-AUC"])
    return results_df, trained_models


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════

def get_confusion_matrices(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: pd.Series,
) -> Dict[str, np.ndarray]:
    """
    Compute confusion matrices for all trained models.

    Returns
    -------
    dict
        ``{model_name: confusion_matrix_array}``
    """
    return {
        name: confusion_matrix(y_test, model.predict(X_test))
        for name, model in models.items()
    }


def get_feature_importances(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
) -> pd.Series:
    """
    Extract feature importances from a tree-based model.

    Parameters
    ----------
    model : estimator
        A fitted model with ``feature_importances_`` attribute.
    feature_names : list of str
        Feature names matching the model's input.
    top_n : int
        Number of top features to return.

    Returns
    -------
    pd.Series
        Sorted importances (ascending) of the top-N features.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ attribute.")
        return pd.Series(dtype=float)

    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=True).tail(top_n)

    return importances


def _sanitize_array(arr: Any) -> Any:
    """Replace Inf/NaN values in array-like input."""
    if isinstance(arr, pd.DataFrame):
        arr = arr.copy()
        arr.replace([np.inf, -np.inf], np.nan, inplace=True)
        if arr.isnull().values.any():
            arr.fillna(arr.median(), inplace=True)
        if arr.isnull().values.any():
            arr.fillna(0, inplace=True)
    elif isinstance(arr, np.ndarray):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr
