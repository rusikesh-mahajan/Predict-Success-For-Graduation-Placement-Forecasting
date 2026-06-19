"""
Hyperparameter tuning, model persistence, and pipeline management.

Handles RandomizedSearchCV tuning of the best-performing model,
saving/loading complete model bundles (model + preprocessor +
metadata), and listing saved models.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

from config.settings import (
    MODELS_DIR,
    PARAM_GRIDS,
    RANDOM_STATE,
    TUNING_CV,
    TUNING_N_ITER,
)
from src.utils import setup_logging

logger = setup_logging(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════

def tune_best_model(
    results_df: pd.DataFrame,
    models_dict: Dict[str, Any],
    X_train: Any,
    y_train: pd.Series,
    X_test: Any,
    y_test: pd.Series,
) -> Tuple[Any, str, Optional[Dict], Optional[Dict]]:
    """
    Tune the top-performing model using RandomizedSearchCV.

    The best model is identified from ``results_df`` (first row,
    sorted by ROC-AUC).  If a param grid is defined in config,
    RandomizedSearchCV is run; otherwise the base model is returned.

    Parameters
    ----------
    results_df : pd.DataFrame
        Model comparison results, sorted by ROC-AUC descending.
    models_dict : dict
        ``{name: fitted_estimator}``
    X_train, y_train : array-like
        Training data.
    X_test, y_test : array-like
        Test data for post-tuning evaluation.

    Returns
    -------
    best_model : estimator
        The tuned (or original) best estimator.
    best_name : str
        Name of the best model.
    tuned_metrics : dict or None
        Post-tuning evaluation metrics.
    best_params : dict or None
        Best hyper-parameters found.
    """
    best_name: str = results_df.index[0]
    logger.info("Tuning best model: %s", best_name)

    if best_name not in PARAM_GRIDS:
        logger.warning("No param grid defined for '%s'. Returning base model.", best_name)
        return models_dict[best_name], best_name, None, None

    param_grid = PARAM_GRIDS[best_name]
    base_model = models_dict[best_name]

    # Strip 'classifier__' prefix for standalone model tuning
    clean_grid = {}
    for key, values in param_grid.items():
        clean_key = key.replace("classifier__", "")
        clean_grid[clean_key] = values

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=clean_grid,
        n_iter=TUNING_N_ITER,
        cv=TUNING_CV,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    logger.info("Running RandomizedSearchCV (n_iter=%d, cv=%d)...", TUNING_N_ITER, TUNING_CV)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    tuned_metrics: Dict[str, float] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Best_CV_Score": round(search.best_score_, 4),
    }

    logger.info(
        "Tuning complete — ROC-AUC: %.4f (CV: %.4f)",
        tuned_metrics["ROC-AUC"], tuned_metrics["Best_CV_Score"],
    )

    return best_model, best_name, tuned_metrics, search.best_params_


# ═══════════════════════════════════════════════════════════════════
#  Model Persistence
# ═══════════════════════════════════════════════════════════════════

def save_model(
    model: Any,
    scaler: Any,
    feature_cols: List[str],
    name: str,
    preprocessor: Any = None,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Save model bundle (model, scaler, feature columns, and metadata).

    Parameters
    ----------
    model : estimator
        Trained model.
    scaler : transformer
        Fitted scaler.
    feature_cols : list of str
        Feature names the model expects.
    name : str
        Model bundle name (used as filename).
    preprocessor : Pipeline, optional
        The full preprocessing pipeline.
    metadata : dict, optional
        Additional metadata (e.g. training metrics).

    Returns
    -------
    str
        Path to the saved ``.joblib`` file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, f"{name}.joblib")

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "preprocessor": preprocessor,
        "metadata": metadata or {},
    }

    joblib.dump(bundle, filepath)
    logger.info("Model saved to '%s'", filepath)

    return filepath


def load_model(name: str) -> Dict[str, Any]:
    """
    Load a saved model bundle.

    Parameters
    ----------
    name : str
        Model name (without ``.joblib`` extension).

    Returns
    -------
    dict
        Keys: ``model``, ``scaler``, ``feature_cols``,
        ``preprocessor``, ``metadata``.

    Raises
    ------
    FileNotFoundError
        If no saved model is found.
    """
    filepath = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved model found at '{filepath}'")

    bundle = joblib.load(filepath)
    logger.info("Model loaded from '%s'", filepath)

    return bundle


def list_saved_models() -> List[str]:
    """
    List names of all saved models (without file extension).

    Returns
    -------
    list of str
    """
    if not os.path.exists(MODELS_DIR):
        return []

    models = [
        f.replace(".joblib", "")
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".joblib")
    ]

    logger.debug("Found %d saved models.", len(models))
    return models
