"""
Hyperparameter tuning and model persistence.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from config.settings import PARAM_GRIDS, TUNING_N_ITER, TUNING_CV, RANDOM_STATE, MODELS_DIR


def tune_best_model(
    results_df: pd.DataFrame,
    models_dict: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    """
    Tune the top-performing model (by ROC-AUC) using RandomizedSearchCV.

    Returns
    -------
    best_model : estimator
    best_name : str
    tuned_metrics : dict
    best_params : dict
    """
    best_name = results_df.index[0]

    if best_name not in PARAM_GRIDS:
        return models_dict[best_name], best_name, None, None

    param_grid = PARAM_GRIDS[best_name]
    base_model = models_dict[best_name]

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=TUNING_N_ITER,
        cv=TUNING_CV,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    tuned_metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Best_CV_Score": round(search.best_score_, 4),
    }

    return best_model, best_name, tuned_metrics, search.best_params_


# ── Persistence ───────────────────────────────────────────────────

def save_model(model, scaler, feature_cols: list, name: str) -> str:
    """
    Save model, scaler, and feature column list to the ``models/`` directory.

    Returns the path to the saved file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(
        {"model": model, "scaler": scaler, "feature_cols": feature_cols},
        filepath,
    )
    return filepath


def load_model(name: str) -> dict:
    """
    Load a saved model bundle.

    Returns
    -------
    dict with keys: model, scaler, feature_cols
    """
    filepath = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved model found at '{filepath}'")
    return joblib.load(filepath)


def list_saved_models() -> list[str]:
    """Return names of all saved models (without extension)."""
    if not os.path.exists(MODELS_DIR):
        return []
    return [
        f.replace(".joblib", "")
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".joblib")
    ]
