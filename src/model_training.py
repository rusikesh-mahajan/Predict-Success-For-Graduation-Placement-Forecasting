"""
Model training and evaluation utilities.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from config.settings import RANDOM_STATE

# ── Optional XGBoost ──────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_models() -> dict:
    """Return an ordered dict of model name → untrained estimator."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=10
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE, learning_rate=0.1
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

    return models


def train_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: dict = None,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Train every model and compute evaluation metrics on the test set.

    Parameters
    ----------
    progress_callback : callable, optional
        ``callback(step, total, model_name)`` – for Streamlit progress bars.

    Returns
    -------
    results_df : pd.DataFrame
        Rows = models, columns = metrics (sorted by ROC-AUC descending).
    trained_models : dict
        Mapping model name → fitted estimator.
    """
    if models is None:
        models = get_models()

    # ── Guard: impute any residual NaN / Inf values ────────────────
    # Feature engineering or scaling may introduce NaN (e.g. division
    # by zero, missing source columns).  Fill with column medians so
    # that estimators like LogisticRegression don't raise ValueError.
    for _df in (X_train, X_test):
        _df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if _df.isnull().values.any():
            _df.fillna(_df.median(), inplace=True)
        # Last-resort: if an entire column was NaN, median is NaN too
        if _df.isnull().values.any():
            _df.fillna(0, inplace=True)

    results = []
    trained_models = {}
    total = len(models)

    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(idx, total, name)

        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            }
        )

    if progress_callback:
        progress_callback(total, total, "Done")

    results_df = (
        pd.DataFrame(results)
        .set_index("Model")
        .sort_values("ROC-AUC", ascending=False)
    )
    return results_df, trained_models


def get_confusion_matrices(
    models: dict, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Return {model_name: confusion_matrix_array}."""
    return {
        name: confusion_matrix(y_test, model.predict(X_test))
        for name, model in models.items()
    }


def get_feature_importances(
    model, feature_names: list, top_n: int = 15
) -> pd.Series:
    """
    Extract feature importances from a tree-based model.

    Returns a sorted Series (ascending) of the top-N features.
    """
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=True).tail(top_n)
    return importances
