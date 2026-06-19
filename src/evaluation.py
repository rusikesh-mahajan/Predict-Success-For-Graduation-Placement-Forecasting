"""
Model evaluation and visualization.

Generates publication-quality evaluation plots and saves them to
``reports/figures/``.  Supports confusion matrices, ROC curves,
Precision-Recall curves, classification reports, and feature
importance charts.
"""
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    roc_auc_score,
)

from config.settings import FIGURES_DIR
from src.utils import setup_logging

logger = setup_logging(__name__)

# ── Plot Style ────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def ensure_figures_dir() -> str:
    """Create the figures directory if it doesn't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return FIGURES_DIR


# ═══════════════════════════════════════════════════════════════════
#  Confusion Matrix
# ═══════════════════════════════════════════════════════════════════

def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: pd.Series,
    target_name: str = "Target",
    save: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrices for all models in a grid.

    Parameters
    ----------
    models : dict
        ``{name: fitted_estimator}``
    X_test : array-like
        Test features.
    y_test : Series
        Test labels.
    target_name : str
        Name of the target variable (for title).
    save : bool
        Whether to save the figure to disk.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            ax=axes[idx],
            cmap="Blues",
            display_labels=["Negative", "Positive"],
        )
        axes[idx].set_title(f"{name}", fontsize=12, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Confusion Matrices — {target_name}", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        path = os.path.join(ensure_figures_dir(), f"confusion_matrices_{target_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrices saved to '%s'", path)

    return fig


# ═══════════════════════════════════════════════════════════════════
#  ROC Curve
# ═══════════════════════════════════════════════════════════════════

def plot_roc_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: pd.Series,
    target_name: str = "Target",
    save: bool = True,
) -> plt.Figure:
    """
    Plot ROC curves for all models overlaid on a single plot.

    Parameters
    ----------
    models : dict
        ``{name: fitted_estimator}``
    X_test : array-like
        Test features.
    y_test : Series
        Test labels.
    target_name : str
        Name of the target variable (for title).
    save : bool
        Whether to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in models.items():
        try:
            RocCurveDisplay.from_estimator(
                model, X_test, y_test,
                ax=ax, name=name,
            )
        except Exception as e:
            logger.warning("ROC curve failed for %s: %s", name, e)

    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    ax.set_title(f"ROC Curves — {target_name}", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    fig.tight_layout()

    if save:
        path = os.path.join(ensure_figures_dir(), f"roc_curves_{target_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("ROC curves saved to '%s'", path)

    return fig


# ═══════════════════════════════════════════════════════════════════
#  Precision-Recall Curve
# ═══════════════════════════════════════════════════════════════════

def plot_precision_recall_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: pd.Series,
    target_name: str = "Target",
    save: bool = True,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for all models.

    Parameters
    ----------
    models : dict
        ``{name: fitted_estimator}``
    X_test, y_test : array-like
        Test data.
    target_name : str
        For the title.
    save : bool
        Whether to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in models.items():
        try:
            PrecisionRecallDisplay.from_estimator(
                model, X_test, y_test,
                ax=ax, name=name,
            )
        except Exception as e:
            logger.warning("PR curve failed for %s: %s", name, e)

    ax.set_title(f"Precision-Recall Curves — {target_name}", fontsize=16, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    fig.tight_layout()

    if save:
        path = os.path.join(ensure_figures_dir(), f"pr_curves_{target_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Precision-Recall curves saved to '%s'", path)

    return fig


# ═══════════════════════════════════════════════════════════════════
#  Classification Report
# ═══════════════════════════════════════════════════════════════════

def generate_classification_reports(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: pd.Series,
    target_name: str = "Target",
    save: bool = True,
) -> Dict[str, str]:
    """
    Generate text classification reports for all models.

    Parameters
    ----------
    models : dict
        ``{name: fitted_estimator}``
    X_test, y_test : array-like
        Test data.
    target_name : str
        Target variable name.
    save : bool
        Whether to save reports to disk.

    Returns
    -------
    dict
        ``{model_name: report_string}``
    """
    reports: Dict[str, str] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=["Negative (0)", "Positive (1)"],
            zero_division=0,
        )
        reports[name] = report
        logger.info("Classification report for %s:\n%s", name, report)

    if save:
        path = os.path.join(ensure_figures_dir(), f"classification_reports_{target_name.lower().replace(' ', '_')}.txt")
        with open(path, "w") as f:
            for name, report in reports.items():
                f.write(f"{'=' * 60}\n")
                f.write(f" {name}\n")
                f.write(f"{'=' * 60}\n")
                f.write(report)
                f.write("\n\n")
        logger.info("Classification reports saved to '%s'", path)

    return reports


# ═══════════════════════════════════════════════════════════════════
#  Feature Importance
# ═══════════════════════════════════════════════════════════════════

def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Model",
    target_name: str = "Target",
    top_n: int = 15,
    save: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot feature importances for a tree-based model.

    Parameters
    ----------
    model : estimator
        Must have ``feature_importances_`` attribute.
    feature_names : list of str
        Feature names.
    model_name : str
        Model name for title.
    target_name : str
        Target variable name.
    top_n : int
        Number of top features to show.
    save : bool
        Whether to save the figure.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model '%s' does not support feature importances.", model_name)
        return None

    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    importances.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(importances)))
    ax.set_title(f"Feature Importances — {model_name} ({target_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("")
    fig.tight_layout()

    if save:
        path = os.path.join(
            ensure_figures_dir(),
            f"feature_importance_{model_name.lower().replace(' ', '_')}_{target_name.lower().replace(' ', '_')}.png"
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance plot saved to '%s'", path)

    return fig


# ═══════════════════════════════════════════════════════════════════
#  Generate All Plots
# ═══════════════════════════════════════════════════════════════════

def generate_all_evaluation_plots(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: List[str],
    target_name: str = "Target",
) -> Dict[str, Any]:
    """
    Generate all evaluation plots and save them to disk.

    Parameters
    ----------
    models : dict
        ``{name: fitted_estimator}``
    X_test, y_test : array-like
        Test data.
    feature_names : list of str
        Feature names.
    target_name : str
        Target variable name.

    Returns
    -------
    dict
        ``{plot_type: figure}``
    """
    logger.info("Generating all evaluation plots for '%s'...", target_name)

    plots: Dict[str, Any] = {}

    # Confusion matrices
    plots["confusion_matrices"] = plot_confusion_matrices(
        models, X_test, y_test, target_name,
    )

    # ROC curves
    plots["roc_curves"] = plot_roc_curves(
        models, X_test, y_test, target_name,
    )

    # Precision-Recall curves
    plots["pr_curves"] = plot_precision_recall_curves(
        models, X_test, y_test, target_name,
    )

    # Classification reports
    plots["classification_reports"] = generate_classification_reports(
        models, X_test, y_test, target_name,
    )

    # Feature importance for best tree-based model
    tree_models = ["Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"]
    for name in tree_models:
        if name in models:
            plots[f"feature_importance_{name}"] = plot_feature_importance(
                models[name], feature_names, name, target_name,
            )

    plt.close("all")  # Free memory
    logger.info("All evaluation plots generated successfully.")

    return plots
