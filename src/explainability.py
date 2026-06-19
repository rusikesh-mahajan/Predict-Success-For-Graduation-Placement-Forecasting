"""
Explainable AI using SHAP (SHapley Additive exPlanations).

Generates SHAP-based explanations for model predictions, including
summary plots, feature importance rankings, and human-readable
reports identifying the most influential factors.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import FIGURES_DIR, REPORTS_DIR
from src.utils import setup_logging

logger = setup_logging(__name__)

# ── Optional SHAP import ──────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE: bool = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


def ensure_figures_dir() -> str:
    """Create figures directory if needed."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return FIGURES_DIR


def _get_shap_explainer(
    model: Any,
    X_background: np.ndarray,
    model_name: str = "",
) -> Any:
    """
    Select and initialise the appropriate SHAP explainer.

    Parameters
    ----------
    model : estimator
        Fitted model.
    X_background : array-like
        Background data for the explainer (typically a sample
        of the training set).
    model_name : str
        Name of the model (for logging).

    Returns
    -------
    shap.Explainer
    """
    tree_models = ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]

    if model_name in tree_models or hasattr(model, "feature_importances_"):
        logger.info("Using TreeExplainer for '%s'.", model_name)
        return shap.TreeExplainer(model)
    else:
        logger.info("Using LinearExplainer for '%s'.", model_name)
        return shap.LinearExplainer(model, X_background)


def compute_shap_values(
    model: Any,
    X_data: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    max_samples: int = 200,
) -> Optional[Any]:
    """
    Compute SHAP values for the given data.

    Parameters
    ----------
    model : estimator
        Fitted model.
    X_data : array-like
        Data to explain (usually test set).
    feature_names : list of str
        Feature names.
    model_name : str
        Model name for logging.
    max_samples : int
        Maximum number of samples to explain.

    Returns
    -------
    shap.Explanation or None
        SHAP values, or None if SHAP is unavailable.
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP is not installed.")
        return None

    logger.info("Computing SHAP values for %s (%d samples)...", model_name, min(len(X_data), max_samples))

    try:
        # Subsample if too large
        if isinstance(X_data, pd.DataFrame):
            X_sample = X_data.iloc[:max_samples]
        elif isinstance(X_data, np.ndarray):
            X_sample = X_data[:max_samples]
        else:
            X_sample = X_data[:max_samples]

        explainer = _get_shap_explainer(model, X_sample, model_name)
        shap_values = explainer(X_sample)

        # Handle multi-output case
        if isinstance(shap_values.values, list):
            shap_values = shap_values[..., 1]

        # Ensure feature names are set
        if shap_values.feature_names is None:
            shap_values.feature_names = feature_names

        logger.info("SHAP values computed successfully.")
        return shap_values

    except Exception as e:
        logger.error("SHAP computation failed for %s: %s", model_name, e)
        return None


def plot_shap_summary(
    shap_values: Any,
    target_name: str = "Target",
    model_name: str = "Model",
    save: bool = True,
) -> Optional[plt.Figure]:
    """
    Generate SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values.
    target_name : str
        Target variable name.
    model_name : str
        Model name for title.
    save : bool
        Whether to save the figure.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if shap_values is None or not SHAP_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, show=False)
    plt.title(f"SHAP Summary — {model_name} ({target_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(
            ensure_figures_dir(),
            f"shap_summary_{model_name.lower().replace(' ', '_')}_{target_name.lower().replace(' ', '_')}.png"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot saved to '%s'", path)

    fig = plt.gcf()
    plt.close("all")
    return fig


def plot_shap_importance(
    shap_values: Any,
    target_name: str = "Target",
    model_name: str = "Model",
    top_n: int = 15,
    save: bool = True,
) -> Optional[plt.Figure]:
    """
    Generate SHAP feature importance bar plot.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values.
    target_name : str
        Target variable name.
    model_name : str
        Model name.
    top_n : int
        Number of top features.
    save : bool
        Whether to save.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if shap_values is None or not SHAP_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=top_n, show=False)
    plt.title(f"SHAP Feature Importance — {model_name} ({target_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(
            ensure_figures_dir(),
            f"shap_importance_{model_name.lower().replace(' ', '_')}_{target_name.lower().replace(' ', '_')}.png"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("SHAP importance plot saved to '%s'", path)

    fig = plt.gcf()
    plt.close("all")
    return fig


def get_top_features(
    shap_values: Any,
    top_n: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Extract the top-N most influential features based on SHAP values.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values.
    top_n : int
        Number of top features to return.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ``Feature``, ``Mean_Abs_SHAP``, ``Rank``.
    """
    if shap_values is None or not SHAP_AVAILABLE:
        return None

    try:
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        feature_names = shap_values.feature_names

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(mean_abs))]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean_Abs_SHAP": mean_abs,
        }).sort_values("Mean_Abs_SHAP", ascending=False).head(top_n)

        importance_df["Rank"] = range(1, len(importance_df) + 1)
        importance_df = importance_df.reset_index(drop=True)

        return importance_df

    except Exception as e:
        logger.error("Failed to extract top features: %s", e)
        return None


def generate_shap_report(
    shap_values_grad: Any = None,
    shap_values_place: Any = None,
    model_name: str = "Model",
) -> str:
    """
    Generate a markdown report explaining SHAP-based feature influence.

    Parameters
    ----------
    shap_values_grad : shap.Explanation, optional
        SHAP values for graduation prediction.
    shap_values_place : shap.Explanation, optional
        SHAP values for placement prediction.
    model_name : str
        Name of the model used.

    Returns
    -------
    str
        Path to the generated markdown report.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "shap_report.md")

    lines = [
        "# 🔍 SHAP Explainability Report",
        "",
        f"**Model:** {model_name}",
        "",
        "This report identifies the most influential factors affecting",
        "student graduation and placement predictions using SHAP",
        "(SHapley Additive exPlanations).",
        "",
        "---",
        "",
    ]

    for label, sv in [
        ("Graduation", shap_values_grad),
        ("Placement", shap_values_place),
    ]:
        lines.append(f"## {label} Prediction")
        lines.append("")

        top_features = get_top_features(sv)
        if top_features is not None:
            lines.append("### Top 10 Most Influential Features")
            lines.append("")
            lines.append("| Rank | Feature | Mean |SHAP| Impact |")
            lines.append("|------|---------|------|")

            for _, row in top_features.iterrows():
                lines.append(
                    f"| {int(row['Rank'])} | {row['Feature']} | {row['Mean_Abs_SHAP']:.4f} |"
                )

            lines.append("")
            lines.append(f"### Key Insights — {label}")
            lines.append("")

            top3 = top_features.head(3)["Feature"].tolist()
            lines.append(f"The top 3 most influential features for {label.lower()} prediction are:")
            lines.append("")
            for i, feat in enumerate(top3, 1):
                lines.append(f"{i}. **{feat}**")
            lines.append("")
        else:
            lines.append("*SHAP analysis not available.*")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.extend([
        "## Methodology",
        "",
        "- **SHAP** uses Shapley values from cooperative game theory to assign",
        "  each feature a contribution to every prediction.",
        "- **TreeExplainer** is used for tree-based models (exact computation).",
        "- **LinearExplainer** is used for linear models.",
        "- Mean absolute SHAP value represents average feature impact.",
        "",
        "---",
        "",
        "*Report generated automatically by the Predict Success ML pipeline.*",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("SHAP report saved to '%s'", report_path)
    return report_path
