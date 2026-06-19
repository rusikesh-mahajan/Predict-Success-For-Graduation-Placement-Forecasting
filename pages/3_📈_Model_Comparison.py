"""
Page 3: Model Comparison — Side-by-side model analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.model_training import get_feature_importances

# ── Page Config & Styles ──────────────────────────────────────────
st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

css_path = os.path.join(root_dir, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📈 Model Comparison</h1>", unsafe_allow_html=True)

# ── Check for training results ───────────────────────────────────
if "training_results" not in st.session_state or st.session_state.training_results is None:
    st.warning("⚠️ No training results found. Please go to **Model Training** and train models first.")
    st.stop()

res = st.session_state.training_results
results_df = res["results_df"]

st.markdown(f"### Comparing models for: **{res['target'].replace('_', ' ')}**")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Metric Comparison",
    "🕸️ Radar Chart",
    "📈 ROC & PR Curves",
    "🌲 Feature Importance",
])

# ── Tab 1: Metric Comparison ─────────────────────────────────────
with tab1:
    st.markdown("### All Models — Holdout Metrics")
    st.dataframe(
        results_df.style.background_gradient(cmap="Blues")
        .format("{:.4f}"),
        use_container_width=True,
    )

    # Cross-validation vs holdout comparison
    if "cv_results" in st.session_state and st.session_state.cv_results is not None:
        st.markdown("---")
        st.markdown("### Cross-Validation vs Holdout Comparison")

        cv_df = st.session_state.cv_results

        comparison_data = []
        for model_name in results_df.index:
            if model_name in cv_df.index:
                comparison_data.append({
                    "Model": model_name,
                    "CV Accuracy": cv_df.loc[model_name].get("accuracy_mean", 0),
                    "Holdout Accuracy": results_df.loc[model_name, "Accuracy"],
                    "CV ROC-AUC": cv_df.loc[model_name].get("roc_auc_mean", 0),
                    "Holdout ROC-AUC": results_df.loc[model_name, "ROC-AUC"],
                })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data).set_index("Model")
            st.dataframe(
                comp_df.style.background_gradient(cmap="Greens").format("{:.4f}"),
                use_container_width=True,
            )

    # Grouped bar chart
    st.markdown("### Visual Comparison")
    metrics_to_plot = st.multiselect(
        "Select Metrics to Compare:",
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        default=["Accuracy", "F1-Score", "ROC-AUC"],
    )

    if metrics_to_plot:
        metrics_long = results_df.reset_index().melt(
            id_vars="Model", value_vars=metrics_to_plot,
            var_name="Metric", value_name="Score",
        )

        fig = px.bar(
            metrics_long, x="Model", y="Score", color="Metric",
            barmode="group", template="plotly_dark",
            title="Model Comparison",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Radar Chart ──────────────────────────────────────────
with tab2:
    st.markdown("### Radar Chart — Model Performance Profile")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    fig_radar = go.Figure()

    colors = px.colors.qualitative.Set2
    for idx, (model_name, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        values.append(values[0])  # Close the polygon

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=model_name,
            line_color=colors[idx % len(colors)],
            opacity=0.7,
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor="rgba(0,0,0,0)",
        ),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        title="Model Performance Radar",
        height=600,
    )

    st.plotly_chart(fig_radar, use_container_width=True)

# ── Tab 3: ROC & PR Curves ──────────────────────────────────────
with tab3:
    st.markdown("### ROC Curves")

    from sklearn.metrics import roc_curve, precision_recall_curve

    # ROC curves
    fig_roc = go.Figure()
    colors = px.colors.qualitative.Set2

    for idx, (name, model) in enumerate(res["models"].items()):
        try:
            y_prob = model.predict_proba(res["X_test"])[:, 1]
            fpr, tpr, _ = roc_curve(res["y_test"], y_prob)
            auc_val = results_df.loc[name, "ROC-AUC"]

            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} (AUC={auc_val:.4f})",
                line=dict(color=colors[idx % len(colors)], width=2),
            ))
        except Exception:
            pass

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random",
        line=dict(color="gray", dash="dash"),
    ))

    fig_roc.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Precision-Recall curves
    st.markdown("### Precision-Recall Curves")
    fig_pr = go.Figure()

    for idx, (name, model) in enumerate(res["models"].items()):
        try:
            y_prob = model.predict_proba(res["X_test"])[:, 1]
            precision, recall, _ = precision_recall_curve(res["y_test"], y_prob)

            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision,
                name=name,
                line=dict(color=colors[idx % len(colors)], width=2),
            ))
        except Exception:
            pass

    fig_pr.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
    )
    st.plotly_chart(fig_pr, use_container_width=True)

# ── Tab 4: Feature Importance Comparison ─────────────────────────
with tab4:
    st.markdown("### Feature Importance Comparison")

    tree_models = ["Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"]
    available = [m for m in tree_models if m in res["models"]]

    if available:
        selected_models = st.multiselect(
            "Select models to compare:", available, default=available[:2],
        )

        if selected_models:
            imp_data = []
            for name in selected_models:
                importances = get_feature_importances(
                    res["models"][name], res["feature_cols"], top_n=10,
                )
                for feat, val in importances.items():
                    imp_data.append({"Model": name, "Feature": feat, "Importance": val})

            if imp_data:
                imp_df = pd.DataFrame(imp_data)

                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature", color="Model",
                    barmode="group", orientation="h",
                    template="plotly_dark",
                    title="Feature Importance Comparison",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_imp.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=600,
                )
                st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance comparison requires tree-based models.")
