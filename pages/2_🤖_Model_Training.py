"""
Page 2: Model Training — Train, Evaluate & Save Models.
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

from src.data_loader import load_data, remove_duplicates
from src.preprocessing import (
    run_preprocessing_pipeline,
    split_data,
    scale_features,
    handle_missing_values,
)
from src.feature_engineering import engineer_features
from src.model_training import (
    train_evaluate_models,
    get_feature_importances,
    cross_validate_models,
    get_models,
)
from src.model_tuning import tune_best_model, save_model
from src.evaluation import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    generate_classification_reports,
    plot_feature_importance,
)

# ── Page Config & Styles ──────────────────────────────────────────
st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

css_path = os.path.join(root_dir, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🤖 Model Training & Evaluation</h1>", unsafe_allow_html=True)

# ── State Management ─────────────────────────────────────────────
if "df" not in st.session_state:
    try:
        st.session_state.df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

if "training_results" not in st.session_state:
    st.session_state.training_results = None
if "tuned_results" not in st.session_state:
    st.session_state.tuned_results = None
if "cv_results" not in st.session_state:
    st.session_state.cv_results = None

# ── Sidebar Controls ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Training Configuration")
    target_var = st.radio(
        "Select Prediction Target:",
        ["Graduation_Status", "Placement_Status"],
        format_func=lambda x: "🎓 Graduation" if "Graduation" in x else "💼 Placement",
    )

    do_feature_eng = st.checkbox("Apply Feature Engineering", value=True)
    do_cv = st.checkbox("Run Cross-Validation (5-Fold)", value=True)
    do_tuning = st.checkbox(
        "Auto-Tune Best Model", value=False,
        help="Runs RandomizedSearchCV on the best model. Can take a few minutes.",
    )
    do_save_plots = st.checkbox("Save Evaluation Plots", value=True)

    start_training = st.button("🚀 Start Training Pipeline", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
#  Training Pipeline
# ═══════════════════════════════════════════════════════════════════
if start_training:
    with st.spinner("Initializing Pipeline..."):
        df = st.session_state.df.copy()

        # 1. Remove duplicates
        df = remove_duplicates(df)

        # 2. Preprocess
        df_clean, encoders = run_preprocessing_pipeline(df)

        # 3. Feature Engineering
        if do_feature_eng:
            df_clean = engineer_features(df_clean)

        # 4. Split Data
        other_target = [c for c in ["Graduation_Status", "Placement_Status"] if c != target_var and c in df_clean.columns]
        df_for_split = df_clean.drop(columns=other_target, errors="ignore")

        feature_cols = [c for c in df_for_split.columns if c != target_var]
        X = df_for_split[feature_cols]
        y = df_for_split[target_var]

        from sklearn.model_selection import train_test_split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        # 5. Scale
        X_train, scaler = scale_features(X_train_raw, feature_cols)
        X_test, _ = scale_features(X_test_raw, feature_cols, scaler=scaler)

    # 6. Cross-Validation (Optional)
    if do_cv:
        with st.spinner("Running Stratified K-Fold Cross-Validation..."):
            cv_progress = st.progress(0)
            cv_status = st.empty()

            def cv_callback(step, total, name):
                if step < total:
                    cv_progress.progress(step / total)
                    cv_status.text(f"Cross-validating: {name}")
                else:
                    cv_progress.progress(1.0)
                    cv_status.text("Cross-validation complete!")

            cv_results = cross_validate_models(
                X_train, y_train,
                progress_callback=cv_callback,
            )
            st.session_state.cv_results = cv_results

    # 7. Train & Evaluate on Holdout
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(step, total, model_name):
        if step < total:
            progress_bar.progress(step / total)
            status_text.text(f"Training {model_name}...")
        else:
            progress_bar.progress(1.0)
            status_text.text("Training complete!")

    results_df, trained_models = train_evaluate_models(
        X_train, X_test, y_train, y_test,
        progress_callback=update_progress,
    )

    st.session_state.training_results = {
        "target": target_var,
        "results_df": results_df,
        "models": trained_models,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

    # 8. Generate evaluation plots
    if do_save_plots:
        with st.spinner("Generating evaluation plots..."):
            target_label = target_var.replace("_", " ")
            plot_confusion_matrices(trained_models, X_test, y_test, target_label)
            plot_roc_curves(trained_models, X_test, y_test, target_label)
            plot_precision_recall_curves(trained_models, X_test, y_test, target_label)
            generate_classification_reports(trained_models, X_test, y_test, target_label)

            # Feature importance for best tree-based model
            best_name = results_df.index[0]
            if best_name in ["Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"]:
                plot_feature_importance(
                    trained_models[best_name], feature_cols,
                    best_name, target_label,
                )

    # 9. Tuning (Optional)
    if do_tuning:
        with st.spinner("Tuning Best Model..."):
            best_model, best_name, tuned_metrics, best_params = tune_best_model(
                results_df, trained_models, X_train, y_train, X_test, y_test,
            )
            st.session_state.tuned_results = {
                "model": best_model,
                "name": best_name,
                "metrics": tuned_metrics,
                "params": best_params,
            }
    else:
        st.session_state.tuned_results = None

# ═══════════════════════════════════════════════════════════════════
#  Display Results
# ═══════════════════════════════════════════════════════════════════
if st.session_state.training_results is not None:
    res = st.session_state.training_results

    st.markdown(f"### 📊 Results for {res['target'].replace('_', ' ')}")

    # ── Cross-Validation Results ──────────────────────────────────
    if st.session_state.cv_results is not None:
        st.markdown("#### 🔄 Stratified K-Fold Cross-Validation Results")
        cv_df = st.session_state.cv_results

        # Format CV results for display
        display_cols = [c for c in cv_df.columns if "_mean" in c]
        cv_display = cv_df[display_cols].copy()
        cv_display.columns = [c.replace("_mean", "").replace("_", " ").title() for c in display_cols]
        st.dataframe(
            cv_display.style.background_gradient(cmap="Blues"),
            use_container_width=True,
        )
        st.caption("Mean scores across 5-fold stratified cross-validation.")

    st.markdown("---")

    # ── Holdout Results ───────────────────────────────────────────
    st.markdown("#### 🏆 Holdout Test Set Results")
    st.dataframe(
        res["results_df"].style.background_gradient(cmap="Blues"),
        use_container_width=True,
    )

    # Top Model Metrics
    best_model_name = res["results_df"].index[0]
    best_metrics = res["results_df"].iloc[0]

    st.markdown(f"#### 🥇 Best Model: **{best_model_name}**")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC", f"{best_metrics['ROC-AUC']:.4f}")
    m2.metric("Accuracy", f"{best_metrics['Accuracy']:.4f}")
    m3.metric("Precision", f"{best_metrics['Precision']:.4f}")
    m4.metric("F1-Score", f"{best_metrics['F1-Score']:.4f}")
    m5.metric("Recall", f"{best_metrics['Recall']:.4f}")

    # Tuned Results
    if st.session_state.tuned_results is not None:
        tuned = st.session_state.tuned_results
        st.markdown("---")
        st.markdown(f"#### 🛠️ Tuned Model: **{tuned['name']}**")

        if tuned["metrics"]:
            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric(
                "Tuned ROC-AUC", f"{tuned['metrics']['ROC-AUC']:.4f}",
                f"{tuned['metrics']['ROC-AUC'] - best_metrics['ROC-AUC']:.4f}",
            )
            tm2.metric(
                "Tuned Accuracy", f"{tuned['metrics']['Accuracy']:.4f}",
                f"{tuned['metrics']['Accuracy'] - best_metrics['Accuracy']:.4f}",
            )
            tm3.metric("Tuned F1", f"{tuned['metrics']['F1-Score']:.4f}")
            tm4.metric("CV Score", f"{tuned['metrics']['Best_CV_Score']:.4f}")

            with st.expander("View Best Parameters"):
                st.json(tuned["params"])

        final_model_to_save = tuned["model"]
        final_model_name = f"tuned_{tuned['name']}_{res['target']}"
    else:
        final_model_to_save = res["models"][best_model_name]
        final_model_name = f"base_{best_model_name}_{res['target']}"

    # Save Action
    st.markdown("---")
    if st.button("💾 Save Best Model for Predictions"):
        path = save_model(
            final_model_to_save, res["scaler"],
            res["feature_cols"], final_model_name,
        )
        st.success(f"Model saved successfully to `{path}`")

    # ── Visualization Tabs ────────────────────────────────────────
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([
        "📊 Metric Comparison",
        "🌲 Feature Importance",
        "📉 Confusion Matrix",
    ])

    with tab1:
        # Multi-metric bar chart
        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        metrics_long = res["results_df"].reset_index().melt(
            id_vars="Model", value_vars=metrics_to_plot,
            var_name="Metric", value_name="Score",
        )

        fig_metrics = px.bar(
            metrics_long, x="Metric", y="Score", color="Model",
            barmode="group", template="plotly_dark",
            title="Model Comparison — All Metrics",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_metrics.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    with tab2:
        tree_models = ["Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"]
        available_tree = [m for m in tree_models if m in res["models"]]

        if available_tree:
            selected_model = st.selectbox("Select Model for Feature Importance:", available_tree)
            importances = get_feature_importances(
                res["models"][selected_model], res["feature_cols"],
            )

            if len(importances) > 0:
                fig_imp = px.bar(
                    x=importances.values, y=importances.index,
                    orientation="h",
                    title=f"Feature Importances — {selected_model}",
                    template="plotly_dark",
                    color=importances.values,
                    color_continuous_scale="Viridis",
                )
                fig_imp.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not available for the selected models.")

    with tab3:
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix

        cm_model = st.selectbox(
            "Select Model for Confusion Matrix:",
            list(res["models"].keys()),
            key="cm_select",
        )

        if cm_model in res["models"]:
            y_pred = res["models"][cm_model].predict(res["X_test"])
            cm = sk_confusion_matrix(res["y_test"], y_pred)

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                colorscale="Blues",
                text=cm, texttemplate="%{text}",
                textfont={"size": 20},
            ))
            fig_cm.update_layout(
                title=f"Confusion Matrix — {cm_model}",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

else:
    st.info("Configure settings in the sidebar and click **Start Training Pipeline** to begin.")
