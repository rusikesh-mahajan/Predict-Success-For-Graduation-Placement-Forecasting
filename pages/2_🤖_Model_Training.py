import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.data_loader import load_data
from src.preprocessing import run_preprocessing_pipeline, split_data, scale_features
from src.feature_engineering import engineer_features
from src.model_training import train_evaluate_models, get_feature_importances
from src.model_tuning import tune_best_model, save_model

# --- Page Config & Styles ---
st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

css_path = os.path.join(root_dir, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🤖 Model Training & Evaluation</h1>", unsafe_allow_html=True)

# --- State Management ---
if 'df' not in st.session_state:
    try:
        st.session_state.df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'tuned_results' not in st.session_state:
    st.session_state.tuned_results = None

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Training Configuration")
    target_var = st.radio(
        "Select Prediction Target:",
        ["Graduation_Status", "Placement_Status"],
        format_func=lambda x: "🎓 Graduation" if "Graduation" in x else "💼 Placement"
    )
    
    do_feature_eng = st.checkbox("Apply Feature Engineering", value=True)
    do_tuning = st.checkbox("Auto-Tune Best Model", value=False, help="Runs RandomizedSearchCV on the best model. Can take a few minutes.")
    
    start_training = st.button("🚀 Start Training Pipeline", use_container_width=True)

# --- Training Pipeline ---
if start_training:
    with st.spinner("Initializing Pipeline..."):
        df = st.session_state.df.copy()
        
        # 1. Preprocess
        df_clean, encoders = run_preprocessing_pipeline(df)
        
        # 2. Feature Engineering
        if do_feature_eng:
            df_clean = engineer_features(df_clean)
            
        # 3. Split Data
        X_train_raw, X_test_raw, y_train, y_test = split_data(df_clean, target_var)
        feature_cols = X_train_raw.columns.tolist()
        
        # 4. Scale
        X_train, scaler = scale_features(X_train_raw, feature_cols)
        X_test, _ = scale_features(X_test_raw, feature_cols, scaler=scaler)
        
        # 5. Train & Evaluate
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(step, total, model_name):
            if step < total:
                progress_bar.progress((step) / total)
                status_text.text(f"Training {model_name}...")
            else:
                progress_bar.progress(1.0)
                status_text.text("Training complete!")
                
        results_df, trained_models = train_evaluate_models(
            X_train, X_test, y_train, y_test, progress_callback=update_progress
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
            "y_test": y_test
        }
        
        # 6. Tuning (Optional)
        if do_tuning:
            with st.spinner("Tuning Best Model..."):
                best_model, best_name, tuned_metrics, best_params = tune_best_model(
                    results_df, trained_models, X_train, y_train, X_test, y_test
                )
                st.session_state.tuned_results = {
                    "model": best_model,
                    "name": best_name,
                    "metrics": tuned_metrics,
                    "params": best_params
                }
        else:
            st.session_state.tuned_results = None

# --- Display Results ---
if st.session_state.training_results is not None:
    res = st.session_state.training_results
    
    st.markdown(f"### Results for {res['target'].replace('_', ' ')}")
    
    # Leaderboard
    st.dataframe(res["results_df"].style.background_gradient(cmap="Blues"), use_container_width=True)
    
    # Top Model Metrics
    best_model_name = res["results_df"].index[0]
    best_metrics = res["results_df"].iloc[0]
    
    st.markdown(f"#### 🏆 Best Base Model: **{best_model_name}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{best_metrics['ROC-AUC']:.4f}")
    m2.metric("Accuracy", f"{best_metrics['Accuracy']:.4f}")
    m3.metric("F1-Score", f"{best_metrics['F1-Score']:.4f}")
    m4.metric("Recall", f"{best_metrics['Recall']:.4f}")
    
    # Tuned Results
    if st.session_state.tuned_results is not None:
        tuned = st.session_state.tuned_results
        st.markdown("---")
        st.markdown(f"#### 🛠️ Tuned Model: **{tuned['name']}**")
        tm1, tm2, tm3, tm4 = st.columns(4)
        tm1.metric("Tuned ROC-AUC", f"{tuned['metrics']['ROC-AUC']:.4f}", 
                   f"{tuned['metrics']['ROC-AUC'] - best_metrics['ROC-AUC']:.4f}")
        tm2.metric("Tuned Accuracy", f"{tuned['metrics']['Accuracy']:.4f}",
                   f"{tuned['metrics']['Accuracy'] - best_metrics['Accuracy']:.4f}")
        tm3.metric("Tuned F1", f"{tuned['metrics']['F1-Score']:.4f}")
        
        with st.expander("View Best Parameters"):
            st.json(tuned['params'])
            
        final_model_to_save = tuned["model"]
        final_model_name = f"tuned_{tuned['name']}_{res['target']}"
    else:
        final_model_to_save = res["models"][best_model_name]
        final_model_name = f"base_{best_model_name}_{res['target']}"
    
    # Save Action
    st.markdown("---")
    if st.button("💾 Save Best Model for Predictions"):
        path = save_model(final_model_to_save, res["scaler"], res["feature_cols"], final_model_name)
        st.success(f"Model saved successfully to `{path}`")

    # Visualizations
    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 Metric Comparison", "🌲 Feature Importance"])
    
    with tab1:
        fig_acc = px.bar(
            res["results_df"].reset_index(), 
            y="Model", x="Accuracy", 
            orientation='h',
            title="Model Accuracy Comparison",
            template="plotly_dark",
            color="Accuracy",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_acc, use_container_width=True)
        
    with tab2:
        if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"]:
            importances = get_feature_importances(res["models"][best_model_name], res["feature_cols"])
            fig_imp = px.bar(
                x=importances.values, y=importances.index,
                orientation='h',
                title=f"Feature Importances ({best_model_name})",
                template="plotly_dark",
                color=importances.values,
                color_continuous_scale="Mint"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(f"Feature importance not available for {best_model_name}")
else:
    st.info("Configure settings in the sidebar and click **Start Training Pipeline** to begin.")
