"""
Predict Success Dashboard — Main Entry Point.

Streamlit multi-page application for student graduation and
placement forecasting using machine learning.
"""
import streamlit as st
import os

# ── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predict Success | ML Project",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load Custom CSS ────────────────────────────────────────────────
def load_css() -> None:
    """Load custom CSS styles from the assets directory."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100)
    st.markdown("## Predict Success")
    st.markdown("ML Models for Graduation & Placement Forecasting")
    st.markdown("---")
    st.markdown("### 📑 Navigation")
    st.markdown("""
    - 📊 **Data Explorer** — EDA & Visualization
    - 🤖 **Model Training** — Train & Evaluate
    - 📈 **Model Comparison** — Compare Models
    - 🔍 **Explainability** — SHAP Analysis
    - 🔮 **Predict** — Real-time Predictions
    """)
    st.markdown("---")
    st.markdown("Created by: **Data Science Team**")
    st.markdown("Version: **2.0.0**")

# ── Main Content ──────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>🎓 Predict Success Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""
<div class='custom-card'>
    <h2>Welcome to the Student Success Prediction Platform</h2>
    <p>This industry-level machine learning application helps educational institutions
    identify students who are at risk of not graduating or not securing placement
    after their studies. Built with production-grade ML pipelines, SHAP
    explainability, and interactive visualizations.</p>
</div>
""", unsafe_allow_html=True)

# ── Project Overview Cards ────────────────────────────────────────
st.markdown("### 🚀 Platform Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3>📊 Data Explorer</h3>
        <p>Interactive EDA of student records. Visualize feature distributions,
        outliers, and correlations with dynamic Plotly charts.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h3>🤖 Model Training</h3>
        <p>Train 5 classifiers with Stratified K-Fold CV. Compare
        metrics, tune hyper-parameters, and save the best model.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <h3>📈 Model Comparison</h3>
        <p>Side-by-side comparison of all trained models. Radar charts,
        ROC curves, and cross-validation vs holdout analysis.</p>
    </div>
    """, unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class='metric-card'>
        <h3>🔍 Explainability</h3>
        <p>SHAP-based model explanations. Understand which factors
        most influence graduation and placement predictions.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class='metric-card'>
        <h3>🔮 Predict Outcomes</h3>
        <p>Input new student data to receive instant probability
        scores for graduation and placement outcomes.</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class='metric-card'>
        <h3>📋 Reports</h3>
        <p>Auto-generated evaluation reports, confusion matrices,
        feature importance charts, and SHAP analysis documents.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Key Objectives ────────────────────────────────────────────────
st.markdown("### 🎯 Objectives")
st.markdown("""
- **Predict Graduation Status:** Binary classification (Will Graduate / At Risk)
- **Predict Placement Status:** Binary classification (Will Place / At Risk)
- **Identify Key Factors:** Determine which student attributes influence outcomes the most
- **Provide Actionable Insights:** Enable data-driven interventions for student success
- **Explain Predictions:** Use SHAP to make model decisions interpretable
""")

# ── Dataset Quick Stats ───────────────────────────────────────────
st.markdown("### 📊 Dataset at a Glance")
q1, q2, q3, q4 = st.columns(4)
q1.metric("📁 Records", "1,005")
q2.metric("📐 Features", "15")
q3.metric("🎯 Targets", "2")
q4.metric("🤖 Models", "5")

st.markdown("---")

# ── Architecture ──────────────────────────────────────────────────
st.markdown("### 🏗️ Architecture")
st.markdown("""
```
                    ┌──────────────┐
                    │  Raw Data    │
                    │  (CSV)       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Data Loader │
                    │  & Validator │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  Preprocessing Pipeline │
              │  (Impute → Cap → Encode │
              │   → Engineer → Scale)   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    Train / Val / Test    │
              │    Stratified Split      │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐     ┌──────▼──────┐   ┌─────▼─────┐
    │   5×    │     │  Stratified │   │  Holdout  │
    │ Models  │     │  K-Fold CV  │   │  Eval     │
    └────┬────┘     └──────┬──────┘   └─────┬─────┘
         │                 │                │
         └─────────────────┼────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Evaluation & SHAP     │
              │   Explainability        │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Streamlit Dashboard   │
              │   (6 Interactive Pages) │
              └─────────────────────────┘
```
""")

st.info("👈 Please select a module from the sidebar to begin.")
