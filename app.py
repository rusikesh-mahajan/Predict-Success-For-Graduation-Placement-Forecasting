import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Predict Success | ML Project",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100)
    st.markdown("## Predict Success")
    st.markdown("ML Models for Graduation & Placement Forecasting")
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("Use the sidebar links to navigate through the app.")
    st.markdown("---")
    st.markdown("Created by: **Data Science Team**")
    st.markdown("Version: **1.0.0**")

# --- Main Content ---
st.markdown("<h1 class='main-header'>🎓 Predict Success Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""
<div class='custom-card'>
    <h2>Welcome to the Student Success Prediction Platform</h2>
    <p>This industry-level machine learning application helps educational institutions identify students who are at risk of not graduating or not securing placement after their studies.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3>1. Data Explorer</h3>
        <p>Interactive Exploratory Data Analysis (EDA) of student records. Visualize feature distributions, outliers, and correlations.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h3>2. Model Training</h3>
        <p>Train and evaluate multiple classification models. Compare performance metrics and explore feature importance.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <h3>3. Real-time Prediction</h3>
        <p>Input new student data to receive instant probability scores for graduation and placement outcomes.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🎯 Objectives")
st.markdown("""
- **Predict Graduation Status:** Binary classification (Will Graduate / At Risk)
- **Predict Placement Status:** Binary classification (Will Place / At Risk)
- **Identify Key Factors:** Determine which student attributes influence outcomes the most
- **Provide Actionable Insights:** Enable data-driven interventions for student success
""")

st.info("👈 Please select a module from the sidebar to begin.")
