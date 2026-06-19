"""
Page 1: Data Explorer — Interactive EDA & Dataset Overview.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys

# Add root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.data_loader import load_data, get_data_summary
from src.feature_engineering import engineer_features

# ── Page Config & Styles ──────────────────────────────────────────
st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

css_path = os.path.join(root_dir, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📊 Data Explorer</h1>", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────
@st.cache_data
def get_data():
    """Load and cache the student dataset."""
    try:
        return load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


df = get_data()
if df is None:
    st.stop()

summary = get_data_summary(df)

# ══════════════════════════════════════════════════════════════════
#  Dataset Overview Section
# ══════════════════════════════════════════════════════════════════
st.markdown("### 📋 Dataset Overview")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("📁 Total Records", f"{summary['n_rows']:,}")
m2.metric("📐 Total Features", summary['n_cols'])
m3.metric("⚠️ Missing Values", summary['missing_total'])
m4.metric("🔄 Duplicates", summary['n_duplicates'])
m5.metric("💾 Memory", f"{summary['memory_mb']} MB")

# Dataset info card
with st.expander("📖 Dataset Description", expanded=False):
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | **Source** | Synthetic (generated for research) |
    | **Records** | ~1,005 student records |
    | **Features** | 13 input features |
    | **Targets** | 2 binary targets (Graduation & Placement) |
    | **Missing Values** | ~2% in selected columns (realistic) |
    | **Duplicates** | 5 intentional duplicate rows |
    """)

    st.markdown("**Feature Descriptions:**")
    feature_info = {
        "Student_ID": "Unique student identifier",
        "Gender": "Male / Female",
        "Age": "Student age (18–26)",
        "Attendance_Percentage": "Class attendance (20–100%)",
        "CGPA": "Cumulative GPA (2.0–10.0)",
        "Internal_Marks": "Internal assessment (10–100)",
        "External_Marks": "External exam marks (5–100)",
        "Backlogs": "Number of backlogs (0–5)",
        "Study_Hours_Per_Week": "Weekly study hours (1–40)",
        "Internship_Experience": "Has internship (0/1)",
        "Projects_Completed": "Number of projects (0–5)",
        "Communication_Skills_Score": "Communication score (1–10)",
        "Aptitude_Test_Score": "Aptitude test (10–100)",
        "Graduation_Status": "🎯 Target: Graduated (1) / Not (0)",
        "Placement_Status": "🎯 Target: Placed (1) / Not (0)",
    }
    st.dataframe(
        pd.DataFrame(feature_info.items(), columns=["Feature", "Description"]),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
#  Tabs: Preview, Distributions, Correlations, Statistics
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data Preview",
    "📈 Distributions",
    "🔗 Correlations",
    "📊 Statistics",
])

# ── Tab 1: Data Preview ──────────────────────────────────────────
with tab1:
    st.markdown("### Raw Data Preview")

    with st.expander("🔍 Filter Data"):
        col1, col2 = st.columns(2)
        with col1:
            if "Gender" in df.columns:
                genders = st.multiselect(
                    "Select Gender", df["Gender"].unique(),
                    default=df["Gender"].unique(),
                )
            else:
                genders = []
        with col2:
            if "Graduation_Status" in df.columns:
                grad_status = st.multiselect(
                    "Graduation Status", [0, 1], default=[0, 1],
                )
            else:
                grad_status = []

    filtered_df = df.copy()
    if genders and "Gender" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Gender"].isin(genders)]
    if grad_status and "Graduation_Status" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Graduation_Status"].isin(grad_status)]

    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.caption(f"Showing {min(100, len(filtered_df))} of {len(filtered_df)} filtered records.")

# ── Tab 2: Distributions ─────────────────────────────────────────
with tab2:
    st.markdown("### Feature Distributions")
    col1, col2 = st.columns([1, 3])

    with col1:
        feature_to_plot = st.selectbox("Select Feature", summary["numerical_cols"])
        plot_type = st.radio("Plot Type", ["Histogram", "Boxplot (Outliers)", "Violin Plot"])
        color_by = st.selectbox(
            "Color By (Optional)",
            ["None", "Graduation_Status", "Placement_Status"],
        )

    with col2:
        color_arg = color_by if color_by != "None" else None

        if plot_type == "Histogram":
            fig = px.histogram(
                df, x=feature_to_plot, color=color_arg,
                marginal="box", template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
        elif plot_type == "Violin Plot":
            fig = px.violin(
                df, y=feature_to_plot, color=color_arg,
                box=True, points="outliers",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
        else:
            fig = px.box(
                df, y=feature_to_plot, color=color_arg,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Target Distributions
    st.markdown("### Target Variable Distributions")
    t1, t2 = st.columns(2)

    with t1:
        if "Graduation_Status" in df.columns:
            grad_counts = df["Graduation_Status"].value_counts().reset_index()
            grad_counts.columns = ["Status", "Count"]
            grad_counts["Status"] = grad_counts["Status"].map({0: "Not Graduated", 1: "Graduated"})

            fig_pie1 = px.pie(
                grad_counts, values="Count", names="Status", hole=0.4,
                title="Graduation Status", template="plotly_dark",
                color_discrete_sequence=["#e74c3c", "#2ecc71"],
            )
            fig_pie1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie1, use_container_width=True)

    with t2:
        if "Placement_Status" in df.columns:
            place_counts = df["Placement_Status"].value_counts().reset_index()
            place_counts.columns = ["Status", "Count"]
            place_counts["Status"] = place_counts["Status"].map({0: "Not Placed", 1: "Placed"})

            fig_pie2 = px.pie(
                place_counts, values="Count", names="Status", hole=0.4,
                title="Placement Status", template="plotly_dark",
                color_discrete_sequence=["#e67e22", "#3498db"],
            )
            fig_pie2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie2, use_container_width=True)

# ── Tab 3: Correlations ──────────────────────────────────────────
with tab3:
    st.markdown("### Correlation Heatmap")

    corr_df = df[summary["numerical_cols"]].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=corr_df.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))

    fig_corr.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Top correlations with targets
    st.markdown("### 🎯 Top Correlations with Target Variables")
    tc1, tc2 = st.columns(2)

    with tc1:
        if "Graduation_Status" in corr_df.columns:
            grad_corr = corr_df["Graduation_Status"].drop(
                ["Graduation_Status", "Placement_Status"], errors="ignore"
            ).sort_values(key=abs, ascending=False)

            fig_gc = px.bar(
                x=grad_corr.values, y=grad_corr.index,
                orientation="h", title="Correlations with Graduation",
                template="plotly_dark",
                color=grad_corr.values,
                color_continuous_scale="RdBu",
            )
            fig_gc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gc, use_container_width=True)

    with tc2:
        if "Placement_Status" in corr_df.columns:
            place_corr = corr_df["Placement_Status"].drop(
                ["Graduation_Status", "Placement_Status"], errors="ignore"
            ).sort_values(key=abs, ascending=False)

            fig_pc = px.bar(
                x=place_corr.values, y=place_corr.index,
                orientation="h", title="Correlations with Placement",
                template="plotly_dark",
                color=place_corr.values,
                color_continuous_scale="RdBu",
            )
            fig_pc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pc, use_container_width=True)

# ── Tab 4: Statistics ─────────────────────────────────────────────
with tab4:
    st.markdown("### 📊 Descriptive Statistics")

    st.dataframe(
        df[summary["numerical_cols"]].describe().round(2).T,
        use_container_width=True,
    )

    # Missing values
    if summary["missing_total"] > 0:
        st.markdown("### ⚠️ Missing Values")
        st.dataframe(summary["missing_by_col"], use_container_width=True)

    # Data types
    st.markdown("### 🏷️ Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Non-Null": df.notnull().sum(),
        "Null": df.isnull().sum(),
        "Unique": df.nunique(),
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)
