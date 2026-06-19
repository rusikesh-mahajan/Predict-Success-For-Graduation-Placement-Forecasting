"""
Page 4: Explainability — SHAP-based model explanations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# ── Page Config & Styles ──────────────────────────────────────────
st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")

css_path = os.path.join(root_dir, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔍 Model Explainability</h1>", unsafe_allow_html=True)

# ── Check dependencies ───────────────────────────────────────────
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.error("⚠️ SHAP is not installed. Run: `pip install shap`")
    st.stop()

from src.explainability import (
    compute_shap_values,
    get_top_features,
    generate_shap_report,
)

# ── Check for training results ───────────────────────────────────
if "training_results" not in st.session_state or st.session_state.training_results is None:
    st.warning("⚠️ No training results found. Please go to **Model Training** first.")
    st.stop()

res = st.session_state.training_results

# ── Model Selection ──────────────────────────────────────────────
st.markdown("### Select Model for SHAP Analysis")

available_models = list(res["models"].keys())
selected_model = st.selectbox("Choose a model:", available_models)

max_samples = st.slider(
    "Number of samples to analyze:", min_value=50, max_value=500,
    value=100, step=50,
    help="More samples = more accurate but slower",
)

if st.button("🔬 Run SHAP Analysis", use_container_width=True):
    model = res["models"][selected_model]

    with st.spinner(f"Computing SHAP values for {selected_model}..."):
        shap_values = compute_shap_values(
            model, res["X_test"], res["feature_cols"],
            model_name=selected_model,
            max_samples=max_samples,
        )

    if shap_values is not None:
        st.session_state.shap_values = shap_values
        st.session_state.shap_model_name = selected_model
        st.success("SHAP analysis complete!")
    else:
        st.error("SHAP computation failed. Check the logs for details.")

# ═══════════════════════════════════════════════════════════════════
#  Display SHAP Results
# ═══════════════════════════════════════════════════════════════════
if "shap_values" in st.session_state and st.session_state.shap_values is not None:
    shap_values = st.session_state.shap_values
    model_name = st.session_state.get("shap_model_name", "Model")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summary Plot",
        "📋 Feature Importance",
        "🏆 Top Features",
        "📄 Report",
    ])

    # ── Tab 1: Summary Plot ──────────────────────────────────────
    with tab1:
        st.markdown(f"### SHAP Summary Plot — {model_name}")
        st.markdown("""
        Each dot represents a single prediction. The color shows the
        feature value (red = high, blue = low), and the horizontal
        position shows the SHAP value (impact on prediction).
        """)

        fig_summary, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, show=False)
        plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_summary)
        plt.close("all")

    # ── Tab 2: Feature Importance Bar ────────────────────────────
    with tab2:
        st.markdown(f"### SHAP Feature Importance — {model_name}")
        st.markdown("Mean absolute SHAP value = average impact of each feature.")

        fig_bar, ax = plt.subplots(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close("all")

    # ── Tab 3: Top Features Table ────────────────────────────────
    with tab3:
        st.markdown(f"### 🏆 Top 10 Most Influential Features — {model_name}")

        top_features = get_top_features(shap_values, top_n=10)

        if top_features is not None:
            st.dataframe(
                top_features.style.background_gradient(
                    subset=["Mean_Abs_SHAP"], cmap="YlOrRd",
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Plotly bar chart
            fig_top = px.bar(
                top_features, x="Mean_Abs_SHAP", y="Feature",
                orientation="h",
                title=f"Top 10 Features — {model_name}",
                template="plotly_dark",
                color="Mean_Abs_SHAP",
                color_continuous_scale="YlOrRd",
            )
            fig_top.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_top, use_container_width=True)

            # Key insights
            st.markdown("### 💡 Key Insights")
            top3 = top_features.head(3)["Feature"].tolist()
            st.markdown(f"""
            The analysis reveals that the top 3 most influential features for
            **{res['target'].replace('_', ' ')}** prediction are:

            1. **{top3[0]}** — Strongest predictor
            2. **{top3[1]}** — Second most important
            3. **{top3[2]}** — Third most important

            These features have the highest mean absolute SHAP values,
            indicating they contribute the most to the model's predictions.
            """)

    # ── Tab 4: Generate Report ───────────────────────────────────
    with tab4:
        st.markdown("### 📄 SHAP Explainability Report")

        if st.button("📝 Generate SHAP Report"):
            with st.spinner("Generating report..."):
                report_path = generate_shap_report(
                    shap_values_grad=shap_values if "Graduation" in res["target"] else None,
                    shap_values_place=shap_values if "Placement" in res["target"] else None,
                    model_name=model_name,
                )
                st.success(f"Report saved to: `{report_path}`")

                # Display report
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        st.markdown(f.read())

else:
    st.info("Click **Run SHAP Analysis** above to generate explanations.")
