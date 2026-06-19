"""
Page 5: Predict — Real-time student outcome predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import io

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.model_tuning import list_saved_models, load_model
from src.predict import prepare_new_student, predict_student, validate_input

# ── Page Config & Styles ──────────────────────────────────────────
st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")

css_path = os.path.join(root_dir, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔮 Predict Outcomes</h1>", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────────
available_models = list_saved_models()

if not available_models:
    st.warning("⚠️ No trained models found. Please go to **Model Training** and save a model first.")
    st.stop()

# ── Model Selection ──────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    grad_models = [m for m in available_models if "Graduation" in m]
    selected_grad_model = st.selectbox(
        "🎓 Select Graduation Model:", grad_models,
    ) if grad_models else None

with col2:
    place_models = [m for m in available_models if "Placement" in m]
    selected_place_model = st.selectbox(
        "💼 Select Placement Model:", place_models,
    ) if place_models else None

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
#  Prediction Mode Selection
# ═══════════════════════════════════════════════════════════════════
pred_mode = st.radio(
    "Prediction Mode:",
    ["📝 Single Student", "📁 Batch Upload (CSV)"],
    horizontal=True,
)

# ═══════════════════════════════════════════════════════════════════
#  Single Student Prediction
# ═══════════════════════════════════════════════════════════════════
if pred_mode == "📝 Single Student":
    st.markdown("### 📝 Enter Student Details")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 30, 21)
            cgpa = st.slider("CGPA", 2.0, 10.0, 7.5, 0.1)
            attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0, 1.0)

        with c2:
            internal_marks = st.number_input("Internal Marks (0-100)", 0.0, 100.0, 60.0)
            external_marks = st.number_input("External Marks (0-100)", 0.0, 100.0, 60.0)
            backlogs = st.number_input("Backlogs", 0, 10, 0)
            study_hours = st.slider("Study Hours/Week", 0.0, 60.0, 15.0)

        with c3:
            internship = st.selectbox("Internship Experience", ["Yes", "No"])
            projects = st.number_input("Projects Completed", 0, 10, 2)
            comm_skills = st.slider("Communication Skills (1-10)", 1.0, 10.0, 7.0)
            aptitude = st.slider("Aptitude Test Score", 0.0, 100.0, 65.0)

        submitted = st.form_submit_button("🔮 Predict Outcomes", use_container_width=True)

    if submitted:
        # Build input DataFrame
        input_data = pd.DataFrame([{
            "Gender": 1 if gender == "Male" else 0,
            "Age": age,
            "Attendance_Percentage": attendance,
            "CGPA": cgpa,
            "Internal_Marks": internal_marks,
            "External_Marks": external_marks,
            "Backlogs": backlogs,
            "Study_Hours_Per_Week": study_hours,
            "Internship_Experience": 1 if internship == "Yes" else 0,
            "Projects_Completed": projects,
            "Communication_Skills_Score": comm_skills,
            "Aptitude_Test_Score": aptitude,
        }])

        # Validate input
        warnings = validate_input(input_data)
        for w in warnings:
            st.warning(f"⚠️ {w}")

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        # Graduation Prediction
        if selected_grad_model:
            try:
                bundle = load_model(selected_grad_model)
                processed = prepare_new_student(
                    input_data, bundle["feature_cols"], bundle["scaler"],
                )
                pred, prob = predict_student(bundle["model"], processed)

                prob_pct = prob[0] * 100
                is_positive = pred[0] == 1

                with res_col1:
                    st.markdown("### 🎓 Graduation Prediction")

                    if is_positive:
                        st.success(f"**Will Graduate** (Confidence: {prob_pct:.1f}%)")
                    else:
                        st.error(f"**At Risk of Not Graduating** (Confidence: {(100 - prob_pct):.1f}%)")

                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_pct,
                        title={"text": "Graduation Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#2ecc71" if is_positive else "#e74c3c"},
                            "steps": [
                                {"range": [0, 50], "color": "rgba(255,0,0,0.15)"},
                                {"range": [50, 100], "color": "rgba(0,255,0,0.15)"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 2},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    ))
                    fig_g.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_g, use_container_width=True)
            except Exception as e:
                with res_col1:
                    st.error(f"Graduation prediction failed: {e}")

        # Placement Prediction
        if selected_place_model:
            try:
                bundle = load_model(selected_place_model)
                processed = prepare_new_student(
                    input_data, bundle["feature_cols"], bundle["scaler"],
                )
                pred, prob = predict_student(bundle["model"], processed)

                prob_pct = prob[0] * 100
                is_positive = pred[0] == 1

                with res_col2:
                    st.markdown("### 💼 Placement Prediction")

                    if is_positive:
                        st.success(f"**Will Get Placed** (Confidence: {prob_pct:.1f}%)")
                    else:
                        st.warning(f"**At Risk of No Placement** (Confidence: {(100 - prob_pct):.1f}%)")

                    fig_p = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_pct,
                        title={"text": "Placement Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#3498db" if is_positive else "#e67e22"},
                            "steps": [
                                {"range": [0, 50], "color": "rgba(255,165,0,0.15)"},
                                {"range": [50, 100], "color": "rgba(52,152,219,0.15)"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 2},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    ))
                    fig_p.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_p, use_container_width=True)
            except Exception as e:
                with res_col2:
                    st.error(f"Placement prediction failed: {e}")

        # Download single result
        st.markdown("---")
        result_data = input_data.copy()
        if selected_grad_model:
            result_data["Graduation_Prediction"] = "Graduate" if pred[0] == 1 else "At Risk"
            result_data["Graduation_Probability"] = f"{prob_pct:.1f}%"

        csv_buffer = io.StringIO()
        result_data.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Download Prediction Result",
            csv_buffer.getvalue(),
            "prediction_result.csv",
            "text/csv",
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════
#  Batch Prediction
# ═══════════════════════════════════════════════════════════════════
else:
    st.markdown("### 📁 Batch Prediction — Upload CSV")
    st.markdown("""
    Upload a CSV file with student records. Required columns:
    `Gender` (Male/Female or 0/1), `Age`, `Attendance_Percentage`, `CGPA`,
    `Internal_Marks`, `External_Marks`, `Backlogs`, `Study_Hours_Per_Week`,
    `Internship_Experience` (0/1), `Projects_Completed`,
    `Communication_Skills_Score`, `Aptitude_Test_Score`
    """)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(batch_df)} records.**")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🔮 Run Batch Predictions", use_container_width=True):
                results = batch_df.copy()

                # Encode Gender if string
                if "Gender" in results.columns and results["Gender"].dtype == "object":
                    results["Gender"] = results["Gender"].map({"Male": 1, "Female": 0})

                # Graduation
                if selected_grad_model:
                    with st.spinner("Predicting graduation..."):
                        bundle = load_model(selected_grad_model)
                        processed = prepare_new_student(
                            results, bundle["feature_cols"], bundle["scaler"],
                        )
                        preds, probs = predict_student(bundle["model"], processed)
                        results["Graduation_Prediction"] = np.where(preds == 1, "Graduate", "At Risk")
                        results["Graduation_Probability"] = (probs * 100).round(1)

                # Placement
                if selected_place_model:
                    with st.spinner("Predicting placement..."):
                        bundle = load_model(selected_place_model)
                        processed = prepare_new_student(
                            results, bundle["feature_cols"], bundle["scaler"],
                        )
                        preds, probs = predict_student(bundle["model"], processed)
                        results["Placement_Prediction"] = np.where(preds == 1, "Placed", "At Risk")
                        results["Placement_Probability"] = (probs * 100).round(1)

                st.success("Batch predictions complete!")
                st.dataframe(results, use_container_width=True)

                # Download results
                csv_buffer = io.StringIO()
                results.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Batch Results",
                    csv_buffer.getvalue(),
                    "batch_predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing batch file: {e}")
