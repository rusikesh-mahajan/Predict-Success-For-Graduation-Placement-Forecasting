import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.model_tuning import list_saved_models, load_model
from src.predict import prepare_new_student, predict_student

# --- Page Config & Styles ---
st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")

css_path = os.path.join(root_dir, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔮 Predict Outcomes</h1>", unsafe_allow_html=True)

# --- Load Models ---
available_models = list_saved_models()

if not available_models:
    st.warning("⚠️ No trained models found. Please go to the **Model Training** page and save a model first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    grad_models = [m for m in available_models if "Graduation" in m]
    selected_grad_model = st.selectbox("Select Graduation Model:", grad_models) if grad_models else None

with col2:
    place_models = [m for m in available_models if "Placement" in m]
    selected_place_model = st.selectbox("Select Placement Model:", place_models) if place_models else None

st.markdown("---")

# --- Input Form ---
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

# --- Prediction Logic ---
if submitted:
    # 1. Create input DataFrame
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
        "Aptitude_Test_Score": aptitude
    }])
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    # 2. Predict Graduation
    if selected_grad_model:
        bundle = load_model(selected_grad_model)
        processed_input = prepare_new_student(input_data, bundle["feature_cols"], bundle["scaler"])
        pred, prob = predict_student(bundle["model"], processed_input)
        
        prob_pct = prob[0] * 100
        is_positive = pred[0] == 1
        
        with res_col1:
            st.markdown("### 🎓 Graduation Prediction")
            
            if is_positive:
                st.success(f"**Will Graduate** (Confidence: {prob_pct:.1f}%)")
            else:
                st.error(f"**At Risk of Not Graduating** (Confidence: {(100-prob_pct):.1f}%)")
                
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_pct,
                title = {'text': "Graduation Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green" if is_positive else "red"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255,0,0,0.2)"},
                        {'range': [50, 100], 'color': "rgba(0,255,0,0.2)"}
                    ]
                }
            ))
            fig_g.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, use_container_width=True)

    # 3. Predict Placement
    if selected_place_model:
        bundle = load_model(selected_place_model)
        processed_input = prepare_new_student(input_data, bundle["feature_cols"], bundle["scaler"])
        pred, prob = predict_student(bundle["model"], processed_input)
        
        prob_pct = prob[0] * 100
        is_positive = pred[0] == 1
        
        with res_col2:
            st.markdown("### 💼 Placement Prediction")
            
            if is_positive:
                st.success(f"**Will Get Placed** (Confidence: {prob_pct:.1f}%)")
            else:
                st.warning(f"**At Risk of No Placement** (Confidence: {(100-prob_pct):.1f}%)")
                
            fig_p = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_pct,
                title = {'text': "Placement Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3498db" if is_positive else "orange"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255,165,0,0.2)"},
                        {'range': [50, 100], 'color': "rgba(52,152,219,0.2)"}
                    ]
                }
            ))
            fig_p.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_p, use_container_width=True)
