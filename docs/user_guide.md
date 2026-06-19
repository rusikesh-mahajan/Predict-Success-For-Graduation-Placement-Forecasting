# 📖 User Guide

## Getting Started

Launch the application:
```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## Navigation

Use the sidebar to navigate between 6 modules:

### 1. 📊 Data Explorer
- **Overview**: See dataset size, missing values, duplicates
- **Preview**: Filter and browse raw data
- **Distributions**: Histograms, boxplots, violin plots for any feature
- **Correlations**: Heatmap showing feature relationships
- **Statistics**: Descriptive statistics and data type information

### 2. 🤖 Model Training
- **Configure**: Select target (Graduation/Placement), toggle feature engineering, CV, and tuning
- **Train**: Click "Start Training Pipeline" to train all 5 models
- **Review**: See cross-validation and holdout results
- **Save**: Save the best model for predictions

### 3. 📈 Model Comparison
- **Metrics Table**: All models ranked by ROC-AUC
- **Radar Chart**: Visual performance profiles
- **ROC/PR Curves**: Interactive plotly charts
- **Feature Importance**: Compare importances across models

### 4. 🔍 Explainability
- **Select Model**: Choose which model to explain
- **Run SHAP**: Compute SHAP values (may take a moment)
- **Summary Plot**: Beeswarm plot showing feature impacts
- **Top Features**: Ranked list of most influential factors
- **Report**: Generate a markdown explainability report

### 5. 🔮 Predict
- **Single Prediction**: Enter one student's details
- **Batch Prediction**: Upload a CSV with multiple students
- **Download**: Export results as CSV

## Tips
- Train models before using Comparison, Explainability, or Predict pages
- Save at least one model for each target to use the Predict page
- Use batch upload for predicting many students at once
