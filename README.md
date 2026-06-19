# 🎓 Predict Success: Graduation & Placement Forecasting

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **production-grade machine learning application** that predicts student graduation and placement outcomes. Built with proper ML engineering practices including sklearn Pipelines, Stratified K-Fold Cross-Validation, SHAP explainability, and an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Architecture](#️-architecture)
- [ML Pipeline Workflow](#-ml-pipeline-workflow)
- [Models & Performance](#-models--performance)
- [Features](#-features)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

Educational institutions need data-driven tools to identify students at risk of not graduating or not securing placements. This project addresses that need by:

- Training **5 classification models** and selecting the best performer
- Providing **SHAP-based explanations** of what drives predictions
- Offering an **interactive Streamlit dashboard** with 6 modules
- Following **production ML engineering** best practices

### Key Objectives

| Objective | Description |
|-----------|-------------|
| 🎓 Graduation Prediction | Binary classification: Will Graduate vs At Risk |
| 💼 Placement Prediction | Binary classification: Will Get Placed vs At Risk |
| 🔍 Factor Identification | Which student attributes influence outcomes most |
| 📊 Actionable Insights | Enable data-driven interventions |

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| **Source** | Synthetic (realistic correlations) |
| **Total Records** | 1,005 (1,000 + 5 intentional duplicates) |
| **Input Features** | 13 |
| **Target Variables** | 2 (Graduation_Status, Placement_Status) |
| **Missing Values** | ~2% in selected columns |
| **Format** | CSV |

### Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Student_ID | ID | STU0001-STU1000 | Unique identifier |
| Gender | Categorical | Male/Female | Student gender |
| Age | Numerical | 18–26 | Student age |
| Attendance_Percentage | Numerical | 20–100 | Class attendance % |
| CGPA | Numerical | 2.0–10.0 | Cumulative GPA |
| Internal_Marks | Numerical | 10–100 | Internal assessment |
| External_Marks | Numerical | 5–100 | External exam marks |
| Backlogs | Numerical | 0–5 | Number of backlogs |
| Study_Hours_Per_Week | Numerical | 1–40 | Weekly study hours |
| Internship_Experience | Binary | 0/1 | Has internship |
| Projects_Completed | Numerical | 0–5 | Number of projects |
| Communication_Skills_Score | Numerical | 1–10 | Communication score |
| Aptitude_Test_Score | Numerical | 10–100 | Aptitude test |
| **Graduation_Status** | **Target** | **0/1** | **Graduated or not** |
| **Placement_Status** | **Target** | **0/1** | **Placed or not** |

### Engineered Features

| Feature | Formula |
|---------|---------|
| Total_Marks | Internal + External |
| Marks_Ratio | Internal / Total |
| Academic_Index | Weighted: CGPA (40%) + Attendance (30%) + Marks (30%) |
| Employability_Score | Weighted: Communication (30%) + Aptitude (30%) + Projects (20%) + Internship (20%) |
| Risk_Flag | Backlogs > 2 AND Attendance < 60% |

---

## 🏗️ Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│    Raw Data      │────▶│  Preprocessing    │────▶│  Train/Val/Test  │
│    (CSV)         │     │  Pipeline         │     │  Split           │
└──────────────────┘     └───────────────────┘     └────────┬─────────┘
                                                            │
                         ┌──────────────────────────────────┤
                         │                                  │
                    ┌────▼────┐                    ┌────────▼────────┐
                    │ 5 Models│                    │ Stratified      │
                    │ Training│                    │ K-Fold CV       │
                    └────┬────┘                    └────────┬────────┘
                         │                                  │
                    ┌────▼──────────────────────────────────▼─┐
                    │        Evaluation & SHAP                │
                    │  (Confusion Matrix, ROC, PR, Reports)   │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │       Streamlit Dashboard (6 Pages)      │
                    │  Data │ Train │ Compare │ SHAP │ Predict │
                    └──────────────────────────────────────────┘
```

---

## 🔄 ML Pipeline Workflow

```
1. Data Loading & Validation
   └── Column checks, type validation

2. Preprocessing (sklearn Pipeline)
   ├── Remove duplicates
   ├── Impute missing values (median/mode)
   ├── Cap outliers (IQR method)
   ├── Feature engineering (5 new features)
   ├── Encode categoricals (OneHotEncoder)
   └── Scale numerics (StandardScaler)

3. Data Splitting
   ├── Train: 70%
   ├── Validation: 10%
   └── Test: 20%

4. Model Training
   ├── Logistic Regression
   ├── Decision Tree
   ├── Random Forest
   ├── Gradient Boosting
   └── XGBoost

5. Evaluation
   ├── Stratified 5-Fold Cross-Validation
   ├── Holdout test set metrics
   ├── Confusion matrices
   ├── ROC & Precision-Recall curves
   └── Classification reports

6. Explainability
   ├── SHAP TreeExplainer / LinearExplainer
   ├── Summary plots (beeswarm)
   ├── Feature importance rankings
   └── Top-10 influential factors report

7. Model Persistence
   └── Best model saved via joblib
```

---

## 🤖 Models & Performance

### Model Comparison

| Model | Key Characteristics |
|-------|-------------------|
| Logistic Regression | Linear, interpretable, fast |
| Decision Tree | Non-linear, prone to overfitting |
| Random Forest | Ensemble, robust, good default |
| Gradient Boosting | Sequential ensemble, high accuracy |
| XGBoost | Optimized boosting, best overall |

### Evaluation Metrics

All models are evaluated using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness |
| **Precision** | Of predicted positives, how many are correct |
| **Recall** | Of actual positives, how many are found |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the ROC curve (primary metric) |

> **Note:** Actual metric values depend on the training run. Use the dashboard to see current results.

---

## ✨ Features

### 📊 Data Explorer
- Dataset overview with metrics
- Interactive histograms, boxplots, violin plots
- Correlation heatmaps with target analysis
- Descriptive statistics tables
- Data filtering by gender and status

### 🤖 Model Training
- One-click training pipeline
- Stratified K-Fold Cross-Validation
- Holdout test evaluation
- Optional hyperparameter tuning (RandomizedSearchCV)
- Auto-save evaluation plots

### 📈 Model Comparison
- Side-by-side metric comparison
- Radar chart performance profiles
- Interactive ROC & PR curves
- CV vs holdout analysis
- Feature importance comparison

### 🔍 Explainability (SHAP)
- SHAP summary plots (beeswarm)
- Feature importance bar charts
- Top-10 influential features table
- Auto-generated explainability reports

### 🔮 Predictions
- Single student prediction with gauges
- Batch prediction via CSV upload
- Downloadable results
- Input validation warnings

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Predict-Success-Graduation-Placement-Forecasting.git
cd Predict-Success-Graduation-Placement-Forecasting

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset (if not present)
python data/generate_dataset.py

# 5. Run the application
streamlit run app.py
```

### Single-Command Quick Start

```bash
pip install -r requirements.txt && python data/generate_dataset.py && streamlit run app.py
```

---

## 🚀 Usage

1. **Start the app:** `streamlit run app.py`
2. **Explore data:** Navigate to 📊 Data Explorer
3. **Train models:** Go to 🤖 Model Training → Configure → Start
4. **Compare models:** Check 📈 Model Comparison
5. **Explain predictions:** Visit 🔍 Explainability
6. **Make predictions:** Use 🔮 Predict (single or batch)

---

## 📂 Project Structure

```
Ml_cloud/
├── app.py                              # Streamlit main entry point
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
├── Procfile                            # Render deployment
├── render.yaml                         # Render blueprint
├── setup.sh                            # Environment setup
│
├── .streamlit/
│   └── config.toml                     # Streamlit theme configuration
│
├── assets/
│   └── style.css                       # Custom CSS (glassmorphism theme)
│
├── config/
│   ├── __init__.py
│   └── settings.py                     # Paths, hyper-parameters, constants
│
├── data/
│   ├── generate_dataset.py             # Synthetic data generator
│   ├── raw/
│   │   └── student_data.csv            # Raw dataset
│   └── processed/                      # Processed data (generated)
│
├── docs/
│   ├── technical_documentation.md      # Architecture & pipeline details
│   ├── user_guide.md                   # Dashboard usage guide
│   ├── model_training_guide.md         # ML training methodology
│   └── deployment_guide.md             # Deployment instructions
│
├── models/                             # Saved model bundles (.joblib)
│
├── notebooks/
│   └── *.ipynb                         # Jupyter notebooks
│
├── pages/                              # Streamlit multi-page app
│   ├── 1_📊_Data_Explorer.py
│   ├── 2_🤖_Model_Training.py
│   ├── 3_📈_Model_Comparison.py
│   ├── 4_🔍_Explainability.py
│   └── 5_🔮_Predict.py
│
├── reports/
│   ├── figures/                        # Generated evaluation plots
│   └── shap_report.md                  # SHAP explainability report
│
├── src/                                # Core ML pipeline
│   ├── __init__.py
│   ├── data_loader.py                  # Data loading & validation
│   ├── evaluation.py                   # Evaluation plots & reports
│   ├── explainability.py               # SHAP analysis
│   ├── feature_engineering.py          # Feature creation (sklearn transformer)
│   ├── model_training.py               # Training & cross-validation
│   ├── model_tuning.py                 # Hyperparameter tuning & persistence
│   ├── predict.py                      # Inference pipeline
│   ├── preprocessing.py                # sklearn Pipeline (impute, scale, encode)
│   └── utils.py                        # Logging, seeds, utilities
│
└── tests/                              # Unit tests
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_preprocessing.py
    └── test_model_training.py
```

---

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file path: `app.py`
5. Deploy!

### Render

1. Push to GitHub
2. Create a new Web Service on [render.com](https://render.com)
3. Connect the repository
4. Render will use `render.yaml` automatically

See [docs/deployment_guide.md](docs/deployment_guide.md) for detailed instructions.

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_data_loader.py -v

# Run with coverage
python -m pytest tests/ -v --tb=short
```

---

## 🔮 Future Improvements

- [ ] **Deep Learning**: Add neural network models (PyTorch/TensorFlow)
- [ ] **Feature Selection**: Automated feature selection (RFE, Boruta)
- [ ] **AutoML**: Integration with AutoML frameworks
- [ ] **Real-time Monitoring**: Model drift detection & alerting
- [ ] **API Endpoint**: FastAPI REST API for predictions
- [ ] **Database Integration**: PostgreSQL for student records
- [ ] **A/B Testing**: Model comparison in production
- [ ] **CI/CD Pipeline**: GitHub Actions for automated testing
- [ ] **Docker**: Containerized deployment
- [ ] **User Authentication**: Role-based access control

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Technical Documentation](docs/technical_documentation.md) | Architecture, pipeline, design decisions |
| [User Guide](docs/user_guide.md) | How to use the Streamlit dashboard |
| [Model Training Guide](docs/model_training_guide.md) | ML methodology & training process |
| [Deployment Guide](docs/deployment_guide.md) | Cloud deployment instructions |

---

## 📝 License

This project is licensed under the MIT License.

---

**Built with ❤️ by the Data Science Team**
