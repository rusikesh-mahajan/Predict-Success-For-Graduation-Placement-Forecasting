# 🎓 Predict Success: ML Models for Graduation & Placement Forecasting

This project uses machine learning to predict whether a student will graduate and secure a placement. It provides an industry-level file structure with a premium Streamlit dashboard.

## 🚀 Features

* **Data Explorer (`pages/1_📊_Data_Explorer.py`)**: Interactive EDA with Plotly, outlier detection, and correlation analysis.
* **Model Training (`pages/2_🤖_Model_Training.py`)**: Train multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost), compare metrics (ROC-AUC, Accuracy, F1), tune hyperparameters using `RandomizedSearchCV`, and save the best model.
* **Predict Outcomes (`pages/3_🔮_Predict.py`)**: Input new student data and get real-time predictions with confidence gauges.

## 📂 Project Structure

```
Ml_cloud/
├── app.py                         # Streamlit main entry point
├── README.md
├── requirements.txt               # Dependencies
├── assets/
│   └── style.css                  # Custom CSS for the dashboard
├── config/
│   ├── __init__.py
│   └── settings.py                # Hyperparameters, paths, feature engineering weights
├── data/
│   ├── processed/
│   └── raw/                       # Put `student_data.csv` here
├── models/                        # Saved Joblib models
├── notebooks/                     # Original Jupyter notebooks
├── pages/                         # Streamlit multi-page app
│   ├── 1_📊_Data_Explorer.py
│   ├── 2_🤖_Model_Training.py
│   └── 3_🔮_Predict.py
└── src/                           # Core ML logic
    ├── __init__.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── model_training.py
    ├── model_tuning.py
    ├── predict.py
    └── preprocessing.py
```

## 🛠️ Setup Instructions

1. **Install Requirements**:
   Ensure you have Python 3.9+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Data**:
   Ensure your dataset `student_data.csv` is located at `data/raw/student_data.csv`.

3. **Run the App**:
   Start the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
