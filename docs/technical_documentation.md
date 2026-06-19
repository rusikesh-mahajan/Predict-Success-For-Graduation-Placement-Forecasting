# 🏗️ Technical Documentation

## Architecture Overview

The Predict Success platform follows a modular architecture with clear separation of concerns:

### Core Layers

1. **Data Layer** (`src/data_loader.py`)
   - CSV ingestion with validation
   - Column type checking
   - Duplicate detection and removal

2. **Preprocessing Layer** (`src/preprocessing.py`)
   - `DataFrameImputer` — median/mode imputation
   - `OutlierCapper` — IQR-based capping with learned bounds
   - `ColumnDropper` — removes ID/target columns
   - `StandardScaler` — zero-mean, unit-variance scaling
   - `OneHotEncoder` — categorical encoding

3. **Feature Engineering Layer** (`src/feature_engineering.py`)
   - `FeatureEngineer` — sklearn `TransformerMixin`
   - Creates 5 derived features from raw inputs
   - Can be included in sklearn Pipeline

4. **Model Layer** (`src/model_training.py`, `src/model_tuning.py`)
   - 5 classifiers with configurable hyperparameters
   - Stratified K-Fold Cross-Validation
   - RandomizedSearchCV tuning
   - Model persistence via joblib

5. **Evaluation Layer** (`src/evaluation.py`)
   - Confusion matrices, ROC/PR curves
   - Classification reports
   - Feature importance plots

6. **Explainability Layer** (`src/explainability.py`)
   - SHAP TreeExplainer / LinearExplainer
   - Summary and importance plots
   - Automated report generation

7. **Presentation Layer** (Streamlit `pages/`)
   - 6 interactive pages
   - Plotly visualizations
   - Custom CSS theme

## Data Flow

```
CSV → Validate → Impute → Cap Outliers → Engineer Features
→ Drop ID/Targets → Encode Categoricals → Scale Numerics
→ Split (70/10/20) → Train Models → Evaluate → SHAP → Dashboard
```

## Design Decisions

### Why sklearn Pipeline?
- Prevents data leakage (transformers fit only on training data)
- Reproducible preprocessing for inference
- Serializable (save/load entire pipeline)

### Why Stratified K-Fold?
- Handles class imbalance
- More robust than single holdout
- Reduces variance in performance estimates

### Why SHAP?
- Model-agnostic explanations
- Theoretically grounded (Shapley values)
- Works with both linear and tree models
