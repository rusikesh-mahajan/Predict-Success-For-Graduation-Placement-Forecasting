# 🤖 Model Training Guide

## Overview

This guide explains the machine learning methodology used in the project.

## Models

| Model | Type | Strengths |
|-------|------|-----------|
| Logistic Regression | Linear | Fast, interpretable, baseline |
| Decision Tree | Tree | Captures non-linearity |
| Random Forest | Ensemble (Bagging) | Robust, reduces overfitting |
| Gradient Boosting | Ensemble (Boosting) | High accuracy, sequential |
| XGBoost | Optimized Boosting | Best performance, regularized |

## Training Pipeline

### 1. Data Preparation
- Load CSV → Validate columns → Remove duplicates
- Impute missing values (median for numerical, mode for categorical)
- Cap outliers using IQR method (bounds learned from training data only)

### 2. Feature Engineering
- 5 derived features are created before encoding/scaling
- Implemented as sklearn TransformerMixin (no data leakage)

### 3. Data Splitting
- **Train**: 70% — used for model fitting
- **Validation**: 10% — used for tuning decisions
- **Test**: 20% — held out for final evaluation
- All splits use stratification to preserve class balance

### 4. Cross-Validation
- Stratified 5-Fold CV on training set
- Reports mean ± std for each metric
- More robust than single holdout

### 5. Hyperparameter Tuning
- RandomizedSearchCV on the best base model
- 50 iterations, 5-fold CV
- Optimizes ROC-AUC
- Param grids defined in `config/settings.py`

### 6. Model Selection
- Primary metric: **ROC-AUC**
- Models ranked by holdout ROC-AUC
- Best model auto-saved with scaler and feature metadata

## Preventing Data Leakage
- Scaling/encoding fitted on training data only
- Outlier bounds learned from training data only
- Feature engineering uses no target information
- Test set never seen during training or tuning
