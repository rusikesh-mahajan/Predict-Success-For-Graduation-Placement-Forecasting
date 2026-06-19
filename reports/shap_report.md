# 🔍 SHAP Explainability Report

**Model:** Logistic Regression

This report identifies the most influential factors affecting
student graduation and placement predictions using SHAP
(SHapley Additive exPlanations).

---

## Graduation Prediction

### Top 10 Most Influential Features

| Rank | Feature | Mean |SHAP| Impact |
|------|---------|------|
| 1 | Backlogs | 1.1985 |
| 2 | CGPA | 0.5028 |
| 3 | Academic_Index | 0.4708 |
| 4 | Marks_Ratio | 0.3016 |
| 5 | Study_Hours_Per_Week | 0.2128 |
| 6 | Attendance_Percentage | 0.2054 |
| 7 | Risk_Flag | 0.1240 |
| 8 | Age | 0.0860 |
| 9 | Gender | 0.0676 |
| 10 | Communication_Skills_Score | 0.0477 |

### Key Insights — Graduation

The top 3 most influential features for graduation prediction are:

1. **Backlogs**
2. **CGPA**
3. **Academic_Index**

---

## Placement Prediction

*SHAP analysis not available.*

---

## Methodology

- **SHAP** uses Shapley values from cooperative game theory to assign
  each feature a contribution to every prediction.
- **TreeExplainer** is used for tree-based models (exact computation).
- **LinearExplainer** is used for linear models.
- Mean absolute SHAP value represents average feature impact.

---

*Report generated automatically by the Predict Success ML pipeline.*