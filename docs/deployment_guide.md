# 🌐 Deployment Guide

## Streamlit Cloud

### Prerequisites
- GitHub account
- Repository pushed to GitHub

### Steps

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Production-ready release"
   git push origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"**

4. **Configure:**
   - Repository: `your-username/Predict-Success-Graduation-Placement-Forecasting`
   - Branch: `main`
   - Main file path: `app.py`

5. **Click "Deploy"**

### Notes
- Streamlit Cloud auto-installs from `requirements.txt`
- The `.streamlit/config.toml` is used automatically
- Ensure `data/raw/student_data.csv` is in the repository

---

## Render

### Prerequisites
- Render account
- Repository on GitHub

### Steps

1. **Create a new Web Service on [render.com](https://render.com)**

2. **Connect your GitHub repository**

3. **Configure:**
   - Name: `predict-success`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Deploy**

### Using render.yaml (Auto-Deploy)

The `render.yaml` file in the repository configures auto-deployment:

```yaml
services:
  - type: web
    name: predict-success
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## Environment Variables

No environment variables are required. All configuration is in `config/settings.py`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | Ensure all files are committed |
| Dataset not found | Run `python data/generate_dataset.py` |
| SHAP errors | Install with `pip install shap` |
| Port in use | Change port in `.streamlit/config.toml` |
