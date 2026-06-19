#!/bin/bash
# ============================================================
# Predict Success — Environment Setup Script
# ============================================================
# Usage: bash setup.sh

echo "============================================================"
echo "  Predict Success: Setup Script"
echo "============================================================"

# 1. Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."
python -m venv venv

# 2. Activate virtual environment
echo "[2/4] Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# 3. Install dependencies
echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate dataset
echo "[4/4] Generating dataset..."
python data/generate_dataset.py

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "  Run: streamlit run app.py"
echo "============================================================"
