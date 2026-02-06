# Early Diabetes Causal AI

## Project Overview
This project implements an AI-based system for early diabetes risk prediction using blood test data.
It combines machine learning models with causal inference to generate interpretable treatment and lifestyle recommendations.

## Team Workflow
- main: stable demo branch
- dev: integration branch
- feature/*: individual work branches

## Tech Stack
Python, scikit-learn, XGBoost, DoWhy, CausalNex, Streamlit

## Setup and Training

### Prerequisites
- Python 3.11+ recommended
- On macOS: Install libomp for XGBoost: `brew install libomp`

### Installation
```bash
pip install -r requirements.txt
```

### Training Artifacts

1. **Train Prediction Model** (combines NHANES + Pima datasets):
```bash
python scripts/train_prediction.py
```
This will:
- Load and combine NHANES and Pima datasets from `data/raw/`
- Train XGBoost (or RandomForest if XGBoost unavailable)
- Save model to `models/prediction_model.joblib`
- Save feature schema to `models/feature_schema.json`

2. **Train Causal Graph** (from NHANES data):
```bash
python scripts/train_causal.py
```
This will:
- Load NHANES dataset from `data/raw/`
- Build causal graph using CausalNex (or correlation fallback)
- Save graph to `models/causal_graph.joblib`

### Running the Dashboard
```bash
streamlit run src/dashboard/app.py
```

The dashboard will use the trained models if they exist, or show friendly warnings if not found.
