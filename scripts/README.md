# Training Scripts

## Quick Start

### 1. Train Prediction Model
```bash
python scripts/train_prediction.py
```

**Requirements:**
- `data/raw/NHANES Blood Panel Dataset.csv` (optional but recommended)
- `data/raw/Pima Indians Diabetes Database.csv` (optional but recommended)
- At least one dataset must be present

**Outputs:**
- `models/prediction_model.joblib` - Trained model
- `models/feature_schema.json` - Feature schema for reference

### 2. Train Causal Graph
```bash
python scripts/train_causal.py
```

**Requirements:**
- `data/raw/NHANES Blood Panel Dataset.csv`

**Outputs:**
- `models/causal_graph.joblib` - Causal graph (NetworkX format)

### 3. Run Dashboard
```bash
streamlit run src/dashboard/app.py
```

The dashboard will automatically use the trained models if they exist.

## Mac Compatibility

If you encounter XGBoost OpenMP errors on macOS:
```bash
brew install libomp
```

The scripts will automatically fall back to RandomForest if XGBoost fails.

