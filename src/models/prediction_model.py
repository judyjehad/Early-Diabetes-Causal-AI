import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

# Try to import XGBoost, fallback to None if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def train_prediction_model(train_csv_path: str, target_column: str, model_output_path: str) -> dict:
    """
    Trains a prediction model with preprocessing pipeline and saves it.
    
    Args:
        train_csv_path: Path to training CSV file
        target_column: Name of the target column
        model_output_path: Path where the trained model will be saved
        
    Returns:
        Dictionary with model_path, model_type, n_samples, and n_features
    """
    # Load training data
    df = pd.read_csv(train_csv_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify numeric and categorical columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Build preprocessing transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ],
        remainder='passthrough'
    )
    
    # Select model: XGBoost if available, otherwise RandomForest
    if XGBOOST_AVAILABLE:
        try:
            # Test if XGBoost works (handles OpenMP issues on Mac)
            if len(X) > 0:
                test_X = X.head(2)
                test_y = y.head(2)
                test_model = xgb.XGBClassifier(n_estimators=1, random_state=42, n_jobs=1)
                test_model.fit(test_X, test_y)
            model = xgb.XGBClassifier(random_state=42, n_estimators=100, n_jobs=1)
            model_type = "XGBoost"
        except Exception as e:
            # XGBoost failed (likely OpenMP issue), use RandomForest
            import warnings
            warnings.warn(f"XGBoost failed ({str(e)}), using RandomForest. On Mac, try: brew install libomp")
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model_type = "RandomForest"
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_type = "RandomForest"
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(model_output_path) if os.path.dirname(model_output_path) else '.', exist_ok=True)
    
    # Save the trained pipeline
    joblib.dump(pipeline, model_output_path)
    
    # Return metadata
    return {
        "model_path": model_output_path,
        "model_type": model_type,
        "n_samples": len(X),
        "n_features": X.shape[1]
    }


def evaluate_prediction_model(model_path: str, test_csv_path: str, target_column: str) -> dict:
    """
    Evaluates a trained model on test data and returns metrics.
    
    Args:
        model_path: Path to the saved model file
        test_csv_path: Path to test CSV file
        target_column: Name of the target column
        
    Returns:
        Dictionary with accuracy, precision, recall, f1, and optionally roc_auc
    """
    # Load the model pipeline
    pipeline = joblib.load(model_path)
    
    # Load test data
    df_test = pd.read_csv(test_csv_path)
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    # Try to calculate ROC AUC if probabilities are available and target is binary
    try:
        y_pred_proba = pipeline.predict_proba(X_test)
        # Check if binary classification (2 classes)
        if y_pred_proba.shape[1] == 2:
            # Use probabilities for positive class
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            metrics["roc_auc"] = float(roc_auc)
    except (AttributeError, IndexError):
        # Model doesn't support predict_proba or other issue
        pass
    
    return metrics


def predict_single(model_path: str, input_features: dict) -> dict:
    """
    Makes a prediction for a single sample using the trained model.
    
    Args:
        model_path: Path to the saved model file
        input_features: Dictionary of feature names and values
        
    Returns:
        Dictionary with label and risk_score (probability if available)
    """
    # Load the model pipeline
    pipeline = joblib.load(model_path)
    
    # Convert input features to DataFrame
    df_input = pd.DataFrame([input_features])
    
    # Make prediction
    label = pipeline.predict(df_input)[0]
    
    # Try to get probability/risk score
    risk_score = None
    try:
        y_proba = pipeline.predict_proba(df_input)
        # For binary classification, use probability of positive class
        if y_proba.shape[1] == 2:
            risk_score = float(y_proba[0, 1])
        else:
            # For multiclass, use max probability
            risk_score = float(np.max(y_proba[0]))
    except (AttributeError, IndexError):
        # Model doesn't support predict_proba
        pass
    
    return {
        "label": int(label) if isinstance(label, (np.integer, np.floating)) else str(label),
        "risk_score": risk_score
    }

