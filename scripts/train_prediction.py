"""
Training script for prediction model.
Combines NHANES and Pima datasets, trains XGBoost/RandomForest/LogReg, saves model and feature schema.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing.pipeline import load_raw_data, clean_data
from src.models.prediction_model import train_prediction_model

# Mac compatibility check for XGBoost
def check_xgboost_openmp():
    """Check and warn about XGBoost OpenMP issues on Mac."""
    import platform
    if platform.system() == "Darwin":  # macOS
        try:
            import xgboost as xgb
            # Try a simple operation to detect OpenMP issues
            test_data = np.array([[1, 2], [3, 4]])
            test_label = np.array([0, 1])
            try:
                model = xgb.XGBClassifier(n_estimators=1, random_state=42)
                model.fit(test_data, test_label)
            except Exception as e:
                if "omp" in str(e).lower() or "openmp" in str(e).lower():
                    print("⚠️  WARNING: XGBoost OpenMP issue detected on macOS.")
                    print("   Install libomp: brew install libomp")
                    print("   Falling back to RandomForest...")
                    return False
        except ImportError:
            pass
    return True


def prepare_combined_dataset(nhanes_path: str, pima_path: str, target_column: str = "diabetes"):
    """
    Load and combine NHANES and Pima datasets.
    
    Args:
        nhanes_path: Path to NHANES CSV
        pima_path: Path to Pima CSV
        target_column: Name of target column
        
    Returns:
        Combined and cleaned DataFrame
    """
    datasets = []
    
    # Load NHANES if available
    if os.path.exists(nhanes_path):
        print(f"Loading NHANES dataset from {nhanes_path}...")
        df_nhanes = load_raw_data(nhanes_path)
        df_nhanes = clean_data(df_nhanes)
        
        # Rename NHANES columns to match Pima for overlaps
        column_mapping = {}
        if "luxcapm" in df_nhanes.columns:
            column_mapping["luxcapm"] = "Glucose"
            print(f"  Renaming 'luxcapm' -> 'Glucose'")
        if "lbdinsi" in df_nhanes.columns:
            column_mapping["lbdinsi"] = "Insulin"
            print(f"  Renaming 'lbdinsi' -> 'Insulin'")
        
        if column_mapping:
            df_nhanes = df_nhanes.rename(columns=column_mapping)
        
        # Create diabetes label from Glucose if it exists (after renaming)
        if target_column not in df_nhanes.columns:
            if "Glucose" in df_nhanes.columns:
                print(f"  Creating {target_column} label from Glucose (threshold >= 126)...")
                df_nhanes[target_column] = (df_nhanes["Glucose"] >= 126).astype(int)
                print(f"  Created {target_column}: {df_nhanes[target_column].sum()} positive cases out of {len(df_nhanes)}")
            else:
                print(f"  ⚠️  WARNING: NHANES dataset has no '{target_column}' label and no 'Glucose' column.")
                print(f"     Skipping NHANES dataset. Available columns: {list(df_nhanes.columns)[:10]}...")
        
        # Only add if target column exists
        if target_column in df_nhanes.columns:
            datasets.append(df_nhanes)
            print(f"  NHANES: {len(df_nhanes)} samples, {df_nhanes.shape[1]} features")
    
    # Load Pima if available
    if os.path.exists(pima_path):
        print(f"Loading Pima dataset from {pima_path}...")
        df_pima = load_raw_data(pima_path)
        df_pima = clean_data(df_pima)
        
        # Rename Outcome to diabetes if it exists
        if target_column not in df_pima.columns:
            if "Outcome" in df_pima.columns:
                print(f"  Renaming 'Outcome' to '{target_column}'...")
                df_pima[target_column] = df_pima["Outcome"]
                # Drop the old column to avoid duplicates
                df_pima = df_pima.drop(columns=["Outcome"], errors='ignore')
            else:
                print(f"  ⚠️  WARNING: Pima dataset has no '{target_column}' label and no 'Outcome' column.")
                print(f"     Skipping Pima dataset. Available columns: {list(df_pima.columns)[:10]}...")
        
        # Only add if target column exists
        if target_column in df_pima.columns:
            datasets.append(df_pima)
            print(f"  Pima: {len(df_pima)} samples, {df_pima.shape[1]} features")
    
    if not datasets:
        raise FileNotFoundError(f"Neither dataset found or neither has a valid label column. Expected at least one of: {nhanes_path}, {pima_path}")
    
    # Combine datasets using union of columns (not intersection)
    if len(datasets) > 1:
        print(f"  Combining datasets using union of columns...")
        # Use pd.concat with union - missing values will be NaN
        combined = pd.concat(datasets, axis=0, sort=False, ignore_index=True)
    else:
        combined = datasets[0]
    
    # Ensure target column exists
    if target_column not in combined.columns:
        raise ValueError(f"Target column '{target_column}' not found in combined dataset. Available columns: {list(combined.columns)}")
    
    # Clean combined dataset
    combined = clean_data(combined)
    
    # Check that we have features (numeric columns except target)
    X = combined.drop(columns=[target_column])
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) == 0:
        raise ValueError(
            f"Feature mapping failed: Found 0 numeric features after combining datasets.\n"
            f"Target column: {target_column}\n"
            f"All columns: {list(combined.columns)}\n"
            f"Numeric columns in X: {list(X.select_dtypes(include=[np.number]).columns)}\n"
            f"Please check column names and ensure datasets have numeric features."
        )
    
    print(f"Combined dataset: {len(combined)} samples, {combined.shape[1]} features")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Target distribution: {combined[target_column].value_counts().to_dict()}")
    return combined


def create_feature_schema(df: pd.DataFrame, target_column: str, output_path: str):
    """
    Create and save feature schema JSON.
    
    Args:
        df: DataFrame with features
        target_column: Name of target column
        output_path: Path to save schema JSON
    """
    X = df.drop(columns=[target_column])
    
    schema = {
        "features": list(X.columns),
        "numeric_features": X.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_features": X.select_dtypes(exclude=[np.number]).columns.tolist(),
        "target_column": target_column,
        "feature_stats": {}
    }
    
    # Add basic stats for numeric features
    for col in schema["numeric_features"]:
        schema["feature_stats"][col] = {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "median": float(X[col].median())
        }
    
    # Save schema
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"Feature schema saved to {output_path}")


def main():
    """Main training function."""
    # Paths
    nhanes_path = "data/raw/NHANES Blood Panel Dataset.csv"
    pima_path = "data/raw/Pima Indians Diabetes Database.csv"
    model_output = "models/prediction_model.joblib"
    schema_output = "models/feature_schema.json"
    target_column = "diabetes"
    
    # Check XGBoost on Mac
    xgb_available = check_xgboost_openmp()
    
    print("=" * 60)
    print("Training Prediction Model")
    print("=" * 60)
    
    # Prepare dataset
    try:
        df = prepare_combined_dataset(nhanes_path, pima_path, target_column)
    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")
        return
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train model
    print("\nTraining model...")
    try:
        # Save to temp CSV since train_prediction_model expects CSV path
        temp_train_path = "data/processed/temp_train.csv"
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(temp_train_path, index=False)
        
        result = train_prediction_model(
            train_csv_path=temp_train_path,
            target_column=target_column,
            model_output_path=model_output
        )
        
        # Clean up temp file
        if os.path.exists(temp_train_path):
            os.remove(temp_train_path)
        
        print(f"✅ Model trained successfully!")
        print(f"   Model type: {result['model_type']}")
        print(f"   Samples: {result['n_samples']}")
        print(f"   Features: {result['n_features']}")
        print(f"   Saved to: {model_output}")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create feature schema
    print("\nCreating feature schema...")
    try:
        create_feature_schema(df, target_column, schema_output)
        print(f"✅ Feature schema saved to {schema_output}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create feature schema: {e}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

