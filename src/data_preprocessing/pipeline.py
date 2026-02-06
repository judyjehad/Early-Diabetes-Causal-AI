import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads a CSV file from the specified path.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    # Check if file exists before attempting to load
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    
    # Load the CSV file
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: Input DataFrame to clean
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Handle missing values in numeric columns using median
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_clean[col].isna().any():
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)
    
    # Handle missing values in non-numeric columns using mode or "unknown"
    non_numeric_columns = df_clean.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        if df_clean[col].isna().any():
            mode_value = df_clean[col].mode()
            if len(mode_value) > 0:
                df_clean[col].fillna(mode_value[0], inplace=True)
            else:
                df_clean[col].fillna("unknown", inplace=True)
    
    return df_clean


def split_data(df: pd.DataFrame, target_column: str):
    """
    Splits the DataFrame into training, validation, and test sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column to predict
        
    Returns:
        Tuple containing X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Check that target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # First split: 70% train, 30% temp (which will become val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Second split: Split temp into 15% validation and 15% test (50% of 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir: str):
    """
    Saves the processed data splits as CSV files in the output directory.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training target
        y_val: Validation target
        y_test: Test target
        output_dir: Directory path where CSV files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save feature sets
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    
    # Save target sets
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

