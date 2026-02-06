"""
Treatment recommendation patterns extracted from UCI 130-US Hospitals dataset.
Maps risk levels and causal drivers to treatment/lab patterns.
"""
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path


def load_treatment_patterns(data_path: str = "data/raw/UCI Diabetes 130-US Hospitals Dataset.csv"):
    """
    Load and extract treatment patterns from UCI dataset.
    
    Args:
        data_path: Path to UCI dataset CSV
        
    Returns:
        Dictionary of treatment patterns by risk band
    """
    if not os.path.exists(data_path):
        return None
    
    try:
        df = pd.read_csv(data_path)
        
        # Common medication/lab columns in UCI dataset
        medication_cols = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['medication', 'drug', 'med', 'insulin', 'metformin'])]
        lab_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['lab', 'test', 'glucose', 'a1c', 'hba1c'])]
        
        # Outcome/readmission columns
        outcome_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['readmission', 'outcome', 'discharge', 'result'])]
        
        patterns = {
            "low_risk": {
                "medications": [],
                "labs": ["HbA1c monitoring", "Basic metabolic panel"],
                "frequency": "Annual"
            },
            "medium_risk": {
                "medications": ["Metformin consideration", "Lifestyle intervention"],
                "labs": ["HbA1c every 6 months", "Lipid panel", "Renal function"],
                "frequency": "Semi-annual"
            },
            "high_risk": {
                "medications": ["Metformin", "Additional glucose-lowering agents if needed"],
                "labs": ["HbA1c every 3 months", "Comprehensive metabolic panel", 
                        "Lipid panel", "Microalbuminuria screening"],
                "frequency": "Quarterly"
            }
        }
        
        # Extract actual patterns from data if columns exist
        if medication_cols and outcome_cols:
            # Group by outcome and extract common medications
            for outcome_col in outcome_cols[:1]:  # Use first outcome column
                if df[outcome_col].dtype in [np.number, 'int64', 'float64']:
                    # Define risk bands based on outcome
                    low_mask = df[outcome_col] <= df[outcome_col].quantile(0.33)
                    med_mask = (df[outcome_col] > df[outcome_col].quantile(0.33)) & \
                              (df[outcome_col] <= df[outcome_col].quantile(0.66))
                    high_mask = df[outcome_col] > df[outcome_col].quantile(0.66)
                    
                    # Extract common medications for each band
                    for mask, band in [(low_mask, "low_risk"), (med_mask, "medium_risk"), 
                                      (high_mask, "high_risk")]:
                        if mask.sum() > 0:
                            band_df = df[mask]
                            for med_col in medication_cols[:3]:  # Top 3 medications
                                if med_col in band_df.columns:
                                    common_meds = band_df[med_col].value_counts().head(2)
                                    if len(common_meds) > 0:
                                        patterns[band]["medications"].extend(
                                            [f"{med_col}: {idx}" for idx in common_meds.index[:1]]
                                        )
        
        return patterns
        
    except Exception as e:
        print(f"Warning: Could not load treatment patterns: {e}")
        return None


def get_treatment_recommendations_by_risk(risk_level: str, top_drivers: list, 
                                         treatment_patterns: dict = None) -> list:
    """
    Get treatment recommendations based on risk level and causal drivers.
    
    Args:
        risk_level: "low", "medium", or "high"
        top_drivers: List of top causal driver features
        treatment_patterns: Treatment patterns dictionary (optional)
        
    Returns:
        List of treatment recommendations
    """
    recommendations = []
    
    # Map risk level to pattern key
    risk_key = f"{risk_level}_risk"
    
    if treatment_patterns and risk_key in treatment_patterns:
        pattern = treatment_patterns[risk_key]
        
        # Add medication recommendations
        if pattern.get("medications"):
            for med in pattern["medications"]:
                recommendations.append(f"Consider {med} (consult clinician)")
        
        # Add lab monitoring
        if pattern.get("labs"):
            for lab in pattern["labs"]:
                recommendations.append(f"Monitor: {lab}")
        
        # Add frequency
        if pattern.get("frequency"):
            recommendations.append(f"Follow-up frequency: {pattern['frequency']}")
    
    # Add driver-specific recommendations
    if top_drivers:
        driver_names = [d.get("feature", d) if isinstance(d, dict) else d 
                       for d in top_drivers[:2]]
        if driver_names:
            recommendations.append(
                f"Focus on managing: {', '.join(driver_names)}"
            )
    
    return recommendations


# Load patterns at module level (will be None if dataset not available)
TREATMENT_PATTERNS = load_treatment_patterns()

