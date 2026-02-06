"""
PDF Input Parser Utility

Extracts patient values from PDF text for auto-filling the assessment form.
"""
import re
from io import BytesIO
from typing import Dict, Tuple, List, Optional

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


# PDF label to feature mapping
PDF_LABEL_TO_FEATURE = {
    "Age": "Age",
    "BMI": "BMI",
    "Blood Pressure": "BloodPressure",
    "BloodPressure": "BloodPressure",
    "Glucose": "Glucose",
    "Insulin": "Insulin",
    "Total Cholesterol": "tc",
    "HDL": "hdl",
    "LDL": "lbdldl",
    "LDL Cholesterol": "lbdldl",
    "Triglycerides": "lbxtr",
    "Cholesterol/HDL Ratio": "tcs",
    "Cholesterol / HDL Ratio": "tcs",
    "ST": "st",
    "dt": "dt",
    "DT": "dt",
    "lbxin": "lbxin",
    "luxsmed": "luxsmed",
    "Pregnancies": "Pregnancies",
    "Diabetes Pedigree Function": "DiabetesPedigreeFunction",
    "SkinThickness": "SkinThickness",
    "Column1": "Column1",
    "Record ID": "Column1"
}

# Feature units and notes for display
FEATURE_UNITS = {
    "Age": ("years", ""),
    "BMI": ("kg/m²", ""),
    "BloodPressure": ("mmHg", "systolic"),
    "Glucose": ("mg/dL", ""),
    "Insulin": ("μU/mL", ""),
    "tc": ("mg/dL", ""),
    "hdl": ("mg/dL", ""),
    "lbdldl": ("mg/dL", ""),
    "lbxtr": ("mg/dL", ""),
    "tcs": ("ratio", ""),
    "st": ("", ""),
    "dt": ("", ""),
    "lbxin": ("", ""),
    "luxsmed": ("", ""),
    "Pregnancies": ("count", ""),
    "DiabetesPedigreeFunction": ("", ""),
    "SkinThickness": ("mm", ""),
    "Column1": ("", "")
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes.
    
    Args:
        file_bytes: PDF file as bytes
        
    Returns:
        Extracted text as string
    """
    if not PYPDF_AVAILABLE:
        raise ImportError("pypdf is required. Install it with: pip install pypdf")
    
    try:
        pdf_file = BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def normalize_number(text: str) -> Optional[float]:
    """
    Normalize and parse a number from text (handles commas, decimals).
    
    Args:
        text: Text containing a number
        
    Returns:
        Parsed float value or None if invalid
    """
    if not text:
        return None
    
    # Remove commas and whitespace
    cleaned = text.replace(",", "").strip()
    
    # Try to extract first number (handles formats like "138/88" -> 138)
    number_match = re.search(r'(\d+\.?\d*)', cleaned)
    if number_match:
        try:
            return float(number_match.group(1))
        except ValueError:
            pass
    
    return None


def parse_blood_pressure(text: str) -> Optional[float]:
    """
    Parse blood pressure, extracting systolic value.
    Handles formats like "138/88", "138-88", "Systolic: 138".
    
    Args:
        text: Text containing blood pressure
        
    Returns:
        Systolic value as float or None
    """
    # Pattern: "138/88" or "138-88"
    bp_pattern = r'(\d+)\s*[/-]\s*\d+'
    match = re.search(bp_pattern, text)
    if match:
        return float(match.group(1))
    
    # Pattern: "Systolic: 138" or "BP: 138"
    systolic_pattern = r'(?:systolic|bp)[:\s]+(\d+)'
    match = re.search(systolic_pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Try to extract first number
    return normalize_number(text)


def parse_patient_values(text: str, schema_features: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Parse patient values from PDF text.
    
    Returns a dictionary mapping feature names to:
    {
        "value": float,
        "unit": str,
        "notes": str,
        "confidence": str  # "high", "medium", "low"
    }
    
    Args:
        text: Extracted PDF text
        schema_features: Optional list of expected feature names to filter results
        
    Returns:
        Dictionary mapping feature names to parsed value dictionaries
    """
    parsed = {}
    text_lower = text.lower()
    
    # Pattern 1: "Label (units): value" or "Label: value" or "Label value"
    for pdf_label, feature_name in PDF_LABEL_TO_FEATURE.items():
        if schema_features and feature_name not in schema_features:
            continue
        
        # Skip if already parsed
        if feature_name in parsed:
            continue
        
        value = None
        confidence = "low"
        escaped_label = re.escape(pdf_label)
        
        # Try patterns with units in parentheses
        patterns_with_units = [
            rf'{escaped_label}\s*\([^)]+\)\s*:?\s*([0-9,.\s/]+)',
        ]
        
        for pattern in patterns_with_units:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                if feature_name == "BloodPressure":
                    value = parse_blood_pressure(matches[0])
                else:
                    value = normalize_number(matches[0])
                if value is not None:
                    confidence = "high"
                    break
        
        # Try patterns with colon
        if value is None:
            patterns_colon = [
                rf'{escaped_label}\s*:\s*([0-9,.\s/]+)',
            ]
            for pattern in patterns_colon:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    if feature_name == "BloodPressure":
                        value = parse_blood_pressure(matches[0])
                    else:
                        value = normalize_number(matches[0])
                    if value is not None:
                        confidence = "high"
                        break
        
        # Try patterns with space
        if value is None:
            patterns_space = [
                rf'{escaped_label}\s+([0-9,.\s/]+)',
            ]
            for pattern in patterns_space:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    if feature_name == "BloodPressure":
                        value = parse_blood_pressure(matches[0])
                    else:
                        value = normalize_number(matches[0])
                    if value is not None:
                        confidence = "medium"
                        break
        
        if value is not None:
            unit, notes = FEATURE_UNITS.get(feature_name, ("", ""))
            parsed[feature_name] = {
                "value": value,
                "unit": unit,
                "notes": notes,
                "confidence": confidence
            }
    
    # Pattern 2: Table format "Feature | Value" or "Feature\tValue"
    table_pattern = r'([^\|\t\n]+?)[\|\t]+([0-9,.\s/]+)'
    table_matches = re.findall(table_pattern, text, re.MULTILINE)
    for label, value_str in table_matches:
        label_clean = label.strip()
        for pdf_label, feature_name in PDF_LABEL_TO_FEATURE.items():
            if schema_features and feature_name not in schema_features:
                continue
            if feature_name in parsed:
                continue
            
            # Check if label matches (case-insensitive)
            if pdf_label.lower() in label_clean.lower() or label_clean.lower() in pdf_label.lower():
                if feature_name == "BloodPressure":
                    value = parse_blood_pressure(value_str)
                else:
                    value = normalize_number(value_str)
                
                if value is not None:
                    unit, notes = FEATURE_UNITS.get(feature_name, ("", ""))
                    parsed[feature_name] = {
                        "value": value,
                        "unit": unit,
                        "notes": notes,
                        "confidence": "high"  # Table format is usually reliable
                    }
                    break
    
    return parsed
