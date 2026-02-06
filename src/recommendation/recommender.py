from .treatment_patterns import get_treatment_recommendations_by_risk, TREATMENT_PATTERNS


# Feature name mapping (raw codes to friendly names)
FEATURE_NAME_MAP = {
    "tc": "Total Cholesterol",
    "tcs": "Cholesterol / HDL Ratio",
    "lbxtr": "Triglycerides",
    "lbdldl": "LDL Cholesterol",
    "st": "ST",
    "dt": "DT",
    "Age": "Age",
    "Glucose": "Glucose",
    "BMI": "BMI",
    "BloodPressure": "Blood Pressure"
}


def get_friendly_feature_name(feature: str) -> str:
    """
    Converts a raw feature code to a friendly human-readable name.
    
    Args:
        feature: Raw feature code (e.g., "tc", "tcs")
        
    Returns:
        Friendly name (e.g., "Total Cholesterol") or title-case version if unknown
    """
    if feature in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[feature]
    # If unknown, return title-case version
    return feature.replace("_", " ").title()


def clean_recommendations(recommendations: list) -> list:
    """
    Post-process recommendations to remove raw UCI dataset keys and format for user-friendly display.
    
    Note: UCI dataset features (medical_specialty, num_medications, metformin, etc.) are used internally
    for pattern extraction but should not appear in the final UI output.
    
    Args:
        recommendations: List of raw recommendation strings
        
    Returns:
        List of cleaned, user-friendly recommendation strings
    """
    cleaned = []
    has_metformin_rec = False
    has_glucose_lowering_rec = False
    
    # Keywords/patterns to exclude
    exclude_keywords = ["medical_specialty", "num_medications"]
    exclude_patterns = [": ?", ": UNKNOWN", ": unknown"]
    
    # Common medication column names from UCI dataset
    medication_keywords = ["metformin", "insulin", "glipizide", "glyburide", "pioglitazone", 
                          "rosiglitazone", "acarbose", "miglitol", "tolazamide", "glimepiride", 
                          "acetohexamide", "chlorpropamide", "tolbutamide", "troglitazone"]
    
    for rec in recommendations:
        # Skip empty or None
        if not rec or not isinstance(rec, str):
            continue
        
        rec_lower = rec.lower()
        
        # Skip if contains excluded keywords
        if any(keyword in rec_lower for keyword in exclude_keywords):
            continue
        
        # Skip if contains unknown/placeholder values
        if any(pattern in rec for pattern in exclude_patterns):
            continue
        
        # Handle raw medication key-value pairs (e.g., "metformin: No", "insulin: Down")
        if ":" in rec:
            parts = rec.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().lower()
                value = parts[1].strip()
                
                # Skip raw medication column key-value pairs
                if any(med_keyword in key for med_keyword in medication_keywords):
                    # Track if we should add a medication recommendation
                    if "metformin" in key and not has_metformin_rec:
                        has_metformin_rec = True
                    continue
        
        # Replace metformin mentions with clean phrasing
        if "metformin" in rec_lower and not has_metformin_rec:
            # Skip the raw recommendation, we'll add a clean one
            if ":" in rec or "consider" in rec_lower:
                has_metformin_rec = True
                continue
        
        # Handle "Consider" medication recommendations with raw values
        if rec_lower.startswith("consider ") and "consult clinician" in rec_lower:
            # Check if it's about medications (but not already clean)
            if any(med_keyword in rec_lower for med_keyword in medication_keywords):
                # Skip raw medication recommendations
                if ":" in rec:
                    if "metformin" in rec_lower and not has_metformin_rec:
                        has_metformin_rec = True
                    elif not has_glucose_lowering_rec:
                        has_glucose_lowering_rec = True
                    continue
        
        # Format monitoring recommendations consistently
        if rec_lower.startswith("monitor:") or rec_lower.startswith("monitoring:"):
            # Extract the monitoring item
            monitoring_item = rec.split(":", 1)[1].strip() if ":" in rec else rec.replace("Monitoring", "").replace("monitoring", "").strip()
            if monitoring_item:
                cleaned.append(f"Monitoring: {monitoring_item}")
            continue
        
        # Format follow-up frequency consistently
        if "follow-up frequency:" in rec_lower or "follow up frequency:" in rec_lower:
            frequency = rec.split(":", 1)[1].strip() if ":" in rec else ""
            if frequency:
                # Map to consistent phrasing
                freq_lower = frequency.lower()
                if "quarterly" in freq_lower or "every 3 months" in freq_lower:
                    cleaned.append("Follow-up: Quarterly (every 3 months)")
                elif "semi-annual" in freq_lower or "semi annual" in freq_lower or "every 6 months" in freq_lower:
                    cleaned.append("Follow-up: Semi-annually (every 6 months)")
                elif "annual" in freq_lower or "yearly" in freq_lower:
                    cleaned.append("Follow-up: Annually")
                else:
                    cleaned.append(f"Follow-up: {frequency}")
            continue
        
        # Keep other recommendations as-is (lifestyle, etc.)
        cleaned.append(rec)
    
    # Add medication recommendations if they were filtered out but should be included
    if has_metformin_rec and not any("first-line medication" in r.lower() for r in cleaned):
        cleaned.append("Discuss first-line medication options (e.g., metformin) with a clinician.")
    
    if has_glucose_lowering_rec and not any("additional glucose-lowering" in r.lower() for r in cleaned):
        cleaned.append("Discuss additional glucose-lowering medications with a clinician if needed.")
    
    return cleaned


def map_risk_to_severity(risk_score: float) -> str:
    """
    Maps a risk score to a severity level.
    
    Args:
        risk_score: Risk score between 0 and 1, or None
        
    Returns:
        Severity level: "LOW", "MODERATE", "HIGH", or "UNKNOWN" (uppercase, no emoji)
    """
    if risk_score is None:
        return "UNKNOWN"
    
    if risk_score < 0.33:
        return "LOW"
    elif risk_score < 0.66:
        return "MODERATE"
    else:
        return "HIGH"


def generate_recommendations(patient_features: dict, risk_score: float, causal_info: dict) -> dict:
    """
    Generates personalized recommendations based on patient features, risk score, and causal analysis.
    
    Returns structured output with clean, limited recommendations:
    {
        "severity": str,  # "LOW", "MODERATE", "HIGH", "UNKNOWN"
        "summary": str,  # 1 sentence max
        "lifestyle": list[str],  # max 2 items
        "medication_discussion": list[str],  # max 1 item
        "monitoring": list[str],  # max 2 items
        "follow_up": str,  # exactly 1 sentence
        "focus_factors": list[str],  # max 2 items, human-readable names
        "debug": dict | None
    }
    
    Args:
        patient_features: Dictionary of patient feature names and values
        risk_score: Risk score between 0 and 1 (can be None)
        causal_info: Dictionary from causal module with top_drivers and method
        
    Returns:
        Structured dictionary with cleaned recommendations
    """
    # Map risk to severity (uppercase, no emoji)
    severity = map_risk_to_severity(risk_score)
    
    # Initialize category lists
    lifestyle = []
    medication_discussion = []
    monitoring = []
    focus_factors = []
    
    # Debug info (internal)
    debug_info = {
        "causal_method": causal_info.get("method") if causal_info else None,
        "raw_drivers": causal_info.get("top_drivers", []) if causal_info else []
    }
    
    # Core lifestyle recommendations (always include, max 2)
    lifestyle_core = [
        "Maintain a balanced diet rich in whole grains, vegetables, and lean proteins",
        "Engage in regular physical activity (at least 150 minutes per week)"
    ]
    lifestyle.extend(lifestyle_core[:2])
    
    # Feature-based recommendations
    # Glucose-related
    glucose_keywords = ["glucose", "blood_glucose", "glucose_level", "Glucose"]
    for key in glucose_keywords:
        if key in patient_features:
            try:
                glucose_value = float(patient_features[key])
                if glucose_value > 100:
                    if len(lifestyle) < 2:
                        lifestyle.append("Reduce added sugars and refined carbohydrates in your diet")
                    if len(monitoring) < 2:
                        monitoring.append("Blood glucose levels")
                    break
            except (ValueError, TypeError):
                pass
    
    # BMI-related
    bmi_keywords = ["bmi", "body_mass_index", "BMI"]
    for key in bmi_keywords:
        if key in patient_features:
            try:
                bmi_value = float(patient_features[key])
                if bmi_value >= 25:
                    if len(lifestyle) < 2:
                        lifestyle.append("Consider a weight management plan with a healthcare provider")
                    break
            except (ValueError, TypeError):
                pass
    
    # Blood pressure-related
    bp_keywords = ["blood_pressure", "systolic", "diastolic", "bp", "BloodPressure"]
    for key in bp_keywords:
        if key in patient_features:
            try:
                bp_value = float(patient_features[key])
                if bp_value > 120:
                    if len(monitoring) < 2:
                        monitoring.append("Blood pressure")
                    break
            except (ValueError, TypeError):
                pass
    
    # Cholesterol/triglycerides-related
    lipid_keywords = ["cholesterol", "triglycerides", "ldl", "hdl", "total_cholesterol", "tc", "lbdldl", "lbxtr"]
    for key in lipid_keywords:
        if key in patient_features:
            try:
                lipid_value = float(patient_features[key])
                if lipid_value > 200 or (key.lower() in ["ldl", "ldl_cholesterol", "lbdldl"] and lipid_value > 100):
                    if len(monitoring) < 2:
                        monitoring.append("Lipid profile")
                    break
            except (ValueError, TypeError):
                pass
    
    # Add treatment recommendations from UCI dataset patterns
    if TREATMENT_PATTERNS and causal_info and "top_drivers" in causal_info:
        treatment_recs = get_treatment_recommendations_by_risk(
            severity.lower(),
            causal_info["top_drivers"],
            TREATMENT_PATTERNS
        )
        # Process treatment recommendations into categories
        for rec in treatment_recs:
            rec_lower = rec.lower()
            if "monitor" in rec_lower or "monitoring" in rec_lower:
                # Extract monitoring item
                if ":" in rec:
                    item = rec.split(":", 1)[1].strip()
                    if item and len(monitoring) < 2:
                        monitoring.append(item)
            elif "medication" in rec_lower or "metformin" in rec_lower or any(med in rec_lower for med in ["first-line", "glucose-lowering"]):
                # Medication discussion (max 1)
                if len(medication_discussion) < 1:
                    if "first-line" in rec_lower or "metformin" in rec_lower:
                        medication_discussion.append("Discuss first-line medication options (e.g., metformin) with a clinician.")
                    elif "additional" in rec_lower or "glucose-lowering" in rec_lower:
                        medication_discussion.append("Discuss additional glucose-lowering medications with a clinician if needed.")
    
    # Extract focus factors from causal drivers (max 2, convert to friendly names)
    if causal_info and "top_drivers" in causal_info:
        driver_features = [driver.get("feature", "") for driver in causal_info["top_drivers"][:2]]
        focus_factors = [get_friendly_feature_name(f) for f in driver_features if f][:2]
    
    # Build summary (1 sentence max, use friendly names)
    if severity == "UNKNOWN":
        summary = "Risk assessment incomplete with general lifestyle recommendations provided."
    elif severity == "LOW":
        if focus_factors:
            focus_str = ", ".join(focus_factors[:2])
            summary = f"Current risk level appears low with focus areas: {focus_str}."
        else:
            summary = "Current risk level appears low with continued healthy lifestyle habits recommended."
    elif severity == "MODERATE":
        if focus_factors:
            focus_str = ", ".join(focus_factors[:2])
            summary = f"Moderate risk level detected with key factors {focus_str}, recommending enhanced monitoring and clinician consultation."
        else:
            summary = "Moderate risk level detected, recommending enhanced monitoring and consultation with a healthcare provider."
    else:  # HIGH
        if focus_factors:
            focus_str = ", ".join(focus_factors[:2])
            summary = f"Elevated risk level identified with primary concerns {focus_str}, recommending clinical consultation for further evaluation."
        else:
            summary = "Elevated risk level identified, recommending clinical consultation for further evaluation."
    
    # Determine follow-up frequency based on severity (exactly 1 sentence)
    if severity == "HIGH":
        follow_up = "Follow-up: Quarterly (every 3 months)"
    elif severity == "MODERATE":
        follow_up = "Follow-up: Semi-annually (every 6 months)"
    else:
        follow_up = "Follow-up: Annually"
    
    # Limit counts and remove duplicates
    lifestyle = list(dict.fromkeys(lifestyle))[:2]  # Max 2
    medication_discussion = list(dict.fromkeys(medication_discussion))[:1]  # Max 1
    monitoring = list(dict.fromkeys(monitoring))[:2]  # Max 2
    focus_factors = focus_factors[:2]  # Max 2
    
    # Filter out raw UCI keys from all lists
    exclude_keywords = ["medical_specialty", "num_medications"]
    exclude_patterns = [": ?", ": UNKNOWN", ": unknown"]
    medication_keywords = ["metformin:", "insulin:", "glipizide:", "glyburide:"]
    
    def clean_item(item):
        """Remove raw UCI patterns from a single item."""
        if not item or not isinstance(item, str):
            return None
        item_lower = item.lower()
        if any(kw in item_lower for kw in exclude_keywords):
            return None
        if any(pattern in item for pattern in exclude_patterns):
            return None
        if any(med_kw in item_lower for med_kw in medication_keywords):
            return None
        if ":" in item and len(item.split(":")) == 2:
            # Might be a raw key-value pair
            key = item.split(":")[0].strip().lower()
            if any(med in key for med in ["metformin", "insulin", "glipizide", "glyburide"]):
                return None
        return item
    
    lifestyle = [clean_item(item) for item in lifestyle if clean_item(item)]
    medication_discussion = [clean_item(item) for item in medication_discussion if clean_item(item)]
    monitoring = [clean_item(item) for item in monitoring if clean_item(item)]
    
    return {
        "severity": severity,
        "summary": summary,
        "lifestyle": lifestyle,
        "medication_discussion": medication_discussion,
        "monitoring": monitoring,
        "follow_up": follow_up,
        "focus_factors": focus_factors,
        "debug": debug_info
    }
