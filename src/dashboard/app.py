import sys
from pathlib import Path
import base64

# Add project root to sys.path to enable imports from src module
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import os
import json
from datetime import datetime
import pandas as pd

# Import modules
from src.models.prediction_model import predict_single
from src.models.causal_module import load_causal_graph, get_causal_effects
from src.recommendation.recommender import generate_recommendations
from src.reporting import build_assessment_pdf
from src.reporting.pdf_input_parser import extract_text_from_pdf, parse_patient_values, PYPDF_AVAILABLE

# Constants
DEFAULT_MODEL_PATH = "models/prediction_model.joblib"
DEFAULT_GRAPH_PATH = "models/causal_graph.joblib"
SCHEMA_PATH = "models/feature_schema.json"
NHANES_PATH = "data/raw/NHANES Blood Panel Dataset.csv"
PIMA_PATH = "data/raw/Pima Indians Diabetes Database.csv"
UCI_PATH = "data/raw/UCI Diabetes 130-US Hospitals Dataset.csv"

# Feature label mapping (UI-only display names - friendly labels hide raw codes)
FEATURE_LABELS = {
    "tc": "Total Cholesterol",
    "tcs": "Cholesterol / HDL Ratio",
    "lbdldl": "LDL Cholesterol",
    "lbxtr": "Triglycerides",
    "hdl": "HDL",
    "st": "ST",
    "dt": "DT",
    "Column1": "Record ID",
    "Age": "Age",
    "Glucose": "Glucose",
    "BMI": "BMI",
    "BloodPressure": "Blood Pressure"
}

# Page configuration
st.set_page_config(page_title="Early Diabetes Causal AI Assistant", layout="wide")

# Load feature schema
def load_feature_schema():
    """Load feature schema from JSON file."""
    if os.path.exists(SCHEMA_PATH):
        try:
            with open(SCHEMA_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

# System status check
def check_system_status():
    """Check what artifacts and datasets are available."""
    status = {
        "model": os.path.exists(DEFAULT_MODEL_PATH),
        "graph": os.path.exists(DEFAULT_GRAPH_PATH),
        "schema": os.path.exists(SCHEMA_PATH),
        "nhanes": os.path.exists(NHANES_PATH),
        "pima": os.path.exists(PIMA_PATH),
        "uci": os.path.exists(UCI_PATH)
    }
    return status

# PDF handling
def save_pdf(uploaded_file, save_path):
    """Save uploaded PDF file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def get_pdf_display_url(pdf_path):
    """Generate base64 data URL for PDF display."""
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    return f"data:application/pdf;base64,{base64_pdf}"

# Load schema
schema = load_feature_schema()

# Initialize form version for widget recreation
if "form_version" not in st.session_state:
    st.session_state["form_version"] = 0

# Sidebar
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    - Enter your health metrics in the form
    - Our AI model analyzes your risk factors
    - Causal analysis identifies key drivers
    - Personalized recommendations are generated
    """)
    
    st.divider()
    
    # Debug mode
    show_debug = st.checkbox("Show debug info", value=False, key="show_debug")
    
    st.divider()
    
    st.header("Configuration")
    model_path = st.text_input("Model Path", value=DEFAULT_MODEL_PATH)
    graph_path = st.text_input("Graph Path", value=DEFAULT_GRAPH_PATH)
    
    st.divider()
    
    # PDF Report section
    st.header("Project Report (PDF)")
    
    # Check if reports directory exists
    reports_dir = "reports" if os.path.exists("reports") else "data/processed"
    pdf_path = os.path.join(reports_dir, "project_report.pdf")
    
    uploaded_pdf = st.file_uploader("Upload PDF Report", type=["pdf"])
    if uploaded_pdf is not None:
        save_pdf(uploaded_pdf, pdf_path)
        st.success(f"PDF saved to {pdf_path}")
    
    # Extract inputs from PDF (in sidebar)
    st.subheader("Extract Inputs from PDF")
    uploaded_pdf_inputs = st.file_uploader("Upload Patient PDF", type=["pdf"], key="pdf_inputs_sidebar")
    if uploaded_pdf_inputs is not None:
        if not PYPDF_AVAILABLE:
            st.error("pypdf is required for PDF parsing. Install it with: pip install pypdf")
        else:
            try:
                pdf_bytes = uploaded_pdf_inputs.read()
                pdf_text = extract_text_from_pdf(pdf_bytes)
                
                if not pdf_text.strip():
                    st.warning("PDF appears to be empty or could not extract text.")
                else:
                    if schema and "features" in schema:
                        schema_features = schema.get("features", [])
                        parsed_values = parse_patient_values(pdf_text, schema_features)
                        
                        if parsed_values:
                            st.session_state["pdf_extracted_inputs"] = parsed_values
                            st.success(f"Extracted {len(parsed_values)} values. See Report tab to preview and apply.")
                        else:
                            st.warning("Could not extract any values from PDF.")
                    else:
                        st.warning("Feature schema not available.")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    # Show existing PDF if available
    if os.path.exists(pdf_path):
        st.divider()
        st.success("PDF report available")
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="project_report.pdf",
                mime="application/pdf"
            )

# Main content
st.title("Early Diabetes Causal AI Assistant")

# System Status Section
st.subheader("System Status")
status = check_system_status()

col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Models:**")
    st.write("✅ Model" if status["model"] else "❌ Model")
    st.write("✅ Graph" if status["graph"] else "❌ Graph")
    st.write("✅ Schema" if status["schema"] else "❌ Schema")

with col2:
    st.write("**Datasets:**")
    st.write("✅ NHANES" if status["nhanes"] else "❌ NHANES")
    st.write("✅ Pima" if status["pima"] else "❌ Pima")
    st.write("✅ UCI" if status["uci"] else "❌ UCI")

with col3:
    if not all([status["model"], status["graph"], status["schema"]]):
        st.warning("⚠️ Missing artifacts")
        st.code("""
python scripts/train_prediction.py
python scripts/train_causal.py
        """, language="bash")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Assessment", "Results", "Report"])

# Tab 1: Assessment
with tab1:
    st.header("Patient Information")
    
    # Show message if values are available from PDF extraction
    if "pdf_extracted_inputs" in st.session_state and st.session_state.get("pdf_extracted_inputs"):
        prefill_count = len(st.session_state.pdf_extracted_inputs)
        if prefill_count > 0:
            st.info(f"✓ {prefill_count} values available from PDF extraction. Form fields below are pre-filled. Please review and adjust as needed.")
    
    if not schema or "features" not in schema:
        st.warning("⚠️ Feature schema not found. Please train a model first.")
        # Fallback form
        with st.form("fallback_form"):
            age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=500.0, value=95.0, step=1.0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            blood_pressure = st.number_input("Blood Pressure (systolic, mmHg)", min_value=0.0, max_value=250.0, value=120.0, step=1.0)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=500.0, value=200.0, step=1.0)
            submitted = st.form_submit_button("Run Assessment")
            if submitted:
                patient_features = {
                    "age": age,
                    "glucose": glucose,
                    "bmi": bmi,
                    "blood_pressure": blood_pressure,
                    "cholesterol": cholesterol
                }
                st.session_state.patient_features = patient_features
                st.session_state.assessment_submitted = True
                st.rerun()
    else:
        # Full form with schema
        all_features = schema.get("features", [])
        feature_stats = schema.get("feature_stats", {})
        common_fields = ["Age", "Glucose", "BMI", "BloodPressure"]
        
        common_features = [f for f in all_features if f in common_fields]
        other_features = [f for f in all_features if f not in common_fields]
        
        # Apply pending inputs before creating widgets
        if "pending_inputs" in st.session_state and st.session_state["pending_inputs"]:
            pending = st.session_state["pending_inputs"]
            # Get schema features to validate
            schema_features = schema.get("features", []) if schema and "features" in schema else []
            for feature_key, value_info in pending.items():
                # Only process if feature is in schema
                if not schema_features or feature_key in schema_features:
                    # Extract the actual value
                    value = value_info.get("value") if isinstance(value_info, dict) else value_info
                    # Set the value in session_state (form_version change forces widget recreation)
                    # Widgets will read from session_state when created
                    st.session_state[feature_key] = value
            # Clear pending_inputs after applying
            del st.session_state["pending_inputs"]
        
        with st.form(key=f"patient_input_form_{st.session_state['form_version']}"):
            patient_features = {}
            
            # Core inputs (always visible)
            st.subheader("Core Inputs")
            for feature in common_features:
                stats = feature_stats.get(feature, {})
                default_value = stats.get("median", 0.0)
                min_val = stats.get("min", 0.0)
                max_val = stats.get("max", 1000.0)
                
                # Use value from session state if available (set by PDF apply), otherwise use median
                if feature in st.session_state:
                    default_value = st.session_state[feature]
                
                # Get display label (use friendly name)
                display_label = FEATURE_LABELS.get(feature, feature)
                
                if feature == "Age":
                    default_value = int(default_value) if default_value > 0 else 45
                    patient_features[feature] = st.number_input(
                        display_label, min_value=0, max_value=120, value=default_value, step=1, key=feature
                    )
                elif feature == "Glucose":
                    default_value = float(default_value) if default_value > 0 else 95.0
                    patient_features[feature] = st.number_input(
                        f"{display_label} (mg/dL)", min_value=0.0, max_value=500.0,
                        value=default_value, step=1.0, key=feature
                    )
                elif feature == "BMI":
                    default_value = float(default_value) if default_value > 0 else 25.0
                    patient_features[feature] = st.number_input(
                        display_label, min_value=10.0, max_value=50.0,
                        value=default_value, step=0.1, key=feature
                    )
                elif feature == "BloodPressure":
                    default_value = float(default_value) if default_value > 0 else 120.0
                    patient_features[feature] = st.number_input(
                        f"{display_label} (systolic, mmHg)", min_value=0.0, max_value=250.0,
                        value=default_value, step=1.0, key=feature
                    )
                else:
                    default_value = float(default_value) if default_value > 0 else 0.0
                    patient_features[feature] = st.number_input(
                        display_label, min_value=float(min_val), max_value=float(max_val),
                        value=default_value, step=0.1, key=feature
                    )
            
            # Advanced inputs (collapsed)
            if other_features:
                with st.expander("Advanced Lab Inputs", expanded=False):
                    for feature in other_features:
                        stats = feature_stats.get(feature, {})
                        default_value = stats.get("median", 0.0)
                        min_val = stats.get("min", 0.0)
                        max_val = stats.get("max", 1000.0)
                        
                        # Use value from session state if available (set by PDF apply), otherwise use median
                        if feature in st.session_state:
                            default_value = st.session_state[feature]
                        
                        default_value = float(default_value) if default_value > 0 else 0.0
                        
                        # Get display label (use friendly name)
                        display_label = FEATURE_LABELS.get(feature, feature)
                        patient_features[feature] = st.number_input(
                            display_label, min_value=float(min_val), max_value=float(max_val),
                            value=default_value, step=0.1, key=feature
                        )
            
            submitted = st.form_submit_button("Run Assessment")
            
            if submitted:
                # Ensure all schema features are present, fill missing with medians
                missing_features = []
                for feature in all_features:
                    if feature not in patient_features:
                        stats = feature_stats.get(feature, {})
                        median_val = stats.get("median", 0.0)
                        patient_features[feature] = float(median_val) if median_val > 0 else 0.0
                        missing_features.append(feature)
                
                if missing_features:
                    st.info(f"Using median values for: {', '.join(missing_features[:5])}")
                
                # Store in session state
                st.session_state.patient_features = patient_features
                st.session_state.assessment_submitted = True
                st.rerun()

# Tab 2: Results
with tab2:
    if "assessment_submitted" not in st.session_state or not st.session_state.assessment_submitted:
        st.info("Please complete the assessment first.")
    else:
        patient_features = st.session_state.patient_features
        
        # Ensure features are in schema order
        if schema and "features" in schema:
            ordered_features = {}
            all_features = schema.get("features", [])
            feature_stats = schema.get("feature_stats", {})
            
            for feature in all_features:
                if feature in patient_features:
                    ordered_features[feature] = patient_features[feature]
                else:
                    # Fill with median if missing
                    stats = feature_stats.get(feature, {})
                    median_val = stats.get("median", 0.0)
                    ordered_features[feature] = float(median_val) if median_val > 0 else 0.0
            
            patient_features = ordered_features
        
        st.header("Assessment Results")
        
        # Risk Assessment
        st.subheader("Risk Assessment")
        prediction = {"label": None, "risk_score": None}
        
        if os.path.exists(model_path):
            try:
                prediction = predict_single(model_path, patient_features)
            except Exception as e:
                st.warning(f"⚠️ Could not load prediction model: {str(e)}")
        else:
            st.warning(f"⚠️ Model file not found at: {model_path}")
        
        # Display prediction
        if prediction.get("risk_score") is not None:
            risk_score = prediction["risk_score"]
            st.metric("Risk Score", f"{risk_score:.3f}", delta=None)
            st.caption("Risk score ranges from 0 (low risk) to 1 (high risk)")
            
            if show_debug and prediction.get("label") is not None:
                st.write(f"**Predicted Label:** {prediction['label']}")
        else:
            st.info("Risk score not available. Model may not be loaded.")
        
        # Causal Analysis
        st.subheader("Causal Drivers")
        graph = load_causal_graph(graph_path)
        if graph is None:
            st.info(f"ℹ️ Causal graph not found. Using heuristic method.")
        
        causal_info = get_causal_effects(patient_features, graph)
        
        # Display causal drivers
        if causal_info.get("top_drivers"):
            for driver in causal_info["top_drivers"]:
                driver_feature = driver.get('feature', '')
                driver_display = FEATURE_LABELS.get(driver_feature, driver_feature)
                st.write(f"- **{driver_display}**: {driver['score']:.3f}")
        else:
            st.info("No causal drivers identified.")
        
        # Recommendations
        st.subheader("Recommendations")
        recommendations = generate_recommendations(
            patient_features,
            prediction.get("risk_score"),
            causal_info
        )
        
        # Store result in session_state for PDF generation
        st.session_state["last_result"] = {
            "timestamp": datetime.now().isoformat(),
            "risk_score": prediction.get("risk_score"),
            "predicted_label": prediction.get("label"),
            "severity": recommendations.get("severity", "UNKNOWN"),
            "causal_drivers": causal_info.get("top_drivers", [])[:5],
            "recommendations": {
                "summary": recommendations.get("summary", ""),
                "lifestyle": recommendations.get("lifestyle", []),
                "medication_discussion": recommendations.get("medication_discussion", []),
                "monitoring": recommendations.get("monitoring", []),
                "follow_up": recommendations.get("follow_up", ""),
                "focus_factors": recommendations.get("focus_factors", [])
            }
        }
        
        # Display severity and summary (no emoji)
        severity = recommendations.get("severity", "UNKNOWN")
        st.write(f"**Severity Level:** {severity}")
        st.write(recommendations.get("summary", ""))
        
        # Lifestyle recommendations
        if recommendations.get("lifestyle"):
            st.write("**Lifestyle:**")
            for item in recommendations["lifestyle"]:
                st.write(f"• {item}")
        
        # Medication discussion
        if recommendations.get("medication_discussion"):
            st.write("**Medication Discussion:**")
            for item in recommendations["medication_discussion"]:
                st.write(f"• {item}")
        
        # Monitoring
        if recommendations.get("monitoring"):
            st.write("**Monitoring:**")
            for item in recommendations["monitoring"]:
                st.write(f"• {item}")
        
        # Follow-up
        if recommendations.get("follow_up"):
            st.write(f"**{recommendations['follow_up']}**")
        
        # Focus factors (already friendly names from recommender)
        if recommendations.get("focus_factors"):
            st.write("**Focus Areas:**")
            st.write(", ".join(recommendations["focus_factors"]))
        
        # PDF Report Download
        st.divider()
        st.subheader("Download Report")
        if "last_result" in st.session_state:
            try:
                pdf_bytes = build_assessment_pdf(st.session_state["last_result"])
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name="diabetes_assessment_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                if show_debug:
                    st.exception(e)
        else:
            st.info("Run an assessment to enable PDF export.")
        
        # Debug info
        if show_debug:
            st.divider()
            st.subheader("Debug Information")
            st.write("**Raw Prediction:**")
            st.json(prediction)
            st.write("**Causal Info:**")
            st.json(causal_info)
            st.write("**Features Sent to Model:**")
            st.json(patient_features)
            if recommendations.get("debug"):
                st.write("**Internal Debug:**")
                st.json(recommendations["debug"])

# Tab 3: Report
with tab3:
    st.header("Project Report")
    
    # Extract Inputs from PDF section
    st.subheader("Extract Inputs from PDF")
    uploaded_pdf_tab3 = st.file_uploader("Upload Patient PDF to Extract Input Values", type=["pdf"], key="pdf_upload_tab3")
    
    # PDF parsing and preview
    if uploaded_pdf_tab3 is not None:
        if not PYPDF_AVAILABLE:
            st.error("pypdf is required for PDF parsing. Install it with: pip install pypdf")
        else:
            try:
                # Extract text
                pdf_bytes = uploaded_pdf_tab3.read()
                pdf_text = extract_text_from_pdf(pdf_bytes)
                
                if not pdf_text.strip():
                    st.warning("PDF appears to be empty or could not extract text.")
                else:
                    # Parse values
                    if schema and "features" in schema:
                        schema_features = schema.get("features", [])
                        parsed_values = parse_patient_values(pdf_text, schema_features)
                        
                        if parsed_values:
                            # Store in session state for preview
                            st.session_state["pdf_extracted_inputs"] = parsed_values
                            
                            # Show preview table
                            st.subheader("Extracted Values Preview")
                            
                            # Prepare table data
                            table_data = []
                            for feature_name, value_info in parsed_values.items():
                                display_label = FEATURE_LABELS.get(feature_name, feature_name)
                                value = value_info.get("value", "")
                                unit = value_info.get("unit", "")
                                notes = value_info.get("notes", "")
                                confidence = value_info.get("confidence", "low")
                                
                                unit_str = f" {unit}" if unit else ""
                                notes_str = f" ({notes})" if notes else ""
                                display_value = f"{value}{unit_str}{notes_str}".strip()
                                
                                table_data.append({
                                    "Field Name": display_label,
                                    "Extracted Value": display_value,
                                    "Unit/Notes": f"{unit}{' - ' + notes if notes else ''}".strip() or "-",
                                    "Confidence": confidence.title()
                                })
                            
                            # Display as table
                            df_preview = pd.DataFrame(table_data)
                            st.dataframe(df_preview, use_container_width=True, hide_index=True)
                            
                            # Determine missing features
                            if schema_features:
                                missing = [f for f in schema_features if f not in parsed_values]
                                if missing:
                                    missing_labels = [FEATURE_LABELS.get(f, f) for f in missing[:10]]
                                    st.warning(f"Could not extract values for: {', '.join(missing_labels)}")
                            
                            # Apply button
                            if st.button("Apply Extracted Inputs", key="apply_pdf_inputs", type="primary"):
                                # Filter and prepare pending inputs
                                pending_inputs = {}
                                applied_keys = []
                                skipped_keys = []
                                
                                # Get debug setting
                                show_debug_flag = st.session_state.get("show_debug", False)
                                
                                for feature_key, value_info in parsed_values.items():
                                    # Check if feature is in schema
                                    if schema_features and feature_key not in schema_features:
                                        skipped_keys.append(feature_key)
                                        continue
                                    
                                    # Store in pending_inputs (will be applied before widget creation)
                                    pending_inputs[feature_key] = value_info
                                    applied_keys.append(feature_key)
                                
                                # Store in pending_inputs buffer and increment form_version to force widget recreation
                                st.session_state["pending_inputs"] = pending_inputs
                                st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1
                                
                                # Show debug info
                                if show_debug_flag:
                                    st.write("**Debug - Extracted Dict:**")
                                    st.json(parsed_values)
                                    st.write("**Debug - Pending Inputs:**")
                                    st.json({k: (v.get("value") if isinstance(v, dict) else v) for k, v in pending_inputs.items()})
                                    st.write(f"**Keys Applied:** {applied_keys}")
                                    if skipped_keys:
                                        st.write(f"**Keys Skipped (not in schema):** {skipped_keys}")
                                
                                st.success("Inputs applied to assessment form.")
                                st.rerun()
                        else:
                            st.warning("Could not extract any values from PDF. Please check the PDF format.")
                    else:
                        st.warning("Feature schema not available. Please train a model first.")
                        
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    # Show existing extracted values if available
    if "pdf_extracted_inputs" in st.session_state and st.session_state.get("pdf_extracted_inputs"):
        st.divider()
        st.subheader("Currently Extracted Values")
        parsed_values = st.session_state["pdf_extracted_inputs"]
        
        table_data = []
        for feature_name, value_info in parsed_values.items():
            display_label = FEATURE_LABELS.get(feature_name, feature_name)
            value = value_info.get("value", "") if isinstance(value_info, dict) else value_info
            unit = value_info.get("unit", "") if isinstance(value_info, dict) else ""
            notes = value_info.get("notes", "") if isinstance(value_info, dict) else ""
            confidence = value_info.get("confidence", "low") if isinstance(value_info, dict) else "low"
            
            unit_str = f" {unit}" if unit else ""
            notes_str = f" ({notes})" if notes else ""
            display_value = f"{value}{unit_str}{notes_str}".strip()
            
            table_data.append({
                "Field Name": display_label,
                "Extracted Value": display_value,
                "Unit/Notes": f"{unit}{' - ' + notes if notes else ''}".strip() or "-",
                "Confidence": confidence.title()
            })
        
        df_preview = pd.DataFrame(table_data)
        st.dataframe(df_preview, use_container_width=True, hide_index=True)
        
        if st.button("Apply Extracted Inputs", key="apply_existing_pdf_inputs", type="primary"):
            # Filter and prepare pending inputs
            pending_inputs = {}
            applied_keys = []
            skipped_keys = []
            
            # Get schema features for validation
            schema_features = schema.get("features", []) if schema and "features" in schema else None
            
            # Get debug setting
            show_debug_flag = st.session_state.get("show_debug", False)
            
            for feature_key, value_info in parsed_values.items():
                # Check if feature is in schema
                if schema_features and feature_key not in schema_features:
                    skipped_keys.append(feature_key)
                    continue
                
                # Store in pending_inputs (will be applied before widget creation)
                pending_inputs[feature_key] = value_info
                applied_keys.append(feature_key)
            
            # Store in pending_inputs buffer and increment form_version to force widget recreation
            st.session_state["pending_inputs"] = pending_inputs
            st.session_state["form_version"] = st.session_state.get("form_version", 0) + 1
            
            # Show debug info
            if show_debug_flag:
                st.write("**Debug - Extracted Dict:**")
                st.json(parsed_values)
                st.write("**Debug - Pending Inputs:**")
                st.json({k: (v.get("value") if isinstance(v, dict) else v) for k, v in pending_inputs.items()})
                st.write(f"**Keys Applied:** {applied_keys}")
                if skipped_keys:
                    st.write(f"**Keys Skipped (not in schema):** {skipped_keys}")
            
            st.success("Inputs applied to assessment form.")
            st.rerun()
    
    # Existing PDF display
    pdf_path = os.path.join("reports" if os.path.exists("reports") else "data/processed", "project_report.pdf")
    
    if os.path.exists(pdf_path):
        st.divider()
        st.subheader("View Project Report")
        # Display PDF using embed
        pdf_display_url = get_pdf_display_url(pdf_path)
        pdf_display_html = f"""
        <embed src="{pdf_display_url}" width="100%" height="800px" type="application/pdf">
        """
        st.components.v1.html(pdf_display_html, height=800)
        
        # Download button
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="project_report.pdf",
                mime="application/pdf"
            )
    elif uploaded_pdf_tab3 is None:
        st.info("No PDF report available. Upload one above or in the sidebar.")
