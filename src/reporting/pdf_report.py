"""
PDF Report Generation Module

Generates PDF reports summarizing diabetes risk assessment results.
"""
from datetime import datetime
from io import BytesIO
from typing import Dict, Any

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
except ImportError:
    raise ImportError("reportlab is required. Install it with: pip install reportlab")


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


def build_assessment_pdf(report_data: Dict[str, Any]) -> bytes:
    """
    Builds a PDF report from assessment results.
    
    Args:
        report_data: Dictionary containing:
            - timestamp: str (ISO format datetime)
            - risk_score: float (0-1)
            - predicted_label: int or str (optional)
            - severity: str (LOW/MODERATE/HIGH/UNKNOWN - uppercase, no emoji)
            - causal_drivers: list[dict] with 'feature' and 'score' keys (max 5)
            - recommendations: dict with keys:
                - summary: str
                - lifestyle: list[str]
                - medication_discussion: list[str]
                - monitoring: list[str]
                - follow_up: str
                - focus_factors: list[str] (already friendly names)
    
    Returns:
        bytes: PDF file as bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    elements.append(Paragraph("Diabetes Risk Assessment Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    timestamp = report_data.get('timestamp', datetime.now().isoformat())
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        timestamp_str = dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        timestamp_str = timestamp
    
    elements.append(Paragraph(f"<b>Report Generated:</b> {timestamp_str}", normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    elements.append(Paragraph("Risk Assessment", heading_style))
    
    risk_score = report_data.get('risk_score')
    severity = report_data.get('severity', 'UNKNOWN')  # Already uppercase, no emoji
    predicted_label = report_data.get('predicted_label')
    
    risk_data = [
        ['Risk Score:', f"{risk_score:.3f}" if risk_score is not None else "N/A"],
        ['Risk Level:', severity],
    ]
    
    if predicted_label is not None:
        risk_data.append(['Predicted Label:', str(predicted_label)])
    
    risk_table = Table(risk_data, colWidths=[2*inch, 4*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 0.3*inch))
    
    causal_drivers = report_data.get('causal_drivers', [])
    if causal_drivers:
        elements.append(Paragraph("Causal Drivers", heading_style))
        
        top_drivers = causal_drivers[:5]
        
        driver_data = [['Feature', 'Score']]
        for driver in top_drivers:
            feature = driver.get('feature', 'N/A')
            # Use friendly name instead of raw code
            feature_display = get_friendly_feature_name(feature)
            score = driver.get('score', 0)
            driver_data.append([feature_display, f"{score:.3f}"])
        
        driver_table = Table(driver_data, colWidths=[4*inch, 2*inch])
        driver_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        elements.append(driver_table)
        elements.append(Spacer(1, 0.3*inch))
    
    recommendations = report_data.get('recommendations', {})
    if recommendations:
        elements.append(Paragraph("Recommendations", heading_style))
        
        summary = recommendations.get('summary', '')
        if summary:
            elements.append(Paragraph(f"<b>Summary:</b> {summary}", normal_style))
            elements.append(Spacer(1, 0.2*inch))
        
        lifestyle = recommendations.get('lifestyle', [])
        if lifestyle:
            elements.append(Paragraph("<b>Lifestyle Modifications:</b>", normal_style))
            for item in lifestyle:
                elements.append(Paragraph(f"• {item}", normal_style))
            elements.append(Spacer(1, 0.15*inch))
        
        medication = recommendations.get('medication_discussion', [])
        if medication:
            elements.append(Paragraph("<b>Pharmacological Considerations:</b>", normal_style))
            for item in medication:
                elements.append(Paragraph(f"• {item}", normal_style))
            elements.append(Spacer(1, 0.15*inch))
        
        monitoring = recommendations.get('monitoring', [])
        if monitoring:
            elements.append(Paragraph("<b>Monitoring Parameters:</b>", normal_style))
            for item in monitoring:
                elements.append(Paragraph(f"• {item}", normal_style))
            elements.append(Spacer(1, 0.15*inch))
        
        follow_up = recommendations.get('follow_up', '')
        if follow_up:
            elements.append(Paragraph(f"<b>{follow_up}</b>", normal_style))
            elements.append(Spacer(1, 0.15*inch))
        
        focus_factors = recommendations.get('focus_factors', [])
        if focus_factors:
            # focus_factors already contains friendly names from recommender
            focus_str = ", ".join(focus_factors)
            elements.append(Paragraph(f"<b>Key Factors to Prioritize:</b> {focus_str}", normal_style))
    
    doc.build(elements)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
