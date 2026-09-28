# utils/sustainability_report.py
"""
Sustainability Report Generator
Creates downloadable PDF reports with CO2 metrics and relatable comparisons
"""

import io
from datetime import datetime
from typing import Dict, List
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from utils import green_metrics


def generate_sustainability_certificate(session_data: Dict) -> bytes:
    """
    Generate a PDF sustainability certificate
    
    Args:
        session_data: Dictionary containing:
            - total_co2_g: Total CO2 emissions in grams
            - queries_executed: Number of queries run
            - avg_green_score: Average green score
            - best_query: Best performing query data
            - worst_query: Worst performing query data
            - session_duration_s: Total session duration
    
    Returns:
        PDF file as bytes
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#00FF9F'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#00FFFF'),
        spaceAfter=12
    )
    
    normal_style = styles['Normal']
    
    # Title
    title = Paragraph("🌱 Green AI & DB Sustainability Certificate", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Date
    date_text = Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style)
    elements.append(date_text)
    elements.append(Spacer(1, 0.3*inch))
    
    # Session Summary
    summary_heading = Paragraph("Session Summary", heading_style)
    elements.append(summary_heading)
    
    total_co2 = session_data.get('total_co2_g', 0)
    queries_count = session_data.get('queries_executed', 0)
    avg_score = session_data.get('avg_green_score', 0)
    duration_s = session_data.get('session_duration_s', 0)
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Queries Executed', str(queries_count)],
        ['Total CO₂ Emissions', f"{total_co2:.6f} g"],
        ['Average Green Score', f"{avg_score:.1f}/100"],
        ['Session Duration', f"{duration_s:.1f} seconds"]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b2b2b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Relatable Metrics
    relatable_heading = Paragraph("Environmental Impact (Relatable Metrics)", heading_style)
    elements.append(relatable_heading)
    
    smartphones = green_metrics.co2_to_smartphones(total_co2)
    car_meters = green_metrics.co2_to_car_km(total_co2) * 1000
    
    relatable_data = [
        ['Comparison', 'Equivalent'],
        ['Smartphones Charged', f"{smartphones:.4f} devices"],
        ['Car Distance', f"{car_meters:.2f} meters"],
        ['LED Lightbulb Hours', f"{total_co2 * 100:.2f} hours (est.)"]
    ]
    
    relatable_table = Table(relatable_data, colWidths=[3*inch, 2*inch])
    relatable_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00FF9F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(relatable_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Best Query
    if session_data.get('best_query'):
        best_heading = Paragraph("🏆 Most Sustainable Query", heading_style)
        elements.append(best_heading)
        
        best_query = session_data['best_query']
        best_text = Paragraph(f"<b>CO₂:</b> {best_query.get('co2_g', 0):.6f}g | "
                             f"<b>Time:</b> {best_query.get('time_s', 0):.2f}s | "
                             f"<b>Score:</b> {best_query.get('green_score', 0)}/100",
                             normal_style)
        elements.append(best_text)
        elements.append(Spacer(1, 0.2*inch))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = Paragraph(
        "<i>This certificate demonstrates your commitment to sustainable AI and database practices. "
        "Continue optimizing your queries to reduce environmental impact!</i>",
        normal_style
    )
    elements.append(footer_text)
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer and return it
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def calculate_session_metrics(monitoring_data_list: List[Dict]) -> Dict:
    """
    Calculate aggregate metrics for the entire session
    
    Args:
        monitoring_data_list: List of monitoring data dictionaries from all queries
    
    Returns:
        Dictionary with aggregated metrics
    """
    if not monitoring_data_list:
        return {
            'total_co2_g': 0,
            'queries_executed': 0,
            'avg_green_score': 0,
            'session_duration_s': 0
        }
    
    total_co2 = 0
    total_duration = 0
    
    for mon_data in monitoring_data_list:
        try:
            df = pd.json_normalize(mon_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
            df['total_co2_gs'] = df.get('cpu.co2_gs_cpu', 0).fillna(0)
            if 'gpu.co2_gs_gpu' in df.columns:
                df['total_co2_gs'] += df['gpu.co2_gs_gpu'].fillna(0)
            
            query_co2 = (df['total_co2_gs'] * df['time_diff_s']).sum()
            query_duration = df['time_diff_s'].sum()
            
            total_co2 += query_co2
            total_duration += query_duration
        except Exception:
            continue
    
    return {
        'total_co2_g': total_co2,
        'queries_executed': len(monitoring_data_list),
        'avg_green_score': 75,  # Placeholder - should calculate from actual scores
        'session_duration_s': total_duration
    }


def format_relatable_metrics(co2_g: float) -> Dict[str, str]:
    """
    Format CO2 emissions into relatable metrics
    
    Args:
        co2_g: CO2 emissions in grams
    
    Returns:
        Dictionary of formatted relatable metrics
    """
    return {
        'smartphones': 
            {"value":f"{green_metrics.co2_to_smartphones(co2_g):.4f}",
            "text": "smartphones charged"},
        'car_distance': 
            {"value":f"{green_metrics.co2_to_car_km(co2_g) * 1000:.2f}",
            "text": "meters driven by car"},
        'lightbulb_hours': 
            {"value":f"{co2_g * 100:.2f}",
            "text": "hours of LED lightbulb (est.)"}
    }
