# GUI/eco_dashboard.py
"""
Real-time Eco-Dashboard for Green AI & DB Demo
Displays live power consumption, CO2 emissions, and Green Score
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils import green_metrics

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils import green_metrics


def create_power_gauge(current_power_w, max_power_w=200):
    """
    Create a speedometer-style gauge for power consumption
    (versione per tema chiaro)
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_power_w,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "⚡ Power Draw (W)", 'font': {'size': 20, 'color': '#333333'}},
        number={'font': {'size': 40, 'color': '#2e7d32'}},  # verde scuro
        gauge={
            'axis': {
                'range': [None, max_power_w],
                'tickwidth': 1,
                'tickcolor': "#555555"
            },
            'bar': {'color': "#2e7d32"},  # barra verde scuro
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "#cccccc",
            'steps': [
                {'range': [0, max_power_w * 0.3], 'color': 'rgba(76, 175, 80, 0.15)'},   # verde chiaro
                {'range': [max_power_w * 0.3, max_power_w * 0.7], 'color': 'rgba(255, 193, 7, 0.18)'},  # giallo
                {'range': [max_power_w * 0.7, max_power_w], 'color': 'rgba(244, 67, 54, 0.18)'}  # rosso
            ],
            'threshold': {
                'line': {'color': "#e53935", 'width': 3},
                'thickness': 0.75,
                'value': max_power_w * 0.9
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#333333"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_co2_ticker(cumulative_co2_g):
    """
    Create an animated CO2 ticker display (versione per tema chiaro)
    """
    co2_mg = cumulative_co2_g * 1000

    smartphones = green_metrics.co2_to_smartphones(cumulative_co2_g)
    car_meters = green_metrics.co2_to_car_km(cumulative_co2_g) * 1000
    bulb_meters = green_metrics.co2_to_led_bulbs_on(cumulative_co2_g)

    ticker_html = f"""
    <div style='background: linear-gradient(135deg, #f5f7ff 0%, #e3f2fd 100%);
                padding: 20px; border-radius: 10px; text-align: center;
                border: 1px solid #1976d2; box-shadow: 0 0 10px rgba(25,118,210,0.2);'>
        <h3 style='color: #1a237e; margin: 0; font-size: 1.2em;'>
            🌍 Cumulative CO₂ Emissions
        </h3>
        <h1 style='color: #2e7d32; margin: 10px 0; font-size: 3.5em; font-family: monospace;'>
            {co2_mg:.2f} <span style='font-size: 0.5em;'>mg</span>
        </h1>
        <div style='color: #333333; font-size: 0.5em; margin-top: 10px;'>
            <div style='margin: 1px 0;'>📱 ≈ {smartphones:.2f} smartphones charged</div>
            <div style='margin: 1px 0;'>🚗 ≈ {car_meters:.2f} meters driven</div>
            <div style='margin: 1px 0;'>💡 ≈ {bulb_meters:.2f} hours of LED lightbulb</div>
        </div>
    </div>
    """

    return ticker_html


def create_green_score_gauge(score):
    """
    Create a 0-100 Green Score gauge with color zones (tema chiaro)
    """
    # Determine color based on score
    if score >= 80:
        bar_color = "#2e7d32"   # Excellent - verde scuro
        zone_text = "Excellent"
    elif score >= 60:
        bar_color = "#fbc02d"   # Good - giallo/ambra
        zone_text = "Good"
    elif score >= 40:
        bar_color = "#fb8c00"   # Fair - arancione
        zone_text = "Fair"
    else:
        bar_color = "#e53935"   # Poor - rosso
        zone_text = "Poor"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"🌱 Green Score ({zone_text})",
            'font': {'size': 20, 'color': '#333333'}
        },
        number={'font': {'size': 45, 'color': bar_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#555555"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "#cccccc",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(229, 57, 53, 0.15)'},
                {'range': [40, 60], 'color': 'rgba(251, 140, 0, 0.15)'},
                {'range': [60, 80], 'color': 'rgba(251, 192, 45, 0.15)'},
                {'range': [80, 100], 'color': 'rgba(76, 175, 80, 0.15)'}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#333333"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig



def display_eco_dashboard(monitoring_data, show_live=True):
    """
    Display the complete real-time eco-dashboard
    
    Args:
        monitoring_data: List of monitoring data dictionaries
        show_live: If True, display live updating dashboard
    
    Returns:
        Dictionary of Streamlit empty containers for live updates
    """
    st.markdown("### 🌿 Real-time Eco-Dashboard")
    
    if not monitoring_data:
        st.info("⏳ Waiting for monitoring data...")
        return None
    
    # Get latest metrics
    latest = monitoring_data[-1] if monitoring_data else {}
    
    # Calculate current power
    cpu_power = latest.get('cpu', {}).get('power_w', 0)
    gpu_power = latest.get('gpu', {}).get('power_w', 0)
    total_power = cpu_power + gpu_power
    
    # Calculate cumulative CO2
    try:
        mon_df = pd.json_normalize(monitoring_data)
        mon_df['timestamp'] = pd.to_datetime(mon_df['timestamp'])
        mon_df['time_diff_s'] = mon_df['timestamp'].diff().dt.total_seconds().fillna(0)
        mon_df['total_co2_gs'] = mon_df.get('cpu.co2_gs_cpu', 0).fillna(0)
        if 'gpu.co2_gs_gpu' in mon_df.columns:
            mon_df['total_co2_gs'] += mon_df['gpu.co2_gs_gpu'].fillna(0)
        cumulative_co2 = (mon_df['total_co2_gs'] * mon_df['time_diff_s']).sum()
        
        # Calculate Green Score (based on latest metrics)
        rows_returned = 100  # Placeholder - should come from query results
        execution_time = mon_df['time_diff_s'].sum()
        green_score = green_metrics.calculate_green_score(cumulative_co2, rows_returned, execution_time)
    except Exception:
        cumulative_co2 = 0
        green_score = 50
    
    # Create dashboard layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(create_power_gauge(total_power), use_container_width=True, key="power_gauge")
    
    with col2:
        st.markdown(create_co2_ticker(cumulative_co2), unsafe_allow_html=True)
    
    with col3:
        st.plotly_chart(
            create_green_score_gauge(green_score), 
            use_container_width=True, key="green_score")
    
    # Return containers for live updates if needed
    if show_live:
        return {
            'col1': col1,
            'col2': col2,
            'col3': col3
        }
    
    return None


def update_eco_dashboard_live(containers, monitoring_data):
    """
    Update the eco-dashboard with latest data (for live updates)
    
    Args:
        containers: Dictionary of Streamlit containers
        monitoring_data: Latest monitoring data list
    """
    if not containers or not monitoring_data:
        return
    
    # This function would be called in a loop to update the dashboard
    # Implementation depends on Streamlit's st.empty() pattern
    pass
