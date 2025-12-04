# charts/plotly_charts.py

from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.translations import get_text


# --- NEON THEME CONSTANTS ---
NEON_COLORS = {
    'cpu': '#00FF9F',      # Neon Green
    'gpu': '#FF00FF',      # Neon Magenta
    'co2': '#00FFFF',      # Neon Cyan
    'grid': '#2b2b2b',     # Dark Gray
    'bg': '#0e1117',       # Streamlit Dark BG approx
    'text': '#FAFAFA'      # White
}

def _get_neon_layout(title: str, xaxis_title: str, yaxis_title: str) -> dict:
    """
    Returns a dictionary representing the layout for a Plotly chart with a neon theme.

    Args:
        title (str): The title of the chart.
        xaxis_title (str): The title for the x-axis.
        yaxis_title (str): The title for the y-axis.

    Returns:
        dict: A dictionary containing layout configuration for Plotly.
    """
    return dict(
        title=dict(text=title, font=dict(color=NEON_COLORS['text'], size=18)),
        xaxis=dict(
            title=xaxis_title, 
            showgrid=True, 
            gridcolor=NEON_COLORS['grid'],
            color=NEON_COLORS['text']
        ),
        yaxis=dict(
            title=yaxis_title, 
            showgrid=True, 
            gridcolor=NEON_COLORS['grid'],
            color=NEON_COLORS['text']
        ),
        paper_bgcolor='rgba(0,0,0,0)', # Transparent to let Streamlit theme show
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.02, 
            xanchor="right", x=1,
            font=dict(color=NEON_COLORS['text'])
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )

def _add_annotations(fig, annotations):
    """
    Adds vertical lines and text annotations to the chart.

    Args:
        fig (go.Figure): The Plotly figure to annotate.
        annotations (list of dict): A list of dictionaries, where each dictionary contains:
            - 'x': The timestamp or x-coordinate for the vertical line and annotation.
            - 'text': The label text to display.

    Returns:
        go.Figure: The annotated Plotly figure.
    """
    if not annotations:
        return fig

    for ann in annotations:
        fig.add_vline(
            x=ann['x'],
            line_width=1,
            line_dash="dash",
            line_color="gray"
        )
        fig.add_annotation(
            x=ann['x'],
            y=1,
            yref="paper",
            text=ann['text'],
            showarrow=False,
            textangle=-90,
            xanchor="right",
            yanchor="top",
            font=dict(size=10, color="gray")
        )
    return fig

def generate_usage_chart(df: pd.DataFrame, annotations: List[Dict] = None):
    """
    Generates a line chart showing CPU and GPU usage percentages over time.

    Args:
        df (pd.DataFrame): The dataframe containing usage data. Expected columns:
            - 'timestamp': Time points.
            - 'cpu_util_percent': CPU usage percentage.
            - 'gpu_util_percent': GPU usage percentage (optional).
        annotations (List[Dict], optional): List of annotations to add to the chart. Defaults to None.

    Returns:
        go.Figure: A Plotly figure object.
    """
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # Trace for CPU usage
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['cpu_util_percent'],
        mode='lines', name=get_text("charts", "cpu_usage"),
        line=dict(color=NEON_COLORS['cpu'], width=2),
        fill='tozeroy', fillcolor='rgba(0, 255, 159, 0.1)'
    ))

    # Trace for GPU usage (if available)
    if 'gpu_util_percent' in df.columns and not df['gpu_util_percent'].isnull().all():
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['gpu_util_percent'],
            mode='lines', name=get_text("charts", "gpu_usage"),
            line=dict(color=NEON_COLORS['gpu'], width=2),
            fill='tozeroy', fillcolor='rgba(255, 0, 255, 0.1)'
        ))

    layout = _get_neon_layout(get_text("charts", "resource_usage"), get_text("charts", "time"), get_text("charts", "usage_pct"))
    layout['yaxis']['range'] = [0, 105]
    fig.update_layout(layout)
    
    if annotations:
        fig = _add_annotations(fig, annotations)

    return fig

def generate_power_chart(df: pd.DataFrame, annotations: List[Dict] = None):
    """
    Generates a line chart showing power consumption (in Watts) for CPU and GPU over time.

    Args:
        df (pd.DataFrame): The dataframe containing power data. Expected columns:
            - 'timestamp': Time points.
            - 'cpu_power_w': CPU power consumption in Watts.
            - 'gpu_power_w': GPU power consumption in Watts (optional).
        annotations (List[Dict], optional): List of annotations to add to the chart. Defaults to None.

    Returns:
        go.Figure: A Plotly figure object.
    """
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    max_power = 0

    # Trace for CPU power
    if 'cpu_power_w' in df.columns and not df['cpu_power_w'].isnull().all():
        max_power = max(max_power, df['cpu_power_w'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['cpu_power_w'],
            mode='lines', name=get_text("charts", "cpu_power"),
            line=dict(color=NEON_COLORS['cpu'], width=2, dash='dot')
        ))

    # Trace for GPU power (if available)
    if 'gpu_power_w' in df.columns and not df['gpu_power_w'].isnull().all():
        max_power = max(max_power, df['gpu_power_w'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['gpu_power_w'],
            mode='lines', name=get_text("charts", "gpu_power"),
            line=dict(color=NEON_COLORS['gpu'], width=2, dash='dot')
        ))

    layout = _get_neon_layout(get_text("charts", "power_consumption"), get_text("charts", "time"), get_text("charts", "power_w"))
    layout['yaxis']['range'] = [0, max_power * 1.1] if max_power > 0 else None
    fig.update_layout(layout)
    
    if annotations:
        fig = _add_annotations(fig, annotations)
        
    return fig

def generate_co2_rate_chart(df: pd.DataFrame, annotations: List[Dict] = None):
    """
    Generates a line chart showing instantaneous CO2 emissions (g/s) over time.

    Args:
        df (pd.DataFrame): The dataframe containing emission data. Expected columns:
            - 'timestamp': Time points.
            - 'cpu.co2_gs_cpu': CPU CO2 emissions in g/s.
            - 'gpu.co2_gs_gpu': GPU CO2 emissions in g/s (optional).
        annotations (List[Dict], optional): List of annotations to add to the chart. Defaults to None.

    Returns:
        go.Figure: A Plotly figure object.
    """
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    max_val = 0

    # CPU CO2 Rate
    if 'cpu.co2_gs_cpu' in df.columns:
        max_val = max(max_val, df['cpu.co2_gs_cpu'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['cpu.co2_gs_cpu'],
            mode='lines', name=get_text("charts", "co2_cpu"),
            line=dict(color=NEON_COLORS['cpu'], width=2),
            fill='tozeroy', fillcolor='rgba(0, 255, 159, 0.1)'
        ))

    # GPU CO2 Rate
    if 'gpu.co2_gs_gpu' in df.columns and not df['gpu.co2_gs_gpu'].isnull().all():
        max_val = max(max_val, df['gpu.co2_gs_gpu'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['gpu.co2_gs_gpu'],
            mode='lines', name=get_text("charts", "co2_gpu"),
            line=dict(color=NEON_COLORS['gpu'], width=2),
            fill='tozeroy', fillcolor='rgba(255, 0, 255, 0.1)'
        ))

    layout = _get_neon_layout(get_text("charts", "instant_co2"), get_text("charts", "time"), get_text("charts", "emissions_gs"))
    layout['yaxis']['range'] = [0, max_val * 1.1] if max_val > 0 else None
    fig.update_layout(layout)
    
    if annotations:
        fig = _add_annotations(fig, annotations)

    return fig

def generate_cumulative_co2_chart(charts_data: List[Dict[str, pd.DataFrame]], title: str):
    """
    Generates a line chart showing cumulative CO2 emissions for one or more execution runs.

    Args:
        charts_data (List[Dict[str, pd.DataFrame]]): A list of dictionaries, each containing:
            - 'df': A DataFrame with a 'cumulative_gco2' column.
            - 'name': The name of the run/trace.
        title (str): The title of the chart.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure()

    if not charts_data:
        return fig

    for i, chart_item in enumerate(charts_data):
        df = chart_item.get('df')
        name = chart_item.get('name')

        if df is None or df.empty or 'cumulative_gco2' not in df.columns:
            continue
            
        color = NEON_COLORS['co2'] if i == 0 else NEON_COLORS['cpu'] # Fallback colors

        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_gco2'],
                mode='lines',
                name=name,
                line=dict(color=color, width=3),
                fill='tozeroy' if len(charts_data) == 1 else None,
                fillcolor='rgba(0, 255, 255, 0.1)' if len(charts_data) == 1 else None
            )
        )

    layout = _get_neon_layout(title, get_text("charts", "time"), get_text("charts", "cumulative_co2"))
    fig.update_layout(layout)
    return fig

def generate_phase_co2_comparison_chart(phases_data: List[Dict]):
    """
    Generates a bar chart comparing the total CO2 consumption of different phases.

    Args:
        phases_data (List[Dict]): A list of dictionaries, each representing a phase with:
            - 'phase': The name of the phase.
            - 'co2': The total CO2 emissions for that phase.

    Returns:
        go.Figure: A Plotly figure object.
    """
    if not phases_data:
        return go.Figure()
        
    df = pd.DataFrame(phases_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['phase'],
            y=df['co2'],
            text=df['co2'].apply(lambda x: f"{x:.4f} g"),
            textposition='auto',
            marker_color=[NEON_COLORS['cpu'], NEON_COLORS['gpu'], NEON_COLORS['co2'], '#AB63FA'][:len(df)]
        )
    ])
    
    layout = _get_neon_layout(get_text("charts", "total_co2_phase"), get_text("charts", "phase"), get_text("charts", "total_co2_g"))
    layout['showlegend'] = False
    fig.update_layout(layout)
    return fig