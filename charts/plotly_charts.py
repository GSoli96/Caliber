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

# def _add_annotations(fig, annotations):
#     """
#     Adds vertical lines and text annotations to the chart.

#     Args:
#         fig (go.Figure): The Plotly figure to annotate.
#         annotations (list of dict): A list of dictionaries, where each dictionary contains:
#             - 'x': The timestamp or x-coordinate for the vertical line and annotation.
#             - 'text': The label text to display.

#     Returns:
#         go.Figure: The annotated Plotly figure.
#     """
#     if not annotations:
#         return fig

#     for ann in annotations:
#         fig.add_vline(
#             x=ann['x'],
#             line_width=2,
#             line_dash="longdashdot",
#             line_color="grey",
#             fillcolor="grey"
#         )
#         fig.add_annotation(
#             x=ann['x'],
#             y=2,
#             yref="paper",
#             text=ann['text'],
#             showarrow=False,
#             textangle=-90,
#             xanchor="right",
#             yanchor="top",
#             font=dict(size=10, color="grey")
#         )
#         print(ann['text'])

#     return fig
def _add_annotations(fig, annotations):
    """
    Aggiunge linee verticali e annotazioni al grafico.
    annotations: list of dict {'x': timestamp, 'text': label}
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
            font=dict(size=15, color="gray")
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

    if df.shape[0] > 20:
        df["cpu_smooth"] = df["cpu_util_percent"].rolling(window=20).mean()
    else:
        df["cpu_smooth"] = df["cpu_util_percent"]

    if df.shape[0] == 1:
        df = add_values_with_time(df)

    # Trace for CPU usage
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['cpu_smooth'],
        mode='lines', name=get_text("charts", "cpu_usage"),
        line=dict(color=NEON_COLORS['cpu'], width=2),
        fill='tozeroy', fillcolor='rgba(0, 255, 159, 0.1)'
    ))

    # Trace for GPU usage (if available)
    if 'gpu_util_percent' in df.columns and not df['gpu_util_percent'].isnull().all():
        if df.shape[0] > 20:
            df["gpu_smooth"] = df["gpu_util_percent"].rolling(window=20).mean()
        else:
            df["gpu_smooth"] = df["gpu_util_percent"]
        if df.shape[0] == 1:
            df = add_values_with_time(df)
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['gpu_smooth'],
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
        if df.shape[0] > 20:
            df["cpu_power_smooth"] = df["cpu_power_w"].rolling(window=20).mean()
        else:
            df["cpu_power_smooth"] = df["cpu_power_w"]
        if df.shape[0] == 1:
            df = add_values_with_time(df)
        max_power = max(max_power, df['cpu_power_smooth'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['cpu_power_smooth'],
            mode='lines', name=get_text("charts", "cpu_power"),
            line=dict(color=NEON_COLORS['cpu'], width=2, dash='dash')
        ))

    # Trace for GPU power (if available)
    if 'gpu_power_w' in df.columns and not df['gpu_power_w'].isnull().all():
        if df.shape[0] > 20:
            df["gpu_power_smooth"] = df["gpu_power_w"].rolling(window=20).mean()
        else:
            df["gpu_power_smooth"] = df["gpu_power_w"]
        if df.shape[0] == 1:
            df = add_values_with_time(df)
        max_power = max(max_power, df['gpu_power_smooth'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['gpu_power_smooth'],
            mode='lines', name=get_text("charts", "gpu_power"),
            line=dict(color=NEON_COLORS['gpu'], width=2, dash='dash')
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
        if df.shape[0] > 20:
            df["cpu_co2_smooth"] = df["cpu.co2_gs_cpu"].rolling(window=20).mean()
        else:
            df["cpu_co2_smooth"] = df["cpu.co2_gs_cpu"]
        if df.shape[0] == 1:
            df = add_values_with_time(df)
        max_val = max(max_val, df['cpu_co2_smooth'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['cpu_co2_smooth'],
            mode='lines', name=get_text("charts", "co2_cpu"),
            line=dict(color=NEON_COLORS['cpu'], width=2),
            fill='tozeroy', fillcolor='rgba(0, 255, 159, 0.1)'
        ))

    # GPU CO2 Rate
    if 'gpu.co2_gs_gpu' in df.columns and not df['gpu.co2_gs_gpu'].isnull().all():
        if df.shape[0] > 20:
            df["gpu_co2_smooth"] = df["gpu.co2_gs_gpu"].rolling(window=20).mean()
        else:
            df["gpu_co2_smooth"] = df["gpu.co2_gs_gpu"]
        if df.shape[0] == 1:
            df = add_values_with_time(df)
        max_val = max(max_val, df['gpu_co2_smooth'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['gpu_co2_smooth'],
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

        if df.shape[0] > 20:
            df["cumulative_gco2_smooth"] = df["cumulative_gco2"].rolling(window=20).mean()
        else:
            df["cumulative_gco2_smooth"] = df["cumulative_gco2"]
        if df.shape[0] == 1:
            df = add_values_with_time(df)
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
    if df.shape[0] > 20:
        df["co2_smooth"] = df["co2"].rolling(window=20).mean()
    else:
        df["co2_smooth"] = df["co2"]

    if df.shape[0] == 1:
        df = add_values_with_time(df)

    fig = go.Figure(data=[
        go.Bar(
            x=df['phase'],
            y=df['co2_smooth'],
            text=df['co2_smooth'].apply(lambda x: f"{x:.4f} g"),
            textposition='auto',
            marker_color=[NEON_COLORS['cpu'], NEON_COLORS['gpu'], NEON_COLORS['co2'], '#AB63FA'][:len(df)]
        )
    ])
    
    layout = _get_neon_layout(get_text("charts", "total_co2_phase"), get_text("charts", "phase"), get_text("charts", "total_co2_g"))
    layout['showlegend'] = False
    fig.update_layout(layout)
    return fig


import pandas as pd

def add_values_with_time(df, time_col: str = "timestamp"):
    """
    df: DataFrame con UNA sola riga.
    time_col: nome della colonna timestamp (datetime64).
    
    Ritorna un nuovo df con:
    - riga 0: tutti 1 tranne timestamp = -1s
    - riga 1: riga originale
    - riga 2: tutti 1 tranne timestamp = +1s
    """
    # prendo l'unica riga
    row = df.iloc[0]

    # creo le due nuove righe come copie della riga originale
    before = row.copy()
    after = row.copy()

    # metto tutte le colonne a 0 tranne il timestamp
    for col in df.columns:
        if col == time_col:
            continue
        before[col] = 0
        after[col] = 0

    # aggiorno il timestamp: -1s e +1s
    before[time_col] = row[time_col] - pd.Timedelta(seconds=1)
    after[time_col] = row[time_col] + pd.Timedelta(seconds=1)

    # trasformo in DataFrame e concateno
    new_df = pd.concat(
        [before.to_frame().T, df, after.to_frame().T],
        ignore_index=True
    )
    return new_df
