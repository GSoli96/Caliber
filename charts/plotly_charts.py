# charts/plotly_charts.py

from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

# --- FUNZIONE REINSERITA PER COMPATIBILITÀ ---
# Questa funzione è richiesta da app.py. La ripristino come placeholder.
def generate_charts():
    """Placeholder per una futura tab di grafici."""
    st.info('TO DO: Questa sezione potrà contenere grafici di riepilogo.')

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
            font=dict(size=10, color="gray")
        )
    return fig

def generate_usage_chart(df: pd.DataFrame, annotations: List[Dict] = None):
    """Genera un grafico dell'utilizzo percentuale di CPU e GPU."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # Traccia per l'utilizzo della CPU
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['cpu_util_percent'],
        mode='lines', name='CPU Usage (%)',
        line=dict(color=NEON_COLORS['cpu'], width=2),
        fill='tozeroy', fillcolor='rgba(0, 255, 159, 0.1)'
    ))

    # Traccia per l'utilizzo della GPU (se disponibile)
    if 'gpu_util_percent' in df.columns and not df['gpu_util_percent'].isnull().all():
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['gpu_util_percent'],
            mode='lines', name='GPU Usage (%)',
            line=dict(color=NEON_COLORS['gpu'], width=2),
            fill='tozeroy', fillcolor='rgba(255, 0, 255, 0.1)'
        ))

    layout = _get_neon_layout("Resource Usage (%)", "Time", "Usage (%)")
    layout['yaxis']['range'] = [0, 105]
    fig.update_layout(layout)
    
    if annotations:
        fig = _add_annotations(fig, annotations)

    return fig

def generate_power_chart(df: pd.DataFrame, annotations: List[Dict] = None):
    """Genera un grafico del consumo energetico (Watt) di CPU e GPU."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    max_power = 0

    # Traccia per il consumo della CPU
    if 'cpu_power_w' in df.columns and not df['cpu_power_w'].isnull().all():
        max_power = max(max_power, df['cpu_power_w'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['cpu_power_w'],
            mode='lines', name='CPU Power (W)',
            line=dict(color=NEON_COLORS['cpu'], width=2, dash='dot')
        ))

    # Traccia per il consumo della GPU (se disponibile)
    if 'gpu_power_w' in df.columns and not df['gpu_power_w'].isnull().all():
        max_power = max(max_power, df['gpu_power_w'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['gpu_power_w'],
            mode='lines', name='GPU Power (W)',
            line=dict(color=NEON_COLORS['gpu'], width=2, dash='dot')
        ))

    layout = _get_neon_layout("Power Consumption (W)", "Time", "Power (W)")
    layout['yaxis']['range'] = [0, max_power * 1.1] if max_power > 0 else None
    fig.update_layout(layout)
    
    if annotations:
        fig = _add_annotations(fig, annotations)
        
    return fig

def generate_co2_rate_chart(df: pd.DataFrame, annotations: List[Dict] = None):
    """Genera un grafico delle emissioni di CO2 istantanee (g/s)."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    max_val = 0

    # CPU CO2 Rate
    if 'cpu.co2_gs_cpu' in df.columns:
        max_val = max(max_val, df['cpu.co2_gs_cpu'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['cpu.co2_gs_cpu'],
            mode='lines', name='CO₂ CPU (g/s)',
            line=dict(color=NEON_COLORS['cpu'], width=2),
            fill='tozeroy', fillcolor='rgba(0, 255, 159, 0.1)'
        ))

    # GPU CO2 Rate
    if 'gpu.co2_gs_gpu' in df.columns and not df['gpu.co2_gs_gpu'].isnull().all():
        max_val = max(max_val, df['gpu.co2_gs_gpu'].max())
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['gpu.co2_gs_gpu'],
            mode='lines', name='CO₂ GPU (g/s)',
            line=dict(color=NEON_COLORS['gpu'], width=2),
            fill='tozeroy', fillcolor='rgba(255, 0, 255, 0.1)'
        ))

    layout = _get_neon_layout("Instant CO₂ Emissions (g/s)", "Time", "Emissions (g/s)")
    layout['yaxis']['range'] = [0, max_val * 1.1] if max_val > 0 else None
    fig.update_layout(layout)
    
    if annotations:
        fig = _add_annotations(fig, annotations)

    return fig

def generate_cumulative_co2_chart(charts_data: List[Dict[str, pd.DataFrame]], title: str):
    """
    Genera un grafico delle emissioni di CO2 cumulative per una o più esecuzioni.
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

    layout = _get_neon_layout(title, "Time", "Cumulative CO₂ (g)")
    fig.update_layout(layout)
    return fig

def generate_phase_co2_comparison_chart(phases_data: List[Dict]):
    """
    Genera un grafico a barre per confrontare il consumo totale di CO2 delle fasi.
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
    
    layout = _get_neon_layout("Total CO₂ Emissions by Phase", "Phase", "Total CO₂ (g)")
    layout['showlegend'] = False
    fig.update_layout(layout)
    return fig