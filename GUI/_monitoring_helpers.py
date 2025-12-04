# Resource monitoring helper functions to be integrated into relational_profiling_tab.py

# Insert these functions after line 418 (after zscore_outlier_rate function)

# ------------------------------------------------------------------------------------
# Resource Monitoring Helpers
# ------------------------------------------------------------------------------------
def start_monitoring():
    """
    Start resource monitoring thread.
    Returns: (data_list, stop_event, monitor_thread)
    """
    data_list = []
    stop_event = threading.Event()
    
    # Get emission factor and CPU TDP from session state or defaults
    emission_factor = st.session_state.get('emission_factor', 250.0)
    cpu_tdp = st.session_state.get('cpu_tdp', 65.0)
    
    monitor = SystemMonitor(data_list, stop_event, emission_factor, cpu_tdp)
    monitor.start()
    
    return data_list, stop_event, monitor

def stop_monitoring(data_list, stop_event, monitor_thread):
    """
    Stop resource monitoring and return collected metrics.
    """
    stop_event.set()
    monitor_thread.join(timeout=2.0)
    return data_list

def create_resource_plots(metrics_data):
    """
    Create Plotly figure with CPU%, GPU%, and CO2 subplots.
    
    Args:
        metrics_data: List of metric dicts from SystemMonitor
    
    Returns:
        plotly.graph_objects.Figure
    """
    if not metrics_data:
        return None
    
    # Extract data
    timestamps = [m.get('timestamp') for m in metrics_data if 'timestamp' in m]
    cpu_percents = [m.get('cpu', {}).get('percent', 0) for m in metrics_data]
    gpu_percents = [m.get('gpu', {}).get('percent', 0) for m in metrics_data]
    co2_gs = [m.get('cpu', {}).get('co2_gs_cpu', 0) + m.get('gpu', {}).get('co2_gs_gpu', 0) for m in metrics_data]
    
    has_gpu = any(gpu_percents)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('CPU Usage (%)', 'GPU Usage (%)' if has_gpu else 'GPU (Not Available)', 'CO2 Emissions (g/s)'),
        vertical_spacing=0.12
    )
    
    # CPU plot
    fig.add_trace(
        go.Scatter(x=timestamps, y=cpu_percents, mode='lines', name='CPU %', 
                   line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    
    # GPU plot
    if has_gpu:
        fig.add_trace(
            go.Scatter(x=timestamps, y=gpu_percents, mode='lines', name='GPU %',
                       line=dict(color='#ff7f0e', width=2)),
            row=2, col=1
        )
    else:
        fig.add_annotation(
            text="GPU not detected or not in use",
            xref="x2", yref="y2",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
    
    # CO2 plot
    fig.add_trace(
        go.Scatter(x=timestamps, y=co2_gs, mode='lines', name='CO2 (g/s)',
                   line=dict(color='#2ca02c', width=2), fill='tozeroy'),
        row=3, col=1
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="%", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="%", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="g/s", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False, title_text="Resource Consumption During Analysis")
    
    return fig
