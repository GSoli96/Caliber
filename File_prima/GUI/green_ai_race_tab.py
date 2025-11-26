# GUI/green_ai_race_tab.py
"""
Green AI Race Tab - Model Comparison for Sustainability
Compare energy consumption and performance of different LLM models
"""

import streamlit as st
import pandas as pd
import threading
import time
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import llm_adapters
import db_adapters
from utils.prompt_builder import create_sql_prompt
from utils.query_cleaner import extract_sql_query
from utils.system_monitor_utilities import SystemMonitor
from utils import green_metrics
from utils.translations import get_text
from utils.icons import Icons

UI_ICONS = Icons.UI_ICONS

def create_comparison_chart(model_a_data, model_b_data):
    """
    Create side-by-side comparison chart for two models
    
    Args:
        model_a_data: Dictionary with metrics for model A
        model_b_data: Dictionary with metrics for model B
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Energy Consumption (J)", "CO₂ Emissions (g)", "Execution Time (s)"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    models = [model_a_data['name'], model_b_data['name']]
    
    # Energy comparison
    energy_values = [model_a_data.get('energy_j', 0), model_b_data.get('energy_j', 0)]
    fig.add_trace(go.Bar(x=models, y=energy_values, name="Energy", 
                         marker_color=['#FF00FF', '#00FF9F']), row=1, col=1)
    
    # CO2 comparison
    co2_values = [model_a_data.get('co2_g', 0), model_b_data.get('co2_g', 0)]
    fig.add_trace(go.Bar(x=models, y=co2_values, name="CO₂",
                         marker_color=['#FF00FF', '#00FF9F']), row=1, col=2)
    
    # Time comparison
    time_values = [model_a_data.get('time_s', 0), model_b_data.get('time_s', 0)]
    fig.add_trace(go.Bar(x=models, y=time_values, name="Time",
                         marker_color=['#FF00FF', '#00FF9F']), row=1, col=3)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA')
    )
    
    return fig


def determine_winner(model_a_data, model_b_data):
    """
    Determine the most sustainable model based on CO2 emissions
    
    Args:
        model_a_data: Metrics for model A
        model_b_data: Metrics for model B
    
    Returns:
        Tuple of (winner_name, winner_data, loser_data)
    """
    co2_a = model_a_data.get('co2_g', float('inf'))
    co2_b = model_b_data.get('co2_g', float('inf'))
    
    if co2_a < co2_b:
        return model_a_data['name'], model_a_data, model_b_data
    else:
        return model_b_data['name'], model_b_data, model_a_data


def green_ai_race_tab():
    """
    Main function for Green AI Race tab
    """
    # st.header("🏁 Green AI Race - Model Sustainability Comparison")
    
    st.markdown("""
    Compare the energy consumption and CO₂ emissions of different LLM models.
    See which model is the **Most Sustainable Choice** for your query!
    """)
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Challenger A")
        col1A, col2A = st.columns(2)
        with col1A:
            model_a_backend = st.selectbox(
                "Backend A",
                ["Choose a model", "Ollama", "LM Studio", "Hugging Face"],
                key="race_backend_a"    
            )
        with col2A:
            model_a_name = select_model('race_model_a', model_a_backend)    
    with col2:
        st.markdown("### 🟢 Challenger B")
        col1B, col2B = st.columns(2)
        with col1B:
            model_b_backend = st.selectbox(
                "Backend B",
                ["Choose a model", "Ollama", "LM Studio", "Hugging Face"],
                key="race_backend_b"
            )
        with col2B:
            model_b_name = select_model('race_model_b', model_b_backend)
    
    if model_a_backend == 'Choose a model' or model_b_backend == 'Choose a model':
        st.toast(
            icon=UI_ICONS['Warning'],  
            body="Please select a model for both challengers")
        return
    else:
        st.toast(
            icon=UI_ICONS['Success'], 
            body="Models selected successfully")
    
        # Query input
        st.markdown("### 📝 Test Query")
        user_question = st.text_area(
            "Enter your question",
            placeholder="e.g., Show me the top 10 customers by revenue",
            height=100,
            key="race_question"
        )
        
        # Start race button
        can_start = (model_a_name and model_b_name and user_question and 
                    st.session_state.get('create_db_done', False))
        
        if st.button("🏁 Start Race!", type="primary", disabled=not can_start):
            st.session_state.race_status = 'running'
            st.session_state.race_results = {
                'model_a': {'name': f"{model_a_backend}/{model_a_name}", 'status': 'running'},
                'model_b': {'name': f"{model_b_backend}/{model_b_name}", 'status': 'running'}
            }
            st.rerun()
        
        # Display results
        if st.session_state.get('race_status') == 'running':
            st.info("🏁 Race in progress...")
            
            # Simulate race execution (in real implementation, run both models in parallel)
            # For now, show placeholder
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"### {model_a_name}")
                st.spinner("Running...")
            
            with col_b:
                st.markdown(f"### {model_b_name}")
                st.spinner("Running...")
            
            # TODO: Implement actual parallel execution with monitoring
            time.sleep(2)
            st.session_state.race_status = 'done'
            st.rerun()
        
        elif st.session_state.get('race_status') == 'done':
            results = st.session_state.get('race_results', {})
            
            # Display comparison
            st.markdown("## 📊 Race Results")
            
            # Mock data for demonstration
            model_a_data = {
                'name': results['model_a']['name'],
                'energy_j': 150.5,
                'co2_g': 0.0375,
                'time_s': 2.3
            }
            
            model_b_data = {
                'name': results['model_b']['name'],
                'energy_j': 89.2,
                'co2_g': 0.0223,
                'time_s': 1.8
            }
            
            # Comparison chart
            st.plotly_chart(create_comparison_chart(model_a_data, model_b_data), use_container_width=True)
            
            # Winner declaration
            winner_name, winner_data, loser_data = determine_winner(model_a_data, model_b_data)
            
            savings_co2 = loser_data['co2_g'] - winner_data['co2_g']
            savings_pct = (savings_co2 / loser_data['co2_g'] * 100) if loser_data['co2_g'] > 0 else 0
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #00ff9f 0%, #00cc7f 100%); 
                        padding: 30px; border-radius: 15px; text-align: center; 
                        margin: 20px 0; box-shadow: 0 6px 12px rgba(0,255,159,0.4);'>
                <h1 style='color: #0e1117; margin: 0; font-size: 2.5em;'>🏆 Winner: {winner_name}</h1>
                <h2 style='color: #0e1117; margin: 15px 0;'>Most Sustainable Choice!</h2>
                <p style='color: #0e1117; font-size: 1.2em; margin: 10px 0;'>
                    <strong>{savings_pct:.1f}%</strong> less CO₂ emissions<br>
                    <strong>{savings_co2:.6f}g</strong> CO₂ saved
                </p>
                <p style='color: #0e1117; margin-top: 15px;'>
                    ≈ {green_metrics.co2_to_smartphones(savings_co2):.4f} smartphones charged<br>
                    ≈ {green_metrics.co2_to_car_km(savings_co2) * 1000:.2f} meters driven
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            with st.expander("📋 Detailed Metrics"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"#### {model_a_data['name']}")
                    st.metric("Energy", f"{model_a_data['energy_j']:.2f} J")
                    st.metric("CO₂", f"{model_a_data['co2_g']:.6f} g")
                    st.metric("Time", f"{model_a_data['time_s']:.2f} s")
                
                with col_b:
                    st.markdown(f"#### {model_b_data['name']}")
                    st.metric("Energy", f"{model_b_data['energy_j']:.2f} J")
                    st.metric("CO₂", f"{model_b_data['co2_g']:.6f} g")
                    st.metric("Time", f"{model_b_data['time_s']:.2f} s")
            
            if st.button("🔄 New Race"):
                st.session_state.race_status = None
                st.session_state.race_results = {}
                st.rerun()


def select_model(key_model, backend):
    if backend == 'LM Studio':
        flag_server = st.session_state['server_lmStudio']
    elif backend == 'Ollama':
        flag_server = st.session_state['server_ollama']
    elif backend == 'Choose a model':
        st.selectbox(f"{UI_ICONS['Select a Model']}", options=['Choose a model'],
                       index=0, key=key_model+'tmp', disabled=True)
        return None
    else:
        flag_server = True

    if flag_server:
        
        models = llm_adapters.list_models(backend, **cfg)
        
        if isinstance(models, dict) and 'error' in models:
            st.toast(models['error'], 'error')
            return None
        elif not isinstance(models, list) or not models:
            st.toast(get_text("conf_model", "no_models_found"), 'warning')
            return None
        else:
            models_by_backend[backend] = models
            selected_by_backend[backend] = selected_by_backend.get(backend) or models[0]
            st.toast(get_text("conf_model", "models_found", n=len(models)), 'success')
    else:
        st.toast(get_text("conf_model", "server_not_running"), 'warning')
        return None

    models = models_by_backend.get(backend) or []
    if models:
        def _label(x):
            if isinstance(x, str): return x
            if isinstance(x, dict): return x.get("id") or x.get("name") or str(x)
            return str(x)

        labels = [_label(m) for m in models]

        current = selected_by_backend.get(backend)
        default_idx = labels.index(_label(current)) if current in models else 0

        sel = st.selectbox(f"{UI_ICONS['Select a Model']} {get_text('conf_model', 'available_model')}", options=labels,
                                   index=default_idx, key=k(f"{key_model}"))
        return sel
    