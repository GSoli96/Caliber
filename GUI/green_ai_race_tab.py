import streamlit as st
import pandas as pd
import threading
import time
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import llm_adapters
import db_adapters
from streamlit.runtime.scriptrunner import add_script_run_ctx
from db_adapters.DBManager import DBManager
from utils.prompt_builder import create_sql_prompt
from utils.query_cleaner import extract_sql_query
from utils.system_monitor_utilities import SystemMonitor
from utils import green_metrics
from utils.translations import get_text
from GUI.load_db_tab import load_db_tab, dataset_tab_dbms
from GUI.message_gui import st_toast_temp
from db_adapters.DBManager import check_service_status
from llm_adapters.lmstudio_adapter import run_server_lmStudio
from llm_adapters.ollama_adapter import run_server_ollama
from utils.translations import get_text
from llm_adapters.huggingface_adapter import ensure_model_cached
from GUI.message_gui import st_toast_temp
from utils.load_config import get_HF_Token  # se già lo importi altrove, lascia pure
from threading import Thread
from llm_adapters.lmstudio_adapter import start_server_background as start_lmstudio_background
from llm_adapters.ollama_adapter import start_server_background as start_ollama_background
import time
import numpy as np
import pandas as pd
import json
import streamlit as st

import spacy.cli
from spacy.util import is_package
from llm_adapters.spacy_adapter import (
    COMMON_SPACY_MODELS, model_details
)
from GUI.message_gui import st_toast_temp
from llm_adapters.lmstudio_adapter import lms_get_stream, get_lmstudio_status
from llm_adapters.lmstudio_adapter import lmstudio_panel
from llm_adapters.sensitive_entity import is_sensitive_column
from llm_adapters.spacy_adapter import model_details, COMMON_SPACY_MODELS, suffix_of
from utils.load_config import get_HF_Token
from llm_adapters.model_downloader import download_model_HF
from llm_adapters.ollama_adapter import ollama_panel   # <— aggiungi
from utils.translations import get_text
from llm_adapters.ollama_adapter import get_ollama_status, get_ollama_pid


# --- Helper Functions for Charts ---
def create_comparison_chart(model_a_data, model_b_data):
    """
    Create side-by-side comparison chart for two models
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Energy Consumption (J)", "CO₂ Emissions (g)", "Total Time (s)"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    models = [model_a_data['name'], model_b_data['name']]
    
    # Energy comparison
    energy_values = [model_a_data.get('total_energy_j', 0), model_b_data.get('total_energy_j', 0)]
    fig.add_trace(go.Bar(x=models, y=energy_values, name="Energy", 
                         marker_color=['#FF00FF', '#00FF9F']), row=1, col=1)
    
    # CO2 comparison
    co2_values = [model_a_data.get('total_co2_g', 0), model_b_data.get('total_co2_g', 0)]
    fig.add_trace(go.Bar(x=models, y=co2_values, name="CO₂",
                         marker_color=['#FF00FF', '#00FF9F']), row=1, col=2)
    
    # Time comparison
    time_values = [model_a_data.get('total_time_s', 0), model_b_data.get('total_time_s', 0)]
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

def create_detailed_breakdown_chart(results_data):
    """
    Create a stacked bar chart showing Generation vs Execution costs for each DB and Model.
    """
    # Prepare data
    data = []
    for model_key, res in results_data.items():
        model_name = res['name']
        for db_name, db_res in res['details'].items():
            data.append({
                'Model': model_name,
                'DB': db_name,
                'Phase': 'Generation',
                'CO2 (g)': db_res['gen_co2_g'],
                'Energy (J)': db_res['gen_energy_j'],
                'Time (s)': db_res['gen_time_s']
            })
            data.append({
                'Model': model_name,
                'DB': db_name,
                'Phase': 'Execution',
                'CO2 (g)': db_res['exec_co2_g'],
                'Energy (J)': db_res['exec_energy_j'],
                'Time (s)': db_res['exec_time_s']
            })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    
    # Iterate to add traces
    for phase in ['Generation', 'Execution']:
        phase_data = df[df['Phase'] == phase]
        fig.add_trace(go.Bar(
            name=phase,
            x=[phase_data['Model'], phase_data['DB']],
            y=phase_data['CO2 (g)'],
            text=phase_data['CO2 (g)'].apply(lambda x: f"{x:.4f}g"),
            textposition='auto'
        ))

    fig.update_layout(
        title="CO₂ Emissions Breakdown by Model, Database, and Phase",
        barmode='stack',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        yaxis_title="CO₂ Emissions (g)"
    )
    return fig

def determine_winner(model_a_data, model_b_data):
    """
    Determine the most sustainable model based on Total CO2 emissions
    """
    co2_a = model_a_data.get('total_co2_g', float('inf'))
    co2_b = model_b_data.get('total_co2_g', float('inf'))
    
    if co2_a < co2_b:
        return model_a_data['name'], model_a_data, model_b_data
    else:
        return model_b_data['name'], model_b_data, model_a_data

# --- Race Logic ---
def run_race_phase(model_name, backend, user_question, loaded_databases, display_name=None, challenge=None, progress_bar=None):

    """
    Executes the race for a single model across all databases.
    Returns a dictionary with detailed metrics.
    """
    if challenge is None:
        st.error("Challenge not specified")
        return
    
    final_name = display_name if display_name else f"{backend}/{model_name}"
    results = {
        'name': final_name,
        'details': {}, # db_name -> metrics
        'total_co2_g': 0.0,
        'total_energy_j': 0.0,
        'total_time_s': 0.0
    }

    if 'A' in challenge:
        txt = 'Start Challenge A'
    elif 'B' in challenge:
        txt = 'Start Challenge B'
    
    with st.expander(txt, expanded = st.session_state.race_status != 'done'):
        create_task_list(challenge, 'Started', 1, progress_bar)

        monitor_data = []
        stop_event = threading.Event()
        
        # Start Monitor
        monitor = SystemMonitor(monitor_data, stop_event, st.session_state.get('emission_factor', 250.0), st.session_state.get('cpu_tdp', 65.0))
        add_script_run_ctx(monitor)
        monitor.start()

        create_task_list(challenge, 'Start Monitoring', 2, progress_bar)

        db_state = st.session_state.get('db', {})
        db_choice = db_state.get('choice')
        db_connection_args = db_state.get('connection_args')

        st.session_state.process_status = 'running'
        st.session_state.process_results = {'timestamps': {}}
        st.session_state.monitoring_data = []
        st.session_state.greenefy_status = 'idle'
        
        try:
            for db_name, db_info in loaded_databases.items():
                db_metrics = {
                    'gen_time_s': 0.0, 'gen_co2_g': 0.0, 'gen_energy_j': 0.0,
                    'exec_time_s': 0.0, 'exec_co2_g': 0.0, 'exec_energy_j': 0.0,
                    'sql': None, 'rows': 0, 'error': None
                }
                
                # --- 1. GENERATION PHASE ---
                t0_gen = time.time()
                start_gen_ts = datetime.now(timezone.utc)
                
                # st_toast_temp("Constructing prompt for SQL query", "info")
                prompt = create_sql_prompt(
                    user_question=user_question, 
                    db_connection_args=db_connection_args, 
                    db_name=db_name, 
                    dfs=db_state)
                
                if prompt is None:
                    db_metrics['error'] = "Prompt Construction Error"
                    continue
                
                create_task_list(challenge, 'Construct prompt', 3, progress_bar)
                
                try:
                    # st_toast_temp("Generating SQL query", "info")
                    create_task_list(challenge, 'Generate SQL query', 4, progress_bar)
                   
                    response = llm_adapters.generate(
                        backend=backend,
                        prompt=prompt,
                        model_name=model_name       
                    )
                    # print(response)
                    create_task_list(challenge, 'LLM has generated SQL query', 5, progress_bar)
                    
                    sql_query = extract_sql_query(response)
                    # print(sql_query)
                    db_metrics['sql'] = sql_query
                except Exception as e:
                    db_metrics['error'] = f"Generation Error: {str(e)}"
                
                t1_gen = time.time()
                end_gen_ts = datetime.now(timezone.utc)
                db_metrics['gen_time_s'] = t1_gen - t0_gen
                
                # Calculate Gen Metrics from Monitor Data
                # We need to wait a bit to ensure monitor captured data or interpolate
                time.sleep(0.2) 
                gen_slice = [m for m in monitor_data if m['timestamp'] >= start_gen_ts and m['timestamp'] <= end_gen_ts]
                db_metrics['gen_co2_g'], db_metrics['gen_energy_j'] = calculate_metrics_from_slice(gen_slice)

                # st_toast_temp("Executing SQL query on database", "info")
                create_task_list(challenge, 'Executing SQL query', 6, progress_bar)
                
                # --- 2. EXECUTION PHASE ---
                if db_metrics['sql'] and not db_metrics['error']:
                    t0_exec = time.time()
                    start_exec_ts = datetime.now(timezone.utc)
                    
                    try:
                        db_config = {"choice_DBMS": st.session_state['db_choice'], 
                        'dfs_dict': st.session_state["dataframes"]["DBMS"]
                        }

                        temp_state = {
                            "config_dict": db_config,
                        }
                    
                        db_choice = db_config["choice_DBMS"]

                        manager = DBManager(temp_state, type="query")
                        engine = manager._db_engine()
                        
                        with engine.connect() as conn:
                            res_dict = db_adapters.execute_query(db_choice, conn, db_metrics['sql'])
                            
                        if "error" in res_dict:
                            db_metrics['error'] = f"Execution Error: {res_dict['error']}"
                        else:
                            db_metrics['rows'] = res_dict.get("rows", 0)
                            
                    except Exception as e:
                        db_metrics['error'] = f"Execution Exception: {str(e)}"
                        traceback.print_exc()
                    
                    create_task_list(challenge, 'SQL query executed', 7, progress_bar)
                    
                    # st_toast_temp("Query execution completed", "success")   
                    t1_exec = time.time()
                    end_exec_ts = datetime.now(timezone.utc)
                    db_metrics['exec_time_s'] = t1_exec - t0_exec
                    
                    time.sleep(0.2)
                    exec_slice = [m for m in monitor_data if m['timestamp'] >= start_exec_ts and m['timestamp'] <= end_exec_ts]
                    db_metrics['exec_co2_g'], db_metrics['exec_energy_j'] = calculate_metrics_from_slice(exec_slice)
                
                results['details'][db_name] = db_metrics
                results['total_co2_g'] += (db_metrics['gen_co2_g'] + db_metrics['exec_co2_g'])
                results['total_energy_j'] += (db_metrics['gen_energy_j'] + db_metrics['exec_energy_j'])
                results['total_time_s'] += (db_metrics['gen_time_s'] + db_metrics['exec_time_s'])
                
                create_task_list(challenge, 'Saving results', 8, progress_bar)
                
        finally:
            stop_event.set()
            monitor.join(timeout=1)
            create_task_list(challenge, 'Stopping Monitoring', 9, progress_bar)
            
            st_toast_temp(f"Challenge {challenge} completed", "success")
            
            create_task_list(challenge, 'End Challenge', 10, progress_bar)
        
    return results

def create_task_list(challenge, task, num, progress_bar):
    time.sleep(1)
    progress = st.session_state['race_progress']
    progress += 5
    st.session_state['race_progress'] = progress
    progress_bar.progress(st.session_state['race_progress'])
    if 'A' in challenge:
        return st.checkbox(
                        f'{num}. {task}',
                        value=True,
                        disabled=True,
                        key=task + f'_{challenge}_{st.session_state["widget_race_idx_counter"]}'
                     )
    elif 'B' in challenge:
        return st.checkbox(
                        f'{num}. {task}',
                        value=True,
                        disabled=True,
                        key=task + f'_{challenge}_{st.session_state["widget_race_idx_counter"]}'
                    )

def calculate_metrics_from_slice(data_slice):
    """
    Calculate total CO2 (g) and Energy (J) from a list of monitor data points.
    """
    if not data_slice:
        return 0.0, 0.0
    
    df = pd.json_normalize(data_slice)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # CO2 Rate (g/s)
    df['total_co2_gs'] = df.get('cpu.co2_gs_cpu', 0).fillna(0)
    if 'gpu.co2_gs_gpu' in df.columns:
        df['total_co2_gs'] += df['gpu.co2_gs_gpu'].fillna(0)
        
    # Power (W) = J/s
    df['total_power_w'] = df.get('cpu.power_w', 0).fillna(0)
    if 'gpu.power_w' in df.columns:
        df['total_power_w'] += df['gpu.power_w'].fillna(0)
        
    total_co2_g = (df['total_co2_gs'] * df['time_diff_s']).sum()
    total_energy_j = (df['total_power_w'] * df['time_diff_s']).sum()
    
    return total_co2_g, total_energy_j

# --- Main UI ---
def green_ai_race_tab():
    """
    Main function for Green AI Race tab
    """
    st.markdown("""
    Compare the energy consumption and CO₂ emissions of different LLM models.
    See which model is the **Most Sustainable Choice** for your query!
    """)
    
    st.markdown("""
    <style>
    /* ✅ Colora il check SVG interno quando è attivo */
    [data-testid="stCheckbox"] svg {
        fill: #9e9e9e;  /* grigio default */
    }

    [data-testid="stCheckbox"] input:checked ~ div svg {
        fill: #00c853 !important;  /* ✅ VERDE */
    }
    </style>
    """, unsafe_allow_html=True)

    model_a_name = None
    model_b_name = None
    
    if st.session_state['race']['submit'] == False:
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔵 Challenger A")
            col1A, col2A = st.columns(2)
            with col1A:
                model_a_backend = st.selectbox(
                    "Backend A",
                    ["Choose a model", "Ollama", "LM Studio", "Hugging Face"],
                    key="race_backend_a_" + str(st.session_state['widget_race_idx_counter']), index=0   
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
                    key="race_backend_b_" + str(st.session_state['widget_race_idx_counter']), index=0
                )
            with col2B:
                model_b_name = select_model('race_model_b', model_b_backend)
        
        can_submit = model_a_backend != 'Choose a model' and model_b_backend != 'Choose a model' and model_a_name and model_b_name
        
        with st.form("🏆 Challenge Selection", border=False, clear_on_submit=False):
            submit_button = st.form_submit_button("Choose this configuration", type="primary", disabled=not can_submit, key="race_submit_form_button_" + str(st.session_state['widget_race_idx_counter']))
        
        if submit_button:
            st.session_state['race']['A']['backend'] = model_a_backend
            st.session_state['race']['A']['model'] = model_a_name
            st.session_state['race']['B']['backend'] = model_b_backend
            st.session_state['race']['B']['model'] = model_b_name
            st.session_state['race']['submit'] = True

    loaded_databases = st.session_state["dataframes"]["DBMS"].keys()
    
    if st.session_state['race']['submit'] == False:
        return
    
    if len(loaded_databases) == 0:
        with st.expander("📋 Load Databases", expanded=len(loaded_databases) == 0):
            load_db_tab(key=f"race_db_{st.session_state['widget_race_idx_counter']}")
    else:
        dataset_tab_dbms(key_alter=f"race_db_{st.session_state['widget_race_idx_counter']}")
    
    loaded_databases = st.session_state["dataframes"]["DBMS"].keys()
    
    if len(loaded_databases) == 0:
        st.warning("Please load at least one database to start the race.")
        return
    
    # Query input
    st.markdown("### 📝 Test Query")
    user_question = st.text_area(
        "Enter your question",
        placeholder="e.g., Show me the top 10 customers by revenue",
        height=100, value='Count all persons with an age > 10',
        key=f"race_question_{st.session_state['widget_race_idx_counter']}"
    )
    
    if not user_question:
        st.warning("Please enter a query to start the race.")
        return
    # Start race button
    can_start = (st.session_state['race']['submit'] == True 
    and st.session_state['race']['A']['model'] is not None 
    and st.session_state['race']['B']['model'] is not None 
    and user_question 
    and len(loaded_databases) > 0 
    and st.session_state.get('race_status') != 'running' 
    and st.session_state.get('race_status') != 'done')
    start_button = st.button("🏁 Start Race!", type="primary", disabled=not can_start, key=f"race_start_button_{st.session_state['widget_race_idx_counter']}")

    if start_button:
        st.session_state.race_status = 'running'
        
    # Display results
    if st.session_state.get('race_status') == 'running':
        st.info("🏁 Race in progress... Please wait.")
        
        progress_bar = st.progress(st.session_state['race_progress'])
        status_text = st.empty()
        model_a_name = st.session_state['race']['A']['model']
        model_b_name = st.session_state['race']['B']['model']
        model_a_backend = st.session_state['race']['A']['backend']
        model_b_backend = st.session_state['race']['B']['backend']

        # Handle identical models
        name_a = model_a_name
        name_b = model_b_name
        if model_a_name == model_b_name and model_a_backend == model_b_backend:
            name_a = f"{model_a_name} (A)"
            name_b = f"{model_b_name} (B)"
        
        loaded_databases = st.session_state["dataframes"]["DBMS"]

        col1, col2 = st.columns(2)

        # Run Model A
        with col1:
            res_a = run_race_phase(model_a_name, model_a_backend, user_question, loaded_databases, display_name=name_a, challenge='A', progress_bar=progress_bar)
        
        # Run Model B
        with col2:
            res_b = run_race_phase(model_b_name, model_b_backend, user_question, loaded_databases, display_name=name_b, challenge='B', progress_bar=progress_bar)
        
        st.session_state.race_results = {'model_a': res_a, 'model_b': res_b}
        st.session_state.race_status = 'done'
    
    if (len(st.session_state.race_results.keys()) == 0) and st.session_state.race_status == 'done':
        st.rerun()

    if ('model_a' in list(st.session_state.race_results.keys()) and 'model_b' in list(st.session_state.race_results.keys())):
        results = st.session_state.race_results
        model_a_data = results['model_a']
        model_b_data = results['model_b']
        
        # Display comparison
        st.markdown("## 📊 Race Results")
        
        # Comparison chart
        st.plotly_chart(create_comparison_chart(model_a_data, model_b_data), use_container_width=True, key=f"race_comparison_chart_{st.session_state['widget_race_idx_counter']}")
        
        # Winner declaration
        winner_name, winner_data, loser_data = determine_winner(model_a_data, model_b_data)
        
        savings_co2 = loser_data['total_co2_g'] - winner_data['total_co2_g']
        savings_pct = (savings_co2 / loser_data['total_co2_g'] * 100) if loser_data['total_co2_g'] > 0 else 0
        
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
        
        # Detailed Breakdown Chart
        st.markdown("### 🔍 Detailed Breakdown (Generation vs Execution)")
        st.plotly_chart(create_detailed_breakdown_chart(results), use_container_width=True, key=f"race_detailed_breakdown_chart_{st.session_state['widget_race_idx_counter']}")
        
        # Detailed Metrics per DB
        with st.expander("📋 Detailed Metrics per Database"):
            for model_key, res in results.items():
                st.markdown(f"#### {res['name']}")
                for db_name, db_res in res['details'].items():
                    st.markdown(f"**Database:** `{db_name}`")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Gen Time", f"{db_res['gen_time_s']:.2f} s")
                    c2.metric("Gen CO₂", f"{db_res['gen_co2_g']:.4f} g")
                    c3.metric("Exec Time", f"{db_res['exec_time_s']:.2f} s")
                    c4.metric("Exec CO₂", f"{db_res['exec_co2_g']:.4f} g")
                    
                    if db_res['error']:
                        st.error(db_res['error'])
                    else:
                        with st.expander("Show SQL & Green Score"):
                            st.code(db_res['sql'], language='sql')
                            st.info(f"Rows returned: {db_res['rows']}")
                            
                            # Green Score
                            score = green_metrics.calculate_green_score(
                                db_res['gen_co2_g'] + db_res['exec_co2_g'],
                                db_res['rows'],
                                db_res['gen_time_s'] + db_res['exec_time_s']
                            )
                            st.metric("Green Score", f"{score}/100")
                    st.divider()
        
        if st.button("🔄 New Race", key=f"race_new_race_{st.session_state['widget_race_idx_counter']}"):
            st.session_state.race_status = None
            st.session_state.race_results = {}
            
            st.session_state.race = {
                'A':{
                'backend': None,  # "LM Studio" | "Ollama" | "Hugging Face" | "Local (Upload)"
                'model': None,  # string (nome/id del modello)
                },
                'B':{
                'backend': None,  # "LM Studio" | "Ollama" | "Hugging Face" | "Local (Upload)"
                'model': None,  # string (nome/id del modello)
                },
                'submit': False
                }

            st.session_state.race_progress = 0

            st.session_state.tasks_A = {
                    "Started": False,
                    "Start Monitoring": False,
                    "Construct prompt": False,
                    "Sending prompt": False,
                    "LLM has generated SQL query": False,
                    "Executing SQL query": False,
                    "SQL query executed": False,
                    "Saving results": False,
                    "Stopping Monitoring": False,
                    "End Challenge": False
        }

            st.session_state.tasks_B = {
                    "Started": False,
                    "Start Monitoring": False,
                    "Construct prompt": False,
                    "Sending prompt": False,
                    "LLM has generated SQL query": False,
                    "Executing SQL query": False,
                    "SQL query executed": False,
                    "Saving results": False,
                    "Stopping Monitoring": False,
                    "End Challenge": False
                }
            st.session_state['widget_race_idx_counter'] += 1
            st.toast("New race started", icon='✅')
            # st.rerun()

def select_model(key_model, backend):

    if backend == 'LM Studio':
        flag_server = st.session_state['server_lmStudio']
    elif backend == 'Ollama':
        flag_server = st.session_state['server_ollama']
    elif backend == 'Choose a model':
        st.selectbox(f"🎯", options=['Choose a model'],
                       index=0, 
                       key=key_model+'tmp', 
                       disabled=True)
        return None
    else:
        flag_server = True

    sel = ''

    if backend == 'Choose a model':
        return 
    models = []
    if flag_server and sel != "Choose a model":
        models = llm_adapters.list_models(backend)

        if models:
            # st.toast(get_text("conf_model", "models_found", n=len(models)), icon='✅')
            pass
        else:
            st.toast(get_text("conf_model", "no_models_found"), icon='⚠️')
            return None
    else:
        # st_toast_temp(get_text("conf_model", "server_not_running"), 'warning')
        if st.session_state['server_lmStudio'] != True or st.session_state['server_ollama'] != True:
            with st.spinner("Waiting for servers to be started..."):
                ensure_llm_status(backend)

    if models:
        # Filter out non-string models if any
        model_options = ["Choose a model"] + [m if isinstance(m, str) else m.get('id', str(m)) for m in models]
        sel = st.selectbox(f"🎯 {get_text('conf_model', 'available_model')}", options=model_options,
                                   index=0, key=f"choose_{key_model}_{backend}")
        if sel == "Choose a model":
            return None

        if backend =="Hugging Face":
            hf_token = get_HF_Token()

            th = Thread(target=ensure_model_cached, args=(sel, backend, hf_token))
            add_script_run_ctx(th)
            th.start()
            # st_toast_temp(f"Checking model for {backend}", "info")

            with st.spinner("Waiting for model to be cached..."):
                while th.is_alive():
                    time.sleep(1)
            # st.toast("Model cached", icon='✅')

        return sel
    return None

def ensure_llm_status(backend):

    if backend == 'LM Studio':
        if st.session_state['server_lmStudio'] == True:
            return
        else:
            res = start_lmstudio_background()
            if res["ok"]:
                st.session_state['server_lmStudio'] = True
                st_toast_temp(get_text("llm_adapters", "started_background"), 'success')
                get_lmstudio_status.clear()
                st.rerun()

            else:
                st.session_state['server_lmStudio'] = False
                st_toast_temp(res["msg"], 'error')
    elif backend == 'Ollama':
        if st.session_state['server_ollama'] == True:
            return
        else:
            res = start_ollama_background()
            if res.get("ok"):
                st.session_state['server_ollama'] = True
                print(f"[INFO] Ollama server started")
                st_toast_temp(get_text("llm_adapters", "ollama_server_started"), 'success')
                if res.get("cmd"):
                    st.toast(res["cmd"])

                get_ollama_status.clear()
                get_ollama_pid.clear()
                st.rerun()
        
            else:
                st.session_state['server_ollama'] = False
                st_toast_temp(res.get("msg", get_text("llm_adapters", "ollama_server_error")), 'error')
    