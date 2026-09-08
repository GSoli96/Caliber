import streamlit as st
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
from utils.system_monitor_utilities import SystemMonitor, watt_to_co2_gs, co2_gs_to_co2_kgh
import threading
import llm_adapters
from utils.prompt_builder import create_sql_prompt
from utils.query_cleaner import extract_sql_query
from utils.utils_gen_eval_query import get_all_loaded_dfs
import db_adapters
from GUI.green_ai_race_tab import green_ai_race_tab
from utils.translations import get_text
from db_adapters.DBManager import DBManager

DB_DIR = st.session_state.get('db_dir', "database")

# --- CO2 Monitoring Helper ---
class CO2Monitor:
    def __init__(self, operation_name="Operation"):
        self.operation_name = operation_name
        self.data_list = []
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.start_time = None
        self.end_time = None
        self.total_co2_g = 0.0
        self.total_energy_wh = 0.0

    def __enter__(self):
        self.start_time = time.time()
        self.data_list = []
        self.stop_event.clear()
        # Start monitoring thread
        emission_factor = st.session_state.get('emission_factor', 250.0)
        cpu_tdp = st.session_state.get('cpu_tdp', 65.0)
        
        self.monitor_thread = SystemMonitor(self.data_list, self.stop_event, emission_factor, cpu_tdp)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join()
        self.end_time = time.time()
        self._calculate_totals()
        self._update_global_kpi()

    def _calculate_totals(self):
        total_co2 = 0.0
        total_energy_ws = 0.0 
        
        if not self.data_list:
            return

        dt = 0.5 
        
        for data in self.data_list:
            if 'cpu' in data:
                total_co2 += data['cpu'].get('co2_gs_cpu', 0) * dt
                total_energy_ws += data['cpu'].get('power_w', 0) * dt
            
            if 'gpu' in data:
                total_co2 += data['gpu'].get('co2_gs_gpu', 0) * dt
                total_energy_ws += data['gpu'].get('power_w', 0) * dt

        self.total_co2_g = total_co2
        self.total_energy_wh = total_energy_ws / 3600.0

    def _update_global_kpi(self):
        if 'benchmarking_kpi' not in st.session_state:
            st.session_state['benchmarking_kpi'] = {
                'total_co2_g': 0.0,
                'total_energy_wh': 0.0,
                'history': []
            }
        
        kpi = st.session_state['benchmarking_kpi']
        kpi['total_co2_g'] += self.total_co2_g
        kpi['total_energy_wh'] += self.total_energy_wh
        kpi['history'].append({
            'timestamp': datetime.now(),
            'operation': self.operation_name,
            'co2_g': self.total_co2_g,
            'energy_wh': self.total_energy_wh,
            'duration_s': self.end_time - self.start_time
        })

def benchmark_tab():
    if 'benchmarking_state' not in st.session_state:
        st.session_state['benchmarking_state'] = {
            'llm_backend': None,
            'llm_model': None,
            'dbms_selection': [],
            'dbms_connections': {},
            'csv_file': None,
            'csv_mapping': {},
            'active_csv_df': None
        }

    st.markdown("## 🎯 Eco Benchmarking")
    
    render_global_kpi()

    tab_names = [
        "🏁 Green AI Race",
        "🧠 NL→SQL Benchmark",
        "🗄️ LLM & DBMS Execution Benchmark",
    ]
        
    tabs = st.tabs(tab_names)
    
    # --- GREEN AI RACE ---
    with tabs[0]:
        st.subheader("🏁 Green AI Race")
        green_ai_race_tab()
        
    with tabs[1]:
        st.subheader("🧠 NL→SQL Benchmark")
        render_llm_nl_sql_tab()

    with tabs[2]:
        st.subheader("🗄️ DBMS Execution Benchmark")
        render_dbms_execution_tab()

def render_global_kpi():
    if 'benchmarking_kpi' in st.session_state:
        kpi = st.session_state['benchmarking_kpi']
        cols = st.columns(4)
        cols[0].metric("Total Session CO₂", f"{kpi['total_co2_g']:.4f} g")
        cols[1].metric("Total Energy", f"{kpi['total_energy_wh']:.4f} Wh")
        
        if kpi['history']:
            last_op = kpi['history'][-1]
            cols[2].metric("Last Op CO₂", f"{last_op['co2_g']:.4f} g")
            cols[3].metric("Last Op Duration", f"{last_op['duration_s']:.2f} s")
    else:
        st.info("Start a benchmark to see CO₂ & Energy KPIs.")

def render_shared_setup(key_suffix=""):
    state = st.session_state['benchmarking_state']
    
    with st.expander("⚙️ Setup: LLM, Database & Data", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🤖 LLM Configuration")
            llm_backend = st.selectbox(
                "LLM Backend", 
                ["LM Studio", "Ollama", "Hugging Face"], 
                key=f"bench_llm_backend_{key_suffix}"
            )
            
            models = llm_adapters.list_models(llm_backend)
            if models:
                llm_model = st.selectbox(
                    "Model", 
                    models, 
                    key=f"bench_llm_model_{key_suffix}"
                )
            else:
                st.warning("No models found for the selected backend.")
                return
            
            state['llm_backend'] = llm_backend
            state['llm_model'] = llm_model

        with col2:
            st.markdown("#### 📁 Active Dataset")
            active_db_name = st.session_state.get('db_name')
            active_tables = st.session_state.get("dataframes", {}).get("DBMS", {}).get(active_db_name, []) if active_db_name else []

            if active_db_name and active_tables:
                st.success(
                    f"Using dataset **{active_db_name}** "
                    f"({st.session_state.get('db_choice', 'N/A')}) — "
                    f"{len(active_tables)} table(s) loaded from Data Hub."
                )
            else:
                st.warning("⚠️ No dataset loaded. Please load one from **Data Hub** first.")

    # if len(list(st.session_state["dataframes"]["DBMS"].keys())) > 0:
    #     dataset_tab_dbms(key)

    with st.expander("📂 Data Source (CSV)", expanded=True):
        uploaded_file = st.file_uploader("Upload Benchmark CSV", type=["csv"], key=f"bench_csv_{key_suffix}")
        
        if uploaded_file:
            if state['csv_file'] != uploaded_file:
                state['csv_file'] = uploaded_file
                state['active_csv_df'] = pd.read_csv(uploaded_file)
                st.toast("CSV Loaded!")
            
            if state['active_csv_df'] is not None:
                st.dataframe(state['active_csv_df'].head(3), height=100)
                
                cols = state['active_csv_df'].columns.tolist()
                nl_col = st.selectbox("Natural Language Query Column", [None] + cols, 
                key=f"bench_nl_col_{key_suffix}")
                
                state['csv_mapping']['nl_query'] = nl_col
                
                dbms_cols = {}
                for db in ["MySQL", "SQLite", "PostgreSQL", "DuckDB", "SQL Server"]:
                    if db in cols:
                        dbms_cols[db] = db
                state['csv_mapping']['dbms_cols'] = dbms_cols
                
                if dbms_cols:
                    st.caption(f"Detected DBMS Ground Truth Columns: {', '.join(dbms_cols.keys())}")

# def dataset_tab_dbms(key_alter="Eco"):
#     """Visualizza i dettagli per TUTTI i database DBMS caricati in session_state."""
#     loaded_databases = st.session_state["dataframes"]["DBMS"]

#     if not loaded_databases:
#         st.info(get_text("load_dataset", "no_db_loaded"))
#         return

#     with st.expander('📁 Dataset Overview', expanded=False):
#         for db_name, tables_data in loaded_databases.items():

#             st.header(f"📁 Database: {db_name}")

#             # 3. Recupera la configurazione specifica per *questo* database
#             #    Questo funziona grazie alle modifiche al prerequisito
#             config_dict = st.session_state.get('uploaded_dbms', {}).get(db_name)

#             with st.container(border=True):
#                 st.subheader(get_text("load_dataset", "db_info"))

#                 # 4. Funzione modulare per mostrare le info
#                 _display_db_info(config_dict, db_name, tables_data, key_alter)

#                 # st.divider()
#                 st.subheader(get_text("load_dataset", "explore_tables"))

#                 # 5. Logica esistente per creare i tab delle tabelle
#                 tab_names = []
#                 tab_dfs = []

#                 if not tables_data:
#                     st.warning(get_text("gen_eval", "no_table_found", db_name=db_name))
#                     continue

#                 for db_dict in tables_data:
#                     tab_names.append(db_dict["table_name"])
#                     tab_dfs.append(db_dict["table"])

#                 tabs = st.tabs(tab_names)

#                 for name, tab, df in zip(tab_names, tabs, tab_dfs):
#                     with tab:
#                         if df is not None and df.shape[0] > 0:
#                             with st.container(border=False):
#                                 st.subheader(get_text("load_dataset", "dataset_details"))
#                                 rows_to_show = st.number_input(
#                                     get_text("load_dataset", "rows_to_show"),
#                                     min_value=1,
#                                     max_value=len(df),
#                                     value=min(5, len(df)),
#                                     step=1,
#                                     help=get_text("load_dataset", "rows_to_show_help"),
#                                     key=f'numberInput_preview_{db_name}_{name}_{tab}'
#                                 )
#                                 st.write(df.head(rows_to_show))
#                         else:
#                             st.error(f'Table {name} in dataset {db_name} non trovato!')


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def reset_all_dbms(key_prefix):
    """Resets all DBMS-related state and removes the database."""
    config_dict = st.session_state['uploaded_dbms']
    db_manager = DBManager({'config_dict': config_dict}, 'remove')
    db_manager._reset_all_db()

    idx = st.session_state['widget_idx_counter'] 
    st.session_state.pop(f'conn_str_{key_prefix}_{idx}', None)
    st.session_state.pop(f'db_first_{key_prefix}_{idx}', None)
    st.session_state.pop(f'table_area_{key_prefix}_{idx}', None)
    st.session_state.pop(f'load_all_table_{key_prefix}_{idx}', None)
    st.session_state.pop(f'button_DBMS_{key_prefix}_{idx}', None)
    st.session_state.pop(f'button_reset_DBMS_{key_prefix}_{idx}', None)

    st.session_state['dataframes'].pop(['DBMS'], {})
    st.session_state['dataframes']['DBMS'] = {}
    st.session_state.pop('uploaded_dbms', {})
    st.session_state.setdefault('uploaded_dbms', {})

    st.cache_data.clear()

    st.session_state['widget_idx_counter'] += 1
    st.rerun()

def run_feature_benchmark(scenario, dataset_size, features, runs, duration):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(runs):
        status_text.text(f"Running iteration {i+1}/{runs}...")
        
        with CO2Monitor(f"Feature Benchmark - Run {i+1}") as monitor:
            time.sleep(duration) 
            
            import random
            latency = random.uniform(10, 50) + (5 if "HE Enabled" in features else 0)
            throughput = random.uniform(1000, 5000)
        
        results.append({
            'run': i + 1,
            'scenario': scenario,
            'dataset_size': dataset_size,
            'features': ", ".join(features),
            'latency_ms': latency,
            'throughput_ops': throughput,
            'co2_g': monitor.total_co2_g,
            'energy_wh': monitor.total_energy_wh
        })
        progress_bar.progress((i + 1) / runs)
    
    status_text.text("Benchmark Complete!")
    st.session_state['feature_benchmark_results'] = pd.DataFrame(results)
    st.rerun()

def render_llm_nl_sql_tab():
    
    
    render_shared_setup(key_suffix="nl_sql")
    
    state = st.session_state['benchmarking_state']
    
    if st.button("Load Example NL->SQL"):
        # Mock loading example data
        mock_df = pd.DataFrame({
            'nl_query': ['Show all users', 'Count orders', 'Find user by name'],
            'MySQL': ['SELECT * FROM users', 'SELECT COUNT(*) FROM orders', 'SELECT * FROM users WHERE name = ?'],
            'PostgreSQL': ['SELECT * FROM users', 'SELECT COUNT(*) FROM orders', 'SELECT * FROM users WHERE name = ?']
        })
        state['active_csv_df'] = mock_df
        state['csv_mapping']['nl_query'] = 'nl_query'
        state['csv_mapping']['dbms_cols'] = {'MySQL': 'MySQL', 'PostgreSQL': 'PostgreSQL'}
        st.toast("Example Loaded (Mock Data)")
        st.rerun()
    
    if state['active_csv_df'] is not None and state['csv_mapping'].get('nl_query'):
        st.divider()
        st.markdown("### 🚀 Execution")
        
        if st.button("Generate SQL for all rows", type="primary"):
            run_nl_sql_benchmark()
            
        if 'nl_sql_results' in st.session_state:
            st.markdown("### Results")
            results_df = st.session_state['nl_sql_results']
            st.dataframe(results_df)
            
            avg_co2 = results_df['co2_g'].mean()
            total_co2 = results_df['co2_g'].sum()
            avg_latency = results_df['latency_s'].mean()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total CO₂", f"{total_co2:.4f} g")
            c2.metric("Avg CO₂ / Query", f"{avg_co2:.4f} g")
            c3.metric("Avg Latency", f"{avg_latency:.2f} s")
            
    else:
        st.info("Please upload a CSV and select the Natural Language Query column.")

def run_nl_sql_benchmark():
    # Check for demo mode
    from utils.demo_data import DEMO_MODE_ENABLED, get_demo_benchmarking_results
    
    if DEMO_MODE_ENABLED:
        # Demo mode: simulate execution with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_queries = 5
        for i in range(total_queries):
            status_text.text(f"Generating query {i+1}/{total_queries}...")
            progress_bar.progress((i + 1) / total_queries)
            time.sleep(0.8)  # Simulate processing time
        
        status_text.text("Generation Complete!")
        st.session_state['nl_sql_results'] = get_demo_benchmarking_results()
        st.rerun()
        return
    
    # Real mode: original implementation
    state = st.session_state['benchmarking_state']
    df = state['active_csv_df']
    nl_col = state['csv_mapping']['nl_query']
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df)
    
    all_loaded_dfs = get_all_loaded_dfs()
    db_choice = st.session_state.get('db_choice', 'Unknown')
    db_connection_args = st.session_state.get('db_connection_args', {})

    for i, row in df.iterrows():
        nl_query = row[nl_col]
        status_text.text(f"Generating query {i+1}/{total_rows}...")
        
        with CO2Monitor(f"NL->SQL Gen - Row {i+1}") as monitor:
            try:
                full_prompt = create_sql_prompt(
                    dfs=all_loaded_dfs, user_question=nl_query, db_name=db_choice,
                    db_connection_args=db_connection_args
                )
                
                llm_args = {
                    "backend": state['llm_backend'], 
                    "prompt": full_prompt, 
                    "model_name": state['llm_model']
                }
                
                if not state['llm_backend']:
                    time.sleep(1)
                    generated_sql = f"SELECT * FROM table WHERE description LIKE '%{nl_query}%'"
                else:
                    raw_output = llm_adapters.generate(**llm_args)
                    generated_sql = extract_sql_query(raw_output) if isinstance(raw_output, str) else "ERROR"
                    
            except Exception as e:
                generated_sql = f"ERROR: {str(e)}"
        
        results.append({
            'nl_query': nl_query,
            'generated_sql': generated_sql,
            'latency_s': monitor.end_time - monitor.start_time,
            'co2_g': monitor.total_co2_g
        })
        progress_bar.progress((i + 1) / total_rows)
        
    status_text.text("Generation Complete!")
    st.session_state['nl_sql_results'] = pd.DataFrame(results)
    st.rerun()

def render_dbms_execution_tab():
    
    render_shared_setup(key_suffix="exec")
    state = st.session_state['benchmarking_state']
    
    st.markdown("### 1. Select Query Source")
    source = st.radio("Source", ["CSV Columns (Ground Truth)", "Generated Queries (Tab 2)"])
    
    queries_to_run = [] 
    
    if source == "CSV Columns (Ground Truth)":
        if state['active_csv_df'] is not None and state['csv_mapping'].get('dbms_cols'):
            dbms_cols = state['csv_mapping']['dbms_cols']
            selected_dbms = st.multiselect("Select DBMS Columns to Test", list(dbms_cols.keys()))
            
            if selected_dbms:
                for dbms in selected_dbms:
                    col_name = dbms_cols[dbms]
                    for sql in state['active_csv_df'][col_name]:
                        queries_to_run.append({'query': sql, 'dbms': dbms})
                st.info(f"Loaded {len(queries_to_run)} queries from CSV.")
        else:
            st.warning("No DBMS columns detected in CSV.")
            
    elif source == "Generated Queries (Tab 2)":
        if 'nl_sql_results' in st.session_state:
            res_df = st.session_state['nl_sql_results']
            target_db = st.session_state.get('db_choice', 'TargetDB')
            for sql in res_df['generated_sql']:
                queries_to_run.append({'query': sql, 'dbms': target_db})
            st.info(f"Loaded {len(queries_to_run)} generated queries.")
        else:
            st.warning("No generated queries found. Run Tab 2 first.")
            
    if queries_to_run:
        st.divider()
        if st.button("🚀 Execute Queries on DBMS", type="primary"):
            run_dbms_execution(queries_to_run)
            
        if 'dbms_exec_results' in st.session_state:
            st.markdown("### Results")
            res_df = st.session_state['dbms_exec_results']
            st.dataframe(res_df)
            
            if not res_df.empty:
                fig = px.box(res_df, x='dbms', y='latency_s', title="Latency Distribution per DBMS")
                st.plotly_chart(fig, use_container_width=True)
                
                fig2 = px.bar(res_df, x='dbms', y='co2_g', title="Total CO₂ per DBMS", barmode='group')
                st.plotly_chart(fig2, use_container_width=True)

def run_dbms_execution(queries):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(queries)
    
    db_choice = st.session_state.get('db_choice', 'Unknown')
    db_connection_args = st.session_state.get('db_connection_args', {})
    print('db_choice', db_choice)
    print('db_connection_args', db_connection_args)
    
    engine = None
    try:
        init_res = db_adapters.initialize_database(db_choice, db_connection_args)
        engine = init_res.get('engine')
    except:
        pass

    for i, item in enumerate(queries):
        sql = item['query']
        dbms = item['dbms']
        status_text.text(f"Executing query {i+1}/{total} on {dbms}...")
        
        with CO2Monitor(f"Exec Query {i+1} on {dbms}") as monitor:
            try:
                if engine:
                    res = db_adapters.execute_query(db_choice, engine, sql)
                    error = res.get('error')
                    rows = res.get('rows', 0)
                else:
                    time.sleep(0.1) 
                    error = None
                    rows = 100
                    
            except Exception as e:
                error = str(e)
                rows = 0
        
        results.append({
            'query': sql,
            'dbms': dbms,
            'latency_s': monitor.end_time - monitor.start_time,
            'co2_g': monitor.total_co2_g,
            'rows': rows,
            'error': error
        })
        progress_bar.progress((i + 1) / total)
        
    status_text.text("Execution Complete!")
    st.session_state['dbms_exec_results'] = pd.DataFrame(results)
    st.rerun()

def render_llm_quality_eval_tab():
    st.subheader("LLM Quality Evaluation")
    
    render_shared_setup(key_suffix="eval")
    state = st.session_state['benchmarking_state']
    
    if state['active_csv_df'] is not None and state['csv_mapping'].get('nl_query'):
        st.divider()
        st.markdown("### 1. Generate Candidates (if needed)")
        
        if st.button("Generate Candidates", help="Run LLM to generate SQL for evaluation"):
            run_nl_sql_benchmark() 
            
        if 'nl_sql_results' in st.session_state:
            st.success("Candidates available from Tab 2/Generation.")
            
            st.markdown("### 2. Evaluate Quality")
            
            dbms_cols = state['csv_mapping'].get('dbms_cols', {})
            if not dbms_cols:
                st.warning("No Ground Truth columns detected in CSV.")
            else:
                gt_dbms = st.selectbox("Select Ground Truth DBMS", list(dbms_cols.keys()), 
                key=f"bench_gt_dbms_{key_suffix}")
                gt_col = dbms_cols[gt_dbms]
                
                if st.button("📊 Calculate Metrics (Accuracy, Precision, Recall)", type="primary"):
                    run_quality_evaluation(gt_col)
                    
                if 'quality_eval_results' in st.session_state:
                    metrics = st.session_state['quality_eval_results']
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
                    c2.metric("Precision", f"{metrics['precision']:.2f}")
                    c3.metric("Recall", f"{metrics['recall']:.2f}")
                    c4.metric("F1 Score", f"{metrics['f1']:.2f}")
                    
                    st.markdown("#### Confusion Matrix")
                    cm = metrics['confusion_matrix']
                    st.write(cm)
                    
                    st.markdown("#### Detailed Results")
                    st.dataframe(metrics['details'])

def run_quality_evaluation(gt_col):
    state = st.session_state['benchmarking_state']
    generated_df = st.session_state['nl_sql_results']
    ground_truth_df = state['active_csv_df']
    
    results = []
    tp = 0
    fp = 0
    fn = 0
    
    for i, row in generated_df.iterrows():
        gen_sql = row['generated_sql'].strip().lower().replace(";", "")
        gt_sql = str(ground_truth_df.iloc[i][gt_col]).strip().lower().replace(";", "")
        
        is_correct = (gen_sql == gt_sql)
        
        status = "Correct" if is_correct else "Wrong"
        
        if is_correct:
            tp += 1
        else:
            fp += 1
            
        results.append({
            'nl_query': row['nl_query'],
            'generated_sql': row['generated_sql'],
            'ground_truth_sql': ground_truth_df.iloc[i][gt_col],
            'status': status
        })
        
    accuracy = (tp / len(results)) * 100 if results else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0 
    
    st.session_state['quality_eval_results'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall, 
        'f1': 0.0, 
        'confusion_matrix': {"Correct": tp, "Wrong": fp},
        'details': pd.DataFrame(results)
    }
    st.rerun()
