import streamlit as st
import pandas as pd
from datetime import datetime, timezone
import time
import threading
import traceback
from typing import Dict
import sys
import db_adapters
import llm_adapters
from utils.prompt_builder import create_sql_prompt, create_sql_optimization_prompt
from utils.query_cleaner import extract_sql_query
from utils.system_monitor_utilities import SystemMonitor
from utils.translations import get_text

def get_all_loaded_dfs() -> Dict[str, pd.DataFrame]:
    all_dfs = {}
    file_dfs = st.session_state.get('dataframes', {}).get('files', {})
    for name, data in file_dfs.items():
        if isinstance(data, dict) and 'df' in data and isinstance(data['df'], pd.DataFrame):
            all_dfs[name] = data['df']
    dbms_data = st.session_state.get('dataframes', {}).get('DBMS', {})
    for db_name, tables_list in dbms_data.items():
        if isinstance(tables_list, list):
            for table_info in tables_list:
                if isinstance(table_info, dict) and 'table_name' in table_info and 'table' in table_info:
                    all_dfs[table_info['table_name']] = table_info['table']
    return all_dfs


def _run_full_process_eval(result_holder, monitoring_data, user_question, all_loaded_dfs, db_choice, db_connection_args,
                           llm_backend, llm_model, cpu_tdp, emission_factor):

    db_engine = None
    try:
        if not all_loaded_dfs:
            raise ValueError("Nessun dato trovato da interrogare.")

        # --- FASE 1: GENERAZIONE QUERY (LLM) ---
        t_start_process = datetime.now(timezone.utc)
        result_holder['timestamps']['start_process'] = t_start_process
        
        full_prompt = create_sql_prompt(
            dfs=all_loaded_dfs, user_question=user_question, db_name=db_choice,
            db_connection_args=db_connection_args
        )
        llm_args = {"backend": llm_backend, "prompt": full_prompt, "model_name": llm_model}
        
        raw_llm_output = llm_adapters.generate(**llm_args)
        t_end_generation = datetime.now(timezone.utc)
        result_holder['timestamps']['end_generation'] = t_end_generation
        
        generated_sql = extract_sql_query(raw_llm_output) if isinstance(raw_llm_output, str) else None
        result_holder['info'] = {'raw_llm_output': raw_llm_output, 'generated_sql': generated_sql}

        if not generated_sql:
            error_msg = get_text("gen_eval", "no_sql_output")
            if isinstance(raw_llm_output, dict) and 'error' in raw_llm_output:
                error_msg = f"{get_text('gen_eval', 'exec_error')} {raw_llm_output['error']}"
            result_holder['info']['error'] = error_msg
            st.session_state.process_results = result_holder
            return 

        # --- FASE 2: ESECUZIONE QUERY (DB) ---
        # Inizializza DB
        db_init_result = db_adapters.initialize_database(db_choice=db_choice, connection_args=db_connection_args, msg="riga 64 utils_gen_eval_query")
        db_engine = db_init_result.get('engine')
        if db_init_result.get('error'): raise Exception(db_init_result.get('error'))

        # Carica dati (se necessario per SQLite/DuckDB in memory, altrimenti skip)
        # Nota: Per DB reali questo step potrebbe essere ridondante o lento, ma lo manteniamo come da logica originale
        with db_engine.connect() as connection:
            for table_name, df in all_loaded_dfs.items():
                try:
                    df.to_sql(table_name, connection.connection, if_exists='replace', index=False)
                except Exception:
                    pass # Gestione casi in cui to_sql non è supportato o necessario

        t_start_db_orig = datetime.now(timezone.utc)
        result_holder['timestamps']['start_db'] = t_start_db_orig
        
        query_result = db_adapters.execute_query(db_choice=db_choice, conn=db_engine, query=generated_sql)
        
        t_end_db_orig = datetime.now(timezone.utc)
        result_holder['timestamps']['end_db'] = t_end_db_orig
        
        duration_s_orig = (t_end_db_orig - t_start_db_orig).total_seconds()

        result_holder['info']['query_result'] = query_result
        result_holder['metrics'] = {"duration_s": duration_s_orig}
        st.session_state.process_results = result_holder
    except Exception as e:
        traceback.print_exc()
        result_holder.setdefault('info', {})['error'] = str(e)
    finally:
        if db_engine:
            db_engine.dispose()
        st.session_state.process_results = result_holder

def run_full_process_eval(user_question):
    
    llm_state = st.session_state.get('llm', {})
    llm_backend = llm_state.get('backend')
    llm_model = llm_state.get('model')

    db_state = st.session_state.get('db', {})
    db_choice = db_state.get('choice')
    db_connection_args = db_state.get('connection_args')

    st.session_state.process_status = 'running'
    st.session_state.process_results = {'timestamps': {}}
    st.session_state.monitoring_data = []
    st.session_state.greenefy_status = 'idle'

    st.session_state.stop_monitor_event = threading.Event()
    monitor = SystemMonitor(st.session_state.monitoring_data, st.session_state.stop_monitor_event,
                            st.session_state.get('emission_factor', 250.0), st.session_state.get('cpu_tdp', 65.0))
    st.session_state.monitor_thread = monitor
    monitor.start()

    worker_thread = threading.Thread(
        target=_run_full_process_eval,
        args=(
            st.session_state.process_results, 
            st.session_state.monitoring_data,
            user_question, get_all_loaded_dfs(), db_choice,
            db_connection_args, llm_backend,
            llm_model, st.session_state.get('cpu_tdp', 65.0),
            st.session_state.get('emission_factor', 250.0)
        ),
        daemon=True
    )
    st.session_state.process_thread = worker_thread
    worker_thread.start()
    st.rerun()

def _run_greenefy_process(result_holder, monitoring_data, user_question, all_loaded_dfs, db_choice, db_connection_args,
                          llm_backend, llm_model, original_query, original_query_co2):
    db_engine = None
    try:
        t_start_greenefy = datetime.now(timezone.utc)
        result_holder['timestamps']['start_greenefy'] = t_start_greenefy
        
        # --- FASE 3: GREENEFY (OTTIMIZZAZIONE) ---
        alt_prompt = create_sql_optimization_prompt(
            dfs=all_loaded_dfs, user_question=user_question, db_name=db_choice, 
            original_query=original_query, original_query_co2=original_query_co2
        )
        
        alt_raw_output = llm_adapters.generate(
            **{"backend": llm_backend, "prompt": alt_prompt, "model_name": llm_model}
        )
        
        # Estrazione query multiple (separate da ;)
        # Qui assumiamo che l'LLM ritorni le query separate da ; come richiesto nel prompt
        # Se l'LLM ritorna testo, proviamo a pulirlo
        
        candidates = []
        if isinstance(alt_raw_output, str):
            # Rimuovi markdown sql
            clean_output = alt_raw_output.replace("```sql", "").replace("```", "").strip()
            candidates = [q.strip() for q in clean_output.split(';') if q.strip()]
        
        result_holder['greenefy_candidates'] = candidates
        result_holder['greenefy_results'] = []

        if not candidates:
             result_holder['greenefy_error'] = "No alternative queries generated."
             return

        # Esecuzione candidati
        db_init_result = db_adapters.initialize_database(db_choice=db_choice, connection_args=db_connection_args, msg="riga 135 utils_gen_eval_query")
        db_engine = db_init_result.get('engine')
        
        # Ricarica dati se necessario (come sopra)
        with db_engine.connect() as connection:
             for table_name, df in all_loaded_dfs.items():
                try:
                    df.to_sql(table_name, connection.connection, if_exists='replace', index=False)
                except Exception:
                    pass

        for i, alt_sql in enumerate(candidates):
            if i >= 5: break # Max 5 tentativi
            
            res = {'sql': alt_sql, 'status': 'pending'}
            try:
                t_start_alt = time.time()
                alt_query_result = db_adapters.execute_query(db_choice=db_choice, conn=db_engine, query=alt_sql)
                duration_alt = time.time() - t_start_alt
                
                if alt_query_result.get('error'):
                    res['status'] = 'error'
                    res['error'] = alt_query_result.get('error')
                else:
                    res['status'] = 'success'
                    res['rows'] = alt_query_result.get('rows')
                    res['duration'] = duration_alt
                    res['result'] = alt_query_result
            except Exception as e:
                res['status'] = 'error'
                res['error'] = str(e)
            
            result_holder['greenefy_results'].append(res)

        t_end_greenefy = datetime.now(timezone.utc)
        result_holder['timestamps']['end_greenefy'] = t_end_greenefy

    except Exception as e:
        traceback.print_exc()
        result_holder['greenefy_error'] = str(e)
    finally:
        if db_engine:
            db_engine.dispose()

def run_greenefy_process():
    st.session_state.greenefy_status = 'running'
    
    # Se il monitor era stato fermato, riavvialo
    if not st.session_state.get('monitor_thread') or not st.session_state.monitor_thread.is_alive():
            st.session_state.stop_monitor_event = threading.Event()
            monitor = SystemMonitor(st.session_state.monitoring_data, st.session_state.stop_monitor_event,
                                st.session_state.get('emission_factor', 250.0), st.session_state.get('cpu_tdp', 65.0))
            st.session_state.monitor_thread = monitor
            monitor.start()

    worker_thread = threading.Thread(
        target=_run_greenefy_process,
        args=(
            st.session_state.process_results, st.session_state.monitoring_data,
            st.session_state.gen_prompt, get_all_loaded_dfs(), st.session_state.db_choice,
            st.session_state.db_connection_args, st.session_state.llm.get('backend'),
            st.session_state.llm.get('model'), generated_sql, orig_co2
        ),
        daemon=True
    )
    st.session_state.greenefy_thread = worker_thread
    worker_thread.start()
    st.rerun()


def dataset_tab_geneval(key_alter="", loaded_databases=None, show_dataset=True):
    """
    Visualizza i dettagli per TUTTI i database DBMS caricati in session_state.
    """

    if not loaded_databases:
        st.info("Nessun database (da DBMS) è stato ancora caricato.")
        return

    with st.container(border=False):
        for db_name, tables_data in loaded_databases.items():

            with st.expander(f"📁 Database: {db_name}", expanded=show_dataset):
                tab_names = []
                tab_dfs = []

                if not tables_data:
                    st.warning(f"Nessuna tabella trovata o caricata per il database '{db_name}'.")
                    continue

                for db_dict in tables_data:
                    tab_names.append(db_dict["table_name"])
                    tab_dfs.append(db_dict["table"])

                tabs = st.tabs(tab_names)

                for name, tab, df in zip(tab_names, tabs, tab_dfs):
                    with tab:
                        if df is not None and df.shape[0] > 0:
                            with st.container(border=False):
                                st.write('🔍 Preview Dataset')
                                st.write(df.head(5))
                        else:
                            st.error(f'Table {name} in dataset {db_name} non trovato!')
