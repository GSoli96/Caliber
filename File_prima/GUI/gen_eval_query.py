# GUI/gen_eval_query.py

import time

from GUI.eco_dashboard import display_eco_dashboard
from utils.sustainability_report import (
    generate_sustainability_certificate,
    calculate_session_metrics,
    format_relatable_metrics
)

from utils.utils_gen_eval_query import dataset_tab_geneval, run_full_process_eval, run_greenefy_process
# Aggiungere import
from GUI.green_ai_race_tab import green_ai_race_tab
import threading
import pandas as pd
import streamlit as st
from spacy import load as spacy_load
from datetime import datetime, timezone
from typing import Dict, List
import json
import db_adapters
import llm_adapters
from charts.plotly_charts import (
    generate_usage_chart, generate_power_chart, generate_cumulative_co2_chart,
    generate_co2_rate_chart, generate_phase_co2_comparison_chart
)
from GUI.green_optimizer_display import display_green_optimizer_results

from utils.prompt_builder import create_sql_prompt, create_sql_optimization_prompt
from utils.query_cleaner import extract_sql_query
from utils.history_manager import add_history_entry
from utils.system_monitor_utilities import get_dynamic_system_info, SystemMonitor
from GUI.load_dataset_gui import dataset_tab_dbms
from utils.load_config import get_num_alternative_queries
import traceback
from utils.translations import get_text
import streamlit as st
from utils import green_metrics
from utils.icons import Icons


# Icone per gli adapter LLM
LLM_ADAPTER_ICONS = Icons.ICONS

# Icone per gli adapter LLM
UI_ICONS = Icons.ICONS

def _display_results_eval():
    results = st.session_state.get('process_results', {})
    info = results.get('info', {})
    timestamps = results.get('timestamps', {}) 
    monitoring_data = st.session_state.get('monitoring_data', [])


    # --- ERRORI PRINCIPALI ---
    if info.get('error'):
        st.error(f"{get_text('gen_eval', 'exec_error')}\n\n{info['error']}")
        return

    generated_sql = info.get("generated_sql")
    query_res = info.get("query_result", {})

    # --- VISUALIZZAZIONE QUERY E RISULTATI (Restructured) ---
    st.subheader(get_text("gen_eval", "original_results"))

    # 1. Query SQL Generata (Full Width)
    st.markdown(f"##### {get_text('gen_eval', 'generated_sql')}")
    st.code(generated_sql or "N/A", language="sql")

    # 2. Risultato Esecuzione (Messaggio Status)
    if query_res.get('error'):
        st.error(f"{get_text('gen_eval', 'db_error')} {query_res['error']}")
    else:
        st.success(get_text("gen_eval", "query_executed"))

    # 3. Metriche: Tempo LLM e Metadati Dataset
    col_time, col_meta = st.columns(2)

    with col_time:
        # Calcolo tempo generazione LLM
        llm_duration_s = 0.0
        if 'start_process' in timestamps and 'end_generation' in timestamps:
            try:
                start = timestamps['start_process']
                end = timestamps['end_generation']
                if isinstance(start, str): start = pd.to_datetime(start)
                if isinstance(end, str): end = pd.to_datetime(end)
                llm_duration_s = (end - start).total_seconds()
            except Exception:
                pass
        st.metric("LLM Generation Time", f"{llm_duration_s:.2f} s")

    with col_meta:
        # Metadati Dataset
        db_choice = st.session_state.get('db_choice', 'Unknown DB')
        rows_count = query_res.get('rows', 'N/A')
        cols_count = "N/A"
        if 'data' in query_res and isinstance(query_res['data'], pd.DataFrame):
            cols_count = query_res['data'].shape[1]
        
        st.markdown(f"**Dataset:** {db_choice}")
        st.markdown(f"**Rows:** {rows_count} | **Cols:** {cols_count}")

    # View Result Data Expander
    if 'data' in query_res and isinstance(query_res['data'], pd.DataFrame):
         with st.expander("View Result Data"):
             st.dataframe(query_res['data'])

    # Prima dei plot consumi:
    st.markdown("---")
    display_eco_dashboard(monitoring_data, show_live=False)
    st.markdown("---")

    # --- PLOT CONSUMI ---
    if monitoring_data:
        monitoring_df = pd.json_normalize(monitoring_data)
        monitoring_df['timestamp'] = pd.to_datetime(monitoring_df['timestamp'])
        monitoring_df.rename(
            columns={'cpu.percent': 'cpu_util_percent', 'cpu.power_w': 'cpu_power_w', 'gpu.percent': 'gpu_util_percent',
                     'gpu.power_w': 'gpu_power_w'}, inplace=True, errors='ignore')

        # Calcolo metriche CO2
        monitoring_df['time_diff_s'] = monitoring_df['timestamp'].diff().dt.total_seconds().fillna(0)
        monitoring_df['total_co2_gs'] = monitoring_df.get('cpu.co2_gs_cpu', 0).fillna(0)
        if 'gpu.co2_gs_gpu' in monitoring_df.columns:
            monitoring_df['total_co2_gs'] += monitoring_df['gpu.co2_gs_gpu'].fillna(0)
        monitoring_df['cumulative_gco2'] = (monitoring_df['total_co2_gs'] * monitoring_df['time_diff_s']).cumsum()

        # --- ANNOTAZIONI ---
        annotations = []
        if 'start_process' in timestamps: annotations.append({'x': timestamps['start_process'], 'text': 'Start Gen'})
        if 'end_generation' in timestamps: annotations.append({'x': timestamps['end_generation'], 'text': 'End Gen / Start DB'})
        if 'end_db' in timestamps: annotations.append({'x': timestamps['end_db'], 'text': 'End DB'})
        if 'start_greenefy' in timestamps: annotations.append({'x': timestamps['start_greenefy'], 'text': 'Start Greenefy'})
        if 'end_greenefy' in timestamps: annotations.append({'x': timestamps['end_greenefy'], 'text': 'End Greenefy'})

        # --- 1. CONSUMI GENERALI ---
        with st.expander("General Consumption", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(generate_usage_chart(monitoring_df, annotations), use_container_width=True)
            with c2:
                st.plotly_chart(generate_co2_rate_chart(monitoring_df, annotations), use_container_width=True)

        # --- 2. GENERAZIONE QUERY ---
        if 'start_process' in timestamps and 'end_generation' in timestamps:
            gen_df = monitoring_df[(monitoring_df['timestamp'] >= timestamps['start_process']) & (monitoring_df['timestamp'] <= timestamps['end_generation'])]
            if not gen_df.empty:
                with st.expander("Query Generation Phase", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(generate_usage_chart(gen_df), use_container_width=True)
                    with c2: st.plotly_chart(generate_co2_rate_chart(gen_df), use_container_width=True)

        # --- 3. ESECUZIONE DB ---
        if 'start_db' in timestamps and 'end_db' in timestamps:
            db_df = monitoring_df[(monitoring_df['timestamp'] >= timestamps['start_db']) & (monitoring_df['timestamp'] <= timestamps['end_db'])]
            if not db_df.empty:
                with st.expander("DB Execution Phase", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(generate_usage_chart(db_df), use_container_width=True)
                    with c2: st.plotly_chart(generate_co2_rate_chart(db_df), use_container_width=True)
        
        # --- 4. GREENEFY ---
        if 'start_greenefy' in timestamps and 'end_greenefy' in timestamps:
             green_df = monitoring_df[(monitoring_df['timestamp'] >= timestamps['start_greenefy']) & (monitoring_df['timestamp'] <= timestamps['end_greenefy'])]
             if not green_df.empty:
                with st.expander("Greenefy Phase", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(generate_usage_chart(green_df), use_container_width=True)
                    with c2: st.plotly_chart(generate_co2_rate_chart(green_df), use_container_width=True)

    # --- GREENEFY BUTTON & RESULTS ---
    st.divider()
    
    # Calcola CO2 query originale per passarlo a Greenefy
    orig_co2 = 0.0
    if monitoring_data and 'start_db' in timestamps and 'end_db' in timestamps:
         # Stima approssimativa basata sul monitoring
         pass 

    if 'greenefy_results' not in results:
        if st.button("🌱 Greenefy", type="primary", help="Optimize query for CO2"):
            run_greenefy_process()
            
    
    if 'greenefy_results' in results:
        display_green_optimizer_results(results, info, query_res, timestamps, monitoring_data)

    with st.container(border=False):
        loaded_databases = st.session_state["dataframes"]["DBMS"]

        if not loaded_databases:
            st.info("Nessun database (da DBMS) è stato ancora caricato.")
        else:
            for db_name, tables_data in loaded_databases.items():
                with st.expander(f"📁 Database: {db_name}", expanded=False):
                    tab_names = []
                    tab_dfs = []
                    if not tables_data:
                        st.warning(get_text("gen_eval", "no_table_found", db_name=db_name))
                        continue
                    for db_dict in tables_data:
                        tab_names.append(db_dict["table_name"])
                        tab_dfs.append(db_dict["table"])
                    tabs = st.tabs(tab_names)
                    for name, tab, df in zip(tab_names, tabs, tab_dfs):
                        with tab:
                            if df is not None and df.shape[0] > 0:
                                st.write(get_text("gen_eval", "preview_dataset"))
                                st.write(df.head(5))
                            else:
                                st.error(get_text("gen_eval", "table_not_found", name=name, db_name=db_name))
        
    # Alla fine di _display_results_eval, aggiungere:
    st.markdown("### 📊 Sustainability Metrics")
    # Mostra metriche comprensibili
    if monitoring_data:
        try:
            mon_df = pd.json_normalize(monitoring_data)
            mon_df['timestamp'] = pd.to_datetime(mon_df['timestamp'])
            mon_df['time_diff_s'] = mon_df['timestamp'].diff().dt.total_seconds().fillna(0)
            mon_df['total_co2_gs'] = mon_df.get('cpu.co2_gs_cpu', 0).fillna(0)
            if 'gpu.co2_gs_gpu' in mon_df.columns:
                mon_df['total_co2_gs'] += mon_df['gpu.co2_gs_gpu'].fillna(0)
            total_co2 = (mon_df['total_co2_gs'] * mon_df['time_diff_s']).sum()
        except Exception:
            total_co2 = 0.0
        relatable = format_relatable_metrics(total_co2)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📱 Smartphones", relatable['smartphones'])
        with col2:
            st.metric("🚗 Car Distance", relatable['car_distance'])
        with col3:
            st.metric("💡 LED Hours", relatable['lightbulb_hours'])
    # Bottone per scaricare certificato
    if st.button("📊 Download Green Certificate", type="primary"):
        session_data = calculate_session_metrics([monitoring_data])
        pdf_bytes = generate_sustainability_certificate(session_data)
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=f"green_certificate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

def query_gen_eval_tab():
    llm_state = st.session_state.get('llm', {})

    if st.session_state['create_db_done'] == True:
        if len(list(st.session_state["dataframes"]["DBMS"].keys())) > 0:
            dataset_tab_geneval("geneval")
    else:
        st.info(get_text("gen_eval", "please_load_dataset"))

    if llm_state['status'] == 'notLoad':
        st.info(get_text("gen_eval", "please_load_llm"))
    elif llm_state['status'] == 'loaded':
        backend_icon = LLM_ADAPTER_ICONS.get(llm_state["backend"], "🤖")
        status_icon = "✅" if llm_state["status"] == "loaded" else "⚠️"
        
        with st.expander(f"### {UI_ICONS['Details']} {get_text('gen_eval', 'selected_model')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True):
                    st.markdown(f"<div style='text-align:center;'><strong>{backend_icon} {get_text('gen_eval', 'backend')}</strong><br>{llm_state['backend']}</div>", unsafe_allow_html=True)
            with col2:
                with st.container(border=True):
                    st.markdown(f"<div style='text-align:center;'><strong>🎯 {get_text('gen_eval', 'model')}</strong><br>{llm_state['model']}</div>", unsafe_allow_html=True)
            with col3:
                with st.container(border=True):
                    st.markdown(f"<div style='text-align:center;'><strong>{status_icon} {get_text('gen_eval', 'status')}</strong><br>{llm_state['status'].capitalize()}</div>", unsafe_allow_html=True)

    user_prompt = st.text_area(get_text("gen_eval", "describe_request"), key="gen_prompt", height=120)
    submit = (user_prompt != '' and st.session_state['create_db_done'] and llm_state['status'] != 'notLoad')

    c1, c2, c3 = st.columns(3)
    with c1:
        run_button = st.button(get_text("gen_eval", "generate_btn"), type="primary", disabled=not submit)
    with c3:
        run_spacy = st.button(get_text("gen_eval", "analyze_spacy"))

    if run_spacy and user_prompt.strip():
        # ... (Logica spaCy esistente) ...
        pass

    output_placeholder = st.empty()

    # --- AVVIO PROCESSO PRINCIPALE ---
    if run_button:
        run_full_process_eval(user_prompt)

    # --- GESTIONE STATO RUNNING ---
    if st.session_state.get('process_status') == 'running':
        with output_placeholder.container():
            st.info(get_text("gen_eval", "running_msg"))
            # Visualizzazione live (opzionale, per ora semplificata)
            
        worker_thread = st.session_state.get('process_thread')
        if worker_thread and not worker_thread.is_alive():
            # Il thread principale è finito. 
            # NON fermiamo il monitor se vogliamo che continui per Greenefy? 
            # Ma Greenefy è su richiesta utente. Quindi fermiamo il monitor ora.
            if st.session_state.get('stop_monitor_event'): st.session_state.stop_monitor_event.set()
            if st.session_state.get('monitor_thread'): st.session_state.monitor_thread.join(timeout=1)
            
            st.session_state.process_status = 'done'
            st.rerun()
        else:
            time.sleep(0.5)
            st.rerun()

    # --- GESTIONE STATO GREENEFY RUNNING ---
    if st.session_state.get('greenefy_status') == 'running':
        with output_placeholder.container():
            st.info("🌱 Greenefy optimization in progress...")
            # Mostra risultati parziali
            _display_results_eval()

        green_thread = st.session_state.get('greenefy_thread')
        if green_thread and not green_thread.is_alive():
            if st.session_state.get('stop_monitor_event'): st.session_state.stop_monitor_event.set()
            if st.session_state.get('monitor_thread'): st.session_state.monitor_thread.join(timeout=1)
            
            st.session_state.greenefy_status = 'done'
            st.rerun()
        else:
            time.sleep(0.5)
            st.rerun()

    # --- VISUALIZZAZIONE RISULTATI FINALI ---
    if st.session_state.get('process_status') == 'done' or st.session_state.get('greenefy_status') == 'done':
        with output_placeholder.container():
            _display_results_eval()


