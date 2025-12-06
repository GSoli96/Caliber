import numpy as np
import streamlit as st

from GUI.relational_profiling_tab import ui_profiling_relazionale, ui_integrita_dataset, missing_value_tab
from utils.translations import get_text
from utils.symbols import symbols
import sys
import os
import re
import streamlit as st
import numpy as np
import streamlit as st

from GUI.relational_profiling_tab import ui_profiling_relazionale, ui_integrita_dataset, missing_value_tab
from utils.translations import get_text
from utils.symbols import symbols
import sys
import os
import re
import streamlit as st
from itertools import count
import db_adapters
from GUI.dataset_explore_gui import info_dataset
from db_adapters.DBManager import DBManager
from utils.load_data import load_data_files
# from GUI.relational_profiling_app import ui_profiling_relazionale, ui_integrita_dataset, ui_export
from GUI.message_gui import st_toast_temp
from utils.translations import get_text
from utils.symbols import symbols
import zipfile
import io
import pandas as pd
import json
import time
from GUI.dataset_explore_gui import get_column_category, TYPE_INFO
from llm_adapters.sensitive_entity import is_sensitive_column

sep_options = symbols.sep_options

def db_analytics_tab():
    # Display uploaded files in tabs
    if len(list(st.session_state["dataframes"]["DBMS"].keys())) > 0:
        """Visualizza i dettagli per TUTTI i database DBMS caricati in session_state."""
        loaded_databases = st.session_state["dataframes"]["DBMS"]

        if not loaded_databases:
            st.info(get_text("load_dataset", "no_db_loaded"))
            return

        with st.container(border=True):
            for db_name, tables_data in loaded_databases.items():

                st.header(f"📁 Database: {db_name}")

                config_dict = st.session_state.get('uploaded_dbms', {}).get(db_name)

                with st.expander(get_text("load_dataset", "db_info"), expanded=True):
                    # 4. Funzione modulare per mostrare le info
                    _display_db_info(config_dict, db_name, tables_data)

                with st.expander(get_text("load_dataset", "explore_tables"), expanded=True):

                    # 5. Logica esistente per creare i tab delle tabelle
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
                                # Use deterministic key for export compatibility
                                key_alter = f"{db_name}_{name}"
                                show_df_details(df, name, key_alter)

                            else:
                                st.error(f'Table {name} in dataset {db_name} non trovato!')

        with st.container(border=True):
            st.subheader(get_text("load_dataset", "export_header"))
            if st.button(get_text("load_dataset", "export_zip_btn"), key="btn_export_analytics_zip"):
                with st.spinner(get_text("load_dataset", "exporting_zip")):
                    zip_bytes, zip_name = export_analytics_zip(loaded_databases)
                    st.download_button(label=get_text("load_dataset", "download_zip"), data=zip_bytes,
                                       file_name=zip_name, mime="application/zip")
        st.warning(get_text("load_dataset", "config_metadata_missing", db_name=db_name))
        return

        with st.container(border=False):

            dbms_type = config_dict.get('db_choice')  # 'db_choice' è la chiave usata da upload_dbms

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(get_text("load_dataset", "dbms_type"), dbms_type or "N/A")

            with col2:
                st.metric(get_text("load_dataset", "tables_loaded"), len(tables_data))

            if dbms_type == "SQLite":
                db_path = config_dict.get('path_to_file') or config_dict.get('db_path')
                if db_path:
                    st.text_input(get_text("load_dataset", "db_path"), db_path, disabled=True,
                                key=f"path_{db_name}_{key_alter}")

                    try:
                        if os.path.exists(db_path):
                            file_size_bytes = os.path.getsize(db_path)
                            file_size_mb = file_size_bytes / (1024 * 1024)
                            with col3:
                                st.metric(get_text("load_dataset", "weight"), f"{file_size_mb:.2f} MB")
                        else:
                            st.caption(get_text("load_dataset", "file_moved"))
                    except Exception as e:
                        st.caption(get_text("load_dataset", "calc_weight_error"))
                else:
                    st.info(get_text("load_dataset", "sqlite_path_missing"))

            # --- Blocco per altri DBMS (MySQL, ecc.) ---
            else:
                conn_str = config_dict.get('conn_str')
                if conn_str:
                    masked_str = re.sub(r"password=([^&@]+)", "password=********", conn_str, flags=re.IGNORECASE)
                    st.text_input(get_text("load_dataset", "conn_string_label"), masked_str, disabled=True,
                                key=f"conn_{db_name}_{dbms_type}_{key_alter}")
                else:
                    st.info(get_text("load_dataset", "conn_string_unavailable"))

            tb_list_config = config_dict.get("table_list", [])
            if isinstance(tb_list_config, list) and len(tb_list_config) > 0:
                st.caption(get_text("load_dataset", "config_req_tables", n=len(tb_list_config)))
            else:
                st.caption(get_text("load_dataset", "config_req_all"))
    else:
        st.info('Please load a dataset first.')
def show_df_details(df, name, key_alter):
    """Displays detailed information about a dataframe in tabs."""
    st.markdown("""
        <style>
        /* Selettore per la barra delle tab */
        div[data-baseweb="tab-list"] {
            justify-content: space-between;
        }

        /* Seleziona i singoli tab e cambia ordine */
        div[data-baseweb="tab-list"] button:nth-child(1) { order: 1; }  /* LLM 1 */
        div[data-baseweb="tab-list"] button:nth-child(2) { order: 2; }  /* LLM 2 */
        div[data-baseweb="tab-list"] button:nth-child(3) { order: 3; }  /* LLM 3 */
        div[data-baseweb="tab-list"] button:nth-child(4) { order: 4; }  /* LLM 3 */
        div[data-baseweb="tab-list"] button:nth-child(5) { order: 5; }  /* LLM 4 */
        div[data-baseweb="tab-list"] button:nth-child(6) { order: 9; }  /* LLM 4 */
        </style>
    """, unsafe_allow_html=True)

    tab2_detalied, tab4_missing_profiling = st.tabs([
        get_text("load_dataset", "tab_detailed"),
        get_text("load_dataset", "tab_missing&profiling"),
    ])

    with tab2_detalied:
        st.subheader(get_text("load_dataset", "dataset_details"))

        # metriche principali (aggiungo % missing e duplicati)
        total_missing = int(df.isna().sum().sum())
        pct_missing = float(total_missing / (df.shape[0] * max(1, df.shape[1])) * 100.0) if df.size else 0.0
        dup_rows = int(df.duplicated().sum())
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(get_text("load_dataset", "num_rows"), f"{df.shape[0]:,}")
        c2.metric(get_text("load_dataset", "num_cols"), f"{df.shape[1]:,}")
        c3.metric(get_text("load_dataset", "missing_values"), f"{pct_missing:.0f}%")
        c4.metric(get_text("load_dataset", "duplicate_rows"), f"{dup_rows:,}")
        c5.metric(get_text("load_dataset", "mem_usage"), f"~{mem_mb:.2f} MB")

        with st.expander(get_text('gen_eval', 'preview_dataset'), expanded=False):
            rows_to_show = st.number_input(
                get_text("load_dataset", "rows_to_show"),
                min_value=1,
                max_value=len(df),
                value=min(5, len(df)),
                step=1,
                help=get_text("load_dataset", "rows_to_show_help"),
                key=f'numberInput_preview_{key_alter}',
            )
            st.write(df.head(rows_to_to_show))

        st.subheader(get_text("load_dataset", "dataset_specs"))
        info_dataset(df, key=f'Info_db_{name}')


    with tab4_missing_profiling:
        tab4_profiling, t_missing, tab5_integrita = st.tabs([
            get_text("load_dataset", "tab_profiling"),
            get_text("load_dataset", "missing_values"),
            get_text("load_dataset", "tab_integrity"),])

        with tab4_profiling:
            ui_profiling_relazionale(
                df,
                key=f"prof_{key_alter}",  # chiave per questa sezione
                name=name,
                related_tables=None  # opzionale
            )
        # ---------- MISSING ----------
        with t_missing:
            missing_value_tab(df, key=f"{name}_{key_alter}")

        with tab5_integrita:
            ui_integrita_dataset(
                df,
                key=f"intg_{key_alter}",
                name=name,
                related_tables=None,
            )
    if not loaded_databases:
        st.info(get_text("load_dataset", "no_db_loaded"))
        return

    with st.container(border=True):
        for db_name, tables_data in loaded_databases.items():

            st.header(f"📁 Database: {db_name}")

            # 3. Recupera la configurazione specifica per *questo* database
            #    Questo funziona grazie alle modifiche al prerequisito
            config_dict = st.session_state.get('uploaded_dbms', {}).get(db_name)

            with st.expander(get_text("load_dataset", "db_info"), expanded=True):
                # 4. Funzione modulare per mostrare le info
                _display_db_info(config_dict, db_name, tables_data)

            with st.expander(get_text("load_dataset", "explore_tables"), expanded=True):

                # 5. Logica esistente per creare i tab delle tabelle
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
                            # Use deterministic key for export compatibility
                            key_alter = f"{db_name}_{name}"
                            show_df_details(df, name, key_alter)
                        else:
                            st.toast(f'Table {name} in dataset {db_name} non trovato!')

    with st.container(border=True):
        st.subheader(get_text("load_dataset", "export_header"))
        st.session_state.generate_zip = False

        col_spacer_left, col_btn1, col_btn2, col_spacer_right = st.columns([1, 2, 2, 1])

        with col_btn1:
            genera = st.button(
                get_text("load_dataset", "export_zip_btn"),
                key="btn_export_analytics_zip",
                disabled=st.session_state.generate_zip
            )

        with col_btn2:
            if st.session_state.generate_zip:
                with st.spinner(get_text("load_dataset", "exporting_zip")):
                    zip_bytes, zip_name = export_analytics_zip(loaded_databases)
                    st.session_state.generate_zip = True
        
                st.download_button(
                    label=get_text("load_dataset", "download_zip"),
                    data=zip_bytes,
                    file_name=zip_name,
                    mime="application/zip",
                    key="download_analytics_zip_btn"
                )



def _display_db_info(config_dict, db_name, tables_data):
    """Helper function to display DBMS configuration details."""
    key_alter = ""
    if not config_dict:
        st.warning(get_text("load_dataset", "config_metadata_missing", db_name=db_name))
        return

    with st.container(border=False):

        dbms_type = config_dict.get('db_choice')  # 'db_choice' è la chiave usata da upload_dbms

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(get_text("load_dataset", "dbms_type"), dbms_type or "N/A")

        with col2:
            st.metric(get_text("load_dataset", "tables_loaded"), len(tables_data))

        # --- Blocco per SQLite ---
        if dbms_type == "SQLite":
            # 'path_to_file' (Tab 2) o 'db_path' (Tab 1)
            db_path = config_dict.get('path_to_file') or config_dict.get('db_path')
            if db_path:
                st.text_input(get_text("load_dataset", "db_path"), db_path, disabled=True,
                              key=f"path_{db_name}_{key_alter}")

                # Prova a ottenere la dimensione del file
                try:
                    if os.path.exists(db_path):
                        file_size_bytes = os.path.getsize(db_path)
                        file_size_mb = file_size_bytes / (1024 * 1024)
                        with col3:
                            st.metric(get_text("load_dataset", "weight"), f"{file_size_mb:.2f} MB")
                    else:
                        st.caption(get_text("load_dataset", "file_moved"))
                except Exception as e:
                    st.caption(get_text("load_dataset", "calc_weight_error"))
            else:
                st.info(get_text("load_dataset", "sqlite_path_missing"))

        # --- Blocco per altri DBMS (MySQL, ecc.) ---
        else:
            conn_str = config_dict.get('conn_str')  # 'conn_str' è la chiave usata da upload_dbms
            if conn_str:
                # Maschera la password per sicurezza
                masked_str = re.sub(r"password=([^&@]+)", "password=********", conn_str, flags=re.IGNORECASE)
                st.text_input(get_text("load_dataset", "conn_string_label"), masked_str, disabled=True,
                              key=f"conn_{db_name}_{dbms_type}_{key_alter}")
            else:
                st.info(get_text("load_dataset", "conn_string_unavailable"))

        # Mostra le tabelle che dovevano essere caricate (dalla config)
        tb_list_config = config_dict.get("table_list", [])
        if isinstance(tb_list_config, list) and len(tb_list_config) > 0:
            st.caption(get_text("load_dataset", "config_req_tables", n=len(tb_list_config)))
        else:
            st.caption(get_text("load_dataset", "config_req_all"))

def show_df_details(df, name, key_alter):
    """Displays detailed information about a dataframe in tabs."""
    st.markdown("""
        <style>
        /* Selettore per la barra delle tab */
        div[data-baseweb="tab-list"] {
            justify-content: space-between;
        }

        /* Seleziona i singoli tab e cambia ordine */
        div[data-baseweb="tab-list"] button:nth-child(1) { order: 1; }  /* LLM 1 */
        div[data-baseweb="tab-list"] button:nth-child(2) { order: 2; }  /* LLM 2 */
        div[data-baseweb="tab-list"] button:nth-child(3) { order: 3; }  /* LLM 3 */
        div[data-baseweb="tab-list"] button:nth-child(4) { order: 4; }  /* LLM 3 */
        div[data-baseweb="tab-list"] button:nth-child(5) { order: 5; }  /* LLM 4 */
        div[data-baseweb="tab-list"] button:nth-child(6) { order: 9; }  /* LLM 4 */
        </style>
    """, unsafe_allow_html=True)

    with st.container(border=False):

        tab2_detalied, tab4_missing_profiling = st.tabs([
            get_text("load_dataset", "tab_detailed"),
            get_text("load_dataset", "tab_missing&profiling"),
        ])

        with tab2_detalied:
            st.subheader(get_text("load_dataset", "dataset_details"))

            # metriche principali (aggiungo % missing e duplicati)
            total_missing = int(df.isna().sum().sum())
            pct_missing = float(total_missing / (df.shape[0] * max(1, df.shape[1])) * 100.0) if df.size else 0.0
            dup_rows = int(df.duplicated().sum())
            mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(get_text("load_dataset", "num_rows"), f"{df.shape[0]:,}")
            c2.metric(get_text("load_dataset", "num_cols"), f"{df.shape[1]:,}")
            c3.metric(get_text("load_dataset", "missing_values"), f"{pct_missing:.0f}%")
            c4.metric(get_text("load_dataset", "duplicate_rows"), f"{dup_rows:,}")
            c5.metric(get_text("load_dataset", "mem_usage"), f"~{mem_mb:.2f} MB")

            with st.expander(get_text('gen_eval', 'preview_dataset'), expanded=False):
                rows_to_show = st.number_input(
                    get_text("load_dataset", "rows_to_show"),
                    min_value=1,
                    max_value=len(df),
                    value=min(5, len(df)),
                    step=1,
                    help=get_text("load_dataset", "rows_to_show_help"),
                    key=f'numberInput_preview_{key_alter}',
                )
                st.write(df.head(rows_to_show))

            st.subheader(get_text("load_dataset", "dataset_specs"))
            info_dataset(df, key=f'Info_db_{name}')


        with tab4_missing_profiling:
            tab4_profiling, t_missing, tab5_integrita = st.tabs([
                get_text("load_dataset", "tab_profiling"),
                get_text("load_dataset", "missing_values"),
                get_text("load_dataset", "tab_integrity"),])

            with tab4_profiling:
                ui_profiling_relazionale(
                    df,
                    key=f"prof_{key_alter}",  # chiave per questa sezione
                    name=name,
                    related_tables=None  # opzionale
                )
            # ---------- MISSING ----------
            with t_missing:
                missing_value_tab(df, key=f"{name}_{key_alter}")

            with tab5_integrita:
                ui_integrita_dataset(
                    df,
                    name=name,
                    key=f"intg_{key_alter}",  # chiave per questa sezione
                )


def _aggregate_metrics(metrics_data):
    """
    Aggregate resource monitoring metrics into a summary DataFrame.
    
    Args:
        metrics_data: List of metric dicts from SystemMonitor
    
    Returns:
        pandas.DataFrame with aggregated metrics or None
    """
    if not metrics_data:
        return None
    
    try:
        cpu_percents = [m.get('cpu', {}).get('percent', 0) for m in metrics_data]
        gpu_percents = [m.get('gpu', {}).get('percent', 0) for m in metrics_data]
        co2_gs_cpu = [m.get('cpu', {}).get('co2_gs_cpu', 0) for m in metrics_data]
        co2_gs_gpu = [m.get('gpu', {}).get('co2_gs_gpu', 0) for m in metrics_data]
        co2_total = [cpu + gpu for cpu, gpu in zip(co2_gs_cpu, co2_gs_gpu)]
        
        summary = {
            'Metric': ['CPU %', 'GPU %', 'CO2 (g/s)'],
            'Min': [
                round(min(cpu_percents), 2) if cpu_percents else 0,
                round(min(gpu_percents), 2) if any(gpu_percents) else 0,
                round(min(co2_total), 6) if co2_total else 0
            ],
            'Max': [
                round(max(cpu_percents), 2) if cpu_percents else 0,
                round(max(gpu_percents), 2) if any(gpu_percents) else 0,
                round(max(co2_total), 6) if co2_total else 0
            ],
            'Mean': [
                round(sum(cpu_percents) / len(cpu_percents), 2) if cpu_percents else 0,
                round(sum(gpu_percents) / len(gpu_percents), 2) if any(gpu_percents) else 0,
                round(sum(co2_total) / len(co2_total), 6) if co2_total else 0
            ],
            'Total_CO2_g': [
                0,  # N/A for CPU%
                0,  # N/A for GPU%
                round(sum(co2_total) * 0.5, 4)  # Total CO2 (0.5s sampling interval)
            ]
        }
        
        return pd.DataFrame(summary)
    except Exception:
        return None


def export_analytics_zip(loaded_databases):
    """
    Genera un file ZIP contenente tutte le statistiche e i risultati del profiling
    per i database caricati.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        
        # Iteriamo su tutti i database
        for db_name, tables_data in loaded_databases.items():
            # Configurazione DB
            config_dict = st.session_state.get('uploaded_dbms', {}).get(db_name, {})
            db_info = {
                "db_name": db_name,
                "dbms_type": config_dict.get('db_choice', 'Unknown'),
                "tables_count": len(tables_data),
                "tables_list": [t["table_name"] for t in tables_data],
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Salva info DB
            z.writestr(f"{db_name}/db_summary.json", json.dumps(db_info, indent=4))
            
            # Iteriamo sulle tabelle
            for db_dict in tables_data:
                table_name = db_dict["table_name"]
                df = db_dict["table"]
                
                if df is None or df.empty:
                    continue
                
                # Chiave deterministica usata nella UI
                key_alter = f"{db_name}_{table_name}"
                
                # 1. Metriche Generali (dataset_details)
                total_missing = int(df.isna().sum().sum())
                pct_missing = float(total_missing / (df.shape[0] * max(1, df.shape[1])) * 100.0) if df.size else 0.0
                dup_rows = int(df.duplicated().sum())
                mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
                
                metrics = {
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "missing_values_total": total_missing,
                    "missing_values_pct": round(pct_missing, 2),
                    "duplicate_rows": dup_rows,
                    "memory_usage_mb": round(mem_mb, 2),
                    "columns": list(df.columns)
                }
                z.writestr(f"{db_name}/{table_name}/metrics.json", json.dumps(metrics, indent=4))
                
                # 2. Schema & Column Insights (dataset_explore_gui logic)
                schema_rows = []
                nlp_model = st.session_state.get('spacy_model', {}).get('model')
                
                for col in df.columns:
                    s = df[col]
                    cat = get_column_category(s)
                    miss_pct = float(s.isna().mean() * 100.0)
                    uniq = s.nunique(dropna=True)
                    total = s.count()
                    uniqs_pct = (uniq / total * 100) if total > 0 else 0.0
                    
                    # Sensitive info
                    try:
                        sens_info = is_sensitive_column(col, nlp_model) if nlp_model else {"sensitive": False, "reasons": []}
                    except:
                        sens_info = {"sensitive": False, "reasons": []}
                        
                    schema_rows.append({
                        "column": col,
                        "type": cat,
                        "unique_pct": round(uniqs_pct, 2),
                        "missing_pct": round(miss_pct, 2),
                        "sensitive": sens_info.get("sensitive", False),
                        "sensitive_reasons": sens_info.get("reasons", [])
                    })
                
                z.writestr(f"{db_name}/{table_name}/schema_insights.json", json.dumps(schema_rows, indent=4))

                # 2.1 Descriptive Statistics (Numeric & Text)
                # Numeric
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if not numeric_cols.empty:
                    desc_num = df[numeric_cols].describe().T
                    z.writestr(f"{db_name}/{table_name}/descriptive_stats_numeric.csv", desc_num.to_csv())
                
                # Text
                text_cols = df.select_dtypes(include=['object', 'string']).columns
                if not text_cols.empty:
                    desc_text = df[text_cols].describe(include=['object', 'string']).T
                    z.writestr(f"{db_name}/{table_name}/descriptive_stats_text.csv", desc_text.to_csv())

                # 2.2 Correlations
                if len(numeric_cols) >= 2:
                    corr = df[numeric_cols].corr()
                    z.writestr(f"{db_name}/{table_name}/correlations_matrix.csv", corr.to_csv())
                    
                    # Top pairs
                    corr_pairs = (
                        corr.where(~np.eye(corr.shape[0], dtype=bool))
                        .stack()
                        .rename("correlation")
                        .abs()
                        .sort_values(ascending=False)
                        .reset_index()
                    )
                    corr_pairs.columns = ["Column A", "Column B", "Abs Correlation"]
                    # Add signed correlation back
                    corr_pairs["Correlation"] = corr_pairs.apply(lambda x: corr.loc[x["Column A"], x["Column B"]], axis=1)
                    z.writestr(f"{db_name}/{table_name}/correlations_top_pairs.csv", corr_pairs.head(50).to_csv(index=False))
                
                # 3. Profiling Relazionale (ui_profiling_relazionale)
                prof_ss_key = f"relprof:prof_{key_alter}:{table_name}"
                if prof_ss_key in st.session_state:
                    results = st.session_state[prof_ss_key].get("results", {})
                    # Semantic
                    if "semantic" in results:
                        z.writestr(f"{db_name}/{table_name}/profiling/semantic.csv", results["semantic"].to_csv(index=False))
                        
                        # Export semantic resource metrics if available
                        futures = st.session_state[prof_ss_key].get("futures", {})
                        if "sem" in futures and futures["sem"].done():
                            try:
                                sem_res = futures["sem"].result()
                                if isinstance(sem_res, dict) and "metrics" in sem_res:
                                    metrics_data = sem_res["metrics"]
                                    if metrics_data:
                                        # Export raw metrics as JSON
                                        z.writestr(f"{db_name}/{table_name}/profiling/semantic_resource_metrics.json", 
                                                   json.dumps(metrics_data, default=str, indent=4))
                                        
                                        # Export aggregated metrics as CSV
                                        metrics_df = _aggregate_metrics(metrics_data)
                                        if metrics_df is not None:
                                            z.writestr(f"{db_name}/{table_name}/profiling/semantic_resource_summary.csv", 
                                                       metrics_df.to_csv(index=False))
                            except:
                                pass
                    
                    # Heatmap data
                    futures = st.session_state[prof_ss_key].get("futures", {})
                    if "heat" in futures and futures["heat"].done():
                        try:
                            heat_res = futures["heat"].result()
                            if isinstance(heat_res, dict) and "miss_by_col" in heat_res:
                                z.writestr(f"{db_name}/{table_name}/profiling/missing_heatmap_stats.csv", heat_res["miss_by_col"].to_csv())
                        except:
                            pass

                # 4. Integrità (ui_integrita_dataset)
                intg_ss_key = f"relprof:intg_{key_alter}:{table_name}"
                if intg_ss_key in st.session_state:
                    results = st.session_state[intg_ss_key].get("results", {})
                    if "anomalies" in results:
                        z.writestr(f"{db_name}/{table_name}/profiling/anomalies.csv", results["anomalies"].to_csv(index=False))
                        
                        # Export anomalies resource metrics if available
                        futures = st.session_state[intg_ss_key].get("futures", {})
                        if "anom" in futures and futures["anom"].done():
                            try:
                                anom_res = futures["anom"].result()
                                if isinstance(anom_res, dict) and "metrics" in anom_res:
                                    metrics_data = anom_res["metrics"]
                                    if metrics_data:
                                        # Export raw metrics as JSON
                                        z.writestr(f"{db_name}/{table_name}/profiling/anomalies_resource_metrics.json", 
                                                   json.dumps(metrics_data, default=str, indent=4))
                                        
                                        # Export aggregated metrics as CSV
                                        metrics_df = _aggregate_metrics(metrics_data)
                                        if metrics_df is not None:
                                            z.writestr(f"{db_name}/{table_name}/profiling/anomalies_resource_summary.csv", 
                                                       metrics_df.to_csv(index=False))
                            except:
                                pass
                            
                # 5. Imputazione (missing_value_tab)
                imp_key = f"{table_name}_{key_alter}"
                queue_key = f'imputation_queue_{imp_key}'
                if queue_key in st.session_state and st.session_state[queue_key]:
                     # Salviamo la coda di imputazione pendente
                     z.writestr(f"{db_name}/{table_name}/imputation_queue.json", json.dumps(st.session_state[queue_key], default=str, indent=4))
                     
    return buf.getvalue(), "dataset_analytics_export.zip"