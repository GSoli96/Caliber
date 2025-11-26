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
                                show_df_details(df, name, f'{db_name}_{name}_{tab}')
                            else:
                                st.error(f'Table {name} in dataset {db_name} non trovato!')
    else:
        st.warning('Please laod a dataset from a DBMS.')

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