import streamlit as st
from utils.translations import get_text
from utils.symbols import symbols
import sys
import os
import re
import streamlit as st
from itertools import count
import db_adapters
from GUI.dataset_explore_gui import preview_dataset, info_dataset, detailed_dataset
from db_adapters.DBManager import DBManager
from utils.load_data import load_data_files
from GUI.relational_profiling_tab import ui_profiling_relazionale, ui_integrita_dataset, ui_export
from GUI.message_gui import st_toast_temp
from utils.translations import get_text
from utils.symbols import symbols

sep_options = symbols.sep_options

def dataset_analytics_tab():
    # Display uploaded files in tabs
    if len(list(st.session_state["dataframes"]["DBMS"].keys())) > 0:
        dataset_tab_dbms()


def dataset_tab_files(name):
    """Displays details for a single uploaded file."""
    file_info = st.session_state["uploaded_files"][name]
    uploaded_file = file_info["uploaded_file"]
    current_sep = file_info["separator"]

    col1, colspan, col2 = st.columns([4, 4, 2], vertical_alignment="center")
    with col1:
        st.subheader(f"📁 {name}")
    with col2:
        # pulsante per entrare in edit mode SOLO per questa tab
        if st.button(get_text("load_dataset", "edit_config"), key=f"edit_{name}", type="secondary"):
            st.session_state["edit_sep_flags"][name] = True
            st.rerun()

    # --- EDIT MODE per questa tab ---
    if st.session_state["edit_sep_flags"].get(name, False):
        st.markdown(f"**{get_text('load_dataset', 'edit_separator')}**")

        # pre-selezione coerente col separatore corrente
        if current_sep == "\\t":
            default_idx = sep_options.index("\\\\t")
        elif current_sep in [";", ",", "|"]:
            default_idx = sep_options.index(current_sep)
        else:
            default_idx = sep_options.index("Custom")

        colA, colB = st.columns([2, 2])
        with colA:
            sep_choice = st.selectbox(
                get_text("load_dataset", "separator"),
                options=sep_options,
                index=default_idx,
                key=f"sep_choice_{name}",
                help="Seleziona il separatore dei CSV",
            )
        with colB:
            if sep_choice == "Custom":
                custom_val = st.text_input(
                    get_text("load_dataset", "custom_sep_1char"),
                    value=current_sep if current_sep not in [";", ",", "|", "\\t"] else "",
                    max_chars=1,
                    key=f"custom_sep_{name}",
                    help="Inserisci un singolo carattere (es. :)",
                )
            else:
                custom_val = None

        colS, colC = st.columns([1, 1])
        with colS:
            if st.button(get_text("load_dataset", "save"), key=f"save_sep_{name}", type="primary"):
                # normalizza il separatore scelto
                if sep_choice == "\\\\t":
                    new_sep = "\\t"
                elif sep_choice == "Custom":
                    if not custom_val:
                        st.error("Inserisci un separatore personalizzato (1 carattere).")
                        st.stop()
                    new_sep = custom_val
                else:
                    new_sep = sep_choice

                # aggiorna lo stato
                st.session_state["uploaded_files"][name]["separator"] = new_sep
                st.session_state["dataframes"]['files'].pop(name, {})

                # ricarica il DF con il nuovo separatore
                if load_data_files(uploaded_file, new_sep):
                    df = st.session_state["dataframes"]["files"][name]["df"]
                    st.session_state["edit_sep_flags"][name] = False
                    st_toast_temp(get_text("load_dataset", "sep_updated", name=name), 'success')
                    st.rerun()
                else:
                    st.error(get_text("load_dataset", "reload_error"))
                    st.stop()

        with colC:
            if st.button(get_text("load_dataset", "cancel"), key=f"cancel_sep_{name}", type="secondary"):
                st.session_state["edit_sep_flags"][name] = False
                st.rerun()

    else:
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"{get_text('load_dataset', 'filename')} {uploaded_file.name}")
            with col2:
                shown_sep = "\\\\t" if current_sep == "\\t" else current_sep
                st.write(f"{get_text('load_dataset', 'separator_label')} `{shown_sep}`")

    file_info = st.session_state["uploaded_files"][name]
    uploaded_file = file_info["uploaded_file"]
    current_sep = file_info["separator"]

    # --- BLOCCO CORRETTO ---
    # 1. Prova a PRENDERE il df dalla session_state
    df_info = st.session_state["dataframes"]["files"].get(name)

    df = None
    if df_info:
        # 2. Se c'è, usalo
        df = df_info["df"]
    else:
        # 3. Se non c'è (es. primo caricamento), PROVA A CARICARLO
        if load_data_files(uploaded_file, current_sep):
            # Ora dovrebbe esserci
            if name in st.session_state["dataframes"]["files"]:
                df = st.session_state["dataframes"]["files"][name]["df"]
            else:
                st.error(get_text("load_dataset", "not_found_error", name=name))
        else:
            st.error(get_text("load_dataset", "load_df_error", name=name))

    # 4. Mostra i dettagli solo se abbiamo un df
    if df is not None and df.shape[0] > 0:
        show_df_details(df, name, name)
    else:
        st.error(get_text("load_dataset", "dataset_not_found", name=name))

def dataset_tab_dbms(key_alter=""):
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

            with st.container(border=True):
                st.subheader(get_text("load_dataset", "db_info"))

                # 4. Funzione modulare per mostrare le info
                _display_db_info(config_dict, db_name, tables_data, key_alter)

                # st.divider()
                st.subheader(get_text("load_dataset", "explore_tables"))

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

def _display_db_info(config_dict, db_name, tables_data, key_alter):
    """Helper function to display DBMS configuration details."""
    if key_alter is None:
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
        st.subheader(get_text("load_dataset", "dataset_details"))

        tab1_prew, tab2_detalied, tab3_info, tab4_profiling, tab5_integrita, tab6_export = st.tabs([
            get_text("load_dataset", "tab_preview"),
            get_text("load_dataset", "tab_detailed"),
            get_text("load_dataset", "tab_info"),
            get_text("load_dataset", "tab_profiling"),
            get_text("load_dataset", "tab_integrity"),
            get_text("load_dataset", "tab_export")
        ])

        with tab1_prew:
            preview_dataset(df, name, key_alter)
        with tab2_detalied:
            detailed_dataset(df, key=f"{name}_{key_alter}")
        with tab3_info:
            info_dataset(df, name)
        with tab4_profiling:
            ui_profiling_relazionale(
                df,
                key=f"prof_{key_alter}",  # chiave per questa sezione
                name=name,
                related_tables=None  # opzionale
            )
        with tab5_integrita:
            ui_integrita_dataset(
                df,
                name=name,
                key=f"intg_{key_alter}",  # chiave per questa sezione
            )
        with tab6_export:
            ui_export(
                df,
                name=name,
                key=f"export_{key_alter}",  # chiave per questa sezione
                depends_on_key="prof"  # usa i risultati del profiling per l'export
            )
