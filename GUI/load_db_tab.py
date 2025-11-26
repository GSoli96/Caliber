import sys
import os
import re
import streamlit as st
from itertools import count
import db_adapters
from GUI.dataset_explore_gui import preview_dataset, info_dataset, detailed_dataset
from db_adapters.DBManager import DBManager
from utils.load_data import load_data_files
# from GUI.relational_profiling_app import ui_profiling_relazionale, ui_integrita_dataset, ui_export
from GUI.message_gui import st_toast_temp
from utils.translations import get_text
from utils.symbols import symbols
sep_options = symbols.sep_options

# ============================================================================
# CONSTANTS AND INITIALIZATION
# ============================================================================

# DB_DIR is now managed via session_state
DB_DIR = st.session_state.get('db_dir', "database")

# ============================================================================
# TAB 2: DBMS CONNECTION (LOADING MODE)
# ============================================================================

def load_db_tab(key):
    """Handles direct DBMS connection and database loading."""
    st.subheader(get_text("load_dataset", "connect_db_header"))
    st.markdown(get_text("load_dataset", "connect_db_info"))

    with st.expander(get_text("load_dataset", "tab_dbms_connection"), expanded=False if len(list(st.session_state["dataframes"]["DBMS"].keys())) > 0 else True):
        dbms_parameters = render_dbms_input_for_loading(key, st.session_state['widget_idx_counter'])

        if dbms_parameters['complete_state']:
            db_name = dbms_parameters['db_name']
            st.session_state['uploaded_dbms'][db_name] = dbms_parameters

            # Clear any existing databases from session state (only one active database at a time)
            st.session_state["dataframes"]["DBMS"] = {}

            with st.spinner(show_time=True):
                dict_to_dbsm = {'config_dict': dbms_parameters}

                mgr_dl = DBManager(dict_to_dbsm, 'download')
                dumped, res = mgr_dl.download_db()
                if res:
                    st.session_state['create_db_done'] = True
                    st.session_state['db_choice'] = dbms_parameters['db_choice']
                    st.session_state['db_name'] = dbms_parameters['db_name']
                    st.session_state["dataframes"]["DBMS"][db_name] = dumped

            st_toast_temp(get_text("load_dataset", "dbms_success").format(db_name=db_name), 'success')

    if len(list(st.session_state["dataframes"]["DBMS"].keys())) > 0:
        dataset_tab_dbms(key)

def dataset_tab_dbms(key_alter=""):
    """Visualizza i dettagli per TUTTI i database DBMS caricati in session_state."""
    loaded_databases = st.session_state["dataframes"]["DBMS"]

    if not loaded_databases:
        st.info(get_text("load_dataset", "no_db_loaded"))
        return

    with st.expander('📁 Dataset Overview', expanded=True):
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
                            with st.container(border=False):
                                st.subheader(get_text("load_dataset", "dataset_details"))

                                preview_dataset(df, name, f'{db_name}_{name}_{tab}')
                        else:
                            st.error(f'Table {name} in dataset {db_name} non trovato!')


# ============================================================================
# SHARED: DBMS INPUT COMPONENTS
# ============================================================================

def render_dbms_input_for_loading(key_prefix, idx):
    """Renders DBMS input widgets for Tab 2 (loading existing database)."""
    ret_item = {}

    with st.container():
        col1db, col2db = st.columns(2)
        with col1db:
            ret_item['db_choice'] = st.selectbox(
                get_text("load_dataset", "db_engine"),
                index=0,
                options=db_adapters.DB_ADAPTERS,
                key=f'db_choice_{key_prefix}_{idx}'
            )

        with col2db:
            # Automatically load available databases when DBMS type changes
            current_dbms = ret_item['db_choice']
            cache_key = f'available_dbs_{current_dbms}_{key_prefix}_{idx}'

            # Check if we need to reload (DBMS changed or first load)
            if cache_key not in st.session_state:
                try:
                    temp_state = {
                        'config_dict': {'choice_DBMS': current_dbms},
                        'dfs_dict': {}
                    }
                    mgr = DBManager(temp_state, 'download')
                    dbs = mgr.get_available_databases()
                    if current_dbms == 'PostgreSQL':
                        dbs = [db for db in dbs if db != 'postgres']
                    elif current_dbms == 'MySQL':
                        dbs = [db for db in dbs if
                               db != 'performance_schema' and db != 'information_schema' and db != 'mysql' and db != 'sys']
                    elif current_dbms == 'SQL Server':
                        dbs = [db for db in dbs if db != 'master' and db != 'model' and db != 'msdb' and db != 'tempdb']

                    st.session_state[cache_key] = dbs
                except Exception as e:
                    st.session_state[cache_key] = []
                    st.warning(f"Could not retrieve databases: {str(e)}")

            available_dbs = st.session_state.get(cache_key, [])

            if available_dbs:
                selected_db = st.selectbox(
                    get_text("load_dataset", "db_name"),
                    options=available_dbs,
                    key=f'db_name_select_{key_prefix}_{idx}'
                )

                if ret_item['db_choice'] in ["SQLite", "DuckDB"]:
                    # Reconstruct path
                    ext = ".db" if ret_item['db_choice'] == "SQLite" else ".duckdb"
                    # If the selected db already has extension, don't add it
                    if not selected_db.endswith(ext):
                        filename = f"{selected_db}{ext}"
                    else:
                        filename = selected_db

                    ret_item['path_to_file'] = filename  # Relative path is fine, DBManager handles it
                    ret_item['db_name'] = selected_db
                else:
                    ret_item['db_name'] = selected_db
            else:
                # Still allow manual input as fallback
                if ret_item['db_choice'] in ["SQLite", "DuckDB"]:
                    ret_item['path_to_file'] = st.text_input(
                        get_text("load_dataset", "path_to_file"),
                        help=f'es. database_name.db (Relative paths are saved in {DB_DIR})',
                        placeholder='database_name.db',
                        key=f'path_to_file_{key_prefix}_{idx}'
                    )
                else:
                    ret_item['db_name'] = st.text_input(
                        get_text("load_dataset", "db_name"),
                        help='Insert Database name',
                        placeholder="database",
                        key=f'db_name_{key_prefix}_{idx}',
                    )

    # Show info message outside container if no databases found
    # Use st.empty() to maintain consistent spacing
    info_placeholder = st.empty()
    available_dbs = st.session_state.get(f'available_dbs_{ret_item["db_choice"]}_{key_prefix}_{idx}', [])
    if not available_dbs:
        info_placeholder.info(
            f"ℹ️ No {ret_item['db_choice']} databases found. "
            f"Try creating one in the **{get_text('load_dataset', 'tab_file_upload')}** tab."
        )

    col1, col2, col3 = st.columns([2, 5, 2])

    with col1:
        submitted = st.button(get_text("load_dataset", "load_db_btn"), key=f'button_DBMS_{key_prefix}_{idx}')

    ret_item['complete_state'] = True
    ret_item['submitted'] = submitted

    if not submitted:
        ret_item['complete_state'] = False
        return ret_item

    # Validation for loading mode
    ret_item = validate_dbms_input(ret_item, is_loading=True)

    return ret_item


def render_dbms_input_for_creation(key_prefix, idx):
    """Renders DBMS input widgets for Tab 1 (creating database from files)."""
    ret_item = {}

    with st.container():
        col1db, col2db = st.columns(2)
        with col1db:
            ret_item['db_choice'] = st.selectbox(
                get_text("load_dataset", "db_engine"),
                index=0,
                options=db_adapters.DB_ADAPTERS,
                key=f'db_choice_{key_prefix}_{idx}'
            )

        with col2db:
            # Manual input for creation
            if ret_item['db_choice'] in ["SQLite", "DuckDB"]:
                ext = ".db" if ret_item['db_choice'] == "SQLite" else ".duckdb"
                label = get_text("load_dataset", "db_name")
                help_text = f"Extension {ext} will be added automatically. Saved in {DB_DIR}"
                placeholder = "database_name"

                user_input = st.text_input(
                    label,
                    help=help_text,
                    placeholder=placeholder,
                    key=f'db_name_input_{key_prefix}_{idx}'
                )

                # Auto-add extension if not present
                if user_input:
                    # Remove extension if user typed it, to avoid double extension logic later
                    if user_input.endswith(ext):
                        base_name = user_input[:-len(ext)]
                    else:
                        base_name = user_input

                    ret_item['db_path'] = f"{base_name}{ext}"
                    ret_item['db_name'] = base_name
                else:
                    ret_item['db_path'] = ''
                    ret_item['db_name'] = ''
            else:
                ret_item['db_name'] = st.text_input(
                    get_text("load_dataset", "db_name"),
                    help='Insert Database name',
                    placeholder="database",
                    key=f'db_name_{key_prefix}_{idx}',
                )

    col1, col2, col3 = st.columns([2, 5, 2])

    with col1:
        submitted = st.button(get_text("load_dataset", "create_db_btn"), key=f'button_DBMS_{key_prefix}_{idx}')

    ret_item['complete_state'] = True
    ret_item['submitted'] = submitted

    # Check if we are in "Overwrite Decision" mode
    if st.session_state.get(f'overwrite_mode_{key_prefix}_{idx}'):
        ret_item = handle_overwrite_rename_cancel(key_prefix, idx, ret_item)
        if not ret_item['complete_state']:
            return ret_item

    if not submitted:
        ret_item['complete_state'] = False
        return ret_item

    # Validation for creation mode
    ret_item = validate_dbms_input(ret_item, is_loading=False)

    # Existence Check (only for creation mode)
    if ret_item['complete_state']:
        db_name_check = ret_item.get('db_name')
        db_path_check = ret_item.get('db_path')

        # Normalize name for check
        if ret_item['db_choice'] in ["SQLite", "DuckDB"] and db_path_check:
            if not os.path.isabs(db_path_check):
                db_path_check = os.path.join(DB_DIR, db_path_check)
            db_name_check = os.path.basename(db_path_check).split('.')[0]

        exists = check_db_exists(db_name_check, ret_item['db_choice'], db_path_check)

        if exists:
            st.session_state[f'overwrite_mode_{key_prefix}_{idx}'] = True
            st.rerun()

    return ret_item


def check_db_exists(db_name, dbms_type, db_path=None):
    """Check if database exists (used by both tabs)."""
    if dbms_type in ["SQLite", "DuckDB"]:
        # Check file existence
        if db_path:
            if not os.path.isabs(db_path):
                db_path = os.path.join(DB_DIR, db_path)
            return os.path.exists(db_path)
        return False
    else:
        # Check server
        try:
            temp_state = {'config_dict': {'choice_DBMS': dbms_type}, 'dfs_dict': {}}
            mgr = DBManager(temp_state, 'download')
            dbs = mgr.get_available_databases()
            return db_name in dbs
        except Exception:
            return False


def validate_dbms_input(ret_item, is_loading):
    """Validate DBMS input parameters."""
    if 'db_choice' not in ret_item:
        st_toast_temp(get_text("load_dataset", "insert_db_warning"), 'warning')
        ret_item['complete_state'] = False

    elif ret_item['db_choice'] in ['MySQL', 'PostgreSQL', 'SQL Server'] and (
            'db_name' not in ret_item or ret_item['db_name'] == ""):
        st_toast_temp(get_text("load_dataset", "check_params_warning"), 'warning')
        ret_item['complete_state'] = False

    elif ret_item['db_choice'] in ['SQLite', 'DuckDB']:
        path_key = 'path_to_file' if is_loading else 'db_path'
        user_path = ret_item.get(path_key)

        if not user_path:
            st.error(get_text("load_dataset", "enter_db_path_error"))
            ret_item['complete_state'] = False
        else:
            # Extension is now automatically added in render_dbms_input_for_creation
            # For loading mode, still validate that the file ends with the correct extension
            ext = ".db" if ret_item['db_choice'] == "SQLite" else ".duckdb"

            if is_loading:
                # Loading mode: validate extension is present
                if not user_path.endswith(ext):
                    st.error(f"Please select a valid {ret_item['db_choice']} database file ({ext})")
                    ret_item['complete_state'] = False
                    return ret_item

            # Process the path
            user_path = user_path.strip()
            if os.path.isabs(user_path):
                absolute_path = user_path
            else:
                absolute_path = os.path.abspath(os.path.join(DB_DIR, user_path))

            if is_loading:
                if os.path.isfile(absolute_path):
                    ret_item['path_to_file'] = absolute_path
                    ret_item['db_name'] = os.path.basename(absolute_path)
                elif os.path.isdir(absolute_path):
                    st.error(get_text("load_dataset", "path_is_dir_error", path=absolute_path))
                    ret_item['complete_state'] = False
                else:
                    st.error(get_text("load_dataset", "file_not_found_error", path=absolute_path))
                    ret_item['complete_state'] = False
            else:
                # Creating
                ret_item['db_path'] = absolute_path
                ret_item['db_name'] = os.path.basename(absolute_path)

    return ret_item


def handle_overwrite_rename_cancel(key_prefix, idx, ret_item):
    """Handle overwrite/rename/cancel UI for existing databases."""
    st.warning(get_text("load_dataset", "db_exists_warning"))

    # 2 columns: Overwrite | (New Name + Rename)
    col_act1, col_act2 = st.columns(2)

    # Column 1: Overwrite button
    with col_act1:
        if st.button("Overwrite", key=f"act_overwrite_{key_prefix}_{idx}", use_container_width=True, type="primary"):
            # Delete the existing database
            try:
                temp_dict = {'config_dict': ret_item, 'dfs_dict': {}}
                mgr = DBManager(temp_dict, 'remove')
                mgr._reset_all_db()

                # Now create the new database
                dfs_dict = st.session_state["dataframes"]["files"]
                dict_to_dbsm = {'config_dict': ret_item, 'dfs_dict': dfs_dict}
                mgr_create = DBManager(dict_to_dbsm, 'create')

                with st.spinner("Creating database..."):
                    dumped, loaded_db = mgr_create.create_db()

                if loaded_db:
                    # Update session state
                    db_name = ret_item['db_name']
                    st.session_state['uploaded_dbms'][db_name] = ret_item
                    st.session_state['db_choice'] = ret_item['db_choice']
                    st.session_state['db_name'] = db_name
                    st.session_state['create_db_done'] = True
                    st.session_state["dataframes"]["DBMS"][db_name] = dumped

                    st.success(f'✅ Database **{loaded_db}** successfully overwritten and recreated!')
                    st_toast_temp(f"Database {loaded_db} overwritten successfully", "success")
                else:
                    st.error("❌ Failed to create database after deletion.")

            except Exception as e:
                st.error(f"❌ Error during overwrite: {str(e)}")

            # Exit overwrite mode
            st.session_state[f'overwrite_mode_{key_prefix}_{idx}'] = False
            st.rerun()

    # Column 2: Reset Name button
    with col_act2:
        if st.button("Reset Name", key=f"act_reset_{key_prefix}_{idx}", use_container_width=True, type="secondary"):
            # Exit overwrite mode and increment widget counter to reset the input field
            st.session_state[f'overwrite_mode_{key_prefix}_{idx}'] = False
            st.session_state['widget_idx_counter'] += 1
            st.rerun()

    # If in overwrite mode, we return incomplete state until user decides
    ret_item['complete_state'] = False
    return ret_item

# ============================================================================
# SHARED: DISPLAY COMPONENTS
# ============================================================================
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
