import sys
import os
import re
import streamlit as st
from itertools import count
import db_adapters
from GUI.dataset_explore_gui import info_dataset
from db_adapters.DBManager import DBManager
from utils.load_data import load_data_files
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
# TAB 1: FILE UPLOAD (CREATION MODE)
# ============================================================================

def load_file_tab(key):
    """Handles CSV file upload and database creation from uploaded files."""
    # Ensure all required session state keys are initialize
    st.subheader(get_text("load_dataset", "upload_header"))
    st.markdown(f"""
        - {get_text("load_dataset", "upload_info_single")}
        - {get_text("load_dataset", "upload_info_multiple")}
    """)

    # File upload section
    with st.expander(get_text("load_dataset", "expander_upload"),
                     expanded=True if len(list(st.session_state["uploaded_files"].keys())) == 0 else False):
        submitted, uploaded_files, separator = upload_files(key)
        if submitted and uploaded_files and separator:
            check_duplicates_files(uploaded_files, separator, key)
            if st.session_state['uploaded_files']:
                st_toast_temp(
                    get_text("load_dataset", "success_upload", n=len(list(st.session_state["uploaded_files"].keys()))),
                    'success')

        # Reset button
        reset = False
        if len(list(st.session_state["uploaded_files"].keys())) > 0:
            col1, col2 = st.columns([1, 1], gap='small', vertical_alignment='center')
            with col1:
                reset = st.button(get_text("load_dataset", "reset_files"), key=f'reset_{key}', type='secondary')
            with col2:
                st.success(
                    get_text("load_dataset", "success_upload", n=len(list(st.session_state["uploaded_files"].keys()))))

        if reset:
            reset_upload(key)
    st.divider()
    # Display uploaded files in tabs
    if st.session_state.get("uploaded_files") and len(list(st.session_state["uploaded_files"].keys())) > 0:
        with st.expander(get_text("load_dataset", "expander_files"),
                         expanded=False):
            tabs = st.tabs(list(st.session_state["uploaded_files"].keys()))
            tb_name =  list(st.session_state["uploaded_files"].keys())
            for name, tab in zip(tb_name, tabs):
                with tab:
                    dataset_tab(name)

    # Database configuration section (create DB from uploaded files)
    if len(list(st.session_state['dataframes']['files'].keys())) > 0:
        st.header(get_text("load_dataset", "db_config"))
        with st.expander(get_text("load_dataset", "dbms_config"), expanded=True):
            configure_file_dbms(key_prefix='configure_dbms_tab1', name = name)

def upload_files(key_prefix):
    """Renders file upload form with separator selection."""
    # Ensure uploaded_files is initialized
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = {}

    with st.form(f"dataset_form_files_{key_prefix}", clear_on_submit=False, border=False):
        col1, col2, col3 = st.columns([5, 1, 1], gap='medium')

        with col1:
            uploaded_files = st.file_uploader(
                get_text("load_dataset", "choose_files_csv"),
                type=["csv", "parquet", "h5", "hdf5"],
                label_visibility="collapsed",
                accept_multiple_files=True,
                key=f'upload_files_{key_prefix}_{st.session_state["widget_idx_counter"]}'
            )

        with col2:
            sep_choice = st.selectbox(get_text("load_dataset", "separator"), options=sep_options, index=0,
                                      placeholder=get_text("load_dataset", "Choose_option"),
                                      help=get_text("load_dataset", "separator"),
                                      key=f'selectbox_{key_prefix}_{st.session_state["widget_idx_counter"]}')

            # Se l'utente sceglie 'Custom', mostra un campo per inserire manualmente
            if sep_choice == "Custom":
                separator = st.text_input(get_text("load_dataset", "custom_separator"), max_chars=1,
                                          value=";",
                                          key=f"separator_first_{key_prefix}_{st.session_state['widget_idx_counter']}")
            else:
                # Converte "\\t" in tab reale
                separator = "\\t" if sep_choice == "\\\\t" else sep_choice

        with col3:
            submitted = st.form_submit_button(get_text("load_dataset", "load_files_btn"),
                                              key=f'button_upload_files_{key_prefix}_{st.session_state["widget_idx_counter"]}')

    new_uploaded_files = []
    for files in uploaded_files:
        files_name = files.name.split('.')[0]
        if files_name not in list(st.session_state['uploaded_files'].keys()):
            new_uploaded_files.append(files)

    return submitted, new_uploaded_files, separator

def check_duplicates_files(uploaded_files, separator, key_prefix):
    """Checks for duplicate files and handles them."""
    cur = st.session_state.setdefault('uploaded_files', {})
    st.session_state.setdefault('duplicate_files', [])

    if cur:
        for uploaded_file in uploaded_files:
            base = uploaded_file.name.split('.')[0]

            if base in cur:
                # metti tra i duplicati solo se non già presente
                already = any(f.name.split('.')[0] == base for f, _ in st.session_state['duplicate_files'])
                if not already:
                    st.session_state['duplicate_files'].append((uploaded_file, separator))
            else:
                cur[base] = {'uploaded_file': uploaded_file, 'separator': separator}
    else:
        # nessun file ancora: aggiungi tutti
        for uploaded_file in uploaded_files:
            base = uploaded_file.name.split('.')[0]
            cur[base] = {'uploaded_file': uploaded_file, 'separator': separator}

    # feedback opzionale
    if st.session_state['duplicate_files']:
        st_toast_temp(
            get_text("load_dataset", "warning_duplicates", n=len(st.session_state["duplicate_files"])),
            'warning'
        )
        action_for_duplicate_radio(key_prefix)
        st.session_state.setdefault('choice_files', 'None')
        if st.session_state['choice_files'] != 'None':
            st_toast_temp(get_text("load_dataset", "choice_success", choice=st.session_state["choice_files"]),
                          'success')

def action_for_duplicate_radio(key_prefix):
    """Handles duplicate file actions using radio buttons."""
    dup = st.session_state["duplicate_files"]
    cur = st.session_state["uploaded_files"]

    # Se non ci sono duplicati, esci senza toccare nulla
    if not dup:
        return None

    choice = st.segmented_control(
        get_text("load_dataset", "choose_action"),
        options=[
            get_text("load_dataset", "action_continue"),
            get_text("load_dataset", "action_remove"),
            get_text("load_dataset", "action_keep"),
        ],
        key=f"dup_action_files_{key_prefix}",
    )

    # --- ESECUZIONE DELLA SCELTA ---
    if choice == get_text("load_dataset", "action_continue"):
        st.session_state['choice_files'] = "Continue anyway"
        # Aggiungi i duplicati mantenendoli, con suffisso progressivo
        for uploaded_file, separator in dup:
            base = uploaded_file.name.split(".")[0]
            suffix = next(st.session_state["idx_counter"])
            new_key = f"{base}_{suffix}"
            cur[new_key] = {"uploaded_file": uploaded_file, "separator": separator}
        st.write(
            get_text("load_dataset", "duplicate_kept")
            + ", ".join(cur_key for cur_key in cur.keys()), file=sys.stdout
        )

    elif choice == get_text("load_dataset", "action_remove"):
        st.session_state['choice_files'] = "Remove previously uploaded datasets"
        # Sostituisci i precedenti con i nuovi aventi lo stesso base-name
        replaced = 0
        for uploaded_file, separator in dup:
            base = uploaded_file.name.split(".")[0]
            if base in cur:
                cur[base].update({"uploaded_file": uploaded_file, "separator": separator})
                replaced += 1
            else:
                # se il vecchio non c'era, aggiungilo comunque come base
                cur[base] = {"uploaded_file": uploaded_file, "separator": separator}
                replaced += 1

        st.success(get_text("load_dataset", "replaced_datasets", n=replaced))

    elif choice == get_text("load_dataset", "action_keep"):
        st.session_state['choice_files'] = "Keep previous and ignore new files"

        # Non fare nulla: ignoriamo i duplicati
        st.info(get_text("load_dataset", "ignored_duplicates", n=len(dup)))

    else:
        st.session_state['choice_files'] = "None"
        # Nessuna scelta (difensivo)
        st.warning(get_text("load_dataset", "select_action_warning"))
        return None

def reset_upload(key_prefix):
    """Reset uploaded files and increment widget counter to regenerate all widgets with new keys."""
    st.session_state['uploaded_files'] = {}
    st.session_state['duplicate_files'] = []
    st.session_state['dataframes']['files'] = {}
    st.session_state['widget_idx_counter'] += 1  # Increment to regenerate widgets
    st.rerun()

def dataset_tab(name):
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
            default_idx = sep_options.index(get_text("load_dataset", "custom"))

        colA, colB = st.columns([2, 2])
        with colA:
            sep_choice = st.selectbox(
                get_text("load_dataset", "separator"),
                options=sep_options,
                index=default_idx,
                key=f"sep_choice_{name}",
                help=get_text("load_dataset", "separator"),
            )
        with colB:
            if sep_choice == get_text("load_dataset", "custom"):
                custom_val = st.text_input(
                    get_text("load_dataset", "custom_sep_1char"),
                    value=current_sep if current_sep not in [";", ",", "|", "\\t"] else "",
                    max_chars=1,
                    key=f"custom_sep_{name}",
                    help=get_text("load_dataset", "insert"),
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
                        st.error(get_text("load_dataset", "custom_sep_1char"))
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
                try:
                    st.write(f"{get_text('load_dataset', 'filename')} {uploaded_file.name}")
                except:
                    st.write(f"{get_text('load_dataset', 'filename')} {uploaded_file}")
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
        with st.container(border=False):
            st.subheader(get_text("load_dataset", "dataset_details"))
            rows_to_show = st.number_input(
                get_text("load_dataset", "rows_to_show"),
                min_value=1,
                max_value=len(df),
                value=min(5, len(df)),
                step=1,
                help=get_text("load_dataset", "rows_to_show_help"),
                key=f'numberInput_preview_{name}'
            )
            st.write(df.head(rows_to_show))
    else:
        st.error(get_text("load_dataset", "dataset_not_found", name=name))

# ============================================================================
# SHARED: DBMS INPUT COMPONENTS
# ============================================================================
def render_dbms_input_for_creation(key_prefix, idx, name):
    """Renders DBMS input widgets for Tab 1 (creating database from files)."""
    ret_item = {}
    name = name.lower()
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
            db_choice = ret_item['db_choice']
            is_file_based_db = db_choice in ["SQLite", "DuckDB"]
            db_extension = ""
            db_name_help_text = ""
            db_name_value = name.lower()

            if is_file_based_db:
                db_extension = ".db" if db_choice == "SQLite" else ".duckdb"
                db_name_help_text = "Il file sarà salvato in " + DB_DIR + " con estensione " + db_extension
            else:
                db_name_help_text = get_text("load_dataset", "insert_db_name")

            user_input_db_name = st.text_input(
                get_text("load_dataset", "db_name"),
                help=db_name_help_text,
                placeholder=db_name_value,
                value=db_name_value,
                key=f'db_name_input_{key_prefix}_{idx}'
            )

            ret_item['db_name'] = None
            ret_item['db_path'] = ''

            if user_input_db_name:
                processed_db_name = user_input_db_name.lower().strip()
                if is_file_based_db:
                    # Remove extension if already present to avoid double extensions
                    if processed_db_name.endswith(db_extension):
                        base_name = processed_db_name[:-len(db_extension)]
                    elif processed_db_name.endswith('.db') or processed_db_name.endswith('.duckdb'):
                        # Remove any db extension  
                        base_name = os.path.splitext(processed_db_name)[0]
                    else:
                        base_name = processed_db_name
                    
                    ret_item['db_path'] = f"{base_name}{db_extension}"
                    ret_item['db_name'] = base_name
                else:
                    # Validate lowercase for other DBMS
                    if not user_input_db_name.islower():
                        st_toast_temp("Database name must be in lowercase.", msg_type="error")
                    else:
                        ret_item['db_name'] = processed_db_name

    col1, col2, col3 = st.columns([2, 5, 2])

    with col1:
        submitted = st.button(get_text("load_dataset", "create_db_btn"), key=f'button_DBMS_{key_prefix}_{idx}', disabled=ret_item['db_name'] is None)

    # Check if we are in "Overwrite Decision" mode
    if st.session_state.get(f'overwrite_mode_{key_prefix}_{idx}'):
        ret_item = handle_overwrite_rename_cancel(key_prefix, idx, ret_item)
        if not ret_item['complete_state']:
            return ret_item

    if not submitted:
        ret_item['complete_state'] = False
        return ret_item
    else:
        ret_item['complete_state'] = True
        ret_item['submitted'] = submitted
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

    if os.path.exists(st.session_state.get('db_dir', "Database")):
        list_files = os.listdir(st.session_state.get('db_dir', "Database"))
        if dbms_type == "SQLite":
            db_name_check = db_name+'.db'
            if db_name_check in list_files:
                return True
            else:
                return False
        elif dbms_type == "DuckDB":
            db_name_check = db_name+'.duckdb'
            if db_name_check in list_files:
                return True
            else:
                return False
        else:
            try:
                temp_state = {'config_dict': {'choice_DBMS': dbms_type}, 'dfs_dict': {}}
                mgr = DBManager(temp_state, 'download')
                dbs = mgr.get_available_databases()
                return db_name in dbs
            except Exception as e:
                print('Exception in check_db_exists')
                print(e)
                return False
    else:
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
                    text = get_text("load_dataset", "error1", db_choice=ret_item['db_choice'], ext=ext)
                    st.error(text)
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
                ret_item['db_name'] = os.path.splitext(os.path.basename(absolute_path))[0]

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
                
                loaded_db, _ = create_dbms(dict_to_dbsm, "riga 562")

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
        if st.button("Annulla", key=f"act_reset_{key_prefix}_{idx}", use_container_width=True, type="secondary"):
            # Exit overwrite mode and increment widget counter to reset the input field
            st.session_state[f'overwrite_mode_{key_prefix}_{idx}'] = False
            st.session_state['widget_idx_counter'] += 1
            st.rerun()

    # If in overwrite mode, we return incomplete state until user decides
    ret_item['complete_state'] = False
    return ret_item
def download_dbms(dbms_parameters, msg):
    """Download database from DBMS."""
    print(msg)
    with st.spinner("Downloading database..."):
        mgr_reload = DBManager({'config_dict': dbms_parameters}, 'download')
        reloaded_data, reload_success = mgr_reload.download_db()
    return reload_success, reloaded_data

def create_dbms(dbms_parameters, msg):
    """Create database from DBMS."""
    print(msg)
    print(dbms_parameters.keys())
    print(dbms_parameters.items())
    with st.spinner("Creating database..."):
        mgr_create = DBManager(dbms_parameters, 'create')

        reloaded_data, reload_success = mgr_create.create_db()
    
    return reload_success, reloaded_data
# ============================================================================
# SHARED: DATABASE CONFIGURATION
# ============================================================================
def configure_file_dbms(key_prefix, name):
    """Configures and creates a database from uploaded files (used by Tab 1)."""
    dbms_parameters = render_dbms_input_for_creation(key_prefix, st.session_state['widget_idx_counter'], name)
    if dbms_parameters['complete_state']:
        db_name = dbms_parameters['db_name']
        st.session_state['uploaded_dbms'][db_name] = dbms_parameters

        # Clear any existing databases from session state (only one active database at a time)
        st.session_state["dataframes"]["DBMS"] = {}

        dfs_dict = st.session_state["dataframes"]["files"]

        config_dict = dbms_parameters
        config_dict['dfs_dict'] = dfs_dict
        dict_to_dbsm = {'config_dict': config_dict}
        loaded_db, _ = create_dbms(dict_to_dbsm, "riga 683")
        if not loaded_db:
            st.error(f"Failed to create database.")
        else:
            # Update session state
            st.session_state['db_choice'] = dbms_parameters['db_choice']
            st.session_state['db_name'] = dbms_parameters['db_name']
            st.session_state['create_db_done'] = True

            # Reload the database from DBMS to ensure consistency
        mgr_reload = DBManager({'config_dict': dbms_parameters}, 'download')
        reloaded_data, reload_success = mgr_reload.download_db()

        if reload_success and reloaded_data:
            # Store ONLY the reloaded database (ensuring single dataset)
            st.session_state["dataframes"]["DBMS"][db_name] = reloaded_data
            st.success(get_text(
                "load_dataset",
                    "dbms_success_upload", 
                    db_name=dbms_parameters['db_name'], 
                    dbms_name=dbms_parameters['db_choice']
                    )
            )
            st_toast_temp(get_text("load_dataset", "dbms_success_upload"), "success")
        else:
            # Fallback: use the reloaded data if reload fails
            st.session_state["dataframes"]["DBMS"][db_name] = reloaded_data
            st.warning(get_text("load_dataset", "reload_failed", db_name=db_name))

        # Invalidate cache for available databases to force refresh
        for key in list(st.session_state.keys()):
            if key.startswith('available_dbs_'):
                del st.session_state[key]