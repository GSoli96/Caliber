import streamlit as st
import pandas as pd
import time
from db_adapters.DBManager import DBManager
from utils.translations import get_text
import sys
from GUI.message_gui import st_toast_temp

def db_management_tab():
    # 2. Tab per DBMS
    tabs = st.tabs(["MySQL", "SQLite", "PostgreSQL", "DuckDB", "SQL Server"])
    
    # Mappa nomi tab a nomi interni DBManager
    dbms_map = {
        "MySQL": "MySQL",
        "SQLite": "SQLite",
        "PostgreSQL": "PostgreSQL",
        "DuckDB": "DuckDB",
        "SQL Server": "SQL Server"
    }

    for i, tab_name in enumerate(tabs):
        dbms_type = list(dbms_map.values())[i]
        with tabs[i]:
            _render_dbms_tab(dbms_type)


def _render_dbms_tab(dbms_type):
    # Istanzia DBManager "dummy" per usare i metodi
    # Usiamo session state attuale, ma forziamo il tipo DBMS per questo contesto
    # Attenzione: DBManager legge choice_DBMS dal config_dict. Dobbiamo mockarlo o passarlo.
    # DBManager è molto legato allo stato globale. Creiamo un config temporaneo.
    
    temp_ss = st.session_state
    # Config dict fittizio per inizializzare DBManager
    # Config dict fittizio per inizializzare DBManager
    temp_ss['config_dict'] = {
        'choice_DBMS': dbms_type,
        'db_name': 'serverStatus', # placeholder
        'connection_string': '', # placeholder, userà defaults o global config
        'dfs_dict': {'dummy': pd.DataFrame()} # Assicuriamoci che dfs_dict esista per evitare return anticipato
    } 

    manager = DBManager(temp_ss, type="status")
    manager.choice_DBMS = dbms_type

    # --- SEZIONE SERVER ---
    
    # Layout differente per Server-based vs Serverless
    if dbms_type in ["MySQL", "PostgreSQL", 'SQL Server']:
        # Riga 1: Titolo

        
        # Riga 2: Status e Controlli
        c1, c2 = st.columns([1, 1])

        if st.session_state.DBMS_Sever[dbms_type]['status'] == 'not_running':
            st.markdown(f"### {get_text('db_management', 'server_title', dbms_type=dbms_type)}")
            with c1:
                # Check status automatico
                ok, msg = manager.server_control('status')
                is_running = ok and msg == "Running"

                if is_running:
                    st.markdown(f"#### {get_text('db_management', 'status_label')} {get_text('db_management', 'status_running')}")
                    st.session_state.DBMS_Sever[dbms_type]['status'] = 'running'
                else:
                    st.markdown(f"#### {get_text('db_management', 'status_label')} {get_text('db_management', 'status_stopped')}")
                    st.session_state.DBMS_Sever[dbms_type]['status'] = 'not_running'
            with c2:
                # Start Button - Disabled if running
                if st.button(
                        get_text('db_management', 'btn_start', dbms_type=dbms_type),
                        key=f"btn_start_{dbms_type}",
                        use_container_width=True,
                        disabled=True if st.session_state.DBMS_Sever[dbms_type]['status'] == 'running' else False):
                    ok_start, msg_start = manager.server_control('start')

                    if ok_start:
                        st.toast(f"✅ {msg_start}")
                        st.session_state.DBMS_Sever[dbms_type]['status'] = 'running'
                    else:
                        st.toast(f"❌ {msg_start}")
                        st.session_state.DBMS_Sever[dbms_type]['status'] = 'not_running'

        elif st.session_state.DBMS_Sever[dbms_type]['status'] == 'running':
            with c1:
                st.markdown(f"### {get_text('db_management', 'server_title', dbms_type=dbms_type)}")
            with c2:
                st.markdown(f"#### {get_text('db_management', 'status_label')} {get_text('db_management', 'status_running')}")
            st.session_state.DBMS_Sever[dbms_type]['status'] = 'running'
        
        # Riga 3: Placeholder per errori (o toast sopra)
        st.empty()
    else:
        # Serverless (SQLite, DuckDB)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"### {get_text('db_management', 'serverless_label', dbms_type=dbms_type)}")
        with c2:
            st.success(get_text('db_management', 'serverless_info'))

    st.markdown('---')
    # --- SEZIONE DATABASE ---
    st.subheader(get_text('db_management', 'db_list_header'))

    # Lista DB
    dbs = manager.get_available_databases()
    
    # Filtri richiesti
    filtered_dbs = []
    selected_db = None
    for db in dbs:
        if dbms_type == 'PostgreSQL' and db == 'postgres': continue
        if dbms_type == 'MySQL' and db in ['performance_schema','information_schema','mysql','sys']: continue
        if dbms_type == 'SQL Server' and db in ['master' ,'model','msdb','tempdb']: continue
        filtered_dbs.append(db)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if not filtered_dbs:
            st.info(get_text('db_management', 'no_databases'))
        else:
            selected_db = st.selectbox(
                get_text('db_management', 'select_database'), 
                ['Choose a Database']+filtered_dbs, 
                key=f"sel_db_{dbms_type}",
                index=0)
    
    if selected_db and selected_db != 'Choose a Database':
        details = manager.get_db_details(selected_db)

        if "error" in details:
            st.error(get_text('db_management', 'error_loading', error=details['error']))
        else:
            with col2:
                st.write("")
                st.write("")
                st.markdown(f"**{get_text('db_management', 'size_label')}** {details.get('size_mb', 'N/A')}")
            with col3:
                st.write("")
                st.write("")
                st.markdown(f"**{get_text('db_management', 'tables_label')}** {len(details['tables'])}")
            with col4:
                st.write("")
                st.write("")
                rename = st.button(get_text('db_management', 'btn_rename_db'), key=f"p_btn_ren_db_{dbms_type}")
            with col5:
                st.write("")
                st.write("")
                delete = st.button('Delete Database', key=f"p_btn_del_db_{dbms_type}", type="primary")

            with col6:
                st.write("")
                st.write("")
                delete_all = st.button('Delete All Databases', key=f"p_btn_del_all_db_{dbms_type}", type="primary")
            with st.container(border=True if rename else False):
                c_ren, c_del = st.columns(2)
                # Rename DB
                if rename:
                    with c_ren:
                        a, b = st.columns(2)
                        with a:
                            new_db_name = st.text_input(get_text('db_management', 'new_name_label'), key=f"ren_db_in_{dbms_type}")
                        with b:
                            st.write("")
                            st.write("")
                            if st.button(get_text('db_management', 'btn_rename_db'), key=f"btn_ren_db_{dbms_type}"):
                                ok, msg = manager.rename_db(selected_db, new_db_name)
                                if ok: 
                                    st.toast(msg)
                                    time.sleep(1)
                                    st.rerun()
                                else: 
                                    st.toast(msg)
                    with c_del:
                        st.write("")
                        st.write("")
                        if st.button(get_text('db_management', 'btn_delete_db'), key=f"btn_del_db_{dbms_type}", type="primary"):
                            ok, msg = manager.delete_db(selected_db)
                            if ok:
                                st.toast(msg)
                                time.sleep(1)
                                st.rerun()
                            else: st.toast(msg)
            
            if delete:
                show_dialog(selected_db, dbms_type, details)
            
            if delete_all:
                show_dialog(filtered_dbs, dbms_type, {}, all=True)
        if details['tables']:
            # --- TABELLE ---
            st.markdown(f"### {get_text('db_management', 'tables_header')}")
            for t in details['tables']:
                with st.expander(get_text('db_management', 'table_expander', name=t['name'], rows=t['rows']), expanded=False):
                    st.write(f"**{get_text('db_management', 'columns_label')}** {len(t['columns'])}")

                    # Schema
                    df_schema = pd.DataFrame(t['columns'])
                    st.dataframe(df_schema, hide_index=True, width='stretch')
                        
                # Preview
                if t['preview'] is not None:
                    st.write(f"**{get_text('db_management', 'preview_label')}**")
                    st.dataframe(t['preview'], hide_index=True)
                
                # Rename Table
                c_ren_t, _ = st.columns([2, 1])
                with c_ren_t:
                    new_tbl_name = st.text_input(
                        get_text('db_management', 'rename_table_label', name=t['name']),
                        key=f"ren_tbl_{dbms_type}_{t['name']}"
                    )

                    if st.button(get_text('db_management', 'btn_rename_table'), key=f"btn_ren_tbl_{dbms_type}_{t['name']}"):
                        ok, msg = manager.rename_table(selected_db, t['name'], new_tbl_name)
                        if ok:
                            st.success(msg)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(msg)

    else:
        if not filtered_dbs:
            return
        else:
            with col2:
                st.write("")
                st.write("")
                delete_all = st.button('Delete All Databases', key=f"p_btn_del_all_db_{dbms_type}", type="primary")
                if delete_all:
                    show_dialog(filtered_dbs, dbms_type, {}, all=True)

@st.dialog("🌱 Delete Database!", width="large")
def show_dialog(selected_db, dbms_type, details, all=False):
    if not all:
        st.info(
            f"Are you sure you want to delete the database **{selected_db}**?"
        )
        st.warning("This action is irreversible!")

        with st.container(border=True):
            st.subheader("Database Details")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Database", selected_db)
            with col2:
                st.metric("Size", details.get("size_mb", "N/A"))
            with col3:
                st.metric("DBMS", dbms_type)
            with col4:
                st.metric("Tables", len(details["tables"]))

        col1, col2 = st.columns(2)

        # --------------- YES DELETE --------------------
        with col1:
            if st.button("Yes, delete", type="primary"):
                st.session_state["confirm_delete"] = True

                temp_ss = st.session_state
                temp_ss["config_dict"] = {
                    "choice_DBMS": dbms_type,
                    "db_name": selected_db,
                    "connection_string": "",
                    "dfs_dict": {"dummy": pd.DataFrame()},
                }

                manager = DBManager(temp_ss, type="status")
                ok, msg = manager.delete_db(selected_db)
                if ok:
                    st_toast_temp(msg, 'success')
                    time.sleep(1)
                    st.rerun()
                else:
                    st_toast_temp(msg, 'error')

        # ---------------- CANCEL ------------------------
        with col2:
            if st.button("Cancel", type="secondary"):
                st.session_state["confirm_delete"] = False
                st_toast_temp("Operation cancelled", 'info')
                st.rerun()
    elif all:
        st.info(
            f"Are you sure you want to delete all databases?"
        )
        st.warning("This action is irreversible!")
        
        for db in selected_db:
            st.write(db)
            st.write(dbms_type)

        col1, col2 = st.columns(2)

        # --------------- YES DELETE --------------------
        with col1:
            if st.button("Yes, delete", type="primary"):
                for db in selected_db:
                    temp_ss = st.session_state
                    temp_ss["config_dict"] = {
                        "choice_DBMS": dbms_type,
                        "db_name": db,
                        "connection_string": "",
                    "dfs_dict": {"dummy": pd.DataFrame()},
                }

                    manager = DBManager(temp_ss, type="status")
                    ok, msg = manager.delete_db(db)
                    if ok:
                        st_toast_temp(f"Database {db} deleted successfully", 'success')
                    else:
                        st_toast_temp(f"Database {db} not deleted", 'error')
                        st.warning(msg)
                        st.info(ok)
                st.session_state["confirm_delete"] = True
                time.sleep(1)
                st.rerun()

        # ---------------- CANCEL ------------------------
        with col2:
            if st.button("Cancel", type="secondary"):
                st.session_state["confirm_delete"] = False
                st_toast_temp("Operation cancelled", 'info')
                st.rerun()