import streamlit as st
import pandas as pd
import time
from db_adapters.DBManager import DBManager
from GUI.message_gui import st_toast_temp
import sys

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
    temp_ss['config_dict'] = {
        'choice_DBMS': dbms_type,
        'db_name': 'temp', # placeholder
        'connection_string': '', # placeholder, userà defaults o global config
    }
    # Assicuriamoci che dfs_dict esista (anche vuoto) per evitare return anticipato in __init__
    temp_ss['dfs_dict'] = {'dummy': pd.DataFrame()} 

    manager = DBManager(temp_ss, type="status")
    manager.choice_DBMS = dbms_type

    # --- SEZIONE SERVER ---
    
    # Layout differente per Server-based vs Serverless
    if dbms_type in ["MySQL", "PostgreSQL", 'SQL Server']:
        # Riga 1: Titolo

        
        # Riga 2: Status e Controlli
        c1, c2 = st.columns([1, 1])

        if st.session_state.DBMS_Sever[dbms_type]['status'] == 'not_running':
            st.markdown(f"### 🖥️ Server: {dbms_type}")
            with c1:
                # Check status automatico
                ok, msg = manager.server_control('status')
                is_running = ok and msg == "Running"

                if is_running:
                    st.markdown("#### Status: ✅ Running")
                    st.session_state.DBMS_Sever[dbms_type]['status'] = 'running'
                else:
                    st.markdown("#### Status: 🔴 Stopped")
                    st.session_state.DBMS_Sever[dbms_type]['status'] = 'not_running'
            with c2:
                # Start Button - Disabled if running
                if st.button(
                        f"▶️ Start {dbms_type}",
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
                st.markdown(f"### 🖥️ Server: {dbms_type}")
            with c2:
                st.markdown("#### Status: ✅ Running")
            st.session_state.DBMS_Sever[dbms_type]['status'] = 'running'
        
        # Riga 3: Placeholder per errori (o toast sopra)
        st.empty()
    else:
        # Serverless (SQLite, DuckDB)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"### 📂 DBMS: {dbms_type}")
        with c2:
            st.success(" Serverless (File-based)")

    st.markdown('---')
    # --- SEZIONE DATABASE ---
    st.subheader("📂 Database List")

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
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if not filtered_dbs:
            st.info("ℹ️ No databases found.")
        else:
            selected_db = st.selectbox("📂 Select Database", filtered_dbs, key=f"sel_db_{dbms_type}")
        
    if selected_db:
        # --- DETTAGLI DB ---
        details = manager.get_db_details(selected_db)

        if "error" in details:
            st.error(f"❌ Error loading details: {details['error']}")
        else:
            with col2:
                st.write("")
                st.write("")
                st.markdown(f"**📏 Size:** {details.get('size_mb', 'N/A')}")
            with col3:
                st.write("")
                st.write("")
                st.markdown(f"**📊 Tables:** {len(details['tables'])}")
            with col4:
                st.write("")
                st.write("")
                rename = st.button("✏️ Rename DB", key=f"p_btn_ren_db_{dbms_type}")

            use_col1, use_col2 = st.columns([2, 5])
            with use_col1:
                use_dataset = st.button(
                    "✅ Use this dataset",
                    key=f"btn_use_db_{dbms_type}_{selected_db}",
                    type="primary",
                    use_container_width=True,
                )
            if use_dataset:
                use_config = {
                    'db_choice': dbms_type,
                    'choice_DBMS': dbms_type,
                    'db_name': selected_db,
                }
                if dbms_type in ["SQLite", "DuckDB"]:
                    use_config['path_to_file'] = selected_db

                st.session_state['uploaded_dbms'][selected_db] = use_config
                # Solo un dataset attivo alla volta
                st.session_state["dataframes"]["DBMS"] = {}

                with st.spinner(f"Loading {selected_db}...", show_time=True):
                    mgr_use = DBManager({'config_dict': use_config}, 'download')
                    dumped, ok_use = mgr_use.download_db()

                if ok_use:
                    st.session_state['create_db_done'] = True
                    st.session_state['db_choice'] = dbms_type
                    st.session_state['db_name'] = selected_db
                    st.session_state["dataframes"]["DBMS"][selected_db] = dumped
                    st_toast_temp(f"✅ Dataset {selected_db} in use.", 'success')
                    st.rerun()
                else:
                    st.error(f"❌ Failed to load dataset {selected_db}.")

            with st.container(border=True if rename else False):
                c_ren, c_del = st.columns(2)
                # Rename DB
                if rename:
                    with c_ren:
                        a, b = st.columns(2)
                        with a:
                            new_db_name = st.text_input("✏️ New Name", key=f"ren_db_in_{dbms_type}")
                        with b:
                            st.write("")
                            st.write("")
                            if st.button("✏️ Rename DB", key=f"btn_ren_db_{dbms_type}"):
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
                        if st.button("🗑️ Delete Dataset", key=f"btn_del_db_{dbms_type}", type="primary"):
                            ok, msg = manager.delete_db(selected_db)
                            if ok:
                                st.toast(msg)
                                time.sleep(1)
                                st.rerun()
                            else: st.toast(msg)

        if details['tables']:
            # --- TABELLE ---
            st.markdown("### 📋 Tables")
            for t in details['tables']:
                with st.expander(f"📄 Table: {t['name']} ({t['rows']} rows)", expanded=False):
                    st.write(f"**📊 Columns:** {len(t['columns'])}")

                    # Schema
                    df_schema = pd.DataFrame(t['columns'])
                    st.dataframe(df_schema, hide_index=True, width='stretch')
                        
                # Preview
                if t['preview'] is not None:
                    st.write("**👁️ Preview:**")
                    st.dataframe(t['preview'], hide_index=True)
                
                # Rename Table
                c_ren_t, _ = st.columns([2, 1])
                with c_ren_t:
                    new_tbl_name = st.text_input(
                        f"✏️ Rename {t['name']} to:",
                        key=f"ren_tbl_{dbms_type}_{t['name']}"
                    )

                    if st.button("✏️ Rename Table", key=f"btn_ren_tbl_{dbms_type}_{t['name']}"):
                        ok, msg = manager.rename_table(selected_db, t['name'], new_tbl_name)
                        if ok:
                            st.success(msg)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(msg)

