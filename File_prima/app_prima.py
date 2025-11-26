import threading  # Importa threading per usarlo in initialize_session_state
import time

import streamlit as st

from GUI.db_management_tab import db_management_tab
from GUI.gen_eval_query import query_gen_eval_tab
from GUI.green_ai_race_tab import green_ai_race_tab
from GUI.history_tab import history_tab
from GUI.load_dataset_gui import load_db
from GUI.load_model_tab import load_model_tab
from GUI.setting_tab import settings_tab
from utils.history_manager import initialize_history_db

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        st.session_state.setdefault('current_df', None)
        st.session_state.setdefault('db_choice',"")
        st.session_state.setdefault('db_name', "")
        st.session_state.setdefault('db_connection_args', {})
        st.session_state.setdefault('current_DBMS', None)
        st.session_state.setdefault('llm_backend', None)
        # --- Contenitori per i risultati (in session_state per persistere tra i rerun) ---
        st.session_state.setdefault('process_results', {})

        st.session_state.setdefault('monitoring_data', [])
        st.session_state.setdefault('open_tab', 'generate_query')

        st.session_state.setdefault('upload_database_from',
                                    None)  # Evento per capire da dove ha caricato i dati (CSV or DBMS)
        st.session_state.setdefault('load_DBMS',
                                    False)  # Evento per capire se ha inserito i dati nella sezione "Select Dataset"

        # ----- Gestione Tab -------
        st.session_state.setdefault('current_tab', 'Dashboard')  # Evento per capire quale TAB ha premuto
        st.session_state.setdefault('selected_bar', 'Barra1')  # Barra sopra

        st.session_state.setdefault('dataframes', {'files': {}, 'DBMS': {}})
        st.session_state.setdefault('uploaded_files', {})  # Scelta per capire quanti file ha caricato
        st.session_state.setdefault('uploaded_dbms', {})
        st.session_state.setdefault('choice_files',
                                    'None')  # Scelta per capire se (nel caso di duplicati) ha fatto una scelta
        st.session_state.setdefault('duplicate_files', [])  # controllo file duplicati
        # flag di edit per-tab
        st.session_state.setdefault('server_lmStudio', False)
        st.session_state.setdefault('server_ollama', False)

        st.session_state.setdefault("edit_sep_flags", {})  # dict: name -> bool
        st.session_state.setdefault('create_db_done', False)
        st.session_state.setdefault('selected_bar', 'Barra1')
        st.session_state.setdefault('results_HF', {})
        st.session_state.setdefault('submit_HF', None)
        st.session_state.setdefault('token_HF', '')
        st.session_state.setdefault('db_dir', "database")

        st.session_state.setdefault('hf_dl', {
            "running": False,
            "stop": False,
            "progress": 0,
            "bytes": 0,
            "total": 0,
            "retries": 0,
            "max_retries": 4,
            "note": "",
            "error": None,
            "local_dir": None,
            "started_at": None,
            "pipe": None,
            "thread": None,
            "model_id": None,  # <-- MODIFICA CHIAVE
        })

        import spacy.util

        st.session_state.setdefault('spacy_model',
                                    {'model': 'en_core_web_sm',
                                     'status': 'Load' if spacy.util.is_package("en_core_web_sm") else 'notLoad'}
                                    )
        # Stato LLM attivo (separato da spaCy)
        st.session_state.setdefault('llm', {
            'backend': None,  # "LM Studio" | "Ollama" | "Hugging Face" | "Local (Upload)"
            'model': None,  # string (nome/id del modello)
            'status': 'notLoad',  # "notLoad" | "selected" | "loaded"
            'kwargs': {}  # es. host/token/filter/... necessari per l'adapter
        })
        st.session_state.setdefault('detailed_spacy', {})

        st.session_state.setdefault('DBMS_Sever', {
            "MySQL": {'status': 'not_running',},
            "SQL Server": {'status': 'not_running',},
            "PostgreSQL": {'status': 'not_running',},
        })


initialize_session_state()

initialize_history_db()

st.set_page_config(page_icon="🧭", page_title="Query with LLMs", layout="wide")
st.title("Generate and Evaluate Query")

html = """
    <style>
    /* Selettore per la barra delle tab */
    div[data-baseweb="tab-list"] {
        justify-content: space-between;
    }

    /* Seleziona i singoli tab e cambia ordine */ """


tab_list = [
    "📄 Load Dataset",
    "🤖 Load Model",
    "🧪 Generate & Evaluate Query",
    "🏁 Green AI Race",  # NUOVO TAB
    "🗄️ Gestione DBMS",
    '📜 History',
    "⚙️ Settings"
]

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
    div[data-baseweb="tab-list"] button:nth-child(5) { order: 5; }  /* LLM 3 */
    div[data-baseweb="tab-list"] button:nth-child(6) { order: 6; }  /* LLM 3 */
    div[data-baseweb="tab-list"] button:nth-child(7) { order: 7; }  /* LLM 3 */
    div[data-baseweb="tab-list"] button:nth-child(8) { order: 8; }  /* LLM 3 */
    </style>
""", unsafe_allow_html=True)


# Tabs con icone + testo in inglese
dashboard, load_model, gen_eval_query, green_race, db_mgmt, History_tab, settings = st.tabs()

with dashboard:
    st.header("📄 Load Dataset")
    with st.container(border=True):
        load_db(key_prefix="main")

with load_model:
    st.header("🤖 Load Model")
    load_model_tab()

with gen_eval_query:
    st.header("🧪 Generate & Evaluate Query")
    query_gen_eval_tab()

with green_race:
    st.header("🏁 Green AI Race")
    green_ai_race_tab()

# --- BLOCCO COMMENTATO E RIMOSSO PER PULIZIA ---
# with charts_tab:
#     st.header("📊 Charts")
#     generate_charts()

with History_tab:
    history_tab()

with db_mgmt:
    db_management_tab()

with settings:
    st.header("⚙️ Settings")
    settings_tab()