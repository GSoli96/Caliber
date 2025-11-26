import os

import streamlit as st

from File_prima.GUI.history_tab import history_tab
from File_prima.GUI.setting_tab import settings_tab
from GUI.gen_eval_query import query_gen_eval_tab
# from File_prima.GUI.green_ai_race_tab import green_ai_race_tab
# from File_prima.GUI.history_tab import history_tab
# from File_prima.GUI.setting_tab import settings_tab
from GUI.dataset_analytics_tab import db_analytics_tab
from GUI.db_management_tab import db_management_tab
from GUI.green_ai_race_tab import green_ai_race_tab
from GUI.load_db_tab import load_db_tab
from GUI.load_file_tab import load_file_tab
from llm_adapters.lmstudio_adapter import start_server_background, run_server_lmStudio
from llm_adapters.ollama_adapter import run_server_ollama
from utils.history_manager import initialize_history_db
from utils.translations import get_text
import streamlit as st
from GUI.conf_model import configure_local_model_tab, configure_online_model
from utils.translations import get_text

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state['dataframes'] = {'files': {}, 'DBMS': {}}
        st.session_state['duplicate_files'] = []
        st.session_state['uploaded_files'] = {}
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
        st.session_state['server_ollama'] = False
        st.session_state['process_status'] = None
        if not os.path.exists(st.session_state.db_dir):
            os.makedirs(st.session_state.db_dir)

        st.session_state["selected_by_backend_a"] = None
        st.session_state["selected_by_backend_b"] = None
        st.session_state["selected_by_backend"] = None

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

        st.session_state['widget_idx_counter'] = 0

        run_server_ollama()
        run_server_lmStudio()

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
"""

tab_list = [
    "📊 Dashboard",
    "📈 Dataset Analytics",
    "🧪 Generate Query",
    "🏁 Green AI Race",
    "🎯 benchmarking",
    "🧬 sintetic data",
    "📜 Hystory",
    "⚙️ Setting",
]

# nth-child è 1-based, quindi start=1
for order, tab in enumerate(tab_list, start=1):
    html += (
        f'div[data-baseweb="tab-list"] button:nth-child({order}) '
        + '{ order: ' + str(order) + '; }\n'
        f'/* {tab} (index: {order}) */\n'
    )

html += "</style>"

st.markdown(html, unsafe_allow_html=True)
(
    dashboard_tab,
    dataset_analytic_tab,
    generate_query_tab,
    green_race_tab,
    benchmarking_tab,
    synthetic_data_tab,
    history_page_tab,
    settings_page_tab,
) = st.tabs(tab_list)

with dashboard_tab:
    st.header("📊 Dashboard (NON TOCCARE)")

    load, managment = st.tabs(['📄Load Dataset', "🗄️ Gestione DBMS"])

    with load:
        with st.container(border=True):
            st.subheader("📄 Load Dataset")

            tab1, tab2 = st.tabs([
                get_text("load_dataset", "tab_file_upload"),
                get_text("load_dataset", "tab_dbms_connection")])

            with tab1:
                load_file_tab("tab1")

            with tab2:
                load_db_tab("tab2")

        if len(list(st.session_state["dataframes"]["DBMS"].keys())) > 0:
            with st.container(border=True):
                st.subheader("🤖 Load Model")
                tab1, tab2 = st.tabs([get_text("load_model", "local"), get_text("load_model", "online")])
                with tab1:
                    configure_local_model_tab(key_prefix='configure_local_model')
                with tab2:
                    configure_online_model(key_prefix='configure_online_model')

    # --- GESTIONE DBMS ---
    with managment:
        with st.container(border=True):
            st.header("🗄️ Gestione DBMS (NON TOCCARE)")
            db_management_tab()

# --- DATASET ANALYTICS ---
with dataset_analytic_tab:
    st.header("📈 Dataset Analytics")
    db_analytics_tab()

# --- GENERATE QUERY ---
with generate_query_tab:
    st.header("🧪 Generate Query")
    # Riutilizzo della tua funzione esistente
    query_gen_eval_tab()

# --- GREEN AI RACE ---
with green_race_tab:
    st.header("🏁 Green AI Race")
    green_ai_race_tab()


# --- BENCHMARKING ---
with benchmarking_tab:
    st.header("🎯 Benchmarking")
    # benchmarking_tab() se/quando la definisci
    st.info("Benchmarking – coming soon.")


# --- SYNTHETIC DATA ---
with synthetic_data_tab:
    st.header("🧬 Synthetic Data")
    # synthetic_data_tab() se/quando la definisci
    st.info("Synthetic data generation – coming soon.")


# --- HISTORY ---
with history_page_tab:
    st.header("📜 History")
    history_tab()


# --- SETTINGS ---
with settings_page_tab:
    st.header("⚙️ Settings")
    settings_tab()
