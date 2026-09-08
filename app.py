import os
from threading import Thread

import streamlit as st

from GUI.benchmarking_tab import benchmark_tab
from GUI.conf_model import configure_local_model_tab, configure_online_model
from GUI.dataset_analytics_tab import db_analytics_tab
from GUI.db_management_tab import db_management_tab
from GUI.gen_eval_query import query_gen_eval_tab
from GUI.load_db_tab import load_db_tab
from GUI.load_file_tab import load_file_tab
from GUI.setting_tab import settings_tab
from GUI.synthetic_data_tab import synthetic_data_tab
from db_adapters.DBManager import check_service_status
from llm_adapters.lmstudio_adapter import run_server_lmStudio
from llm_adapters.ollama_adapter import run_server_ollama
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
        st.session_state.setdefault('show_welcome', True)  # Flag per mostrare il messaggio di benvenuto
        activate_service()

def activate_service():
    run_server_ollama()
    run_server_lmStudio()
    
    threads = []
    for dbms in ['MySQL', 'SQL Server', 'PostgreSQL']:
        thread = Thread(target=check_and_save_status, args=(dbms,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def check_and_save_status(dbms):
    status = check_service_status(dbms)
    if 'DBMS_Sever' not in st.session_state:
        st.session_state.DBMS_Sever = {
            "MySQL": {'status': 'not_running',},
            "SQL Server": {'status': 'not_running',},
            "PostgreSQL": {'status': 'not_running',},
        }
    st.session_state.DBMS_Sever[dbms]['status'] = status

initialize_session_state()

st.set_page_config(
    page_icon="🌱",
    page_title="CALIBER",
    layout="wide"
)
st.set_page_config(layout="wide")  # opzionale, solo per usare tutta la larghezza

st.markdown("""
    <style>
    /* Riduci il padding verticale del contenuto principale */
    .block-container {
        padding-top: 1.5rem;      /* prova 0, 0.5, 1… */
        padding-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
/* Riduci il margine sotto il titolo h1 (st.title) */
.block-container h1 {
    margin-bottom: 0rem;   /* prova 0 / 0.3 / 0.5 ecc. */
    padding-top: 0rem;      /* prova 0, 0.5, 1… */
    padding-bottom: 1rem;
}

/* Riduci il margine sopra il blocco delle tab */
.stTabs {
    margin-top: 0rem;        /* anche qui puoi giocare con i valori */
    padding-top: 0rem;      /* prova 0, 0.5, 1… */
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)



st.title("🌳 CALIBER")


html = """
<style>
/* Selettore per la barra delle tab */
div[data-baseweb="tab-list"] {
    justify-content: space-between;
}
"""

tab_list = [
    "📊 Data Hub",
    "🤖 Load Model",
    "📈 Data Insights",
    "🧪 Green Query Builder",
    "🎯 Eco Benchmark",
    "🧬 Synthetic Lab",
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

st.markdown("""
<style>
.stTabs [data-baseweb="tab"] p {
    font-weight: 700;
}
</style>
""",
unsafe_allow_html=True
)

st.markdown(html, unsafe_allow_html=True)
(
    dashboard_tab,
    load_model_tab,
    dataset_analytic_tab,
    generate_query_tab,
    benchmarking_tab,
    synthetic_tab,
    settings_page_tab,
) = st.tabs(tab_list)

with dashboard_tab:
    st.header("📊 Data Hub")

    load, managment = st.tabs(['📄Load Dataset', "🗄️ DBMS Management"])

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

    # --- GESTIONE DBMS ---
    with managment:
        with st.container(border=True):
            st.header("🗄️ DBMS Management")
            db_management_tab()

# --- LOAD MODEL (attiva solo dopo aver caricato un dataset, da qualunque sotto-tab di Data Hub) ---
with load_model_tab:
    st.header("🤖 Load Model")

    if len(list(st.session_state["dataframes"]["DBMS"].keys())) > 0:
        with st.container(border=True):
            tab1, tab2 = st.tabs([get_text("load_model", "local"), get_text("load_model", "online")])
            with tab1:
                configure_local_model_tab(key_prefix='configure_local_model')
            with tab2:
                configure_online_model(key_prefix='configure_online_model')
    else:
        st.info(get_text("gen_eval", "please_load_dataset"))

# --- DATASET ANALYTICS ---
with dataset_analytic_tab:
    st.header("📈 Data Insights")
    db_analytics_tab()

# --- GENERATE QUERY ---
with generate_query_tab:
    st.header("🧪 Green Query Builder")
    query_gen_eval_tab()


# --- BENCHMARKING ---
with benchmarking_tab:
    benchmark_tab()


# --- SYNTHETIC DATA ---
with synthetic_tab:
    st.header("🧬 Synthetic Lab")
    synthetic_data_tab()

# --- SETTINGS ---
with settings_page_tab:
    st.header("⚙️ Settings")
    settings_tab()
