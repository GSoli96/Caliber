import os
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx
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
from llm_adapters.huggingface_adapter import ensure_model_cached
from GUI.message_gui import st_toast_temp
from utils.load_config import get_HF_Token  # se già lo importi altrove, lascia pure

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
        st.session_state['widget_race_idx_counter'] = 0
        st.session_state.setdefault('show_welcome', True)  # Flag per mostrare il messaggio di benvenuto
        
        st.session_state.setdefault('race', {
            'A':{
            'backend': None,  # "LM Studio" | "Ollama" | "Hugging Face" | "Local (Upload)"
            'model': None,  # string (nome/id del modello)
            },
            'B':{
            'backend': None,  # "LM Studio" | "Ollama" | "Hugging Face" | "Local (Upload)"
            'model': None,  # string (nome/id del modello)
            },
            'submit': False
        })

        st.session_state.setdefault('race_progress', 0)
        st.session_state.setdefault('race_status', 'not_running')

        if "tasks_A" not in st.session_state:
            st.session_state.tasks_A = {
                "Started": False,
                "Start Monitoring": False,
                "Construct prompt": False,
                "Sending prompt": False,
                "LLM has generated SQL query": False,
                "Executing SQL query": False,
                "SQL query executed": False,
                "Saving results": False,
                "Stopping Monitoring": False,
                "End Challenge": False
    }

        if "tasks_B" not in st.session_state:
            st.session_state.tasks_B = {
                "Started": False,
                "Start Monitoring": False,
                "Construct prompt": False,
                "Sending prompt": False,
                "LLM has generated SQL query": False,
                "Executing SQL query": False,
                "SQL query executed": False,
                "Saving results": False,
                "Stopping Monitoring": False,
                "End Challenge": False
    }
        st.session_state.setdefault('race_results', {})
        activate_service()

def activate_service():   
    threads = []
    dbms = ['MySQL', 'SQL Server', 'PostgreSQL']
    for dbm in dbms:
        thread = Thread(target=check_and_save_status, args=(dbm,))
        add_script_run_ctx(thread)
        thread.start()
        threads.append(thread)
    
    th = Thread(target=run_server_ollama)
    add_script_run_ctx(th)
    th.start()
    threads.append(th)

    th = Thread(target=run_server_lmStudio)
    add_script_run_ctx(th)
    th.start()
    threads.append(th)

    backend = "Hugging Face"
    model_id = "meta-llama/Meta-Llama-3-8B"
    hf_token = get_HF_Token()

    th = Thread(target=ensure_model_cached, args=(model_id, backend, hf_token))
    add_script_run_ctx(th)
    th.start()
    threads.append(th)
    st_toast_temp("Initializing DBMS and LLMs Server Completed!", "success")

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
    .block-container {
        padding-top: 1.5rem;      
        padding-bottom: 0rem;
        margin_button: 0rem;
    }

.block-container h1 {
    margin-bottom: 0rem;   
    padding-top: 1rem;      
    padding-bottom: 0.5rem;
}

.stTabs {
    margin-top: 0rem;           
    padding-top: 0rem;      
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

tab_list = [get_text("app_home", "data_hub"),
    get_text("app_home", "data_insight"),
    get_text("app_home", "green_query_builder"),
    get_text("app_home", "eco_benchmark"),
    get_text("app_home", "synthetic_data"),
    get_text("app_home", "settings"),
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
    dataset_analytic_tab,
    generate_query_tab,
    benchmarking_tab,
    synthetic_tab,
    settings_page_tab,
) = st.tabs(tab_list)

with dashboard_tab:
    st.header(get_text("app_home", "data_hub"))

    load, managment = st.tabs([
        get_text("app_home", "load_dataset"), 
        get_text("app_home", "db_management")])

    with load:
        with st.container(border=True):
            st.subheader(get_text("app_home", "load_dataset"))

            tab1, tab2 = st.tabs([
                get_text("load_dataset", "tab_file_upload"),
                get_text("load_dataset", "tab_dbms_connection")])

            with tab1:
                load_file_tab("tab1")

            with tab2:
                load_db_tab("tab2")

        if len(list(st.session_state["dataframes"]["DBMS"].keys())) > 0:
            with st.container(border=True):
                st.subheader(get_text("app_home", "load_model"))
                tab1, tab2 = st.tabs([
                    get_text("load_model", "local"), 
                    get_text("load_model", "online")])
                with tab1:
                    configure_local_model_tab(key_prefix='configure_local_model')
                with tab2:
                    configure_online_model(key_prefix='configure_online_model')

    # --- GESTIONE DBMS ---
    with managment:
        with st.container(border=True):
            st.header(get_text("app_home", "db_management"))
            db_management_tab()

# --- DATASET ANALYTICS ---
with dataset_analytic_tab:
    st.header(get_text("app_home", "data_insight"))
    db_analytics_tab()

# --- GENERATE QUERY ---
with generate_query_tab:
    st.header(get_text("app_home", "green_query_builder"))
    query_gen_eval_tab()


# --- BENCHMARKING ---
with benchmarking_tab:
    st.header(get_text("app_home", "eco_benchmark2")) 
    benchmark_tab()


# --- SYNTHETIC DATA ---
with synthetic_tab:
    st.header(get_text("app_home", "synthetic_data"))
    synthetic_data_tab()

# --- SETTINGS ---
with settings_page_tab:
    st.header(get_text("app_home", "settings"))
    settings_tab()


# CSS personalizzato per styling premium
st.markdown("""
    <style>
    .footer {
        margin: 0rem;
        padding: 0.5rem;
        text-align: center;
        color: #888;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>🌳 CALIBER - Carbon-Aware LLM-Integrated Benchmarking & Eco-Responsible Query Rewriting</p>
        <p>Developed with 💚 for a sustainable future</p>
    </div>
""", unsafe_allow_html=True)