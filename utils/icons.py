from __future__ import annotations


class Icons:
    """
    Centralized icon repository for the application UI.
    
    Provides emoji icons for various UI elements, tabs, actions, metrics,
    and status indicators. Icons can be accessed via class attributes or
    the get_icon() method using human-readable keys.
    """
    # App & Tabs
    APP_ICON = "🧭"
    TAB_DATASET = "📄"
    TAB_MODEL = "🤖"
    TAB_QUERY = "🧪"
    TAB_RACE = "🏁"
    TAB_DBMS = "🗄️"
    TAB_SETTINGS = "⚙️"

    # LLM Adapters
    HUGGING_FACE = "🤗"
    OLLAMA = "🦙"
    LM_STUDIO = "🧪"
    LOCAL_UPLOAD = "📤"
    SPACY = "🧠"
    MODEL = "📥"

    # Actions & UI Elements
    HOST = "🌐"
    FILTER = "🔎"
    SUBMIT = "✅"
    REFRESH = "🔄"
    RESET = "♻️"
    SAVE = "💾"
    HF_TOKEN = "🔑🤗"
    LOAD_LIST = "📥"
    CLEAR_LIST = "🗑️"
    GLOBAL_RESET = "🧹♻️"
    LOAD_MODELS = "📥🤖"
    DETAILS = "🔎"
    JSON = "🧾"
    RUN = "▶️"
    RESPONSE = "💬"
    CLEAR_OUTPUT = "🧹"
    SELECT_MODEL = "🎯"
    METADATA = "🏷️"
    LOAD_MODEL = "📥🤖"
    LLM_SOURCE = "🌐"
    INSTALLED_MODELS = "📦"
    INSTALL_MODEL = "⬇️🤖"
    CHOOSE_MODEL = "🎯"
    DOWNLOAD_MODELS = "⬇️📦"

    # Misc
    FOLDER = "📁"
    INFO = "ℹ️"
    WARNING = "⚠️"
    ERROR = "❌"
    SUCCESS = "✅"
    LOADING = "⏳"
    PAUSE = "⏸️"
    STOP = "⏹️"
    PLAY = "▶️"

    # Green AI Race
    CHALLENGER_A = "🔵"
    CHALLENGER_B = "🟢"
    TEST_QUERY = "📝"
    RACE_RESULTS = "📊"
    WINNER = "🏆"
    DETAILED_METRICS = "📋"
    NEW_RACE = "🔄"

    # Dashboard / General
    DASHBOARD = "📊"
    OVERVIEW = "📈"
    HOME = "🏠"
    CONSOLE = "🖥️"
    PIN = "📌"
    FAVORITE = "⭐"

    # Metrics & Performance
    CPU_USAGE = "🧠"
    GPU_USAGE = "🎛️"
    RAM_USAGE = "💽"
    LATENCY = "⏱️"
    THROUGHPUT = "📡"
    TOKENS_PER_SECOND = "🔢"
    COST = "💸"
    REQUESTS = "📨"
    TEMPERATURE = "🌡️"

    # Status / Health
    STATUS_OK = "🟢"
    STATUS_WARNING = "🟡"
    STATUS_ERROR = "🔴"
    ONLINE = "🟢"
    OFFLINE = "⚫"
    DEGRADED = "🟠"
    HEALTH_CHECK = "🩺"

    # Logs / Events / Monitoring
    LOGS = "📚"
    EVENTS = "📅"
    TIMELINE = "🕒"
    MONITORING = "📡"
    ALERTS = "🚨"
    NOTIFICATIONS = "🔔"

    # Trends / Comparison
    TREND_UP = "📈"
    TREND_DOWN = "📉"
    COMPARE = "⚖️"
    BENCHMARK = "🎯"

    # Green / Energy
    ENERGY = "⚡"
    POWER_USAGE = "🔋"
    CO2_EMISSIONS = "🌫️"
    GREEN_SCORE = "🌱"

    # 🔹 UN SOLO DIZIONARIO PER TUTTO
    ICONS = {
        # App & Tabs (nomi "umani")
        "App": APP_ICON,
        "Dataset Tab": TAB_DATASET,
        "Model Tab": TAB_MODEL,
        "Query Tab": TAB_QUERY,
        "Race Tab": TAB_RACE,
        "DBMS Tab": TAB_DBMS,
        "Settings Tab": TAB_SETTINGS,

        # LLM Adapters (etichette UI)
        "Hugging Face": HUGGING_FACE,
        "Ollama": OLLAMA,
        "LM Studio": LM_STUDIO,
        "Local (Upload)": LOCAL_UPLOAD,
        "Spacy": SPACY,
        "Model": MODEL,

        # UI labels (già presenti, INVARIATE)
        "Host": HOST,
        "Filtro": FILTER,
        "Submit": SUBMIT,
        "Refresh": REFRESH,
        "Reset": RESET,
        "Save": SAVE,
        "Hugging Face Token": HF_TOKEN,
        "Load List": LOAD_LIST,
        "Clear List": CLEAR_LIST,
        "Global Reset": GLOBAL_RESET,
        "Load Models": LOAD_MODELS,
        "Details": DETAILS,
        "JSON": JSON,
        "Run": RUN,
        "Response": RESPONSE,
        "Clear Output": CLEAR_OUTPUT,
        "Select a Model": SELECT_MODEL,
        "Metadata": METADATA,
        "Load Model": LOAD_MODEL,
        "LLM Model Source": LLM_SOURCE,
        "Installed Models": INSTALLED_MODELS,
        "Install Model": INSTALL_MODEL,
        "Choose a Model": CHOOSE_MODEL,
        "Selected Model Details": DETAILS,
        "Download Models": DOWNLOAD_MODELS,
        "Warning": WARNING,
        "Error": ERROR,
        "Success": SUCCESS,
        "Info": INFO,
        "Loading": LOADING,
        "Pause": PAUSE,
        "Stop": STOP,
        "Play": PLAY,

        # Green AI Race (etichette UI)
        "Challenger A": CHALLENGER_A,
        "Challenger B": CHALLENGER_B,
        "Test Query": TEST_QUERY,
        "Race Results": RACE_RESULTS,
        "Winner": WINNER,
        "Detailed Metrics": DETAILED_METRICS,
        "New Race": NEW_RACE,

        # Dashboard / General (NUOVE CHIAVI IN INGLESE)
        "Dashboard": DASHBOARD,
        "Overview": OVERVIEW,
        "Home": HOME,
        "Console": CONSOLE,
        "Pin": PIN,
        "Favorite": FAVORITE,

        # Metrics & Performance
        "CPU Usage": CPU_USAGE,
        "GPU Usage": GPU_USAGE,
        "RAM Usage": RAM_USAGE,
        "Latency": LATENCY,
        "Throughput": THROUGHPUT,
        "Tokens per Second": TOKENS_PER_SECOND,
        "Cost": COST,
        "Requests": REQUESTS,
        "Temperature": TEMPERATURE,

        # Status / Health
        "Status OK": STATUS_OK,
        "Status Warning": STATUS_WARNING,
        "Status Error": STATUS_ERROR,
        "Online": ONLINE,
        "Offline": OFFLINE,
        "Degraded": DEGRADED,
        "Health Check": HEALTH_CHECK,

        # Logs / Events / Monitoring
        "Logs": LOGS,
        "Events": EVENTS,
        "Timeline": TIMELINE,
        "Monitoring": MONITORING,
        "Alerts": ALERTS,
        "Notifications": NOTIFICATIONS,

        # Trends / Comparison
        "Trend Up": TREND_UP,
        "Trend Down": TREND_DOWN,
        "Compare": COMPARE,
        "Benchmark": BENCHMARK,

        # Green / Energy
        "Energy": ENERGY,
        "Power Usage": POWER_USAGE,
        "CO2 Emissions": CO2_EMISSIONS,
        "Green Score": GREEN_SCORE,

        # Opzionale: chiavi "costanti" usate nel codice
        "APP_ICON": APP_ICON,
        "TAB_DATASET": TAB_DATASET,
        "TAB_MODEL": TAB_MODEL,
        "TAB_QUERY": TAB_QUERY,
        "TAB_RACE": TAB_RACE,
        "TAB_DBMS": TAB_DBMS,
        "TAB_SETTINGS": TAB_SETTINGS,
        "WARNING": WARNING,
        "ERROR": ERROR,
        "SUCCESS": SUCCESS,
        "INFO": INFO,
    }

    @classmethod
    def get_icon(cls, key: str, default: str = "") -> str:
        """
        Ritorna un'icona dal dizionario unico `ICONS`.
        - `key` può essere una label UI (es. "Host", "Hugging Face")
          oppure una costante (es. "WARNING", "TAB_MODEL").
        """
        # Prima cerca nel dizionario principale
        if key in cls.ICONS:
            return cls.ICONS[key]

        # Fallback: prova a usare il nome di una costante di classe
        if hasattr(cls, key):
            value = getattr(cls, key)
            if isinstance(value, str):
                return value

        return default
