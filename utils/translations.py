# utils/translations.py
import streamlit as st

TRANSLATIONS = {
    "sidebar": {
        "system_info": {"en": "🧠 System Info", "it": "🧠 Info Sistema"},
        "cpu": {"en": "🧩 CPU", "it": "🧩 CPU"},
        "cores": {"en": "🧮 Cores:", "it": "🧮 Cores:"},
        "frequency": {"en": "⏱️ Frequency:", "it": "⏱️ Frequenza:"},
        "tdp_max": {"en": "🔥 TDP Max (Est.):", "it": "🔥 TDP Max (Stim.):"},
        "cpu_watt": {"en": "⚡ CPU Watt:", "it": "⚡ CPU Watt:"},
        "usage": {"en": "📊 Usage:", "it": "📊 Utilizzo:"},

        "ram": {"en": "💾 RAM", "it": "💾 RAM"},
        "total": {"en": "Total:", "it": "Totale:"},
        "available": {"en": "Available:", "it": "Disponibile:"},

        "gpu": {"en": "🎮 GPU", "it": "🎮 GPU"},
        "name": {"en": "Name:", "it": "Nome:"},
        "memory_total": {"en": "Total Memory:", "it": "Memoria Totale:"},
        "gpu_watt": {"en": "⚡ GPU Watt:", "it": "⚡ GPU Watt:"},

        "co2_instant": {"en": "🌱 CO₂ Emissions (Instant)", "it": "🌱 CO₂ Emissioni Istantanee"},
        "cpu_gs": {"en": "CPU g/s:", "it": "CPU g/s:"},
        "cpu_kgh": {"en": "CPU kg/h:", "it": "CPU kg/h:"},
        "gpu_gs": {"en": "GPU g/s:", "it": "GPU g/s:"},
        "gpu_kgh": {"en": "GPU kg/h:", "it": "GPU kg/h:"},

        "co2_total": {"en": "🌱 CO₂ Emissions (Total)", "it": "🌱 CO₂ Emissioni Totali"},
        "cpu_total": {"en": "CPU Total:", "it": "CPU Totale:"},
        "gpu_total": {"en": "GPU Total:", "it": "GPU Totale:"},

        "total_consumption": {"en": "📉 Total Consumption:", "it": "📉 Consumi Totali:"},
    },

    "tabs": {
        "load_dataset": {"en": "📄 Load Dataset", "it": "📄 Carica Dataset"},
        "load_model": {"en": "🤖 Load Model", "it": "🤖 Carica Modello"},
        "gen_eval": {"en": "🧪 Generate & Evaluate Query", "it": "🧪 Genera & Valuta Query"},
        "history": {"en": "📜 History", "it": "📜 Cronologia"},
        "settings": {"en": "⚙️ Settings", "it": "⚙️ Impostazioni"},
    },
    "settings": {
        "header": {
            "en": "Customize application settings according to your preferences.",
            "it": "Personalizza le impostazioni dell'applicazione secondo le tue preferenze."
        },

        "language": {"en": "🌐 Language", "it": "🌐 Lingua"},
        "select_language": {"en": "🗣️ Select language:", "it": "🗣️ Seleziona la lingua:"},
        "current_language": {"en": "📍 Current language:", "it": "📍 Lingua attuale:"},

        "theme": {"en": "🎨 Theme", "it": "🎨 Tema"},
        "theme_info": {
            "en": "💡 You can change the light/dark theme from the top right menu (☀️/🌙).",
            "it": "💡 Puoi cambiare il tema light/dark dal menu in alto a destra (☀️/🌙)."
        },

        "co2_config": {"en": "🌱 CO₂ Estimation Config", "it": "🌱 Configurazione Stime CO₂"},
        "emission_factor": {"en": "⚖️ Emission Factor (gCO₂/kWh)", "it": "⚖️ Fattore Emissione (gCO₂/kWh)"},
        "cpu_tdp": {"en": "🔥 CPU TDP (W) for estimates", "it": "🔥 CPU TDP (W) per stime"},

        "db_dir": {"en": "📁 Database Directory", "it": "📁 Directory Database"},
    },

    "gen_eval": {
        "describe_request": {"en": "Describe your request", "it": "Descrivi la tua richiesta"},
        "generate_btn": {"en": "🚀 Generate", "it": "🚀 Genera"},
        "analyze_spacy": {"en": "🧪 Analyze with spaCy", "it": "🧪 Analizza con spaCy"},
        "spacy_error": {"en": "Unable to load spaCy model", "it": "Impossibile caricare il modello spaCy"},
        "ner_expander": {"en": "Named Entity Recognition (NER)", "it": "Riconoscimento Entità (NER)"},
        "running_msg": {"en": "Execution and monitoring in progress...", "it": "Esecuzione e monitoraggio in corso…"},
        "live_co2_chart": {"en": "Total Cumulative CO₂ (Live)", "it": "CO₂ Cumulativo Totale (Live)"},
        "original_results": {"en": "Original Generation Results", "it": "Risultati della Generazione Originale"},
        "generated_sql": {"en": "Generated SQL Query", "it": "Query SQL Generata"},
        "execution_result": {"en": "Execution Result", "it": "Risultato Esecuzione"},
        "db_error": {"en": "DB Error:", "it": "Errore DB:"},
        "query_executed": {"en": "Query executed.", "it": "Query eseguita."},
        "rows": {"en": "Rows", "it": "Righe"},
        "optimized_proposals": {"en": "Optimized Query Proposals", "it": "Proposte di Query Ottimizzate"},
        "no_alternatives": {"en": "No alternative queries were generated.", "it": "Nessuna query alternativa è stata generata."},
        "alternative_attempt": {"en": "Alternative Attempt", "it": "Tentativo Alternativa"},
        "no_sql_output": {"en": "No SQL output generated.", "it": "Nessun output SQL generato."},
        "success_exec": {"en": "Execution completed successfully.", "it": "Esecuzione completata con successo."},
        "query_not_exec": {"en": "Query not executed:", "it": "Query non eseguita:"},
        "reason_unspecified": {"en": "Reason unspecified", "it": "Motivo non specificato"},
        "all_failed": {"en": "None of the generated alternative queries could be executed successfully.", "it": "Nessuna delle query alternative generate ha potuto essere eseguita con successo."},
        "report_header": {"en": "Execution and Sustainability Report", "it": "Report di Esecuzione e Sostenibilità"},
        "start_gen": {"en": "Start LLM Gen", "it": "Inizio Generazione LLM"},
        "end_gen_start_db": {"en": "End Gen / Start DB", "it": "Fine Gen / Inizio DB"},
        "end_db": {"en": "End DB Exec", "it": "Fine Esecuzione DB"},
        "phase_gen": {"en": "LLM Generation", "it": "Generazione LLM"},
        "phase_db": {"en": "DB Execution", "it": "Esecuzione DB"},
        "insufficient_data": {"en": "Insufficient data for phase comparison.", "it": "Dati insufficienti per il confronto delle fasi."},
        "preview_dataset": {"en": "🔍 Preview Dataset", "it": "🔍 Preview Dataset"},
        "table_not_found": {"en": "Table {name} in dataset {db_name} not found!", "it": "Table {name} in dataset {db_name} non trovato!"},
        "no_db_loaded": {"en": "No database (from DBMS) has been loaded yet.", "it": "Nessun database (da DBMS) è stato ancora caricato."},
        "no_table_found": {"en": "No table found or loaded for database '{db_name}'.", "it": "Nessuna tabella trovata o caricata per il database '{db_name}'."},
        "please_load_dataset": {"en": "Please Load dataset.", "it": "Please Load dataset."},
        "please_load_llm": {"en": "Please Load an LLM.", "it": "Please Load an LLM."},
        "model_config": {"en": "Model Configuration", "it": "Configurazione Modello"},
        "selected_model": {"en": "Selected Model", "it": "Modello Selezionato"},
        "backend": {"en": "Backend", "it": "Backend"},
        "model": {"en": "Model", "it": "Modello"},
        "status": {"en": "Status", "it": "Stato"},
        "exec_error": {"en": "An error occurred during execution:", "it": "Si è verificato un errore durante l'esecuzione:"},
    },
    "load_dataset": {
        "tab_discovery": {'en': 'FD Discovery',"it": "FD Discovery"},
        "header": {"en": "Load Dataset", "it": "Carica Dataset"},
        "tab_file_upload": {"en": "📁 File Upload", "it": "📁 Caricamento File"},
        "tab_dbms_connection": {"en": "📂 Connection to DBMS", "it": "📂 Connessione a DBMS"},
        "upload_header": {"en": "Upload one or more files", "it": "Carica uno o più file"},
        "upload_info_single": {"en": "If you upload **a single file**, it will be interpreted as **a table**.", "it": "Se carichi **un singolo file**, verrà interpretato come **una tabella**."},
        "upload_info_multiple": {"en": "If you upload **multiple files**, each file will represent a different table.", "it": "Se carichi **più file**, ogni file rappresenterà una tabella diversa."},
        "expander_upload": {"en": "📂 Upload one or more files", "it": "📂 Carica uno o più file"},
        "success_upload": {"en": "Successfully Uploaded {n} Files.", "it": "Caricati con successo {n} File."},
        "reset_files": {"en": "🗑️ Reset all Files", "it": "🗑️ Resetta tutti i File"},
        "db_config": {"en": "🗄️⚙️ Database Configuration", "it": "🗄️⚙️ Configurazione Database"},
        "dbms_config": {"en": "DBMS Configuration", "it": "Configurazione DBMS"},
        "connect_db_header": {"en": "Connecting to a Database", "it": "Connessione a un Database"},
        "connect_db_info": {"en": "Enter the connection string to the database.", "it": "Inserisci la stringa di connessione al database."},
        "dbms_success": {"en": "DBMS successfully uploaded.", "it": "DBMS caricato con successo."},
        "choose_files": {"en": "Choose the file files", "it": "Scegli i file"},
        "load_files_btn": {"en": "📥 Load Files", "it": "📥 Carica File"},
        "choose_files_csv": {"en": "Choose one or more files", "it": "Scegli uno o più file"},
        "separator": {"en": "Separator", "it": "Separatore"},
        "custom_separator": {"en": "Enter custom separator:", "it": "Inserisci il separatore personalizzato:"},
        "warning_duplicates": {"en": "Warning. {n} duplicate files detected.", "it": "Attenzione. Rilevati {n} file duplicati."},
        "choice_success": {"en": "Choice made successfully. Choice made: {choice}", "it": "Scelta effettuata con successo. Scelta: {choice}"},
        "choose_action": {"en": "👉 Choose an action:", "it": "👉 Scegli un'azione:"},
        "action_continue": {"en": "Continue anyway", "it": "Continua comunque"},
        "action_remove": {"en": "Remove previously uploaded datasets", "it": "Rimuovi dataset caricati precedentemente"},
        "action_keep": {"en": "Keep previous and ignore new files", "it": "Mantieni precedenti e ignora nuovi file"},
        "duplicate_kept": {"en": "Duplicate kept. Now loaded: ", "it": "Duplicato mantenuto. Ora caricati: "},
        "replaced_datasets": {"en": "Replaced {n} dataset(s) with the newly uploaded ones.", "it": "Sostituiti {n} dataset con quelli appena caricati."},
        "ignored_duplicates": {"en": "Ignored {n} duplicate file(s); previous versions kept.", "it": "Ignorati {n} file duplicati; versioni precedenti mantenute."},
        "select_action_warning": {"en": "Please select an action to continue. This action cannot be undone.", "it": "Seleziona un'azione per continuare. Questa azione non può essere annullata."},
        "db_engine": {"en": "Database Engine", "it": "Motore Database"},
        "conn_string": {"en": "Connection string (JDBC)", "it": "Stringa di connessione (JDBC)"},
        "password": {"en": "Password", "it": "Password"},
        "db_name": {"en": "Database name", "it": "Nome Database"},
        "path_to_file": {"en": "Path to file DB", "it": "Percorso file DB"},
        "db_path_create": {"en": "Database file path (to create)", "it": "Percorso file Database (da creare)"},
        "db_path_help": {"en": "Enter the path and name for the new database file (e.g., ./my_new_db.db)", "it": "Inserisci percorso e nome per il nuovo file database (es. ./mio_nuovo_db.db)"},
        "load_db_btn": {"en": "📥 Load Database", "it": "📥 Carica Database"},
        "create_db_btn": {"en": "🆕 Create Database", "it": "🆕 Crea Database"},
        "insert_db_warning": {"en": "Please insert a DB.", "it": "Inserisci un DB."},
        "check_params_warning": {"en": "Please check params (MySQL)", "it": "Controlla i parametri (MySQL)"},
        "enter_db_path_error": {"en": "Please enter the database file path.", "it": "Per favore, inserisci il percorso del file database."},
        "correct_db_path_error": {"en": "Please enter a correct .db file path", "it": "Per favore, un percorso corretto di un file .db"},
        "path_is_dir_error": {"en": "The specified path is a directory, not a file: '{path}'.", "it": "Il percorso specificato è una directory, non un file: '{path}'."},
        "file_not_found_error": {"en": "File not found: '{path}'. Check the path and try again.", "it": "File non trovato: '{path}'. Verifica il percorso e riprova."},
        "enter_create_path_error": {"en": "Please enter the path to create the database.", "it": "Per favore, inserisci il percorso per creare il database."},
        "edit_config": {"en": "⚙️ Edit configuration", "it": "⚙️ Modifica configurazione"},
        "edit_separator": {"en": "✏️ Edit separator", "it": "✏️ Modifica separatore"},
        "save": {"en": "💾 Save", "it": "💾 Salva"},
        "cancel": {"en": "❌ Cancel", "it": "❌ Annulla"},
        "custom_sep_1char": {"en": "Custom separator (1 char)", "it": "Separatore personalizzato (1 car)"},
        "sep_updated": {"en": "Separator for **{name}** updated.", "it": "Separatore per **{name}** aggiornato."},
        "reload_error": {"en": "Error reloading dataset with new separator.", "it": "Errore nel ricaricare il dataset con il nuovo separatore."},
        "filename": {"en": "Filename:", "it": "Nome file:"},
        "separator_label": {"en": "Separator:", "it": "Separatore:"},
        "not_found_error": {"en": "Error: {name} not found in session_state after loading.", "it": "Errore: {name} non trovato in session_state dopo il caricamento."},
        "load_df_error": {"en": "Unable to load DataFrame for {name}.", "it": "Impossibile caricare il DataFrame per {name}."},
        "dataset_not_found": {"en": "❌ Dataset {name} not found!", "it": "❌ Dataset {name} non trovato!"},
        "dataset_details": {"en": "📊 Dataset Details", "it": "📊 Dettagli Dataset"},
        "tab_preview": {"en": "🔍 Preview", "it": "🔍 Anteprima"},
        "tab_detailed": {"en": "📊 Detailed Statistic", "it": "📊 Statistiche Dettagliate"},
        "tab_info": {"en": "🧾 Info Dataset", "it": "🧾 Info Dataset"},
        "tab_missing&profiling": {"en": "📚 Missing values & Profiling", "it": "📚 Valori Mancanti e Profiling"},
        "tab_profiling": {"en": "📚 Profiling", "it": "📚 Profiling"},
        "tab_integrity": {"en": "🛡️ Integrity", "it": "🛡️ Integrità"},
        "tab_export": {"en": "📦 Export", "it": "📦 Export"},
        "config_metadata_missing": {"en": "Configuration metadata not found for '{db_name}'. Displayed info might be incomplete.", "it": "Metadati di configurazione non trovati per '{db_name}'. Le informazioni visualizzate potrebbero essere incomplete."},
        "dbms_type": {"en": "🖥️ DBMS Type", "it": "🖥️ Tipo DBMS"},
        "tables_loaded": {"en": "🗂️ Loaded Tables", "it": "🗂️ Tabelle Caricate"},
        "db_path": {"en": "Database Path", "it": "Percorso Database"},
        "weight": {"en": "⚖️ Weight", "it": "⚖️ Peso"},
        "file_moved": {"en": "File not found (might have been moved).", "it": "Il file non è stato trovato (potrebbe essere stato spostato)."},
        "calc_weight_error": {"en": "Unable to calculate file weight.", "it": "Impossibile calcolare il peso del file."},
        "sqlite_path_missing": {"en": "SQLite file path not specified.", "it": "Percorso del file SQLite non specificato."},
        "conn_string_label": {"en": "Connection String", "it": "Stringa di Connessione"},
        "conn_string_unavailable": {"en": "Connection string unavailable.", "it": "Stringa di connessione non disponibile."},
        "config_req_tables": {"en": "Configuration: Loading required for {n} tables/columns.", "it": "Configurazione: Caricamento richiesto per {n} tabelle/colonne."},
        "config_req_all": {"en": "Configuration: Loading required for all tables.", "it": "Configurazione: Caricamento richiesto per tutte le tabelle."},
        "db_info": {"en": "🗄️ Database Info", "it": "🗄️ Info Database"},
        "explore_tables": {"en": "📋 Explore Tables", "it": "📋 Esplora Tabelle"},
        "db_exists_warning": {"en": "⚠️ A database with this name already exists. Choose an action:", "it": "⚠️ Esiste già un database con questo nome. Scegli un'azione:"},
        "datetime": {"en": "Date/Time", "it": "Data/Ora"},
        "boolean": {"en": "Boolean", "it": "Booleano"},
        "other": {"en": "Other", "it": "Altro"},
        "dataset_specs": {"en": "🧭 Dataset Exploration", "it": "🧭 Esplorazione del Dataset"},
        "num_rows": {"en": "📏 Number of rows", "it": "📏 Numero di righe"},
        "num_cols": {"en": "📐 Number of columns", "it": "📐 Numero di colonne"},
        "missing_values": {"en": "⚠️ Missing values (%)", "it": "⚠️ Valori mancanti (%)"},
        "duplicate_rows": {"en": "🔁 Duplicate rows", "it": "🔁 Righe duplicate"},
        "mem_usage": {"en": "💾 Memory usage:", "it": "💾 Utilizzo memoria:"},

        "tab_overview": {"en": "📊 Overview", "it": "📊 Panoramica"},
        "tab_cardinality": {"en": "🔢 Cardinality", "it": "🔢 Cardinalità"},
        "tab_columns": {"en": "📋 Columns", "it": "📋 Colonne"},
        "tab_missing": {"en": "⚠️ Missing", "it": "⚠️ Missing"},
        "tab_correlations": {"en": "🔗 Correlations", "it": "🔗 Correlazioni"},

        "desc_stats": {"en": "Descriptive statistics", "it": "Statistiche descrittive"},
        "cardinality_col": {"en": "Cardinality by column", "it": "Cardinalità per colonna"},
        "unique_count": {"en": "Unique Count", "it": "Conteggio Unici"},
        "column": {"en": "Column", "it": "Colonna"},
        "unique_pct": {"en": "Unique (%)", "it": "Valori Unici (%)"},
        "schema_cols": {"en": "Column Schema", "it": "Schema delle colonne"},
        "schema_caption": {"en": "Name color = type | ⚠️ Missing % | 🔒 Sensitive (PII/Pseudonyms detected via spaCy/heuristic keywords)", "it": "Colore del nome = tipo | ⚠️ Missing % | 🔒 Sensibile (PII/Pseudonimi rilevati via spaCy/keyword euristiche)"},
        "sensitive": {"en": "Sensitive", "it": "Sensibile"},
        "reason": {"en": "Reason", "it": "Motivo"},
        "example": {"en": "Example", "it": "Esempio"},
        "missing_dist": {"en": "Missing values distribution", "it": "Distribuzione dei valori mancanti"},
        "missing_hint": {"en": "Tip: consider imputation or exclusion for columns with high Missing %.", "it": "Suggerimento: considera imputazione o esclusione per le colonne con Missing % elevato."},
        "corr_matrix": {"en": "Correlation matrix (numeric columns only)", "it": "Matrice di correlazione (solo colonne numeriche)"},
        "top_corr_pairs": {"en": "Top correlated pairs (absolute):", "it": "Coppie più correlate (assoluto):"},
        "no_num_cols": {"en": "Not enough numeric columns to calculate correlation.", "it": "Non ci sono almeno due colonne numeriche per calcolare la correlazione."},
        "rows_to_show": {"en": "Rows to display", "it": "Righe da visualizzare"},
        "rows_to_show_help": {"en": "Select the number of rows to show in the table preview.", "it": "Seleziona il numero di righe da mostrare nell'anteprima della tabella."},
        "legend": {"en": "Legend", "it": "Legenda"},
        "legend_title": {"en": "Legend of symbols and colors", "it": "Legenda dei simboli e dei colori"},
        "missing_pct": {"en": "Missing %", "it": "Missing %"},
        "legend_col_color": {"en": "**Column name color** → indicates the *data type*:", "it": "**Colore del nome della colonna** → indica la *tipologia di dato*:"},
        "legend_missing": {"en": "**⚠️ Missing %** → percentage of missing values in the column.", "it": "**⚠️ Missing %** → percentuale di valori mancanti nella colonna."},
        "legend_sensitive": {"en": "**🔒 Sensitive** → the column contains or might contain *personal or sensitive information* identified via linguistic analysis (spaCy model).", "it": "**🔒 Sensibile** → la colonna contiene o potrebbe contenere *informazioni personali o sensibili* identificate tramite analisi linguistica (modello spaCy)."},
        "expander_files": {"en": "📋 Tables Overview", "it": "📋 Panoramica delle tabelle"},
        "expander_schema": {"en": "Schema", "it": "Schema"},
        "expander_cardinality": {"en": "Cardinality", "it": "Cardinalità"},
        "expander_missing": {"en": "Missing", "it": "Missing"},
        "expander_correlations": {"en": "Correlations", "it": "Correlazioni"},
        "expander_desc_stats": {"en": "Descriptive Statistics", "it": "Statistiche descrittive"},
        "expander_corr_matrix": {"en": "Correlation Matrix", "it": "Matrice di correlazione"},
        "text": {"en": "Text", "it": "Testo"},
        "numeric": {"en": "Numeric", "it": "Numerico"},
        "categorical": {"en": "Categorical", "it": "Categorico"},
        "type": {"en": "Type", "it": "Tipo"},
        "export_header": {"en": "📦 Export Analytics", "it": "📦 Export Analisi"},
        "export_zip_btn": {"en": "📦 Generate ZIP Export", "it": "📦 Genera Export ZIP"},
        "exporting_zip": {"en": "Preparing export archive...", "it": "Preparazione archivio export..."},
        "download_zip": {"en": "📥 Download Analytics ZIP", "it": "📥 Scarica ZIP Analisi"},
    },
    "history": {
        "no_entries": {"en": "No entries in history. Run a query in the 'Generate & Evaluate Query' tab to see results here.", "it": "Nessuna entry nella cronologia. Esegui una query nella scheda 'Generate & Evaluate Query' per vedere i risultati qui."},
        "entries_found": {"en": "**{n}** executions found.", "it": "**{n}** esecuzioni trovate."},
        "run_title": {"en": "Run of {date} (Model: {model})", "it": "Run del {date} (Modello: {model})"},
        "user_question": {"en": "User Question:", "it": "Domanda Utente:"},
        "status": {"en": "Status", "it": "Stato"},
        "total_duration": {"en": "Total Duration (s)", "it": "Durata Totale (s)"},
        "llm_backend": {"en": "LLM Backend", "it": "LLM Backend"},
        "db_engine": {"en": "DB Engine", "it": "Engine DB"},
        "main_error": {"en": "Main Error:", "it": "Errore Principale:"},
        "original_query": {"en": "Original Query", "it": "Query Originale"},
        "db_exec_duration": {"en": "DB Execution Duration (s)", "it": "Durata Esecuzione DB (s)"},
        "db_exec_error": {"en": "DB Execution Error:", "it": "Errore Esecuzione DB:"},
        "query_executed_rows": {"en": "Query executed. Rows returned: {rows}", "it": "Query eseguita. Righe restituite: {rows}"},
        "json_error": {"en": "Unable to read query result (malformed JSON).", "it": "Impossibile leggere il risultato della query (JSON malformattato)."},
        "no_orig_query": {"en": "No original query generated (probably due to an error).", "it": "Nessuna query originale generata (probabilmente a causa di un errore)."},
        "alt_queries": {"en": "Alternative Queries", "it": "Query Alternative"},
        "no_alt_queries": {"en": "No alternative queries were generated.", "it": "Nessuna query alternativa è stata generata."},
        "alt_attempts": {"en": "{n} alternative attempts generated.", "it": "{n} tentativi di alternative generati."},
        "alternative_n": {"en": "Alternative {n}", "it": "Alternativa {n}"},
        "no_sql_output": {"en": "No SQL output generated.", "it": "Nessun output SQL generato."},
        "success_duration": {"en": "Status: Success (Duration: {s:.3f}s)", "it": "Stato: Successo (Durata: {s:.3f}s)"},
        "failed_reason": {"en": "Status: Failed. Reason: {reason}", "it": "Stato: Fallito. Motivo: {reason}"},
        "alt_json_error": {"en": "Unable to read alternative queries (malformed JSON).", "it": "Impossibile leggere le query alternative (JSON malformattato)."},
        "consumption_report": {"en": "Consumption Report", "it": "Report Consumi"},
        "no_monitor_data": {"en": "No monitoring data saved.", "it": "Nessun dato di monitoraggio salvato."},
        "monitor_json_error": {"en": "Unable to read monitoring data (malformed JSON).", "it": "Impossibile leggere i dati di monitoraggio (JSON malformattato)."},
        "chart_error": {"en": "Error generating consumption charts: {e}", "it": "Errore durante la generazione dei grafici di consumo: {e}"},
    },
    "load_model": {
        "local": {"en": "🖥️ Local Model", "it": "🖥️ Modello Locale"},
        "online": {"en": "☁️ Online Model", "it": "☁️ Modello Online"},
    },
    "profiling": {
        "anomalies_integrity_header": {"en": "🧮 Anomalies & Integrity", "it": "🧮 Anomalie & Integrità"},
        "export_header": {"en": "📦 Export", "it": "📦 Export"},
        "integrity_quality_header": {"en": "**Integrity & Quality** — rows: **{rows:,}**, columns: **{cols:,}**", "it": "**Integrità & Qualità** — righe: **{rows:,}**, colonne: **{cols:,}**"},
        "outlier_threshold": {"en": "Outlier threshold (|z|>…)", "it": "Soglia outlier (|z|>…)"},
        "future_date_tolerance": {"en": "Future date tolerance (days)", "it": "Tolleranza date future (giorni)"},
        "min_year": {"en": "Minimum valid year", "it": "Anno minimo valido"},
        "run": {"en": "Run", "it": "Esegui"},
        "analyzing_anomalies": {"en": "Analyzing Anomalies...", "it": "Analisi Anomalie..."},
        "export_info": {"en": "The package includes: summary.json, uniqueness.csv, anomalies.csv, fds.csv, semantic.csv", "it": "Il pacchetto include: summary.json, uniqueness.csv, anomalies.csv, fds.csv, semantic.csv"},
        "preparing_zip": {"en": "Preparing ZIP...", "it": "Preparazione ZIP..."},
        "download_zip": {"en": "📦 Download profile (ZIP)", "it": "📦 Scarica profilo (ZIP)"},
        "rel_profiling_header": {"en": "**Relational Profiling** — rows: **{rows:,}**, columns: **{cols:,}**", "it": "**Profiling relazionale** — righe: **{rows:,}**, colonne: **{cols:,}**"},
        "lazy_analysis_info": {"en": "Run analyses only when needed. Each tab launches a thread with a progress bar.", "it": "Esegui le analisi solo quando ti servono. Ogni tab lancia un thread con progress bar."},
        "semantic_profiling": {"en": "🧠 Semantic Profiling", "it": "🧠 Profiling Semantico"},
        "heatmap_types": {"en": "🌡️ Heatmap & Types", "it": "🌡️ Heatmap & Tipi"},
        "sample_rows": {"en": "Sample rows per column", "it": "Sample righe per colonna"},
        "use_spacy": {"en": "Use spaCy (if available)", "it": "Usa spaCy (se disponibile)"},
        "semantic_profiling_progress": {"en": "Semantic Profiling...", "it": "Profiling Semantico..."},
        "max_rows_sample": {"en": "Max rows to sample", "it": "Max righe da campionare"},

        "merge_dtypes": {"en": "Merge with types (dtype)", "it": "Unisci con tipi (dtype)"},
        "preparing_heatmap": {"en": "Preparing heatmap...", "it": "Preparazione heatmap..."},
        "heatmap_unavailable": {"en": "Heatmap unavailable: matplotlib/seaborn not installed.", "it": "Heatmap non disponibile: matplotlib/seaborn non installati."},
    },
    "conf_model": {
    "select_model_header": {"en": "🔀 Model Selection (Local / Server)", "it": "🔀 Selezione modello (locale / server)"},
    "host_lm_studio": {"en": "🌐 LM Studio Host", "it": "🌐 Host LM Studio"},
    "filter_contains": {"en": "🔎 Filter (contains)", "it": "🔎 Filtro (contiene)"},
    "host_ollama": {"en": "🌐 Ollama Host", "it": "🌐 Host Ollama"},
    "hf_token_opt": {"en": "🔑 HF Token (opt.)", "it": "🔑 HF token (opz.)"},
"spacy_warning": {
  "en": "⚠️ Note:**spaCy models are _not_ generative models.** They are used only for **linguistic analysis** (e.g., tokenization, part-of-speech tagging, named entity recognition) and for **detecting potentially sensitive information (PII)** in your data or natural-language prompts.\n\nAll processing runs **locally on your machine**; however, **you remain responsible** for ensuring that any personal or confidential data is handled in compliance with your own policies and applicable regulations.\n\nDetected entities may be **incomplete or contain errors** and should always be **manually reviewed** before use.",
  "it": "⚠️ Nota:I modelli **spaCy non sono modelli generativi.** Vengono utilizzati esclusivamente per **l’analisi linguistica** (ad es. tokenizzazione, part-of-speech tagging, named entity recognition) e per **individuare potenziali informazioni sensibili (PII)** nei dati o nei prompt in linguaggio naturale.\n\nL’elaborazione avviene **localmente sulla tua macchina**; tuttavia, **resta tua responsabilità** garantire che eventuali dati personali o confidenziali siano gestiti in conformità alle tue policy e alle normative vigenti.\n\nLe entità rilevate possono essere **incomplete o contenere errori** e dovrebbero sempre essere **verificate manualmente** prima dell’uso."
},



    "load_list": {"en": "📥 Load List", "it": "📥 Carica elenco"},
    "clear_list": {"en": "🗑️ Clear", "it": "🗑️ Svuota"},
    "global_reset": {"en": "🧹♻️ Global Reset", "it": "🧹♻️ Reset globale"},
    "list_cleared": {"en": "List cleared.", "it": "Elenco svuotato."},

    "loading_models": {"en": "⏳ Loading models from {backend}…", "it": "⏳ Carico modelli da {backend}…"},
    "no_models_found": {"en": "❌ No models found.", "it": "❌ Nessun modello trovato."},
    # "models_found": {"en": "📦 Found {n} models.", "it": "📦 Trovati {n} modelli."},
    "server_not_running": {"en": "🚨 Server is not Running!", "it": "🚨 Il server non è in esecuzione!"},

    "available_model": {"en": " Available Model", "it": "Modello disponibile"},
    "details": {"en": "🔍 Details", "it": "🔍 Dettagli"},
    "test_generation": {"en": "Test Generation", "it": "Prova generazione"},
    "server_cli": {"en": "💻 Server (CLI)", "it": "💻 Server (CLI)"},

    "model_overview": {"en": "🧩 Model Overview", "it": "🧩 Panoramica Modello"},
    "explanation_details": {"en": "📘 Explanation Details", "it": "📘 Dettagli Spiegazione"},
    "open_explanations": {"en": "📖 Open detailed explanations", "it": "📖 Apri spiegazioni dettagliate"},
    "raw_json": {"en": "🪶 Raw JSON", "it": "🪶 JSON Grezzo"},

    "prompt": {"en": "Prompt", "it": "Prompt"},
    "prompt_placeholder": {"en": "Hello model! Summarize this sentence in 10 words.", "it": "Ciao modello! Riassumi questa frase in 10 parole."},

    "max_new_tokens": {"en": "max_new_tokens", "it": "max_new_tokens"},
    "run": {"en": "▶️ Run", "it": "▶️ Esegui"},
    "inference": {"en": "⚙️ Inference…", "it": "⚙️ Inferenza…"},
    "response": {"en": "💬 Response", "it": "💬 Risposta"},
    "clear_output": {"en": "🧹 Clear Output", "it": "🧹 Pulisci output"},
    "use_model": {"en": "✅ Use this model", "it": "✅ Usa questo modello"},
    "active_model": {"en": "🎯 Active Model: {backend} / {sel}", "it": "🎯 Modello attivo: {backend} / {sel}"},
    "set_filters_info": {"en": "🔎 Set filters and start a search.", "it": "🔎 Imposta i filtri e avvia una ricerca."},
    "no_models_criteria": {"en": "❌ No models found with these criteria.", "it": "❌ Nessun modello trovato con questi criteri."},
    # "models_found_hf": {"en": "📦 Found {n} models", "it": "📦 Trovati {n} modelli"},
    # "models_found_header": {"en": "📦 Models Found", "it": "📦 Modelli Trovati"},
    "search_invalid_id": {"en": "❗ Search returned items without valid ID.", "it": "❗ La ricerca ha restituito elementi senza id valido."},
    "select_model": {"en": "🎯Select a model", "it": "🎯Seleziona un modello"},
    "language": {"en": "Language: {lang}", "it": "Lingua: {lang}"},
    "full_metadata": {"en": "📘 Full Metadata", "it": "📘 Metadati completi"},
    "load_model_btn": {"en": "📥 Load Model", "it": "📥 Carica Modello"},
    "local_cache": {"en": "📁 Local Cache: {path}", "it": "📁 Cache locale: {path}"},
    "set_active_llm": {"en": "✅ Set as active LLM", "it": "✅ Imposta come LLM attivo"},
    "item_not_found": {"en": "❌ Selected item not found in results.", "it": "❌ Elemento selezionato non trovato tra i risultati."},
    "model_source": {"en": "Model Source", "it": "Fonte del Modello"},
    "local_model_available": {"en": "📦 Local model available", "it": "📦 Modello locale disponibile"},
    "model_details": {"en": "🔍 Model Details", "it": "🔍 Dettagli Modello"},
    "details_error": {"en": "❌ Unable to load details: {error}", "it": "❌ Impossibile caricare i dettagli: {error}"},
    "installed_spacy": {"en": "🧠 Installed spaCy models", "it": "🧠 Modelli spaCy installati"},
    "select_below": {"en": "Select the model in the loading section below.", "it": "Seleziona il modello nella sezione di caricamento qui sotto."},

    "no_spacy_found": {
        "en": "No spaCy model found.\n\nYou can install one from the list below or \n\nvia terminal: `python -m spacy download it_core_news_sm`",
        "it": "Nessun modello spaCy trovato.\n\nPuoi installarne uno dalla lista qui sotto oppure \n\nda terminale: `python -m spacy download it_core_news_sm`"
    },

    "install_spacy": {"en": "⬇️ Install a spaCy model", "it": "⬇️ Installa un modello spaCy"},
    "choose_install": {"en": "Choose a model to install", "it": "Scegli un modello da installare"},
    "install_help": {"en": "List of frequent models; you can reinstall even if already present.", "it": "Lista di modelli frequenti; puoi reinstallare anche se già presente."},
    "verbose_logs": {"en": "Verbose logs", "it": "Log verbosi"},
    "verbose_help": {"en": "Show full downloader output.", "it": "Mostra l'output completo del downloader."},
    "selected_details": {"en": "📘 Selected Model Details", "it": "📘 Dettagli modello selezionato"},

    "download_model_btn": {"en": "⬇️ Download model", "it": "⬇️ Scarica modello"},
    "installing": {"en": "Installing {name}…", "it": "Installazione di {name} in corso…"},
    "install_success": {"en": "✅ Model `{name}` installed successfully.", "it": "✅ Modello `{name}` installato correttamente."},
    "install_failed": {"en": "❌ Installation failed for `{name}` (return code {rc}). Check logs above and retry.", "it": "❌ Installazione fallita per `{name}` (return code {rc}). Controlla i log sopra e riprova."},
    "downloading": {"en": "**⬇️ Downloading:** {text}", "it": "**⬇️ Download in corso:** {text}"},

    "overview": {"en": "🧩 Overview", "it": "🧩 Panoramica"},
    "pipeline_task": {"en": "🛠️ Pipeline & Task", "it": "🛠️ Pipeline & Task"},
    "size_resources": {"en": "📏 Size & Resources", "it": "📏 Dimensioni & risorse"},

    "model_label": {"en": "Model", "it": "Modello"},
    "lang_label": {"en": "Language", "it": "Lingua"},
"version_label": {"en": "Version", "it": "Versione"},
"spacy_compat": {"en": "spaCy compatible", "it": "spaCy compatibile"},
"vectors": {"en": "Vectors", "it": "Vettori"},
"installed": {"en": "Installed", "it": "Installato"},
"yes": {"en": "Yes", "it": "Sì"},
"no": {"en": "No", "it": "No"},

"path_label": {"en": "📦 Path: `{path}`", "it": "📦 Path: `{path}`"},
"notes": {"en": "Notes: ", "it": "Note: "},

"pipeline_components": {"en": "🛠️ Pipeline components", "it": "🛠️ Componenti pipeline"},
"supported_tasks": {"en": "📋 Supported tasks", "it": "📋 Task supportati"},
"expected_size": {"en": "📏 Expected size", "it": "📏 Dimensione attesa"},
"dependencies": {"en": "🔗 Dependencies", "it": "🔗 Dipendenze"},

"deps_transformers": {"en": "`transformers`, `torch` (GPU/CUDA recommended)", "it": "`transformers`, `torch` (consigliata GPU/CUDA)"},
"deps_spacy": {"en": "only `spacy`", "it": "solo `spacy`"},
"vectors_included": {"en": "vectors included in model package", "it": "vettori inclusi nel pacchetto del modello"},

"upload_local_header": {"en": "📦 Upload a local model", "it": "📦 Carica un modello locale"},
"upload_local_caption": {"en": "You can upload one or more files, including `.zip`/`.tar.gz` archives. They will be extracted and saved to a local folder.", "it": "Puoi caricare uno o più file, anche archivi `.zip`/`.tar.gz`. Verranno estratti e salvati in una cartella locale."},

"select_files": {"en": "📁 Select one or more model files", "it": "📁 Seleziona uno o più file del modello"},
"model_name_folder": {"en": "Model name (destination folder)", "it": "Nome modello (cartella di destinazione)"},
"hard_validation": {"en": "Block save if essential files are missing ('hard' validation)", "it": "Blocca salvataggio se mancano file essenziali (validazione “hard”)"},
"save_local_btn": {"en": "💾 Save model locally", "it": "💾 Salva modello localmente"},

"no_file_selected": {"en": "❌ No file selected.", "it": "❌ Nessun file selezionato."},
"save_cancelled": {"en": "❌ Save cancelled: essential files missing for detected format.", "it": "❌ Salvataggio annullato: mancano file essenziali per il formato rilevato."},
"missing_files": {"en": "Missing: ", "it": "Mancanti: "},
"model_saved": {"en": "✅ Model '{name}' saved in {folder}", "it": "✅ Modello '{name}' salvato in {folder}"},
"format_detected": {"en": "Format detected: **{fmt}**", "it": "Formato rilevato: **{fmt}**"},
"soft_validation_warning": {"en": "⚠️ Some suggested files are missing (soft validation): ", "it": "⚠️ Alcuni file suggeriti non sono presenti (validazione soft): "},

"uploaded_models_header": {"en": "### 📂 Uploaded Models", "it": "### 📂 Modelli caricati"},
"no_models_uploaded": {"en": "No models uploaded. Use the form above to add one.", "it": "Nessun modello caricato. Usa il form sopra per aggiungerne uno."},
"delete_model": {"en": "🗑️ Delete model", "it": "🗑️ Elimina modello"},
"model_deleted": {"en": "Model '{name}' deleted.", "it": "Modello '{name}' eliminato."},

"hf_explorer_title": {"en": "🤗 Hugging Face Model Explorer", "it": "🤗 Hugging Face Model Explorer"},
"filter_task": {"en": "Filter by task", "it": "Filtro per task"},
"filter_task_help": {"en": "Select a model type (empty = no filter)", "it": "Seleziona un tipo di modello (vuoto = nessun filtro)"},
"author_org": {"en": "Author / Organization", "it": "Autore / Organizzazione"},
"sort_by": {"en": "Sort by", "it": "Ordina per"},
"search_placeholder": {"en": "🔎 Search in name or description", "it": "🔎 Cerca nel nome o nella descrizione"},
"max_results": {"en": "Max results", "it": "Numero massimo di risultati"},
"hf_token_help": {"en": "HF token (optional, for private models)", "it": "HF token (opzionale, per modelli privati)"},
"search_btn": {"en": "Search models", "it": "Cerca modelli"},
"insert_author": {"en": "✏️ Please insert the author.", "it": "✏️ Please insert the author."},
"insert_author_token": {"en": "🔑 Please insert the author and your HF token.", "it": "🔑 Please insert the author and your HF token."},
"no_models_found_retry": {"en": "❌ No models found, change filters and retry!", "it": "❌ Nessun modello trovato, cambia i filtri e riprova!"},
# "models_found_count": {"en": "📦 Found {n} models", "it": "📦 Trovati {n} modelli"},

"lm_studio_title": {"en": "🧪 LM Studio", "it": "🧪 LM Studio"},
"lm_host_help": {"en": "Local OpenAI-compatible server (e.g. http://localhost:1234)", "it": "Server locale OpenAI-compatible (es. http://localhost:1234)"},

"refresh_cache": {"en": "🔄 Refresh LM Studio (cache)", "it": "🔄 Aggiorna LM Studio (cache)"},
"refresh_cache_help": {"en": "🧹 Clear cache for /v1/models", "it": "🧹 Svuota cache per /v1/models"},

"models_exposed": {"en": "📡 Models exposed by server: {n}", "it": "📡 Modelli esposti dal server: {n}"},
"models_exposed_help": {"en": "These are the models *currently exposed* by the local server.", "it": "Questi sono i modelli *attualmente esposti* dal server locale."},

"lms_connection_error": {"en": "❌ LM Studio server unreachable. Open LM Studio and enable local server (port 1234).", "it": "❌ LM Studio server non raggiungibile. Apri LM Studio e abilita il server locale (porta 1234)."},
"lms_error": {"en": "❌ LM Studio Error: {e}", "it": "❌ Errore LM Studio: {e}"},

"set_active_lms": {"en": "🎯 Set active model (LM Studio)", "it": "🎯 Imposta modello attivo (LM Studio)"},
"model_id_exposed": {"en": "Model ID exposed by server (copy from list above)", "it": "Model ID esposto dal server (copialo dall'elenco sopra)"},
"use_lms_model": {"en": "✅ Use this LM Studio model", "it": "✅ Usa questo modello LM Studio"},
"insert_valid_id": {"en": "Insert a valid Model ID.", "it": "Inserisci un Model ID valido."},

"download_lms_hub": {"en": "⬇️ Download from LM Studio Hub (CLI)", "it": "⬇️ Scarica da LM Studio Hub (CLI)"},
"download_lms_help": {"en": "Use `lms get` CLI to search and download a model (e.g. `qwen2.5:7b`, `mistral:7b-instruct`, URL or alias supported).", "it": "Usa la CLI `lms get` per cercare e scaricare un modello (es. `qwen2.5:7b`, `mistral:7b-instruct`, URL o alias supportati)."},

"model_name_query": {"en": "Model name / query", "it": "Nome modello / query"},

"no_confirm": {"en": "--no-confirm", "it": "--no-confirm"},
"no_confirm_help": {"en": "Skip interactive prompts where possible.", "it": "Evita prompt interattivi, dove possibile."},

"force_install": {"en": "--force", "it": "--force"},
"force_install_help": {"en": "Force reinstall/update even if already present.", "it": "Forza reinstall/update anche se già presente."},

"download_with_lms": {"en": "⬇️ Download with `lms get`", "it": "⬇️ Scarica con `lms get`"},
"insert_valid_query": {"en": "Insert a valid model name or query.", "it": "Inserisci un nome modello o una query valida."},

"executing_cmd": {"en": "Executing: lms get {q} {args}", "it": "Esecuzione: lms get {q} {args}"},
"download_error": {"en": "❌ Download error: {e}", "it": "❌ Errore durante il download: {e}"},

"download_complete_lms": {"en": "✅ Download complete. If it's an LLM, you'll see `lms ls` updated or can expose it in the server.", "it": "✅ Download completato. Se è un LLM, vedrai `lms ls` aggiornato o potrai esporlo nel server."},
"lms_exit_code": {"en": "`lms get` finished with code {rc}.", "it": "`lms get` terminato con codice {rc}."},

"lms_hint": {"en": "💡 Tip: after installation, you can expose the model in LM Studio server and see it in the list above.", "it": "💡 Suggerimento: dopo l’installazione puoi esporre il modello nel server LM Studio e lo vedrai nella lista in alto."},

"ollama_registry_title": {"en": "🦙 Ollama – Online Registry", "it": "🦙 Ollama – Registry online"},
"ollama_host_help": {"en": "Usually http://localhost:11434", "it": "Di solito http://localhost:11434"},

"repo_filter": {"en": "Repository filter", "it": "Filtro repository"},

"registry_browse": {"en": "🌐 Online Registry (browse & pull)", "it": "🌐 Registry online (browse & pull)"},
"search_registry": {"en": "Search in registry", "it": "Cerca nel registry"},
"search_repo_contains": {"en": "Search repository (contains)", "it": "Cerca repository (contiene)"},
"refresh_registry": {"en": "Refresh registry list", "it": "Aggiorna elenco registry"},

"manual_pull_info": {
    "en": "You can still manually enter `model:tag` below to pull.",
    "it": "Puoi comunque inserire manualmente `modello:tag` più sotto per fare il pull."
},

"repos_in_registry": {"en": "📦 Repositories in registry (filtered): {n}", "it": "📦 Repository nel registry (filtrati): {n}"},
"select_repo": {"en": "Select repository", "it": "Seleziona repository"},

"tags_available": {"en": "🏷️ Tags available for **{repo}**: {n}", "it": "🏷️ Tag disponibili per **{repo}**: {n}"},

"pull_btn": {"en": "⬇️ Pull", "it": "⬇️ Pull"},
"pull_help": {"en": "Download and materialize this tag locally.", "it": "Scarica e materializza localmente questo tag."},
"pulling_model": {"en": "⬇️ Pulling `{name}`…", "it": "⬇️ Pull `{name}` in corso…"},
"pull_complete": {"en": "✅ Pull complete.", "it": "✅ Pull completato."},
"pull_error": {"en": "❌ Pull error: {e}", "it": "❌ Errore durante il pull: {e}"},

"manual_pull_header": {"en": "⬇️ Manual Pull (name:tag)", "it": "⬇️ Pull manuale (nome:tag)"},
"remote_model_name": {"en": "Remote model name", "it": "Nome modello remoto"},
"pull_from_registry": {"en": "⬇️ Pull from registry", "it": "⬇️ Pull da registry"},
"insert_valid_tag": {"en": "Insert a valid name (e.g. `model:tag`).", "it": "Inserisci un nome valido (es. `modello:tag`)."},
"pull_error_short": {"en": "❌ Pull error: {e}", "it": "❌ Errore pull: {e}"},

"set_active_ollama": {"en": "🎯 Set active model (Ollama)", "it": "🎯 Imposta modello attivo (Ollama)"},
"ollama_model_name": {"en": "Model name in Ollama", "it": "Nome modello in Ollama"},
"use_ollama_model": {"en": "✅ Use this Ollama model", "it": "✅ Usa questo modello Ollama"},
"insert_valid_ollama": {"en": "Insert a valid model name (repo:tag).", "it": "Inserisci un nome di modello valido (repo:tag)."},

"spacy_title": {"en": "🧠 spaCy", "it": "🧠 spaCy"},

"choose_source": {"en": "Choose model source", "it": "Scegli sorgente modello"},
"source_radio": {"en": "Start from a common model or enter a custom package?", "it": "Vuoi partire da un modello comune o inserire un pacchetto custom?"},

"common_models": {"en": "Common models", "it": "Modelli comuni"},
"custom_package": {"en": "Custom package", "it": "Pacchetto custom"},
"no_common_match": {"en": "❌ No common model matches the filter.", "it": "❌ Nessun modello comune corrisponde al filtro."},

"common_model_label": {"en": "Common model", "it": "Modello comune"},
"custom_package_input": {"en": "Custom package", "it": "Pacchetto custom"},

"custom_package_help": {
    "en": "Package name (e.g. it_core_news_sm, en_core_web_sm)",
    "it": "Nome del pacchetto (es. it_core_news_sm, en_core_web_sm)"
},

"select_or_insert": {"en": "Select or insert a model.", "it": "Seleziona o inserisci un modello."},

"details_no_download": {"en": "📘 Model details (without downloading)", "it": "📘 Dettagli modello (senza scaricare)"},

"download_update": {"en": "Download / Update", "it": "Scarica / Aggiorna"},
"show_details_after": {"en": "Show full details after download", "it": "Mostra dettagli completi dopo il download"},

"downloading_candidate": {"en": "⬇️ Downloading `{candidate}`…", "it": "⬇️ Download di `{candidate}` in corso…"},
"model_installed": {"en": "✅ Model `{candidate}` installed/updated.", "it": "✅ Modello `{candidate}` installato/aggiornato."},
"failed_exit_code": {"en": "❌ Failed with exit code {code}.", "it": "❌ Fallito con exit code {code}."},

"full_details_installed": {"en": "📘 Full details (installed)", "it": "📘 Dettagli completi (installato)"},

"vectors_dim": {"en": "Vectors (dim)", "it": "Vettori (dim)"},
"vectors_keys": {"en": "Vectors (keys)", "it": "Vettori (keys)"},
"description": {"en": "Description", "it": "Descrizione"},
"cant_load_details": {"en": "❌ Cannot load full details: {e}", "it": "❌ Non riesco a caricare i dettagli completi: {e}"},
"test_ner_pos": {"en": "🧪 Test NER/POS", "it": "🧪 Prova NER/POS"},

"test_text": {"en": "📝 Test text", "it": "📝 Testo di prova"},
"test_text_default": {"en": "Hello, I am in Milan for EDBT.", "it": "Ciao, sono a Milano per EDBT."},

"run_analysis": {"en": "▶️ Run analysis", "it": "▶️ Esegui analisi"},
"error_loading_model": {"en": "❌ Error loading model: {e}", "it": "❌ Errore durante il caricamento del modello: {e}"},

"ner_highlighted": {"en": "🔍 NER highlighted", "it": "🔍 NER evidenziato"},
"pos_table": {"en": "🧩 POS (table)", "it": "🧩 POS (tabella)"},
"json_raw": {"en": "🪶 JSON raw", "it": "🪶 JSON raw"},

"legend": {"en": "📘 Legend", "it": "📘 Legenda"},
"no_entities": {"en": "ℹ️ No entities recognized.", "it": "ℹ️ Nessuna entità riconosciuta."},

"use_analysis_model": {"en": "✅ Use this model for Analysis", "it": "✅ Usa questo modello per le Analisi"},
"analysis_model_chosen": {"en": "📌 Analysis Model Chosen {candidate}", "it": "📌 Modello per Analisi Scelto {candidate}"},

"person": {"en": "Person (proper name)", "it": "Persona (nome proprio)"},
"norp": {"en": "Groups (nat., relig., pol.)", "it": "Gruppi (naz., relig., politici)"},
"fac": {"en": "Facility (building, bridge, etc.)", "it": "Struttura fisica (edificio, ponte, ecc.)"},
"org": {"en": "Organization (company, agency, team)", "it": "Organizzazione (azienda, ente, squadra)"},
"gpe": {"en": "Country/Region/City", "it": "Paese/Regione/Città"},
"loc": {"en": "Location (mountains, rivers…)", "it": "Località non politica (montagne, fiumi…)"},
"product": {"en": "Product/Object", "it": "Prodotto/oggetto"},
"event": {"en": "Event", "it": "Evento nominato"},
"work_of_art": {"en": "Work of Art (book, film…)", "it": "Opera (libro, film, quadro…)"},
"law": {"en": "Law", "it": "Legge o atto normativo"},
"language_ent": {"en": "Language", "it": "Lingua"},
"date": {"en": "Date/Period", "it": "Data/periodo"},
"time": {"en": "Time", "it": "Orario"},
"percent": {"en": "Percentage", "it": "Percentuale"},
"money": {"en": "Money", "it": "Valuta/importo"},
"quantity": {"en": "Quantity", "it": "Quantità/misura"},
"ordinal": {"en": "Ordinal", "it": "Ordinale (1º, 2º…)"},
"cardinal": {"en": "Cardinal", "it": "Cardinale"},
"spacy_entity": {"en": "spaCy Entity", "it": "Entità spaCy"},
"sensitive_tag": {"en": "🔒 sensitive ({n}/{tot})", "it": "🔒 sensibile ({n}/{tot})"},
"not_sensitive_tag": {"en": "🟢 not sensitive ({n})", "it": "🟢 non sensibile ({n})"},

"searching_ollama": {"en": "🔎 Searching in Ollama registry…", "it": "🔎 Cerco nel registry Ollama…"},
"ollama_registry_error": {"en": "❌ Unable to query Ollama registry: {e}", "it": "❌ Impossibile interrogare il registry Ollama: {e}"},

"fetching_tags": {"en": "📦 Fetching tags…", "it": "📦 Recupero tag…"},
"tags_error": {"en": "❌ Unable to fetch tags for '{model}': {e}", "it": "❌ Impossibile recuperare i tag per '{model}': {e}"}
    },
"imputation": {
    "header": {"en": "🧩 Missing Value Imputation", "it": "🧩 Imputazione Valori Mancanti"},

    "select_cols": {"en": "📌 Select columns to impute", "it": "📌 Seleziona colonne da imputare"},
    "method": {"en": "🔧 Imputation Method", "it": "🔧 Metodo di Imputazione"},

    "simple": {"en": "Simple Imputation (Univariate)", "it": "Imputazione Semplice (Univariata)"},
    "knn": {"en": "KNN Imputation (Multivariate)", "it": "Imputazione KNN (Multivariata)"},
    "iterative": {"en": "Iterative Imputation (MICE)", "it": "Imputazione Iterativa (MICE)"},

    "strategy": {"en": "Strategy", "it": "Strategia"},
    "mean": {"en": "Mean", "it": "Media"},
    "median": {"en": "Median", "it": "Mediana"},
    "most_frequent": {"en": "Most Frequent (Mode)", "it": "Più Frequente (Moda)"},
    "constant": {"en": "Constant Value", "it": "Valore Costante"},

    "fill_value": {"en": "Fill Value", "it": "Valore di Riempimento"},
    "n_neighbors": {"en": "Number of Neighbors (k)", "it": "Numero di Vicini (k)"},
    "max_iter": {"en": "Max Iterations", "it": "Max Iterazioni"},

    "rows_limit": {"en": "Limit rows for calculation (0 = all)", "it": "Limita righe per calcolo (0 = tutte)"},
    "rows_limit_help": {
        "en": "⚠️ Complex algorithms (KNN, MICE) can be slow on large datasets. Limit rows to speed up.",
        "it": "⚠️ Algoritmi complessi (KNN, MICE) possono essere lenti su grandi dataset. Limita le righe per velocizzare."
    },

    "apply_btn": {"en": "▶️ Apply Imputation", "it": "▶️ Applica Imputazione"},
    "success": {"en": "✅ Imputation applied successfully!", "it": "✅ Imputazione applicata con successo!"},
    "error": {"en": "❌ Error during imputation: {e}", "it": "❌ Errore durante l'imputazione: {e}"},

    "no_cols": {"en": "⚠️ Please select at least one column.", "it": "⚠️ Seleziona almeno una colonna."},

    "preview": {"en": "📊 Preview Imputed Data", "it": "📊 Anteprima Dati Imputati"},
    "save_update": {"en": "💾 Update Dataset in Memory", "it": "💾 Aggiorna Dataset in Memoria"},
    "updated": {"en": "Dataset updated in session state.", "it": "Dataset aggiornato nello stato della sessione."},

    "cols_with_missing": {"en": "Columns with missing values:", "it": "Colonne con valori mancanti:"},
    "imputation_type": {"en": "Imputation Type", "it": "Tipo di Imputazione"},

    "numeric_standard": {"en": "🔢 Numeric (Standard Algorithms)", "it": "🔢 Numerico (Algoritmi Standard)"},
    "text_llm": {"en": "🧠 Text (LLM - Generative)", "it": "🧠 Testo (LLM - Generativo)"},

    "select_backend": {"en": "Select Backend", "it": "Seleziona Backend"},
    "select_model": {"en": "Select Model", "it": "Seleziona Modello"},

    "start_imputation": {"en": "▶️ Start Imputation", "it": "▶️ Avvia Imputazione"},
    "imputing_progress": {"en": "⏳ Imputing... ({current}/{total})", "it": "⏳ Imputazione in corso... ({current}/{total})"},

    "review_header": {"en": "📝 Review Imputed Values", "it": "📝 Revisiona Valori Imputati"},
    "original": {"en": "Original", "it": "Originale"},
    "imputed": {"en": "Imputed", "it": "Imputato"},

    "accept": {"en": "✔️ Accept", "it": "✔️ Accetta"},
    "reject": {"en": "❌ Reject", "it": "❌ Rifiuta"},
    "accept_all": {"en": "✔️ Accept All", "it": "✔️ Accetta Tutto"},
    "reject_all": {"en": "❌ Reject All", "it": "❌ Rifiuta Tutto"},

    "no_model_selected": {"en": "⚠️ Please select a model.", "it": "⚠️ Seleziona un modello."},
    "no_text_cols": {
        "en": "No text/object columns with missing values found.",
        "it": "Nessuna colonna di testo/oggetto con valori mancanti trovata."
    },

    "imputation_complete": {"en": "🎉 Imputation review complete.", "it": "🎉 Revisione imputazione completata."},

    "confirm_apply": {"en": "💾 Apply Accepted Changes to Dataset", "it": "💾 Applica Modifiche Accettate al Dataset"}
},
"welcome": {
    "title": {"en": "🌳 Welcome to CALIBER", "it": "🌳 Benvenuto in CALIBER"},
    "subtitle": {"en": "### 🌍 Green AI-Powered SQL Query Generation & Optimization", "it": "### 🌍 Green AI-Powered SQL Query Generation & Optimization"},
    "intro": {
        "en": "**CALIBER** (Carbon-Aware LLM-Integrated Benchmarking and Eco-friendly Rewriting) is a comprehensive tool for SQL query generation via artificial intelligence, with real-time CO₂ emissions monitoring and green query optimization.",
        "it": "**CALIBER** (Carbon-Aware LLM-Integrated Benchmarking and Eco-friendly Rewriting) è uno strumento completo per la generazione di query SQL tramite intelligenza artificiale, con monitoraggio in tempo reale delle emissioni di CO₂ e ottimizzazione green delle query."
    },
    "features_header": {"en": "### 📋 Features Overview", "it": "### 📋 Panoramica delle Funzionalità"},
    
    "data_hub_title": {"en": "#### 📊 Data Hub", "it": "#### 📊 Data Hub"},
    "data_hub_desc": {
        "en": "Load data from local files (CSV, Parquet, HDF5) or connect to existing databases (MySQL, PostgreSQL, SQLite, DuckDB). Configure LLM models for query generation.",
        "it": "Carica dati da file locali (CSV, Parquet, HDF5) o connettiti a database esistenti (MySQL, PostgreSQL, SQLite, DuckDB). Configura i modelli LLM per la generazione di query."
    },
    
    "data_insights_title": {"en": "#### 📈 Data Insights", "it": "#### 📈 Data Insights"},
    "data_insights_desc": {
        "en": "In-depth dataset analysis with detailed statistics, sensitive data (PII) detection, relational profiling, primary key identification, and interactive visualizations.",
        "it": "Analisi approfondita dei dataset con statistiche dettagliate, rilevamento di dati sensibili (PII), profilazione relazionale, identificazione di chiavi primarie e visualizzazioni interattive."
    },
    
    "green_query_title": {"en": "#### 🧪 Green Query Builder", "it": "#### 🧪 Green Query Builder"},
    "green_query_desc": {
        "en": "Generate SQL queries from natural language using LLMs. Includes real-time CO₂ monitoring, alternative query execution, and the **Greenify** function to optimize queries by reducing carbon emissions.",
        "it": "Genera query SQL da linguaggio naturale utilizzando LLM. Include monitoraggio CO₂ in tempo reale, esecuzione di query alternative e la funzione **Greenify** per ottimizzare le query riducendo le emissioni di carbonio."
    },
    
    "eco_benchmark_title": {"en": "#### 🎯 Eco Benchmark", "it": "#### 🎯 Eco Benchmark"},
    "eco_benchmark_desc": {
        "en": "Compare different LLMs based on energy efficiency, accuracy, and carbon footprint. Includes NL→SQL benchmarks, DBMS execution, quality evaluation, and Green AI Race.",
        "it": "Confronta diversi LLM in base a efficienza energetica, accuratezza e impronta di carbonio. Include benchmark NL→SQL, esecuzione DBMS, valutazione qualità e Green AI Race."
    },
    
    "synthetic_lab_title": {"en": "#### 🧬 Synthetic Lab", "it": "#### 🧬 Synthetic Lab"},
    "synthetic_lab_desc": {
        "en": "Generate synthetic datasets with various strategies (Faker, GaussianCopula, CTGAN, TVAE) while monitoring the environmental impact of each approach.",
        "it": "Genera dataset sintetici con diverse strategie (Faker, GaussianCopula, CTGAN, TVAE) monitorando l'impatto ambientale di ogni approccio."
    },
    
    "settings_title": {"en": "#### ⚙️ Settings", "it": "#### ⚙️ Settings"},
    "settings_desc": {
        "en": "Customize key parameters such as CO₂ emission factor and CPU TDP for more accurate estimates based on your hardware and geographic location.",
        "it": "Personalizza i parametri chiave come il fattore di emissione CO₂ e il TDP della CPU per stime più accurate basate sul tuo hardware e posizione geografica."
    },
    
    "key_features_header": {"en": "### 🌱 Key Features", "it": "### 🌱 Caratteristiche Principali"},
    "feature_monitoring": {
        "en": "**Real-time CO₂ Monitoring**: Complete tracking of energy consumption and emissions",
        "it": "**Monitoraggio CO₂ in tempo reale**: Tracciamento completo del consumo energetico e delle emissioni"
    },
    "feature_greenify": {
        "en": "**Green Optimization (Greenify)**: Automatic query rewriting to reduce environmental impact",
        "it": "**Ottimizzazione Green (Greenify)**: Riscrittura automatica delle query per ridurre l'impatto ambientale"
    },
    "feature_multi_llm": {
        "en": "**Multi-LLM Support**: Integration with Hugging Face, Ollama, LM Studio, and custom models",
        "it": "**Multi-LLM Support**: Integrazione con Hugging Face, Ollama, LM Studio e modelli personalizzati"
    },
    "feature_benchmarking": {
        "en": "**Comprehensive Benchmarking**: Systematic comparison of LLMs and DBMS based on sustainability metrics",
        "it": "**Benchmarking Completo**: Confronto sistematico di LLM e DBMS basato su metriche di sostenibilità"
    },
    "feature_synthetic": {
        "en": "**Synthetic Data Generation**: Dataset creation with energy impact tracking",
        "it": "**Generazione Dati Sintetici**: Creazione di dataset con tracciamento dell'impatto energetico"
    },
    "feature_analytics": {
        "en": "**Advanced Analytics**: Relational profiling, PII detection, and integrity analysis",
        "it": "**Analisi Avanzata**: Profilazione relazionale, rilevamento PII e analisi di integrità"
    },
    
    "getting_started_header": {"en": "### 🚀 Getting Started", "it": "### 🚀 Per Iniziare"},
    "step1": {"en": "**Load your data** in the *Data Hub* tab", "it": "**Carica i tuoi dati** nel tab *Data Hub*"},
    "step2": {"en": "**Configure an LLM model** (local or online)", "it": "**Configura un modello LLM** (locale o online)"},
    "step3": {"en": "**Generate queries** from natural language in the *Green Query Builder* tab", "it": "**Genera query** dal linguaggio naturale nel tab *Green Query Builder*"},
    "step4": {"en": "**Optimize** your queries with the Greenify function", "it": "**Ottimizza** le tue query con la funzione Greenify"},
    "step5": {"en": "**Compare** different LLMs in the *Eco Benchmark* tab", "it": "**Confronta** diversi LLM nel tab *Eco Benchmark*"},
    
    "dont_show_again": {"en": "✅ Don't show this message again", "it": "✅ Non mostrare più questo messaggio"},
    "dont_show_help": {
        "en": "Check to hide this message on future launches",
        "it": "Seleziona per non visualizzare più questo messaggio all'avvio"
    },
    "start_button": {"en": "🚀 Get Started", "it": "🚀 Inizia"}
},
}


def get_text(section, key, **kwargs):
    lang = st.session_state.get("language", "English")
    # Map full language names to codes
    lang_code = "en" if lang == "English" else "it"
    
    try:
        text = TRANSLATIONS[section][key][lang_code]
        if kwargs:
            return text.format(**kwargs)
        return text
    except KeyError:
        return f"MISSING: {section}.{key}"
