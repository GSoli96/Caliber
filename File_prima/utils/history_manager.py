# utils/history_manager.py

import sqlite3
import json
import os
from datetime import datetime

# Definisci il percorso del database.
# Puoi cambiarlo se preferisci (es. in una cartella 'data')
DB_PATH = "history/query_history.db"


def get_db_connection():
    """Restituisce una connessione al database SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Permette di accedere ai risultati come dizionari
        return conn
    except sqlite3.Error as e:
        print(f"Errore durante la connessione al DB della cronologia: {e}")
        return None


def initialize_history_db():
    """
    Crea la tabella 'query_history' se non esiste.
    Da chiamare una volta all'avvio dell'app.
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn:
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS query_history
                         (
                             id
                             INTEGER
                             PRIMARY
                             KEY
                             AUTOINCREMENT,
                             run_timestamp
                             TEXT
                             NOT
                             NULL,
                             user_question
                             TEXT,
                             llm_backend
                             TEXT,
                             llm_model
                             TEXT,
                             db_engine_type
                             TEXT,
                             tables_queried
                             TEXT,
                             status
                             TEXT
                             NOT
                             NULL,
                             total_duration_s
                             REAL,
                             original_query
                             TEXT,
                             original_query_duration_s
                             REAL,
                             original_query_result
                             TEXT,
                             alternatives
                             TEXT,
                             monitoring_data
                             TEXT,
                             error_message
                             TEXT
                         );
                         """)
    except sqlite3.Error as e:
        print(f"Errore durante l'inizializzazione del DB della cronologia: {e}")
    finally:
        if conn:
            conn.close()


def add_history_entry(entry_data: dict):
    """
    Aggiunge una nuova riga alla tabella 'query_history'.
    'entry_data' è un dizionario con chiavi che corrispondono alle colonne della tabella.
    """
    conn = get_db_connection()
    if conn is None:
        print("Impossibile aggiungere l'entry alla cronologia: connessione non riuscita.")
        return

    # Pulisce i dati per l'inserimento
    columns = [
        'run_timestamp', 'user_question', 'llm_backend', 'llm_model',
        'db_engine_type', 'tables_queried', 'status', 'total_duration_s',
        'original_query', 'original_query_duration_s', 'original_query_result',
        'alternatives', 'monitoring_data', 'error_message'
    ]

    # Assicura che solo le chiavi attese siano presenti e abbiano un valore
    data_to_insert = {col: entry_data.get(col) for col in columns}

    try:
        with conn:
            conn.execute(f"""
            INSERT INTO query_history (
              {', '.join(columns)}
            ) VALUES (
              {', '.join([f':{col}' for col in columns])}
            );
            """, data_to_insert)
    except sqlite3.Error as e:
        print(f"Errore durante il salvataggio dell'entry nella cronologia: {e}")
    finally:
        if conn:
            conn.close()


def get_all_history_entries() -> list:
    """
    Recupera tutte le entry dalla cronologia, ordinate dalla più recente.
    Restituisce una lista di dizionari.
    """
    conn = get_db_connection()
    if conn is None:
        return []

    try:
        entries = conn.execute('SELECT * FROM query_history ORDER BY run_timestamp DESC').fetchall()
        # Converte gli oggetti sqlite3.Row in dizionari standard
        return [dict(row) for row in entries]
    except sqlite3.Error as e:
        print(f"Errore durante il recupero della cronologia: {e}")
        return []
    finally:
        if conn:
            conn.close()