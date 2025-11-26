import os

import json5


def load_config(config_path="setup/config_parameters.jsonc"):
    """Carica il file di configurazione JSON."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return json5.load(f)


config = load_config()


def get_color_from_dtype(dtype):
    """Restituisce un colore esadecimale basato sul tipo di dato di Pandas."""
    dtype_str = str(dtype).lower()
    color_map = config.get("dtype_colors", {})
    # Cerca una chiave che sia contenuta nel dtype_str
    for key, color in color_map.items():
        if key in dtype_str:
            return color
    return color_map.get("default", "#4A4A4A")


def get_connectionMySQL():
    MYSQL = config.get("dtype_colors", {})
    return MYSQL


def get_HF_Token():
    token = config.get("HF_Token", {})
    return token['token']


def get_num_alternative_queries():
    """Restituisce il numero di query alternative da generare."""
    # Aggiungo una riga di debug per mostrare il valore caricato
    num_queries = config.get("num_alternative_queries", 0)
    print(f"[DEBUG-CONFIG] Numero di query alternative da generare: {num_queries}")
    return num_queries


def get_db_adapter(available_adapters, db_name=None):
    """
    Restituisce l'adapter Python associato a un database, in base alla configurazione JSON.
    `available_adapters` deve essere un dizionario che contiene i riferimenti effettivi
    agli adapter importati nel codice.
    """
    if db_name is None:
        adapter_key = config.get("db_adapters", None)
    else:
        adapter_key = config.get("db_adapters", {}).get(db_name)
        if not adapter_key:
            raise ValueError(f"Nessun adapter configurato per il database: {db_name}")

    if adapter_key not in available_adapters:
        raise ValueError(f"Adapter '{adapter_key}' non trovato nel codice Python.")
    return available_adapters[adapter_key]


if __name__ == "__main__":
    config = load_config("config.json")
    print(config)