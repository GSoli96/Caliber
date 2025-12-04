import os

import json5


def load_config(config_path="setup/config_parameters.jsonc"):
    """
    Loads the JSON configuration file.
    
    Args:
        config_path (str): Path to the configuration file. Defaults to "setup/config_parameters.jsonc".
    
    Returns:
        dict: Parsed configuration dictionary.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return json5.load(f)


config = load_config()


def get_color_from_dtype(dtype):
    """
    Returns a hexadecimal color based on the Pandas data type.
    
    Args:
        dtype: Pandas data type object.
    
    Returns:
        str: Hexadecimal color code corresponding to the data type.
    """
    dtype_str = str(dtype).lower()
    color_map = config.get("dtype_colors", {})
    # Search for a key contained in dtype_str
    for key, color in color_map.items():
        if key in dtype_str:
            return color
    return color_map.get("default", "#4A4A4A")


def get_connectionMySQL():
    """
    Retrieves MySQL connection configuration.
    
    Returns:
        dict: MySQL connection parameters from configuration.
    """
    MYSQL = config.get("dtype_colors", {})
    return MYSQL


def get_HF_Token():
    """
    Retrieves the Hugging Face API token from configuration.
    
    Returns:
        str: Hugging Face authentication token.
    """
    token = config.get("HF_Token", {})
    return token['token']


def get_num_alternative_queries():
    """
    Returns the number of alternative queries to generate.
    
    Returns:
        int: Number of alternative queries configured.
    """
    num_queries = config.get("num_alternative_queries", 0)
    print(f"[DEBUG-CONFIG] Number of alternative queries to generate: {num_queries}")
    return num_queries


def get_db_adapter(available_adapters, db_name=None):
    """
    Returns the Python adapter associated with a database based on JSON configuration.
    
    Args:
        available_adapters (dict): Dictionary containing references to imported adapters.
        db_name (str, optional): Name of the database. If None, uses default adapter.
    
    Returns:
        object: The database adapter object.
    
    Raises:
        ValueError: If no adapter is configured for the database or adapter is not found.
    """
    if db_name is None:
        adapter_key = config.get("db_adapters", None)
    else:
        adapter_key = config.get("db_adapters", {}).get(db_name)
        if not adapter_key:
            raise ValueError(f"No adapter configured for database: {db_name}")

    if adapter_key not in available_adapters:
        raise ValueError(f"Adapter '{adapter_key}' not found in Python code.")
    return available_adapters[adapter_key]


if __name__ == "__main__":
    config = load_config("config.json")
    print(config)