# db_adapters/__init__.py
import pandas as pd
from sqlalchemy import create_engine, text

DB_ADAPTERS = [
    "MySQL",
    "SQLite",
    "PostgreSQL",
    "DuckDB",
    "SQL Server",
]

def initialize_database(db_choice: str, connection_args: dict, msg):
    """
    Initializes a database engine for temporary query execution.
    For simplicity and speed, this will always create an in-memory SQLite engine.

    Args:
        db_choice (str): The type of database selected (e.g., "SQLite", "MySQL").
        connection_args (dict): A dictionary containing connection arguments.

    Returns:
        dict: A dictionary containing the initialized engine key "engine" or an error message key "error".
    """
    print(msg)
    try:
        engine = create_engine("sqlite:///:memory:")
        return {"engine": engine}
    except Exception as e:
        return {"error": str(e)}


def execute_query(db_choice: str, conn, query: str):
    """
    Executes a SQL query on the given connectable (engine or connection)
    and returns the result.

    Args:
        db_choice (str): The type of database selected.
        conn: The database connection or engine object.
        query (str): The SQL query string to execute.

    Returns:
        dict: A dictionary containing the result DataFrame and row count, or an error message.
    """
    # === DEBUGGING E CONTROLLO DI ROBUSTEZZA ===
    print(f"\n[DEBUG-DB_ADAPTER] execute_query chiamata con:\n  - Query: '{query}'\n  - Tipo Query: {type(query)}\n")
    if not isinstance(query, str) or not query.strip():
        return {"error": "La query ricevuta è vuota o non è una stringa."}

    try:
        result_df = pd.read_sql_query(sql=text(query), con=conn)
        return {
            "dataframe": result_df,
            "rows": len(result_df)
        }
    except Exception as e:
        return {"error": str(e)}