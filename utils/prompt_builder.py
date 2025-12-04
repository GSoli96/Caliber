import pandas as pd
from typing import Dict


def create_sql_prompt(
        dfs: Dict[str, pd.DataFrame],
        user_question: str,
        db_name: str,
        db_connection_args: dict | None = None
) -> str:
    """
    Creates a clear, direct, and robust prompt for SQL query generation
    for a database with one or more tables.
    
    Args:
        dfs (Dict[str, pd.DataFrame]): Dictionary mapping table names to DataFrames.
        user_question (str): Natural language question from the user.
        db_name (str): Name/type of the database (e.g., "SQLite", "PostgreSQL").
        db_connection_args (dict, optional): Database connection arguments.
    
    Returns:
        str: Formatted prompt for LLM to generate SQL query.
    """
    # Fallback for when no DataFrame is loaded
    if not dfs:
        return f"""
                ### TASK ###
                You are an expert SQL assistant for a {db_name} database.
                Write a valid SQL query based on the user's request.
                Return ONLY the raw SQL query. Do not add any explanation or markdown formatting.
            
                ### USER REQUEST ###
                "{user_question}"
            
                ### SQL QUERY ###
                """

    # --- Build database info section ---
    db_info_lines = [f"- SQL Dialect: {db_name}"]
    if db_name == "SQLite" and db_connection_args and db_connection_args.get('db_path'):
        db_path = db_connection_args['db_path']
        if db_path != ':memory:':
            db_info_lines.append(f"- Database File: `{db_path}`")
    db_info_str = "\n".join(db_info_lines)

    # --- Build schema info section for EACH TABLE ---
    schema_info_parts = []
    print('Dfs', dfs)
    for table_name, df in dfs.items():
        if df is None:
            continue

        schema_parts = []
        for col, dtype in df.dtypes.items():
            # Clean column names for safe SQL usage
            clean_col = f'"{col}"' if ' ' in col or ':' in col else col
            schema_parts.append(f"  - {clean_col} (type: {dtype})")

        schema_str = "\n".join(schema_parts)
        row_count = len(df)
        col_count = len(df.columns)
        table_info = f"""
                    - Table Name: `{table_name}`
                    - Rows: {row_count}
                    - Columns: {col_count}
                    - Table Schema:
                    {schema_str}
                    """
        schema_info_parts.append(table_info)

    full_schema_str = "\n".join(schema_info_parts)

    # Assemble the final prompt
    prompt_template = f"""
    ### TASK ###
    You are an expert AI assistant that writes a single, valid SQL query for a {db_name} database.
    Your job is to translate the user's request into a SQL query that may involve multiple tables.
    You MUST return ONLY the SQL query itself. Do not write any other text, comments, or markdown.

    ### DATABASE INFO ###
    {db_info_str}

    ### DATABASE SCHEMA ###
    The database contains the following tables:
    {full_schema_str}

    ### USER REQUEST ###
    "{user_question}"

    ### SQL QUERY ###
    """
    return prompt_template.strip()


def create_sql_optimization_prompt(
        dfs: Dict[str, pd.DataFrame],
        user_question: str,
        db_name: str,
        original_query: str,
        original_query_co2: float = None,
        db_metadata: dict = None
) -> str:
    """
    Create a prompt for generating optimized, CO2-efficient SQL query alternatives.
    
    Generates a prompt that instructs the LLM to create 3-5 alternative queries
    that are more energy-efficient than the original while maintaining correctness.
    
    Args:
        dfs: Dictionary mapping table names to pandas DataFrames
        user_question: Original natural language question from user
        db_name: Database type (SQLite, PostgreSQL, MySQL, etc.)
        original_query: The original SQL query to optimize
        original_query_co2: Optional CO2 consumption of original query in grams
        db_metadata: Optional additional database metadata
    
    Returns:
        str: Formatted prompt for LLM to generate optimized queries
    
    Note:
        The prompt includes green optimization strategies like column selection,
        efficient joins, result limiting, and index-friendly operations.
    """
    db_info_str = f"- SQL Dialect: {db_name}"

    schema_info_parts = []
    for table_name, df in dfs.items():
        if df is None:
            continue

        # --- CORRECTED LINE ---
        # Replaced complex list comprehension with a clearer and safer approach.
        schema_parts = []
        for c, t in df.dtypes.items():
            clean_col = f'"{c}"' if ' ' in c else c
            schema_parts.append(f"  - {clean_col} (type: {t})")
        # --- END CORRECTION ---

        schema_str = "\n".join(schema_parts)
        row_count = len(df)
        col_count = len(df.columns)
        table_info = f"""- Table: `{table_name}`
    - Rows: {row_count}
    - Columns: {col_count}
    - Schema:
    {schema_str}"""
        schema_info_parts.append(table_info)

    full_schema_str = "\n".join(schema_info_parts)

    co2_info = ""
    if original_query_co2 is not None:
        co2_info = f"\nThe original query consumed approximately {original_query_co2:.4f} grams of CO2."

    prompt_template = f"""
    ### TASK ###
        You are an expert AI assistant that writes CO2-consumption-optimized SQL queries for a {db_name} database.
    Your task is to provide 3 to 5 DIFFERENT and more efficient alternatives to the original query that still correctly answers the user's request.

        ### GREEN OPTIMIZATION STRATEGIES ###
        Apply these techniques to reduce energy consumption and CO2 emissions:
        1. **Column Selection**: Replace SELECT * with specific columns needed
        2. **Efficient Joins**: Use INNER JOIN instead of subqueries where possible
        3. **Result Limiting**: Add LIMIT clause if appropriate for the user's request
        4. **Index-Friendly**: Write queries that can use indexes (avoid functions on indexed columns)
        5. **Reduce Data Movement**: Minimize rows processed early in the query (WHERE before JOIN)
        6. **Avoid Redundancy**: Remove unnecessary DISTINCT, GROUP BY, or ORDER BY operations

        ### OUTPUT FORMAT ###
        You MUST return ONLY the alternative SQL queries, separated by a semicolon ';'.
        Do not include explanations, comments, or markdown formatting.{co2_info}

    ### DATABASE SCHEMA ###
    {full_schema_str}

    ### USER REQUEST ###
    "{user_question}"

    ### ORIGINAL QUERY (DO NOT REPEAT) ###
    ```sql
    {original_query}
    ```

    ### ALTERNATIVE AND OPTIMIZED SQL QUERIES ###
    """
    return prompt_template.strip()