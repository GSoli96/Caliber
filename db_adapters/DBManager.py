import os
import re
import subprocess
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import URL

from utils.load_config import load_config

# Try to use pymysql as fallback for MySQL if MySQLdb is not available
try:
    import pymysql
    pymysql.install_as_MySQLdb()
except ImportError:
    pass

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError:
    mysql = None

try:
    import pyodbc
except ImportError:
    pyodbc = None

try:
    import psycopg2
    from psycopg2 import OperationalError as PgError
except ImportError:
    psycopg2 = None

# DB_DIR is now managed via session_state
DB_DIR = st.session_state.get("db_dir", "database")
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)


class DBManager:
    """
    Database Manager class for handling multiple DBMS types (SQLite, DuckDB, MySQL, PostgreSQL, SQL Server).
    
    This class provides a unified interface for creating, loading, and managing databases across different
    database management systems. It handles both file-based databases (SQLite, DuckDB) and server-based
    databases (MySQL, PostgreSQL, SQL Server).
    
    Attributes:
        RESERVED_KEYWORDS (set): SQL reserved keywords that cannot be used as database names.
        SERVICE_MAP (dict): Mapping of DBMS types to their Windows service names.
    """
    RESERVED_KEYWORDS = {"select", "from", "where", "union", "join", "group", "order", "limit"}

    # Mappatura dei servizi Windows per i DBMS server-based
    SERVICE_MAP = {
        "PostgreSQL": "postgresql-x64-18",
        "MySQL": "MySQL80",
        "SQL Server": "MSSQLSERVER",
    }

    def __init__(self, dict_stato: dict, type: str):
        """
        Initializes the DBManager instance.
        
        Args:
            dict_stato (dict): Session state dictionary containing configuration and data.
            type (str): Operation type - "create_db" for creating databases, "download" for loading existing databases.
        """
        self.ss = dict_stato
        self.type = type
        print("[DBManager] Type: ", self.type)

        config_dict = self.ss.get("config_dict", {})
        print("[DBManager] Config dict: ", config_dict)
        self.dfs_dict = config_dict.get("dfs_dict", {})
        print("[DBManager] DFS dict: ", self.dfs_dict)

        # Config globale (parametri server, ecc.)
        self.global_config = load_config()
        print("[DBManager] Global config: ", self.global_config)

        if not config_dict:
            # Nessuna configurazione -> istanza "vuota"
            print("[DBManager] Nessuna configurazione -> istanza \"vuota\"")
            return

        if not self.dfs_dict and self.type != "download":
            # Per create_db servono i DataFrame; per download_db no
            print("[DBManager] Nessun DataFrame -> istanza \"vuota\"")
            return

        # --- Lettura parametri dalla GUI ---

        # Connection string (nuova chiave 'conn_str' con fallback a 'connection_string')
        self.connection_string: str = (
            config_dict.get("connection_string")
            or config_dict.get("conn_str")
            or ""
        )
        print("[DBManager] Connection string: ", self.connection_string)

        # Nome database
        self.db_name: str = self._safe_db_name((config_dict.get("db_name", "").strip()).replace(" ", "_"))
        print("[DBManager] DB Name: ", self.db_name)
        # Lista tabelle / colonne
        tblist = config_dict.get("table_list", [])
        if tblist:
            self.table_list = [table.strip().replace(" ", "_") for table in tblist]
        else:
            self.table_list = "Load all tables"
        print("[DBManager] Table list: ", self.table_list)

        self.load_all_tables: bool = bool(config_dict.get("load_all_tables", False))
        print("[DBManager] Load all tables: ", self.load_all_tables)

        # Scelta DBMS
        self.choice_DBMS: str = (
            config_dict.get("choice_DBMS")
            or config_dict.get("db_choice")
            or ""
        )
        print("[DBManager] DBMS: ", self.choice_DBMS)

        # Percorso file per SQLite / DuckDB
        self.sqlite_db_path: Optional[str] = None
        if self.choice_DBMS in ("SQLite", "DuckDB"):
            # 'db_path' (creazione) o 'path_to_file' (download)
            self.sqlite_db_path = config_dict.get("db_path") or config_dict.get("path_to_file")
            # Se arriva solo il path, deriva il nome db dal file (come fa la GUI)
            if self.sqlite_db_path and not config_dict.get("db_name"):
                self.db_name = os.path.splitext(os.path.basename(self.sqlite_db_path))[0]
        print("[DBManager] SQLite DB Path: ", self.sqlite_db_path)

        # Carica i DataFrame solo se non siamo in modalità download
        if self.type != "download":
            self.db_to_load = self.load_df()
            print("[DBManager] DataFrame loaded: ", self.db_to_load)
        else:
            self.db_to_load = None
            print("[DBManager] DataFrame not loaded")

        self.psw_conn = config_dict.get("psw_conn")
        print("[DBManager] Password connection: ", self.psw_conn)

    # ---------------------- Utility interne ----------------------
    def _normalize_db_path(self):
        """
        Normalizza il percorso del file DB per SQLite e DuckDB.
        Garantisce:
            - che l'estensione sia corretta (.db o .duckdb)
            - che il path sia assoluto
            - che DB_DIR venga sempre applicato correttamente
        """
        filename = self.sqlite_db_path or self.db_name

        # Aggiungi estensione se manca
        if not filename.lower().endswith(".db") and not filename.lower().endswith(".duckdb"):
            ext = ".db" if self.choice_DBMS == "SQLite" else ".duckdb"
            filename += ext

        # Se l’utente ha passato un percorso assoluto, usa quello
        if os.path.isabs(filename):
            return filename

        # Altrimenti applica DB_DIR
        return os.path.abspath(os.path.join(DB_DIR, filename))

    def _check_service_status(self, dbms_type: str) -> bool:
        """
        Checks if the DBMS service is running (only for server-based DBMS on Windows).
        
        For SQLite/DuckDB or unmapped DBMS types, always returns True.
        
        Args:
            dbms_type (str): The type of DBMS (e.g., "MySQL", "PostgreSQL", "SQL Server").
        
        Returns:
            bool: True if the service is running or not applicable, False otherwise.
        """
        service_name = self.SERVICE_MAP.get(dbms_type)
        if not service_name:
            # es. SQLite / DuckDB o DBMS non gestito a livello di servizio
            return True

        try:
            result = subprocess.run(
                ["sc", "query", service_name],
                capture_output=True,
                text=True,
                check=False,  # non sollevare eccezione se il comando fallisce
            )

            if result.returncode != 0:
                print(f"[Service Check] Errore nel controllo del servizio {service_name}: {result.stderr}")
                return False

            if "RUNNING" in result.stdout:
                return True
            else:
                print(f"[Service Check] Servizio {service_name} non è in esecuzione. Output:\n{result.stdout}")
                return False
        except Exception as e:
            print(f"[Service Check] Eccezione durante il controllo del servizio {service_name}: {e}")
            return False

    def load_df(self) -> Dict[str, pd.DataFrame]:
        """
        Extracts DataFrames from the session state dictionary.
        
        The GUI passes self.dfs_dict with the structure:
        { "table_name": {"df": pd.DataFrame, ...} }
        
        This function extracts the pd.DataFrame from entry["df"] if present.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping table names to their DataFrames.
        """
        items = list(self.dfs_dict.keys())
        db_to_load: Dict[str, pd.DataFrame] = {}
        for key in items:
            entry = self.dfs_dict[key]
            df = entry["df"] if isinstance(entry, dict) and "df" in entry else entry
            if isinstance(df, pd.DataFrame):
                db_to_load[key] = df
        return db_to_load

    def _sanitize_table_name(self, name: str) -> str:
        """
        Sanitizes a table name by removing invalid characters.
        
        Args:
            name (str): The original table name.
        
        Returns:
            str: Sanitized table name with only alphanumeric characters and underscores.
        """
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).strip("_")
        return safe or "table_unnamed"

    def _sqlalchemy_scheme(self) -> str:
        """
        Returns the SQLAlchemy connection scheme for the selected DBMS.
        
        Returns:
            str: SQLAlchemy connection scheme string (e.g., "sqlite", "postgresql+psycopg").
        """
        m = {
            "SQLite": "sqlite",
            "PostgreSQL": "postgresql+psycopg",
            "MySQL": "mysql+pymysql",
            "SQL Server": "mssql+pyodbc",
            "DuckDB": "duckdb",
        }
        return m.get(self.choice_DBMS, "sqlite")

    # ---------------------- DB disponibili ----------------------

    def get_available_databases(self) -> List[str]:
        """
        Retrieves the list of available databases on the server or in the local directory
        (depending on the chosen DBMS).
        
        For file-based databases (SQLite, DuckDB), lists files in the database directory.
        For server-based databases, queries the server for available databases.
        
        Returns:
            List[str]: List of available database names.
        """
        try:
            if self.choice_DBMS == "SQLite":
                if os.path.exists(DB_DIR):
                    files = [f for f in os.listdir(DB_DIR) if f.endswith(".db")]
                    return [os.path.splitext(f)[0] for f in files]
                return []

            elif self.choice_DBMS == "DuckDB":
                if os.path.exists(DB_DIR):
                    files = [f for f in os.listdir(DB_DIR) if f.endswith(".duckdb")]
                    return [os.path.splitext(f)[0] for f in files]
                return []

            elif self.choice_DBMS == "MySQL":
                conf = self.global_config.get("MYSQL", {})
                default_db = ""
                driver = "mysql+pymysql"
                query = "SHOW DATABASES"

            elif self.choice_DBMS == "PostgreSQL":
                conf = self.global_config.get("POSTGRES", {})
                default_db = "postgres"
                driver = "postgresql+psycopg"
                query = "SELECT datname FROM pg_database WHERE datistemplate = false;"

            elif self.choice_DBMS == "SQL Server":
                # Qui nel file originale c'era "MYSQLSERVER": è stato corretto in "SQLSERVER".
                conf = self.global_config.get("SQLSERVER", {})
                default_db = "master"
                driver = "mssql+pyodbc"
                query = "SELECT name FROM sys.databases"

            else:
                return []

            user = conf.get("user")
            password = conf.get("password")
            host = conf.get("HOST")
            port = conf.get("PORT")

            if not user or not host:
                return []

            url = URL.create(
                drivername=driver,
                username=user,
                password=password,
                host=host,
                port=port,
                database=default_db,
            )

            if self.choice_DBMS == "SQL Server":
                url = url.update_query_dict(
                    {
                        "driver": "ODBC Driver 17 for SQL Server",
                        "TrustServerCertificate": "yes",
                    }
                )

            engine = create_engine(url)
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return [row[0] for row in result]

        except Exception as e:
            print(f"Error listing databases: {e}")
            return []

    # ---------------------- Connection URL helpers ----------------------

    def _normalize_connection_url(self) -> str:
        """
        Normalizes the connection string (even if it arrives in JDBC format)
        into something SQLAlchemy can use.
        
        Parses connection parameters from various formats and constructs a valid
        SQLAlchemy connection URL with appropriate authentication and options.
        
        Returns:
            str: Normalized SQLAlchemy connection URL.
        """
        raw = (self.connection_string or "").strip()
        if raw.startswith("jdbc:"):
            raw = raw[5:]

        parsed = urlparse(raw)
        q = parse_qs(parsed.query or "")

        user = parsed.username or q.get("user", [None])[0] or q.get("username", [None])[0]
        pwd = parsed.password or q.get("password", [None])[0]
        host = parsed.hostname or q.get("host", [None])[0] or "localhost"
        port = parsed.port or (q.get("port", [None])[0] and int(q.get("port", [None])[0]))
        # jdbc_url ora viene preso solo dal parametro esplicito, non dal path
        jdbc_url = q.get("jdbc_url", [None])[0]

        # Se non presenti nella connection string, usa quelli da config_parameters.jsonc
        if self.choice_DBMS == "PostgreSQL":
            pg_conf = self.global_config.get("POSTGRES", {})
            if not user:
                user = pg_conf.get("user")
            if not pwd:
                pwd = pg_conf.get("password")
            if not host:
                host = pg_conf.get("HOST")
            if not port:
                port = pg_conf.get("PORT")

        elif self.choice_DBMS == "MySQL":
            mysql_conf = self.global_config.get("MYSQL", {})
            if not user:
                user = mysql_conf.get("user")
            if not pwd:
                pwd = mysql_conf.get("password")
            if not host:
                host = mysql_conf.get("HOST")
            if not port:
                port = mysql_conf.get("PORT")
            if not jdbc_url:
                jdbc_url = mysql_conf.get("jdbc_url")

        elif self.choice_DBMS == "SQL Server":
            mssql_conf = self.global_config.get("SQLSERVER", {})
            # Usiamo solo HOST dal config, niente user/password, niente porta
            if not host:
                host = mssql_conf.get("HOST", "localhost")

            # Per Windows authentication ignoriamo sempre user e password
            user = None
            pwd = None

            # E NON forziamo la porta: lasciamo che pyodbc usi la stessa risoluzione di "localhost" che ti funziona
            port = None

        default_ports = {"PostgreSQL": 5432, "MySQL": 3306}
        if port is None:
            port = default_ports.get(self.choice_DBMS)

        # Se è stato fornito un jdbc_url esplicito lo usiamo così com'è (tolto l'eventuale prefisso jdbc:)
        if jdbc_url:
            jdbc_url = jdbc_url.strip()
            if jdbc_url.startswith("jdbc:"):
                return jdbc_url[5:]
            return jdbc_url

        scheme = self._sqlalchemy_scheme()

        # Per database embedded, l'URL completo viene costruito altrove
        if self.choice_DBMS in ("SQLite", "DuckDB"):
            return f"{scheme}://"

        auth = ""
        if user:
            auth = user if pwd is None else f"{user}:{pwd}"
            auth += "@"

        hostport = host if port is None else f"{host}:{port}"
        netloc = f"{auth}{hostport}"

        # Dentro _normalize_connection_url, nella parte finale:
        query_params: Dict[str, Any] = {}
        if self.choice_DBMS == "SQL Server":
            if not parsed.query or ("driver=" not in parsed.query.lower()):
                query_params["driver"] = "ODBC Driver 17 for SQL Server"
            # In ogni caso, per Windows auth:
            query_params["TrustServerCertificate"] = "yes"
            query_params["Trusted_Connection"] = "yes"

        query_str = urlencode(query_params) if query_params else ""
        url = urlunparse((scheme, netloc, "", "", query_str, ""))
        return url

    def _server_engine(self):
        """
        Returns an engine connected to the *server* (not to a specific database).
        
        Used for creating/dropping databases on server-based DBMS.
        
        Returns:
            Engine: SQLAlchemy engine connected to the server root.
        """
        url = self._normalize_connection_url()
        connect_args: Dict[str, Any] = {}
        if self.choice_DBMS == "SQL Server":
            connect_args["fast_executemany"] = True
        engine = create_engine(url, connect_args=connect_args, future=True)
        return engine

    def _db_engine(self):
        """
        Returns an engine pointed to the specific database (file or server).
        
        Constructs the appropriate connection URL based on the DBMS type and
        database name/path.
        
        Returns:
            Engine: SQLAlchemy engine connected to the target database.
        """
        if self.choice_DBMS == "SQLite":
            if not self.sqlite_db_path:
                db_path = os.path.abspath(
                    os.path.join(DB_DIR, f"{self.db_name}")
                )
            else:
                if os.path.isabs(self.sqlite_db_path):
                    db_path = self.sqlite_db_path
                else:
                    db_path = os.path.abspath(os.path.join(DB_DIR, self.sqlite_db_path))
            url = f"sqlite:///{db_path}"

        elif self.choice_DBMS == "DuckDB":
            db_path = self._normalize_db_path()
            url = f"duckdb:///{db_path}"

        else:
            server_url = self._normalize_connection_url()
            parsed = urlparse(server_url)
            url = urlunparse(parsed._replace(path=f"/{self.db_name}"))

            if self.choice_DBMS == "SQL Server":
                q = parse_qs(parsed.query or "")
                if "driver" not in {k.lower() for k in q.keys()}:
                    sep = "&" if parsed.query else ""
                    url += f"{sep}driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"

        connect_args: Dict[str, Any] = {}
        if self.choice_DBMS == "SQL Server":
            connect_args["fast_executemany"] = True

        return create_engine(url=url, connect_args=connect_args, future=True)

    # ---------------------- Gestione creazione / reset DB ----------------------

    def _reset_all_db(self) -> None:
        """
        Deletes the database (file or server-based database) if it exists.
        
        For file-based databases, removes the file from disk.
        For server-based databases, executes DROP DATABASE command.
        """
        if not self.db_name and not self.sqlite_db_path:
            raise ValueError("db_name o db_path mancante in config_dict.")

        # SQLite
        if self.choice_DBMS == "SQLite":
            if self.sqlite_db_path:
                if os.path.isabs(self.sqlite_db_path):
                    db_path = self.sqlite_db_path
                else:
                    db_path = os.path.abspath(os.path.join(DB_DIR, self.sqlite_db_path))
            else:
                db_path = os.path.abspath(
                    os.path.join(DB_DIR, f"{self.db_name}")
                )
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except OSError as e:
                    print(f"Error removing SQLite file {db_path}: {e}")
            return

        # DuckDB
        if self.choice_DBMS == "DuckDB":
            db_path = self._normalize_db_path()
            if os.path.exists(db_path):
                os.remove(db_path)
            return


        base_engine = None
        drop_stmt = None

        if self.choice_DBMS == "PostgreSQL":
            server_url = self._normalize_connection_url()
            parsed = urlparse(server_url)
            url_postgres = urlunparse(parsed._replace(path="/postgres"))
            base_engine = create_engine(url_postgres, future=True)
            drop_stmt = text(f'DROP DATABASE IF EXISTS "{self.db_name}"')

        elif self.choice_DBMS == "MySQL":
            base_engine = self._server_engine()
            drop_stmt = text(f"DROP DATABASE IF EXISTS `{self.db_name}`")

        elif self.choice_DBMS == "SQL Server":
            server_url = self._normalize_connection_url()
            parsed = urlparse(server_url)
            url_master = urlunparse(parsed._replace(path="/master"))
            base_engine = create_engine(url_master, future=True, isolation_level="AUTOCOMMIT")
            drop_stmt = text(
                f"""
                IF DB_ID(N'{self.db_name}') IS NOT NULL
                BEGIN
                    ALTER DATABASE [{self.db_name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
                    DROP DATABASE [{self.db_name}];
                END
                """
            )

        if base_engine is not None and drop_stmt is not None:
            with base_engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                try:
                    conn.execute(drop_stmt)
                except Exception as e:
                    print(f"Could not drop database {self.db_name}. It might not exist. Error: {e}")

    def _safe_db_name(self, name: str) -> str:
        """
        Sanitizes a database name to ensure it's safe for use.
        
        Converts to lowercase, removes file extensions, avoids reserved keywords,
        and replaces invalid characters with underscores.
        
        Args:
            name (str): The original database name.
        
        Returns:
            str: Sanitized database name.
        """
        name = name.strip().lower()
        if name in self.RESERVED_KEYWORDS:
            name = f"{name}_db"
        # Rimuove l'estensione .db/.duckdb se presente prima di sanitizzare
        if name.endswith(".db"):
            name = name[:-3]
        elif name.endswith(".duckdb"):
            name = name[:-7]
        
        return re.sub(r"\W+", "_", name)

    def _create_database_if_needed(self) -> None:
        """
        Creates the target database from scratch, deleting it if it already exists.
        
        For SQLite/DuckDB, manages files on disk.
        For server-based DBMS, executes DROP + CREATE commands.
        """
        if not self.db_name and not self.sqlite_db_path:
            raise ValueError("Database name or path are missing.")

        dbname = self.db_name

        # --- SQLite ---
        if self.choice_DBMS == "SQLite":
            print(f"[DBManager Create] SQLite. dbname: {dbname}, sqlite_db_path: {self.sqlite_db_path}")
            if self.sqlite_db_path:
                if os.path.isabs(self.sqlite_db_path):
                    db_path = self.sqlite_db_path
                else:
                    db_path = os.path.abspath(os.path.join(DB_DIR, self.sqlite_db_path))
            else:
                db_path = os.path.abspath(os.path.join(DB_DIR, f"{dbname}.db"))
            
            print(f"[DBManager Create] Final db_path: {db_path}")
            if os.path.exists(db_path):
                os.remove(db_path)
            return

        # --- DuckDB ---
        if self.choice_DBMS == "DuckDB":
            db_path = self._normalize_db_path()
            if os.path.exists(db_path):
                os.remove(db_path)
            return

        # --- DBMS server-based ---

        if self.choice_DBMS == "PostgreSQL":
            server_url = self._normalize_connection_url()
            parsed = urlparse(server_url)
            url_postgres = urlunparse(parsed._replace(path="/postgres"))
            base_engine = create_engine(url_postgres, future=True)

            drop_stmt = text(f'DROP DATABASE IF EXISTS "{dbname}"')
            # Termina connessioni attive al DB prima del DROP
            kill_conn_stmt = text(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = :db AND pid <> pg_backend_pid()
                """
            )
            create_stmt = text(f'CREATE DATABASE "{dbname}"')

            with base_engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                # Chiudi eventuali sessioni che tengono vivo il DB target
                conn.execute(kill_conn_stmt, {"db": dbname})
                conn.execute(drop_stmt)
                conn.execute(create_stmt)
            base_engine.dispose()
            return

        if self.choice_DBMS == "MySQL":
            base_engine = self._server_engine()
            drop_stmt = text(f"DROP DATABASE IF EXISTS `{dbname}`")
            create_stmt = text(
                f"CREATE DATABASE `{dbname}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )

            with base_engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute(drop_stmt)
                conn.execute(create_stmt)
            base_engine.dispose()
            return

        if self.choice_DBMS == "SQL Server":
            server_url = self._normalize_connection_url()
            parsed = urlparse(server_url)
            url_master = urlunparse(parsed._replace(path="/master"))
            base_engine = create_engine(url_master, future=True, isolation_level="AUTOCOMMIT")

            drop_if_exists_stmt = text(
                f"""
                IF DB_ID(N'{dbname}') IS NOT NULL
                BEGIN
                    ALTER DATABASE [{dbname}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
                    DROP DATABASE [{dbname}];
                END
                """
            )
            create_stmt = text(f"CREATE DATABASE [{dbname}]")

            with base_engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute(drop_if_exists_stmt)
                conn.execute(create_stmt)
            base_engine.dispose()
            return

        # Fallback nel caso compaia un DBMS non gestito
        raise ValueError(f"The DBMS {self.choice_DBMS} is not supported")

    # ---------------------- Scelta tabelle e I/O ----------------------

    def _pick_tables_to_load(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Selects which tables to load based on configuration.
        
        Uses self.db_to_load (generated by load_df) with structure:
        { "table_name": pd.DataFrame }
        and filters it based on self.table_list / self.load_all_tables.
        
        Returns:
            List[Tuple[str, pd.DataFrame]]: List of (table_name, dataframe) tuples to load.
        """
        if self.load_all_tables or (
            isinstance(self.table_list, str)
            and self.table_list.strip().lower() == "load all tables"
        ):
            items = list(self.db_to_load.items())

        elif isinstance(self.table_list, (list, tuple)) and self.table_list:
            final_items: List[Tuple[str, pd.DataFrame]] = []
            columns_to_find = set(self.table_list)

            # Itera su ogni (nome_tabella, df_completo) nel dizionario
            for key, original_df in self.db_to_load.items():
                cols_found_in_this_df = [
                    col for col in original_df.columns if col in columns_to_find
                ]
                if cols_found_in_this_df:
                    filtered_df = original_df[cols_found_in_this_df]
                    final_items.append((key, filtered_df))
            items = final_items
        else:
            items = list(self.db_to_load.items())

        return [(self._sanitize_table_name(k), df) for k, df in items]

    # ---------------------- Metodo 1: create_db ----------------------
    def create_db(self,):
        """
        Creates the database and loads the selected DataFrames into it.
        
        This method:
        1. Checks if the DBMS service is running (for server-based databases)
        2. Creates the database structure
        3. Loads all selected tables from DataFrames into the database
        
        Returns:
            tuple: (out_list, True) where out_list is a list of dictionaries with format:
                   [{"table": df, "table_name": ...}, ...]
        """
        results_store = self.ss.setdefault("result", {})
        results_store.setdefault(self.db_name, [])
        out_list: List[Dict[str, Any]] = []

        def worker():
            # Check servizio (per i DBMS server-based)
            if not self._check_service_status(self.choice_DBMS):
                print(f"[DBManager] Servizio {self.choice_DBMS} non attivo.")
                return

            self._create_database_if_needed()
            engine = self._db_engine()
            items = self._pick_tables_to_load()

            with engine.begin() as conn:
                for table_name, df in items:
                    if not isinstance(df, pd.DataFrame):
                        continue
                    try:
                        df.to_sql(
                            name=table_name,
                            con=conn,
                            if_exists="replace",
                            index=False,
                            method="multi",
                            chunksize=1000,
                        )
                        rec = {"table": df, "table_name": table_name}
                        out_list.append(rec)
                    except Exception:
                        print(
                            f"[DBManager Worker Error] Failed to write table '{table_name}' "
                            f"to DB '{self.db_name}'."
                        )
                        traceback.print_exc()
                        # Non rilanciamo l'eccezione perché siamo in un thread

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join()

        results_store[self.db_name] = out_list
        return out_list, True

    # ---------------------- Metodo 2: download_db ----------------------

    def download_db(self):
        """
        Downloads all tables from the selected database.
        
        Connects to an existing database and retrieves all tables as DataFrames.
        
        Returns:
            tuple: (out_list, True) if at least one table exists, otherwise (None, False).
                   out_list format: [{"table": df, "table_name": "tbl1"}, ...]
        """
        results_store = self.ss.setdefault("result", {})
        out_list: List[Dict[str, Any]] = []

        def worker():
            # Check servizio (per i DBMS server-based)
            if not self._check_service_status(self.choice_DBMS):
                print(f"[DBManager] Servizio {self.choice_DBMS} non attivo.")
                return

            engine = self._db_engine()
            insp = inspect(engine)
            tables = insp.get_table_names()
            tables = [t for t in tables if not t.startswith("sqlite_")]

            with engine.connect() as conn:
                for tbl in tables:
                    try:
                        df = pd.read_sql_table(tbl, conn)
                    except Exception:
                        # Dialetti che richiedono quoting diverso
                        if self.choice_DBMS == "SQL Server":
                            quoted = f"[{tbl}]"
                        elif self.choice_DBMS == "MySQL":
                            quoted = f"`{tbl}`"
                        else:
                            quoted = f'"{tbl}"'
                        df = pd.read_sql(text(f"SELECT * FROM {quoted}"), conn)

                    # Converti il nome della tabella in lowercase come richiesto
                    tbl_lower = tbl.lower()
                    out_list.append({"table": df, "table_name": tbl_lower})

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join()

        results_store[self.db_name] = out_list
        if len(out_list) == 0:
            return None, False
        else:
            return out_list, True

    # ---------------------- Helper Methods ----------------------

    def _get_admin_engine(self):
        """
        Returns an engine connected to the server root (or default DB) for admin operations.
        
        For MySQL, connects to the server without specifying a database.
        For PostgreSQL, connects to the 'postgres' default database.
        For SQL Server, connects to the 'master' database.
        
        Returns:
            Engine: SQLAlchemy engine for administrative operations, or None if not applicable.
        """
        if self.choice_DBMS == "MySQL":
            return self._server_engine()
        elif self.choice_DBMS == "PostgreSQL":
            server_url = self._normalize_connection_url()
            parsed = urlparse(server_url)
            url = urlunparse(parsed._replace(path="/postgres"))
            return create_engine(url, future=True)
        elif self.choice_DBMS == "SQL Server":
            server_url = self._normalize_connection_url()
            parsed = urlparse(server_url)
            url = urlunparse(parsed._replace(path="/master"))
            connect_args = {"fast_executemany": True}
            return create_engine(url, connect_args=connect_args, future=True)
        return None

    def _safe_db_name(self, db_name: str) -> str:
        """Sanitizes the database name."""
        return db_name.strip().replace(" ", "_")

    def _check_service_status(self, dbms_type: str) -> bool:
        """
        Checks if the service for the given DBMS is running using connection attempts.
        
        Args:
            dbms_type (str): The type of DBMS to check.
        
        Returns:
            bool: True if the service is accessible, False otherwise.
        """
        if dbms_type == "MySQL":
            return self._check_mysql_status()
        elif dbms_type == "PostgreSQL":
            return self._check_postgres_status()
        elif dbms_type == "SQL Server":
            return self._check_sqlserver_status()
        return True

    def _check_mysql_status(self) -> bool:
        """
        Checks if MySQL service is running by attempting a connection.
        
        Tries pymysql first as fallback, then mysql.connector if available.
        
        Returns:
            bool: True if MySQL is accessible, False otherwise.
        """
        if 'mysql.connector' not in sys.modules and mysql is None:
             # Fallback to pymysql if mysql-connector is not available
             try:
                conf = self.global_config.get("MYSQL", {})
                conn = pymysql.connect(
                    host=conf.get("HOST", "127.0.0.1"),
                    port=int(conf.get("PORT", 3306)),
                    user=conf.get("user", "root"),
                    password=conf.get("password", ""),
                    connect_timeout=3
                )
                conn.close()
                return True
             except Exception:
                 return False

        try:
            conf = self.global_config.get("MYSQL", {})
            conn = mysql.connector.connect(
                host=conf.get("HOST", "127.0.0.1"),
                port=int(conf.get("PORT", 3306)),
                user=conf.get("user", "root"),
                password=conf.get("password", ""),
                connection_timeout=3
            )
            if conn.is_connected():
                conn.close()
                return True
        except Exception:
            return False
        return False

    def _check_postgres_status(self) -> bool:
        """
        Checks if PostgreSQL service is running by attempting a connection.
        
        Returns:
            bool: True if PostgreSQL is accessible, False otherwise.
        """
        if psycopg2 is None: return False
        try:
            conf = self.global_config.get("POSTGRES", {})
            conn = psycopg2.connect(
                host=conf.get("HOST", "127.0.0.1"),
                port=int(conf.get("PORT", 5432)),
                user=conf.get("user", "postgres"),
                password=conf.get("password", ""),
                dbname="postgres",
                connect_timeout=3
            )
            conn.close()
            return True
        except Exception:
            return False

    def _check_sqlserver_status(self) -> bool:
        """
        Checks if SQL Server is running by attempting a Windows Authentication connection.
        
        Uses ODBC Driver 17 for SQL Server with Trusted_Connection.
        
        Returns:
            bool: True if SQL Server is accessible, False otherwise.
        """
        if pyodbc is None:
            return False

        conf = self.global_config.get("SQLSERVER", {})

        driver = "{ODBC Driver 17 for SQL Server}"

        # Leggo host/port/istanza dal json, ma senza user/password
        host = conf.get("HOST", "localhost")
        port = conf.get("PORT")  # es. 1433, opzionale
        instance = conf.get("INSTANCE")  # opzionale, es. "MYSQLSERVER"

        if instance:
            # es: localhost\MYSQLSERVER
            server = rf"{host}\{instance}"
        elif port:
            # es: localhost,1433
            server = f"{host},{port}"
        else:
            # es: localhost
            server = host

        conn_str = (
            f"DRIVER={driver};"
            f"SERVER={server};"
            "DATABASE=master;"
            "Trusted_Connection=yes;"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )

        try:
            with pyodbc.connect(conn_str, timeout=3):
                return True
        except Exception as e:
            print(f"[Windows auth] Connection failed: {e}")
            return False

    # ---------------------- DBMS Management Methods ---------------------
    def server_control(self, action: str) -> Tuple[bool, str]:
        """
        Controls the database server (start, stop, status).
        
        Requires Administrator privileges for start/stop operations.
        
        Args:
            action (str): The action to perform - 'start', 'stop', or 'status'.
        
        Returns:
            Tuple[bool, str]: (success, message) indicating the result of the operation.
        """
        service_name = self.SERVICE_MAP.get(self.choice_DBMS)
        if not service_name:
            if self.choice_DBMS in ["SQLite", "DuckDB"]:
                return True, "Serverless (File-based)"
            return False, f"No service map for {self.choice_DBMS}"

        if action == 'status':
            # Qui usiamo il NOME DEL SERVIZIO, non il nome logico del DBMS
            is_running = self._check_service_status(service_name)
            return True, "Running" if is_running else "Stopped"

        if action == 'start':
            cmd = ["net", "start", service_name]
        elif action == 'stop':
            cmd = ["net", "stop", service_name]
        else:
            return False, "Invalid action"

        try:
            # Richiede privilegi amministrativi
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                return True, f"Service {action}ed successfully"
            else:
                err = res.stderr.strip() or res.stdout.strip()
                if "Access is denied" in err or "Accesso negato" in err:
                    return False, "Access Denied: Run as Administrator"
                return False, f"Error: {err}"
        except Exception as e:
            return False, str(e)

    def get_available_databases(self) -> List[str]:
        """
        Returns a list of available databases for the current DBMS.
        
        For file-based databases, lists files in the database directory.
        For server-based databases, queries the system catalogs.
        
        Returns:
            List[str]: List of database names or file names.
        """
        dbs = []
        try:
            if self.choice_DBMS in ["SQLite", "DuckDB"]:
                # List files in DB_DIR
                if os.path.exists(DB_DIR):
                    exts = {".sqlite", ".db"} if self.choice_DBMS == "SQLite" else {".duckdb"}
                    for f in os.listdir(DB_DIR):
                        if any(f.endswith(ext) for ext in exts):
                            dbs.append(f)
            elif self.choice_DBMS == "MySQL":
                engine = self._get_admin_engine()
                with engine.connect() as conn:
                    res = conn.execute(text("SHOW DATABASES"))
                    dbs = [row[0] for row in res]
            elif self.choice_DBMS == "PostgreSQL":
                engine = self._get_admin_engine()
                with engine.connect() as conn:
                    res = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false"))
                    dbs = [row[0] for row in res]
            elif self.choice_DBMS == "SQL Server":
                engine = self._get_admin_engine()
                with engine.connect() as conn:
                    res = conn.execute(text("SELECT name FROM sys.databases"))
                    dbs = [row[0] for row in res]
        except Exception as e:
            print(f"Error getting databases: {e}")
            # traceback.print_exc()
        return dbs

    def get_db_details(self, db_name: str) -> Dict[str, Any]:
        """
        Retrieves detailed information for a specific database.
        
        Includes database size, list of tables, column information, row counts, and data previews.
        
        Args:
            db_name (str): The name of the database to inspect.
        
        Returns:
            Dict[str, Any]: Dictionary containing database details with keys:
                - name: database name
                - size_mb: size in megabytes (for file-based databases)
                - tables: list of table details (name, columns, rows, preview)
                - error: error message if inspection failed
        """
        details = {
            "name": db_name,
            "size_mb": "N/A",
            "tables": []
        }

        # ===== 1. File size for SQLite / DuckDB =====
        if self.choice_DBMS in ["SQLite", "DuckDB"]:
            # Costruisco percorso file
            # Per sicurezza aggiungo estensione se manca
            filename = db_name
            if not filename.lower().endswith((".db", ".duckdb")):
                if self.choice_DBMS == "SQLite":
                    filename += ".db"
                else:
                    filename += ".duckdb"

            db_path = os.path.join(DB_DIR, filename)

            if os.path.exists(db_path):
                size_bytes = os.path.getsize(db_path)
                details["size_mb"] = f"{size_bytes / (1024 * 1024):.2f} MB"

        # ===== 2. Estraggo tabelle tramite SQLAlchemy =====
        old_db_name = self.db_name
        self.db_name = db_name

        try:
            engine = self._db_engine()
            insp = inspect(engine)
            table_names = insp.get_table_names()

            # Filtra tabelle di sistema SQLite
            if self.choice_DBMS == "SQLite":
                table_names = [t for t in table_names if not t.startswith("sqlite_")]

            with engine.connect() as conn:
                for t in table_names:
                    t_det = {"name": t, "columns": [], "rows": 0, "preview": None}

                    # Columns
                    try:
                        cols = insp.get_columns(t)
                        t_det["columns"] = [
                            {"name": c["name"], "type": str(c["type"])}
                            for c in cols
                        ]
                    except Exception:
                        pass

                    # Rows + Preview
                    try:
                        if self.choice_DBMS == "SQL Server":
                            quoted = f"[{t}]"
                        elif self.choice_DBMS == "MySQL":
                            quoted = f"`{t}`"
                        else:
                            quoted = f'"{t}"'

                        # Count
                        count_res = conn.execute(text(f"SELECT COUNT(*) FROM {quoted}"))
                        t_det["rows"] = count_res.scalar()

                        # Preview
                        if self.choice_DBMS == "SQL Server":
                            query = f"SELECT TOP 5 * FROM {quoted}"
                        else:
                            query = f"SELECT * FROM {quoted} LIMIT 5"

                        df_prev = pd.read_sql(text(query), conn)
                        t_det["preview"] = df_prev

                    except Exception as e:
                        print(f"Error reading table {t}: {e}")

                    details["tables"].append(t_det)

        except Exception as e:
            details["error"] = str(e)

        finally:
            self.db_name = old_db_name

        return details

    def rename_db(self, old_name: str, new_name: str) -> Tuple[bool, str]:
        """
        Renames a database.
        
        For file-based databases, renames the file on disk.
        For server-based databases, uses ALTER DATABASE or equivalent commands.
        
        Args:
            old_name (str): Current database name.
            new_name (str): New database name.
        
        Returns:
            Tuple[bool, str]: (success, message) indicating the result.
        """
        safe_old = self._safe_db_name(old_name)
        safe_new = self._safe_db_name(new_name)
        
        if self.choice_DBMS in ("SQLite", "DuckDB"):
            old_path = os.path.join(DB_DIR, safe_old)
            new_path = os.path.join(DB_DIR, safe_new)
            
            if not os.path.exists(old_path):
                return False, "Source database not found"
            if os.path.exists(new_path):
                return False, "Destination database already exists"
                
            try:
                os.rename(old_path, new_path)
                return True, "Database renamed successfully"
            except Exception as e:
                return False, str(e)
                
        elif self.choice_DBMS == "SQL Server":
            # ALTER DATABASE [Old] MODIFY NAME = [New]
            # Requires single user mode usually
            try:
                engine = self._db_engine(no_db=True)
                with engine.connect() as conn:
                    conn.execute(text(f"ALTER DATABASE [{old_name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE"))
                    conn.execute(text(f"ALTER DATABASE [{old_name}] MODIFY NAME = [{new_name}]"))
                    conn.execute(text(f"ALTER DATABASE [{new_name}] SET MULTI_USER"))
                return True, "Database renamed"
            except Exception as e:
                return False, str(e)
                
        elif self.choice_DBMS == "PostgreSQL":
            # ALTER DATABASE old RENAME TO new
            # Cannot be connected to the db being renamed
            try:
                engine = self._db_engine(no_db=True) # Connect to default (postgres)
                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    # Terminate connections
                    conn.execute(text(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{old_name}'"))
                    conn.execute(text(f"ALTER DATABASE \"{old_name}\" RENAME TO \"{new_name}\""))
                return True, "Database renamed"
            except Exception as e:
                return False, str(e)
                
        elif self.choice_DBMS == "MySQL":
            return False, "Rename not supported for MySQL directly (requires dump/restore)"
            
        return False, "Not implemented"

    def rename_table(self, db_name: str, old_table_name: str, new_table_name: str) -> Tuple[bool, str]:
        """
        Renames a table within a database.
        
        Uses DBMS-specific syntax for renaming tables.
        
        Args:
            db_name (str): The database containing the table.
            old_table_name (str): Current table name.
            new_table_name (str): New table name.
        
        Returns:
            Tuple[bool, str]: (success, message) indicating the result.
        """
        old_db_name = self.db_name
        self.db_name = db_name
        
        try:
            engine = self._db_engine()
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                if self.choice_DBMS == "SQL Server":
                    # sp_rename
                    conn.execute(text(f"EXEC sp_rename '{old_table_name}', '{new_table_name}'"))
                elif self.choice_DBMS == "MySQL":
                    conn.execute(text(f"RENAME TABLE `{old_table_name}` TO `{new_table_name}`"))
                elif self.choice_DBMS == "PostgreSQL":
                    conn.execute(text(f"ALTER TABLE \"{old_table_name}\" RENAME TO \"{new_table_name}\""))
                else: # SQLite / DuckDB
                    conn.execute(text(f"ALTER TABLE \"{old_table_name}\" RENAME TO \"{new_table_name}\""))
            return True, "Table renamed"
        except Exception as e:
            return False, str(e)
        finally:
            self.db_name = old_db_name

    def delete_db(self, db_name: str) -> Tuple[bool, str]:
        """
        Deletes a database.
        
        For file-based databases, removes the file from disk.
        For server-based databases, executes DROP DATABASE command.
        
        Args:
            db_name (str): The name of the database to delete.
        
        Returns:
            Tuple[bool, str]: (success, message) indicating the result.
        """
        if self.choice_DBMS in ("SQLite", "DuckDB"):
            db_path = os.path.join(DB_DIR, db_name)
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
                    return True, "Database deleted"
                return False, "Database file not found"
            except Exception as e:
                return False, str(e)
        else:
            # Server based DROP DATABASE
            try:
                engine = self._db_engine(no_db=True)
                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    if self.choice_DBMS == "SQL Server":
                        conn.execute(text(f"DROP DATABASE [{db_name}]"))
                    elif self.choice_DBMS == "MySQL":
                         conn.execute(text(f"DROP DATABASE `{db_name}`"))
                    elif self.choice_DBMS == "PostgreSQL":
                        # Terminate connections first
                        conn.execute(text(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{db_name}'"))
                        conn.execute(text(f"DROP DATABASE \"{db_name}\""))
                return True, "Database deleted"
            except Exception as e:
                return False, str(e)



def check_service_status(service_name: str) -> bool:
    """
    Standalone function to check if a database service is running.
    
    Creates a temporary DBManager instance to check service status.
    
    Args:
        service_name (str): The name of the service to check.
    
    Returns:
        str: 'running' if the service is active, 'not_running' otherwise.
    """
    db_manager = DBManager(type='status', dict_stato={})

    return 'running' if db_manager._check_service_status(service_name) else 'not_running'
