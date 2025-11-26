# Query with LLMs: An AI-Powered SQL Query Generator and Analyzer

This tool is an advanced web application, built with Streamlit, designed to allow users to interact with their data using natural language. The application acts as a bridge between user questions and databases, leveraging the power of Large Language Models (LLMs) to generate, optimize, and evaluate SQL queries.

Beyond simple query generation, the application offers in-depth dataset analysis, detailed system performance monitoring, and an estimation of CO₂ emissions, promoting a conscious and sustainable approach to data analysis.

## 🌟 Key Features

The application is divided into logical modules that offer a complete and integrated experience:

### 1. Data Connectivity and Management
- **Flexible Data Loading**: Supports loading data from multiple sources:
    - **Local Files**: Upload one or more files (`.csv`, `.parquet`, `.h5`, `.hdf5`). The application can handle multiple files as separate tables within a logical database.
    - **DBMS Connection**: Connects to database management systems like **MySQL** and **SQLite** to load existing tables.
- **Database Creation**: Allows users to upload data from files and materialize them into a new database (currently supporting SQLite and MySQL), facilitating the execution of complex queries.

### 2. In-Depth Dataset Analysis and Profiling
- **Detailed Statistics**: Provides a comprehensive overview of the dataset with metrics such as the number of rows/columns, missing values, duplicate rows, and memory usage.
- **Column-Level Analysis**: Offers a detailed analysis of each column, including:
    - **Detected Data Type**: Classifies columns into categories like Text, Numeric, Categorical, Datetime, and Boolean.
    - **Cardinality**: Shows the number and percentage of unique values.
    - **Sensitive Data (PII) Detection**: Uses **spaCy** models and heuristic rules to identify and flag columns that may contain personally identifiable information (names, emails, addresses, etc.).
- **Relational Profiling and Integrity**:
    - **Primary Key Identification**: Suggests columns or combinations of columns that are candidates for primary keys.
    - **Anomaly Detection**: Detects constant values, numerical outliers (via Z-score), and anomalous dates (in the future or too far in the past).
    - **Semantic Profiling**: Identifies specific formats like emails, tax codes, IBANs, and suggests optimal SQL data types.
- **Interactive Visualizations**: Includes correlation matrices and heatmaps of missing values for intuitive visual exploration.

### 3. Multi-Backend LLM Integration
The application is not tied to a single model or service but offers an "adapter" architecture to connect to various LLM sources:
- **🤗 Hugging Face**:
    - **Online Search**: Search and download models directly from the Hugging Face Hub.
    - **Local Cache**: Utilizes models already present in the local Hugging Face cache.
- **🦙 Ollama**: Integrates with local Ollama servers, allowing you to list, manage, and use any locally served model.
- **🧪 LM Studio**: Connects to local LM Studio servers, with control panels to start/stop the server and manage loaded models.
- **📤 Local Model**: Allows uploading models in standard formats (e.g., `GGUF`, `Transformers`) directly through the interface.
- **🧠 spaCy**: Integrates spaCy models not for generation, but for linguistic analysis of user text and for detecting sensitive entities in the data.

### 4. SQL Query Generation and Evaluation
- **From Natural Language to SQL**: Translates user requests (e.g., "show me customers from Rome with more than 3 orders") into valid SQL queries.
- **Intelligent Prompt Construction**: Generates prompts optimized for the LLM, including the database schema (table and column names with their types) to maximize the accuracy of the generated query.
- **Alternative Query Generation**: It doesn't stop at the first query. The application asks the LLM to generate alternative and potentially more optimized versions of the same query.
- **Execution and Comparison**: Executes both the original query and the alternatives, measuring and comparing execution times for an objective evaluation.

### 5. Performance Monitoring and Sustainability Estimation
- **Real-Time Dashboard**: A sidebar constantly monitors system resources:
    - **CPU** and **GPU** utilization (%).
    - Instantaneous power consumption of CPU and GPU (Watts).
- **CO₂ Emission Estimation**:
    - Calculates instantaneous CO₂ emissions (in g/s and kg/h) based on power consumption and a configurable emission factor.
    - Tracks **cumulative emissions** throughout the entire generation and evaluation process, providing comparative charts for each phase.
- **Comprehensive Reporting**: Each run generates a detailed report with charts on resource usage and CO₂ emissions, separating the LLM generation phase from the database execution phase.

### 6. History and Settings
- **Persistent History**: Saves every run to a local database (SQLite). The history includes the user's question, generated queries, results, execution times, error messages, and all monitoring data.
- **Configurable Settings**: Allows the user to customize key parameters such as the **CO₂ emission factor** and **CPU TDP** to make the estimates more accurate for their specific hardware and location.

## 🔧 Project Architecture

The code is organized into specialized modules to ensure maintainability and scalability:

- **`app.py`**: The entry point of the Streamlit application. It manages the session state and orchestrates the various UI components.
- **`GUI/`**: Contains all modules that define the user interface. Each file corresponds to a specific tab or component (e.g., `load_dataset_gui.py`, `gen_eval_query.py`).
- **`llm_adapters/`**: The core of the model integration. Each file (`huggingface_adapter.py`, `ollama_adapter.py`, etc.) implements a common logic (`list_models`, `get_model_details`, `generate`) for a specific backend.
- **`db_adapters/`**: Manages the logic for connecting to, creating, and executing queries on databases. `DBManager.py` abstracts the creation and population of databases.
- **`utils/`**: A collection of reusable utility functions, such as `prompt_builder.py` (for creating prompts for LLMs), `query_cleaner.py` (for extracting SQL code from LLM responses), and `system_monitor_utilities.py` (for resource monitoring).
- **`charts/`**: Contains functions for generating interactive charts with Plotly.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git

**External Dependencies (optional, depending on the features you use):**
- For the **LM Studio** backend: You need to have [LM Studio](https://lmstudio.ai/) installed and its command-line interface (`lms`) enabled.
- For the **Ollama** backend: You need to have [Ollama](https://ollama.com/) installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd <PROJECT_FOLDER_NAME>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    You need to create a `requirements.txt` file with the following libraries, derived from the source code:
    ```
    streamlit
    pandas
    sqlalchemy
    transformers
    torch
    huggingface_hub
    spacy
    psutil
    GPUtil
    plotly
    requests
    humanize
    numpy
    # Add drivers for the databases you intend to use
    pymysql  # For MySQL
    ```
    Then, run the command:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download a language model for spaCy:**
    For the text analysis and PII detection features, a spaCy model is required.
    ```bash
    python -m spacy download en_core_web_sm  # English model
    python -m spacy download it_core_news_sm  # Italian model
    ```

### Running the Application

Once the installation is complete, start the application with the following command:
```bash
streamlit run app.py
```

The application will automatically open in your web browser.

## 📋 Usage Guide

1.  **Tab 📄 `Load Dataset`**:
    - Choose whether to load data from local files or connect to a DBMS.
    - If loading files, select the files and the correct separator (for CSVs).
    - If connecting to a DBMS, enter the connection parameters.
    - Once loaded, you can explore the dataset, view its preview, detailed statistics, and profiling.

2.  **Tab 🤖 `Load Model`**:
    - Choose the source of your LLM (e.g., "Local Model" for Ollama/LM Studio, "Online Model" for Hugging Face Hub).
    - **Local Model**: Select the backend (e.g., Ollama), use the panel to ensure the server is running, load the list of available models, and select one. Click "Use this model" to activate it.
    - **Online Model**: Use the filters to search for models on Hugging Face, select a model from the results, and click "Load Model" to download it. Once ready, click "Set as active LLM".

3.  **Tab 🧪 `Generate & Evaluate Query`**:
    - Ensure a dataset and an LLM have been loaded and activated.
    - Write your question in natural language in the text area (e.g., "What are the top 5 best-selling products?").
    - Click "🚀 Generate".
    - The application will display real-time resource monitoring. Upon completion, you will see:
        - The original generated SQL query.
        - The proposed alternative queries.
        - The execution results.
        - Comparative charts of performance and CO₂ consumption.

4.  **Tab 📜 `History`**:
    - Here you can review all past runs.
    - Expand each entry to see the full details, including queries, results, errors, and consumption graphs.

5.  **Tab ⚙️ `Settings`**:
    - Customize the CO₂ emission factor and your CPU's TDP to refine the sustainability estimates.