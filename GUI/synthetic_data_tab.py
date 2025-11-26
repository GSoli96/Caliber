"""
Synthetic Data Generation Tab

This module provides a comprehensive UI for generating synthetic datasets using various strategies.
"""

import streamlit as st
import pandas as pd
import io
from typing import Dict, List, Optional
import traceback

from utils.synthetic_generators import (
    check_library_available,
    generate_faker_data,
    generate_gaussian_copula_data,
    generate_ctgan_data,
    generate_tvae_data,
    generate_hma1_data,
    COMMON_FAKER_PROVIDERS
)
from GUI.dataset_analytics_tab import show_df_details
from GUI.load_file_tab import configure_file_dbms
from utils.translations import get_text


def synthetic_data_tab():
    """Main synthetic data generation tab."""
    
    # Initialize session state for synthetic data
    if 'synthetic_datasets' not in st.session_state:
        st.session_state['synthetic_datasets'] = {}
    if 'synthetic_generation_state' not in st.session_state:
        st.session_state['synthetic_generation_state'] = {
            'strategy': None,
            'generated': False,
            'dataset_name': None
        }
    
    st.markdown("""
    Generate synthetic datasets using various strategies and libraries. Each strategy has different strengths:
    - **Faker**: Simple, realistic tabular data (names, addresses, etc.)
    - **GaussianCopula**: Statistical modeling for numerical data
    - **CTGAN/TVAE**: Deep learning for complex, high-dimensional data
    - **HMA1**: Multi-table relational data with referential integrity
    """)
    
    # Strategy selection
    with st.container(border=True):
        st.subheader("📊 Select Generation Strategy")
        
        strategy = st.selectbox(
            "Choose a synthetic data generation strategy:",
            options=[
                "Faker - Simple Tabular Data",
                "SDV GaussianCopula - Statistical Modeling",
                "SDV CTGAN - Deep Learning (GAN)",
                "SDV TVAE - Deep Learning (VAE)",
                "SDV HMA1 - Multi-table Relational"
            ],
            key="synthetic_strategy_select"
        )
        
        st.session_state['synthetic_generation_state']['strategy'] = strategy
    
    # Strategy-specific UI
    if "Faker" in strategy:
        _render_faker_ui()
    elif "GaussianCopula" in strategy:
        _render_gaussian_copula_ui()
    elif "CTGAN" in strategy:
        _render_ctgan_ui()
    elif "TVAE" in strategy:
        _render_tvae_ui()
    elif "HMA1" in strategy:
        _render_hma1_ui()
    
    # Display generated datasets
    if st.session_state['synthetic_datasets']:
        st.divider()
        _render_generated_datasets()


# ============================================================================
# FAKER UI
# ============================================================================

def _render_faker_ui():
    """Render UI for Faker-based generation."""
    
    with st.container(border=True):
        st.subheader("🎭 Faker - Simple Tabular Data")
        
        st.info("""
        **About Faker**: Generates realistic fake data like names, addresses, emails, etc.
        
        **Best for**: Simple single-table datasets with common data types
        
        **Not suitable for**: Complex relationships, statistical properties, or domain-specific data
        """)
        
        # Check library availability
        available, error = check_library_available("faker")
        if not available:
            st.error(f"""
            ❌ **Faker library not installed**
            
            Install with: `pip install faker`
            
            Error: {error}
            """)
            return
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input(
                "Dataset Name",
                value="faker_dataset",
                help="Name for the generated dataset",
                key="faker_dataset_name"
            )
            
            num_rows = st.number_input(
                "Number of Rows",
                min_value=1,
                max_value=1000000,
                value=1000,
                step=100,
                help="How many rows to generate. More rows = longer generation time.",
                key="faker_num_rows"
            )
        
        with col2:
            locale = st.selectbox(
                "Locale",
                options=['en_US', 'it_IT', 'fr_FR', 'de_DE', 'es_ES', 'pt_BR', 'ja_JP', 'zh_CN'],
                help="Language/region for generated data",
                key="faker_locale"
            )
            
            seed = st.number_input(
                "Random Seed (optional)",
                min_value=0,
                value=42,
                help="Set a seed for reproducible results. Same seed = same data.",
                key="faker_seed"
            )
        
        # Column configuration
        st.markdown("### Column Configuration")
        st.caption("Add columns to your dataset. Each column uses a Faker provider to generate realistic data.")
        
        # Initialize column list in session state
        if 'faker_columns' not in st.session_state:
            st.session_state['faker_columns'] = [
                ("name", "name"),
                ("email", "email"),
                ("address", "address")
            ]
        
        # Display current columns
        for idx, (col_name, provider) in enumerate(st.session_state['faker_columns']):
            col_a, col_b, col_c = st.columns([3, 3, 1])
            
            with col_a:
                new_name = st.text_input(
                    "Column Name",
                    value=col_name,
                    key=f"faker_col_name_{idx}",
                    label_visibility="collapsed"
                )
            
            with col_b:
                # Flatten provider options
                provider_options = []
                for category, providers in COMMON_FAKER_PROVIDERS.items():
                    for prov_key, prov_label in providers:
                        provider_options.append(f"{prov_label} ({prov_key})")
                
                # Find current selection
                current_selection = f"{provider} ({provider})"
                for prov_str in provider_options:
                    if f"({provider})" in prov_str:
                        current_selection = prov_str
                        break
                
                try:
                    default_idx = provider_options.index(current_selection)
                except ValueError:
                    default_idx = 0
                
                selected = st.selectbox(
                    "Faker Provider",
                    options=provider_options,
                    index=default_idx,
                    key=f"faker_provider_{idx}",
                    label_visibility="collapsed"
                )
                
                # Extract provider key from selection
                new_provider = selected.split("(")[-1].rstrip(")")
            
            with col_c:
                if st.button("🗑️", key=f"faker_remove_{idx}", help="Remove column"):
                    st.session_state['faker_columns'].pop(idx)
                    st.rerun()
            
            # Update column if changed
            if new_name != col_name or new_provider != provider:
                st.session_state['faker_columns'][idx] = (new_name, new_provider)
        
        # Add column button
        if st.button("➕ Add Column", key="faker_add_column"):
            st.session_state['faker_columns'].append(("new_column", "word"))
            st.rerun()
        
        # Show provider reference
        with st.expander("📚 Available Faker Providers Reference"):
            for category, providers in COMMON_FAKER_PROVIDERS.items():
                st.markdown(f"**{category}**")
                for prov_key, prov_label in providers:
                    st.caption(f"- `{prov_key}`: {prov_label}")
        
        # Generate button
        st.divider()
        if st.button("🚀 Generate Synthetic Data", key="faker_generate", type="primary", use_container_width=True):
            _generate_faker_dataset(dataset_name, num_rows, locale, seed)


def _generate_faker_dataset(dataset_name: str, num_rows: int, locale: str, seed: int):
    """Generate dataset using Faker."""
    
    if not st.session_state.get('faker_columns'):
        st.error("Please add at least one column")
        return
    
    # Validate unique column names
    col_names = [col[0] for col in st.session_state['faker_columns']]
    if len(col_names) != len(set(col_names)):
        st.error("Column names must be unique")
        return
    
    progress_bar = st.progress(0, text="Generating synthetic data...")
    
    try:
        def progress_callback(current, total):
            progress_bar.progress(current / total, text=f"Generating row {current}/{total}")
        
        df = generate_faker_data(
            num_rows=num_rows,
            columns=st.session_state['faker_columns'],
            locale=locale,
            seed=seed,
            progress_callback=progress_callback
        )
        
        # Store in session state
        st.session_state['synthetic_datasets'][dataset_name] = {
            'dataframe': df,
            'strategy': 'Faker',
            'params': {
                'num_rows': num_rows,
                'locale': locale,
                'seed': seed,
                'columns': st.session_state['faker_columns'].copy()
            }
        }
        
        progress_bar.progress(100, text="✅ Generation complete!")
        st.success(f"Successfully generated dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")
        st.code(traceback.format_exc())
    finally:
        progress_bar.empty()


# ============================================================================
# GAUSSIAN COPULA UI
# ============================================================================

def _render_gaussian_copula_ui():
    """Render UI for GaussianCopula generation."""
    
    with st.container(border=True):
        st.subheader("📊 SDV GaussianCopula - Statistical Modeling")
        
        st.info("""
        **About GaussianCopula**: Uses statistical modeling to learn distributions and correlations from real data.
        
        **Best for**: Numerical data where statistical properties matter (correlations, distributions)
        
        **Requires**: Sample real data to learn from
        
        **Parameters**:
        - **Sample Data**: Upload a CSV file with real data to learn from
        - **Number of Rows**: How many synthetic rows to generate
        """)
        
        # Check library availability
        available, error = check_library_available("sdv.single_table")
        if not available:
            st.error(f"""
            ❌ **SDV library not installed**
            
            Install with: `pip install sdv`
            
            Error: {error}
            """)
            return
        
        # Configuration
        dataset_name = st.text_input(
            "Dataset Name",
            value="gaussian_copula_dataset",
            key="gc_dataset_name"
        )
        
        # Upload sample data
        uploaded_file = st.file_uploader(
            "Upload Sample Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with real data. The model will learn from this data.",
            key="gc_upload"
        )
        
        if uploaded_file:
            try:
                real_data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(real_data)} rows, {len(real_data.columns)} columns")
                
                with st.expander("Preview Sample Data"):
                    st.dataframe(real_data.head(10))
                
                num_rows = st.number_input(
                    "Number of Synthetic Rows to Generate",
                    min_value=1,
                    max_value=1000000,
                    value=len(real_data),
                    help="How many synthetic rows to generate",
                    key="gc_num_rows"
                )
                
                if st.button("🚀 Generate Synthetic Data", key="gc_generate", type="primary", use_container_width=True):
                    _generate_gaussian_copula_dataset(dataset_name, real_data, num_rows)
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")


def _generate_gaussian_copula_dataset(dataset_name: str, real_data: pd.DataFrame, num_rows: int):
    """Generate dataset using GaussianCopula."""
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        def progress_callback(message):
            status_text.text(message)
        
        progress_bar.progress(33)
        df = generate_gaussian_copula_data(
            real_data=real_data,
            num_rows=num_rows,
            progress_callback=progress_callback
        )
        
        progress_bar.progress(100)
        
        # Store in session state
        st.session_state['synthetic_datasets'][dataset_name] = {
            'dataframe': df,
            'strategy': 'GaussianCopula',
            'params': {
                'num_rows': num_rows,
                'original_rows': len(real_data)
            }
        }
        
        status_text.text("✅ Generation complete!")
        st.success(f"Successfully generated dataset '{dataset_name}' with {len(df)} rows")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")
        st.code(traceback.format_exc())
    finally:
        status_text.empty()
        progress_bar.empty()


# ============================================================================
# CTGAN UI
# ============================================================================

def _render_ctgan_ui():
    """Render UI for CTGAN generation."""
    
    with st.container(border=True):
        st.subheader("🧠 SDV CTGAN - Deep Learning (GAN)")
        
        st.info("""
        **About CTGAN**: Conditional Tabular GAN uses deep learning to generate high-quality synthetic data.
        
        **Best for**: Complex, high-dimensional datasets with mixed data types
        
        **Requires**: Sample real data to train on
        
        **Training Time**: Can be slow (minutes to hours depending on data size and epochs)
        
        **Parameters**:
        - **Epochs**: Number of training iterations (more = better quality, slower). Default: 300
        - **Batch Size**: Number of samples per training batch. Default: 500
        """)
        
        # Check library availability
        available, error = check_library_available("sdv.single_table")
        if not available:
            st.error(f"""
            ❌ **SDV library not installed**
            
            Install with: `pip install sdv`
            
            Error: {error}
            """)
            return
        
        st.warning("⚠️ CTGAN training can be computationally intensive. Start with small datasets and fewer epochs for testing.")
        
        # Configuration
        dataset_name = st.text_input(
            "Dataset Name",
            value="ctgan_dataset",
            key="ctgan_dataset_name"
        )
        
        # Upload sample data
        uploaded_file = st.file_uploader(
            "Upload Sample Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with real data to train the model",
            key="ctgan_upload"
        )
        
        if uploaded_file:
            try:
                real_data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(real_data)} rows, {len(real_data.columns)} columns")
                
                with st.expander("Preview Sample Data"):
                    st.dataframe(real_data.head(10))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    num_rows = st.number_input(
                        "Number of Synthetic Rows",
                        min_value=1,
                        max_value=1000000,
                        value=len(real_data),
                        key="ctgan_num_rows"
                    )
                
                with col2:
                    epochs = st.number_input(
                        "Training Epochs",
                        min_value=1,
                        max_value=10000,
                        value=300,
                        help="More epochs = better quality but slower training",
                        key="ctgan_epochs"
                    )
                
                with col3:
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        max_value=10000,
                        value=500,
                        help="Larger batch = faster but more memory",
                        key="ctgan_batch_size"
                    )
                
                if st.button("🚀 Train and Generate", key="ctgan_generate", type="primary", use_container_width=True):
                    _generate_ctgan_dataset(dataset_name, real_data, num_rows, epochs, batch_size)
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")


def _generate_ctgan_dataset(dataset_name: str, real_data: pd.DataFrame, num_rows: int, epochs: int, batch_size: int):
    """Generate dataset using CTGAN."""
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        def progress_callback(message):
            status_text.text(message)
        
        progress_bar.progress(10, text="Initializing CTGAN...")
        
        df = generate_ctgan_data(
            real_data=real_data,
            num_rows=num_rows,
            epochs=epochs,
            batch_size=batch_size,
            progress_callback=progress_callback
        )
        
        progress_bar.progress(100)
        
        # Store in session state
        st.session_state['synthetic_datasets'][dataset_name] = {
            'dataframe': df,
            'strategy': 'CTGAN',
            'params': {
                'num_rows': num_rows,
                'epochs': epochs,
                'batch_size': batch_size,
                'original_rows': len(real_data)
            }
        }
        
        status_text.text("✅ Generation complete!")
        st.success(f"Successfully generated dataset '{dataset_name}' with {len(df)} rows")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")
        st.code(traceback.format_exc())
    finally:
        status_text.empty()
        progress_bar.empty()


# ============================================================================
# TVAE UI
# ============================================================================

def _render_tvae_ui():
    """Render UI for TVAE generation."""
    
    with st.container(border=True):
        st.subheader("🧠 SDV TVAE - Deep Learning (VAE)")
        
        st.info("""
        **About TVAE**: Tabular Variational AutoEncoder uses deep learning to generate synthetic data.
        
        **Best for**: Mixed data types, generally faster than CTGAN
        
        **Requires**: Sample real data to train on
        
        **Advantages over CTGAN**: Usually faster training, works well with continuous variables
        
        **Parameters**:
        - **Epochs**: Number of training iterations. Default: 300
        - **Batch Size**: Number of samples per training batch. Default: 500
        """)
        
        # Check library availability
        available, error = check_library_available("sdv.single_table")
        if not available:
            st.error(f"""
            ❌ **SDV library not installed**
            
            Install with: `pip install sdv`
            
            Error: {error}
            """)
            return
        
        # Configuration
        dataset_name = st.text_input(
            "Dataset Name",
            value="tvae_dataset",
            key="tvae_dataset_name"
        )
        
        # Upload sample data
        uploaded_file = st.file_uploader(
            "Upload Sample Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with real data to train the model",
            key="tvae_upload"
        )
        
        if uploaded_file:
            try:
                real_data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(real_data)} rows, {len(real_data.columns)} columns")
                
                with st.expander("Preview Sample Data"):
                    st.dataframe(real_data.head(10))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    num_rows = st.number_input(
                        "Number of Synthetic Rows",
                        min_value=1,
                        max_value=1000000,
                        value=len(real_data),
                        key="tvae_num_rows"
                    )
                
                with col2:
                    epochs = st.number_input(
                        "Training Epochs",
                        min_value=1,
                        max_value=10000,
                        value=300,
                        help="More epochs = better quality but slower training",
                        key="tvae_epochs"
                    )
                
                with col3:
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        max_value=10000,
                        value=500,
                        help="Larger batch = faster but more memory",
                        key="tvae_batch_size"
                    )
                
                if st.button("🚀 Train and Generate", key="tvae_generate", type="primary", use_container_width=True):
                    _generate_tvae_dataset(dataset_name, real_data, num_rows, epochs, batch_size)
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")


def _generate_tvae_dataset(dataset_name: str, real_data: pd.DataFrame, num_rows: int, epochs: int, batch_size: int):
    """Generate dataset using TVAE."""
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        def progress_callback(message):
            status_text.text(message)
        
        progress_bar.progress(10, text="Initializing TVAE...")
        
        df = generate_tvae_data(
            real_data=real_data,
            num_rows=num_rows,
            epochs=epochs,
            batch_size=batch_size,
            progress_callback=progress_callback
        )
        
        progress_bar.progress(100)
        
        # Store in session state
        st.session_state['synthetic_datasets'][dataset_name] = {
            'dataframe': df,
            'strategy': 'TVAE',
            'params': {
                'num_rows': num_rows,
                'epochs': epochs,
                'batch_size': batch_size,
                'original_rows': len(real_data)
            }
        }
        
        status_text.text("✅ Generation complete!")
        st.success(f"Successfully generated dataset '{dataset_name}' with {len(df)} rows")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")
        st.code(traceback.format_exc())
    finally:
        status_text.empty()
        progress_bar.empty()


# ============================================================================
# HMA1 UI
# ============================================================================

def _render_hma1_ui():
    """Render UI for HMA1 multi-table generation."""
    
    with st.container(border=True):
        st.subheader("🔗 SDV HMA1 - Multi-table Relational Data")
        
        st.info("""
        **About HMA1**: Hierarchical Modeling Algorithm for generating relational databases.
        
        **Best for**: Multi-table datasets with primary/foreign key relationships
        
        **Preserves**: Referential integrity, cardinality, table relationships
        
        **Requires**: 
        - Multiple related tables (CSV files)
        - Metadata defining relationships (primary keys, foreign keys)
        
        **Note**: This is an advanced feature. For simple use cases, consider other strategies.
        """)
        
        # Check library availability
        available, error = check_library_available("sdv.multi_table")
        if not available:
            st.error(f"""
            ❌ **SDV multi-table library not installed**
            
            Install with: `pip install sdv`
            
            Error: {error}
            """)
            return
        
        st.warning("""
        ⚠️ **Advanced Feature**: HMA1 requires careful metadata configuration.
        
        For a simpler alternative, use the "Load File" tab to upload multiple CSV files,
        then use the relational profiling features to analyze relationships.
        """)
        
        st.markdown("""
        ### Implementation Note
        
        Full HMA1 implementation requires:
        1. Multiple table uploads
        2. Relationship definition UI
        3. Metadata configuration
        
        This is a complex feature that would benefit from a dedicated workflow.
        Consider using existing database loading features combined with other synthesis strategies.
        """)


# ============================================================================
# DISPLAY GENERATED DATASETS
# ============================================================================

def _render_generated_datasets():
    """Display all generated synthetic datasets."""
    
    st.header("📦 Generated Synthetic Datasets")
    
    dataset_names = list(st.session_state['synthetic_datasets'].keys())
    
    if not dataset_names:
        st.info("No datasets generated yet. Use the forms above to generate synthetic data.")
        return
    
    # Tabs for each dataset
    tabs = st.tabs(dataset_names)
    
    for tab, dataset_name in zip(tabs, dataset_names):
        with tab:
            dataset_info = st.session_state['synthetic_datasets'][dataset_name]
            df = dataset_info['dataframe']
            strategy = dataset_info['strategy']
            params = dataset_info['params']
            
            # Dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strategy", strategy)
            with col2:
                st.metric("Rows", f"{len(df):,}")
            with col3:
                st.metric("Columns", len(df.columns))
            
            # Parameters
            with st.expander("Generation Parameters"):
                st.json(params)
            
            # Analytics (reuse existing component)
            st.subheader("📊 Dataset Analytics")
            show_df_details(df, dataset_name, f"synthetic_{dataset_name}")
            
            # Download options
            st.divider()
            st.subheader("💾 Download Dataset")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                # CSV download
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"{dataset_name}.csv",
                    mime="text/csv",
                    key=f"download_csv_{dataset_name}",
                    use_container_width=True
                )
            
            with col_d2:
                # Parquet download
                parquet_buffer = io.BytesIO()
                df.to_parquet(parquet_buffer, index=False)
                st.download_button(
                    label="📦 Download as Parquet",
                    data=parquet_buffer.getvalue(),
                    file_name=f"{dataset_name}.parquet",
                    mime="application/octet-stream",
                    key=f"download_parquet_{dataset_name}",
                    use_container_width=True
                )
            
            with col_d3:
                # Excel download
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"{dataset_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_excel_{dataset_name}",
                    use_container_width=True
                )
            
            # Create database option
            st.divider()
            st.subheader("🗄️ Create Database from Synthetic Data")
            
            if st.button(f"Create Database from '{dataset_name}'", key=f"create_db_{dataset_name}"):
                _create_database_from_synthetic(dataset_name, df)


def _create_database_from_synthetic(dataset_name: str, df: pd.DataFrame):
    """Create a database from synthetic dataset."""
    
    # Add to uploaded files (mimicking file upload)
    st.session_state['uploaded_files'][dataset_name] = {
        'uploaded_file': None,  # No actual file
        'separator': ','
    }
    
    # Add to dataframes
    st.session_state['dataframes']['files'][dataset_name] = {
        'df': df,
        'separator': ','
    }
    
    st.success(f"""
    ✅ Dataset '{dataset_name}' added to available datasets!
    
    You can now:
    1. Go to the "Dashboard" tab
    2. Scroll to "Database Configuration"
    3. Create a database from this synthetic data
    """)
    
    st.info("💡 Tip: The dataset is now available in the file list and can be used to create SQLite, DuckDB, or other database types.")
