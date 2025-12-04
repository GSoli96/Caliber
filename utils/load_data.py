import os
import csv
import io
import pandas as pd
import streamlit as st

from GUI.message_gui import st_toast_temp

# --- Function for Data Loading (with cache) ---
# @st.cache_data
def load_data_files(uploaded_files, separator_first):
    """
    Loads one or more files into Pandas DataFrames.
    
    Supports: csv, txt, parquet, h5/hdf5 file formats.
    
    Args:
        uploaded_files: Single file or list of uploaded file objects.
        separator_first (str): Separator character for CSV/TXT files.
    
    Returns:
        bool: True if ALL files were loaded successfully, False otherwise.
    """
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    if not uploaded_files:
        st.warning("No files uploaded.")
        return False

    all_success = True
    first_error_shown = False

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower().lstrip('.')

        db_name = file_name.split('.')[0]

        if db_name in st.session_state['dataframes']['files'].keys():
            st_toast_temp(f'{db_name} already loaded.', 'warning')
            continue

        try:
            uploaded_file.seek(0)  # Return to the beginning of the file
            df = None

            if file_extension in ["csv", "txt"]:
                try:
                    df = pd.read_csv(uploaded_file, sep=separator_first)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=separator_first, encoding='latin1')
                except pd.errors.ParserError:
                    # Attempt to sniff the separator - show error only once
                    if not first_error_shown:
                        uploaded_file.seek(0)
                        try:
                            # Read a sample of the file (first 2048 bytes)
                            sample = uploaded_file.read(2048).decode('utf-8', errors='ignore')
                            uploaded_file.seek(0)
                            dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
                            guessed_sep = dialect.delimiter
                            
                            st.error(
                                f"Error loading **{file_name}**. "
                                f"It seems the separator is incorrect. "
                                f"Detected separator: `{guessed_sep}` (Hex: {hex(ord(guessed_sep))}). "
                                f"Please update the selection."
                            )
                        except Exception as sniff_err:
                            st.error(f"Error loading **{file_name}**: Could not automatically detect separator. Please select the correct separator manually.")
                        
                        first_error_shown = True
                    
                    return False  # Return immediately after first error

                except Exception as e:
                    # Fallback for other errors
                    if not first_error_shown:
                        st.error(f"Error reading {file_name}: {e}")
                        first_error_shown = True
                    return False

            elif file_extension == "parquet":
                df = pd.read_parquet(uploaded_file)

            elif file_extension in ["h5", "hdf5"]:
                try:
                    df = pd.read_hdf(uploaded_file)
                except ValueError:
                    import h5py
                    uploaded_file.seek(0)
                    with h5py.File(uploaded_file, 'r') as f:
                        key = list(f.keys())[0] if f.keys() else None
                        if key:
                            st.warning(
                                f"HDF5 file '{file_name}' read using key: '{key}'. If it contains multiple datasets, manual selection may be required."
                            )
                            uploaded_file.seek(0)
                            df = pd.read_hdf(uploaded_file, key=key)
                        else:
                            raise

            else:
                if not first_error_shown:
                    st.error(f"Unsupported extension: **.{file_extension}**")
                    first_error_shown = True
                return False

            if df is not None and not df.empty:
                if db_name not in list(st.session_state['dataframes']['files'].keys()):
                    st.session_state['dataframes']['files'][db_name] = {
                        'df':df,
                        'file_name':file_name
                    }
                    st_toast_temp(f"{db_name} loaded with {len(df)} rows.", 'success')
            else:
                if not first_error_shown:
                    st.error(f"Error reading {file_name}: DataFrame is empty or None.")
                    first_error_shown = True
                return False

        except Exception as e:
            if not first_error_shown:
                st.error(f"Error in file {file_name}: {e}")
                first_error_shown = True
            return False

    return all_success