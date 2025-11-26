import os
import pandas as pd
import streamlit as st

from GUI.message_gui import st_toast_temp

# --- Funzione per Caricamento Dati (con cache) ---
# @st.cache_data
def load_data_files(uploaded_files, separator_first):
    """
    Carica uno o più file in DataFrame di Pandas.
    Supporta: csv, txt, parquet, h5/hdf5.

    Restituisce:
        dict: {nome_file: DataFrame}
    """
    if uploaded_files is isinstance(uploaded_files, list):
        pass
    else:
        uploaded_files = [uploaded_files]

    if not uploaded_files:
        st.warning("Nessun file caricato.")
        return None

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower().lstrip('.')

        db_name = file_name.split('.')[0]

        if db_name in st.session_state['dataframes']['files'].keys():
            st_toast_temp(f'{db_name} già caricato.', 'warning')
            pass
        try:
            uploaded_file.seek(0)  # Ritorna all'inizio del file

            if file_extension in ["csv", "txt"]:
                try:
                    df = pd.read_csv(uploaded_file, sep=separator_first)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=separator_first, encoding='latin1')
                except Exception:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=separator_first, encoding='utf-8')

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
                                f"File HDF5 '{file_name}' letto utilizzando la chiave: '{key}'. Se contiene più dataset, potrebbe essere necessaria una selezione manuale."
                            )
                            uploaded_file.seek(0)
                            df = pd.read_hdf(uploaded_file, key=key)
                        else:
                            raise

            else:
                st.error(f"Estensione non supportata: **.{file_extension}**")
                continue

            if df is not None and not df.empty:
                if db_name not in list(st.session_state['dataframes']['files'].keys()):
                    st.session_state['dataframes']['files'][db_name] = {
                        'df':df,
                         'file_name':file_name}
                    st_toast_temp(f"{db_name} caricato con {len(df)} righe.", 'success')
            else:
                st.error(f"Errore durante la lettura di {file_name}.")

        except Exception as e:
            st.error(f"Errore nel file {file_name}: {e}")

    return True