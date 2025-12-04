# -*- coding: utf-8 -*-
"""
relational_profiling_app.py — versione on-demand con thread + progress bar per ogni tab.

Caratteristiche:
- Ogni tab ha un pulsante **Esegui** che avvia l'analisi in un THREAD separato.
- Barra di progresso live per mostrare l'avanzamento.
- Risultati persistenti in st.session_state, quindi restano dopo i rerun di Streamlit.
- Utility robuste (unicità/PK, FK, anomalie, semantica, FD, heatmap, export).

Nota: questa versione evita blocchi lunghi del main thread. Le operazioni usano threading;
per lavoro Python puro (CPU-bound) il guadagno è variabile, ma molte parti Pandas/NumPy
rilasciano il GIL, mantenendo la UI reattiva.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import io
import json
import time
import math
import itertools
import datetime as dt
import zipfile
from collections import Counter
from typing import Dict, Tuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import pandas as pd
import streamlit as st
from utils.translations import get_text
# --- IMPORTS FOR IMPUTATION ---
from sklearn.impute import SimpleImputer, KNNImputer
# Enable IterativeImputer (experimental)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import llm_adapters
# --- IMPORTS FOR RESOURCE MONITORING ---
import threading
from utils.system_monitor_utilities import SystemMonitor, get_static_system_info, get_dynamic_system_info
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _render_numeric_imputation(df, missing_cols, key):

    if 'imputed_df_cache_{key}' not in st.session_state:
        st.session_state[f'imputed_df_cache_{key}'] = None
    
    # Filtra solo colonne numeriche che hanno missing
    numeric_cols_missing = [c for c in missing_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols_missing:
        st.warning("No numeric columns with missing values found.")
        return

    # 2. Selezione colonne
    cols_to_impute = st.multiselect(
        get_text('imputation', 'select_cols'),
        options=numeric_cols_missing,
        default=numeric_cols_missing[:1],
        key=f"num_cols_sel_{key}"
    )

    if not cols_to_impute:
        st.warning(get_text('imputation', 'no_cols'))
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        # 3. Selezione Metodo
        method = st.selectbox(
            get_text('imputation', 'method'),
            options=["simple", "knn", "iterative"],
            format_func=lambda x: get_text('imputation', x),
        key=f"num_method_sel_{key}"
    )

    # 4. Parametri
    imputer = None

    cols_numeric = df[cols_to_impute].select_dtypes(include=[np.number]).columns.tolist()
    cols_non_numeric = [c for c in cols_to_impute if c not in cols_numeric]

    if method == "simple":
        with col2:
            strategy = st.selectbox(
                get_text('imputation', 'strategy'),
                options=["mean", "median", "most_frequent", "constant"],
                format_func=lambda x: get_text('imputation', x),
                key=f"num_strategy_sel_{key}"
            )

        fill_value = None
        if strategy == "constant":
            with col3:
                fill_value = st.text_input(get_text('imputation', 'fill_value'), value="0", key=f"num_fill_val_{key}")
                if cols_numeric and not cols_non_numeric:
                    try:
                        fill_value = float(fill_value)
                    except:
                        pass

        if cols_non_numeric and strategy in ["mean", "median"]:
            st.warning(
                f"⚠️ Strategy '{strategy}' cannot be applied to non-numeric columns: {cols_non_numeric}. They will be skipped or cause error.")

        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

    elif method == "knn":
        if cols_non_numeric:
            st.warning(f"⚠️ KNN Imputation supports only numeric columns. Ignored: {cols_non_numeric}")

        with col2:
            k = st.slider(get_text('imputation', 'n_neighbors'), 1, 20, 5, key=f"num_k_slider_{key}")
            imputer = KNNImputer(n_neighbors=k)

    elif method == "iterative":
        if cols_non_numeric:
            st.warning(f"⚠️ Iterative Imputation supports only numeric columns. Ignored: {cols_non_numeric}")

        with col2:
            max_iter = st.slider(get_text('imputation', 'max_iter'), 1, 50, 10, key=f"num_iter_slider_{key}")
            imputer = IterativeImputer(max_iter=max_iter, random_state=0)
    
    with col3:
    # 5. Row Limit
        row_limit = st.number_input(
            get_text('imputation', 'rows_limit'),
            min_value=0,
            max_value=len(df),
            value=0,
            help=get_text('imputation', 'rows_limit_help'),
            key=f"num_row_limit_{key}"
        )
    impute = st.button(get_text('imputation', 'apply_btn'), key=f"num_apply_btn_{key}")

    metrics = None
    target_df = None

    # 6. Applica
    if impute:
        target_df = df.copy()
        if row_limit > 0:
            target_df = target_df.head(row_limit)

        valid_cols = cols_to_impute
        if method in ["knn", "iterative"]:
            valid_cols = cols_numeric

        if not valid_cols:
            st.error("No valid columns for this method.")
            return

        data_to_impute = target_df[valid_cols]

        with st.spinner("Imputing..."):
            # START MONITORING
            data_list, stop_event, monitor = start_monitoring()
            try:
                imputed_data = imputer.fit_transform(data_to_impute)
            finally:
                # STOP MONITORING
                metrics = stop_monitoring(data_list, stop_event, monitor)

            imputed_df = pd.DataFrame(imputed_data, columns=valid_cols, index=target_df.index)
            target_df[valid_cols] = imputed_df

            st.toast(get_text('imputation', 'success'))
            
    # DISPLAY RESOURCE PLOTS
    if metrics is not None:
        with st.expander("🌳Resource Consumption", expanded=False):
            fig_cpu, fig_co2 = create_resource_plots_two_columns(metrics)
            if fig_cpu and fig_co2:
                c_res1, c_res2 = st.columns(2)
                c_res1.plotly_chart(fig_cpu, use_container_width=True)
                c_res2.plotly_chart(fig_co2, use_container_width=True)
            
    if target_df is not None:
        with st.expander(get_text('imputation', 'preview'), expanded=False):
            rows_to_show = st.number_input(
                get_text("load_dataset", "rows_to_show"),
                min_value=1,
                max_value=len(df),
                value=min(5, len(df)),
                step=1,
                help=get_text("load_dataset", "rows_to_show_help"),
                key=f'numberInput_preview_missing_{key}',
            )
            st.write(target_df.head(rows_to_show))

        if row_limit == 0 or row_limit == len(df):
            st.session_state[f'imputed_df_cache_{key}'] = target_df 
        else:
            st.warning("Imputation performed on partial dataset. To save, set limit to 0 (all rows).")

    if st.session_state[f'imputed_df_cache_{key}'] is not None:
        if st.button(
            "💾 Save imputed dataset",
            key=f"save_imputed_{key}",
            help="Save the modified dataset with the imputed data."
        ):
            st.toast('Dataset saved successfully!')

            dbms_parameters = st.session_state["dataframes"]["DBMS"][st.session_state["db_name"]]

            config_dict = dbms_parameters
            config_dict['dfs_dict'] = target_df
            dict_to_dbsm = {'config_dict': config_dict}
            loaded_db, _ = create_dbms(dict_to_dbsm, "riga 221")
            
            if not loaded_db:
                st.toast(f"Failed to create database.")
            else:
                st.session_state['db_choice'] = dbms_parameters['db_choice']
                st.session_state['db_name'] = dbms_parameters['db_name']
                st.session_state['create_db_done'] = True
                st.session_state["dataframes"]["DBMS"][st.session_state["db_name"]] = target_df


from db_adapters.DBManager import DBManager

def create_dbms(dbms_parameters, msg):
    """Create database from DBMS."""
    print(msg)
    print(dbms_parameters.keys())
    print(dbms_parameters.items())
    with st.spinner("Creating database..."):
        mgr_create = DBManager(dbms_parameters, 'create')

        reloaded_data, reload_success = mgr_create.create_db()
    
    return reload_success, reloaded_data

def _render_llm_imputation(df, missing_cols, key):
    # Filtra solo colonne object/string che hanno missing
    text_cols_missing = [c for c in missing_cols if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c])]

    if not text_cols_missing:
        st.warning(get_text('imputation', 'no_text_cols'))
        return

    # 1. Selezione Colonne
    cols_to_impute = st.multiselect(
        get_text('imputation', 'select_cols'),
        options=text_cols_missing,
        default=text_cols_missing[:1],
        key=f"llm_cols_sel_{key}"
    )

    if not cols_to_impute:
        return

    # 2. Selezione Backend e Modello
    c1, c2 = st.columns(2)
    with c1:
        backend = st.selectbox(get_text('imputation', 'select_backend'), options=list(llm_adapters.LLM_ADAPTERS.keys()),
                               key=f"llm_backend_sel_{key}")

    with c2:
        # Carica modelli dinamicamente
        models = []
        if backend:
            try:
                # Usa config salvata se esiste, altrimenti default
                cfg = st.session_state.get(f"lm_selector__cfg_by_backend", {}).get(backend, {})
                models = llm_adapters.list_models(backend, **cfg)
                if isinstance(models, dict) and 'error' in models:
                    models = []
            except:
                models = []

        model_options = [m if isinstance(m, str) else (m.get('id') or m.get('name')) for m in models] if isinstance(
            models, list) else []
        model = st.selectbox(get_text('imputation', 'select_model'), options=model_options, key=f"llm_model_sel_{key}")

    # 3. Avvio Imputazione
    if st.button(get_text('imputation', 'start_imputation'), key=f"llm_start_btn_{key}"):
        if not model:
            st.error(get_text('imputation', 'no_model_selected'))
            return

        st.session_state[f'imputation_queue_{key}'] = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Trova righe da imputare
        # Limitiamo a max 20 righe per demo/performance se non specificato diversamente?
        # L'utente non ha chiesto limite esplicito qui, ma è "testo lungo", quindi lento.
        # Facciamo tutte le righe con missing nelle colonne selezionate.

        rows_to_process = []
        for col in cols_to_impute:
            missing_indices = df[df[col].isna()].index.tolist()
            for idx in missing_indices:
                rows_to_process.append((idx, col))

        total = len(rows_to_process)
        completed = 0

        # Configurazione LLM
        llm_kwargs = st.session_state.get(f"lm_selector__cfg_by_backend", {}).get(backend, {})

        # START GLOBAL MONITORING
        global_data_list, global_stop_event, global_monitor = start_monitoring()
        llm_intervals = []

        for idx, col in rows_to_process:
            status_text.text(get_text('imputation', 'imputing_progress', current=completed + 1, total=total))

            # Context Row
            row_data = df.loc[idx].to_dict()
            # Rimuovi il valore missing corrente per chiarezza nel prompt
            row_data[col] = "MISSING"

            # Esempi (5 righe non missing per questa colonna)
            examples_df = df[df[col].notna()].sample(n=min(5, len(df[df[col].notna()])), random_state=42)
            examples_str = ""
            for _, ex_row in examples_df.iterrows():
                examples_str += f"- {ex_row.to_dict()}\n"

            prompt = f"""
            Task: Impute the missing value for column '{col}'.

            Context Row (JSON):
            {row_data}

            Similar Examples (JSON):
            {examples_str}

            Instructions:
            - Analyze the context and examples.
            - Predict the most likely value for '{col}' in the Context Row.
            - Return ONLY the value. Do not add quotes or explanations.
            """

            try:
                # Chiamata LLM
                t_start = dt.datetime.now(dt.timezone.utc)
                imputed_val = llm_adapters.generate(
                    backend=backend,
                    prompt=prompt,
                    model_name=model,
                    max_tokens=50,  # Breve
                    **llm_kwargs
                )
                t_end = dt.datetime.now(dt.timezone.utc)
                llm_intervals.append((t_start, t_end))

                st.session_state.setdefault(f'imputation_queue_{key}', []).append({
                    "index": idx,
                    "column": col,
                    "original": None,  # Era NaN
                    "imputed": imputed_val.strip(),
                    "row_data": row_data
                })
            except Exception as e:
                st.error(f"Error imputing row {idx}, col {col}: {e}")

            completed += 1
            progress_bar.progress(completed / total)

        # STOP GLOBAL MONITORING
        global_metrics = stop_monitoring(global_data_list, global_stop_event, global_monitor)

        # Filter for LLM specific
        llm_metrics = []
        for m in global_metrics:
            t = m.get('timestamp')
            # timestamp in metrics is already datetime with timezone (from system_monitor_utilities)
            # We check if it falls in any interval
            if t and any(start <= t <= end for start, end in llm_intervals):
                llm_metrics.append(m)

        st.session_state[f'imputation_metrics_{key}'] = {
            'total': global_metrics,
            'llm': llm_metrics
        }

        st.success("Imputation generation complete. Please review below.")
        st.rerun()

    # 4. Review UI
    if f'imputation_queue_{key}' in st.session_state and st.session_state[f'imputation_queue_{key}']:
        
        # DISPLAY METRICS IF AVAILABLE
        if f'imputation_metrics_{key}' in st.session_state:
            metrics_data = st.session_state[f'imputation_metrics_{key}']
            total_metrics = metrics_data.get('total', [])
            llm_metrics = metrics_data.get('llm', [])
            
            if total_metrics:
                st.markdown("### Resource Consumption")
                
                # Row 1: Total System
                st.markdown("**Total System Consumption**")
                fig_cpu_tot, fig_co2_tot = create_resource_plots_two_columns(total_metrics)
                if fig_cpu_tot and fig_co2_tot:
                    c_tot1, c_tot2 = st.columns(2)
                    c_tot1.plotly_chart(fig_cpu_tot, use_container_width=True)
                    c_tot2.plotly_chart(fig_co2_tot, use_container_width=True)
                
                # Row 2: LLM Specific
                if llm_metrics:
                    st.markdown("**LLM-Only Consumption**")
                    fig_cpu_llm, fig_co2_llm = create_resource_plots_two_columns(llm_metrics)
                    if fig_cpu_llm and fig_co2_llm:
                        c_llm1, c_llm2 = st.columns(2)
                        c_llm1.plotly_chart(fig_cpu_llm, use_container_width=True)
                        c_llm2.plotly_chart(fig_co2_llm, use_container_width=True)
            
            st.divider()

        st.divider()
        st.subheader(get_text('imputation', 'review_header'))

        queue = st.session_state[f'imputation_queue_{key}']

        # Bottoni Globali
        c_all_1, c_all_2 = st.columns(2)
        if c_all_1.button(get_text('imputation', 'accept_all'), type="primary", key=f"accept_all_{key}"):
            for item in queue:
                df.at[item['index'], item['column']] = item['imputed']
            st.session_state[f'imputation_queue_{key}'] = []
            st.success("All changes accepted.")
            st.rerun()

        if c_all_2.button(get_text('imputation', 'reject_all'), key=f"reject_all_{key}"):
            st.session_state[f'imputation_queue_{key}'] = []
            st.warning("All changes rejected.")
            st.rerun()

        st.divider()

        # Lista items
        # Usiamo un indice per rimuovere elementi dalla lista man mano
        indices_to_remove = []

        for i, item in enumerate(queue):
            with st.container(border=True):
                c_info, c_act = st.columns([3, 1])
                with c_info:
                    st.markdown(f"**Row {item['index']} - Column `{item['column']}`**")
                    # Mostra riga completa evidenziando la cella
                    # Creiamo un df di una riga per visualizzazione
                    disp_row = item['row_data'].copy()
                    disp_row[item['column']] = f"✨ {item['imputed']} (was NaN)"
                    st.json(disp_row, expanded=False)
                    st.markdown(f"**Proposed Value:** `{item['imputed']}`")

                with c_act:
                    if st.button(get_text('imputation', 'accept'), key=f"acc_{i}_{key}"):
                        df.at[item['index'], item['column']] = item['imputed']
                        indices_to_remove.append(i)
                        st.rerun()  # Rerun necessario per aggiornare coda

                    if st.button(get_text('imputation', 'reject'), key=f"rej_{i}_{key}"):
                        indices_to_remove.append(i)
                        st.rerun()

        # Pulizia coda (se gestita senza rerun immediato, ma qui usiamo rerun per semplicità)
        if indices_to_remove:
            # Rimuovi in ordine inverso per non sballare indici
            for i in sorted(indices_to_remove, reverse=True):
                del st.session_state[f'imputation_queue_{key}'][i]
            st.rerun()
# ------------------------------------------------------------------------------------
# Esecutore di thread riutilizzabile
# ------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_thread_pool() -> ThreadPoolExecutor:
    max_workers = min(32, (os.cpu_count() or 4))
    return ThreadPoolExecutor(max_workers=max_workers)

# ------------------------------------------------------------------------------------
# Helpers progress
# ------------------------------------------------------------------------------------
def make_progress_reporter(ss_key: str, tag: str, total: int) -> Callable[[int], None]:
    """
    Crea un reporter che aggiorna st.session_state[ss_key]['progress'][tag] in [0,1].
    - total: numero di step previsti
    - il reporter accetta 'done' (step completati) e aggiorna la frazione
    """
    if ss_key not in st.session_state:
        st.session_state[ss_key] = {}
    if 'progress' not in st.session_state[ss_key]:
        st.session_state[ss_key]['progress'] = {}
    st.session_state[ss_key]['progress'][tag] = 0.0

    def _report(done: int):
        frac = max(0.0, min(1.0, (done / total) if total else 1.0))
        st.session_state[ss_key]['progress'][tag] = frac

    return _report

def start_threaded_job(ss_key: str, tag: str, func: Callable[..., Any], *args, **kwargs) -> None:
    """
    Avvia un job in thread e memorizza il Future in session_state.
    - tag distingue i job (es. 'keys', 'card', 'anom', 'sem',  'heat', 'export')
    """
    pool = get_thread_pool()
    if ss_key not in st.session_state:
        st.session_state[ss_key] = {}
    if 'futures' not in st.session_state[ss_key]:
        st.session_state[ss_key]['futures'] = {}
    fut = pool.submit(func, *args, **kwargs)
    st.session_state[ss_key]['futures'][tag] = fut

def job_running(ss_key: str, tag: str) -> bool:
    fut: Future | None = st.session_state.get(ss_key, {}).get('futures', {}).get(tag)
    return bool(fut and not fut.done())

def job_result(ss_key: str, tag: str):
    fut: Future | None = st.session_state.get(ss_key, {}).get('futures', {}).get(tag)
    if fut and fut.done():
        try:
            return fut.result()
        except Exception as e:
            return e
    return None

# ------------------------------------------------------------------------------------
# Utility base
# ------------------------------------------------------------------------------------
def is_null_series(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(object) == "")

def safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.astype(str).nunique(dropna=True))

def zscore_outlier_rate(series: pd.Series, z: float = 3.0) -> float:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    zvals = (s - s.mean()) / std
    return float((zvals.abs() > z).mean())

# ------------------------------------------------------------------------------------
# Resource Monitoring Helpers
# ------------------------------------------------------------------------------------
def start_monitoring():
    """Start resource monitoring thread."""
    data_list = []
    stop_event = threading.Event()
    emission_factor = st.session_state.get('emission_factor', 250.0)
    cpu_tdp = st.session_state.get('cpu_tdp', 65.0)
    monitor = SystemMonitor(data_list, stop_event, emission_factor, cpu_tdp)
    monitor.start()
    return data_list, stop_event, monitor
def stop_monitoring(data_list, stop_event, monitor_thread):
    """Stop resource monitoring and return collected metrics."""
    stop_event.set()
    monitor_thread.join(timeout=2.0)
    return data_list

import plotly.graph_objects as go
from plotly.subplots import make_subplots


import plotly.graph_objects as go
import streamlit as st

import plotly.graph_objects as go
import streamlit as st


def create_resource_plots_two_columns(metrics_data):
    """Return two Plotly figures (CPU/GPU and CO2) with perfect dark/light support."""
    if not metrics_data:
        return None, None

    # Detect Streamlit theme
    theme = st.get_option("theme.base")
    st.write(theme)
    if theme == "dark":
        BG = "#111418"            # dark background
        GRID = "rgba(190,190,190,0.20)"
        AXIS = "#111418"          # visible light gray
        TITLE = "#ffffff"
    else:
        BG = "white"
        GRID = "rgba(0,0,0,0.12)"
        AXIS = "#2c2c2c"
        TITLE = "#1a1a1a"

    # Parse data
    timestamps = [m.get("timestamp") for m in metrics_data]
    cpu = [m.get("cpu", {}).get("percent", 0) for m in metrics_data]
    gpu = [m.get("gpu", {}).get("percent", 0) for m in metrics_data]
    co2 = [
        m.get("cpu", {}).get("co2_gs_cpu", 0) +
        m.get("gpu", {}).get("co2_gs_gpu", 0)
        for m in metrics_data
    ]
    has_gpu = any(gpu)

    # ----------------------
    # CPU + GPU FIGURE
    # ----------------------
    fig_cpu = go.Figure()

    fig_cpu.add_trace(go.Scatter(
        x=timestamps, y=cpu,
        mode="lines", name="CPU %",
        line=dict(color="#4cc3ff", width=3),
        hovertemplate="CPU: %{y:.1f}%<br>%{x}<extra></extra>"
    ))

    if has_gpu:
        fig_cpu.add_trace(go.Scatter(
            x=timestamps, y=gpu,
            mode="lines", name="GPU %",
            line=dict(color="#ff9d40", width=3),
            hovertemplate="GPU: %{y:.1f}%<br>%{x}<extra></extra>"
        ))

    fig_cpu.update_layout(
        height=380,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title=dict(
            text="🧠 CPU Usage (%)" if not has_gpu else "🧠 CPU & 🎨 GPU Usage",
            font=dict(size=20, color=TITLE),
            x=0.02,
            xanchor="left"
        ),
        showlegend=has_gpu,
        legend=dict(
            font=dict(color=AXIS),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        font=dict(color=AXIS)
    )

    fig_cpu.update_xaxes(showgrid=True, gridcolor=GRID, color=AXIS)
    fig_cpu.update_yaxes(showgrid=True, gridcolor=GRID, color=AXIS, range=[0, 100])

    # ----------------------
    # CO2 FIGURE
    # ----------------------
    fig_co2 = go.Figure()

    fig_co2.add_trace(go.Scatter(
        x=timestamps, y=co2,
        mode="lines", fill="tozeroy",
        name="CO₂ (g/s)",
        line=dict(color="#6edc6e", width=3),
        fillcolor="rgba(110,220,110,0.35)",
        hovertemplate="CO₂: %{y:.4f} g/s<br>%{x}<extra></extra>"
    ))

    fig_co2.update_layout(
        height=380,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title=dict(
            text="🌍 CO₂ Emissions (g/s)",
            font=dict(size=20, color=TITLE),
            x=0.02,
            xanchor="left"
        ),
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        font=dict(color=AXIS)
    )

    fig_co2.update_xaxes(showgrid=True, gridcolor=GRID, color=AXIS)
    fig_co2.update_yaxes(showgrid=True, gridcolor=GRID, color=AXIS)

    return fig_cpu, fig_co2



# ------------------------------------------------------------------------------------
# Task functions con PROGRESS (chiamate nei thread)
# ------------------------------------------------------------------------------------
def task_anomalies(
    df: pd.DataFrame,
    z_thresh: float,
    min_year: Optional[int],
    future_days: Optional[int],
    ss_key: str,
    tag: str
) -> Dict[str, Any]:

    # START MONITORING
    data_list, stop_event, monitor = start_monitoring()

    reporter = make_progress_reporter(ss_key, tag, total=max(1, len(df.columns)))

    has_dates = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)

    rows_ = []

    for i, col in enumerate(df.columns, start=1):

        ser = df[col]
        miss_rate = is_null_series(ser).mean()
        nunique = safe_nunique(ser)
        is_const = (nunique == 1)

        # Numeric stats
        if pd.api.types.is_numeric_dtype(ser):
            std = ser.std(skipna=True)
            skew_ = ser.skew(skipna=True)
            kurt_ = ser.kurt(skipna=True)
            out_z = zscore_outlier_rate(ser, z=z_thresh)
        else:
            std = skew_ = kurt_ = np.nan
            out_z = np.nan

        # Date-only stats (only if dataset contains dates)
        if has_dates and pd.api.types.is_datetime64_any_dtype(ser):
            s = pd.to_datetime(ser, errors="coerce", utc=True)
            now = pd.Timestamp.utcnow()

            future = (s > now + pd.Timedelta(days=future_days)).mean() if future_days is not None else 0
            ancient = (s < pd.Timestamp(f"{min_year}-01-01", tz="UTC")).mean() if min_year else 0

            future_pct = round(future * 100, 2)
            ancient_pct = round(ancient * 100, 2)
        else:
            future_pct = np.nan
            ancient_pct = np.nan

        # Severity index: weighted anomalies
        severity = (
            miss_rate * 0.35 +
            (out_z if not np.isnan(out_z) else 0) * 0.35 +
            (abs(skew_) / 10 if not np.isnan(skew_) else 0) * 0.15 +
            (abs(kurt_) / 10 if not np.isnan(kurt_) else 0) * 0.15
        )

        rows_.append({
            "Column": col,
            "Missing %": round(miss_rate * 100, 2),
            "Constant?": "Yes" if is_const else "",
            f"Outliers |z|>{z_thresh} %": round(out_z * 100, 2) if not np.isnan(out_z) else np.nan,
            "Std": std,
            "Skewness": skew_,
            "Kurtosis": kurt_,
            "Future dates %": future_pct,
            f"Dates < {min_year} %": ancient_pct,
            "Severity": round(float(severity), 4)
        })

        reporter(i)

    anomalies = pd.DataFrame(rows_)

    # STOP MONITORING
    metrics = stop_monitoring(data_list, stop_event, monitor)

    return {
        "anomalies": anomalies,
        "metrics": metrics
    }



def task_semantic(df: pd.DataFrame, sem_sample: int, enable_spacy: bool, ss_key: str, tag: str) -> Dict[str, Any]:
    # START MONITORING
    data_list, stop_event, monitor = start_monitoring()
    reporter = make_progress_reporter(ss_key, tag, total=max(1, len(df.columns)))

    # INTERNATIONAL PATTERNS
    patterns = {
        "email": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
        "phone_international": r"^\+?[1-9]\d{6,14}$",
        "italian_tax_code": r"^[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]$",
        "iban": r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$",
        "postal_code_generic": r"^[A-Za-z0-9\s\-]{3,10}$",
        "ssn_usa": r"^\d{3}-?\d{2}-?\d{4}$",
        "nin_uk": r"^[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]$",
        "sin_canada": r"^\d{3}\s?\d{3}\s?\d{3}$",
        "tfn_australia": r"^\d{8,9}$",
        "passport_generic": r"^[A-Za-z0-9]{6,9}$",
        "passport_uk": r"^\d{9}$",
        "passport_usa": r"^\d{9}$",
        "aadhaar_india": r"^\d{4}\s?\d{4}\s?\d{4}$",
        "credit_card": r"^(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})$",
        "ipv4": r"^(?:\d{1,3}\.){3}\d{1,3}$",
        "ipv6": r"^([0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}$",
        "mac_address": r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$",
    }

    # OPTIONAL SPACY
    nlp = None
    if enable_spacy:
        try:
            import spacy
            for model in ("it_core_news_sm", "en_core_web_sm"):
                try:
                    nlp = spacy.load(model)
                    break
                except Exception:
                    continue
        except Exception:
            nlp = None

    rows_sem = []
    patterns_found = set()

    for i, col in enumerate(df.columns, start=1):
        ser = df[col]
        dtype = str(ser.dtype)
        sample = ser.dropna().astype(str).head(int(sem_sample))

        # MATCHES ONLY FOR PATTERNS WITH >0% MATCH
        matches = {}
        for label, rx in patterns.items():
            try:
                m = sample.str.fullmatch(rx, regex=True).mean() if not sample.empty else 0.0
            except Exception:
                m = 0.0
            m = round(float(m) * 100, 2)
            if m > 0:     # <---------- Only keep patterns that matched
                matches[label] = m
                patterns_found.add(label)

        pii_hint = "🔒 sensitive" if any(v > 10 for v in matches.values()) else ""

        # NER (minimal)
        ner_labels = Counter()
        if nlp and not sample.empty:
            try:
                text = "\n".join(sample.sample(min(20, len(sample)), random_state=42))
                doc = nlp(text)
                ner_labels.update(ent.label_ for ent in doc.ents)
            except Exception:
                pass

        # SQL TYPE INFERENCE
        sql_type = "TEXT"
        if pd.api.types.is_integer_dtype(ser):
            sql_type = "BIGINT" if pd.to_numeric(ser, errors="coerce").max() > 2**31 - 1 else "INT"
        elif pd.api.types.is_float_dtype(ser):
            sql_type = "DECIMAL(18,6)"
        elif pd.api.types.is_bool_dtype(ser):
            sql_type = "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(ser):
            sql_type = "TIMESTAMP"
        elif pd.api.types.is_string_dtype(ser):
            try:
                maxlen = int(ser.astype(str).str.len().max())
                if maxlen <= 50: 
                    sql_type = "VARCHAR(50)"
                elif maxlen <= 255: 
                    sql_type = "VARCHAR(255)"
            except Exception:
                pass

        row = {
            "Column": col,
            "dtype": dtype,
            "Suggested_SQL": sql_type,
            "NER_labels": ", ".join(f"{k}:{v}" for k, v in ner_labels.items()) if ner_labels else "",
            "Note": pii_hint
        }

        # Dynamically add only the matching patterns
        for p_label, p_val in matches.items():
            row[f"match_{p_label}_%"] = p_val

        rows_sem.append(row)
        reporter(i)

    sem_df = pd.DataFrame(rows_sem)

    # STOP MONITORING
    metrics = stop_monitoring(data_list, stop_event, monitor)

    miss_matrix = df.isna().astype(int)

    return {
        "miss_matrix": miss_matrix,
        "semantic": sem_df,
        "patterns_found": sorted(patterns_found),
        "metrics": metrics
    }

# ------------------------------------------------------------------------------------
# UI helpers per progress
# ------------------------------------------------------------------------------------
def render_progress(ss_key: str, tag: str, label: str):
    """Mostra (e aggiorna) una progress bar finché il job 'tag' è in corso."""
    prog = st.session_state.get(ss_key, {}).get('progress', {}).get(tag, 0.0)
    bar = st.progress(prog, text=label)
    return bar

# ------------------------------------------------------------------------------------
# Funzione principale: 7 tab, pulsante Esegui per ciascuna, thread + progress
# ------------------------------------------------------------------------------------
def ui_integrita_dataset(
        df: pd.DataFrame,
        name: str = "def",
        key: str = "intg",  # chiave sessione per questa sezione
):
    """Integrità/qualità: Anomalie & Integrità + Esportabilità (ZIP)."""
    ss_key = f"relprof:{key}:{name}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = {"results": {}, "progress": {}, "futures": {}}
    results = st.session_state[ss_key]["results"]
    st.header(get_text("profiling", "anomalies_integrity_header"))
    # header
    n_rows, n_cols = df.shape
    with st.container(border=True):
        st.markdown(get_text("profiling", "integrity_quality_header", rows=n_rows, cols=n_cols))

    has_dates = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)

    c1, c2, c3 = st.columns(3)
    with c1:
        z_thresh = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1, key=f"{ss_key}-z_thresh")

    if has_dates:
        with c2:
            future_grace_days = st.slider("Future date tolerance (days)", 0, 30, 1, key=f"{ss_key}-future_grace_days")
        with c3:
            min_year = st.number_input("Minimum acceptable year", 1800, 2100, 1900)
    else:
        future_grace_days = None
        min_year = None

    tag = "anom"
    if st.button(get_text("profiling", "run"), key=f"{ss_key}-{tag}-run"):
        start_threaded_job(ss_key, 
        tag, task_anomalies, df, float(z_thresh),
                           ss_key, tag)

    if job_running(ss_key, tag):
        st.progress(st.session_state[ss_key]["progress"].get(tag, 0.0), text=get_text("profiling", "analyzing_anomalies"))
        time.sleep(0.1)
        st.rerun()


    res = job_result(ss_key, tag)
    # Display anomalies
    if isinstance(res, dict) and "anomalies" in res:
        results["anomalies"] = res["anomalies"]
        st.dataframe(res["anomalies"], hide_index=True, width='stretch')

        # Resource consumption
        if "metrics" in res and res["metrics"]:
            st.divider()
            st.subheader("📊 Resource Consumption")

            fig_cpu, fig_co2 = create_resource_plots_two_columns(res["metrics"])

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_cpu, use_container_width=True)
            with col2:
                st.plotly_chart(fig_co2, use_container_width=True)


        # --- PDF/HTML Export ---
        with st.container(border=True):
            st.subheader("📄 Export Report")

            import base64
            from io import BytesIO

            def export_html(df):
                html = df.to_html(index=False)
                return html

            def export_pdf(df):
                import pdfkit
                html = df.to_html(index=False)
                pdf = pdfkit.from_string(html, False)
                return pdf

            if "anomalies" in res:
                html = export_html(res["anomalies"])
                st.download_button("⬇️ Export HTML report", data=html, file_name="anomalies_report.html")

                try:
                    pdf = export_pdf(res["anomalies"])
                    st.download_button("⬇️ Export PDF report", data=pdf, file_name="anomalies_report.pdf")
                except:
                    st.info("wkhtmltopdf non presente, impossibile generare PDF.")

def missing_value_tab(df: pd.DataFrame, key=""):
    """
    Tab per gestire i valori mancanti con vari algoritmi.
    """
    st.markdown(f"### {get_text('imputation', 'header')}")

    # 1. Identifica colonne con missing
    missing_cols = df.columns[df.isna().any()].tolist()

    if not missing_cols:
        st.success("No missing values found in this dataset! 🎉")
        return

    st.info(f"{get_text('imputation', 'cols_with_missing')} {', '.join(missing_cols)}")

    tab1, tab2 = st.tabs([get_text('imputation', 'numeric_standard'), get_text('imputation', 'text_llm')])

    with tab1:
        # --- LOGICA ESISTENTE (NUMERICA) ---
        _render_numeric_imputation(df, missing_cols, key)
    with tab2:
        # --- LOGICA LLM (TESTO) ---
        _render_llm_imputation(df, missing_cols, key)


def ui_profiling_relazionale(
        df: pd.DataFrame,
        key: str = "prof",  # chiave sessione per questa sezione
        name: str = "def",
        related_tables: dict | None = None,  # {"clienti": df_clienti, ...} per FK
        fd_max_cols: int = 8,
        fd_sample_rows: int = 100_000,
        fd_tolerance: float = 0.01,
):
    """Profiling relazionale: Chiavi/PK, Cardinalità & FK, FD, Semantica, Heatmap & Tipi."""
    ss_key = f"relprof:{key}:{name}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = {"results": {}, "progress": {}, "futures": {}}
    results = st.session_state[ss_key]["results"]

    # testata compatta
    n_rows, n_cols = df.shape
    st.subheader(get_text("profiling", "semantic_profiling") + " & " + get_text("profiling", "heatmap_types"))
    with st.container(border=True):
        st.markdown(get_text("profiling", "rel_profiling_header", rows=n_rows, cols=n_cols))
        st.caption(get_text("profiling", "lazy_analysis_info"))

    # ordina come richiesto + icone

    c1, c2 = st.columns(2)
    with c1:
        sem_sample = st.slider(get_text("profiling", "sample_rows"), 20, 2000, 200, 20, key=f'slider4_{ss_key}')
    with c2:
        enable_spacy = st.checkbox(get_text("profiling", "use_spacy"), True, key=f'check4_{ss_key}')
    
    tag = "sem&heat"
    if st.button(get_text("profiling", "run"), key=f"{ss_key}-{tag}-run"):
        start_threaded_job(ss_key, tag, task_semantic, df, int(sem_sample), bool(enable_spacy), ss_key, tag)

    if job_running(ss_key, tag):
        st.progress(st.session_state[ss_key]["progress"].get(tag, 0.0), text=get_text("profiling", "semantic_profiling_progress"))
        time.sleep(0.1)
        st.rerun()

    res = job_result(ss_key, tag)
    if isinstance(res, dict) and "semantic" in res:
        results["semantic"] = res["semantic"]
        st.dataframe(res["semantic"], hide_index=True, width='stretch')

        miss_matrix = res["miss_matrix"]

        try:
            mm = (
                miss_matrix
                if len(miss_matrix) <= int(sem_sample)
                else miss_matrix.sample(int(sem_sample), random_state=42)
            )

            fig, ax = plt.subplots(figsize=(min(12, 2 + df.shape[1] * 0.5), 6))
            sns.heatmap(mm.T, cbar=True, ax=ax)

            ax.set_xlabel("righe")
            ax.set_ylabel("colonne")

            with st.expander('🟥 '+get_text("profiling", "heatmap_types"), expanded=False):
                st.pyplot(fig)

        except Exception as e:
            st.info(get_text("profiling", "heatmap_unavailable"))
            st.error(f"Errore nella heatmap: {e}")

        # Display resource consumption plots
        if "metrics" in res and res["metrics"]:
            st.divider()
            st.subheader("📊 Resource Consumption")

            fig_cpu, fig_co2 = create_resource_plots_two_columns(res["metrics"])

            if fig_cpu:
                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(fig_cpu, use_container_width=True)

                with col2:
                    st.plotly_chart(fig_co2, use_container_width=True)




