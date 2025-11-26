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


def _render_numeric_imputation(df, missing_cols, key):
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
        strategy = st.selectbox(
            get_text('imputation', 'strategy'),
            options=["mean", "median", "most_frequent", "constant"],
            format_func=lambda x: get_text('imputation', x),
            key=f"num_strategy_sel_{key}"
        )

        fill_value = None
        if strategy == "constant":
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

        k = st.slider(get_text('imputation', 'n_neighbors'), 1, 20, 5, key=f"num_k_slider_{key}")
        imputer = KNNImputer(n_neighbors=k)

    elif method == "iterative":
        if cols_non_numeric:
            st.warning(f"⚠️ Iterative Imputation supports only numeric columns. Ignored: {cols_non_numeric}")

        max_iter = st.slider(get_text('imputation', 'max_iter'), 1, 50, 10, key=f"num_iter_slider_{key}")
        imputer = IterativeImputer(max_iter=max_iter, random_state=0)

    # 5. Row Limit
    row_limit = st.number_input(
        get_text('imputation', 'rows_limit'),
        min_value=0,
        max_value=len(df),
        value=0,
        help=get_text('imputation', 'rows_limit_help'),
        key=f"num_row_limit_{key}"
    )

    # 6. Applica
    if st.button(get_text('imputation', 'apply_btn'), key=f"num_apply_btn_{key}"):
        try:
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
                imputed_data = imputer.fit_transform(data_to_impute)
                imputed_df = pd.DataFrame(imputed_data, columns=valid_cols, index=target_df.index)
                target_df[valid_cols] = imputed_df

                st.success(get_text('imputation', 'success'))
                st.write(get_text('imputation', 'preview'))
                st.dataframe(target_df[valid_cols].head(10))

                if row_limit == 0 or row_limit == len(df):
                    st.session_state[f'imputed_df_cache_{key}'] = target_df
                else:
                    st.warning("Imputation performed on partial dataset. To save, set limit to 0 (all rows).")

        except Exception as e:
            st.error(get_text('imputation', 'error', e=e))

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
                imputed_val = llm_adapters.generate(
                    backend=backend,
                    prompt=prompt,
                    model_name=model,
                    max_tokens=50,  # Breve
                    **llm_kwargs
                )

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

        st.success("Imputation generation complete. Please review below.")
        st.rerun()

    # 4. Review UI
    if f'imputation_queue_{key}' in st.session_state and st.session_state[f'imputation_queue_{key}']:
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
    - tag distingue i job (es. 'keys', 'card', 'anom', 'sem', 'fd', 'heat', 'export')
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
# Task functions con PROGRESS (chiamate nei thread)
# ------------------------------------------------------------------------------------


def task_anomalies(df: pd.DataFrame, z_thresh: float, min_year: int, future_days: int, ss_key: str, tag: str) -> Dict[str, Any]:
    reporter = make_progress_reporter(ss_key, tag, total=max(1, len(df.columns)))
    rows_ = []
    for i, col in enumerate(df.columns, start=1):
        ser = df[col]
        miss = is_null_series(ser).mean()
        const = int((safe_nunique(ser) == 1))
        out_frac = zscore_outlier_rate(ser, z=z_thresh) if pd.api.types.is_numeric_dtype(ser) else float("nan")
        # date
        if pd.api.types.is_datetime64_any_dtype(ser):
            s = pd.to_datetime(ser, errors="coerce", utc=True)
            now = pd.Timestamp.utcnow()
            future = float((s > now + pd.Timedelta(days=future_days)).mean())
            ancient = float((s < pd.Timestamp(f"{min_year}-01-01", tz="UTC")).mean())
            date_future = round(future * 100, 2)
            date_old = round(ancient * 100, 2)
        else:
            date_future = float("nan"); date_old = float("nan")
        rows_.append({
            "Colonna": col,
            "Missing %": round(miss * 100, 2),
            "Costante?": "✅" if const else "",
            f"Outlier |z|>{z_thresh:g} %": (round(out_frac * 100, 2) if not np.isnan(out_frac) else np.nan),
            "Date future %": date_future,
            f"Date < {min_year} %": date_old,
        })
        reporter(i)
    anomalies = pd.DataFrame(rows_)
    return {"anomalies": anomalies}

def task_semantic(df: pd.DataFrame, sem_sample: int, enable_spacy: bool, ss_key: str, tag: str) -> Dict[str, Any]:
    reporter = make_progress_reporter(ss_key, tag, total=max(1, len(df.columns)))

    patterns = {
        "email": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        "telefono_it": r"^\+?3?9?\s?[\d\s\-]{6,}$",
        "codice_fiscale_it": r"^[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]$",
        "iban_it": r"^IT[0-9A-Z]{25}$",
        "cap_it": r"^\d{5}$",
    }
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
    for i, col in enumerate(df.columns, start=1):
        ser = df[col]
        dtype = str(ser.dtype)
        sample = ser.dropna().astype(str).head(int(sem_sample))
        matches = {}
        for label, rx in patterns.items():
            try:
                m = sample.str.fullmatch(rx, regex=True).mean() if not sample.empty else 0.0
            except Exception:
                m = 0.0
            matches[label] = round(float(m) * 100, 2)
        pii_hint = "🔒 possibile PII" if any(v > 10 for v in matches.values()) else ""
        ner_labels = Counter()
        if nlp and not sample.empty:
            try:
                text = "\n".join(sample.sample(min(20, len(sample)), random_state=42))
                doc = nlp(text)
                ner_labels.update([ent.label_ for ent in doc.ents])
            except Exception:
                pass
        # tipo SQL suggerito (euristico)
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
                if maxlen <= 50: sql_type = "VARCHAR(50)"
                elif maxlen <= 255: sql_type = "VARCHAR(255)"
            except Exception:
                pass
        rows_sem.append({
            "Colonna": col, "dtype": dtype, "Suggerito_SQL": sql_type,
            "Match email %": matches["email"], "Match tel %": matches["telefono_it"],
            "Match CF %": matches["codice_fiscale_it"], "Match IBAN %": matches["iban_it"],
            "Match CAP %": matches["cap_it"],
            "NER labels (sample)": ", ".join(f"{k}:{v}" for k, v in ner_labels.items()) if ner_labels else "",
            "Nota": pii_hint
        })
        reporter(i)
    sem_df = pd.DataFrame(rows_sem)
    return {"semantic": sem_df}



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

    c1, c2, c3 = st.columns(3)
    with c1:
        z_thresh = st.slider(get_text("profiling", "outlier_threshold"), 2.0, 5.0, 3.0, 0.1, key=f"sliderc1_{ss_key}")
    with c2:
        future_grace_days = st.slider(get_text("profiling", "future_date_tolerance"), 0, 30, 1, key=f"sliderc2_{ss_key}")
    with c3:
        min_year = st.number_input(get_text("profiling", "min_year"), 1800, 2100, 1900, key=f"ninputc3_{ss_key}")
    tag = "anom"
    if st.button(get_text("profiling", "run"), key=f"{ss_key}-{tag}-run"):
        start_threaded_job(ss_key, tag, task_anomalies, df, float(z_thresh), int(min_year), int(future_grace_days),
                           ss_key, tag)

    if job_running(ss_key, tag):
        st.progress(st.session_state[ss_key]["progress"].get(tag, 0.0), text=get_text("profiling", "analyzing_anomalies"))
        time.sleep(0.1)
        st.rerun()


    res = job_result(ss_key, tag)
    if isinstance(res, dict) and "anomalies" in res:
        results["anomalies"] = res["anomalies"]
        st.dataframe(res["anomalies"].sort_values("Missing %", ascending=False), hide_index=True, width='stretch')

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

    # --- SCELTA TIPO IMPUTAZIONE ---
    imputation_type = st.radio(
        get_text('imputation', 'imputation_type'),
        options=["numeric", "text"],
        format_func=lambda x: get_text('imputation', 'numeric_standard') if x == "numeric" else get_text('imputation',
                                                                                                         'text_llm'),
        key=f"imputation_type_{key}"
    )

    if imputation_type == "numeric":
        # --- LOGICA ESISTENTE (NUMERICA) ---
        _render_numeric_imputation(df, missing_cols, key)
    else:
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
    with st.container(border=True):
        st.markdown(get_text("profiling", "rel_profiling_header", rows=n_rows, cols=n_cols))
        st.caption(get_text("profiling", "lazy_analysis_info"))

    # ordina come richiesto + icone
    t4, t5 = st.tabs([
        # "🧩 Chiavi & PK",
        # "🔗 Cardinalità & FK",
        # "⚖️ Dipendenze Funzionali",
        get_text("profiling", "semantic_profiling"),
        get_text("profiling", "heatmap_types"),
    ])

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            sem_sample = st.slider(get_text("profiling", "sample_rows"), 20, 2000, 200, 20, key=f'slider4_{ss_key}')
        with c2:
            enable_spacy = st.checkbox(get_text("profiling", "use_spacy"), False, key=f'check4_{ss_key}')
        tag = "sem"
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

    # --- 🌡️ Heatmap & Tipi
    with t5:
        c1, c2 = st.columns(2)
        with c1:
            hm_max_rows = st.number_input(get_text("profiling", "max_rows_sample"), 500, 100_000, 2_000, step=500,
                                          key=f'ninput5_{ss_key}')
        with c2:
            show_types = st.checkbox(get_text("profiling", "merge_dtypes"), True, key=f'check5_{ss_key}')

        def _task_heat(df: pd.DataFrame, hm_max_rows: int, show_types: bool, ss_key: str, tag: str):
            reporter = make_progress_reporter(ss_key, tag, total=3)
            miss_matrix = df.isna().astype(int);
            reporter(1)
            miss_by_col = miss_matrix.mean().to_frame("Missing %").assign(
                **{"Missing %": lambda x: (x["Missing %"] * 100).round(2)});
            reporter(2)
            reporter(3)
            return {"miss_by_col": miss_by_col, "miss_matrix": miss_matrix, "hm_max_rows": hm_max_rows,
                    "show_types": show_types}

        tag = "heat"
        if st.button(get_text("profiling", "run"), key=f"{ss_key}-{tag}-run"):
            start_threaded_job(ss_key, tag, _task_heat, df, int(hm_max_rows), bool(show_types), ss_key, tag)

        # --- BLOCCO CORRETTO ---
        if job_running(ss_key, tag):
            st.progress(st.session_state[ss_key]["progress"].get(tag, 0.0), text=get_text("profiling", "preparing_heatmap"))
            time.sleep(0.1)  # Aggiunto: Polling
            st.rerun()  # Aggiunto: Forza rerun
        # --- FINE BLOCCO ---

        res = job_result(ss_key, tag)
        if isinstance(res, dict):
            miss_by_col = res["miss_by_col"].reset_index().rename(columns={"index": "Colonna"})
            miss_matrix = res["miss_matrix"]
            if res.get("show_types"):
                type_map = df.dtypes.astype(str).rename("dtype").reset_index().rename(columns={"index": "Colonna"})
                out_df = miss_by_col.merge(type_map, on="Colonna", how="left").sort_values("Missing %", ascending=False)
                st.dataframe(out_df, hide_index=True, width='stretch')
            else:
                st.dataframe(miss_by_col.sort_values("Missing %", ascending=False), hide_index=True, width='stretch')

            try:
                import matplotlib.pyplot as plt, seaborn as sns
                mm = miss_matrix if len(miss_matrix) <= res["hm_max_rows"] else miss_matrix.sample(res["hm_max_rows"],
                                                                                                   random_state=42)
                fig, ax = plt.subplots(figsize=(min(12, 2 + df.shape[1] * 0.5), 6))
                sns.heatmap(mm.T, cbar=True, ax=ax);
                ax.set_xlabel("righe");
                ax.set_ylabel("colonne")
                st.pyplot(fig, width='stretch')
            except Exception:
                st.info(get_text("profiling", "heatmap_unavailable"))
