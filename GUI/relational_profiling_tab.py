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

def top_informative_columns(df: pd.DataFrame, k: int) -> list[str]:
    n_rows = len(df)
    scores = {}
    for c in df.columns:
        u = safe_nunique(df[c])
        if n_rows <= 1:
            score = 0.0
        else:
            frac = u / max(1, n_rows)
            score = frac * (1 - abs(frac - 0.5))  # preferisci cardinalità intermedie
        scores[c] = score
    return [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

def detect_candidate_pk(df: pd.DataFrame, max_k: int = 3, max_combos: int = 50_000) -> list[tuple[tuple[str, ...], float]]:
    """Ritorna [(combo_colonne, missing_rate)] ordinate per dimensione. Early-stop se trova PK."""
    n_rows = len(df)
    results = []
    cols = list(df.columns)

    # singole
    for c in cols:
        nulls = is_null_series(df[c]).sum()
        if nulls == 0 and safe_nunique(df[c]) == n_rows:
            results.append(((c,), 0.0))
    if results:
        return results

    tried = 0
    for k_ in range(2, min(max_k, len(cols)) + 1):
        for combo in itertools.combinations(cols, k_):
            tried += 1
            if tried > max_combos:
                break
            subset = df[list(combo)]
            if is_null_series(subset).any(axis=1).any():
                continue
            try:
                unique_pairs = subset.drop_duplicates().shape[0]
            except TypeError:
                unique_pairs = subset.astype(str).drop_duplicates().shape[0]
            if unique_pairs == n_rows:
                results.append((combo, 0.0))
        if results:
            break
    return results

def candidate_fk_to_related(df_left: pd.DataFrame, related: Dict[str, pd.DataFrame], min_coverage: float = 0.9, progress: Callable[[int], None] | None = None) -> pd.DataFrame:
    """FK semplici: colonna sinistra ⊆ chiave unica destra (singola o coppia). Usa progress su tabelle destre."""
    findings = []
    if not related:
        return pd.DataFrame(findings)
    steps_total = len(related)
    step = 0
    for name_r, dfr in related.items():
        step += 1
        if progress: progress(step)
        unique_right = []
        for c in dfr.columns:
            if dfr[c].notna().all() and safe_nunique(dfr[c]) == len(dfr) and len(dfr) > 0:
                unique_right.append((c,))
        for combo in itertools.combinations(dfr.columns, 2):
            sub = dfr[list(combo)].dropna()
            try:
                if sub.drop_duplicates().shape[0] == len(dfr):
                    unique_right.append(combo)
            except TypeError:
                if sub.astype(str).drop_duplicates().shape[0] == len(dfr):
                    unique_right.append(combo)

        for c_left in df_left.columns:
            left_vals = set(df_left[c_left].dropna().astype(str).unique())
            if not left_vals:
                continue
            for combo in unique_right:
                right_vals = set(
                    dfr[list(combo)].dropna().astype(str).agg("§§".join, axis=1).unique()
                )
                inter = left_vals & right_vals
                coverage = len(inter) / max(1, len(left_vals))
                cover = left_vals.issubset(right_vals)
                if coverage >= min_coverage:
                    findings.append({
                        "colonna_sinistra": c_left if len(combo) == 1 else f"{c_left} (combinata?)",
                        "tabella_destra": name_r,
                        "chiave_destra": " + ".join(combo),
                        "coverage_fk(%)": round(coverage * 100, 2),
                        "ipotetica_FK": "✅" if cover else "⚠️ quasi"
                    })
    return pd.DataFrame(findings)

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
def task_keys_pk(df: pd.DataFrame, pk_max_combo: int, pk_max_combos: int, ss_key: str, tag: str) -> Dict[str, Any]:
    # progress: stimiamo step = n_col + 1 (unicità per colonna) + ricerca PK composte
    reporter = make_progress_reporter(ss_key, tag, total=max(1, len(df.columns) + 1))
    n_rows = len(df)
    # step 1: tabella unicità
    rows = []
    for i, c in enumerate(df.columns, start=1):
        rows.append({
            "Colonna": c,
            "Unici": safe_nunique(df[c]),
            "Missing %": round(is_null_series(df[c]).mean() * 100, 2)
        })
        reporter(i)
    uniq = pd.DataFrame(rows)
    uniq["Unici %"] = (uniq["Unici"] / max(1, n_rows) * 100).round(2)
    uniq["Candidata_PK_singola"] = (uniq["Unici"] == n_rows) & (uniq["Missing %"] == 0)

    # step 2: PK composte (contiamo come un singolo step per non bloccare)
    if pk_max_combo > 1:
        _ = detect_candidate_pk(df, max_k=int(pk_max_combo), max_combos=int(pk_max_combos))
    reporter(len(df.columns) + 1)
    return {"uniq": uniq}

def task_card_fk(df: pd.DataFrame, related_tables: dict | None, min_cov: float, ss_key: str, tag: str) -> Dict[str, Any]:
    reporter = make_progress_reporter(ss_key, tag, total=max(1, len(related_tables) if related_tables else 1))
    # cardinalità intra-tabella (veloce, conteggiamo metà progresso)
    n_rows = len(df)
    card = pd.DataFrame({
        "Colonna": df.columns,
        "Unici": [safe_nunique(df[c]) for c in df.columns]
    })
    card["Unici %"] = (card["Unici"] / max(1, n_rows) * 100).round(2)
    # metà progresso
    reporter( max(1, (len(related_tables) if related_tables else 1)//2) )
    # FK (se presenti)
    fk = None
    if related_tables:
        fk = candidate_fk_to_related(df, related_tables, min_coverage=min_cov, progress=reporter)
    else:
        reporter(1)
    return {"card": card, "fk": fk}

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

def task_fds(df: pd.DataFrame, fd_max_cols: int, fd_sample_rows: int, fd_tolerance: float, ss_key: str, tag: str) -> Dict[str, Any]:
    # progress per colonne considerate
    cols = top_informative_columns(df, min(fd_max_cols, df.shape[1]))
    df2 = df[cols].copy()
    if len(df2) > fd_sample_rows:
        df2 = df2.sample(fd_sample_rows, random_state=42)
    reporter = make_progress_reporter(ss_key, tag, total=max(1, len(cols)))

    findings = []
    for i, a in enumerate(cols, start=1):
        a_vals = df2[a]
        if is_null_series(a_vals).mean() > 0.2:
            reporter(i); continue
        for b in cols:
            if a == b: continue
            g = df2[[a, b]].dropna()
            if g.empty: continue
            try:
                k = g.groupby(a, dropna=False)[b].nunique()
            except TypeError:
                k = g.astype({a: str, b: str}).groupby(a, dropna=False)[b].nunique()
            viol_rate = (k > 1).sum() / max(1, k.shape[0])
            conf = 1.0 - viol_rate
            if conf >= (1 - fd_tolerance):
                findings.append({
                    "A → B": f"{a} → {b}",
                    "violations(%)": round(viol_rate * 100, 2),
                    "confidence(%)": round(conf * 100, 2),
                    "status": "✅ quasi certa" if viol_rate <= fd_tolerance else "⚠️ debole"
                })
        reporter(i)
    fds = pd.DataFrame(findings).sort_values(["violations(%)", "A → B"], ascending=[True, True]) if findings else pd.DataFrame()
    return {"fds": fds}

def task_export_zip(df: pd.DataFrame, parts: dict, key: str, ss_key: str, tag: str) -> Dict[str, Any]:
    reporter = make_progress_reporter(ss_key, tag, total=5)  # 5 file
    n_rows, n_cols = df.shape
    miss_pct = float(df.isna().sum().sum()) / max(1, df.size) * 100.0
    dup_rows = int(df.duplicated().sum())
    candidate_pks = detect_candidate_pk(df, max_k=3)

    summary = {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "missing_pct": round(miss_pct, 2),
        "duplicate_rows": int(dup_rows),
        "candidate_pks": [list(x) for x, _ in candidate_pks] if candidate_pks else [],
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "schema": {c: str(t) for c, t in df.dtypes.items()},
    }
    reporter(1)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")); reporter(2)
        z.writestr("uniqueness.csv", (parts.get("uniq") or pd.DataFrame()).to_csv(index=False).encode("utf-8")); reporter(3)
        z.writestr("anomalies.csv", (parts.get("anomalies") or pd.DataFrame()).to_csv(index=False).encode("utf-8")); reporter(4)
        z.writestr("fds.csv", (parts.get("fds") or pd.DataFrame()).to_csv(index=False).encode("utf-8"))
        z.writestr("semantic.csv", (parts.get("semantic") or pd.DataFrame()).to_csv(index=False).encode("utf-8")); reporter(5)
    return {"zip_bytes": buf.getvalue(), "file_name": f"profilo_dataset_{key}.zip"}

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


def ui_export(
        df: pd.DataFrame,
        name: str = "def",
        key: str = "intg",  # chiave sessione per questa sezione
        depends_on_key: str = "prof"  # chiave dei risultati della sezione profiling (per l'export)
):
    ss_key = f"relprof:{key}:{name}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = {"results": {}, "progress": {}, "futures": {}}
    results = st.session_state[ss_key]["results"]

    # --- 📦 Export
    # --- 📦 Export
    st.header(get_text("profiling", "export_header"))
    st.caption(get_text("profiling", "export_info"))
    tag = "export"
    # recupera pezzi anche dalla sezione Profiling (se già eseguita)
    prof_results = st.session_state.get(f"relprof:{depends_on_key}", {}).get("results", {})
    parts = {
        "uniq": prof_results.get("uniq"),
        "fds": prof_results.get("fds"),
        "semantic": prof_results.get("semantic"),
        "anomalies": results.get("anomalies"),
    }
    if st.button(get_text("profiling", "run"), key=f"{ss_key}-{tag}-run"):
        start_threaded_job(ss_key, tag, task_export_zip, df, parts, key, ss_key, tag)

    # --- BLOCCO CORRETTO ---
    if job_running(ss_key, tag):
        st.progress(st.session_state[ss_key]["progress"].get(tag, 0.0), text=get_text("profiling", "preparing_zip"))
        time.sleep(0.1)  # Aggiunto: Polling
        st.rerun()  # Aggiunto: Forza rerun
    # --- FINE BLOCCO ---

    res = job_result(ss_key, tag)
    if isinstance(res, dict) and "zip_bytes" in res:
        st.download_button(get_text("profiling", "download_zip"), data=res["zip_bytes"], file_name=res["file_name"],
                           mime="application/zip", width='stretch')


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

    # --- 🧩 Chiavi & PK (Corretto anche se commentato)
    # with t1:
    #     ...
    #     tag = "keys"
    #     if st.button("Esegui", key=f"{ss_key}-{tag}-run"):
    #         start_threaded_job(ss_key, tag, task_keys_pk, df, int(pk_max_combo), int(pk_max_combos), ss_key, tag)
    #     if job_running(ss_key, tag):
    #         st.progress(st.session_state[ss_key]["progress"].get(tag, 0.0), text="Calcolo Chiavi & PK...")
    #         time.sleep(0.1) # Aggiunto
    #         st.rerun()      # Aggiunto
    #     res = job_result(ss_key, tag)
    #     ...
    #
    # # --- 🔗 Cardinalità & FK (Corretto anche se commentato)
    # with t2:
    #     ...
    #     tag = "card"
    #     if st.button("Esegui", key=f"{ss_key}-{tag}-run"):
    #         start_threaded_job(ss_key, tag, task_card_fk, df, related_tables if enable_fk else None, float(fk_min_coverage), ss_key, tag)
    #     if job_running(ss_key, tag):
    #         st.progress(st.session_state[ss_key]["progress"].get(tag, 0.0), text="Cardinalità & FK...")
    #         time.sleep(0.1) # Aggiunto
    #         st.rerun()      # Aggiunto
    #     res = job_result(ss_key, tag)
    #     ...
    #
    # # --- ⚖️ FD (Corretto anche se commentato)
    # with t3:
    #     ...
    #     tag = "fd"
    #     if st.button("Esegui", key=f"{ss_key}-{tag}-run"):
    #         start_threaded_job(ss_key, tag, task_fds, df, int(fd_cols_ctl), int(fd_sample_ctl), float(fd_tol_ctl)/100.0, ss_key, tag)
    #     if job_running(ss_key, tag):
    #         st.progress(st.session_state[ss_key]["progress"].get(tag, 0.0), text="Analisi FD...")
    #         time.sleep(0.1) # Aggiunto
    #         st.rerun()      # Aggiunto
    #     res = job_result(ss_key, tag)
    #     ...

    # --- 🧠 Semantico (LA TUA CORREZIONE SPECIFICA)
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

# --- Facoltativo: demo rapida ---
if __name__ == "__main__":
    st.set_page_config(page_title="Profiling Relazionale (on-demand)", layout="wide")
    st.title("Profiling Relazionale & Integrità — on‑demand con thread + progress")
    st.caption("Ogni tab esegue il proprio task in un thread con barra di progresso.")

    st.sidebar.header("Demo dataset")
    demo = st.sidebar.selectbox("Scegli un dataset demo", ["Iris", "Random"])
    if demo == "Iris":
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df_demo = iris.frame
    else:
        df_demo = pd.DataFrame({
            "id": np.arange(5000),
            "user": np.random.choice([f"user_{i}" for i in range(800)], size=5000),
            "amount": np.random.lognormal(mean=2.0, sigma=1.0, size=5000),
            "email": np.where(np.random.rand(5000) < 0.2, [f"u{i}@mail.com" for i in range(5000)], None),
            "when": pd.date_range("2024-01-01", periods=5000, freq="min")
        })

    st.dataframe(df_demo.head(), width='stretch')
    profiling_relazionale_integrita(df_demo, key="demo")
