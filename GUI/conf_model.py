import time
import numpy as np
import pandas as pd
import json
import streamlit as st

import spacy.cli
from spacy.util import is_package
from llm_adapters.spacy_adapter import (
    COMMON_SPACY_MODELS, model_details
)
from GUI.message_gui import st_toast_temp
from llm_adapters.lmstudio_adapter import lms_get_stream
from llm_adapters.lmstudio_adapter import lmstudio_panel
from llm_adapters.sensitive_entity import is_sensitive_column
from llm_adapters.spacy_adapter import model_details, COMMON_SPACY_MODELS, suffix_of
from utils.load_config import get_HF_Token
from llm_adapters.model_downloader import download_model_HF
from llm_adapters.ollama_adapter import ollama_panel   # <— aggiungi
from utils.translations import get_text

import time

from utils.icons import Icons

# Icone per gli adapter LLM
ICONS = Icons.ICONS

import llm_adapters
import streamlit as st
import llm_adapters
import llm_adapters
from utils.load_config import get_HF_Token  # se già lo importi altrove, lascia pure
import sys
import subprocess
from typing import Dict, Any
import io, tarfile, zipfile, json, re, time, shutil
from pathlib import Path
import json
import requests

def label_with_icon(name: str) -> str:
    return f"{ICONS.get(name, '🧩')} {name}"

def get_nonce(key_prefix: str) -> int:
    k = f"{key_prefix}__nonce"
    if k not in st.session_state:
        st.session_state[k] = 0
    return st.session_state[k]


def do_reset(key_prefix: str):
    keys_to_drop = [k for k in list(st.session_state.keys()) if k.startswith(f"{key_prefix}_")]
    for kdrop in keys_to_drop:
        st.session_state.pop(kdrop, None)
    nonce_key = f"{key_prefix}__nonce"
    st.session_state[nonce_key] = st.session_state.get(nonce_key, 0) + 1
    try:
        llm_adapters.clear_cache_for("list_models")
    except Exception as e:
        st.warning(get_text("conf_model", "unable_to_clean", e=e))
    st.rerun()

# --- sostituisci in conf_model.py ---
def configure_local_model_tab(key_prefix: str = "lm_selector"):
    st.markdown("""
        <style>
        div[data-baseweb="tab-list"] {
            justify-content: space-between;
        }
        div[data-baseweb="tab-list"] button:nth-child(1) { order: 1; }
        div[data-baseweb="tab-list"] button:nth-child(2) { order: 2; }  /* LLM 2 */
        div[data-baseweb="tab-list"] button:nth-child(3) { order: 3; }  /* LLM 3 */
        div[data-baseweb="tab-list"] button:nth-child(5) { order: 9; }  /* LLM 4 */
        </style>
    """, unsafe_allow_html=True)
    section_id = key_prefix

    def k(name: str) -> str:
        # chiavi stabili per tutti i widget e lo stato
        return f"{section_id}__{name}"

    st.subheader(get_text("conf_model", "select_model_header"))

    backends = list(llm_adapters.LLM_ADAPTERS.keys())
    # Tab con icone
    tabs = st.tabs([label_with_icon(b) for b in backends])

    # Stato persistente per backend
    st.session_state.setdefault(k("cfg_by_backend"), {b: {} for b in backends})
    st.session_state.setdefault(k("models_by_backend"), {b: None for b in backends})
    st.session_state.setdefault(k("selected_by_backend"), {b: None for b in backends})

    cfg_by_backend = st.session_state[k("cfg_by_backend")]
    models_by_backend = st.session_state[k("models_by_backend")]
    selected_by_backend = st.session_state[k("selected_by_backend")]

    # ---- UI per ogni tab ----
    for backend, tab in zip(backends, tabs):
        with tab:
            st.markdown(f"{label_with_icon(backend)}")
            cfg = dict(cfg_by_backend.get(backend) or {})
            c1, c2 = st.columns(2)
            # parametri specifici
            if backend == "LM Studio":
                with c1:
                    host = st.text_input(f"{get_text('conf_model', 'host_lm_studio')}",
                                         value=cfg.get("host", "http://localhost:1234"),
                                         key=k(f"lm_host_{backend}"))
                with c2:
                    flt = st.text_input(f"{get_text('conf_model', 'filter_contains')}", value=cfg.get("filter", ""),
                                        key=k(f"lm_filter_{backend}"))
                cfg.update(host=host, filter=flt)
                lmstudio_panel(host=host, key=f'{backend}_local')
            elif backend == "Ollama":
                with c1:
                    host = st.text_input(f"{get_text('conf_model', 'host_ollama')}",
                                         value=cfg.get("host", "http://localhost:11434"),
                                         key=k(f"ol_host_{backend}"))
                with c2:
                    flt = st.text_input(f"{get_text('conf_model', 'filter_contains')}", value=cfg.get("filter", ""),
                                        key=k(f"ol_filter_{backend}"))
                cfg.update(host=host, filter=flt)
                ollama_panel(host=host, key=f'{backend}_local')
            elif backend == "Hugging Face":
                with c1:
                    tok = st.text_input(f"{get_text('conf_model', 'hf_token_opt')}",
                                        value=cfg.get("token", get_HF_Token()),
                                        type="password", key=k(f"hf_token_{backend}"))
                cfg.update(token=tok)
            elif backend == "Spacy":
                st.warning(get_text("conf_model", "spacy_warning"))

            cfg_by_backend[backend] = cfg
            st.divider()
            a1, a2, a3 = st.columns([1, 1, 2])
            with a1:
                do_load = st.button(f"{get_text('conf_model', 'load_list')}", key=k(f"load_{backend}"))
            with a2:
                do_clear = st.button(f"{get_text('conf_model', 'clear_list')}", key=k(f"clear_{backend}"))
            with a3:
                do_reset_btn = st.button(f"{get_text('conf_model', 'global_reset')}", key=k(f"reset_{backend}"))

            if do_reset_btn:
                if "do_reset" in globals() and callable(globals()["do_reset"]):
                    globals()["do_reset"](key_prefix=section_id)
                # in ogni caso, puliamo anche il nostro namespace
                for key in list(st.session_state.keys()):
                    if key.startswith(f"{section_id}__"):
                        del st.session_state[key]
                st.rerun()

            if do_clear:
                models_by_backend[backend] = None
                selected_by_backend[backend] = None
                st.success(get_text("conf_model", "list_cleared"))
                st.rerun()

            if do_load:
                if backend == 'LM Studio':
                    flag_server = st.session_state['server_lmStudio']
                elif backend == 'Ollama':
                    flag_server = st.session_state['server_ollama']
                else:
                    flag_server = True

                if flag_server:
                    with st.spinner(get_text("conf_model", "loading_models", backend=backend)):
                        models = llm_adapters.list_models(backend, **cfg)
                    if isinstance(models, dict) and 'error' in models:
                        st.error(models['error'])
                    elif not isinstance(models, list) or not models:
                        st.warning(get_text("conf_model", "no_models_found"))
                    else:
                        models_by_backend[backend] = models
                        selected_by_backend[backend] = selected_by_backend.get(backend) or models[0]
                        st.success(get_text("conf_model", "models_found", n=len(models)))
                else:
                    st_toast_temp(get_text("conf_model", "server_not_running"), 'warning')
                    st.warning(get_text("conf_model", "server_not_running"))


            models = models_by_backend.get(backend) or []
            if models:
                def _label(x):
                    if isinstance(x, str): return x
                    if isinstance(x, dict): return x.get("id") or x.get("name") or str(x)
                    return str(x)

                labels = [_label(m) for m in models]

                current = selected_by_backend.get(backend)
                default_idx = labels.index(_label(current)) if current in models else 0

                sel = st.selectbox(f"{get_text('conf_model', 'available_model')}", options=labels,
                                   index=default_idx, key=k(f"model_{backend}"))
                selected_by_backend[backend] = sel

                with st.expander(f"{get_text('conf_model', 'details')}", expanded=False):
                    # Tab interne per azioni
                    if backend in ["LM Studio", "Ollama"]:
                        t1, t2, t3 = st.tabs([f"{get_text('conf_model', 'details')}",
                                              f"🧪 {get_text('conf_model', 'test_generation')}",
                                              f"{get_text('conf_model', 'server_cli')}"])
                    elif backend in ("Hugging Face"):
                        t1, t2 = st.tabs([f"{get_text('conf_model', 'details')}", f"🧪 {get_text('conf_model', 'test_generation')}"])
                    else:
                        t1, = st.tabs([f"{get_text('conf_model', 'details')}"])

                    with t1:
                        details = llm_adapters.get_model_details(backend=backend, model_name=sel, **cfg)

                        if isinstance(details, dict):
                            nested_keys = [z for z, v in details.items() if isinstance(v, dict)]
                            flat_part = {z: v for z, v in details.items() if z not in nested_keys}
                            nested_part = {z: v for z, v in details.items() if z in nested_keys}

                            rows = _dict_to_table_rows(flat_part, section="Overview")
                            for z, sub in nested_part.items():
                                rows.extend(_dict_to_table_rows(sub, section=z))

                            df = pd.DataFrame(rows, columns=[
                                get_text('conf_model', 'dataframe_section'), 
                                get_text('conf_model', 'dataframe_field'), 
                                get_text('conf_model', 'dataframe_value')])

                            df = df[
                                (~df[get_text('conf_model', 'dataframe_value')].apply(_is_empty_value)) &
                                (~df[get_text('conf_model', 'dataframe_value')].apply(_is_complex_value))
                                ].reset_index(drop=True)

                            # pulizia nomi "Campo"
                            df = _clean_campo_names(df)

                            # separa overview e spiegazioni
                            overview_df = df[
                                df[
                                    get_text('conf_model', 'dataframe_section')] == get_text('conf_model', 'dataframe_overview')].drop(columns=[get_text('conf_model', 'dataframe_section')])
                            expl_df_full = df[df[get_text('conf_model', 'dataframe_section')] != get_text('conf_model', 'dataframe_overview')]

                            # memorizza i nomi delle sezioni PRIMA di droppare la colonna
                            section_names = sorted(expl_df_full[get_text('conf_model', 'dataframe_section')].unique()) if not expl_df_full.empty else []

                            # poi elimina la colonna
                            expl_df = expl_df_full.drop(columns=[get_text('conf_model', 'dataframe_section')])

                            # ---- 1️⃣ TABELLONA OVERVIEW ----
                            st.markdown(f"### {get_text('conf_model', 'model_overview')}")
                            st.dataframe(overview_df, hide_index=True, width="stretch")

                            # ---- 2️⃣ EXPLANATION DETAILS ----
                            if not expl_df.empty:
                                st.markdown(f"### {get_text('conf_model', 'explanation_details')}")
                                with st.expander(get_text("conf_model", "open_explanations"), expanded=False):
                                    tabs = st.tabs(section_names)
                                    for tab_i, sec in zip(tabs, section_names):
                                        with tab_i:
                                            sec_df = expl_df_full[
                                                expl_df_full[
                                                    get_text('conf_model', 'dataframe_section')] == sec][
                                                [get_text('conf_model', 'dataframe_field'),
                                                get_text('conf_model', 'dataframe_value')]].reset_index(drop=True)
                                            st.table(sec_df)

                    if backend in ("Hugging Face", "LM Studio"):
                        with (t2 if backend != "LM Studio" else t2):
                            prompt = st.text_area(get_text("conf_model", "prompt"), get_text("conf_model", "prompt_placeholder"),
                                                  key=k(f"prompt_{backend}"))
                            cols = st.columns([1, 1])
                            with cols[0]:
                                max_tokens = st.number_input(get_text("conf_model", "max_new_tokens"), 1, 512, value=64,
                                                             key=k(f"max_new_tokens_{backend}"))
                            out_key = k(f"last_out_{backend}")
                            if st.button(f"{get_text('conf_model', 'run')}", key=k(f"run_{backend}")):
                                with st.spinner(get_text("conf_model", "inference")):
                                    out = llm_adapters.generate(
                                        backend=backend,
                                        prompt=prompt,
                                        model_name=sel,
                                        max_tokens=int(max_tokens),
                                        **cfg
                                    )
                                st.session_state[out_key] = out

                            if st.session_state.get(out_key) is not None:
                                with st.expander(f"{get_text('conf_model', 'response')}", expanded=True):
                                    st.write(st.session_state[out_key])
                                    if st.button(f"{get_text('conf_model', 'clear_output')}",
                                                 key=k(f"clear_out_{backend}")):
                                        st.session_state[out_key] = None
                                        st.rerun()

                    if backend  == "LM Studio":
                        with t3:
                            lmstudio_panel(host=cfg.get("host", "http://localhost:1234"), key=f'local_tab3_{backend}')

                    if backend == 'Ollama':
                        with t3:
                            ollama_panel(host=cfg.get("host", "http://localhost:11434"), key=f'local_tab3_{backend}')

                col_set, col_sp = st.columns([3, 5])
                with col_sp:
                    empty = st.empty()
                with col_set:
                    if st.button(get_text("conf_model", "use_model"), key=k(f"set_active_{backend}")):
                        st.session_state.setdefault('llm', {})
                        st.session_state['llm'] = {
                            'backend': backend,
                            'model': sel if isinstance(sel, str) else str(sel),
                            'status': 'loaded',  # per i backend locali/serviti lo consideriamo pronto
                            'kwargs': dict(cfg)  # host/token/filter ecc. salvati sopra per quel backend
                        }
                        # per retro-compatibilità con codice esistente:
                        st.session_state['llm_backend'] = backend
                        st.session_state['llm_model'] = sel if isinstance(sel, str) else str(sel)
                        empty.success(get_text("conf_model", "active_model", backend=backend, sel=sel))
                        st_toast_temp(get_text("conf_model", "active_model", backend=backend, sel=sel), 'success')

def configure_online_model(key_prefix):
    backend_display_options = list(llm_adapters.LLM_ADAPTERS.keys())

    tab1, tab2, tab3, tab4 = st.tabs([label_with_icon(n) for n in backend_display_options])

    with tab1:
        hugging_face_tab()

        results = st.session_state.get('results_HF', [])
        submitted = st.session_state.get('submit_HF', None)

        if not results and submitted is None:
            st.info(get_text("conf_model", "set_filters_info"))
        elif not results and submitted is True:
            st.warning(get_text("conf_model", "no_models_criteria"))
        elif results and submitted:
            st.success(get_text("conf_model", "models_found_hf", n=len(results)))
            st.subheader(get_text("conf_model", "models_found_header"))

            # Se sono tanti, usa selectbox invece di tab infinite
            names = []
            seen = set()
            for m in results:
                mid = (m.get('modelId') or m.get('id') or '').strip()
                if not mid:
                    continue
                # evita duplicati
                if mid in seen:
                    continue
                seen.add(mid)
                names.append(mid)

            if not names:
                st.warning(get_text("conf_model", "search_invalid_id"))
                return

            selected = st.selectbox(get_text('conf_model', 'select_model'), names, key="hf_model_select")

            # dettaglio del selezionato
            m = next((x for x in results if (x.get('modelId') or x.get('id')) == selected), None)
            if m:
                st.markdown(f"### [{selected}](https://huggingface.co/{selected})")
                st.write(
                    f"**{get_text('conf_model', 'hugging_task')}:** {m.get('pipeline_tag', '—')}  |  "
                    f"**{get_text('conf_model', 'hugging_download')}:** {m.get('downloads', '—')}  |  "
                    f"**{get_text('conf_model', 'hugging_likes')}:** {m.get('likes', '—')}  |  "
                    f"**{get_text('conf_model', 'hugging_authors')}:** {m.get('author', '—')}"
                )
                tags = m.get('tags') or []
                if tags:
                    st.caption("Tags: " + ", ".join(tags[:8]))
                    m['tags'] = tags[:8]
                card = m.get('cardData') or {}
                lang = card.get('language')
                if lang:
                    st.caption(get_text("conf_model", "language", lang=lang))

                load_model = st.button(f"{get_text('conf_model', 'load_model_btn')}", key="load_model_{}".format(selected))

                # --- MODIFICA CHIAVE: Gestione dello stato al click ---
                state = st.session_state.get('hf_dl', {})

                if load_model:
                    # L'utente ha cliccato "Load".
                    # Resettiamo lo stato hf_dl per forzare un nuovo download.
                    state.clear()
                    state.update({
                        "running": False, "stop": False, "progress": 0, "bytes": 0,
                        "total": 0, "retries": 0, "max_retries": 4,
                        "note": "", "error": None, "local_dir": None,
                        "started_at": None, "pipe": None, "thread": None,
                        "model_id": selected  # Imposta subito il modello target
                    })

                if state.get("model_id") == selected:
                    token = st.session_state['token_HF']
                    task = m.get('pipeline_tag') or 'text-generation'

                    download_model_HF(model_id=selected, task=task, token=token)
                ready = (
                        bool(state.get("local_dir") or state.get("pipe")) and
                        state.get("model_id") == selected
                )

                if ready:
                    st.caption(get_text("conf_model", "local_cache", path=state.get('local_dir') or '—'))

                col_set, col_sp = st.columns([3, 5])
                with col_sp:
                    empty = st.empty()
                with col_set:
                    set_active = st.button(get_text("conf_model", "set_active_llm"),
                                           key=f"hf_set_active_{selected}",
                                           disabled=not ready)
                if set_active:
                    st.session_state['llm'] = {
                        'backend': "Hugging Face",
                        'model': selected,
                        'status': 'loaded',  # pronto: file scaricati e/o pipeline creata
                        'kwargs': {
                            'token': st.session_state.get('token_HF', ''),
                            'local_dir': state.get('local_dir')  # utile per l'adapter
                        }
                    }
                    st.session_state['llm_backend'] = "Hugging Face"
                    st.session_state['llm_model'] = selected
                    empty.success(get_text("conf_model", "active_model", backend="Hugging Face", sel=selected))
                    st_toast_temp(get_text("conf_model", "active_model", backend="Hugging Face", sel=selected), 'success')

            else:
                st.warning(get_text("conf_model", "item_not_found"))

    # --- TAB: Spacy ---
    with tab4:
        spacy_tab()
        spacy_ner_load()
    # --- TAB: Ollama ---
    with tab2:
        ollama_tab()

    # --- TAB: LM Studio ---
    with tab3:
        lmstudio_tab()

def upload_llm():
    backend_display_options = list(llm_adapters.LLM_ADAPTERS.keys())
    llm_backend = st.selectbox(get_text('conf_model', 'model_source'), options=backend_display_options,
                               key='llm_backend',
                               on_change=lambda: st.session_state.update({'llm_model': None}))
    if llm_backend:
        available_models = llm_adapters.list_models(llm_backend)
        if isinstance(available_models, list) and available_models:
            st.selectbox(get_text("conf_model", "local_model_available"), options=available_models, key='llm_model')
            if st.session_state.llm_model:
                with st.expander(get_text('conf_model', 'model_details'), expanded=True):
                    model_details_dict = llm_adapters.get_model_details(backend=llm_backend,
                                                                        model_name=st.session_state.llm_model)
                    if 'error' not in model_details_dict:
                        st.markdown("\n".join([f"- **{key}:** {value}" for key, value in model_details_dict.items()]))
                    else:
                        st.warning(get_text("conf_model", "details_error", error=model_details_dict['error']))
        elif isinstance(available_models, dict) and 'error' in available_models:
            st.error(available_models['error'])

# ---------- Helpers ----------
def _norm(s: str) -> str:
    return (s or "").strip().lower()

def spacy_show_model(name):
    # usa la cache di streamlit per evitare doppi rendering
    @st.cache_data(show_spinner=False)
    def cached_spacy_models() -> list[str] | dict:
        return llm_adapters.list_models(name)

    with st.expander(f"{ICONS['Installed Models']} {get_text('conf_model', 'installed_spacy')}", expanded=False):
        spacy_models = cached_spacy_models()

        if isinstance(spacy_models, list) and spacy_models:
            st.markdown("\n".join([f"- `{m}`" for m in spacy_models]))
            st.caption(get_text("conf_model", "select_below"))
        else:
            st.info(get_text("conf_model", "no_spacy_found"))

        # st.divider()
        st.subheader(f"{ICONS['Install Model']} {get_text('conf_model', 'install_spacy')}")

        col1, col2 = st.columns([3, 1])
        with col1:
            default_idx = 0 if "it_core_news_sm" in COMMON_SPACY_MODELS else 0
            model_to_install = st.selectbox(
                f"{ICONS['Choose a Model']} {get_text('conf_model', 'choose_install')}",
                COMMON_SPACY_MODELS,
                index=default_idx,
                help=get_text("conf_model", "install_help")
            )
        with col2:
            verbose = st.toggle(get_text("conf_model", "verbose_logs"), value=True, help=get_text("conf_model", "verbose_help"))

        # === Nuova sezione: Dettagli del modello selezionato ===
        st.markdown(f"### {ICONS['Selected Model Details']} {get_text('conf_model', 'selected_details')}")
        details = model_details(model_to_install)
        _render_model_tabs(details)

        # Evita doppi click durante l'installazione
        installing_key = "spacy_installing"
        if installing_key not in st.session_state:
            st.session_state[installing_key] = False

        disabled = st.session_state[installing_key]

        if st.button(f"{ICONS['Download Models']} {get_text('conf_model', 'download_model_btn')}", type="primary", disabled=disabled):
            st.session_state[installing_key] = True
            with st.spinner(get_text("conf_model", "installing", name=model_to_install)):
                rc = _download_spacy_model(model_to_install, verbose=verbose)
            # quando cambia la select del modello spaCy (es. in spacy_show_model o spacy_tab)
            if model_to_install and st.session_state['spacy_model']['model'] != model_to_install:
                st.session_state['spacy_model'] = {'model': model_to_install, 'status': 'notLoad'}

            if rc == 0:
                st.success(get_text("conf_model", "install_success", name=model_to_install))
                st.session_state['spacy_model'] = {
                    'model': model_to_install,
                    'status': 'Load'
                }
                cached_spacy_models.clear()
                time.sleep(0.3)
                st.session_state[installing_key] = False

                st.rerun()
            else:
                st.error(get_text("conf_model", "install_failed", name=model_to_install, rc=rc))
                st.session_state[installing_key] = False



def _download_spacy_model(model_name: str, *, verbose: bool = True) -> int:
    cmd = [sys.executable, "-m", "spacy", "download", model_name]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    progress = st.progress(0)
    status = st.empty()
    log_box = st.empty() if verbose else None
    logs = []
    step = 0
    max_step = 90
    tick = 0

    while True:
        line = proc.stdout.readline()
        if line:
            tick += 1
            status_text = line.strip()
            if status_text:
                status.write(get_text("conf_model", "downloading", text=status_text))
            if verbose:
                logs.append(line.rstrip("\n"))
                if len(logs) > 400:
                    logs = logs[-400:]
                log_box.code("\n".join(logs), language="bash")
            if step < max_step:
                step += 1
                progress.progress(step / 100.0)
        else:
            if proc.poll() is not None:
                break
            if step < max_step:
                step += 1
                progress.progress(step / 100.0)
            time.sleep(0.1)

    rc = proc.returncode
    progress.progress(1.0 if rc == 0 else max(step, 5) / 100.0)
    return rc

def _render_model_tabs(details: Dict[str, Any]) -> None:
    t_overview, t_pipeline, t_size = st.tabs([get_text("conf_model", "overview"), get_text("conf_model", "pipeline_task"), get_text("conf_model", "size_resources")])

    with t_overview:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(get_text("conf_model", "model_label"), details["model"])
            st.metric(get_text("conf_model", "lang_label"), f'{details["language"]} ({details["lang_code"]})')
            st.metric(get_text("conf_model", "version_label"), details["version"] or "N/D")
        with col2:
            st.metric(get_text("conf_model", "spacy_compat"), details["spacy_version"] or "N/D")
            st.metric(get_text("conf_model", "vectors"), details["vectors"])
            st.metric(get_text("conf_model", "installed"), get_text("conf_model", "yes") if details["installed"] else get_text("conf_model", "no"))
            if details["installed_path"]:
                st.caption(get_text("conf_model", "path_label", path=details['installed_path']))

        if details.get("notes"):
            st.info(get_text("conf_model", "notes") + "  •  ".join(details["notes"]))

    with t_pipeline:
        st.write(get_text("conf_model", "pipeline_components"))
        st.write(" → " + "  →  ".join(f"`{c}`" for c in details["pipeline"]))
        st.write(get_text("conf_model", "supported_tasks"))
        for task in details["tasks"]:
            st.write(f"- {task}")

    with t_size:
        st.write(get_text("conf_model", "expected_size"))
        st.write(details["size_hint"])
        st.write(get_text("conf_model", "dependencies"))
        deps = []
        if suffix_of(details["model"]) == "trf":
            deps.append(get_text("conf_model", "deps_transformers"))
        else:
            deps.append(get_text("conf_model", "deps_spacy"))
        if details["vectors"] == "presenti":
            deps.append(get_text("conf_model", "vectors_included"))
        st.write("- " + "\n- ".join(deps))

def show_tab_local_upload(key_prefix, name):
    nonce = get_nonce(key_prefix)

    def k(name: str) -> str:
        return f"{key_prefix}_{name}_{nonce}"

    st.markdown(f"### {get_text('conf_model', 'upload_local_header')}")
    st.caption(get_text("conf_model", "upload_local_caption"))

    upload_root = Path(".uploaded_models")
    upload_root.mkdir(exist_ok=True)

    def slugify(name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r"[^\w\-\.]+", "-", s)
        s = re.sub(r"-{2,}", "-", s).strip("-")
        return s or f"model-{int(time.time())}"

    def detect_fmt(files: list[str]) -> str:
        lower = [f.lower() for f in files]
        if any(f.endswith(".gguf") for f in lower):
            return "GGUF (llama.cpp)"

        has_config = any(f.endswith("config.json") for f in lower)
        has_weights = any(
            f.endswith(".safetensors") or f.endswith("pytorch_model.bin")
            for f in lower
        )
        has_tokenizer = any(
            f.endswith("tokenizer.json")
            or f.endswith("tokenizer.model")
            or f.endswith("vocab.json")
            for f in lower
        )
        if has_config and has_weights and has_tokenizer:
            return "Transformers (HF)"

        if any(f.endswith(".onnx") for f in lower):
            return "ONNX"

        return get_text("conf_model", "other_unknown")

    def required_by_format(fmt: str) -> list[str]:
        if fmt == "GGUF (llama.cpp)":
            return [".gguf"]
        if fmt == "Transformers (HF)":
            return [
                "config.json",
                "tokenizer.json|tokenizer.model|vocab.json",
                ".safetensors|pytorch_model.bin",
            ]
        if fmt == "ONNX":
            return ["*.onnx"]
        return []

    def check_missing(all_files: list[str], fmt: str) -> list[str]:
        if fmt == get_text("conf_model", "other_unknown"):
            return []
        present = set(f.lower() for f in all_files)
        missing = []
        for req in required_by_format(fmt):
            choices = [x.strip().lower() for x in req.split("|")]
            ok = False
            for ch in choices:
                if ch.startswith("*."):
                    ext = ch[1:]
                    if any(p.endswith(ext) for p in present):
                        ok = True;
                        break
                elif ch.startswith("."):
                    if any(p.endswith(ch) for p in present):
                        ok = True;
                        break
                else:
                    if ch in present or any(p.endswith("/" + ch) or p.endswith("\\" + ch) for p in present):
                        ok = True;
                        break
            if not ok:
                missing.append(req)
        return missing

    files = st.file_uploader(
        get_text("conf_model", "select_files"),
        accept_multiple_files=True,
        key=k("local_upload_files"),
    )

    model_name = st.text_input(f"{ICONS['Metadata']} {get_text('conf_model', 'model_name_folder')}",
                               key=k("local_upload_name"))
    if not model_name and files:
        model_name = files[0].name.split(".")[0]

    hard_validate = st.checkbox(get_text("conf_model", "hard_validation"), value=True,
                                key=k("local_upload_hard"))

    if st.button(f"{ICONS['Save']} {get_text('conf_model', 'save_local_btn')}", key=k("save_local_model")):
        if not files:
            st.warning(get_text("conf_model", "no_file_selected"))
            st.stop()

        folder = upload_root / slugify(model_name or "model")
        folder.mkdir(exist_ok=True, parents=True)
        saved_files = []

        for f in files:
            data = f.read()
            name = f.name
            lower = name.lower()

            try:
                if lower.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        zf.extractall(folder)
                        saved_files.extend(zf.namelist())
                        continue
                if lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
                    mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r"
                    with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as tf:
                        tf.extractall(folder)
                        saved_files.extend([m.name for m in tf.getmembers() if m.isfile()])
                        continue
            except Exception as e:
                st.warning(get_text("conf_model", "archivio", name=name, e=e))

            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(data)
            saved_files.append(name)

        all_files = [p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()]
        fmt = detect_fmt(all_files)
        missing = check_missing(all_files, fmt)

        if hard_validate and missing:
            shutil.rmtree(folder, ignore_errors=True)
            st.error(get_text("conf_model", "save_cancelled"))
            st.info(get_text("conf_model", "missing_files") + ", ".join(missing))
            st.stop()

        # Scrivi metadata
        meta = {
            "name": folder.name,
            "created_at": int(time.time()),
            "format": fmt,
            "files": all_files,
            "missing": missing,  # anche in modalità soft mostriamo cosa manca
        }
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        st.success(get_text("conf_model", "model_saved", name=folder.name, folder=folder))
        st.write(get_text("conf_model", "format_detected", fmt=fmt))
        if missing:
            st.warning(get_text("conf_model", "soft_validation_warning") + ", ".join(missing))

    # === Elenco modelli ===
    st.markdown(get_text("conf_model", "uploaded_models_header"))
    models = [d.name for d in upload_root.iterdir() if d.is_dir()]
    if not models:
        st.info(get_text("conf_model", "no_models_uploaded"))
    else:
        sel = st.selectbox(f"{ICONS['Select a Model']} " + get_text("conf_model", "select_model"), options=models, key=k("local_pick"))
        meta_file = upload_root / sel / "metadata.json"
        if meta_file.exists():
            # mostra anche “missing” se presente
            st.json(json.loads(meta_file.read_text(encoding="utf-8")))
        if st.button(get_text("conf_model", "delete_model"), key=k(f"del_{sel}")):
            shutil.rmtree(upload_root / sel)
            st.success(get_text("conf_model", "model_deleted", name=sel))

@st.cache_data(show_spinner=False)
def list_ollama_models(host: str = "http://localhost:11434"):
    r = requests.get(f"{host}/api/tags", timeout=5)
    r.raise_for_status()
    js = r.json()
    return js.get("models", [])

def pull_ollama(model_name: str, host: str = "http://localhost:11434"):
    with requests.post(
            f"{host}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=60,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            yield line  # ogni linea è un frammento JSON con status/progress

@st.cache_data(show_spinner=False)
def list_lmstudio_models(host: str = "http://localhost:1234"):
    r = requests.get(f"{host}/v1/models", timeout=5)
    r.raise_for_status()
    js = r.json()
    return js.get("data", [])

@st.cache_data(show_spinner=get_text("conf_model", "searching_hugging"))
def hf_search_models(task, author, search, sort, limit, token):
    HF_API = "https://huggingface.co/api/models"
    params = {"limit": limit, "full": 1}
    if task:   params["pipeline_tag"] = task
    if author: params["author"] = author
    if search: params["search"] = search
    if sort:   params["sort"] = sort

    headers = {"User-Agent": "streamlit-hf-explorer/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(HF_API, params=params, headers=headers, timeout=5)
    r.raise_for_status()
    data = r.json()
    results = []
    for m in data:
        results.append({
            "id": m.get("modelId") or m.get("id"),
            "author": m.get("author"),
            "pipeline_tag": m.get("pipeline_tag"),
            "downloads": m.get("downloads"),
            "likes": m.get("likes"),
            "tags": m.get("tags") or [],
            "private": bool(m.get("private")),
            "gated": bool(m.get("gated")),
            "lastModified": m.get("lastModified"),
        })

    return results, {"url": r.url, "status": r.status_code, "count": len(results)}

def hugging_face_tab():
    st.title(get_text("conf_model", "hf_explorer_title"))

    # --- FILTRI ---
    with st.form(key="formHF"):
        col1, col2, col3 = st.columns(3)
        with col1:
            task = st.selectbox(
                get_text("conf_model", "filter_task"),
                ["", "text-generation", "translation", "summarization",
                 "image-classification", "question-answering", "text2text-generation"],
                help=get_text("conf_model", "filter_task_help")
            )
            task = task or None

        with col2:
            author = st.text_input(get_text("conf_model", "author_org"), placeholder="es. google, meta, bigscience")
            author = author.strip() or None

        with col3:
            sort = st.selectbox(get_text("conf_model", "sort_by"), ["", "downloads", "likes", "lastModified"])
            sort = sort or None

        search = st.text_input(get_text("conf_model", "search_placeholder"), placeholder="es. llama, bert, gpt2")
        limit = st.slider(
            label=get_text("conf_model", "max_results"),
            min_value=1,
            max_value=100,
            value=5,  # default
            step=5,
            key="hf_limit"
        )
        use_token = st.text_input(f"{ICONS['Hugging Face Token']} " + get_text("conf_model", "hf_token_help"),
                                  type="password", value=get_HF_Token())
        submit = st.form_submit_button(f"{ICONS['Filtro']} " + get_text("conf_model", "search_btn"))

        if not author and submit:
            st_toast_temp(get_text("conf_model", "insert_author"))
        elif not use_token and submit:
            st_toast_temp(get_text("conf_model", "insert_author"))
        elif not author and not use_token and submit:
            st_toast_temp(get_text("conf_model", "insert_author_token"))
        elif submit and author and use_token:
            results, url = hf_search_models(task=task, author=author, search=search, sort=sort, limit=limit,
                                            token=use_token)
            if len(results) == 0:
                st.session_state['results_HF'] = []
                st.session_state['submit_HF'] = False
                st.session_state['token_HF'] = ''
                st_toast_temp(get_text("conf_model", "no_models_found_retry"))
            else:
                st_toast_temp(get_text("conf_model", "models_found_count", n=len(results)))
                st.session_state['results_HF'] = [res for res in results]
                st.session_state['submit_HF'] = True
                st.session_state['token_HF'] = use_token

def lmstudio_tab():
    st.title(get_text("conf_model", "lm_studio_title"))

    # --- Sezione A: elenco modelli esposti dal server OpenAI-compatible ---
    col1, col2 = st.columns(2)
    with col1:
        lm_host = st.text_input(f"{ICONS['Host']} " + get_text("conf_model", "host_lm_studio"), "http://localhost:1234",
                                help=get_text("conf_model", "lm_host_help"))
    with col2:
        lm_filter = st.text_input(f"{ICONS['Filtro']} " + get_text("conf_model", "filter_contains"), "", placeholder="es. qwen, mistral, llama")
    # lm_host lo stai già ottenendo da st.text_input(...)
    lmstudio_panel(host=lm_host, key='online_tab')

    if st.session_state['server_lmStudio'] is True:

        cols_srv = st.columns([3, 1])
        with cols_srv[0]:
            if st.button(f"{ICONS['Refresh']} " + get_text("conf_model", "refresh_cache"), help=get_text("conf_model", "refresh_cache_help")):
                st.cache_data.clear()

        try:
            data = list_lmstudio_models(lm_host)  # [{"id": "..."}]
            items = [{"id": d.get("id", "")} for d in data]
            f = _norm(lm_filter)
            if f:
                items = [x for x in items if f in _norm(x["id"])]

            st.caption(get_text("conf_model", "models_exposed", n=len(items)))
            for it in items:
                mid = it["id"]
                with st.container(border=True):
                    st.markdown(f"**{mid}**")
                    st.caption(get_text("conf_model", "models_exposed_help"))
        except requests.exceptions.ConnectionError:
            st.error(get_text("conf_model", "lms_connection_error"))
        except Exception as e:
            st.error(get_text("conf_model", "lms_error", e=e))

        # === Imposta come LLM attivo (LM Studio) ===
        st.markdown("---")
        st.subheader(get_text("conf_model", "set_active_lms"))
        chosen_lms = st.text_input(
            get_text("conf_model", "model_id_exposed"),
            placeholder="es. Meta-Llama-3.1-8B-Instruct-GGUF"
        )
        col_set, col_sp = st.columns([3, 5])
        with col_sp:
            empty = st.empty()
        with col_set:
            if st.button(get_text("conf_model", "use_lms_model")):
                if not chosen_lms.strip():
                    st.warning(get_text("conf_model", "insert_valid_id"))
                else:
                    st.session_state['llm'] = {
                        'backend': "LM Studio",
                        'model': chosen_lms.strip(),
                        'status': 'loaded',
                        'kwargs': {'host': lm_host}  # usa l'host letto sopra
                    }
                    # retro-compat
                    st.session_state['llm_backend'] = "LM Studio"
                    st.session_state['llm_model'] = chosen_lms.strip()
                    st.success(get_text("conf_model", "active_model", backend="LM Studio", sel=chosen_lms.strip()))
                    empty.success(get_text("conf_model", "active_model", backend="LM Studio", sel=chosen_lms.strip()))
                    st_toast_temp(get_text("conf_model", "active_model", backend="LM Studio", sel=chosen_lms.strip()), 'success')

        st.divider()

        # --- Sezione B: scarica/aggiorna modelli tramite CLI "lms get" ---
        st.subheader(f"{ICONS['Download Models']} " + get_text("conf_model", "download_lms_hub"))
        st.caption(get_text("conf_model", "download_lms_help"))

        with st.form(key="lms_get_form"):
            q = st.text_input(get_text("conf_model", "model_name_query"), placeholder="es. llama3.1:8b-instruct")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                no_confirm = st.checkbox(get_text("conf_model", "no_confirm"), value=True, help=get_text("conf_model", "no_confirm_help"))
            with c2:
                fresh = st.checkbox(get_text("conf_model", "force_install"), value=False, help=get_text("conf_model", "force_install_help"))
            submit_get = st.form_submit_button(f"{ICONS['Download Models']} " + get_text("conf_model", "download_with_lms"))

        if submit_get:
            if not q.strip():
                st.warning(get_text("conf_model", "insert_valid_query"))
            else:
                args = []
                if no_confirm: args.append("--no-confirm")
                if fresh: args.append("--force")

                log_area = st.empty()
                prog = st.progress(0)
                status = st.empty()

                st.info(get_text("conf_model", "executing_cmd", q=q, args=' '.join(args)))
                rc = None
                pct_seen = 0
                try:
                    gen = lms_get_stream(q.strip(), extra_args=args)
                    for line in gen:
                        status.write(line)
                        log_area.code(line, language="bash")
                        import re
                        m = re.search(r"(\d{1,3})\s*%", line)
                        if m:
                            pct = max(0, min(100, int(m.group(1))))
                            if pct != pct_seen:
                                prog.progress(pct)
                                pct_seen = pct
                    prog.progress(100)
                    rc = 0
                except Exception as e:
                    rc = 1
                    st.error(get_text("conf_model", "download_error", e=e))

                if rc == 0:
                    st.success(get_text("conf_model", "download_complete_lms"))
                else:
                    st.error(get_text("conf_model", "lms_exit_code", rc=rc))

        st.caption(get_text("conf_model", "lms_hint"))
    elif st.session_state.get('server_lmStudio') is False:
        st.warning(get_text("conf_model", "server_not_running"))

# ==============================================================
# 🦙 OLLAMA TAB — solo registry online + pull
# ==============================================================

def ollama_tab():
    st.title(get_text("conf_model", "ollama_registry_title"))

    colh1, colh2 = st.columns(2)
    with colh1:
        ollama_host = st.text_input(f"{ICONS['Host']} " + get_text("conf_model", "host_ollama"), "http://localhost:11434",
                                    help=get_text("conf_model", "ollama_host_help"))
    with colh2:
        reg_query = st.text_input(f"{ICONS['Filtro']} " + get_text("conf_model", "repo_filter"), "",
                                  placeholder="es. llama3, qwen2.5, mistral")
    ollama_panel(host=ollama_host, key='online_tab')
    if st.session_state['server_ollama'] is True:
        st.divider()
        st.subheader(get_text("conf_model", "registry_browse"))

        with st.expander(f"{ICONS['Filtro']} " + get_text("conf_model", "search_registry"), expanded=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                reg_filter = st.text_input(f"{ICONS['Filtro']} " + get_text("conf_model", "search_repo_contains"), reg_query or "",
                                           placeholder="es. llama3, mistral, qwen")
            with c2:
                refresh_reg = st.button(f"{ICONS['Refresh']} " + get_text("conf_model", "refresh_registry"))

            if refresh_reg:
                ollama_registry_catalog.clear()
                ollama_registry_tags.clear()

            catalog = ollama_registry_catalog()
            if isinstance(catalog, dict) and "error" in catalog:
                st.warning(catalog["error"])
                st.info(get_text("conf_model", "manual_pull_info"))
                repos = []
            else:
                repos = catalog or []

            fq = (reg_filter or "").strip().lower()
            if fq:
                repos = [r for r in repos if fq in r.lower()]

            st.caption(get_text("conf_model", "repos_in_registry", n=len(repos)))

            if repos:
                repo = st.selectbox(f"{ICONS['Select a Model']} " + get_text("conf_model", "select_repo"), repos, index=0,
                                    key="oll_reg_repo")
                tags = ollama_registry_tags(repo)
                if isinstance(tags, dict) and "error" in tags:
                    st.warning(tags["error"])
                    tags = []
                st.caption(get_text("conf_model", "tags_available", repo=repo, n=len(tags)))

                for tg in tags:
                    with st.container(border=True):
                        st.markdown(f"**{repo}:{tg}**")
                        cta1, cta2 = st.columns([1, 4])

                        with cta1:
                            do_pull = st.button(get_text("conf_model", "pull_btn"), key=f"pull_{repo}_{tg}")
                        with cta2:
                            st.caption(get_text("conf_model", "pull_help"))
                        if do_pull and st.session_state['server_ollama'] is True:
                            try:
                                with st.spinner(get_text("conf_model", "pulling_model", name=f"{repo}:{tg}")):
                                    url = f"{ollama_host}/api/pull"
                                    payload = {"name": f"{repo}:{tg}"}
                                    resp = requests.post(url, json=payload, stream=True, timeout=600)
                                    resp.raise_for_status()
                                    prog = st.progress(0)
                                    last_pct = 0
                                    for raw in resp.iter_lines(decode_unicode=True):
                                        if not raw:
                                            continue
                                        try:
                                            ev = json.loads(raw)
                                        except Exception:
                                            st.write(raw)
                                            continue
                                        if "status" in ev:
                                            st.write(ev["status"])
                                        if "completed" in ev and "total" in ev:
                                            try:
                                                comp = int(ev["completed"])
                                                total = int(ev["total"])
                                                pct = int(100 * comp / total) if total > 0 else last_pct
                                                pct = max(min(pct, 100), 0)
                                                if pct != last_pct:
                                                    prog.progress(pct)
                                                    last_pct = pct
                                            except Exception:
                                                pass
                                    prog.progress(100)
                                    st.success(get_text("conf_model", "pull_complete"))
                                    st.cache_data.clear()
                            except Exception as e:
                                st.error(get_text("conf_model", "pull_error", e=e))
                        elif do_pull and st.session_state['server_ollama'] is False:
                            st_toast_temp(get_text("conf_model", "server_not_running"), 'warning')
                            st.warning(get_text("conf_model", "server_not_running"))
        st.divider()

        # ====== SEZIONE PULL MANUALE ======
        st.subheader(get_text("conf_model", "manual_pull_header"))
        with st.form("ollama_manual_pull"):
            model_to_pull = st.text_input(get_text("conf_model", "remote_model_name"), placeholder="es. llama3:8b-instruct")
            submit_pull = st.form_submit_button(get_text("conf_model", "pull_from_registry"))
        if submit_pull and st.session_state['server_ollama'] is True:
            if not model_to_pull.strip():
                st.warning(get_text("conf_model", "insert_valid_tag"))
            else:
                try:
                    with st.spinner(get_text("conf_model", "pulling_model", name=model_to_pull)):
                        url = f"{ollama_host}/api/pull"
                        payload = {"name": model_to_pull.strip()}
                        resp = requests.post(url, json=payload, stream=True, timeout=600)
                        resp.raise_for_status()
                        prog = st.progress(0)
                        last_pct = 0
                        for raw in resp.iter_lines(decode_unicode=True):
                            if not raw:
                                continue
                            try:
                                ev = json.loads(raw)
                            except Exception:
                                st.write(raw)
                                continue
                            if "status" in ev:
                                st.write(ev["status"])
                            if "completed" in ev and "total" in ev:
                                try:
                                    comp = int(ev["completed"])
                                    total = int(ev["total"])
                                    pct = int(100 * comp / total) if total > 0 else last_pct
                                    pct = max(min(pct, 100), 0)
                                    if pct != last_pct:
                                        prog.progress(pct)
                                        last_pct = pct
                                except Exception:
                                    pass
                        prog.progress(100)
                        st.success(get_text("conf_model", "pull_complete"))
                        st.cache_data.clear()
                except Exception as e:
                    st.error(get_text("conf_model", "pull_error_short", e=e))
        elif st.session_state['server_ollama'] is False:
            st_toast_temp(get_text("conf_model", "server_not_running"), 'warning')
            st.warning(get_text("conf_model", "server_not_running"))
        st.markdown("---")
        st.subheader(get_text("conf_model", "set_active_ollama"))
        chosen_ol = st.text_input(
            get_text("conf_model", "ollama_model_name"),
            placeholder="es. llama3.1:8b-instruct"
        )
        col_set, col_sp = st.columns([3, 5])
        with col_sp:
            empty = st.empty()
        with col_set:
            load_ollama = st.button(get_text("conf_model", "use_ollama_model"))
            if load_ollama and st.session_state['server_ollama'] is True:
                if not chosen_ol.strip():
                    st.warning(get_text("conf_model", "insert_valid_ollama"))
                else:
                    st.session_state['llm'] = {
                        'backend': "Ollama",
                        'model': chosen_ol.strip(),
                        'status': 'loaded',
                        'kwargs': {'host': ollama_host}  # usa l'host letto sopra
                    }
                    # retro-compat
                    st.session_state['llm_backend'] = "Ollama"
                    st.session_state['llm_model'] = chosen_ol.strip()
                    st.success(get_text("conf_model", "active_model", backend="Ollama", sel=chosen_ol.strip()))
                    empty.success(get_text("conf_model", "active_model", backend="Ollama", sel=chosen_ol.strip()))
                    st_toast_temp(get_text("conf_model", "active_model", backend="Ollama", sel=chosen_ol.strip()), 'success')
            elif load_ollama and st.session_state['server_ollama'] is False:
                st_toast_temp(get_text("conf_model", "server_not_running"), 'warning')
                st.warning(get_text("conf_model", "server_not_running"))
    elif st.session_state.get('server_ollama') is False:
        st.warning(get_text("conf_model", "server_not_running"))
import html

def _darken(hex_color: str, pct: float = 0.12) -> str:
    # pct=0.12 => scurisce del 12%
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    r = max(0, int(r * (1 - pct)))
    g = max(0, int(g * (1 - pct)))
    b = max(0, int(b * (1 - pct)))
    return f"#{r:02x}{g:02x}{b:02x}"

def _best_text_color(bg_hex: str) -> str:
    # YIQ contrast: se lo sfondo è chiaro → testo scuro; altrimenti testo chiaro
    h = bg_hex.lstrip("#")
    r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    yiq = (r*299 + g*587 + b*114) / 1000
    return "#0b1220" if yiq >= 140 else "#F8FAFC"  # scuro vs bianco

def _style_for(hex_color: str) -> str:
    bg = _darken(hex_color, 0.10)        # appena più scuro del tuo pastello
    fg = _best_text_color(bg)            # testo leggibile
    return (
        f"background-color:{bg};"
        f"color:{fg};"
        f"padding:2px 6px;border-radius:6px;"
        f"box-shadow: inset 0 0 0 1px rgba(0,0,0,.08);"  # separazione sottile
        f"line-height:1.6;"
    )

LABEL_COLORS = {
    "PERSON": "#ffb3ba", "ORG": "#ffdfba", "GPE": "#ffffba", "LOC": "#baffc9",
    "DATE": "#bae1ff", "TIME": "#d5baff", "EVENT": "#ffc2e2", "NORP": "#aaffff",
    "CARDINAL": "#ffd6a5", "MONEY": "#bde0fe", "PERCENT": "#caffbf", "FAC": "#e2afff",
    "PRODUCT": "#ffd1dc", "LAW": "#e0ffe0", "LANGUAGE": "#f1f0ff"
}
FALLBACK = ["#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff", "#d5baff", "#ffc2e2", "#aaffff"]

# ==== OLLAMA REGISTRY HELPERS (online) ====
# Nota: sono endpoint "docker v2 style" su registry.ollama.ai
#       Se il registry cambia, la UI mostra un fallback senza bloccare l’app.
def spacy_tab():
    # ========== UI TESTA ==========
    st.title(get_text("conf_model", "spacy_title"))
    st.warning(get_text("conf_model", "spacy_warning"))
    # ========== BLOCCO A: SCEGLI SORGENTE MODELLO ==========
    st.subheader(get_text("conf_model", "choose_source"))
    src = st.radio(
        get_text("conf_model", "source_radio"),
        [get_text("conf_model", "common_models"), get_text("conf_model", "custom_package")],
        horizontal=True,
    )
    candidate = None
    show_after = False
    common = list(COMMON_SPACY_MODELS)

    if 'detailed_spacy' not in st.session_state:
        st.session_state['detailed_spacy'] = {}

    colh1, colh2, colh3 = st.columns([3, 2, 3])
    with colh1:
        filtro = st.text_input(f"{ICONS['Filtro']} " + get_text("conf_model", "filter_contains"), "",
                               placeholder="es. it_core, en_core, transformer")
    if src == get_text("conf_model", "common_models"):
        # lista filtrabile dei più comuni (anche se non installati)
        with colh3:
            if filtro:
                common = [m for m in COMMON_SPACY_MODELS if filtro.lower() in m.lower()]
            if not common:
                st.info(get_text("conf_model", "no_common_match"))
                return
            sel = st.selectbox(get_text("conf_model", "common_model_label"), common, index=0, key="spacy_common_pick")
            candidate = sel
            if candidate not in list(st.session_state['detailed_spacy'].keys()):
                st.session_state['detailed_spacy'] = {}
                st.session_state['detailed_spacy'][candidate] = model_details(candidate)

    else:
        with colh3:
            default_download = st.text_input(get_text("conf_model", "custom_package_input"), "it_core_news_sm",
                                             help=get_text("conf_model", "custom_package_help"))
        candidate = default_download.strip()
        if candidate not in list(st.session_state['detailed_spacy'].keys()):
            st.session_state['detailed_spacy'] = {}
            st.session_state['detailed_spacy'][candidate] = model_details(candidate)

    if not candidate:
        st.info(get_text("conf_model", "select_or_insert"))
        return

# ========== DETTAGLI (anche senza download) ==========
    st.markdown(get_text("conf_model", "details_no_download"))
    pre = st.session_state['detailed_spacy'][candidate]
    # tabella orizzontale dei principali
    base = [{
        get_text("conf_model", "model"): pre.get("model"),
        get_text("conf_model", "language"): f'{pre.get("language")} ({pre.get("lang_code")})',
        get_text("conf_model", "version"): pre.get("version") or "N/D",
        get_text("conf_model", "pipeline"): ", ".join(pre.get("pipeline") or []) or "—",
        get_text("conf_model", "tasks"): ", ".join(pre.get("tasks") or []) or "—",
        get_text("conf_model", "vectors"): pre.get("vectors") or "—",
        get_text("conf_model", "installed"): get_text("conf_model", "si") if pre.get("installed") else get_text("conf_model", "no"),
        get_text("conf_model", "size_hint"): pre.get("size_hint") or "N/D",
    }]
    st.dataframe(pd.DataFrame(base), hide_index=True, width='stretch')

    # download / update
    dl_cols = st.columns([3, 3])
    with dl_cols[0]:
        do_dl = st.button(f"{ICONS['Download Models']} " + get_text("conf_model", "download_update"), key="spacy_dl_go")
    with dl_cols[1]:
        show_after = st.toggle(get_text("conf_model", "show_details_after"), value=True)

    if do_dl:
        try:
            with st.spinner(get_text("conf_model", "downloading_candidate", candidate=candidate)):
                spacy.cli.download(candidate)
            st.success(get_text("conf_model", "model_installed", candidate=candidate))
            st.cache_data.clear()
        except SystemExit as se:
            code = int(getattr(se, "code", 1) or 1)
            if code == 0:
                st.success(get_text("conf_model", "model_installed", candidate=candidate))
                st.cache_data.clear()
            else:
                st.error(get_text("conf_model", "failed_exit_code", code=code))
        except Exception as e:
            st.error(get_text("conf_model", "download_error", e=e))

    # se installato (già prima o appena adesso), mostra dettagli completi
    if candidate is not None and show_after and (pre.get("installed") or is_package(candidate)):
        with st.expander(get_text("conf_model", "full_details_installed"), expanded=True):
            try:
                nlp = spacy.load(candidate)
                meta = getattr(nlp, "meta", {}) or {}
                pipe_names = list(nlp.pipe_names)
                vec_dim = int(getattr(nlp.vocab, "vectors_length", 0) or 0)
                vec_keys = int(getattr(getattr(nlp.vocab, "vectors", None), "n_keys", 0) or 0)
                full = {
                    get_text("conf_model", "model"): meta.get("name", candidate),
                    get_text("conf_model", "language"): meta.get("lang", getattr(nlp, "lang", "N/A")),
                    get_text("conf_model", "version"): meta.get("version", "N/A"),
                    get_text("conf_model", "pipeline"): ", ".join(pipe_names) or "—",
                    get_text("conf_model", "vectors_dim"): vec_dim,
                    get_text("conf_model", "vectors_keys"): vec_keys,
                    get_text("conf_model", "description"): meta.get("description", "—"),
                    get_text("conf_model", "spaCy_compatible"): meta.get("spacy_version", "—"),
                }
                st.dataframe(pd.DataFrame([full]), hide_index=True, width='stretch')
            except Exception as e:
                st.warning(get_text("conf_model", "cant_load_details", e=e))

def spacy_ner_load():
    candidate = list(st.session_state['detailed_spacy'].keys())[0]
    # ========== BLOCCO B: PROVA NER/POS ==========
    st.subheader(get_text("conf_model", "test_ner_pos"))
    sample = st.text_area(get_text("conf_model", "test_text"), get_text("conf_model", "test_text_default"), key="spacy_sample_text", disabled= True if len(list(st.session_state['detailed_spacy'].keys())) ==0 else False)
    go = st.button(get_text("conf_model", "run_analysis"), key="spacy_run_btn", disabled= True if len(list(st.session_state['detailed_spacy'].keys())) ==0 else False)

    if go:
        try:
            nlp = spacy.load(candidate)
            doc = nlp(sample)
        except Exception as e:
            st.error(get_text("conf_model", "error_loading_model", e=e))
            return

        tab1, tab2, tab3 = st.tabs([get_text("conf_model", "ner_highlighted"), get_text("conf_model", "pos_table"), get_text("conf_model", "json_raw")])

        # 1) NER evidenziato nel testo
        with tab1:
            ents = [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents]
            if ents:
                html_out = sample
                for ent in sorted(ents, key=lambda x: x["start"], reverse=True):
                    base = LABEL_COLORS.get(ent["label"], FALLBACK[hash(ent["label"]) % len(FALLBACK)])
                    style = _style_for(base)
                    # proteggiamo il testo (niente HTML iniettato)
                    text = html.escape(ent["text"])
                    span = (f"<span style='{style}' title='{ent['label']}'>{text}</span>")
                    html_out = html_out[: ent["start"]] + span + html_out[ent["end"]:]
                st.markdown(html_out, unsafe_allow_html=True)

                # legenda coerente
                legend_html = build_legend_html(ents, nlp)
                st.markdown(f"**{get_text('conf_model', 'legend')}**", unsafe_allow_html=True)
                st.markdown(legend_html, unsafe_allow_html=True)
            else:
                st.info(get_text("conf_model", "no_entities"))
        # 2) POS in DataFrame
        with tab2:
            pos_rows = [{"Token": t.text, "Lemma": t.lemma_, "POS": t.pos_, "Tag": t.tag_} for t in doc]
            st.dataframe(pd.DataFrame(pos_rows), hide_index=True, width='stretch')

        # 3) JSON raw (utile ma opzionale)
        with tab3:
            res = {
                "ents": ents,
                "pos": [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_} for t in doc],
                "sents": [s.text for s in doc.sents],
            }
            st.code(json.dumps(res, ensure_ascii=False, indent=2), language="json")

    col_set, col_sp = st.columns([4, 5])
    with col_sp:
        empty = st.empty()
    with col_set:
        if st.button(get_text("conf_model", "use_analysis_model"), disabled= True if len(list(st.session_state['detailed_spacy'].keys())) ==0 else False):
            st.session_state['spacy_model'] ={
                                        'model': candidate,
                                         'status': 'Load'
                                         }

            empty.success(get_text("conf_model", "analysis_model_chosen", candidate=candidate))
            st_toast_temp(get_text("conf_model", "analysis_model_chosen", candidate=candidate), 'success')

# --- utility: riassume la “sensibilità” per label usando la tua funzione
def summarize_sensitivity_by_label(ents, nlp):
    summary = {}
    for ent in ents:
        label = ent["label"]
        text = ent["text"]
        res = is_sensitive_column(text, nlp)
        s = summary.setdefault(label, {"count": 0, "sens_count": 0, "examples": []})
        s["count"] += 1
        if res.get("sensitive"):
            s["sens_count"] += 1
            if len(s["examples"]) < 2:
                s["examples"].append(text)
    return summary

def build_legend_html(ents, nlp):
    NER_LABEL_INFO = {
        "PERSON": f'{get_text("conf_model", "ner_label_person")}',
        "NORP": f'{get_text("conf_model", "ner_label_norp")}',
        "FAC": f'{get_text("conf_model", "ner_label_fac")}',
        "ORG": f'{get_text("conf_model", "ner_label_org")}',
        "GPE": f'{get_text("conf_model", "ner_label_gpe")}',
        "LOC": f'{get_text("conf_model", "ner_label_loc")}',
        "PRODUCT": f'{get_text("conf_model", "ner_label_product")}',
        "EVENT": f'{get_text("conf_model", "ner_label_event")}',
        "WORK_OF_ART": f'{get_text("conf_model", "ner_label_work_of_art")}',
        "LAW": f'{get_text("conf_model", "ner_label_law")}',
        "LANGUAGE": f'{get_text("conf_model", "ner_label_language")}',
        "DATE": f'{get_text("conf_model", "ner_label_date")}',
        "TIME": f'{get_text("conf_model", "ner_label_time")}',
        "PERCENT": f'{get_text("conf_model", "ner_label_percent")}',
        "MONEY": f'{get_text("conf_model", "ner_label_money")}',
        "QUANTITY": f'{get_text("conf_model", "ner_label_quantity")}',
        "ORDINAL": f'{get_text("conf_model", "ner_label_ordinal")}',
        "CARDINAL": f'{get_text("conf_model", "ner_label_cardinal")}',
    }
    labs = sorted({e["label"] for e in ents})
    sens = summarize_sensitivity_by_label(ents, nlp)
    items = []
    for l in labs:
        base = LABEL_COLORS.get(l, FALLBACK[hash(l) % len(FALLBACK)])
        style = _style_for(base)  # già definita prima per contrasto
        desc = NER_LABEL_INFO.get(l, "spaCy entity")
        stats = sens.get(l, {"count": 0, "sens_count": 0, "examples": []})
        if stats["sens_count"] > 0:
            tag = f"🔒 {get_text("conf_model", "sensitive")} ({stats['sens_count']}/{stats['count']})"
        else:
            tag = f"🟢 {get_text("conf_model", "non_sensitive")} ({stats['count']})"
        examples = f" — es.: {', '.join(stats['examples'])}" if stats["examples"] else ""
        items.append(
            f"<div style='margin:4px 0;'>"
            f"<span style='{style}'>{l}</span>"
            f"<span style='opacity:.78;margin-left:8px'>{desc} · {tag}{examples}</span>"
            f"</div>"
        )
    return "".join(items)

@st.cache_data(show_spinner=get_text("conf_model", "searching_ollama"), ttl=3600)
def ollama_registry_catalog(limit: int = 2000) -> list[str] | dict:
    try:
        url = "https://registry.ollama.ai/v2/library/_catalog"
        resp = requests.get(url, params={"n": limit}, timeout=10)
        resp.raise_for_status()
        js = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        repos = js.get("repositories", [])
        # Alcuni registry ritornano path 'library/<name>': normalizziamo
        repos = [r.split("/", 1)[-1] if "/" in r else r for r in repos]
        return sorted(set(repos))
    except Exception as e:
        return {"error": get_text("conf_model", "ollama_registry_error", e=e)}

@st.cache_data(show_spinner=get_text("conf_model", "fetching_tags"), ttl=3600)
def ollama_registry_tags(model: str, limit: int = 200) -> list[str] | dict:
    try:
        url = f"https://registry.ollama.ai/v2/library/{model}/tags/list"
        resp = requests.get(url, params={"n": limit}, timeout=10)
        resp.raise_for_status()
        js = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return js.get("tags", []) or []
    except Exception as e:
        return {"error": get_text("conf_model", "tags_error", model=model, e=e)}

def _dict_to_table_rows(d: dict, section: str | None = None):
    rows = []
    for k, v in d.items():
        if isinstance(v, dict):
            # appiattisci di un livello (es. "Spiegazioni")
            for k2, v2 in v.items():
                rows.append({
                    get_text("conf_model", "section"): k if section is None else f"{section} / {k}",
                    get_text("conf_model", "field"): k2,
                    get_text("conf_model", "value"): str(v2),
                })
        else:
            rows.append({
                get_text("conf_model", "section"): section or get_text("conf_model", "overview"),
                get_text("conf_model", "field"): k,
                get_text("conf_model", "value"): str(v)
            })
    return rows

def _is_complex_value(v):
    return isinstance(v, (dict, list, tuple, set, pd.DataFrame))

def _is_empty_value(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str):
        v_clean = v.strip().lower()
        if v_clean in {"", "none", "nan", "null", "_", '—'}:
            return True
    return False

def _clean_campo_names(df: pd.DataFrame) -> pd.DataFrame:
    import re
    campi = df[get_text("conf_model", "field")].tolist()
    base_names = [re.sub(f"\s*\({get_text("conf_model", "by_name")}\)\s*$", "", c).strip() for c in campi]
    df[get_text("conf_model", "campo_base")] = base_names

    duplicates = df[get_text("conf_model", "campo_base")].duplicated(keep=False)

    df_clean = df[~((duplicates) & (df[get_text("conf_model", "field")].str.contains(f"\({get_text("conf_model", "by_name")}\)", case=False)))]

    df_clean.loc[df_clean[get_text("conf_model", "field")].str.contains(f"\({get_text("conf_model", "by_name")}\)", case=False), get_text("conf_model", "field")] = \
        df_clean.loc[df_clean[get_text("conf_model", "field")].str.contains(f"\({get_text("conf_model", "by_name")}\)", case=False), get_text("conf_model", "campo_base")]

    df_clean = df_clean.drop(columns=[get_text("conf_model", "campo_base")])

    return df_clean.reset_index(drop=True)
