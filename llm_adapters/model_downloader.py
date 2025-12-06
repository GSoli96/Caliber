import inspect
import os
import random
import socket
import threading
import time

import streamlit as st
from huggingface_hub import snapshot_download, login
from transformers import pipeline

from utils.translations import get_text

# Tweak utili
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # downloader Rust
os.environ.setdefault("HF_HUB_TIMEOUT", "60")  # timeout singola richiesta (s)


# NOTA: L'inizializzazione di 'hf_dl' DEVE avvenire in app.py


def download_model_HF(
        model_id: str,
        task: str = "text-generation",
        token: str | None = None,
        allow_patterns=None,
        refresh_sec: float = 0.5,
):
    """
    Scarica un modello Hugging Face mostrando barra progresso + pulsante annulla.
    Tutta la UI è dentro questa funzione. Restituisce la pipeline quando pronta o None.
    """
    # Ottieni lo stato. Assicurati che sia inizializzato in app.py!
    if "hf_dl" not in st.session_state:
        st.error(get_text("llm_adapters", "dl_state_error"))
        return None

    state = st.session_state.hf_dl

    # --- MODIFICA CHIAVE 1: Controllo coerenza model_id ---
    # Se lo stato NON appartiene a questo modello, non fare nulla.
    # La UI in conf_model.py deciderà quando resettare lo stato (al click).
    # Qui ci limitiamo a non mostrare UI se il modello non corrisponde.
    if state.get("model_id") and state.get("model_id") != model_id:
        return None

    # Se state.model_id è None (appena avviato) E
    # non stiamo avviando un thread (blocco "if not state['running']..."),
    # allora stiamo per avviarne uno, va bene.

    # ---------- util locali ----------
    def _supports_progress_callback() -> bool:
        try:
            return "progress_callback" in inspect.signature(snapshot_download).parameters
        except Exception:
            return False

    def _dns_ok(host="huggingface.co", timeout_s=3.0) -> bool:
        try:
            socket.setdefaulttimeout(timeout_s)
            socket.gethostbyname(host)
            return True
        except socket.gaierror:
            return False
        except Exception:
            return True

    def _mb(x):
        return f"{(x or 0) / 1e6:.1f} MB"

    def _sleep_backoff(i, base=1.5, cap=12.0):
        time.sleep(min(cap, base ** i) * (0.6 + random.random() * 0.8))

    # ---------- worker in background ----------
    # --- MODIFICA CHIAVE ---
    # Il worker ora accetta il dizionario di stato 's' come argomento
    def _worker(s: dict):
        # --- MODIFICA CHIAVE ---
        # NON usiamo più st.session_state.hf_dl qui dentro. Usiamo 's'.
        # s = st.session_state.hf_dl  <- RIMOSSO

        s.update({
            "running": True, "stop": False, "progress": 0, "bytes": 0, "total": 0,
            "retries": 0, "note": "", "error": None, "local_dir": None,
            "started_at": time.time(),
            "pipe": None,
            "thread": threading.current_thread(),
            # model_id viene settato dal chiamante prima di avviare il thread
        })

        def _cb(done, total):
            if s["stop"]:
                raise KeyboardInterrupt("Interrotto dall’utente")
            s["bytes"] = done or 0
            s["total"] = total or 0
            if total > 0:
                s["progress"] = int(done * 100 / total)
            else:
                s["progress"] = 0  # Evita division by zero

        # login se repo privato
        if token:
            try:
                login(token=token, write_permission=False, add_to_git_credential=False)
            except Exception:
                pass

        if not _dns_ok():
            s["error"] = get_text("llm_adapters", "dl_dns_error")
            s["running"] = False
            return

        kwargs = dict(repo_id=model_id, token=token or None)
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        if _supports_progress_callback():
            kwargs["progress_callback"] = _cb

        try:
            # retry con backoff per errori di rete
            for attempt in range(s["max_retries"] + 1):
                if s["stop"]:
                    raise KeyboardInterrupt("Annullato prima dell’avvio del tentativo")
                try:
                    s["note"] = get_text("llm_adapters", "dl_attempt", attempt=attempt + 1, total=s['max_retries'] + 1)
                    local_dir = snapshot_download(**kwargs)
                    if s["stop"]:
                        raise KeyboardInterrupt("Annullato")
                    s["local_dir"] = local_dir
                    s["note"] = get_text("llm_adapters", "dl_complete_waiting")
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    net_like = isinstance(e, (TimeoutError, socket.timeout, ConnectionError, OSError))
                    try:
                        import requests
                        if isinstance(e, requests.exceptions.RequestException):
                            net_like = True
                    except ImportError:
                        pass

                    if net_like and attempt < s["max_retries"]:
                        s["retries"] = attempt + 1
                        s["note"] = get_text("llm_adapters", "dl_network_retry", error_type=type(e).__name__, error_msg=str(e)[:120], retry=s['retries'], max_retries=s['max_retries'])
                        _sleep_backoff(attempt)
                        continue
                    else:
                        s["note"] = get_text("llm_adapters", "dl_check_cache")
                        try:
                            local_dir_off = snapshot_download(
                                **{k: v for k, v in kwargs.items() if k != "progress_callback"}
                            )
                            s["local_dir"] = local_dir_off
                            s["note"] = get_text("llm_adapters", "dl_cached")
                            break
                        except Exception as e_off:
                            s["error"] = get_text("llm_adapters", "dl_failed", error_type=type(e).__name__, error=e) + "\n" + \
                                         f"Offline fallback assente: {type(e_off).__name__}: {e_off}"
                            s["running"] = False
                            return

            if not s.get("local_dir"):
                s["error"] = get_text("llm_adapters", "dl_no_directory")
                s["running"] = False
                return

        except KeyboardInterrupt:
            s["error"] = get_text("llm_adapters", "dl_cancelled")
        except Exception as e:
            s["error"] = get_text("llm_adapters", "dl_unexpected_error", error_type=type(e).__name__, error=e)
        finally:
            s["running"] = False

    # ---------- UI + orchestration ----------
    # Avvio thread se necessario (solo se non c'è nulla in corso)
    if (
            not state["running"] and
            state["pipe"] is None and
            state["local_dir"] is None and
            state["error"] is None
    ):
        # Questo blocco ora viene eseguito solo se l'utente ha cliccato "Load"
        # e conf_model.py ha resettato lo stato.

        st.success(get_text("llm_adapters", "dl_started"))

        # --- MODIFICA CHIAVE 2: Salva il model_id nello stato ---
        state["model_id"] = model_id

        state["thread"] = threading.Thread(target=_worker, args=(state,), daemon=True)
        state["thread"].start()
        time.sleep(0.1)
        st.rerun()

    # --- MODIFICA CHIAVE 3: Spostato st.info qui ---
    # Mostra l'info solo se il download è effettivamente per questo modello
    st.info(get_text("llm_adapters", "dl_model_info", model_id=model_id, task=task))

    # UI in corso (DOWNLOAD)
    if state["running"]:
        if state["total"]:
            pct = max(0, min(100, state["progress"]))
            frac = pct / 100
            text = f"{_mb(state['bytes'])}/{_mb(state['total'])}  ({pct}%)"
        else:
            frac = 0.0
            text = state["note"] or get_text("llm_adapters", "dl_downloading")

        st.progress(frac, text=text)
        if state["note"]:
            st.caption(state["note"])

        if st.button(get_text("llm_adapters", "dl_btn_cancel")):
            state["stop"] = True
            st.warning(get_text("llm_adapters", "dl_cancel_sent"))

        time.sleep(refresh_sec)
        st.rerun()
        return None

    # UI: esito (ERRORE)
    if state["error"]:
        st.error(f"❌ {state['error']}")
        if state["note"]:
            st.caption(state["note"])
        if st.button(get_text("llm_adapters", "dl_btn_retry")):
            # Reset COMPLETO dello stato
            state.update({
                "running": False, "stop": False, "progress": 0, "bytes": 0,
                "total": 0, "retries": 0, "max_retries": state.get("max_retries", 4),
                "note": "", "error": None, "local_dir": None,
                "started_at": None, "pipe": None, "thread": None,
                "model_id": None,  # <-- MODIFICA CHIAVE 4a
            })
            # --- MODIFICA CHIAVE 4b: Riassegna il target ---
            state["model_id"] = model_id
            state["thread"] = threading.Thread(target=_worker, args=(state,), daemon=True)
            state["thread"].start()
            st.rerun()
        return None

    # UI: esito (DOWNLOAD COMPLETATO, ORA CREO LA PIPELINE)
    if state["local_dir"] and state["pipe"] is None:
        try:
            with st.spinner(get_text("llm_adapters", "dl_creating_pipeline", path=state['local_dir'])):
                state["pipe"] = pipeline(task, model=state["local_dir"], token=token or None)
            st.rerun()
        except Exception as e:
            st.error(get_text("llm_adapters", "dl_pipeline_error", error=e))
            state["error"] = get_text("llm_adapters", "dl_pipeline_fatal", error=e)
            return None

    # UI: esito (PIPELINE PRONTA)
    if state["pipe"] is not None:
        st.success(get_text("llm_adapters", "dl_model_ready", path=state['local_dir']))
        if state["started_at"] and state["bytes"]:
            dt = max(1e-6, time.time() - state["started_at"])
            speed = state["bytes"] / dt / 1e6  # MB/s
            st.caption(get_text("llm_adapters", "dl_complete_stats", time=dt, speed=speed))
        return state["pipe"]

    return None