# -*- coding: utf-8 -*-
# llm_adapters/ollama_adapter.py
import os, shutil, subprocess, json, time
from typing import Optional
import re
import pandas as pd

from GUI.message_gui import st_toast_temp
from utils.translations import get_text

try:
    import streamlit as st
except Exception:
    st = None

# ====== HTTP base (per ping e generate) ======
import requests
import humanize

DEFAULT_HOST = "http://localhost:11434"
API = lambda host: f"{host.rstrip('/')}/api"

from typing import Tuple

@st.cache_data(ttl=5, show_spinner=get_text("llm_adapters", "ollama_verify_status"))
def get_ollama_status(host: str) -> Tuple[bool, int]:
    """
    Controlla lo stato del server Ollama e conta i modelli.
    Questa funzione verrà eseguita al massimo una volta ogni 5 secondi.
    """
    try:
        r = requests.get(f"{API(host)}/tags", timeout=2)
        if r.status_code == 200:
            count = len((r.json() or {}).get("models", []))
            return True, count  # (online, model_count)
    except Exception:
        pass  # La connessione fallisce, ecc.
    return False, 0  # (offline, model_count)

@st.cache_data(ttl=10, show_spinner=False)
def get_ollama_pid(port: int = 11434) -> Optional[int]:
    """
    Trova il PID del server (usa la cache per 10s).
    """
    return _get_pid() or _discover_pid_on_port(port)

def _which_ollama() -> Optional[str]:
    """Trova l'eseguibile 'ollama' su Windows/macOS/Linux."""
    for name in ("ollama", "ollama.exe", "ollama.cmd", "ollama.bat"):
        p = shutil.which(name)
        if p:
            return p
    # percorsi comuni aggiuntivi (best-effort)
    candidates = []
    if os.name == "nt":
        candidates += [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe"),
        ]
    else:
        candidates += ["/usr/local/bin/ollama", "/usr/bin/ollama"]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None

def _cmd_str(parts: list[str]) -> str:
    try:
        import shlex
        return shlex.join(parts)
    except Exception:
        return " ".join(parts)

def _run(cmd: list[str], timeout: int | None = 15):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
        return {
            "ok": p.returncode == 0, "code": p.returncode,
            "stdout": p.stdout or "", "stderr": p.stderr or "",
            "cmd_str": _cmd_str(cmd), "dur_s": round(time.time()-t0,3),
        }
    except FileNotFoundError:
        return {"ok": False, "code": 127, "stdout": "", "stderr": get_text("llm_adapters", "ollama_file_not_found"),
                "cmd_str": _cmd_str(cmd), "dur_s": round(time.time()-t0,3)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": 124, "stdout": "", "stderr": get_text("llm_adapters", "ollama_timeout"),
                "cmd_str": _cmd_str(cmd), "dur_s": round(time.time()-t0,3)}
    except Exception as e:
        return {"ok": False, "code": 1, "stdout": "", "stderr": get_text("llm_adapters", "ollama_error_generic", e=e),
                "cmd_str": _cmd_str(cmd), "dur_s": round(time.time()-t0,3)}

# ====== CLI wrappers ======

def _cli(args: list[str]):
    ollama = _which_ollama()
    if not ollama:
        return {"ok": False, "code": 127, "stdout": "", "stderr": get_text("llm_adapters", "ollama_cli_not_found_path"),
                "cmd_str": "ollama list"}
    cmd = (["cmd", "/c", ollama, "list"] if os.name == "nt" and ollama.lower().endswith((".cmd", ".bat")) else [ollama,"list"])
    return _run(cmd, timeout=20)

def cli_list_raw():
    """`ollama list` (testuale)."""
    return _cli(["list"])

def cli_show_json(model: str):
    """`ollama show --format json <model>`."""
    return _cli(["show", "--format", "json", model])

def cli_ps_raw():
    """`ollama ps` per processi di generazione caricati in RAM."""
    return _cli(["ps"])

# ====== Server background (pattern come lmstudio_adapter) ======
# Memorizziamo un PID se avviamo noi il server con `ollama serve`

_BG = {}
def _set_pid(pid: int | None):
    if st is not None:
        st.session_state["ollama_server_pid"] = pid
    else:
        _BG["pid"] = pid

def _get_pid() -> Optional[int]:
    if st is not None:
        return st.session_state.get("ollama_server_pid")
    return _BG.get("pid")

def _pid_by_port_win(port: int) -> Optional[int]:
    try:
        out = subprocess.check_output(["cmd","/c",f"netstat -ano | findstr :{port}"], text=True, timeout=5)
        # righe tipo: TCP    0.0.0.0:11434   0.0.0.0:0   LISTENING   12345
        for ln in out.splitlines():
            ln = ln.strip()
            if "LISTENING" in ln:
                pid = int(ln.split()[-1])
                return pid
    except Exception:
        pass
    return None

def _pid_by_port_nix(port: int) -> Optional[int]:
    for cmd in (["lsof","-i",f":{port}","-t"], ["sh","-lc", f"ss -lptn 'sport = :{port}' | awk '{{print $NF}}' | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p' | head -n1"]):
        try:
            out = subprocess.check_output(cmd, text=True, timeout=5)
            out = out.strip()
            if out:
                return int(out.splitlines()[0])
        except Exception:
            continue
    return None

def _discover_pid_on_port(port: int=11434) -> Optional[int]:
    return _pid_by_port_win(port) if os.name=="nt" else _pid_by_port_nix(port)

def is_online(host: str = DEFAULT_HOST) -> bool:
    try:
        r = requests.get(f"{API(host)}/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def start_server_background(host: str = DEFAULT_HOST):
    """
    Avvia `ollama serve` in background e ritorna subito.
    Se il server risponde già su HTTP, non fa nulla.
    """
    if is_online(host):
        return {"ok": True, "msg": get_text("llm_adapters", "ollama_server_online"), "pid": _discover_pid_on_port(11434)}

    info = _cli(["serve"])
    # `ollama serve` è un processo long-running → dobbiamo staccarlo.
    # Rilanciamo *detached* perché _cli usa run() bloccante:
    oll = _which_ollama()
    if not oll:
        return {"ok": False, "msg": get_text("llm_adapters", "ollama_cli_not_found")}
    cmd = [oll, "serve"]
    if os.name == "nt":
        creationflags = 0x00000200 | 0x00000008  # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
        p = subprocess.Popen(cmd, creationflags=creationflags, close_fds=True)
    else:
        p = subprocess.Popen(cmd, start_new_session=True, close_fds=True)
    _set_pid(p.pid)
    return {"ok": True, "pid": p.pid, "cmd": _cmd_str(cmd)}

def stop_server_background(host: str = DEFAULT_HOST):
    """
    Tenta di spegnere il server lanciato da noi (PID memorizzato).
    In assenza di PID, prova a scovare il PID che ascolta su 11434 e ucciderlo (best-effort).
    Nota: `ollama stop` ferma i *modelli* in esecuzione, non il server.
    """
    pid = _get_pid() or _discover_pid_on_port(11434)
    if not pid:
        return {"ok": False, "msg": get_text("llm_adapters", "ollama_pid_not_found")}
    try:
        if os.name == "nt":
            subprocess.run(["taskkill","/PID",str(pid),"/T","/F"], capture_output=True, text=True, timeout=10)
        else:
            os.kill(pid, 15)  # SIGTERM
        _set_pid(None)
        return {"ok": True, "msg": get_text("llm_adapters", "ollama_pid_terminated", pid=pid)}
    except Exception as e:
        return {"ok": False, "msg": get_text("llm_adapters", "ollama_stop_error", pid=pid, e=e)}

# ====== Funzioni richieste dall’infrastruttura ======

def list_models(host: str | None = None, filter: str | None = None):
    """
    Elenco modelli tramite CLI `ollama list`. Fallback a HTTP /api/tags.
    Ritorna una lista di stringhe (nomi modello).
    """
    info = cli_list_raw()
    models: list[str] = []
    if not info["ok"]:
        return {"error": f"{info['stderr']} (code={info['code']}) — cmd: {info['cmd_str']}"}
    if info.get("ok"):
        out = info["stdout"].strip().splitlines()
        # salta header "NAME ID SIZE MODIFIED"
        for ln in out:
            s = ln.strip()
            if not s or s.lower().startswith("name ") or s.lower().startswith("repository "):
                continue
            # la prima colonna è il nome (fino a due+ spazi)
            parts = re.split(r"\s{2,}", s)
            if parts:
                models.append(parts[0].strip())
    else:
        # Fallback HTTP
        try:
            r = requests.get(f"{API(host or DEFAULT_HOST)}/tags", timeout=5)
            r.raise_for_status()
            js = r.json()
            models = [m.get("name","").strip() for m in js.get("models", []) if m.get("name")]
        except Exception as e:
            return {"error": get_text("llm_adapters", "ollama_cli_fallback_failed", stderr=info.get('stderr') or 'CLI failed', error=e)}

    if filter:
        f = filter.lower().strip()
        models = [m for m in models if f in m.lower()]
    return models

def get_model_details(model_name: str, host: str | None = None):
    """
    Dettagli modello via CLI (`ollama show --format json`), con fallback HTTP /api/show.
    Ritorna dict strutturato (compatibile con tabella dettagli in conf_model).
    """
    info = cli_show_json(model_name)
    data = None
    if info.get("ok"):
        try:
            data = json.loads(info["stdout"] or "{}")
        except Exception:
            data = None
    if data is None:
        try:
            r = requests.post(f"{API(host or DEFAULT_HOST)}/show", json={"name": model_name}, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return {"error": get_text("llm_adapters", "ollama_details_fail", e=e)}

    # Normalizzazione campi (robusta su versioni diverse)
    det = data.get("details", {}) if isinstance(data, dict) else {}
    families = det.get("families") or det.get("family") or []
    if isinstance(families, str): families = [families]

    size_bytes = data.get("size") or data.get("bytes") or 0
    params = data.get("parameters") or det.get("parameter_size") or det.get("parameters") or "—"
    quant = det.get("quantization") or det.get("quantize") or "—"
    arch  = det.get("arch") or det.get("architecture") or "—"

    out = {
        get_text("llm_adapters", "ollama_det_name"): str(data.get("name", model_name)),
        get_text("llm_adapters", "ollama_det_family"): ", ".join(families) if families else "—",  # Già str
        get_text("llm_adapters", "ollama_det_disk"): humanize.naturalsize(size_bytes) if size_bytes else "—",  # Già str
        get_text("llm_adapters", "ollama_det_params"): str(params),  # Cast a str
        get_text("llm_adapters", "ollama_det_arch"): str(arch),  # Cast a str
        get_text("llm_adapters", "ollama_det_quant"): str(quant),  # Cast a str
        get_text("llm_adapters", "ollama_det_modelfile"): str(data.get("modelfile") or "—"),  # Cast a str
        "Raw": data,  # Questo rimane un dict, gestito correttamente da conf_model.py
    }
    return out

def generate(prompt: str, model_name: str, max_tokens: int = 1024, host: str = DEFAULT_HOST):
    """
    Generazione via HTTP (API ufficiale), così è rapida e portabile.
    """
    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": int(max_tokens)}
        }
        r = requests.post(f"{API(host)}/generate", json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        return j.get("response", "")
    except Exception as e:
        return get_text("llm_adapters", "ollama_http_error", e=e)

def run_server_ollama(host: str = DEFAULT_HOST, key: str = "ollama_panel"):
    res = start_server_background(host=host)
    if res.get("ok"):
        st.session_state['server_ollama'] = True
        st_toast_temp(get_text("llm_adapters", "ollama_server_started"), 'success')
        if res.get("cmd"):
            st.code(res["cmd"], language="bash")

        get_ollama_status.clear()
        get_ollama_pid.clear()
        st.rerun()
    else:
        st.session_state['server_ollama'] = False
        st_toast_temp(res.get("msg", get_text("llm_adapters", "ollama_server_error")), 'error')

def ollama_panel(host: str = DEFAULT_HOST, key: str = "ollama_panel"):
    if st is None:
        raise RuntimeError(get_text("llm_adapters", "streamlit_not_avail"))

    st.subheader(get_text("llm_adapters", "ollama_title"))
    init_key = f"{key}_initialized"

    if init_key not in st.session_state:
        online = False
        count = 0
        pid = None
        st.session_state[init_key] = True
    else:
        try:
            online, count = get_ollama_status(host)
            pid = get_ollama_pid(11434)
        except Exception as e:
            st.error(get_text("llm_adapters", "ollama_connection_error", e=e))
            online, count, pid = False, 0, None

    st.session_state['server_ollama'] = online

    # Pulsanti Start / Stop
    b1, b2, b3 = st.columns([3, 2, 3])

    with b1:
        if st.button(get_text("llm_adapters", "ollama_start_btn"), key=key + "_start"):
            res = start_server_background(host=host)
            if res.get("ok"):
                st.session_state['server_ollama'] = True
                st_toast_temp(get_text("llm_adapters", "ollama_server_started"), 'success')
                if res.get("cmd"):
                    st.code(res["cmd"], language="bash")

                get_ollama_status.clear()
                get_ollama_pid.clear()
                st.rerun()
            else:
                st.session_state['server_ollama'] = False
                st_toast_temp(res.get("msg", get_text("llm_adapters", "ollama_server_error")), 'error')

    with b3:
        if st.button(get_text("llm_adapters", "ollama_stop_btn"), key=key + "_stop"):
            res = stop_server_background(host=host)
            st.toast(res.get("msg", "—"))

            if res.get("ok"):
                st.session_state['server_ollama'] = False
                st_toast_temp(get_text("llm_adapters", "ollama_stop_toast"), 'warning')
                get_ollama_status.clear()
                get_ollama_pid.clear()
                st.rerun()
            else:
                st.session_state['server_ollama'] = True
                st_toast_temp(res["msg"], 'warning')

            if hasattr(st, "cache_data"):
                st.cache_data.clear()

    # Stato server
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.metric(
            label=get_text("llm_adapters", "http_server_label"),
            value=get_text("llm_adapters", "online") if online else get_text("llm_adapters", "offline")
        )
    with c2:
        st.metric(get_text("llm_adapters", "ollama_models_count"), count)
    with c3:
        st.caption(get_text("llm_adapters", "endpoint", url=f"{API(host)}/tags"))

    # Pulsanti elenco modelli & processi
    col1, col2, col3 = st.columns([3, 2, 3])

    with col1:
        ollama_list = st.button(get_text("llm_adapters", "ollama_list_btn"), key=key + "_list")

    with col3:
        button_Ps = st.button(get_text("llm_adapters", "ollama_ps_btn"), key=key + "_ps")

    # --- LIST MODELS ---
    if ollama_list:
        if online:
            info = cli_list_raw()

            stdout = (info.get("stdout") or "").strip()
            stderr = (info.get("stderr") or "").strip()

            primary, primary_label = (
                (stdout, get_text("llm_adapters", "stdout")) if stdout else (stderr, get_text("llm_adapters", "stderr_fallback"))
            )

            with st.expander(primary_label):
                st.code(primary or "<vuoto>", language="bash")

            if stdout and stderr:
                with st.expander(get_text("llm_adapters", "stderr") if primary_label == get_text("llm_adapters", "stdout") else get_text("llm_adapters", "stdout")):
                    st.code(stderr if primary_label == get_text("llm_adapters", "stdout") else stdout, language="bash")

            st.success(get_text("llm_adapters", "ollama_cmd_executed") if info.get("ok") else get_text("llm_adapters", "ollama_cmd_failed"))

            if info.get("ok") and primary:
                parsed = parse_ollama_list(primary)

                c1, c2 = st.columns(2)
                with c1:
                    st.metric(get_text("llm_adapters", "ollama_models_count"), parsed["count"])
                with c2:
                    st.metric(get_text("llm_adapters", "ollama_total_size"), parsed["total_size"])

                if parsed["rows"]:
                    df = pd.DataFrame(
                        parsed["rows"],
                        columns=[get_text("llm_adapters", "ollama_col_name"), get_text("llm_adapters", "ollama_col_id"), get_text("llm_adapters", "ollama_col_size"), get_text("llm_adapters", "ollama_col_modified")]
                    )
                    st.dataframe(df, width='stretch', hide_index=True)
                else:
                    st.info(get_text("llm_adapters", "ollama_no_output"))
        else:
            st_toast_temp(get_text("llm_adapters", "ollama_server_not_running"), 'warning')

    # --- RUNNING MODELS ---
    if button_Ps:
        if online:
            info = cli_ps_raw()
            with st.expander(get_text("llm_adapters", "ollama_ps_title")):
                st.code(info.get("stdout") or "<vuoto>", language="bash")
        else:
            st_toast_temp(get_text("llm_adapters", "ollama_server_not_running"), 'warning')

def _parse_size_bytes(s: str) -> int:
    if not s: return 0
    s = s.strip().replace(",", ".")
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGT]?i?B)?\s*$", s, re.I)
    if not m: return 0
    val = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    factor = {
        "B": 1,
        "KB": 1024, "KIB": 1024,
        "MB": 1024**2, "MIB": 1024**2,
        "GB": 1024**3, "GIB": 1024**3,
        "TB": 1024**4, "TIB": 1024**4,
    }.get(unit, 1)
    return int(val * factor)

def _format_bytes(n: int) -> str:
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit=="B" else f"{n:.2f} {unit}"
        n /= 1024.0

def parse_ollama_list(text: str):
    """
    Parsifica 'ollama list' (tabellare):
    NAME  ID  SIZE  MODIFIED
    llama3.1:latest  46e0c10c039e  4.9 GB  3 months ago
    """
    res = {"rows": [], "count": 0, "total_bytes": 0, "total_size": "—"}
    if not text: return res

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    # salta l'header (NAME ID SIZE MODIFIED)
    for ln in lines:
        if re.match(r"^NAME\s+ID\s+SIZE\s+MODIFIED", ln, re.I):
            continue
        # split su >=2 spazi per non rompere i nomi con ':' (es. llama3.2:latest)
        parts = re.split(r"\s{2,}", ln.strip())
        if len(parts) < 4:
            # a volte su Windows le colonne hanno spacing diversi, prova split 1 volta
            parts = re.split(r"\s{2,}", re.sub(r"\s{2,}", "  ", ln.strip()))
            if len(parts) < 4:
                continue
        name, mid, size_s, modified = parts[0], parts[1], parts[2], parts[3]
        size_b = _parse_size_bytes(size_s)
        res["rows"].append({
            get_text("llm_adapters", "ollama_col_name"): name,
            get_text("llm_adapters", "ollama_col_id"): mid,
            get_text("llm_adapters", "ollama_col_size"): size_s,
            get_text("llm_adapters", "ollama_col_modified"): modified
        })
        res["total_bytes"] += size_b

    res["count"] = len(res["rows"])
    res["total_size"] = _format_bytes(res["total_bytes"]) if res["total_bytes"] else "—"
    return res
