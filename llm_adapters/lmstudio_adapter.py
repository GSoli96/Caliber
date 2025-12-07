# lmstudio_adapter.py
# -*- coding: utf-8 -*-
import os, sys, shutil, subprocess, json, time
import requests
from typing import Tuple, Optional
import re
import pandas as pd

import streamlit as st

from GUI.message_gui import st_toast_temp
from utils.translations import get_text

if 'server_lmStudio' not in st.session_state:
    st.session_state['server_lmStudio'] = False


# ================= NUOVA FUNZIONE CACHATA =================

@st.cache_data(ttl=5, show_spinner=get_text("llm_adapters", "checking_status"))
def get_lmstudio_status(host: str) -> Tuple[bool, int]:
    """
    Controlla lo stato del server LM Studio e conta i modelli.
    Questa funzione verrà eseguita al massimo una volta ogni 5 secondi.
    """
    try:
        url = f"{host.rstrip('/')}/v1/models"
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            count = len((r.json() or {}).get("data", []))
            return True, count
    except Exception:
        pass
    return False, 0

# ================= helpers base =================
def _which_lms() -> str | None:
    for name in ("lms", "lms.exe", "lms.cmd"):
        p = shutil.which(name)
        if p:
            return p

    mac_candidates = [
        "/Applications/LM Studio.app/Contents/Resources/bin/lms",
        os.path.expanduser("~/Applications/LM Studio.app/Contents/Resources/bin/lms"),
    ]

    linux_candidates = [
        "/usr/local/bin/lms",
        "/usr/bin/lms",
        "/opt/LM Studio/resources/bin/lms",
        os.path.expanduser("~/.local/share/LM Studio/resources/bin/lms"),
    ]

    for p in (mac_candidates + linux_candidates):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    for p in (
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\LM Studio\resources\bin\lms.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\LM Studio\resources\bin\lms.cmd"),
    ):
        if p and os.path.isfile(p):
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
            "cmd_str": _cmd_str(cmd), "dur_s": round(time.time() - t0, 3),
        }
    except FileNotFoundError:
        return {"ok": False, "code": 127, "stdout": "", "stderr": get_text("llm_adapters", "file_not_found"),
                "cmd_str": _cmd_str(cmd), "dur_s": round(time.time() - t0, 3)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": 124, "stdout": "", "stderr": get_text("llm_adapters", "timeout"),
                "cmd_str": _cmd_str(cmd), "dur_s": round(time.time() - t0, 3)}
    except Exception as e:
        return {"ok": False, "code": 1, "stdout": "", "stderr": get_text("llm_adapters", "error_generic", e=e),
                "cmd_str": _cmd_str(cmd), "dur_s": round(time.time() - t0, 3)}

# ================= CLI: ls =================
def lms_ls_raw():
    lms = _which_lms()
    if not lms:
        return {"ok": False, "code": 127, "stdout": "", "stderr": get_text("llm_adapters", "cli_not_found"), "cmd_str": "lms ls"}
    cmd = (["cmd", "/c", lms, "ls"] if os.name == "nt" and lms.lower().endswith((".cmd", ".bat")) else [lms, "ls"])
    return _run(cmd, timeout=20)

def list_models(host: str | None = None, filter: str | None = None):
    info = lms_ls_raw()
    if not info["ok"]:
        return {"error": f"{info['stderr']} (code={info['code']}) — cmd: {info['cmd_str']}"}
    out = info["stdout"].strip()
    models: list[str] = []
    if out.startswith("{") or out.startswith("["):
        try:
            data = json.loads(out)
            seq = data if isinstance(data, list) else (data.get("data") or data.get("models") or [])
            for it in seq:
                models.append(str(it["id"] if isinstance(it, dict) and "id" in it else it))
        except Exception:
            pass
    if not models:
        for ln in out.splitlines():
            s = ln.strip()
            if s and not s.lower().startswith(("llm", "embedding", "params", "arch")):
                models.append(s.split()[0])
    if filter:
        f = filter.lower().strip()
        models = [m for m in models if f in m.lower()]
    return models

# ================= Server: start/stop non-bloccanti =================

_BG_STATE = {}
def _set_pid(pid: int | None):
    if st is not None:
        st.session_state["lmstudio_server_pid"] = pid
    else:
        _BG_STATE["pid"] = pid

def _get_pid() -> int | None:
    if st is not None:
        return st.session_state.get("lmstudio_server_pid")
    return _BG_STATE.get("pid")

def start_server_background():
    """
    Avvia `lms server start` in background e ritorna subito.
    Salva il PID per un eventuale stop.
    """
    lms = _which_lms()
    if not lms:
        return {"ok": False, "msg": get_text("llm_adapters", "cli_not_found")}
    args = ["server", "start"]
    cmd = ([lms, *args] if not (os.name == "nt" and lms.lower().endswith((".cmd", ".bat")))
           else ["cmd", "/c", lms, *args])

    if os.name == "nt":
        creationflags = 0x00000200 | 0x00000008
        p = subprocess.Popen(cmd, creationflags=creationflags, close_fds=True)
    else:
        p = subprocess.Popen(cmd, start_new_session=True, close_fds=True)
    _set_pid(p.pid)
    return {"ok": True, "pid": p.pid, "cmd": _cmd_str(cmd)}

def stop_server_background():
    """
    Prova a terminare il processo memorizzato (best-effort).
    Se non c'è PID, tenta `lms server stop` quick.
    """
    pid = _get_pid()
    if pid:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=10)

            else:
                os.kill(pid, 15)  # SIGTERM
            _set_pid(None)
            return {"ok": True, "msg": get_text("llm_adapters", "terminated_pid", pid=pid)}
        except Exception as e:
            return {"ok": False, "msg": get_text("llm_adapters", "error_stop_pid", pid=pid, e=e)}

    lms = _which_lms()
    if not lms:
        return {"ok": False, "msg": get_text("llm_adapters", "cli_not_found")}
    cmd = ([lms, "server", "stop"] if not (os.name == "nt" and lms.lower().endswith((".cmd", ".bat")))
           else ["cmd", "/c", lms, "server", "stop"])
    info = _run(cmd, timeout=10)
    return {"ok": info["ok"], "msg": (info["stdout"] or info["stderr"] or "done").strip()}

# ================= HTTP (facoltativo) =================
def generate(prompt: str, model_name: str, max_tokens: int = 128, host: str = "http://localhost:1234"):
    try:
        url = f"{host.rstrip('/')}/v1/chat/completions"
        payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.7, "max_tokens": int(max_tokens)}
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        return get_text("llm_adapters", "http_error", e=e)

def _parse_lms_ls(stdout: str) -> dict:
    lines = stdout.splitlines()
    if lines and lines[0].lower().startswith("you have"):
        lines = lines[1:]
    if lines and lines[0].lower().startswith("you have"):
        lines = lines[1:]
    cur = None
    llm_rows, emb_rows = [], []
    for raw in lines:
        ln = raw.rstrip()
        s = ln.strip()
        if not s:
            continue
        if s.startswith("LLM"):
            cur = "llm";
            continue
        if s.startswith("EMBEDDING"):
            cur = "emb";
            continue
        if re.match(r"^(PARAMS|ARCH|SIZE)\b", s):
            continue
        if cur in ("llm", "emb"):
            parts = re.split(r"\s{2,}", s)
            while len(parts) < 4:
                parts.append("")
            row = {
                "name": parts[0].strip(),
                "params": parts[1].strip(),
                "arch": parts[2].strip(),
                "size": _pretty_size(parts[3].strip()),
            }
            (llm_rows if cur == "llm" else emb_rows).append(row)
    return {"llm": llm_rows, "embedding": emb_rows}

def _pretty_size(s: str) -> str:
    s = s.strip()
    m = re.match(r"^\s*([\d\.]+)\s*([GMK]B)\s*$", s, flags=re.I)
    if not m:
        return s
    num, unit = m.groups()
    # taglia zeri inutili (8.00 -> 8)
    num = f"{float(num):.2f}".rstrip("0").rstrip(".")
    unit = unit.upper()
    return f"{num} {unit}"

def _get_model_details_core(model_name: str) -> dict:
    """
    Tutta la logica: esegue `lms ls`, estrae i metadati, arricchisce dal nome
    e aggiunge spiegazioni. Non fa rendering.
    """

    def _which_lms():
        for nm in ("lms", "lms.exe", "lms.cmd"):
            p = shutil.which(nm)
            if p: return p
        return None

    lms = _which_lms()
    if not lms:
        info = {
            "name": model_name, "type": get_text("llm_adapters", "unknown_type"),
            "params": "—", "arch": "—", "size": "—",
        }
        info.update(_enrich_from_name(model_name))
        info["explanations"] = _short_explanations(info)
        return info

    try:
        p = subprocess.run([lms, "ls"], capture_output=True, text=True, timeout=10)
        stdout = p.stdout or ""
    except Exception as e:
        return {"Errore": get_text("llm_adapters", "lms_ls_error", e=e)}

    parsed = _parse_lms_ls(stdout)

    def _find(rows):
        for r in rows:
            if r["name"].lower() == model_name.lower():
                return r
        for r in rows:
            if model_name.lower() in r["name"].lower():
                return r
        return None

    row, tipo = None, None
    row = _find(parsed.get("llm", []))
    if row: tipo = "LLM"
    if not row:
        row = _find(parsed.get("embedding", []))
        if row: tipo = "Embedding"

    if row:
        info = {
            "name": row["name"],
            "type": tipo,
            "params": row["params"] or "—",
            "arch": row["arch"] or "—",
            "size": row["size"] or "—",
        }
    else:
        info = {
            "name": model_name, "type": get_text("llm_adapters", "unknown_type"),
            "params": "—", "arch": "—", "size": "—",
        }

    info.update(_enrich_from_name(info["name"]))
    info["training"] = "Instruct/Chat" if "instruct" in info["name"].lower() else "Base"
    info["explanations"] = _short_explanations(info)
    return info

def _short_explanations(info: dict) -> dict:
    """
    Piccole spiegazioni didattiche, pensate per essere lette velocemente in UI.
    """
    tipo = info.get("type", "—")
    spieg_tipo = (
        get_text("llm_adapters", "type_llm")
        if tipo == "LLM"
        else get_text("llm_adapters", "type_embedding")
        if tipo == "Embedding"
        else get_text("llm_adapters", "type_unknown")
    )
    params = info.get("params", "—")
    spieg_params = (
        get_text("llm_adapters", "params_approx", params=params)
        if params != "—" else
        get_text("llm_adapters", "params_unknown")
    )
    arch = info.get("arch", "—")
    spieg_arch = (
        get_text("llm_adapters", "arch_fam", arch=arch)
        if arch != "—" else
        get_text("llm_adapters", "arch_unknown")
    )
    size = info.get("size", "—")
    spieg_size = (
        get_text("llm_adapters", "size_disk", size=size)
        if size != "—" else
        get_text("llm_adapters", "size_unknown")
    )
    addestr = info.get("training", "—")
    spieg_add = (
        get_text("llm_adapters", "train_instruct")
        if addestr.startswith("Instruct") else
        get_text("llm_adapters", "train_base")
    )
    quant = info.get("quantization", "—")
    spieg_quant = (
        get_text("llm_adapters", "quant_desc", quant=quant)
        if quant != "—" else
        get_text("llm_adapters", "quant_unknown")
    )
    ctx = info.get("ctx_est", "—")
    spieg_ctx = (
        get_text("llm_adapters", "ctx_window", ctx=ctx)
        if ctx != "—" else
        get_text("llm_adapters", "ctx_unknown")
    )
    return {
        get_text("llm_adapters", "type_col"): spieg_tipo,
        get_text("llm_adapters", "params_col"): spieg_params,
        get_text("llm_adapters", "arch_col"): spieg_arch,
        get_text("llm_adapters", "size_disk"): spieg_size,
        get_text("llm_adapters", "training_col"): spieg_add,
        get_text("llm_adapters", "quant_col"): spieg_quant,
        get_text("llm_adapters", "ctx_col"): spieg_ctx,
    }

def _enrich_from_name(name: str) -> dict:
    lower = name.lower()
    addestr = "Instruct/Chat" if re.search(r"\b(instruct|chat|sft|it)\b", lower) else "Base"
    params = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:b|bn)\b", lower)
    if m: params = f"{m.group(1)}B"
    # Quantizzazione
    quant = None
    q = re.search(r"\b(q\d(?:_[a-z0-9]+)?)\b", lower)
    if q: quant = q.group(1).upper()
    for kw in ("int4", "int8", "fp16", "bf16", "fp8", "nf4", "qnn"):
        if kw in lower and not quant:
            quant = kw.upper()
    # Formato
    fmt = "GGUF" if ".gguf" in lower else (
        "GGML" if ".ggml" in lower else ("SAFETENSORS" if "safetensors" in lower else None))
    # Context window
    ctx = None
    kctx = re.search(r"\b(\d+)\s*k\b", lower)
    if kctx:
        ctx = f"{kctx.group(1)}k token"
    else:
        nctx = re.search(r"\b(131072|65536|32768|16384|8192|4096)\b", lower)
        if nctx:
            ctx = f"{int(nctx.group(1)) // 1024}k token"
    # Versione
    ver = None
    v = re.search(r"\bv(\d+(?:\.\d+)*)\b", lower)
    if v: ver = v.group(1)
    return {
        "training_from_name": addestr,
        "quantization": quant or "—",
        "format": fmt or "—",
        "ctx_est": ctx or "—",
        "ver_est": ver or "—",
    }

def run_server_lmStudio(host: str = "http://localhost:1234", key='lmstudo'):
    import streamlit as st

    res = start_server_background()
    if res["ok"]:
        st.session_state['server_lmStudio'] = True
        st_toast_temp(get_text("llm_adapters", "started_background"), 'success')
        get_lmstudio_status.clear()
        print(f"[INFO] LM Studio server started")
        st.toast("LM Studio server started!")
    else:
        st.session_state['server_lmStudio'] = False
        st_toast_temp(res["msg"], 'error')
        st.toast("LM Studio server error!")
    
    # st.rerun()


# ================= Pannello Streamlit snello (MODIFICATO) =================
def lmstudio_panel(host: str = "http://localhost:1234", key='lmstudo'):
    if st is None:
        raise RuntimeError("Streamlit non disponibile")

    st.subheader(get_text("llm_adapters", "panel_title"))

    init_key = f"{key}_initialized"

    if init_key not in st.session_state:
        online = False
        count = 0
        st.session_state[init_key] = True
    else:
        try:
            online, count = get_lmstudio_status(host)
        except Exception as e:
            st.error(get_text("llm_adapters", "connection_error", e=e))
            online, count = False, 0

    st.session_state['server_lmStudio'] = online

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        if st.button(get_text("llm_adapters", "start_btn"), key=key + '_Start', disabled=online):
            res = start_server_background()
            if res["ok"]:
                st.session_state['server_lmStudio'] = True
                st_toast_temp(get_text("llm_adapters", "started_background"), 'success')
                get_lmstudio_status.clear()
                st.rerun()
            else:
                st.session_state['server_lmStudio'] = False
                st_toast_temp(res["msg"], 'error')

    with c2:
        if st.button(get_text("llm_adapters", "stop_btn"), key=key + '_Stop', disabled=not online):
            res = stop_server_background()
            if res["ok"]:
                st.session_state['server_lmStudio'] = False
                st_toast_temp(get_text("llm_adapters", "stop_toast"), 'warning')
                get_lmstudio_status.clear()
                st.rerun()
            else:
                st.session_state['server_lmStudio'] = True
                st_toast_temp(res["msg"], 'warning')

            if hasattr(st, "cache_data"):
                st.cache_data.clear()

    list_model = False
    with c3:
        if online:
            list_model = st.button(get_text("llm_adapters", "list_models_btn"), key=key + '_lms', disabled=not online)
        elif st.session_state.get(init_key, False):
            st.warning(get_text("llm_adapters", "start_server_warning"))

    cols_status = st.columns([2, 2, 3])
    with cols_status[0]:
        st.metric(
            label=get_text("llm_adapters", "http_server_label"),
            value=get_text("llm_adapters", "online") if online else get_text("llm_adapters", "offline")
        )
    with cols_status[1]:
        st.metric(get_text("llm_adapters", "num_models"), count)
    with cols_status[2]:
        st.caption(get_text("llm_adapters", "endpoint", url=f"{host.rstrip('/')}/v1/models"))

    pid = _get_pid()

    if list_model and online:
        info = lms_ls_raw()

        stdout = (info.get("stdout") or "").strip()
        stderr = (info.get("stderr") or "").strip()

        primary = stdout or stderr
        primary_label = get_text("llm_adapters", "stdout") if stdout else get_text("llm_adapters", "stderr_fallback")

        with st.expander(primary_label):
            st.code(primary or "<vuoto>", language="bash")

        if stdout and stderr:
            with st.expander(get_text("llm_adapters", "stderr") if primary_label == get_text("llm_adapters", "stdout") else get_text("llm_adapters", "stdout")):
                st.code(stderr if primary_label == get_text("llm_adapters", "stdout") else stdout, language="bash")

        if info.get("ok"):
            st.toast(get_text("llm_adapters", "cmd_success"))

            parsed = parse_lmstudio_ls(primary)

            tot = parsed.get("summary", {}).get("total_models",
                                                len(parsed["llms"]) + len(parsed["embeddings"]))
            siz = parsed.get("summary", {}).get("total_size", "—")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(get_text("llm_adapters", "total_models"), tot)
            with c2:
                st.metric(get_text("llm_adapters", "disk_space"), siz)

            if parsed["llms"]:
                st.markdown(get_text("llm_adapters", "llm_header"))
                st.table(pd.DataFrame(parsed["llms"]))

            if parsed["embeddings"]:
                st.markdown(get_text("llm_adapters", "embedding_header"))
                st.table(pd.DataFrame(parsed["embeddings"]))

        else:
            st.error(get_text("llm_adapters", "cmd_failed"))
    elif list_model and not online:
        st_toast_temp(get_text("llm_adapters", "server_not_running"), 'warning')

def parse_lmstudio_ls(text: str):
    """
    Parsifica l'output di `lmstudio ls` tipo:
    You have 3 models, taking up 16.32 GB of disk space.

    LLM                           PARAMS    ARCH     SIZE
    meta-llama-3.1-8b-instruct    8B        Llama    8.54 GB
    ...

    EMBEDDING                     PARAMS    ARCH          SIZE
    text-embedding-...                      Nomic BERT    84.11 MB
    """
    res = {"summary": {}, "llms": [], "embeddings": []}
    if not text:
        return res

    # 1) summary
    m = re.search(r"You have\s+(\d+)\s+models,\s+taking up\s+(.+?)\s+of disk space", text, re.I)
    if m:
        res["summary"] = {"total_models": int(m.group(1)), "total_size": m.group(2)}

    # 2) normalizza righe (niente righe vuote)
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]

    section = None
    for ln in lines:
        if re.match(r"^LLM\s+PARAMS\s+ARCH\s+SIZE", ln):
            section = "llm";
            continue
        if re.match(r"^EMBEDDING\s+PARAMS\s+ARCH\s+SIZE", ln):
            section = "emb";
            continue
        if ln.lower().startswith("you have"):
            continue  # riga di summary

        parts = re.split(r"\s{2,}", ln.strip())
        if section == "llm":
            if len(parts) >= 4:
                name, params, arch, size = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) == 3:
                name, params, arch, size = parts[0], "—", parts[1], parts[2]
            else:
                continue
            res["llms"].append({
                get_text("llm_adapters", "model_col"): name, 
                get_text("llm_adapters", "params_col"): params, 
                get_text("llm_adapters", "arch_col"): arch, 
                get_text("llm_adapters", "size_col"): size
            })
        elif section == "emb":
            if len(parts) >= 4:
                name, params, arch, size = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) == 3:
                name, params, arch, size = parts[0], "—", parts[1], parts[2]
            else:
                continue
            res["embeddings"].append({
                get_text("llm_adapters", "embedding_col"): name, 
                get_text("llm_adapters", "params_col"): params, 
                get_text("llm_adapters", "arch_col"): arch, 
                get_text("llm_adapters", "size_col"): size
            })
    return res

def get_model_details(model_name: str):
    """
    Recupera i dettagli di un modello da LM Studio.
    Restituisce solo dati strutturati, senza scrivere direttamente su Streamlit.
    """
    details = _get_model_details_core(model_name)

    # prepara i campi principali
    fields = [
        (get_text("llm_adapters", "name_col"), details.get("name")),
        (get_text("llm_adapters", "type_col"), details.get("type")),
        (get_text("llm_adapters", "params_col"), details.get("params")),
        (get_text("llm_adapters", "arch_col"), details.get("arch")),
        (get_text("llm_adapters", "size_disk"), details.get("size")),
        (get_text("llm_adapters", "training_col"), details.get("training")),
        (get_text("llm_adapters", "quant_col"), details.get("quantization")),
        (get_text("llm_adapters", "format_col"), details.get("format")),
        (get_text("llm_adapters", "ctx_col"), details.get("ctx_est")),
        (get_text("llm_adapters", "version_col"), details.get("ver_est")),
    ]

    # filtra solo i valori significativi
    data = [(k, v) for k, v in fields if v and v not in ("—", "", None)]

    # se non c'è nulla, ritorna solo l'avviso nel dict
    if not data:
        details["warning"] = get_text("llm_adapters", "no_details")
        return details

    # costruisce un DataFrame e lo aggiunge al dict
    df = pd.DataFrame([dict(data)])  # una riga con colonne dinamiche
    details["dataframe"] = df

    # aggiungi anche eventuale DataFrame con spiegazioni
    exp = details.get("explanations", {})
    if exp:
        exp_df = pd.DataFrame(
            [{get_text("llm_adapters", "param_col"): k, get_text("llm_adapters", "desc_col"): v} for k, v in exp.items() if v and v.strip()]
        )
        details["explanations_df"] = exp_df

    return details

# === LM Studio: helper "lms get" (CLI) ===
def _which_lms_cli() -> str | None:
    # prova PATH standard
    for nm in ("lms", "lms.exe", "lms.cmd"):
        p = shutil.which(nm)
        if p:
            return p
    # alcuni path noti su Windows
    for p in (
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\LM Studio\resources\bin\lms.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\LM Studio\resources\bin\lms.cmd"),
    ):
        if p and os.path.isfile(p):
            return p
    return None

def lms_get_stream(model_or_query: str, extra_args: list[str] | None = None):
    """
    Avvia `lms get <model_or_query> [extra_args...]` e produce righe di output.
    Ritorna (returncode, generator_di_righe).
    """
    lms = _which_lms_cli()
    if not lms:
        yield get_text("llm_adapters", "cli_error_path")
        return 127

    # Costruisci comando cross-platform, supportando .cmd su Windows
    base = [lms, "get", model_or_query]
    if extra_args:
        base += list(extra_args)

    if os.name == "nt" and lms.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c"] + base
    else:
        cmd = base

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True
        )
    except FileNotFoundError:
        yield get_text("llm_adapters", "cli_exec_error")
        return 127
    except Exception as e:
        yield get_text("llm_adapters", "start_fail", e=e)
        return 1

    # stream delle righe
    last_line_time = time.time()
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            yield line.rstrip("\n")
            last_line_time = time.time()
        elif proc.poll() is not None:
            break
        else:
            # nessuna riga per un po' → evita spin
            time.sleep(0.05)

    return proc.returncode