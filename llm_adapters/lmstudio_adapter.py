# lmstudio_adapter.py
# -*- coding: utf-8 -*-
import os, sys, shutil, subprocess, json, time
import requests  # <-- SPOSTATO QUI
from typing import Tuple, Optional  # <-- AGGIUNTO

import streamlit as st  # opzionale

from GUI.message_gui import st_toast_temp

if 'server_lmStudio' not in st.session_state:
    st.session_state['server_lmStudio'] = False


# ================= NUOVA FUNZIONE CACHATA =================

@st.cache_data(ttl=5, show_spinner="Verifico stato server LM Studio...")
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
            return True, count  # (online, model_count)
    except Exception:
        pass  # La connessione fallisce, ecc.
    return False, 0  # (offline, model_count)


# ================= helpers base =================

def _which_lms() -> str | None:
    # 1) PATH standard
    for name in ("lms", "lms.exe", "lms.cmd"):
        p = shutil.which(name)
        if p:
            return p

    # 2) macOS install path (App bundle)
    mac_candidates = [
        "/Applications/LM Studio.app/Contents/Resources/bin/lms",
        os.path.expanduser("~/Applications/LM Studio.app/Contents/Resources/bin/lms"),
    ]

    # 3) Linux common installs / symlinks
    linux_candidates = [
        "/usr/local/bin/lms",
        "/usr/bin/lms",
        "/opt/LM Studio/resources/bin/lms",
        os.path.expanduser("~/.local/share/LM Studio/resources/bin/lms"),
    ]

    for p in (mac_candidates + linux_candidates):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    # 4) Windows fallback già presente (mantieni il tuo blocco attuale se vuoi)
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
        return {"ok": False, "code": 127, "stdout": "", "stderr": "file non trovato",
                "cmd_str": _cmd_str(cmd), "dur_s": round(time.time() - t0, 3)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": 124, "stdout": "", "stderr": "timeout",
                "cmd_str": _cmd_str(cmd), "dur_s": round(time.time() - t0, 3)}
    except Exception as e:
        return {"ok": False, "code": 1, "stdout": "", "stderr": f"errore: {e}",
                "cmd_str": _cmd_str(cmd), "dur_s": round(time.time() - t0, 3)}


# ================= CLI: ls =================
def lms_ls_raw():
    lms = _which_lms()
    if not lms:
        return {"ok": False, "code": 127, "stdout": "", "stderr": "CLI 'lms' non trovata nel PATH", "cmd_str": "lms ls"}
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

# Fallback storage PID se Streamlit non c'è
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
        return {"ok": False, "msg": "CLI 'lms' non trovata nel PATH"}
    # Comando
    args = ["server", "start"]
    cmd = ([lms, *args] if not (os.name == "nt" and lms.lower().endswith((".cmd", ".bat")))
           else ["cmd", "/c", lms, *args])

    # Opzioni di detach cross-platform
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
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
            return {"ok": True, "msg": f"Terminato PID {pid}"}
        except Exception as e:
            return {"ok": False, "msg": f"Errore stop PID {pid}: {e}"}

    # fallback: CLI stop (sincrono ma veloce)
    lms = _which_lms()
    if not lms:
        return {"ok": False, "msg": "CLI 'lms' non trovata nel PATH"}
    cmd = ([lms, "server", "stop"] if not (os.name == "nt" and lms.lower().endswith((".cmd", ".bat")))
           else ["cmd", "/c", lms, "server", "stop"])
    info = _run(cmd, timeout=10)
    return {"ok": info["ok"], "msg": (info["stdout"] or info["stderr"] or "done").strip()}


# ================= HTTP (facoltativo) =================
def generate(prompt: str, model_name: str, max_tokens: int = 128, host: str = "http://localhost:1234"):
    try:
        # 'requests' è ora importato all'inizio del file
        url = f"{host.rstrip('/')}/v1/chat/completions"
        payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.7, "max_tokens": int(max_tokens)}
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Errore HTTP: {e}"


# --- lmstudio_adapter.py ---

# def _badge(text: str):
#     return f"<span style='padding:2px 8px;border:1px solid rgba(255,255,255,.15);border-radius:999px;font-size:0.85rem'>{text}</span>"

# -- Helper: parsing robusto di `lms ls` in due tabelle (LLM / EMBEDDING) --
def _parse_lms_ls(stdout: str) -> dict:
    lines = stdout.splitlines()
    # rimuovi la riga iniziale tipo "You have 3 models..."
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
        # salta header delle colonne
        if re.match(r"^(PARAMS|ARCH|SIZE)\b", s):
            continue
        if cur in ("llm", "emb"):
            # split su 2+ spazi: [name, params, arch, size]
            parts = re.split(r"\s{2,}", s)
            # fai padding per sicurezza
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


# -- Helper: normalizza dimensioni per mostrarle belle --
def _pretty_size(s: str) -> str:
    # accetta "8.54 GB", "84.11 MB", "7.7 GB" ecc. -> restituisce lo stesso in formato pulito
    s = s.strip()
    m = re.match(r"^\s*([\d\.]+)\s*([GMK]B)\s*$", s, flags=re.I)
    if not m:
        return s
    num, unit = m.groups()
    # taglia zeri inutili (8.00 -> 8)
    num = f"{float(num):.2f}".rstrip("0").rstrip(".")
    unit = unit.upper()
    return f"{num} {unit}"


# ---- Core vero (quello che avevi già, refactor in funzione privata) ----
def _get_model_details_core(model_name: str) -> dict:
    """
    Tutta la logica: esegue `lms ls`, estrae i metadati, arricchisce dal nome
    e aggiunge spiegazioni. Non fa rendering.
    """

    # -------------- COPIA qui la tua versione “bella” --------------
    # Usa gli helper già presenti: _parse_lms_ls, _enrich_from_name, _short_explanations
    # Esempio minimo (riassunto dalla versione che ti ho dato prima):

    def _which_lms():
        for nm in ("lms", "lms.exe", "lms.cmd"):
            p = shutil.which(nm)
            if p: return p
        return None

    lms = _which_lms()
    if not lms:
        info = {
            "Nome": model_name, "Tipo": "Sconosciuto",
            "Parametri": "—", "Architettura": "—", "Dimensione su disco": "—",
        }
        info.update(_enrich_from_name(model_name))
        info["Spiegazioni"] = _short_explanations(info)
        return info

    try:
        p = subprocess.run([lms, "ls"], capture_output=True, text=True, timeout=10)
        stdout = p.stdout or ""
    except Exception as e:
        return {"Errore": f"Impossibile eseguire `lms ls`: {e}"}

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
            "Nome": row["name"],
            "Tipo": tipo,
            "Parametri": row["params"] or "—",
            "Architettura": row["arch"] or "—",
            "Dimensione su disco": row["size"] or "—",
        }
    else:
        info = {
            "Nome": model_name, "Tipo": "Sconosciuto",
            "Parametri": "—", "Architettura": "—", "Dimensione su disco": "—",
        }

    info.update(_enrich_from_name(info["Nome"]))
    info["Addestramento"] = "Instruct/Chat" if "instruct" in info["Nome"].lower() else "Base"
    info["Spiegazioni"] = _short_explanations(info)
    return info


def _short_explanations(info: dict) -> dict:
    """
    Piccole spiegazioni didattiche, pensate per essere lette velocemente in UI.
    """
    tipo = info.get("Tipo", "—")
    spieg_tipo = (
        "LLM: modello generativo di linguaggio, capace di comprendere e produrre testo."
        if tipo == "LLM"
        else "Embedding: trasforma testi in vettori numerici per ricerca semantica, clustering e RAG."
        if tipo == "Embedding"
        else "Tipo non riconosciuto direttamente da `lms ls`."
    )
    params = info.get("Parametri", "—")
    spieg_params = (
        f"Parametri ≈ {params}: i pesi del modello. Più parametri ⇒ più capacità, ma servono più memoria e calcoli."
        if params != "—" else
        "Parametri non rilevati: dipendono dal modello (es. 7B=7 miliardi di pesi)."
    )
    arch = info.get("Architettura", "—")
    spieg_arch = (
        f"Architettura {arch}: la 'famiglia' (Llama, Mistral, Gemma…). Cambia struttura e abilità del modello."
        if arch != "—" else
        "Architettura non specificata."
    )
    size = info.get("Dimensione su disco", "—")
    spieg_size = (
        f"Occupa ~{size} su disco: dipende da quantizzazione e formato (GGUF/FP16…)."
        if size != "—" else
        "Dimensione su disco non disponibile."
    )
    addestr = info.get("Addestramento", "—")
    spieg_add = (
        "Instruct/Chat: fine-tuning per seguire istruzioni in linguaggio naturale."
        if addestr.startswith("Instruct") else
        "Base: modello 'fundation' non specializzato per dialogo; utile per finetuning o compiti generici."
    )
    quant = info.get("Quantizzazione", "—")
    spieg_quant = (
        f"Quantizzazione {quant}: riduce precisione per diminuire memoria e aumentare velocità, con un po' di perdita qualitativa."
        if quant != "—" else
        "Quantizzazione non rilevata nel nome (esempi: Q4_K_M, Q5_0, INT8, BF16)."
    )
    ctx = info.get("Context window (stima)", "—")
    spieg_ctx = (
        f"Finestra di contesto ≈ {ctx}: numero massimo di token considerati in input/output."
        if ctx != "—" else
        "Finestra di contesto non deducibile dal nome (tipico 4k–128k token)."
    )
    return {
        "Tipo": spieg_tipo,
        "Parametri": spieg_params,
        "Architettura": spieg_arch,
        "Dimensione su disco": spieg_size,
        "Addestramento": spieg_add,
        "Quantizzazione": spieg_quant,
        "Context window": spieg_ctx,
    }


import re, subprocess, shutil


# -- Helper: arricchisce i metadati con info dedotte dal nome file --
def _enrich_from_name(name: str) -> dict:
    lower = name.lower()
    # Instruct/Base
    addestr = "Instruct/Chat" if re.search(r"\b(instruct|chat|sft|it)\b", lower) else "Base"
    # Parametri se non già dati (7b, 8b, 70b…)
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
        "Addestramento (dal nome)": addestr,
        "Quantizzazione": quant or "—",
        "Formato file": fmt or "—",
        "Context window (stima)": ctx or "—",
        "Versione (stima)": ver or "—",
    }


# ================= Pannello Streamlit snello (MODIFICATO) =================
# ================= Pannello Streamlit snello (MODIFICATO) =================
def lmstudio_panel(host: str = "http://localhost:1234", key='lmstudo'):
    if st is None:
        raise RuntimeError("Streamlit non disponibile")

    st.subheader("🧪 LM Studio — controllo CLI")

    # --- NUOVA LOGICA "DEFERRED LOAD" ---
    init_key = f"{key}_initialized"

    if init_key not in st.session_state:
        # PRIMO AVVIO: caricamento istantaneo, nessuno spinner.
        online = False
        count = 0
        st.session_state[init_key] = True
    else:
        # RERUN (dopo prima interazione): usa la cache.
        try:
            online, count = get_lmstudio_status(host)
        except Exception as e:
            st.error(f"❌ Errore connessione LM Studio: {e}")
            online, count = False, 0

    # Aggiorna lo stato globale (usato dal resto dell'app)
    st.session_state['server_lmStudio'] = online
    # --- FINE NUOVA LOGICA ---

    # --- bottoni start/stop/ls ---
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        if st.button("🚀 Start (background)", key=key + '_Start'):
            res = start_server_background()
            if res["ok"]:
                st.session_state['server_lmStudio'] = True
                st_toast_temp("Avviato in background.", 'success')
                get_lmstudio_status.clear()
                st.rerun()
            else:
                st.session_state['server_lmStudio'] = False
                st_toast_temp(res["msg"], 'error')

    with c2:
        if st.button("🛑 Stop Server", key=key + '_Stop'):
            res = stop_server_background()
            if res["ok"]:
                st.session_state['server_lmStudio'] = False
                st_toast_temp("🛑 Server Stop!", 'warning')
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
            list_model = st.button("📚 Lista modelli", key=key + '_lms')
        elif st.session_state.get(init_key, False):
            st.warning("🚨 Avviare il server!")

    cols_status = st.columns([2, 2, 3])
    with cols_status[0]:
        st.metric(
            label="🌐 HTTP Server",
            value="🟢 ONLINE" if online else "🔴 OFFLINE"
        )
    with cols_status[1]:
        st.metric("📦 # Modelli", count)
    with cols_status[2]:
        st.caption(f"🔗 Endpoint: {host.rstrip('/')}/v1/models")

    # STATUS PROCESSO CLI
    pid = _get_pid()

    if list_model and online:
        info = lms_ls_raw()

        stdout = (info.get("stdout") or "").strip()
        stderr = (info.get("stderr") or "").strip()

        primary = stdout or stderr
        primary_label = "📤 STDOUT" if stdout else "⚠️ STDERR (fallback)"

        with st.expander(primary_label):
            st.code(primary or "<vuoto>", language="bash")

        if stdout and stderr:
            with st.expander("⚠️ STDERR" if primary_label == "📤 STDOUT" else "📤 STDOUT"):
                st.code(stderr if primary_label == "📤 STDOUT" else stdout, language="bash")

        if info.get("ok"):
            st.success("✅ Comando eseguito con successo.")

            parsed = parse_lmstudio_ls(primary)

            tot = parsed.get("summary", {}).get("total_models",
                                                len(parsed["llms"]) + len(parsed["embeddings"]))
            siz = parsed.get("summary", {}).get("total_size", "—")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("🤖 Modelli totali", tot)
            with c2:
                st.metric("💾 Spazio su disco", siz)

            # tabella LLM
            if parsed["llms"]:
                st.markdown("### 🧠 LLM")
                st.table(pd.DataFrame(parsed["llms"]))

            # tabella EMBEDDING
            if parsed["embeddings"]:
                st.markdown("### 🪐 Embedding")
                st.table(pd.DataFrame(parsed["embeddings"]))

        else:
            st.error("❌ Comando fallito.")
    elif list_model and not online:
        st_toast_temp("⚠️ Server is not Running", 'warning')


import re
import pandas as pd


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
                "Modello": name, "Parametri": params, "Architettura": arch, "Dimensione": size
            })
        elif section == "emb":
            if len(parts) >= 4:
                name, params, arch, size = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) == 3:
                name, params, arch, size = parts[0], "—", parts[1], parts[2]
            else:
                continue
            res["embeddings"].append({
                "Embedding": name, "Parametri": params, "Architettura": arch, "Dimensione": size
            })
    return res


import pandas as pd


def get_model_details(model_name: str):
    """
    Recupera i dettagli di un modello da LM Studio.
    Restituisce solo dati strutturati, senza scrivere direttamente su Streamlit.
    """
    details = _get_model_details_core(model_name)

    # prepara i campi principali
    fields = [
        ("Nome", details.get("Nome")),
        ("Tipo", details.get("Tipo")),
        ("Parametri", details.get("Parametri")),
        ("Architettura", details.get("Architettura")),
        ("Dimensione su disco", details.get("Dimensione su disco")),
        ("Addestramento", details.get("Addestramento")),
        ("Quantizzazione", details.get("Quantizzazione")),
        ("Formato file", details.get("Formato file")),
        ("Context window", details.get("Context window (stima)")),
        ("Versione", details.get("Versione (stima)")),
    ]

    # filtra solo i valori significativi
    data = [(k, v) for k, v in fields if v and v not in ("—", "", None)]

    # se non c'è nulla, ritorna solo l'avviso nel dict
    if not data:
        details["warning"] = (
            "Nessun dettaglio disponibile per questo modello. "
            "LM Studio non ha fornito metadati utili o il parsing non ha trovato nulla."
        )
        return details

    # costruisce un DataFrame e lo aggiunge al dict
    df = pd.DataFrame([dict(data)])  # una riga con colonne dinamiche
    details["dataframe"] = df

    # aggiungi anche eventuale DataFrame con spiegazioni
    exp = details.get("Spiegazioni", {})
    if exp:
        exp_df = pd.DataFrame(
            [{"Parametro": k, "Descrizione": v} for k, v in exp.items() if v and v.strip()]
        )
        details["explanations_df"] = exp_df

    return details


# === LM Studio: helper "lms get" (CLI) ===
import shutil, subprocess, os, time


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
        yield "[errore] CLI 'lms' non trovata nel PATH. Apri LM Studio e usa 'Install CLI'."
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
        yield "[errore] Impossibile eseguire la CLI 'lms'."
        return 127
    except Exception as e:
        yield f"[errore] Avvio fallito: {e}"
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