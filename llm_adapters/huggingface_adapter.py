import os
import humanize
import json
from pathlib import Path
import time
from datetime import datetime

# --- HF hub opzionale, solo per list/cache details ---
try:
    from huggingface_hub import scan_cache_dir

    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

# --- Transformers + torch opzionali per generate ---
try:
    import torch
except Exception:
    torch = None

try:
    from transformers import pipeline, AutoConfig, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _select_device_map_and_dtype():
    """
    Sceglie impostazioni per esecuzione preferendo GPU.
    Ritorna: (device_map, torch_dtype, device_legacy)
      - device_map: "auto" | None
      - torch_dtype: torch.dtype | "auto" | None
      - device_legacy: int | str | None (per pipeline vecchie che usano `device=...`)
    """
    device_map = None
    torch_dtype = None
    device_legacy = None

    # Preferisci CUDA
    if torch is not None and torch.cuda.is_available():
        # device_map="auto" delega a accelerate l’allocazione su GPU
        device_map = "auto"
        # dtype automatico (fp16/ bf16 se supportato)
        torch_dtype = "auto"
        device_legacy = 0  # fallback
    # Apple Silicon (Metal)
    elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Per MPS l’ecosistema è più sensibile: usiamo device legacy se serve
        device_map = None
        torch_dtype = "auto"
        device_legacy = "mps"
    else:
        # CPU
        device_map = None
        torch_dtype = None
        device_legacy = -1

    return device_map, torch_dtype, device_legacy


def list_models():
    """
    Scansiona la cache di Hugging Face e restituisce la lista dei modelli trovati.
    """
    if not HUGGINGFACE_HUB_AVAILABLE:
        return {'error': "Libreria 'huggingface_hub' non installata. Esegui: pip install huggingface_hub"}
    try:
        cache_info = scan_cache_dir()
        model_ids = [repo.repo_id for repo in cache_info.repos]
        if not model_ids:
            return {'error': "Nessun modello trovato nella cache di Hugging Face."}
        return model_ids
    except Exception as e:
        return {'error': f"Errore durante la scansione della cache di Hugging Face: {e}"}


def get_model_details(model_name: str):
    """
    Recupera i dettagli di un modello specifico dalla cache di Hugging Face.
    """
    if not HUGGINGFACE_HUB_AVAILABLE:
        return {'error': "Libreria 'huggingface_hub' non disponibile."}
    try:
        cache_info = scan_cache_dir()
        repo_info = next((repo for repo in cache_info.repos if repo.repo_id == model_name), None)
        if not repo_info:
            return {'error': f"Dettagli non trovati per il modello '{model_name}' nella cache."}

        last_modified_str = "N/A"
        if isinstance(repo_info.last_modified, datetime):
            last_modified_str = repo_info.last_modified.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(repo_info.last_modified, (int, float)):
            last_modified_datetime = datetime.fromtimestamp(repo_info.last_modified)
            last_modified_str = last_modified_datetime.strftime('%Y-%m-%d %H:%M:%S')

        details = {
            "Nome (Repo ID)": repo_info.repo_id,
            "Dimensione su Disco": humanize.naturalsize(repo_info.size_on_disk),
            "Percorso Cache": str(repo_info.repo_path),
            "Ultima Modifica": last_modified_str
        }

        config_file = Path(repo_info.repo_path) / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            archs = config.get("architectures", [])
            details["Architettura"] = archs[0] if archs else "N/A"
            details["Tipo Modello"] = config.get("model_type", "N/A")

        return details
    except Exception as e:
        return {'error': f"Errore nel recuperare i dettagli del modello '{model_name}': {e}"}


def generate(prompt: str, model_name: str, max_tokens=128):
    """
    Genera testo usando un modello dalla cache di Hugging Face,
    dopo aver verificato che sia un modello adatto alla generazione.
    Restituisce SEMPRE una stringa (compat con la tua UI).
    """
    print(f"\n[DEBUG-ADAPTER] Funzione 'generate' chiamata con modello: {model_name}")
    print(f"[DEBUG-ADAPTER] Prompt ricevuto (prime 100 chars): {prompt[:100]}...")


    if not TRANSFORMERS_AVAILABLE:
        return "Libreria 'transformers' non installata. Esegui 'pip install transformers torch'."

    try:
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)

        is_generative = bool(getattr(config, "is_encoder_decoder", False))
        if not is_generative:
            archs = getattr(config, "architectures", []) or []
            for arch in archs:
                if arch.endswith("ForCausalLM") or arch.endswith("LMHeadModel"):
                    is_generative = True
                    break
        if not is_generative and hasattr(config, "task_specific_params"):
            params = config.task_specific_params or {}
            if isinstance(params, dict) and "text-generation" in params:
                is_generative = True

        if not is_generative:
            error_msg = f"ERRORE: Il modello '{model_name}' non è generativo (CausalLM o Encoder-Decoder)."
            return error_msg

        device_map, torch_dtype, device_legacy = _select_device_map_and_dtype()

        try:
            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception as e:
            msg = str(e)

            if "SentencePiece" in msg or "tokenizer.model" in msg:
                return (
                    f"Errore: il tokenizer di '{model_name}' richiede il file "
                    "'tokenizer.model' (SentencePiece) ma non è presente nella cache locale. "
                    "Scarica il modello completo oppure disabilita local_files_only."
                )

            # fallback: prova a scaricare i file mancanti
            try:
                tok = AutoTokenizer.from_pretrained(model_name)
            except Exception as e2:
                return f"Errore nel caricamento del tokenizer per '{model_name}': {e2}"

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        pipe = None
        try:
            # --- BLOCCO CORRETTO ---
            # Sostituito il parametro deprecato 'torch_dtype' con 'dtype'.
            pipe = pipeline(
                'text-generation',
                model=model_name,
                tokenizer=tok,
                device_map=device_map,
                dtype=torch.float16 if torch_dtype == "auto" and torch and torch.cuda.is_available() else None,
            )
            # --- FINE BLOCCO CORRETTO ---
        except TypeError:
            pipe = pipeline(
                'text-generation',
                model=model_name,
                tokenizer=tok,
                device=device_legacy
            )

        gen_kwargs = dict(
            max_new_tokens=int(max_tokens),
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            eos_token_id=tok.eos_token_id
        )

        output = pipe(prompt, **gen_kwargs)

        if output and isinstance(output, list):
            obj = output[0]
            full_text = obj.get('generated_text') or obj.get('text')
        else:
            full_text = None

        if not full_text:
            return "Errore: L'output del modello era vuoto o malformato."

        generated_text = full_text
        p = prompt.strip()
        if generated_text.strip().startswith(p):
            generated_text = generated_text.strip()[len(p):]

        cleaned_text = generated_text.replace("- Query:", "").replace("`", "").strip()
        return cleaned_text

    except RuntimeError as rte:
        msg = str(rte)
        if "CUDA out of memory" in msg or "out of memory" in msg.lower():
            return ("Errore: memoria GPU esaurita durante la generazione. "
                    "Riduci max_new_tokens/batch o usa un modello più piccolo. "
                    "In alternativa forza l'uso della CPU.")
        return f"Errore runtime durante la generazione con '{model_name}': {rte}"

    except Exception as e:
        return f"Errore durante la generazione con il modello '{model_name}': {e}"