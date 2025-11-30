import os
import humanize
import json
from pathlib import Path
import time
from datetime import datetime
from utils.translations import get_text

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
        return {'error': get_text('llm_adapters', 'hf_hub_not_installed')}
    try:
        cache_info = scan_cache_dir()
        model_ids = [repo.repo_id for repo in cache_info.repos]
        if not model_ids:
            return {'error': get_text('llm_adapters', 'hf_no_models')}
        return model_ids
    except Exception as e:
        return {'error': get_text('llm_adapters', 'hf_scan_error', error=e)}


def get_model_details(model_name: str):
    """
    Recupera i dettagli di un modello specifico dalla cache di Hugging Face.
    """
    if not HUGGINGFACE_HUB_AVAILABLE:
        return {'error': get_text('llm_adapters', 'hf_hub_not_available')}
    try:
        cache_info = scan_cache_dir()
        repo_info = next((repo for repo in cache_info.repos if repo.repo_id == model_name), None)
        if not repo_info:
            return {'error': get_text('llm_adapters', 'hf_details_not_found', model=model_name)}

        last_modified_str = "N/A"
        if isinstance(repo_info.last_modified, datetime):
            last_modified_str = repo_info.last_modified.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(repo_info.last_modified, (int, float)):
            last_modified_datetime = datetime.fromtimestamp(repo_info.last_modified)
            last_modified_str = last_modified_datetime.strftime('%Y-%m-%d %H:%M:%S')

        details = {
            get_text('llm_adapters', 'hf_detail_name'): repo_info.repo_id,
            get_text('llm_adapters', 'hf_detail_size'): humanize.naturalsize(repo_info.size_on_disk),
            get_text('llm_adapters', 'hf_detail_cache_path'): str(repo_info.repo_path),
            get_text('llm_adapters', 'hf_detail_last_modified'): last_modified_str
        }

        config_file = Path(repo_info.repo_path) / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            archs = config.get("architectures", [])
            details[get_text('llm_adapters', 'hf_detail_architecture')] = archs[0] if archs else "N/A"
            details[get_text('llm_adapters', 'hf_detail_model_type')] = config.get("model_type", "N/A")

        return details
    except Exception as e:
        return {'error': get_text('llm_adapters', 'hf_details_error', model=model_name, error=e)}


def generate(prompt: str, model_name: str, max_tokens=128):
    """
    Genera testo usando un modello dalla cache di Hugging Face,
    dopo aver verificato che sia un modello adatto alla generazione.
    Restituisce SEMPRE una stringa (compat con la tua UI).
    """
    print(f"\n[DEBUG-ADAPTER] Funzione 'generate' chiamata con modello: {model_name}")
    print(f"[DEBUG-ADAPTER] Prompt ricevuto (prime 100 chars): {prompt[:100]}...")


    if not TRANSFORMERS_AVAILABLE:
        return get_text('llm_adapters', 'transformers_not_installed')

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
            error_msg = get_text('llm_adapters', 'hf_model_not_generative', model=model_name)
            return error_msg

        device_map, torch_dtype, device_legacy = _select_device_map_and_dtype()

        try:
            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception as e:
            msg = str(e)

            if "SentencePiece" in msg or "tokenizer.model" in msg:
                return (
                    get_text('llm_adapters', 'hf_tokenizer_error', model=model_name) + " " +
                    get_text('llm_adapters', 'hf_tokenizer_sentencepiece_missing')
                )

            # fallback: prova a scaricare i file mancanti
            try:
                tok = AutoTokenizer.from_pretrained(model_name)
            except Exception as e2:
                return get_text('llm_adapters', 'hf_tokenizer_load_error', model=model_name, error=e2)

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
            return get_text('llm_adapters', 'hf_empty_output')

        generated_text = full_text
        p = prompt.strip()
        if generated_text.strip().startswith(p):
            generated_text = generated_text.strip()[len(p):]

        cleaned_text = generated_text.replace("- Query:", "").replace("`", "").strip()
        return cleaned_text

    except RuntimeError as rte:
        msg = str(rte)
        if "CUDA out of memory" in msg or "out of memory" in msg.lower():
            return (
                get_text('llm_adapters', 'hf_gpu_memory') + " " +
                get_text('llm_adapters', 'hf_gpu_memory_advice')
            )
        return get_text('llm_adapters', 'hf_runtime_error', model=model_name, error=rte)

    except Exception as e:
        return get_text('llm_adapters', 'hf_generation_error', model=model_name, error=e)