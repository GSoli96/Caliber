import inspect

import streamlit as st
from utils.translations import get_text

from . import huggingface_adapter
from . import lmstudio_adapter
from . import ollama_adapter
from . import spacy_adapter

LLM_ADAPTERS = {
    "Hugging Face": huggingface_adapter,
    "Ollama": ollama_adapter,
    "LM Studio": lmstudio_adapter,
    "Spacy": spacy_adapter
}

def _adapter_for(backend: str):
    if backend not in LLM_ADAPTERS:
        raise KeyError(get_text("llm_adapters", "backend_not_supported", backend=backend))
    return LLM_ADAPTERS[backend]

def _filter_kwargs(fn, kwargs: dict):
    """Passa solo gli argomenti supportati dalla firma di fn (salvo **kwargs)."""
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}

def _fingerprint_kwargs(fn, kwargs: dict):
    """Crea una chiave cache coerente con i kwargs effettivamente usati da fn."""
    used = _filter_kwargs(fn, kwargs)
    return tuple(sorted(used.items()))

@st.cache_data(show_spinner=False)
def _cached_list_models(backend: str, kwargs_fp: tuple):
    adapter = _adapter_for(backend)
    fn = getattr(adapter, "list_models", None)
    if not fn:
        return []
    kwargs = dict(kwargs_fp)
    return fn(**kwargs)

def list_models(backend: str, **kwargs):
    adapter = _adapter_for(backend)
    fn = getattr(adapter, "list_models", None)
    if not fn:
        return []
    fp = _fingerprint_kwargs(fn, kwargs)
    return _cached_list_models(backend, fp)

@st.cache_data(show_spinner=get_text("llm_adapters", "loading_details"))
def _cached_get_details(backend: str, model_name: str, kwargs_fp: tuple):
    adapter = _adapter_for(backend)
    fn = getattr(adapter, "get_model_details", None)
    if not fn:
        return {'error': get_text("llm_adapters", "adapter_no_get_details", backend=backend)}
    kwargs = dict(kwargs_fp)
    return fn(model_name=model_name, **kwargs)

def get_model_details(backend: str, model_name: str, **kwargs):
    adapter = _adapter_for(backend)
    fn = getattr(adapter, "get_model_details", None)
    if not fn:
        return {'error': get_text("llm_adapters", "adapter_no_get_details", backend=backend)}
    fp = _fingerprint_kwargs(fn, kwargs)
    return _cached_get_details(backend, model_name, fp)

def generate(backend: str, prompt: str, **kwargs):
    adapter = _adapter_for(backend)
    fn = getattr(adapter, "generate", None)
    if not fn:
        return {'error': get_text("llm_adapters", "adapter_no_generate", backend=backend)}
    # niente cache qui; filtra kwargs e chiama
    call_kwargs = _filter_kwargs(fn, kwargs)
    return fn(prompt=prompt, **call_kwargs)

def clear_cache_for(kind: str | None = None):
    if kind is None:
        st.cache_data.clear()
        return
    if kind == "list_models":
        _cached_list_models.clear()  # type: ignore[attr-defined]
    elif kind == "get_model_details":
        _cached_get_details.clear()  # type: ignore[attr-defined]

def ensure_model_cached(model_id: str, backend: str = "Hugging Face", hf_token: str = None) -> str:
    adapter = _adapter_for(backend)
    fn = getattr(adapter, "ensure_model_cached", None)
    if not fn:
        raise ValueError(get_text("llm_adapters", "adapter_no_ensure_model_cached", backend=backend))
    return fn(model_id=model_id, hf_token=hf_token)