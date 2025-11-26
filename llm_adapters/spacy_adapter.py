# =========================
# SPA CY ADAPTER (compat con llm_adapters)
# =========================
from typing import Optional, Dict, Any, List
import json
import time

# Flag di debug sicuro
DEBUG_VAR = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# 3) Ultimo tentativo: probe dei nomi più comuni
COMMON_SPACY_MODELS = [
    "en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf",
    "it_core_news_sm", "it_core_news_md", "it_core_news_lg",
    "es_core_news_sm", "de_core_news_sm", "fr_core_news_sm",
]

LANG_NAMES = {
    "it": "Italiano",
    "en": "Inglese",
    "es": "Spagnolo",
    "de": "Tedesco",
    "fr": "Francese",
    "pt": "Portoghese",
    "nl": "Olandese",
    "xx": "Multilingua",
}

def suffix_of(model: str) -> str:
    parts = model.split("_")
    return parts[-1] if parts else model

def _lang_of(model: str) -> str:
    # es: it_core_news_sm -> "it"
    return model.split("_")[0] if "_" in model else "xx"

def _size_hint(suffix: str) -> str:
    # Stima prudente, visuale e non vincolante
    if suffix == "sm":
        return "Piccolo (~15–30 MB), senza vettori"
    if suffix == "md":
        return "Medio (~50–200 MB), vettori di dimensione media"
    if suffix == "lg":
        return "Grande (~300–900+ MB), vettori di alta qualità"
    if suffix == "trf":
        return "Transformer (centinaia di MB), richiede PyTorch/transformers"
    return "N/D"

def _default_pipeline(suffix: str, lang: str) -> Dict[str, Any]:
    # Pipeline tipica dei modelli 'core'
    base = ["tok2vec", "senter", "tagger", "morphologizer", "lemmatizer", "parser", "ner"]
    # Alcuni sm non includono esplicitamente 'tok2vec' come componente separato
    if suffix == "sm":
        base = ["senter", "tagger", "morphologizer", "lemmatizer", "parser", "ner"]
    if suffix == "trf":
        base = ["transformer", "tagger", "morphologizer", "lemmatizer", "parser", "ner"]
    tasks = [
        "Tokenizzazione",
        "POS tagging",
        "Morfologia/lemmatizzazione",
        "Parsing di dipendenza",
        "Riconoscimento entità (NER)",
        "Fraseamento (senter)",
    ]
    if suffix in ("md", "lg"):
        tasks.append("Similarità semantica (vettori)")
    if suffix == "trf":
        tasks.append("Rappresentazioni Transformer")
    return {"pipeline": base, "tasks": tasks}

import importlib.util
from pathlib import Path

def _read_installed_meta(model_name: str) -> Optional[Dict[str, Any]]:
    """Se il modello è installato, tenta di leggere meta.json/config.cfg."""
    try:
        spec = importlib.util.find_spec(model_name)
        if not spec or not spec.submodule_search_locations:
            return None
        pkg_dir = Path(list(spec.submodule_search_locations)[0])
        # spaCy v3: meta.json + config.cfg
        meta_path = pkg_dir / "meta.json"
        cfg_path = pkg_dir / "config.cfg"
        meta: Dict[str, Any] = {"installed_path": str(pkg_dir)}
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta.update(json.load(f))
        # leggi pipeline da config.cfg se presente
        if cfg_path.exists():
            try:
                import configparser
                cp = configparser.ConfigParser()
                # Config di spaCy non è INI puro, ma possiamo cercare la sezione [nlp]
                text = cfg_path.read_text(encoding="utf-8", errors="ignore")
                # estrai pipeline da riga "pipeline = [ ... ]"
                import re
                m = re.search(r"pipeline\s*=\s*\[(.*?)\]", text, re.S)
                if m:
                    items = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
                    meta.setdefault("pipeline", items)
            except Exception:
                pass
        return meta
    except Exception:
        return None

def model_details(model_name: str) -> Dict[str, Any]:
    """Restituisce dettagli combinando metadati reali (se installato) e fallback."""
    suffix = suffix_of(model_name)
    lang_code = _lang_of(model_name)
    language = LANG_NAMES.get(lang_code, lang_code)
    installed_meta = _read_installed_meta(model_name)

    details: Dict[str, Any] = {
        "model": model_name,
        "language": language,
        "lang_code": lang_code,
        "size_hint": _size_hint(suffix),
        "installed": installed_meta is not None,
        "installed_path": installed_meta.get("installed_path") if installed_meta else None,
        "version": installed_meta.get("version") if installed_meta else None,
        "spacy_version": installed_meta.get("spacy_version") if installed_meta else None,
        "vectors": None,
        "pipeline": None,
        "tasks": None,
        "notes": [],
    }

    # Pipeline effettiva se disponibile
    pipe = None
    if installed_meta:
        pipe = installed_meta.get("pipeline") or installed_meta.get("components")
        # meta.json di solito ha "components": {...} e "pipeline": [..]
        if isinstance(pipe, dict):
            pipe = list(pipe.keys())
        details["pipeline"] = pipe

        # info vettori (se presenti nel meta)
        if "vectors" in installed_meta:
            v = installed_meta["vectors"]
            if isinstance(v, dict):
                n_keys = v.get("keys") or v.get("vectors") or v.get("nlp_vectors")
                details["vectors"] = f"presenti ({n_keys} chiavi)" if n_keys else "presenti"
        # qualche nota simpatica
        if suffix == "trf":
            details["notes"].append("Usa un backbone Transformer; può richiedere CUDA per velocità adeguate.")

    # Fallback sensato
    if details["pipeline"] is None:
        dflt = _default_pipeline(suffix, lang_code)
        details["pipeline"] = dflt["pipeline"]
        details["tasks"] = dflt["tasks"]
    else:
        # Deriva tasks dalla pipeline reale
        tasks = []
        pset = set(details["pipeline"])
        if {"tagger", "morphologizer", "lemmatizer"} & pset:
            tasks.append("POS tagging & morfologia/lemmatizzazione")
        if "parser" in pset:
            tasks.append("Parsing di dipendenza")
        if "ner" in pset:
            tasks.append("Riconoscimento entità (NER)")
        if "senter" in pset:
            tasks.append("Fraseamento (senter)")
        if "transformer" in pset:
            tasks.append("Rappresentazioni Transformer")
        details["tasks"] = tasks or ["(pipeline personalizzata)"]

    # Vettori: inferisci dal suffisso se non noto
    if details["vectors"] is None:
        if suffix in ("md", "lg"):
            details["vectors"] = "presenti"
        elif suffix == "sm":
            details["vectors"] = "assenti"
        elif suffix == "trf":
            details["vectors"] = "non applicabile (usa embeddings dal transformer)"

    return details

# --------- util: scan modelli installati ----------
def _spacy_list_installed_models() -> List[str]:
    names = set()
    if not SPACY_AVAILABLE:
        return []

    # 1) API ufficiale, se presente
    try:
        from spacy.util import get_installed_models
        models = get_installed_models()
        for m in models:
            # m può essere Path o oggetto con "name"
            names.add(str(getattr(m, "name", m)))
    except Exception:
        pass

    # 2) Fallback via importlib.metadata
    try:
        import importlib.metadata as importlib_metadata
        for dist in importlib_metadata.distributions():
            meta = dist.metadata or {}
            pkg_name = (meta.get("Name") or "").strip()
            lower = pkg_name.lower()
            # euristica per i pacchetti ufficiali spaCy
            if "_core_" in lower:
                names.add(pkg_name)
    except Exception:
        pass

    if SPACY_AVAILABLE:
        from spacy.util import is_package
        for cname in COMMON_SPACY_MODELS:
            try:
                if is_package(cname):
                    names.add(cname)
            except Exception:
                pass

    return sorted(names)

def list_spacy_models() -> List[str] | Dict[str, Any]:
    if not SPACY_AVAILABLE:
        return {"error": "Libreria 'spacy' non installata. Esegui: pip install spacy"}
    try:
        models = _spacy_list_installed_models()
        if not models:
            return {"error": "Nessun modello spaCy trovato. Installa, ad es.: python -m spacy download it_core_news_sm"}
        return models
    except Exception as e:
        return {"error": f"Errore durante la scansione dei modelli spaCy: {e}"}

def get_spacy_model_details(model_name: str) -> Dict[str, Any]:
    if not SPACY_AVAILABLE:
        return {"error": "Libreria 'spacy' non disponibile."}
    try:
        if DEBUG_VAR:
            return {
                "Modello": model_name,
                "Lingua": "debug",
                "Versione": "debug",
                "Pipeline": ["tok2vec", "tagger", "morphologizer", "parser", "ner"],
                "Descrizione": "",
                "Vettori": {"dim": 0, "n_keys": 0},
                "Has_TextCat": False,
                "Has_Transformer": False,
            }

        nlp = spacy.load(model_name)  # gestisce sia package che path
        meta = getattr(nlp, "meta", {}) or {}
        pipe_names = list(nlp.pipe_names)
        vec_dim = int(getattr(nlp.vocab, "vectors_length", 0) or 0)
        vec_keys = int(getattr(getattr(nlp.vocab, "vectors", None), "n_keys", 0) or 0)

        return {
            "Modello": meta.get("name", model_name),
            "Lingua": meta.get("lang", getattr(nlp, "lang", "N/A")),
            "Versione": meta.get("version", "N/A"),
            "Pipeline": pipe_names,
            "Descrizione": meta.get("description", ""),
            "Vettori": {"dim": vec_dim, "n_keys": vec_keys},
            "Has_TextCat": ("textcat" in pipe_names) or ("textcat_multilabel" in pipe_names),
            "Has_Transformer": ("transformer" in pipe_names),
        }
    except Exception as e:
        return {"error": f"Errore nel recuperare i dettagli del modello spaCy '{model_name}': {e}"}

def spacy_process(
    text: str,
    model_name: str,
    *,
    return_ents: bool = True,
    return_pos: bool = True,
    return_dep: bool = True,
    return_noun_chunks: bool = True,
    return_sents: bool = True,
    return_textcat: bool = True,
    max_len: Optional[int] = None
) -> Dict[str, Any] | str:
    if DEBUG_VAR:
        time.sleep(0.2)
        return {
            "ents": [{"text": "Milano", "label": "LOC"}],
            "pos": [{"text": "Ciao", "pos": "INTJ"}],
            "dep": [],
            "noun_chunks": ["la rete neurale"],
            "sents": ["Ciao mondo."],
            "textcat": []
        }

    if not SPACY_AVAILABLE:
        return "Libreria 'spacy' non installata. Esegui 'pip install spacy' e il relativo modello (es. it_core_news_sm)."

    try:
        nlp = spacy.load(model_name)
        if max_len is not None:
            nlp.max_length = max_len
        doc = nlp(text)

        result: Dict[str, Any] = {}

        if return_ents:
            result["ents"] = [{
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            } for ent in doc.ents]

        if return_pos:
            result["pos"] = [{
                "text": tok.text,
                "lemma": tok.lemma_,
                "pos": tok.pos_,
                "tag": tok.tag_,
                "morph": tok.morph.to_dict() if tok.morph else {},
                "is_stop": bool(tok.is_stop)
            } for tok in doc]

        if return_dep:
            result["dep"] = [{
                "text": tok.text,
                "dep": tok.dep_,
                "head": tok.head.text,
                "head_i": tok.head.i,
            } for tok in doc]

        if return_noun_chunks:
            result["noun_chunks"] = [chunk.text for chunk in doc.noun_chunks]

        if return_sents:
            result["sents"] = [sent.text for sent in doc.sents]

        if return_textcat:
            cats = []
            if "textcat" in nlp.pipe_names or "textcat_multilabel" in nlp.pipe_names:
                cats = [{"label": k, "score": float(v)} for k, v in doc.cats.items()]
            result["textcat"] = cats

        return result
    except Exception as e:
        return f"Errore durante l'elaborazione con spaCy '{model_name}': {e}"

# --------- API attesa da llm_adapters ---------
def list_models(**kwargs) -> List[str]:
    """Usata da llm_adapters.list_models('Spacy', **kwargs)."""
    models = list_spacy_models()
    if isinstance(models, dict) and "error" in models:
        # llm_adapters si aspetta una lista; qui restituiamo lista vuota,
        # il messaggio verrà gestito a livello UI.
        return []
    return models  # type: ignore[return-value]

def get_model_details(model_name: str, **kwargs) -> Dict[str, Any]:
    return get_spacy_model_details(model_name)

def generate(prompt: str, *, model_name: str, **kwargs):
    """Compat: llm_adapters.generate(..., prompt=..., model_name=...)"""
    return spacy_process(
        text=prompt,
        model_name=model_name,
        return_ents=kwargs.get("return_ents", True),
        return_pos=kwargs.get("return_pos", True),
        return_dep=kwargs.get("return_dep", True),
        return_noun_chunks=kwargs.get("return_noun_chunks", True),
        return_sents=kwargs.get("return_sents", True),
        return_textcat=kwargs.get("return_textcat", True),
        max_len=kwargs.get("max_len"),
    )
