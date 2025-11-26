import re
import threading
from typing import Dict, Any, List

# Set di etichette NER che consideriamo sensibili (spaCy standard)
SENSITIVE_NER_LABELS = {
    "PERSON",   # nomi di persone
    "GPE",      # luoghi politici (città, paesi)
    "LOC",      # luoghi generici
    "FAC",      # strutture fisiche (meno comune ma può contenere indirizzi)
    "DATE",     # date (es. data di nascita)
    "MONEY",    # salari, importi
    "NORP",     # nazionalità/gruppi: potenzialmente sensibile (GDPR)
}

# Keyword/regex indicative di PII (IT + EN). Le chiavi sono "categoria" (solo per report)
SENSITIVE_KEYWORDS: Dict[str, List[re.Pattern]] = {
    "personal_identifier": [
        re.compile(r"\b(nome|first[_ ]?name)\b", re.I),
        re.compile(r"\b(cognome|last[_ ]?name)\b", re.I),
        re.compile(r"\b(username|user[_ ]?id|account[_ ]?id)\b", re.I),
        re.compile(r"\b(codice[_ ]?fiscale|cf)\b", re.I),
        re.compile(r"\b(ssn|nin|tax[_ ]?id|codice[_ ]?tributario)\b", re.I),
        re.compile(r"\b(passport|passaporto)\b", re.I),
        re.compile(r"\b(driver[_ ]?license|patente)\b", re.I),
    ],
    "contact": [
        re.compile(r"\b(email|e[-_ ]?mail)\b", re.I),
        re.compile(r"\b(telefono|phone|cell(ulare)?|mobile|whatsapp)\b", re.I),
        re.compile(r"\b(handle|twitter|instagram|linkedin)\b", re.I),
    ],
    "address": [
        re.compile(r"\b(indirizzo|address|via|street|avenue|viale|piazza)\b", re.I),
        re.compile(r"\b(cap|zip|post[_ ]?code)\b", re.I),
        re.compile(r"\b(citt[aà]|city|provincia|province|paese|country|stato|region)\b", re.I),
    ],
    "demographic": [
        re.compile(r"\b(data[_ ]?di[_ ]?nascita|dob|birth(date| day)?)\b", re.I),
        re.compile(r"\b(luogo[_ ]?di[_ ]?nascita|place[_ ]?of[_ ]?birth)\b", re.I),
        re.compile(r"\b(et[aà]|age)\b", re.I),
        re.compile(r"\b(genere|sesso|gender)\b", re.I),
        re.compile(r"\b(nazionalit[aà]|cittadinanza|nationality)\b", re.I),
        re.compile(r"\b(religion(e)?|religious|politic(al|he)|partito)\b", re.I),  # categorie particolari
        re.compile(r"\b(salute|health|disabilit[aà]|disability|diagnos[ie])\b", re.I),
    ],
    "financial": [
        re.compile(r"\b(iban|bic|swift)\b", re.I),
        re.compile(r"\b(carta[_ ]?di[_ ]?credito|credit[_ ]?card|pan)\b", re.I),
        re.compile(r"\b(iban|conto|account|iban)\b", re.I),
        re.compile(r"\b(salary|stipendio|reddito|income|busta[_ ]?paga)\b", re.I),
    ],
}

def is_sensitive_column(col_name: str, nlp) -> Dict[str, Any]:
    """
    Valuta se un nome colonna è potenzialmente sensibile (PII) usando:
    - NER di spaCy sul nome colonna (poco testo, ma utile per label come GPE/DATE/PERSON)
    - keyword/regex multilingua per PII comuni.

    Ritorna un dizionario con:
      - sensitive: bool
      - score: int (euristico)
      - reasons: List[str]
      - matched_keywords: List[str]
      - matched_ner_labels: List[str]
    """
    text = (col_name or "").strip().replace("_", " ")
    reasons = []
    matched_keywords = []
    matched_ner_labels = []
    score = 0

    # 1) Keyword matching
    for category, patterns in SENSITIVE_KEYWORDS.items():
        for pat in patterns:
            if pat.search(text):
                matched_keywords.append(f"{category}:{pat.pattern}")
                reasons.append(f"matched keyword in {category}")
                score += 2  # keyword hit: peso medio

    # 2) NER sul nome colonna (può essere breve, ma a volte 'Rome', 'Birth date', ecc.)
    try:
        doc = nlp(text)
        labels_found = {ent.label_ for ent in doc.ents}
    except Exception:
        labels_found = set()

    labels_sensitive = labels_found & SENSITIVE_NER_LABELS
    if labels_sensitive:
        matched_ner_labels.extend(sorted(labels_sensitive))
        reasons.append(f"NER labels: {', '.join(sorted(labels_sensitive))}")
        score += 1 * len(labels_sensitive)  # ogni label sensibile vale 1

    # 3) Heuristics extra: pattern comuni “tecnici”
    # Es: colonna 'emailAddress', 'userEmail', 'billing_zip', ecc.
    camel_or_concat = re.sub(r"[^A-Za-z0-9]", "", col_name or "")
    if re.search(r"email", camel_or_concat, re.I):
        matched_keywords.append("contact:email(camelCase)")
        reasons.append("matched keyword in contact (camelCase)")
        score += 2
    if re.search(r"(phone|tel|cell)", camel_or_concat, re.I):
        matched_keywords.append("contact:phone(camelCase)")
        reasons.append("matched keyword in contact (camelCase)")
        score += 2

    # 4) Soglia decisione
    sensitive = score >= 2  # soglia conservativa (>=1 keyword forte o NER+qualcos'altro)

    return {
        "column": col_name,
        "sensitive": sensitive,
        "score": score,
        "reasons": sorted(set(reasons)),
        "matched_keywords": matched_keywords,
        "matched_ner_labels": matched_ner_labels,
    }

import spacy
from spacy.cli import download

def load_spacy_model(model_name: str):
    """
    Carica un modello spaCy, e se non è installato, lo scarica automaticamente.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Modello {model_name} non trovato. Download in corso...")
        download(model_name)
        nlp = spacy.load(model_name)
    return nlp
