import re

# --- INIZIO BLOCCO CORRETTO ---
# Ho rimosso i flag inline (?m) e (?i) dalle singole stringhe.
# Questi flag verranno applicati globalmente in re.compile().
TESTCASE_MARKERS = [
    r'^\s*#{2,}.*$',  # Titoli tipo "### TEST CASES ###"
    r'^\s*[-*]\s',  # Bullet list
    r'```',  # Inizio/chiusura di un altro fenced block
    r'\btest\s*cases?\b',
    r'\bexamples?\b',
]
# --- FINE BLOCCO CORRETTO ---

SQL_START = r'(?is)\b(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b'


def extract_sql_query(raw_text: str) -> str | None:
    """
    Extract a clean SQL query from LLM-generated text.
    
    Handles various formats including markdown code blocks, plain SQL,
    and text with multiple queries. Removes test cases, examples, and
    non-SQL content.
    
    Args:
        raw_text: Raw text output from LLM that may contain SQL
    
    Returns:
        str | None: Cleaned SQL query with semicolon, or None if no query found
    
    Example:
        >>> extract_sql_query("```sql\\nSELECT * FROM users\\n```")
        'SELECT * FROM users;'
    """
    if not raw_text:
        return None

    # 1) Fenced code block ```sql ... ``` oppure ``` ... ```
    m = re.search(r"```(?:\s*sql)?\s*\n(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if m:
        q = m.group(1).strip()
        return q if q else None

    # 2) Prima statement che inizia con keyword SQL
    m = re.search(SQL_START + r'.*', raw_text)
    if not m:
        return None

    tail = m.group(0)

    # 2a) Taglia al primo ';' se c'è
    semi = tail.find(';')
    if semi != -1:
        q = tail[:semi + 1]
    else:
        # 2b) Taglia prima di marker non-SQL (###, bullet, ecc.)
        # --- RIGA CORRETTA ---
        # Applico i flag re.IGNORECASE e re.MULTILINE qui.
        cut_re = re.compile('|'.join(TESTCASE_MARKERS), re.IGNORECASE | re.MULTILINE)
        # --- FINE CORREZIONE ---

        cut = cut_re.search(tail)
        if cut:
            q = tail[:cut.start()]
        else:
            # 2c) O prima di un'eventuale seconda statement SQL su nuova riga
            m2 = re.search(r'(?is)\A(.*?)(?=\n\s*' + SQL_START + r')', tail)
            q = m2.group(1) if m2 else tail

    # Pulizia finale (senza schiacciare per forza i newline)
    q = q.strip()
    # Aggiungi ';' se manca e non è una DDL che spesso non richiede il punto e virgola nel tuo esecutore
    if q and not q.endswith(';'):
        q += ';'
    return q or None