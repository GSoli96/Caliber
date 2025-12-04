# ============================================================================
# MODIFICHE PER task_anomalies (circa riga 425)
# ============================================================================

def task_anomalies(df: pd.DataFrame, z_thresh: float, min_year: int, future_days: int, ss_key: str, tag: str) -> Dict[str, Any]:
    # START MONITORING
    data_list, stop_event, monitor = start_monitoring()
    
    reporter = make_progress_reporter(ss_key, tag, total=max(1, len(df.columns)))
    rows_ = []
    for i, col in enumerate(df.columns, start=1):
        ser = df[col]
        miss = is_null_series(ser).mean()
        const = int((safe_nunique(ser) == 1))
        out_frac = zscore_outlier_rate(ser, z=z_thresh) if pd.api.types.is_numeric_dtype(ser) else float("nan")
        # date
        if pd.api.types.is_datetime64_any_dtype(ser):
            s = pd.to_datetime(ser, errors="coerce", utc=True)
            now = pd.Timestamp.utcnow()
            future = float((s > now + pd.Timedelta(days=future_days)).mean())
            ancient = float((s < pd.Timestamp(f"{min_year}-01-01", tz="UTC")).mean())
            date_future = round(future * 100, 2)
            date_old = round(ancient * 100, 2)
        else:
            date_future = float("nan"); date_old = float("nan")
        rows_.append({
            "Colonna": col,
            "Missing %": round(miss * 100, 2),
            "Costante?": "✅" if const else "",
            f"Outlier |z|>{z_thresh:g} %": (round(out_frac * 100, 2) if not np.isnan(out_frac) else np.nan),
            "Date future %": date_future,
            f"Date < {min_year} %": date_old,
        })
        reporter(i)
    anomalies = pd.DataFrame(rows_)
    
    # STOP MONITORING
    metrics = stop_monitoring(data_list, stop_event, monitor)
    
    return {"anomalies": anomalies, "metrics": metrics}


# ============================================================================
# MODIFICHE PER task_semantic (circa riga 455)
# ============================================================================

def task_semantic(df: pd.DataFrame, sem_sample: int, enable_spacy: bool, ss_key: str, tag: str) -> Dict[str, Any]:
    # START MONITORING
    data_list, stop_event, monitor = start_monitoring()
    
    reporter = make_progress_reporter(ss_key, tag, total=max(1, len(df.columns)))

    patterns = {
        "email": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        "telefono_it": r"^\+?3?9?\s?[\d\s\-]{6,}$",
        "codice_fiscale_it": r"^[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]$",
        "iban_it": r"^IT[0-9A-Z]{25}$",
        "cap_it": r"^\d{5}$",
    }
    nlp = None
    if enable_spacy:
        try:
            import spacy
            for model in ("it_core_news_sm", "en_core_web_sm"):
                try:
                    nlp = spacy.load(model)
                    break
                except Exception:
                    continue
        except Exception:
            nlp = None
    rows_sem = []
    for i, col in enumerate(df.columns, start=1):
        ser = df[col]
        dtype = str(ser.dtype)
        sample = ser.dropna().astype(str).head(int(sem_sample))
        matches = {}
        for label, rx in patterns.items():
            try:
                m = sample.str.fullmatch(rx, regex=True).mean() if not sample.empty else 0.0
            except Exception:
                m = 0.0
            matches[label] = round(float(m) * 100, 2)
        pii_hint = "🔒 possibile PII" if any(v > 10 for v in matches.values()) else ""
        ner_labels = Counter()
        if nlp and not sample.empty:
            try:
                text = "\n".join(sample.sample(min(20, len(sample)), random_state=42))
                doc = nlp(text)
                ner_labels.update([ent.label_ for ent in doc.ents])
            except Exception:
                pass
        # tipo SQL suggerito (euristico)
        sql_type = "TEXT"
        if pd.api.types.is_integer_dtype(ser):
            sql_type = "BIGINT" if pd.to_numeric(ser, errors="coerce").max() > 2**31 - 1 else "INT"
        elif pd.api.types.is_float_dtype(ser):
            sql_type = "DECIMAL(18,6)"
        elif pd.api.types.is_bool_dtype(ser):
            sql_type = "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(ser):
            sql_type = "TIMESTAMP"
        elif pd.api.types.is_string_dtype(ser):
            try:
                maxlen = int(ser.astype(str).str.len().max())
                if maxlen <= 50: sql_type = "VARCHAR(50)"
                elif maxlen <= 255: sql_type = "VARCHAR(255)"
            except Exception:
                pass
        rows_sem.append({
            "Colonna": col, "dtype": dtype, "Suggerito_SQL": sql_type,
            "Match email %": matches["email"], "Match tel %": matches["telefono_it"],
            "Match CF %": matches["codice_fiscale_it"], "Match IBAN %": matches["iban_it"],
            "Match CAP %": matches["cap_it"],
            "NER labels (sample)": ", ".join(f"{k}:{v}" for k, v in ner_labels.items()) if ner_labels else "",
            "Nota": pii_hint
        })
        reporter(i)
    sem_df = pd.DataFrame(rows_sem)
    
    # STOP MONITORING
    metrics = stop_monitoring(data_list, stop_event, monitor)
    
    return {"semantic": sem_df, "metrics": metrics}
