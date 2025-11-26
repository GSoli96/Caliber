import re
import numpy as np
import pandas as pd
import streamlit as st
from llm_adapters.sensitive_entity import is_sensitive_column, load_spacy_model
from utils.load_config import get_color_from_dtype
from utils.translations import get_text

# --- IMPORTS FOR IMPUTATION ---
from sklearn.impute import SimpleImputer, KNNImputer
# Enable IterativeImputer (experimental)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

if 'spacy_model' not in st.session_state:
    st.session_state.setdefault('spacy_model',
                                {'model': 'en_core_web_sm',
                                 'status': 'notLoad'}
                                )

if st.session_state['spacy_model'].get('status') == 'notLoad':
    load_spacy_model(st.session_state['spacy_model'].get('model'))
else:
    pass

# --- mapping tipo -> etichetta/colore ---
# --- mapping tipo -> etichetta/colore ---
TYPE_INFO = {
    "text":        {"color": "#2EC4B6"},
    "numeric":     {"color": "#7C4DFF"},
    "categorical": {"color": "#FF9800"},
    "datetime":    {"color": "#00ACC1"},
    "boolean":     {"color": "#8BC34A"},
    "other":       {"color": "#9E9E9E"},
}

# --- NUOVE FUNZIONI PER L'ANALISI DEL CONTENUTO ---
def is_likely_url(series: pd.Series) -> bool:
    """Verifica se una serie di oggetti contiene probabilmente degli URL."""
    # Controlla solo un campione per efficienza e ignora i valori nulli
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    # Regex semplice per http/https/ftp/file
    url_pattern = re.compile(r'^(https?|ftp|file)://', re.IGNORECASE)
    return sample.astype(str).str.match(url_pattern).any()

def is_likely_path(series: pd.Series) -> bool:
    """Verifica se una serie di oggetti contiene probabilmente dei path di file."""
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    # Regex per path Unix-like (/) o Windows-like (C:\)
    path_pattern = re.compile(r'^(?:[a-zA-Z]:\\|/|\\|\.\.?\\)')
    return sample.astype(str).str.match(path_pattern).any()

def is_likely_email(series: pd.Series) -> bool:
    """Verifica se una serie di oggetti contiene probabilmente degli indirizzi email."""
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    # Regex per un formato email standard
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return sample.astype(str).str.match(email_pattern).any()

def get_column_category(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        # euristica: pochi valori distinti => categorico
        uniq = s.nunique(dropna=True)
        if 1 < uniq <= max(20, int(0.05*len(s))):
            return "categorical"
        return "numeric"
    # oggetti/stringhe
    return "text" if pd.api.types.is_string_dtype(s) or s.dtype == "object" else "other"

# ---------- Funzione principale ----------
def detailed_dataset(df: pd.DataFrame, key=""):
    st.subheader(get_text("load_dataset", "dataset_specs"))

    # metriche principali (aggiungo % missing e duplicati)
    total_missing = int(df.isna().sum().sum())
    pct_missing   = float(total_missing / (df.shape[0]*max(1, df.shape[1])) * 100.0) if df.size else 0.0
    dup_rows = int(df.duplicated().sum())
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(get_text("load_dataset", "num_rows"), f"{df.shape[0]:,}")
    c2.metric(get_text("load_dataset", "num_cols"), f"{df.shape[1]:,}")
    c3.metric(get_text("load_dataset", "missing_values"), f"{total_missing:,}", f"{pct_missing:.1f}%")
    c4.metric(get_text("load_dataset", "duplicate_rows"), f"{dup_rows:,}")
    st.caption(get_text("load_dataset", "mem_usage", mb=mem_mb))

    # Tabs per organizzare: Panoramica | Colonne | Missing | Correlazioni
    t_overview, t_cardinality, t_columns, t_missing, t_corr = st.tabs(
        [get_text("load_dataset", "tab_overview"), get_text("load_dataset", "tab_cardinality"), get_text("load_dataset", "tab_columns"), get_text("load_dataset", "tab_missing"), get_text("load_dataset", "tab_correlations")]
    )

    # ---------- PANORAMICA ----------
    with t_overview:
        st.markdown(f"#### {get_text('load_dataset', 'desc_stats')}")
        # describe “all” formattato meglio
        # --- Compatibilità per versioni di pandas diverse ---
        desc_df = df.copy()
        for col in desc_df.select_dtypes(include=["datetime", "datetimetz"]).columns:
            # converte in timestamp numerico (secondi unix) per calcolare min/max/mean
            desc_df[col] = desc_df[col].astype("int64") / 1e9

        try:
            # tenta con datetime_is_numeric (pandas >= 1.1)
            desc = desc_df.describe(include="all", datetime_is_numeric=True).T
        except TypeError:
            # fallback compatibile per pandas più vecchi
            desc = desc_df.describe(include="all").T

        st.dataframe(
            desc.style.format(na_rep="—"),  # 👈 qui
            width='stretch'
        )
        # st.dataframe(desc, width='stretch')
    with t_cardinality:
        st.markdown(f"#### {get_text('load_dataset', 'cardinality_col')}")
        unique_df = (
            df.nunique(dropna=True)
            .rename(get_text("load_dataset", "unique_count"))
            .reset_index()
            .rename(columns={"index": get_text("load_dataset", "column")})
        )
        unique_df[get_text("load_dataset", "unique_pct")] = (unique_df[get_text("load_dataset", "unique_count")] / len(df) * 100).round(2) if len(df) else 0.0
        st.dataframe(
            unique_df,
            column_config={
                get_text("load_dataset", "unique_pct"): st.column_config.ProgressColumn(
                    get_text("load_dataset", "unique_pct"), min_value=0.0, max_value=100.0, format="%.1f%%  "
                )
            },
            hide_index=True, width='stretch'
        )

    # ---------- COLONNE (schema) ----------
    with t_columns:
        st.markdown(f"#### {get_text('load_dataset', 'schema_cols')}")
        st.caption(get_text("load_dataset", "schema_caption"))
        schema_df = build_schema_df(df)

        # Applichiamo lo style, poi passiamo a dataframe i dati SENZA la colonna tecnica
        styler = style_schema(schema_df)
        df_visible = styler.data.drop(columns=["tipo_color"], errors="ignore")
        st.dataframe(
            df_visible,
            column_config={
                get_text("load_dataset", "missing_pct"): st.column_config.ProgressColumn(get_text("load_dataset", "missing_pct"), min_value=0.0, max_value=100.0, format="%.1f%%"),
                get_text("load_dataset", "unique_pct"): st.column_config.ProgressColumn(get_text("load_dataset", "unique_pct"), min_value=0.0, max_value=100.0, format="%.1f%%"),
                get_text("load_dataset", "sensitive"): st.column_config.TextColumn(get_text("load_dataset", "sensitive"),
                                                         help="🔒 = colonna potenzialmente sensibile",
                                                         width="small"),
                get_text("load_dataset", "reason"): st.column_config.TextColumn(get_text("load_dataset", "reason"),
                                                      help="Indizi/keyword/NER",
                                                      width="medium",
                                                      max_chars=100),
                get_text("load_dataset", "example"): st.column_config.TextColumn(get_text("load_dataset", "example"), help="Primo valore non nullo osservato",
                                                        width="medium"),
            },
            hide_index=True,
            width='stretch',
        )

    # ---------- MISSING ----------
    with t_missing:
        missing_value_tab(df, key)
    # ---------- CORRELAZIONI ----------
    with t_corr:
        st.markdown(f"#### {get_text('load_dataset', 'corr_matrix')}")
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True)
            # evidenzia con background gradient
            corr_style = corr.style.background_gradient(cmap="seismic",
                                                        vmin=-1,
                                                        vmax=1)
            st.dataframe(corr_style, width='stretch')
            # top coppie per valore assoluto
            corr_pairs = (
                corr.where(~np.eye(corr.shape[0], dtype=bool))  # rimuovi diagonale
                    .stack()
                    .rename("corr")
                    .abs()
                    .sort_values(ascending=False)
                    .reset_index()
            )
            corr_pairs.columns = ["Colonna A", "Colonna B", "|corr|"]
            st.markdown(f"**{get_text('load_dataset', 'top_corr_pairs')}**")
            st.dataframe(corr_pairs.head(10), hide_index=True, width='stretch')
        else:
            st.info(get_text("load_dataset", "no_num_cols"))

def preview_dataset(df, name, key_alter):
    rows_to_show = st.number_input(
        get_text("load_dataset", "rows_to_show"),
        min_value=1,
        max_value=len(df),
        value=min(5, len(df)),
        step=1,
        help=get_text("load_dataset", "rows_to_show_help"),
        key=f'numberInput_{key_alter}'
    )
    st.write(df.head(rows_to_show))

def info_dataset(df, key):
    """
    Renderizza un Dataset Explorer migliorato.
    Mostra un'anteprima del DataFrame con un'intestazione ricca di informazioni
    e un expander con statistiche dettagliate.
    """
    with st.container():
        # -------- UI --------
        st.markdown(f"### {get_text('load_dataset', 'legend')}")
        st.markdown(f"""
        <div style='line-height: 1.6; font-size: 0.95rem;'>

        - {get_text('load_dataset', 'legend_col_color')}
          &nbsp;&nbsp;● <span style='color:#2EC4B6;'>{get_text('load_dataset', 'text')}</span>
          &nbsp;● <span style='color:#7C4DFF;'>{get_text('load_dataset', 'numeric')}</span>
          &nbsp;● <span style='color:#FF9800;'>{get_text('load_dataset', 'categorical')}</span>
          &nbsp;● <span style='color:#00ACC1;'>{get_text('load_dataset', 'datetime')}</span>
          &nbsp;● <span style='color:#8BC34A;'>{get_text('load_dataset', 'boolean')}</span><br>

        - {get_text('load_dataset', 'legend_missing')}<br>

        - {get_text('load_dataset', 'legend_sensitive')}

        </div>
        """, unsafe_allow_html=True)
        schema_df = build_schema_df(df)

        # Progress bar per i missing nella tabella di schema
        st.dataframe(
            style_schema(schema_df),
            column_config={
                get_text("load_dataset", "missing_pct"): st.column_config.ProgressColumn(
                    get_text("load_dataset", "missing_pct"),
                    help="Percentuale di valori NaN/null",
                    min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                get_text("load_dataset", "sensitive"): st.column_config.TextColumn(
                    get_text("load_dataset", "sensitive"),
                    help="🔒 = colonna potenzialmente sensibile",
                    width="small",
                ),
            },
            hide_index=True,
            width='stretch',
        )


def build_schema_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        cat = get_column_category(s)
        tinfo = TYPE_INFO.get(cat, TYPE_INFO["other"])

        miss_pct = float(s.isna().mean() * 100.0)
        uniq = s.nunique(dropna=True)
        uniq_pct = float(uniq / n * 100.0) if n else 0.0

        # Valuta sensibilità in modo robusto
        try:
            nlp_model = st.session_state['spacy_model']['model']
        except Exception:
            nlp_model = None

        try:
            sens_info = is_sensitive_column(col, nlp_model) if nlp_model else {"sensitive": False, "reasons": []}
        except Exception:
            sens_info = {"sensitive": False, "reasons": []}

        is_sens = bool(sens_info.get("sensitive", False))
        reason = ", ".join(sens_info.get("reasons", [])) or "—"

        # Esempio: primo valore non nullo “leggibile”
        try:
            sample_val = next((str(v) for v in s.dropna().head(1).tolist()), "—")
            sample_val = (sample_val[:77] + "…") if len(sample_val) > 80 else sample_val
        except Exception:
            sample_val = "—"

        rows.append({
            get_text("load_dataset", "column"): col,
            get_text("load_dataset", "type"): get_text("load_dataset", cat),
            get_text("load_dataset", "missing_pct"): miss_pct,
            get_text("load_dataset", "unique_pct"): uniq_pct,
            get_text("load_dataset", "sensitive"): "🔒" if is_sens else "—",
            get_text("load_dataset", "reason"): reason,
            get_text("load_dataset", "example"): sample_val,
            "tipo_color": tinfo["color"],  # per lo styling (non visibile all’utente)
        })

    schema = pd.DataFrame(rows)
    # Ordina: sensibili, poi missing desc, poi unici desc
    schema = schema.sort_values(by=[get_text("load_dataset", "sensitive"), get_text("load_dataset", "missing_pct"), get_text("load_dataset", "unique_pct")], ascending=[False, False, False]).reset_index(drop=True)
    return schema

def style_schema(df_schema: pd.DataFrame):
    # Crea la mappa Colonna -> colore
    if "tipo_color" in df_schema.columns:
        color_map = df_schema.set_index(get_text("load_dataset", "column"))["tipo_color"].to_dict()
    else:
        color_map = {}

    df_visible = df_schema.drop(columns=["tipo_color"], errors="ignore")

    def color_colname(val):
        return f"color: {color_map.get(val, 'inherit')}; font-weight:600"

    # Applica lo stile solo alla colonna "Colonna"
    styler = df_visible.style.map(color_colname, subset=[get_text("load_dataset", "column")])
    return styler



 # --- IMPORTS FOR IMPUTATION ---
from sklearn.impute import SimpleImputer, KNNImputer
# Enable IterativeImputer (experimental)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import llm_adapters

def missing_value_tab(df: pd.DataFrame, key=""):
    """
    Tab per gestire i valori mancanti con vari algoritmi.
    """
    st.markdown(f"### {get_text('imputation', 'header')}")

    # 1. Identifica colonne con missing
    missing_cols = df.columns[df.isna().any()].tolist()
    
    if not missing_cols:
        st.success("No missing values found in this dataset! 🎉")
        return

    st.info(f"{get_text('imputation', 'cols_with_missing')} {', '.join(missing_cols)}")

    # --- SCELTA TIPO IMPUTAZIONE ---
    imputation_type = st.radio(
        get_text('imputation', 'imputation_type'),
        options=["numeric", "text"],
        format_func=lambda x: get_text('imputation', 'numeric_standard') if x == "numeric" else get_text('imputation', 'text_llm'),
        key=f"imputation_type_{key}"
    )

    if imputation_type == "numeric":
        # --- LOGICA ESISTENTE (NUMERICA) ---
        _render_numeric_imputation(df, missing_cols, key)
    else:
        # --- LOGICA LLM (TESTO) ---
        _render_llm_imputation(df, missing_cols, key)


def _render_numeric_imputation(df, missing_cols, key):
    # Filtra solo colonne numeriche che hanno missing
    numeric_cols_missing = [c for c in missing_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols_missing:
        st.warning("No numeric columns with missing values found.")
        return

    # 2. Selezione colonne
    cols_to_impute = st.multiselect(
        get_text('imputation', 'select_cols'),
        options=numeric_cols_missing,
        default=numeric_cols_missing[:1],
        key=f"num_cols_sel_{key}"
    )

    if not cols_to_impute:
        st.warning(get_text('imputation', 'no_cols'))
        return

    # 3. Selezione Metodo
    method = st.selectbox(
        get_text('imputation', 'method'),
        options=["simple", "knn", "iterative"],
        format_func=lambda x: get_text('imputation', x),
        key=f"num_method_sel_{key}"
    )

    # 4. Parametri
    imputer = None
    
    cols_numeric = df[cols_to_impute].select_dtypes(include=[np.number]).columns.tolist()
    cols_non_numeric = [c for c in cols_to_impute if c not in cols_numeric]

    if method == "simple":
        strategy = st.selectbox(
            get_text('imputation', 'strategy'),
            options=["mean", "median", "most_frequent", "constant"],
            format_func=lambda x: get_text('imputation', x),
            key=f"num_strategy_sel_{key}"
        )
        
        fill_value = None
        if strategy == "constant":
            fill_value = st.text_input(get_text('imputation', 'fill_value'), value="0", key=f"num_fill_val_{key}")
            if cols_numeric and not cols_non_numeric:
                try:
                    fill_value = float(fill_value)
                except:
                    pass

        if cols_non_numeric and strategy in ["mean", "median"]:
            st.warning(f"⚠️ Strategy '{strategy}' cannot be applied to non-numeric columns: {cols_non_numeric}. They will be skipped or cause error.")
        
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

    elif method == "knn":
        if cols_non_numeric:
            st.warning(f"⚠️ KNN Imputation supports only numeric columns. Ignored: {cols_non_numeric}")
        
        k = st.slider(get_text('imputation', 'n_neighbors'), 1, 20, 5, key=f"num_k_slider_{key}")
        imputer = KNNImputer(n_neighbors=k)

    elif method == "iterative":
        if cols_non_numeric:
            st.warning(f"⚠️ Iterative Imputation supports only numeric columns. Ignored: {cols_non_numeric}")
        
        max_iter = st.slider(get_text('imputation', 'max_iter'), 1, 50, 10, key=f"num_iter_slider_{key}")
        imputer = IterativeImputer(max_iter=max_iter, random_state=0)

    # 5. Row Limit
    row_limit = st.number_input(
        get_text('imputation', 'rows_limit'),
        min_value=0,
        max_value=len(df),
        value=0,
        help=get_text('imputation', 'rows_limit_help'),
        key=f"num_row_limit_{key}"
    )

    # 6. Applica
    if st.button(get_text('imputation', 'apply_btn'), key=f"num_apply_btn_{key}"):
        try:
            target_df = df.copy()
            if row_limit > 0:
                target_df = target_df.head(row_limit)
            
            valid_cols = cols_to_impute
            if method in ["knn", "iterative"]:
                valid_cols = cols_numeric
            
            if not valid_cols:
                st.error("No valid columns for this method.")
                return

            data_to_impute = target_df[valid_cols]
            
            with st.spinner("Imputing..."):
                imputed_data = imputer.fit_transform(data_to_impute)
                imputed_df = pd.DataFrame(imputed_data, columns=valid_cols, index=target_df.index)
                target_df[valid_cols] = imputed_df
                
                st.success(get_text('imputation', 'success'))
                st.write(get_text('imputation', 'preview'))
                st.dataframe(target_df[valid_cols].head(10))
                
                if row_limit == 0 or row_limit == len(df):
                    st.session_state[f'imputed_df_cache_{key}'] = target_df
                else:
                    st.warning("Imputation performed on partial dataset. To save, set limit to 0 (all rows).")

        except Exception as e:
            st.error(get_text('imputation', 'error', e=e))


def _render_llm_imputation(df, missing_cols, key):
    # Filtra solo colonne object/string che hanno missing
    text_cols_missing = [c for c in missing_cols if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c])]
    
    if not text_cols_missing:
        st.warning(get_text('imputation', 'no_text_cols'))
        return

    # 1. Selezione Colonne
    cols_to_impute = st.multiselect(
        get_text('imputation', 'select_cols'),
        options=text_cols_missing,
        default=text_cols_missing[:1],
        key=f"llm_cols_sel_{key}"
    )
    
    if not cols_to_impute:
        return

    # 2. Selezione Backend e Modello
    c1, c2 = st.columns(2)
    with c1:
        backend = st.selectbox(get_text('imputation', 'select_backend'), options=list(llm_adapters.LLM_ADAPTERS.keys()), key=f"llm_backend_sel_{key}")
    
    with c2:
        # Carica modelli dinamicamente
        models = []
        if backend:
            try:
                # Usa config salvata se esiste, altrimenti default
                cfg = st.session_state.get(f"lm_selector__cfg_by_backend", {}).get(backend, {})
                models = llm_adapters.list_models(backend, **cfg)
                if isinstance(models, dict) and 'error' in models:
                    models = []
            except:
                models = []
        
        model_options = [m if isinstance(m, str) else (m.get('id') or m.get('name')) for m in models] if isinstance(models, list) else []
        model = st.selectbox(get_text('imputation', 'select_model'), options=model_options, key=f"llm_model_sel_{key}")

    # 3. Avvio Imputazione
    if st.button(get_text('imputation', 'start_imputation'), key=f"llm_start_btn_{key}"):
        if not model:
            st.error(get_text('imputation', 'no_model_selected'))
            return
        
        st.session_state[f'imputation_queue_{key}'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Trova righe da imputare
        # Limitiamo a max 20 righe per demo/performance se non specificato diversamente?
        # L'utente non ha chiesto limite esplicito qui, ma è "testo lungo", quindi lento.
        # Facciamo tutte le righe con missing nelle colonne selezionate.
        
        rows_to_process = []
        for col in cols_to_impute:
            missing_indices = df[df[col].isna()].index.tolist()
            for idx in missing_indices:
                rows_to_process.append((idx, col))
        
        total = len(rows_to_process)
        completed = 0
        
        # Configurazione LLM
        llm_kwargs = st.session_state.get(f"lm_selector__cfg_by_backend", {}).get(backend, {})

        for idx, col in rows_to_process:
            status_text.text(get_text('imputation', 'imputing_progress', current=completed+1, total=total))
            
            # Context Row
            row_data = df.loc[idx].to_dict()
            # Rimuovi il valore missing corrente per chiarezza nel prompt
            row_data[col] = "MISSING"
            
            # Esempi (5 righe non missing per questa colonna)
            examples_df = df[df[col].notna()].sample(n=min(5, len(df[df[col].notna()])), random_state=42)
            examples_str = ""
            for _, ex_row in examples_df.iterrows():
                examples_str += f"- {ex_row.to_dict()}\n"
            
            prompt = f"""
            Task: Impute the missing value for column '{col}'.
            
            Context Row (JSON):
            {row_data}
            
            Similar Examples (JSON):
            {examples_str}
            
            Instructions:
            - Analyze the context and examples.
            - Predict the most likely value for '{col}' in the Context Row.
            - Return ONLY the value. Do not add quotes or explanations.
            """
            
            try:
                # Chiamata LLM
                imputed_val = llm_adapters.generate(
                    backend=backend,
                    prompt=prompt,
                    model_name=model,
                    max_tokens=50, # Breve
                    **llm_kwargs
                )
                
                st.session_state.setdefault(f'imputation_queue_{key}', []).append({
                    "index": idx,
                    "column": col,
                    "original": None, # Era NaN
                    "imputed": imputed_val.strip(),
                    "row_data": row_data
                })
            except Exception as e:
                st.error(f"Error imputing row {idx}, col {col}: {e}")
            
            completed += 1
            progress_bar.progress(completed / total)
            
        st.success("Imputation generation complete. Please review below.")
        st.rerun()

    # 4. Review UI
    if f'imputation_queue_{key}' in st.session_state and st.session_state[f'imputation_queue_{key}']:
        st.divider()
        st.subheader(get_text('imputation', 'review_header'))
        
        queue = st.session_state[f'imputation_queue_{key}']
        
        # Bottoni Globali
        c_all_1, c_all_2 = st.columns(2)
        if c_all_1.button(get_text('imputation', 'accept_all'), type="primary", key=f"accept_all_{key}"):
            for item in queue:
                df.at[item['index'], item['column']] = item['imputed']
            st.session_state[f'imputation_queue_{key}'] = []
            st.success("All changes accepted.")
            st.rerun()
            
        if c_all_2.button(get_text('imputation', 'reject_all'), key=f"reject_all_{key}"):
            st.session_state[f'imputation_queue_{key}'] = []
            st.warning("All changes rejected.")
            st.rerun()
            
        st.divider()
        
        # Lista items
        # Usiamo un indice per rimuovere elementi dalla lista man mano
        indices_to_remove = []
        
        for i, item in enumerate(queue):
            with st.container(border=True):
                c_info, c_act = st.columns([3, 1])
                with c_info:
                    st.markdown(f"**Row {item['index']} - Column `{item['column']}`**")
                    # Mostra riga completa evidenziando la cella
                    # Creiamo un df di una riga per visualizzazione
                    disp_row = item['row_data'].copy()
                    disp_row[item['column']] = f"✨ {item['imputed']} (was NaN)"
                    st.json(disp_row, expanded=False)
                    st.markdown(f"**Proposed Value:** `{item['imputed']}`")
                
                with c_act:
                    if st.button(get_text('imputation', 'accept'), key=f"acc_{i}_{key}"):
                        df.at[item['index'], item['column']] = item['imputed']
                        indices_to_remove.append(i)
                        st.rerun() # Rerun necessario per aggiornare coda
                    
                    if st.button(get_text('imputation', 'reject'), key=f"rej_{i}_{key}"):
                        indices_to_remove.append(i)
                        st.rerun()

        # Pulizia coda (se gestita senza rerun immediato, ma qui usiamo rerun per semplicità)
        if indices_to_remove:
            # Rimuovi in ordine inverso per non sballare indici
            for i in sorted(indices_to_remove, reverse=True):
                del st.session_state[f'imputation_queue_{key}'][i]
            st.rerun()
