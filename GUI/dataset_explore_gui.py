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
    # 1. Boolean
    if pd.api.types.is_bool_dtype(s):
        return "boolean"

    # 2. Datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    # 3. Numeric "pulito"
    if pd.api.types.is_numeric_dtype(s):
        uniq = s.nunique(dropna=True)
        if uniq <= 20:
            return "categorical"
        return "numeric"

    # 4. Tentativo: numeric "sporco" (object con numeri)
    if s.dtype == "object":
        s_numeric = pd.to_numeric(s, errors="coerce")
        percent_numeric = s_numeric.notna().mean()

        # Se almeno il 60% è numerico → è numerico
        if percent_numeric >= 0.6:
            uniq = s_numeric.nunique(dropna=True)
            if uniq <= 20:
                return "categorical"
            return "numeric"

        # Altrimenti è testo / categorico
        uniq = s.nunique(dropna=True)
        if uniq <= 20:
            return "categorical"
        return "text"

    # 5. Tutto il resto
    return "other"

def info_dataset(df, key):
    """
    Renderizza un Dataset Explorer migliorato.
    Mostra un'anteprima del DataFrame con un'intestazione ricca di informazioni
    e un expander con statistiche dettagliate.
    """
    tab1, tab2, tab3  = st.tabs([
        '🧩 Schema & Column Insights',
        '🧮 Descriptive Statistics',
        get_text("load_dataset", "tab_correlations")
    ])

    with tab1:
        with st.container(border=True):
            # -------- UI --------
            st.markdown(f"##### {get_text('load_dataset', 'legend')}")
            st.markdown(f"""
                    <div style='line-height: 1; font-size: 0.85rem;'>
    
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

        rows = []

        for col in df.columns:
            s = df[col]
            cat = get_column_category(s)
            tinfo = TYPE_INFO.get(cat, TYPE_INFO["other"])

            miss_pct = float(s.isna().mean() * 100.0)

            uniq = s.nunique(dropna=True)
            total = s.count()  # conta solo i non-NaN

            uniqs_pct = (uniq / total) * 100

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
                get_text("load_dataset", "unique_pct"): uniqs_pct,
                get_text("load_dataset", "missing_pct"): miss_pct,
                get_text("load_dataset", "sensitive"): "🔒" if is_sens else "—",
                get_text("load_dataset", "reason"): reason,
                get_text("load_dataset", "example"): sample_val,
                "tipo_color": tinfo["color"],  # per lo styling (non visibile all’utente)
            })

        schema = pd.DataFrame(rows)

        schema = schema.sort_values(by=[get_text("load_dataset", "sensitive"),
                                        get_text("load_dataset", "missing_pct")],
                                    ascending=[False, False]).reset_index(drop=True)

        schema = schema.sort_values(by=
        [
            get_text("load_dataset", "column"), ],
            ascending=[True]
        ).reset_index(drop=True)

        schema.drop(columns=[get_text("load_dataset", "type")], inplace=True)

        df_union = schema[[
            get_text("load_dataset", "column"),
            get_text("load_dataset", "unique_pct"),
            get_text("load_dataset", "missing_pct"),
            get_text("load_dataset", "sensitive"),
            get_text("load_dataset", "reason"),
            get_text("load_dataset", "example"),
            "tipo_color"]]

        # Progress bar per i missing nella tabella di schema
        st.dataframe(
            style_schema(df_union),
            column_config={
                get_text("load_dataset", "missing_pct"): st.column_config.ProgressColumn(
                    get_text("load_dataset", "missing_pct"),
                    # help="Percentuale missing valori NaN/null",
                    min_value=0.0, max_value=100.0, format="%.1f%%", width='medium'
                ),
                get_text("load_dataset", "unique_pct"): st.column_config.ProgressColumn(
                    get_text("load_dataset", "unique_pct"),
                    # help="Percentuale di valori NaN/null",
                    min_value=0.0, max_value=100.0, format="%.1f%%", width='medium'
                ),
                get_text("load_dataset", "sensitive"): st.column_config.TextColumn(
                    get_text("load_dataset", "sensitive"),
                    help="🔒 = colonna potenzialmente sensibile",
                    width="small",
                ),
                get_text("load_dataset", "column"): st.column_config.TextColumn(
                    get_text("load_dataset", "column"),
                    width="medium",
                ),
                get_text("load_dataset", "reason"): st.column_config.TextColumn(
                    get_text("load_dataset", "reason"),
                    width="large",
                ),
                get_text("load_dataset", "example"): st.column_config.TextColumn(
                    get_text("load_dataset", "example"),
                    width="medium",
                ),


            },
            hide_index=True,
            width='content',
        )

    with tab2:
        categories = {col: get_column_category(df[col]) for col in df.columns}
        numeric_cols = [col for col, cat in categories.items() if cat == "numeric"]
        text_cols = [col for col, cat in categories.items() if cat == "text"]

        # ====================
        #     NUMERICHE
        # ====================
        st.markdown("### 🔢 Colonne numeriche")
        if numeric_cols:
            stats = df[numeric_cols].describe()

            # Arrotondamenti
            stats.loc["mean"] = stats.loc["mean"].round(2)
            stats.loc["std"] = stats.loc["std"].round(2)

            # Stile tabella
            st.dataframe(
                stats.style.format("{:.2f}")
                .set_table_styles([
                    {"selector": "th", "props": [("background-color", "#f0f2f6"), ("font-weight", "bold")]},
                    {"selector": "tbody tr:hover", "props": [("background-color", "#f5f5f5")]}
                ])
            )
        else:
            st.info("Nessuna colonna numerica trovata.")

        # ====================
        #     TESTUALI
        # ====================
        st.markdown("### 📝 Colonne testuali")

        if text_cols:
            df_text = df[text_cols].copy()

            # Tabella delle statistiche testuali
            desc_text = df_text.describe(include="object").T  # Trasposta per una visualizzazione migliore

            # Aggiungo un conteggio parole medio (bonus estetico)
            desc_text["avg_len"] = df_text.apply(lambda col: col.astype(str).str.len().mean()).round(1)

            # Mostra la tabella in modo elegante
            st.dataframe(
                desc_text.style
                .set_table_styles([
                    {"selector": "th", "props": [("background-color", "#f0f2f6"), ("font-weight", "bold")]},
                    {"selector": "tbody tr:hover", "props": [("background-color", "#f9f9f9")]}
                ])
            )
        else:
            st.info("Nessuna colonna testuale trovata.")

    # ---------- CORRELAZIONI ----------
    with tab3:
        col1, col2 = st.columns(2)



        num_df = df.select_dtypes(include=[np.number])

        if num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True)

            # evidenzia con background gradient
            corr_style = corr.style.background_gradient(cmap="seismic",
                                                        vmin=-1,
                                                        vmax=1)
            with col1:
                st.markdown(f"#### {get_text('load_dataset', 'corr_matrix')}")
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
            corr_pairs["|corr|"] = corr_pairs["|corr|"].apply(lambda x: round(x, 2))

            with col2:
                st.markdown(f"#### {get_text('load_dataset', 'top_corr_pairs')}")
                st.dataframe(corr_pairs.head(10), hide_index=True, width='stretch')
        else:
            st.info(get_text("load_dataset", "no_num_cols"))

def style_schema(df_schema: pd.DataFrame):
    # Crea la mappa Colonna -> colore
    if "tipo_color" in df_schema.columns:
        color_map = df_schema.set_index(
            get_text("load_dataset", "column"))["tipo_color"].to_dict()
    else:
        color_map = {}

    df_visible = df_schema.drop(columns=["tipo_color"], errors="ignore")

    def color_colname(val):
        return f"color: {color_map.get(val, 'inherit')}; font-weight:600"

    # Applica lo stile solo alla colonna "Colonna"
    styler = df_visible.style.map(color_colname, subset=[
        get_text("load_dataset", "column")])
    return styler




