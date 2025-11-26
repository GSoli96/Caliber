import streamlit as st


# === Tema dinamico (CSS) ===
def apply_theme():
    theme = st.session_state.get("theme", "Light")
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)


def apply_language():
    pass


from utils.translations import get_text

def settings_tab():
    st.markdown(get_text("settings", "header"))

    # --- Lingua ---
    st.subheader(get_text("settings", "language"))
    with st.container(border=True, horizontal=False):
        language = st.selectbox(
            get_text("settings", "select_language"),
            ["Italiano", "English"],
            index=0 if st.session_state.get("language", "Italiano") == "Italiano" else 1,
            key="language_select",
            on_change=lambda: st.session_state.update({"language": st.session_state.language_select})
        )
        if st.session_state.language_select:
            st.session_state["language"] = language
            st.info(f"{get_text('settings', 'current_language')} **{st.session_state['language']}**")


    # --- Tema ---
    st.subheader(get_text("settings", "theme"))
    with st.container(border=True):
        st.info(get_text("settings", "theme_info"))

    # --- Configurazione Stime CO₂ ---
    st.subheader(get_text("settings", "co2_config"))
    with st.container(border=True):
        st.number_input(
            get_text("settings", "emission_factor"),
            value=st.session_state.get('emission_factor', 400.0),
            key="emission_factor_input",
            on_change=lambda: st.session_state.update(
                {'emission_factor': st.session_state.emission_factor_input}
            )
        )

        st.number_input(
            get_text("settings", "cpu_tdp"),
            value=st.session_state.get('cpu_tdp', 65.0),
            key="cpu_tdp_input",
            on_change=lambda: st.session_state.update(
                {'cpu_tdp': st.session_state.cpu_tdp_input}
            )
        )

    # --- Configurazione Database ---
    st.subheader(get_text("settings", "db_config"))
    with st.container(border=True):
        st.text_input(
            get_text("settings", "db_dir"),
            value=st.session_state.get('db_dir', "Database"),
            key="db_dir_input",
            on_change=lambda: st.session_state.update(
                {'db_dir': st.session_state.db_dir_input}
            )
        )

    # # --- Pulsante Salva ---
    # st.divider()
    # if st.button("💾 Salva Impostazioni"):
    #     st.session_state["settings_saved"] = True
    #     st.success("Impostazioni salvate con successo!")
    #
    # # --- Mostra riepilogo ---
    # if st.session_state.get("settings_saved"):
    #     st.markdown("### 🔧 Riepilogo Impostazioni Correnti")
    #
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.markdown(f"**🌐 Lingua:**  {st.session_state.get('language', 'Italiano')}")
    #         st.markdown(f"**🎨 Tema:**  {st.session_state.get('theme', 'Light')}")
    #     with col2:
    #         st.markdown(f"**🌱 Fattore Emissione:**  {st.session_state.get('emission_factor')} gCO₂/kWh")
    #         st.markdown(f"**⚡ CPU TDP:**  {st.session_state.get('cpu_tdp')} W")
    #
    #     st.info("Le modifiche saranno applicate alla prossima sessione o refresh dell’app.")
