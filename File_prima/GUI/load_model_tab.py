import streamlit as st
from GUI.conf_model import configure_local_model_tab, configure_online_model
from utils.translations import get_text

def load_model_tab():
    tab1, tab2 = st.tabs([get_text("load_model", "local"), get_text("load_model", "online")])
    with tab1:
        configure_local_model_tab(key_prefix='configure_local_model')
    with tab2:
        configure_online_model(key_prefix='configure_online_model')