# ============================================================================
# WELCOME DIALOG
# ============================================================================
import streamlit as st
from utils.translations import get_text

@st.dialog(get_text("welcome", "title"), width="large")
def show_welcome_dialog():
    """Display welcome dialog with tool information and tab descriptions."""
    
    # Subtitle and intro
    st.markdown(get_text("welcome", "subtitle"))
    st.markdown(get_text("welcome", "intro"))
    st.markdown("---")
    
    # Features header
    st.markdown(get_text("welcome", "features_header"))
    
    # Tab descriptions in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(get_text("welcome", "data_hub_title"))
        st.markdown(get_text("welcome", "data_hub_desc"))
        st.write("")
        
        st.markdown(get_text("welcome", "data_insights_title"))
        st.markdown(get_text("welcome", "data_insights_desc"))
        st.write("")
        
        st.markdown(get_text("welcome", "green_query_title"))
        st.markdown(get_text("welcome", "green_query_desc"))
    
    with col2:
        st.markdown(get_text("welcome", "eco_benchmark_title"))
        st.markdown(get_text("welcome", "eco_benchmark_desc"))
        st.write("")
        
        st.markdown(get_text("welcome", "synthetic_lab_title"))
        st.markdown(get_text("welcome", "synthetic_lab_desc"))
        st.write("")
        
        st.markdown(get_text("welcome", "settings_title"))
        st.markdown(get_text("welcome", "settings_desc"))
    
    st.markdown("---")
    
    # Key features
    st.markdown(get_text("welcome", "key_features_header"))
    st.markdown(f"""
    - {get_text("welcome", "feature_monitoring")}
    - {get_text("welcome", "feature_greenify")}
    - {get_text("welcome", "feature_multi_llm")}
    - {get_text("welcome", "feature_benchmarking")}
    - {get_text("welcome", "feature_synthetic")}
    - {get_text("welcome", "feature_analytics")}
    """)
    
    st.markdown("---")
    
    # Getting started
    st.markdown(get_text("welcome", "getting_started_header"))
    st.markdown(f"""
    1. {get_text("welcome", "step1")}
    2. {get_text("welcome", "step2")}
    3. {get_text("welcome", "step3")}
    4. {get_text("welcome", "step4")}
    5. {get_text("welcome", "step5")}
    """)
    
    st.divider()

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button(get_text("welcome", "start_button"), use_container_width=True, type="primary"):
            st.session_state['show_welcome'] = False
            st.rerun()


