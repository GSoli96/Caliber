
import streamlit as st

# Page configuration
st.set_page_config(
    page_icon="🌱",
    page_title="CALIBER - Thank You",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .big-title {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for dialog
if 'show_closing_dialog' not in st.session_state:
    st.session_state['show_closing_dialog'] = True

# Main title with logo
st.markdown('<div class="big-title">🌳 CALIBER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Carbon-Aware LLM-Integrated Benchmarking and Eco-friendly Rewriting</div>', unsafe_allow_html=True)

# Center button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("📊 Show Summary", use_container_width=True, type="primary"):
        st.session_state['show_closing_dialog'] = True

# Closing dialog
@st.dialog("🌱 Thank You for Your Attention!", width="large")
def show_closing_dialog():
    """Display closing dialog with key features and insights."""
    
    st.markdown("""
    ### 🎯 CALIBER: Key Takeaways
    
    Thank you for exploring **CALIBER** - the first comprehensive tool that brings together 
    **AI-powered SQL generation** and **environmental sustainability**.
    """)
    
    st.divider()
    
    st.markdown("### 💡 Why CALIBER Matters")
    # Key Features in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🌍 Real-time CO₂ Monitoring**
        - Track energy consumption for every operation
        - Visualize environmental impact in real-time
        
        **🔄 Green Query Optimization**
        - Automatic query rewriting
        - Maintains semantic equivalence
        
        **🤖 Multi-LLM Support**
        - Hugging Face, Ollama, LM Studio
        - Custom model uploads
        """)
    
    with col2:
        st.markdown("""        
        **⚖️ Comprehensive Benchmarking**
        - Compare LLMs on sustainability
        - Green AI Race competitions
        
        **🧬 Sustainable Data Generation**
        - Multiple synthesis strategies
        - Energy impact tracking
        
        **🔍 Intelligent Analytics**
        - PII detection & relational profiling
        - Interactive visualizations
        """)
  
    st.divider()
    
    # Future Work - Simplified and Visual
    st.markdown("### 🚀 Future Work")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("""
        - 🤖 **More LLM & DBMS backends**
        - 📊 **Richer benchmark suites**
        - 🌍 **Real-world workloads**
        """)
    
    with col_f2:
        st.markdown("""
        - 🔬 **Fine-grained hardware measurements**
        - 🌐 **External carbon-intensity data**
        - 👥 **User studies with practitioners**
        """)
    
    st.divider()
    
    # GitHub Repository
    col_git1, col_git2, col_git3 = st.columns([1, 2, 1])
    with col_git2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px;'>
            <h3>⭐ Star us on GitHub!</h3>
            <a href='https://github.com/GSoli96/Caliber.git' target='_blank' style='text-decoration: none;'>
                <button style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer;'>
                    🔗 github.com/GSoli96/Caliber
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Thank you message - Concise
    st.markdown("""
    <div style='text-align: center; padding: 30px;'>
        <h2>🙏 Thank You!</h2>
        <h3 style='color: #667eea;'>Making AI Greener, One Query at a Time 🌱</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Close button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("✅ Close", use_container_width=True, type="primary"):
            st.session_state['show_closing_dialog'] = False
            st.rerun()

# Show dialog if flag is set
if st.session_state.get('show_closing_dialog', False):
    show_closing_dialog()

# Footer with statistics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌍 CO₂ Tracked", "Real-time", help="Monitor every operation's carbon footprint")
with col2:
    st.metric("🤖 LLM Backends", "4+", help="Hugging Face, Ollama, LM Studio, Custom")
with col3:
    st.metric("💾 DBMS Supported", "4+", help="MySQL, PostgreSQL, SQLite, DuckDB")
with col4:
    st.metric("📊 Avg. CO₂ Reduction", "30-50%", help="With Greenify optimization")
