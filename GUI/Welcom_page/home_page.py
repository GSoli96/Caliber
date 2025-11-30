import streamlit as st

# Configurazione della pagina
st.set_page_config(
    page_icon="🌳",
    page_title="CALIBER - Making AI Greener",
    layout="wide"
)

# CSS personalizzato per styling premium
st.markdown("""
    <style>
    /* Sfondo gradiente verde */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    
    /* Rimuovi padding superiore di Streamlit */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Nascondi header Streamlit */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Rimuovi toolbar */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Container principale */
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 0;
        text-align: center;
        padding: 0rem 0rem;
    }
    
    /* Header container per logo + titolo */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
        width: 100%;
    }
    
    /* Logo e titolo */
    .logo {
        font-size: 130px;
        animation: float 3s ease-in-out infinite;
        margin: 0rem;
        padding: 0rem;
        margin-left: 9rem;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .app-title {
        font-size: 15rem;
        font-weight: 1000;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 3px;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.3);
        margin: 0rem;
        padding: 0rem;
        line-height: 1;
        text-align: center;
        animation: float 3s ease-in-out infinite;
    }
    
    .motto {
        font-size: 2.5rem;
        color: #a8dadc;
        font-weight: 600;
        margin-bottom: 1rem;
        margin-top: 0rem;
        padding-top: 0rem;
        font-style: italic;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    
    /* Features grid */
    .features-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        max-width: 1200px;
        margin: 1rem auto;
        padding: 0 2rem;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 0rem;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 1rem;
        color: #b0b0b0;
        line-height: 1.6;
    }
    
    /* Footer */
    .footer {
        margin: 0rem;
        padding: 0.5rem;
        text-align: center;
        color: #888;
        font-size: 1rem;
    }
    
    /* Animazione particelle */
    .particle {
        position: fixed;
        font-size: 2rem;
        opacity: 0.3;
        animation: rise 10s infinite ease-in;
    }
    
    @keyframes rise {
        0% {
            transform: translateY(100vh) rotate(0deg);
            opacity: 0;
        }
        50% {
            opacity: 0.3;
        }
        100% {
            transform: translateY(-100vh) rotate(360deg);
            opacity: 0;
        }
    }

    .feature-title {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    </style>
""", unsafe_allow_html=True)

# Particelle animate di sfondo
particles_html = ""
for i in range(100):
    left = i * 10
    delay = i * 0.5
    particles_html += f'<div class="particle" style="left: {left}%; animation-delay: {delay}s;">🌱</div>'

st.markdown(particles_html, unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 3, 3])
with col1:
    st.markdown("<div></div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="logo">🌳</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">CALIBER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="motto">Making AI Greener, One Query at a Time 🌱</p>', unsafe_allow_html=True)
with col3:
    st.markdown("<div></div>", unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)



col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    st.markdown("""
    <div class="features-container">
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">🌍</span>
                Eco-Friendly AI
            </div>
            <div class="feature-desc">
                Monitor and reduce CO₂ emissions generated by your SQL queries and AI operations
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="features-container">
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">⚡</span>
                Query Optimization
            </div>
            <div class="feature-desc">
                Automatically optimize queries to maximize performance and minimize environmental impact
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
     st.markdown("""
    <div class="features-container">
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">📊</span>
                Advanced Analytics
            </div>
            <div class="feature-desc">
                In-depth dataset analysis with interactive visualizations and intelligent insights
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("""
    <div class="features-container">
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">🧠</span>
                Multi-LLM Support
            </div>
            <div class="feature-desc">
                Support for LM Studio, Ollama, Hugging Face, and custom models
                <br>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
   

with col2:
    st.markdown("""
    <div class="features-container">
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">🎯</span>
                Benchmarking
            </div>
            <div class="feature-desc">
                Compare performance, quality, and environmental impact of different models and configurations
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="features-container">
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">🧬</span>
                Synthetic Data
            </div>
            <div class="feature-desc">
                Generate realistic synthetic datasets for testing and development with guaranteed privacy
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>🌳 CALIBER - Carbon-Aware LLM-Integrated Benchmarking & Eco-Responsible Query Rewriting</p>
        <p>Developed with 💚 for a sustainable future</p>
    </div>
""", unsafe_allow_html=True)
