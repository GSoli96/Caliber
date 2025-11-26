# GUI/sidebar.py
import time
import streamlit as st
from utils.system_monitor_utilities import get_static_system_info, get_dynamic_system_info
from utils.translations import get_text

def render_detailed_sidebar(sysinfo_static, sysinfo_dynamic):
    """
    Renderizza il contenuto dettagliato della sidebar, disegnando i componenti
    direttamente con i valori corretti per evitare il flicker.
    """
    st.header(get_text("sidebar", "system_info"))
    st.markdown("<hr style='border: none; border-top: 0.5px solid #ccc;'>", unsafe_allow_html=True)

    # --- CPU ---
    st.markdown(f"**{get_text('sidebar', 'cpu')}**", unsafe_allow_html=True)
    st.markdown(f"**{get_text('sidebar', 'cores')}** {sysinfo_static['cpu']['Cores']}\n\n**{get_text('sidebar', 'frequency')}** {sysinfo_static['cpu']['Frequency']}")
    st.markdown(f"**{get_text('sidebar', 'tdp_max')}** {sysinfo_static['cpu']['TDP_W']}")
    st.markdown(f"**{get_text('sidebar', 'cpu_watt')}** {sysinfo_dynamic['cpu']['power_w']:.2f} W")
    st.markdown(f"**{get_text('sidebar', 'usage')}** {sysinfo_dynamic['cpu']['Utilizzo']}")
    st.progress(min(100, int(sysinfo_dynamic['cpu']['percent'])))
    st.markdown("<hr style='margin:5px 0; border: none; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)

    # --- RAM ---
    st.markdown(f"**{get_text('sidebar', 'ram')}**", unsafe_allow_html=True)
    st.markdown(f"**{get_text('sidebar', 'total')}** {sysinfo_static['ram']['Totale']}")
    st.markdown(f"**{get_text('sidebar', 'available')}** {sysinfo_dynamic['ram']['Disponibile']}")
    st.markdown(f"**{get_text('sidebar', 'usage')}** {sysinfo_dynamic['ram']['Utilizzo']}")
    st.progress(min(100, int(sysinfo_dynamic['ram']['percent'])))
    st.markdown("<hr style='margin:5px 0; border: none; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)

    # --- GPU ---
    if "gpu" in sysinfo_static and "gpu" in sysinfo_dynamic:
        st.markdown(f"**{get_text('sidebar', 'gpu')}**", unsafe_allow_html=True)
        st.markdown(
            f"**{get_text('sidebar', 'name')}** {sysinfo_static['gpu']['Nome']}\n\n**{get_text('sidebar', 'memory_total')}** {sysinfo_static['gpu']['Memoria Totale']}")
        st.markdown(f"**{get_text('sidebar', 'available')}** {sysinfo_dynamic['gpu']['Memoria Disponibile']}")
        st.markdown(f"**{get_text('sidebar', 'gpu_watt')}** {sysinfo_dynamic['gpu']['power_w']:.2f} W")
        st.markdown(f"**{get_text('sidebar', 'usage')}** {sysinfo_dynamic['gpu']['Utilizzo']}")
        st.progress(min(100, int(sysinfo_dynamic['gpu']['percent'])))
        st.markdown("<hr style='margin:5px 0; border: none; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)

    # --- CO₂ Emissioni (Istantanee)---
    st.markdown(f"**{get_text('sidebar', 'co2_instant')}**", unsafe_allow_html=True)
    st.markdown(f"**{get_text('sidebar', 'cpu_gs')}** {sysinfo_dynamic['cpu']['co2_gs_cpu']:.4f} g/s")
    st.markdown(f"**{get_text('sidebar', 'cpu_kgh')}** {sysinfo_dynamic['cpu']['co2_kgh_cpu']:.4f} kg/h")

    if "gpu" in sysinfo_dynamic:
        st.markdown(f"**{get_text('sidebar', 'gpu_gs')}** {sysinfo_dynamic['gpu']['co2_gs_gpu']:.4f} g/s")
        st.markdown(f"**{get_text('sidebar', 'gpu_kgh')}** {sysinfo_dynamic['gpu']['co2_kgh_gpu']:.4f} kg/h")

    st.markdown("<hr style='margin:5px 0; border: none; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)

    # Recupera i totali dal session_state, inizializzandoli a 0 se non esistono
    total_cpu_g = st.session_state.get("total_cpu_g", 0.0)
    total_gpu_g = st.session_state.get("total_gpu_g", 0.0)
    total_co2_g = st.session_state.get("total_co2_g", 0.0)

    # --- CO₂ Emissioni (Istantanee)---
    st.markdown(f"**{get_text('sidebar', 'co2_total')}**", unsafe_allow_html=True)

    # Mostra valori formattati come "g" (quantità cumulative), non ratei
    st.markdown(f"**{get_text('sidebar', 'cpu_total')}** {total_cpu_g:.4f} g")

    if "gpu" in sysinfo_dynamic:
        st.markdown(f"**{get_text('sidebar', 'gpu_total')}** {total_gpu_g:.4f} g")

    st.markdown(f"**{get_text('sidebar', 'total_consumption')}** {total_co2_g:.4f} g/s")

    st.markdown("<hr style='margin:5px 0; border: none; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)


def create_sidebar():
    """Funzione principale per la creazione e l'aggiornamento della sidebar."""

    st.markdown("""
    <style>
    [data-testid="stAppViewBlock"] > div[style*="background-color: rgba(0, 0, 0, 0.4)"],
    [data-testid="stSidebar"] div[style*="background-color: rgba(0, 0, 0, 0.4)"] {
        opacity: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # Rimosso st_autorefresh per evitare aggiornamenti non controllati
        sysinfo_static = get_static_system_info()
        sysinfo_dynamic = get_dynamic_system_info()

        render_detailed_sidebar(sysinfo_static, sysinfo_dynamic)

        # --- Stato per integrazione temporale ---
        now = time.time()
        if 'last_ts' not in st.session_state:
            st.session_state.last_ts = now
            st.session_state.total_cpu_g = 0.0
            st.session_state.total_gpu_g = 0.0

        dt = max(0.001, now - st.session_state.last_ts)  # secondi; evita zero
        st.session_state.last_ts = now

        # Ratei istantanei
        cpu_gs = float(sysinfo_dynamic['cpu']['co2_gs_cpu'])
        gpu_gs = float(sysinfo_dynamic['gpu']['co2_gs_gpu']) if "gpu" in sysinfo_dynamic else 0.0

        # Integrazione: g = (g/s) * s
        st.session_state.total_cpu_g += cpu_gs * dt
        st.session_state.total_gpu_g += gpu_gs * dt

        total_g = st.session_state.total_cpu_g + st.session_state.total_gpu_g

        if 'total_co2_g' not in st.session_state:
            st.session_state.total_co2_g = 0.0

        # Aggiungi l'incremento al totale della sessione
        st.session_state.total_co2_g = total_g