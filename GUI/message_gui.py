import streamlit as st


def st_toast_temp(message: str, msg_type: str = "success"):
    """
    Mostra un toast temporaneo in Streamlit.

    Args:
        message (str): Testo del messaggio.
        msg_type (str): Tipo di messaggio: "success", "info", "warning", "error".
    """
    icons = {
        "success": "✅",
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌"
    }
    icon = icons.get(msg_type, "ℹ️")

    st.toast(message, icon=icon)


# Esempio d'uso
# st_toast_temp("✅ Operazione completata!", msg_type="success")
# st_toast_temp("⚠️ Attenzione, qualcosa non va", msg_type="warning")
# st_toast_temp("ℹ️ Informazione importante", msg_type="info")
# st_toast_temp("❌ Errore critico", msg_type="error")
