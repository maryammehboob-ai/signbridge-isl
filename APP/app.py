"""SignBridge - ISL Sign Language Studio: one Streamlit app with live Sign-to-Text, Text-to-Sign and Speech-to-Text.

Run (from the project folder):
    .venv_app\\Scripts\\python.exe -m streamlit run APP\\app.py
"""
import streamlit as st

import core  # noqa: F401  (sets the import paths to your existing modules)

st.set_page_config(page_title="SignBridge - ISL Studio", page_icon=":material/sign_language:", layout="wide")

import ui  # noqa: E402  (the look of the app: stylesheet and HTML building blocks; presentation only)

ui.inject()

pages = [
    st.Page("views/page_home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/page_live.py", title="Live Sign-to-Text", icon=":material/videocam:"),
    st.Page("views/page_text_to_sign.py", title="Text-to-Sign", icon=":material/text_fields:"),
    st.Page("views/page_speech_to_text.py", title="Speech-to-Text", icon=":material/mic:"),
    st.Page("views/page_model_performance.py", title="Model Performance", icon=":material/monitoring:"),
]
st.navigation(pages).run()
ui.side_footer()
