"""Speech-to-Text page: your existing SPEECH_TO_TEXT/speech_to_text.py, plus a hand-off button."""
import streamlit as st

import ui
from core import handoff, speech_ui

import speech_to_text

ui.eyebrow("mic", "Speech to text")
with handoff.no_page_config():
    speech_to_text.main()

st.divider()
text = st.session_state.get("speech_text", "").strip()
if st.button("Open in Text-to-Sign (one sign at a time)", disabled=not text):
    handoff.send_to_text_to_sign(text)
    st.switch_page(handoff.TEXT_TO_SIGN_PAGE)

st.divider()
st.subheader("4. Read the text aloud")
voice, rate = speech_ui.voice_controls("sp")
if st.button("Read aloud", key="sp_read", disabled=not text):
    speech_ui.speak("sp", text, voice, rate)
speech_ui.player("sp")
