"""Streamlit pieces for reading text aloud (voice choice, Speak button, the audio player)."""
import streamlit as st

from core import tts


def voice_controls(prefix):
    """Voice and speed selectors. Returns (voice name or None, rate)."""
    names = list(tts.voices())
    if not names:
        st.caption("Reading aloud is not available in the online version (it uses voices installed with Windows).")
        return None, 0
    left, right = st.columns(2)
    voice = left.selectbox("Voice", names, key=f"{prefix}_voice")
    rate = right.slider("Speed", -5, 5, 0, key=f"{prefix}_rate", help="0 is normal speed; negative is slower.")
    return voice, rate


def speak(prefix, text, voice, rate):
    """Make the speech now and keep it for the next draw of the player. Sets a message on failure."""
    state = st.session_state
    state[f"{prefix}_audio"], state[f"{prefix}_speak_error"] = None, ""
    try:
        state[f"{prefix}_audio"] = tts.speak_wav(text, voice, rate)
    except tts.SpeechError as problem:
        state[f"{prefix}_speak_error"] = str(problem)


def player(prefix):
    """Show the error or play the audio that speak() prepared (once)."""
    state = st.session_state
    error = state.pop(f"{prefix}_speak_error", "")
    audio = state.pop(f"{prefix}_audio", None)
    if error:
        st.error(error)
    if audio:
        st.audio(audio, format="audio/wav", autoplay=True)
