"""Text-to-Sign page: type text and see the sign picture for each letter and number.

Same logic as TEXT_TO_SIGN/text_to_sign.py (reused, unchanged); the pictures are embedded in the page so
they cannot be broken by the image server, accented letters are converted, and the signs can be played as
an animated GIF. Every word is fingerspelled letter by letter.
"""
import streamlit as st

import ui
from core import signs, speech_ui

import text_to_sign as t2s

state = st.session_state
for key, default in (("text_input", ""), ("tokens", []), ("warnings", []), ("index", 0), ("show_full", False),
                     ("converted", False), ("message", ""), ("t2s_notes", [])):
    state.setdefault(key, default)

ui.page_header("Text-to-Sign",
               "Type letters (A-Z) and numbers (1-9). Every word is shown **letter by letter** (fingerspelling): "
               "the project has pictures for the 35 signs A-Z and 1-9 only, so there are no whole-word signs. "
               "This is how names and new words are signed.",
               icon_name="signs", eyebrow="Text to signs")


def convert():
    text = state.text_input
    if not text.strip():
        state.tokens, state.warnings, state.converted, state.t2s_notes = [], [], False, []
        state.message = "Please type some text first (letters A-Z or numbers 1-9)."
        return
    clean, notes = signs.normalize_text(text)
    state.tokens, state.warnings = t2s.text_to_tokens(clean)
    state.t2s_notes = notes
    state.index, state.show_full, state.converted, state.message = 0, False, True, ""
    if not t2s.displayable(state.tokens):
        state.converted = False
        state.message = "Nothing in this text can be shown as a sign (use letters A-Z or numbers 1-9)."


def clear():
    state.text_input = ""
    state.tokens, state.warnings, state.t2s_notes, state.index = [], [], [], 0
    state.show_full, state.converted, state.message = False, False, ""


def step(delta):
    steps = t2s.displayable(state.tokens)
    state.index = max(0, min(len(steps) - 1, state.index + delta))
    state.show_full = False


def show_full():
    state.show_full = True


with st.container(border=True, key="t2s_input"):
    st.text_area("Type your text (letters A-Z, numbers 1-9, spaces):", key="text_input", height=100,
                 placeholder="For example: HELLO, MY NAME IS 123, A1B2C3")

    steps = t2s.displayable(state.tokens) if state.converted else []
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("Convert to Sign", on_click=convert, type="primary", use_container_width=True)
    c2.button("Clear Text", on_click=clear, use_container_width=True)
    c3.button("Previous Sign", on_click=step, args=(-1,), disabled=not steps or state.index <= 0, use_container_width=True)
    c4.button("Next Sign", on_click=step, args=(1,), disabled=not steps or state.index >= len(steps) - 1, use_container_width=True)
    c5.button("Display Full Sequence", on_click=show_full, disabled=not steps, use_container_width=True)

if state.message:
    st.warning(state.message)
for note in state.t2s_notes:
    st.info(note)
for warning in state.warnings:
    st.warning(warning)

if steps:
    st.markdown("**Sequence:** " + t2s.sequence_text(state.tokens))
    missing = sorted({t.char for t in steps if t.kind == "sign" and t.image is None})
    if missing:
        st.error("Sign pictures are missing for: " + ", ".join(missing) +
                 ". Run TEXT_TO_SIGN\\build_references.py to create them.")
    if state.show_full:
        st.subheader("Full sequence")
        with st.container(border=True, key="t2s_result"):
            st.html(signs.sequence_html(state.tokens))
    else:
        index = max(0, min(len(steps) - 1, state.index))
        token = steps[index]
        st.subheader(f"Sign {index + 1} of {len(steps)}: {token.label}")
        with st.container(border=True, key="t2s_result"):
            st.html(signs.single_html(token))

    with st.expander("Play as an animated GIF"):
        gif = signs.make_gif(state.tokens)
        if gif is None:
            st.write(f"The GIF is available for up to {signs.MAX_GIF_SIGNS} signs.")
        else:
            st.html(signs.gif_html(gif))
            st.download_button("Download the GIF", gif, file_name="signs.gif", mime="image/gif")

with st.expander("Read the text aloud"):
    voice, rate = speech_ui.voice_controls("ts")
    if st.button("Read aloud", key="ts_read", disabled=not state.text_input.strip()):
        speech_ui.speak("ts", state.text_input, voice, rate)
    speech_ui.player("ts")
ui.footer()
