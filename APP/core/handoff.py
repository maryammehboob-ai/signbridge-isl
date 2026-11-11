"""Small helpers that let the existing pages work inside the combined app.

- no_page_config(): your Text-to-Sign and Speech-to-Text pages call st.set_page_config(), which
  Streamlit allows only once per app (app.py already does it). This switches it off while an
  existing page runs, so those files can be reused without changing them.
- send_to_text_to_sign(): puts text into the Text-to-Sign page and converts it, so the signs are
  ready when that page opens.
"""
import contextlib

from core import PROJECT_DIR  # noqa: F401  (sets the import paths)

TEXT_TO_SIGN_PAGE = "views/page_text_to_sign.py"


@contextlib.contextmanager
def no_page_config():
    import streamlit as st
    original = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        yield
    finally:
        st.set_page_config = original


def send_to_text_to_sign(text, state=None):
    """Fill the Text-to-Sign page (same session keys as TEXT_TO_SIGN/text_to_sign.py) and convert."""
    import streamlit as st
    import text_to_sign as t2s
    from core import signs
    state = st.session_state if state is None else state
    state["text_input"] = text
    clean, notes = signs.normalize_text(text)
    tokens, warnings = t2s.text_to_tokens(clean)
    state["tokens"], state["warnings"], state["t2s_notes"] = tokens, warnings, notes
    state["index"], state["show_full"] = 0, False
    state["converted"] = bool(t2s.displayable(tokens))
    state["message"] = "" if state["converted"] else "Nothing in this text can be shown as a sign (use letters A-Z or numbers 1-9)."
    return state["converted"]
