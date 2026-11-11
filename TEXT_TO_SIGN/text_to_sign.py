"""ISL Text-to-Sign (fingerspelling): type text, see the sign picture for each letter / number.

This is fingerspelling only: every letter A-Z and every number 1-9 is shown with its own
hand-sign picture. It does NOT translate whole words into ISL word signs.

Run (from the project folder):
    .venv\\Scripts\\python.exe -m streamlit run TEXT_TO_SIGN\\text_to_sign.py

The first part of this file is plain Python (no Streamlit), so it can be tested without a
browser: see test_text_to_sign.py. The screen (Streamlit) part is in main() at the bottom.
It does not touch the models, the dataset, the labels or the webcam scripts.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE / "alphabet_images"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBERS = "123456789"
SUPPORTED = LETTERS + NUMBERS
SPACE_LABEL = "SPACE"


@dataclass
class Token:
    """One character of the typed text, and what to show for it."""
    char: str                    # the (upper-case) character, or " " for a space
    kind: str                    # "sign" | "space" | "skipped"
    image: Optional[Path] = None # the picture to show (None if missing or skipped)
    note: str = ""               # why it was skipped / what is missing

    @property
    def label(self) -> str:
        return SPACE_LABEL if self.kind == "space" else self.char


def image_path(char: str, image_dir: Path = IMAGE_DIR) -> Path:
    return Path(image_dir) / f"{char}.jpg"


def text_to_tokens(text: str, image_dir: Path = IMAGE_DIR) -> Tuple[List[Token], List[str]]:
    """Turn typed text into tokens and a list of warning messages.

    - letters are made upper-case (a -> A);
    - spaces, tabs and new lines become one SPACE token each (several in a row are kept);
    - 0 and every other character are skipped, with a clear message;
    - a missing picture is reported as a warning and the token has image=None.
    """
    tokens: List[Token] = []
    warnings: List[str] = []
    skipped = {}
    missing = []

    for raw in text or "":
        if raw in " \t\r\n":
            tokens.append(Token(" ", "space"))
            continue
        char = raw.upper()
        if char in SUPPORTED:
            path = image_path(char, image_dir)
            if path.is_file():
                tokens.append(Token(char, "sign", path))
            else:
                kind = "alphabet" if char in LETTERS else "number"
                tokens.append(Token(char, "sign", None, f"missing {kind} image"))
                if char not in missing:
                    missing.append(char)
        elif char == "0":
            tokens.append(Token(char, "skipped", None, "0 has no sign in this dataset"))
            skipped.setdefault("0", "0 has no sign in this dataset (only the numbers 1-9 are supported).")
        elif char.isalpha() or char.isdigit():
            tokens.append(Token(char, "skipped", None, "not an English letter or 1-9"))
            skipped.setdefault(char, f"'{raw}' is not supported (only English letters A-Z and numbers 1-9).")
        else:
            tokens.append(Token(char, "skipped", None, "symbol / punctuation"))
            skipped.setdefault(char, f"'{raw}' is a symbol or punctuation mark and has no sign, so it was skipped.")

    warnings.extend(skipped.values())
    for char in missing:
        kind = "alphabet" if char in LETTERS else "number"
        warnings.append(f"Missing {kind} image for '{char}': expected {image_path(char, image_dir).name} "
                        f"in {Path(image_dir)}. Run build_references.py to create it.")
    return tokens, warnings


def displayable(tokens: List[Token]) -> List[Token]:
    """The tokens that are shown one by one (signs and spaces; skipped characters are left out)."""
    return [t for t in tokens if t.kind != "skipped"]


def sequence_text(tokens: List[Token]) -> str:
    """H -> E -> L -> L -> O   (a space is written as SPACE)."""
    return " \u2192 ".join(t.label for t in displayable(tokens))


def split_words(tokens: List[Token]) -> List[List[Token]]:
    """Group the displayable tokens into words (a SPACE token ends a word)."""
    words, current = [], []
    for token in displayable(tokens):
        if token.kind == "space":
            if current:
                words.append(current)
            current = []
        else:
            current.append(token)
    if current:
        words.append(current)
    return words


# ----------------------------------------------------------------------------------------
# The screen (Streamlit). Everything below only runs when you start the app.
# ----------------------------------------------------------------------------------------
def main():
    import streamlit as st

    st.set_page_config(page_title="ISL Text-to-Sign", layout="wide")
    st.title("ISL Text-to-Sign")
    st.caption("Fingerspelling: each letter (A-Z) and number (1-9) is shown with its own sign picture. "
               "This does not translate whole words into ISL word signs.")

    state = st.session_state
    state.setdefault("text_input", "")
    state.setdefault("tokens", [])
    state.setdefault("warnings", [])
    state.setdefault("index", 0)
    state.setdefault("show_full", False)
    state.setdefault("converted", False)
    state.setdefault("message", "")

    def convert():
        text = state.text_input
        if not text.strip():
            state.tokens, state.warnings, state.converted = [], [], False
            state.message = "Please type some text first (letters A-Z or numbers 1-9)."
            return
        state.tokens, state.warnings = text_to_tokens(text)
        state.index, state.show_full, state.converted = 0, False, True
        state.message = ""
        if not displayable(state.tokens):
            state.converted = False
            state.message = "Nothing in this text can be shown as a sign (use letters A-Z or numbers 1-9)."

    def clear():
        state.text_input = ""
        state.tokens, state.warnings, state.index = [], [], 0
        state.show_full, state.converted, state.message = False, False, ""

    def step(delta):
        steps = displayable(state.tokens)
        state.index = max(0, min(len(steps) - 1, state.index + delta))
        state.show_full = False

    def show_full():
        state.show_full = True

    st.text_area("Type your text (letters A-Z, numbers 1-9, spaces):", key="text_input", height=100,
                 placeholder="For example: HELLO, MY NAME IS 123, A1B2C3")

    steps = displayable(state.tokens) if state.converted else []
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("Convert to Sign", on_click=convert, type="primary", use_container_width=True)
    c2.button("Clear Text", on_click=clear, use_container_width=True)
    c3.button("Previous Sign", on_click=step, args=(-1,), disabled=not steps or state.index <= 0,
              use_container_width=True)
    c4.button("Next Sign", on_click=step, args=(1,), disabled=not steps or state.index >= len(steps) - 1,
              use_container_width=True)
    c5.button("Display Full Sequence", on_click=show_full, disabled=not steps, use_container_width=True)

    if state.message:
        st.warning(state.message)
    for warning in state.warnings:
        st.warning(warning)

    if not steps:
        return

    st.markdown("**Sequence:** " + sequence_text(state.tokens))

    def show_token(token, width):
        if token.kind == "space":
            st.markdown(f"<div style='text-align:center;padding:40px 0;border:2px dashed #888;"
                        f"border-radius:8px'><b>{SPACE_LABEL}</b></div>", unsafe_allow_html=True)
        elif token.image is not None:
            try:
                st.image(str(token.image), caption=token.label, width=width)
            except Exception:                                   # unreadable picture: do not crash
                st.warning(f"The picture for '{token.char}' could not be opened.")
        else:
            st.warning(f"No picture for '{token.char}' ({token.note}).")

    if state.show_full:
        st.subheader("Full sequence")
        for number, word in enumerate(split_words(state.tokens), start=1):
            st.markdown(f"**Word {number}:** {''.join(t.char for t in word)}")
            columns = st.columns(min(len(word), 8))
            for i, token in enumerate(word):
                with columns[i % len(columns)]:
                    show_token(token, 120)
            if number < len(split_words(state.tokens)):
                st.markdown(f"<div style='text-align:center;color:#888'>\u2014 {SPACE_LABEL} \u2014</div>",
                            unsafe_allow_html=True)
    else:
        index = max(0, min(len(steps) - 1, state.index))
        token = steps[index]
        st.subheader(f"Sign {index + 1} of {len(steps)}: {token.label}")
        left, _ = st.columns([1, 2])
        with left:
            show_token(token, 320)


if __name__ == "__main__":
    main()
