"""Home page: what the app does and whether everything it needs is in place."""
import json

import streamlit as st

import ui
from core import PROJECT_DIR

MODELS = PROJECT_DIR / "MODELS"

ui.page_header("Talk with signs, in both directions",
               "Indian Sign Language fingerspelling: 26 letters (A-Z) and the numbers 1-9. It reads signs from your camera, "
               "shows signs for your text, and listens to your voice. It does not translate whole ISL words.",
               icon_name="home", eyebrow="SignBridge")

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown(f"<div class='sb-feature'><div class='ico'>{ui.theme.icon('camera')}</div><h4>Live Sign-to-Text</h4>"
                    "<p>Show signs to the webcam. The app reads them continuously and builds words and sentences when you "
                    "press <b>Add Letter</b>.</p></div>", unsafe_allow_html=True)
        st.page_link("views/page_live.py", label="Open the live camera", icon=":material/videocam:")
with c2:
    with st.container(border=True):
        st.markdown(f"<div class='sb-feature'><div class='ico'>{ui.theme.icon('signs')}</div><h4>Text-to-Sign</h4>"
                    "<p>Type text (letters and numbers) and see the sign picture for each character, in order, "
                    "or play it as an animated GIF.</p></div>", unsafe_allow_html=True)
        st.page_link("views/page_text_to_sign.py", label="Open Text-to-Sign", icon=":material/text_fields:")
with c3:
    with st.container(border=True):
        st.markdown(f"<div class='sb-feature'><div class='ico'>{ui.theme.icon('mic')}</div><h4>Speech-to-Text</h4>"
                    "<p>Speak English, get the text, and send it to Text-to-Sign. Needs internet.</p></div>",
                    unsafe_allow_html=True)
        st.page_link("views/page_speech_to_text.py", label="Open Speech-to-Text", icon=":material/mic:")

st.subheader("Setup check")
checks = {
    "Class labels (35)": MODELS / "class_names.json",
    "Hand-point model (used for live recognition)": MODELS / "isl_landmarks.keras",
    "MobileNetV2 model (loaded by the recogniser)": MODELS / "isl_mobilenetv2.keras",
    "Sign pictures for Text-to-Sign (35)": PROJECT_DIR / "TEXT_TO_SIGN" / "alphabet_images" / "A.jpg",
}
with st.container(border=True):
    for name, path in checks.items():
        st.markdown(ui.status_check(path.is_file(), ("OK  " if path.is_file() else "MISSING  ") + name), unsafe_allow_html=True)
    try:
        labels = json.loads((MODELS / "class_names.json").read_text(encoding="utf-8"))
        st.caption(f"Labels: {len(labels)} classes ({labels[0]} ... {labels[-1]})")
    except Exception:
        pass

with st.expander("Good to know"):
    st.markdown(
        "- **Camera:** the live page uses your own browser camera (press START and allow it). It needs a secure (https) page and a network that allows live video.\n"
        "- **Speech-to-Text** sends your recording to Google's speech service, so it needs internet. "
        "The live recognition does not use Google.\n"
        "- **Accuracy:** the live page uses your hand-point model. Signs that need two hands work best with both hands fully in view.\n"
        "- Live camera pictures are analysed on the server and are not stored. Reading aloud (text-to-speech) is not available in the online version."
    )
ui.footer()
