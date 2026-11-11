"""Live Sign-to-Text page (online version): the visitor's browser camera streams to the app over WebRTC."""
import secrets
import time

import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import ui
from core import handoff, words
from core import webrtc_session

MODE_LABELS = {"letters": "Letters (A-Z)", "numbers": "Numbers (1-9)", "all": "All signs"}
CONNECT_WARNING_SECONDS = 20


@st.cache_resource(show_spinner="Loading the models (the first time takes 10-60 seconds)...")
def get_shared_models():
    models = webrtc_session.load_shared_models()
    import gc
    gc.collect()
    gc.freeze()          # keeps Python's garbage collector from pausing the live threads (measured on the local app)
    return models


@st.cache_resource(show_spinner=False)
def get_speller():
    return words.Speller()


def rtc_configuration():
    """STUN always; a TURN relay too when the app owner put one in Streamlit Secrets (never in the code):
        [turn]
        urls = ["turn:example.com:3478", "turns:example.com:5349"]
        username = "..."
        credential = "..."
    """
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        turn = st.secrets["turn"]
        urls = turn["urls"]
        servers.append({"urls": [urls] if isinstance(urls, str) else list(urls),
                        "username": turn["username"], "credential": turn["credential"]})
    except Exception:                      # no secrets file / no [turn] section: STUN only
        pass
    return {"iceServers": servers}


speller = get_speller()
state = st.session_state

ui.page_header("Live Sign-to-Text",
               "Press **START**, allow the camera, then show a sign. Check the prediction and press **Add Letter**. "
               "Nothing is added to the text until you press it (or turn on Auto-add).",
               icon_name="camera", eyebrow="Live recognition")

try:
    shared = get_shared_models()
except Exception as problem:               # missing model, broken install, ...
    st.error(f"The recognition models could not be loaded: {type(problem).__name__}: {problem}")
    st.stop()

state.setdefault("visitor_id", secrets.token_urlsafe(8))
session = webrtc_session.get_session(state.visitor_id, shared)
state.setdefault("live_mode", session.mode)
state.setdefault("live_threshold", int(round(session.committer.threshold * 100)))
state.setdefault("live_auto", session.auto_add)


def space_pressed():
    """Space finishes the word; with 'fix spelling' on, an obvious one-letter mistake is corrected first."""
    if state.get("live_autofix"):
        fix = speller.best_fix(session.text.current_word)
        if fix:
            session.replace_word(fix)
    session.perform("space")


def on_video_frame(frame):
    """Runs in a worker thread for every frame from the visitor's camera (no Streamlit calls here)."""
    image = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(session.feed(image), format="bgr24")


# ------------------------------------------------------------------ settings
with st.container(border=True, key="controls"):
    c3, c4, c5 = st.columns([3.7, 2.6, 1.5])
    c3.radio("Mode", list(MODE_LABELS), format_func=MODE_LABELS.get, key="live_mode", horizontal=True,
             on_change=lambda: session.set_mode(state.live_mode))
    c4.slider("Confidence threshold (%)", 30, 99, key="live_threshold",
              on_change=lambda: session.set_threshold(state.live_threshold / 100))
    c5.toggle("Auto-add", key="live_auto", on_change=lambda: session.set_auto_add(state.live_auto),
              help="ON: hold a sign for about 1 second and the letter is added by itself.")

# ------------------------------------------------------------------ live view
left, right = st.columns([3, 2])
with left:
    with st.container(border=True, key="video_card"):
        ctx = webrtc_streamer(
            key="isl-live", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_configuration(),
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 24}},
                                      "audio": False},
            video_frame_callback=on_video_frame, async_processing=True,
            translations={"start": "START camera", "stop": "STOP", "select_device": "Select camera",
                          "device_ask_permission": "Allow the camera when your browser asks.",
                          "device_not_available": "No camera found. Connect a camera and reload the page.",
                          "device_access_denied": "Camera access was denied. Allow the camera for this site in your browser, then reload."})
        if ctx.state.signalling and not ctx.state.playing:
            state.setdefault("connect_since", time.monotonic())
        else:
            state.pop("connect_since", None)
        if not ctx.state.playing and not ctx.state.signalling:
            st.caption("The camera is off. Press START and allow the camera.")


@st.fragment(run_every=0.4)
def live_text():
    snap = session.snapshot()
    since = state.get("connect_since")
    if since and not snap.camera_on:
        waited = time.monotonic() - since
        if waited > CONNECT_WARNING_SECONDS:
            st.warning("Still connecting. Your network may block the live video connection (WebRTC). Try another "
                       "network (for example Wi-Fi instead of mobile data), then press STOP and START again.")
        else:
            st.info("Connecting to the camera stream ...")
    if session.busy:
        st.warning(f"The server is busy ({webrtc_session.MAX_LIVE_VIEWERS} people are using the live camera). Please try again in a minute.")
    if snap.error:
        st.error(snap.error)
    chips = [ui.chip("LIVE" if snap.camera_on else "CAMERA OFF", "ok" if snap.camera_on else "", live=snap.camera_on)]
    if snap.camera_on:
        chips.append(ui.chip(f"{snap.fps:.0f} video fps"))
        chips.append(ui.chip("hand in view" if snap.hands else "no hand", "ok" if snap.hands else "warn"))
    st.markdown(f"<div class='sb-chips'>{''.join(chips)}</div>", unsafe_allow_html=True)
    with st.container(border=True, key="pred_card"):
        st.markdown("<div class='sb-label'>CURRENT PREDICTION</div>", unsafe_allow_html=True)
        if snap.camera_on and snap.hands == 0:
            st.warning("No hand detected")
        elif snap.camera_on and snap.label:
            size = " long" if len(str(snap.label)) > 2 else ""
            st.markdown(f"<div class='pred'><div class='pred-letter{size}'>{snap.label}</div><div class='pred-side'>"
                        f"<div class='pred-conf'>Confidence: {snap.confidence:.0%}</div>"
                        f"<div class='sb-bar'><i style='width:{min(100, max(0, snap.confidence * 100)):.0f}%'></i></div></div></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='pred'><div class='pred-letter none'>&ndash;</div><div class='pred-side'>"
                        "<div class='pred-hint'>Waiting for a clear sign.</div></div></div>", unsafe_allow_html=True)
        if snap.camera_on:
            st.caption(snap.status)
            if snap.auto_add:
                st.progress(min(1.0, max(0.0, snap.progress)))
        st.caption(f"Engine: {snap.engine} | Mode: {snap.mode} | Threshold: {snap.threshold:.0%}")
        st.caption("Confidence is how sure the model is, not how often it is right.")
    if snap.notice:
        st.info(snap.notice)
    with st.container(border=True, key="text_card"):
        st.markdown("<div class='sb-label'>WORD (being built)</div>", unsafe_allow_html=True)
        st.code(snap.word or " ", language=None)
        if snap.word:
            suggestions = speller.suggestions(snap.word)
            if suggestions:
                st.caption(f"{snap.word} is not in the dictionary. Did you mean:")
                columns = st.columns(len(suggestions))
                for column, suggestion in zip(columns, suggestions):
                    column.button(suggestion, key=f"suggest_{snap.word}_{suggestion}", on_click=session.replace_word,
                                  args=(suggestion,), use_container_width=True)
            elif speller.is_word(snap.word):
                st.caption(f"{snap.word} is a word.")
            else:
                st.caption(f"{snap.word} is not in the dictionary (fine for names).")
        st.markdown("<div class='sb-label'>SENTENCE (finished words)</div>", unsafe_allow_html=True)
        st.code(snap.sentence or " ", language=None)
        if snap.text:
            st.caption("Readable: " + words.readable_sentence(snap.text))


with right:
    live_text()

# ------------------------------------------------------------------ text buttons
st.subheader("Build your text")
with st.container(border=True, key="actions"):
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.button("Add Letter", on_click=session.perform, args=("add",), type="primary", use_container_width=True)
    b2.button("Space", on_click=space_pressed, use_container_width=True)
    b3.button("Delete Last", on_click=session.perform, args=("delete",), use_container_width=True)
    b4.button("Clear Word", on_click=session.perform, args=("clear_word",), use_container_width=True)
    b5.button("Clear Sentence", on_click=session.perform, args=("clear_sentence",), use_container_width=True)
    st.toggle("Fix spelling when I press Space", key="live_autofix",
              help="ON: an obvious one-letter mistake (HELO -> HELLO) is corrected when you finish the word. "
                   "Off by default; you can always click a suggestion instead.")

st.subheader("Use your text")
s1, s2 = st.columns(2)
s1.download_button("Download text", data=session.text.text or "", file_name="signbridge_text.txt", mime="text/plain",
                   use_container_width=True, disabled=not session.text.text,
                   help="Saves the text on your own device. Nothing is stored on the server.")
if s2.button("Send to Text-to-Sign", use_container_width=True):
    text = session.text.text
    if text:
        handoff.send_to_text_to_sign(text)
        st.switch_page(handoff.TEXT_TO_SIGN_PAGE)
    else:
        st.warning("There is no text to send yet.")

with st.expander("Tips, permissions and problems"):
    st.markdown(
        "- Keep **both hands fully in the picture** for two-hand signs, with good light and a plain background.\n"
        "- Your browser asks for **camera permission**. If you pressed Block, click the camera icon in the address bar, "
        "allow it, and reload. The camera works only on secure (https) pages, which this site is.\n"
        "- The video goes from your browser to the app only while you use this page. Nothing is recorded or stored.\n"
        "- **Stuck on connecting?** Some networks (mobile data, school or office Wi-Fi) block live video. Try another network.\n"
        "- The camera stops by itself a few seconds after you press STOP or leave this page.\n"
        "- Your text belongs to your visit only. Use **Download text** to keep it.\n"
        f"- To keep the free server fast, at most {webrtc_session.MAX_LIVE_VIEWERS} people can use the live camera at the same time."
    )
ui.footer()
