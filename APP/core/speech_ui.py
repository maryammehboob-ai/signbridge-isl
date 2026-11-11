"""Streamlit pieces for reading text aloud (voice choice, Speak button, the audio player).

Two ways, chosen automatically:
    * on Windows (your own PC): the Windows voices, through core/tts.py (unchanged)
    * everywhere else (Streamlit Cloud runs Linux): the visitor's OWN BROWSER voice (the Web Speech API).
      Nothing is sent to a server or to Google: the text is spoken on the visitor's device.
"""
import json
import os

import streamlit as st
import streamlit.components.v1 as components

from core import tts

_BROWSER_PAGE = """<!doctype html><meta charset="utf-8">
<style>
  html,body{margin:0;background:transparent;font-family:"Source Sans Pro",system-ui,sans-serif;color:#e8eef8}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  button,select{background:#101b2e;color:#e8eef8;border:1px solid #2a3f5f;border-radius:10px;padding:9px 14px;font-size:14px}
  button{cursor:pointer} button.p{background:#14917f;border-color:#14917f;font-weight:600}
  select{max-width:100%;min-width:0;flex:1 1 160px}
  .n{font-size:13px;color:#8aa0bf;margin-top:8px;min-height:18px}
</style>
<div class="row"><button class="p" id="play">&#9654; Play</button><button id="stop">&#9632; Stop</button>
<select id="voice" aria-label="Voice"><option value="">Default voice</option></select></div>
<div class="n" id="note"></div>
<script>
const TEXT = __TEXT__, RATE = __RATE__, AUTO = __AUTO__;
const synth = window.speechSynthesis, note = document.getElementById("note"), sel = document.getElementById("voice");
function say(m) { note.textContent = m; }
if (!synth || typeof SpeechSynthesisUtterance === "undefined") { say("This browser cannot read aloud."); }
else {
  function fill() {
    const voices = synth.getVoices(), keep = sel.value;
    sel.innerHTML = '<option value="">Default voice</option>';
    voices.forEach((v, i) => { const o = document.createElement("option"); o.value = String(i); o.textContent = v.name + " (" + v.lang + ")"; sel.appendChild(o); });
    sel.value = keep;
  }
  fill(); synth.onvoiceschanged = fill;
  function play() {
    synth.cancel();
    const u = new SpeechSynthesisUtterance(TEXT);
    u.rate = RATE;
    const v = sel.value === "" ? null : synth.getVoices()[Number(sel.value)];
    if (v) { u.voice = v; u.lang = v.lang; }
    u.onstart = () => say("Playing ...");
    u.onend = () => say("");
    u.onerror = (e) => {
      if (e.error === "canceled" || e.error === "interrupted") return say("");
      if (e.error === "not-allowed") return say("Your browser blocked automatic playback. Press Play to listen.");
      say("Could not play the voice (" + e.error + "). Check the device volume and try Play again.");
    };
    synth.speak(u);
  }
  document.getElementById("play").onclick = play;
  document.getElementById("stop").onclick = () => { synth.cancel(); say(""); };
  if (AUTO) setTimeout(play, 200);
}
</script>"""


def _browser_mode():
    return not tts.available() or os.environ.get("ISL_FORCE_BROWSER_VOICE") == "1"     # the variable is only for testing on Windows


def voice_controls(prefix):
    """Voice and speed selectors. Returns (voice name or None, rate)."""
    if _browser_mode():
        rate = st.slider("Speed", -5, 5, 0, key=f"{prefix}_rate", help="0 is normal speed; negative is slower.")
        st.caption("The voice is spoken by your own browser, so it sounds like the voices on your device. You can pick another voice after pressing Read aloud.")
        return None, rate
    names = list(tts.voices())
    if not names:
        st.caption("No Windows voice was found, so reading aloud is not available on this PC.")
        return None, 0
    left, right = st.columns(2)
    voice = left.selectbox("Voice", names, key=f"{prefix}_voice")
    rate = right.slider("Speed", -5, 5, 0, key=f"{prefix}_rate", help="0 is normal speed; negative is slower.")
    return voice, rate


def speak(prefix, text, voice, rate):
    """Windows: make the speech now. Browser: remember what to say; player() hands it to the browser."""
    state = st.session_state
    if _browser_mode():
        state[f"{prefix}_say"] = {"text": str(text), "rate": rate, "auto": True}
        return
    state[f"{prefix}_audio"], state[f"{prefix}_speak_error"] = None, ""
    try:
        state[f"{prefix}_audio"] = tts.speak_wav(text, voice, rate)
    except tts.SpeechError as problem:
        state[f"{prefix}_speak_error"] = str(problem)


def player(prefix):
    """Windows: play the audio speak() prepared (once). Browser: show the Play controls and start the speech."""
    state = st.session_state
    if _browser_mode():
        job = state.get(f"{prefix}_say")
        if job:
            text = job["text"].replace("</", "<\\/")
            page = (_BROWSER_PAGE.replace("__RATE__", repr(2 ** (job["rate"] / 5))).replace("__AUTO__", "true" if job["auto"] else "false")
                    .replace("__TEXT__", json.dumps(text)))          # the visitor's text goes in last
            shown = job["text"] if len(job["text"]) <= 60 else job["text"][:57] + "..."
            st.caption(f"Reading: {shown}")
            components.html(page, height=96)
            job["auto"] = False                  # start by itself only once; later reruns just show the controls
        return
    error = state.pop(f"{prefix}_speak_error", "")
    audio = state.pop(f"{prefix}_audio", None)
    if error:
        st.error(error)
    if audio:
        st.audio(audio, format="audio/wav", autoplay=True)
