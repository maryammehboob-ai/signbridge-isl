"""ISL Speech-to-Text: record your voice in the browser and turn it into English text.
The text can then be shown as ISL fingerspelling signs (Text-to-Sign).

How it works
  1. st.audio_input records your voice in the browser (a WAV file).
  2. The SpeechRecognition library sends that audio to Google's free speech service and
     gets English text back.  >>> INTERNET IS REQUIRED, and your audio is sent to Google. <<<
  3. The text appears in a box you can edit.  "Send to Text-to-Sign" shows the sign pictures.

Run (from the project folder):
    TEXT_TO_SIGN\\.venv_streamlit\\Scripts\\python.exe -m streamlit run SPEECH_TO_TEXT\\speech_to_text.py

The first part of this file is plain Python (no Streamlit) so it can be tested without a
browser or a microphone: see test_speech_to_text.py.  The screen is in main() at the bottom.
It does not touch the MobileNetV2 model, the webcam scripts or the 35 class labels.
"""
import array
import io
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "TEXT_TO_SIGN"))      # to reuse the Text-to-Sign code

LANGUAGE = "en-US"
MIN_AUDIO_BYTES = 1000          # smaller than this cannot hold any speech
MIN_PEAK = 150                  # quietest "loud enough" sample (16-bit audio goes up to 32767)

MESSAGES = {
    "empty": "No audio was recorded. Click the microphone button, speak, then click the red square to stop. "
             "After you stop, the recording needs a few seconds to upload: the text appears by itself, "
             "or press Convert again if it does not.",
    "silent": "The recording is silent. Check that the right microphone is selected and speak a little louder.",
    "bad_audio": "The recording could not be read. Please record again.",
    "not_understood": "Speech was recorded but no words were recognised. Speak clearly in English and try again.",
    "network": "Could not reach the speech service. Speech-to-Text needs an internet connection: check your connection and try again.",
    "service": "The speech service returned an error. Wait a moment and try again.",
    "other": "Something went wrong while converting the speech.",
}


@dataclass
class SpeechResult:
    ok: bool
    text: str = ""
    error: str = ""             # one of the keys of MESSAGES, or "" when ok
    message: str = ""           # a plain-language message for the user


def fail(kind: str, extra: str = "") -> SpeechResult:
    message = MESSAGES[kind] + (f" ({extra})" if extra else "")
    return SpeechResult(False, "", kind, message)


def clean_text(text: str) -> str:
    """Remove extra spaces and make the first letter a capital: 'hello' -> 'Hello'."""
    text = " ".join((text or "").split())
    return text[:1].upper() + text[1:]


def check_audio(wav_bytes: Optional[bytes]) -> Optional[SpeechResult]:
    """Return an error result if the audio is empty, unreadable or silent; None if it looks usable."""
    if not wav_bytes or len(wav_bytes) < MIN_AUDIO_BYTES:
        return fail("empty")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            width = wav.getsampwidth()
    except (wave.Error, EOFError):
        return fail("bad_audio")
    if not frames:
        return fail("empty")
    if width == 2:                                  # 16-bit audio (what the browser recorder gives)
        samples = array.array("h")
        samples.frombytes(frames[: len(frames) // 2 * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        if not samples or max(max(samples), -min(samples)) < MIN_PEAK:
            return fail("silent")
    return None


def transcribe_wav(wav_bytes: Optional[bytes], recognizer=None, language: str = LANGUAGE) -> SpeechResult:
    """Turn WAV audio into text. Never raises: every problem becomes a SpeechResult with a message.

    `recognizer` can be replaced by a fake one in tests (no internet needed).
    """
    problem = check_audio(wav_bytes)
    if problem is not None:
        return problem
    try:
        import speech_recognition as sr
    except ImportError:
        return fail("other", "the SpeechRecognition library is not installed in this Python")
    recognizer = recognizer or sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = recognizer.record(source)
    except Exception as error:                      # unreadable audio file
        return fail("bad_audio", type(error).__name__)
    try:
        text = recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return fail("not_understood")
    except sr.RequestError as error:
        return fail("network" if "connection failed" in str(error).lower() else "service", str(error))
    except OSError as error:                        # e.g. no network at all
        return fail("network", type(error).__name__)
    except Exception as error:
        return fail("other", f"{type(error).__name__}: {error}")
    text = clean_text(text)
    if not text:
        return fail("not_understood")
    return SpeechResult(True, text)


# ----------------------------------------------------------------------------------------
# The screen (Streamlit). Everything below only runs when you start the app.
# ----------------------------------------------------------------------------------------
def main():
    import streamlit as st
    import text_to_sign as t2s

    st.set_page_config(page_title="ISL Speech-to-Text", layout="wide")
    st.title("ISL Speech-to-Text")
    st.caption("Speak English, get text, then show it as ISL fingerspelling signs. "
               "Internet is required: your audio is sent to Google's speech service.")

    state = st.session_state
    state.setdefault("audio_round", 0)
    state.setdefault("last_audio_key", None)
    state.setdefault("speech_text", "")
    state.setdefault("show_signs", False)
    state.setdefault("error", "")
    state.setdefault("info", "")

    with st.expander("Microphone help"):
        st.markdown(
            "- The first time, the browser asks **Allow microphone?**. Choose **Allow**.\n"
            "- If you blocked it: click the **lock / tune icon** in the address bar, set "
            "**Microphone** to Allow, then reload the page.\n"
            "- If nothing is recorded, check Windows Settings > Privacy > Microphone, "
            "and that the right microphone is selected in the recorder.\n"
            "- Only **localhost** or **https** pages may use the microphone (the normal "
            "`localhost:8501` address is fine)."
        )

    def convert(audio_file):
        state.show_signs = False
        state.info = ""
        if audio_file is None:
            state.error = MESSAGES["empty"]
            return
        with st.spinner("Converting speech to text..."):
            result = transcribe_wav(audio_file.getvalue())
        if result.ok:
            state.speech_text, state.error, state.info = result.text, "", "Speech converted."
        else:
            state.error = result.message

    def clear():
        state.speech_text, state.error, state.info = "", "", ""
        state.last_audio_key = None
        state.show_signs = False
        state.audio_round += 1                        # a new recorder key gives an empty recorder

    def send():
        state.show_signs = True

    st.subheader("1. Record")
    audio = st.audio_input("Click the microphone, speak, then click again to stop.",
                           key=f"audio_{state.audio_round}", sample_rate=16000)

    # A recording that has just arrived is converted by itself (no need to find the Convert button; pressing it
    # too early, while the recording was still uploading, used to give a confusing "No audio" message).
    if audio is not None:
        data = audio.getvalue()
        key = (state.audio_round, len(data), hash(data))
        if state.get("last_audio_key") != key:
            state.last_audio_key = key
            convert(audio)

    c1, c2 = st.columns(2)
    if c1.button("Convert Speech to Text", type="primary", use_container_width=True):
        convert(audio)
    c2.button("Clear Text", on_click=clear, use_container_width=True)

    if state.error:
        st.error(state.error)
    elif state.info:
        st.success(state.info)

    st.subheader("2. Recognised text")
    st.text_area("You can correct the text here if a word is wrong:", key="speech_text", height=100,
                 placeholder="Your sentence will appear here")

    st.subheader("3. Show as signs")
    st.button("Send to Text-to-Sign", on_click=send, disabled=not state.speech_text.strip())
    if state.show_signs and state.speech_text.strip():
        tokens, warnings = t2s.text_to_tokens(state.speech_text)
        for warning in warnings:
            st.warning(warning)
        words = t2s.split_words(tokens)
        if not words:
            st.warning("Nothing in this text can be shown as a sign (only letters A-Z and numbers 1-9).")
        else:
            st.markdown("**Sequence:** " + t2s.sequence_text(tokens))
            for number, word in enumerate(words, start=1):
                st.markdown(f"**Word {number}:** {''.join(t.char for t in word)}")
                columns = st.columns(min(len(word), 8))
                for i, token in enumerate(word):
                    with columns[i % len(columns)]:
                        if token.image is not None:
                            st.image(str(token.image), caption=token.label, width=110)
                        else:
                            st.warning(f"No picture for '{token.char}'")
            st.caption("For one-sign-at-a-time viewing, paste this text into the Text-to-Sign app.")


if __name__ == "__main__":
    main()
