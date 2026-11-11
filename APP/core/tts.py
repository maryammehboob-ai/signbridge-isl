"""Text-to-speech: turn text into an audible WAV with the voices that come with Windows.

Uses Windows' built-in System.Speech engine (the voices "Microsoft David Desktop" and "Microsoft Zira
Desktop" here). It works offline: no internet, no extra package. The WAV is played in the browser by the
page (st.audio), so the sound comes from the browser, like any other web audio.

The text is passed to PowerShell through environment variables (never put into the command line), so quotes,
$, ; and backticks in the text cannot run anything.
"""
import functools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_CHARS = 600
TIMEOUT_SECONDS = 30

_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:ISL_TTS_VOICE) { try { $voice.SelectVoice($env:ISL_TTS_VOICE) } catch { } }
$voice.Rate = [int]$env:ISL_TTS_RATE
$voice.SetOutputToWaveFile($env:ISL_TTS_OUT)
$voice.Speak($env:ISL_TTS_TEXT)
$voice.Dispose()
"""

_LIST_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() |
    Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }
"""


class SpeechError(Exception):
    """The text could not be turned into speech; the message is meant for the person using the app."""


def available():
    return sys.platform == "win32"


def _powershell(script, env_extra, timeout=TIMEOUT_SECONDS):
    env = dict(os.environ, **env_extra)
    try:
        return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
                              capture_output=True, text=True, timeout=timeout, env=env,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError:
        raise SpeechError("Windows PowerShell was not found, so speech output is not available.")
    except subprocess.TimeoutExpired:
        raise SpeechError("The speech engine took too long. Try a shorter text.")


@functools.lru_cache(maxsize=1)
def voices():
    """Names of the installed voices (empty list if none or not on Windows)."""
    if not available():
        return ()
    try:
        done = _powershell(_LIST_SCRIPT, {}, timeout=20)
    except SpeechError:
        return ()
    return tuple(line.strip() for line in done.stdout.splitlines() if line.strip())


@functools.lru_cache(maxsize=32)
def _synthesize(text, voice, rate):
    if not available():
        raise SpeechError("Speech output needs Windows (it uses the voices installed with Windows).")
    handle, out_path = tempfile.mkstemp(suffix=".wav", prefix="isl_tts_")
    os.close(handle)
    try:
        done = _powershell(_SCRIPT, {"ISL_TTS_TEXT": text, "ISL_TTS_VOICE": voice or "", "ISL_TTS_RATE": str(int(rate)),
                                     "ISL_TTS_OUT": out_path})
        data = Path(out_path).read_bytes() if Path(out_path).exists() else b""
        if done.returncode != 0 or len(data) < 1000:
            detail = (done.stderr or "").strip().splitlines()[:1]
            raise SpeechError("The speech engine could not create the audio" + (f" ({detail[0][:120]})" if detail else "") + ".")
        return data
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def clean_for_speech(text):
    """Text as it should be spoken: lower case words (so HELLO is read as a word, not letter by letter)."""
    words = " ".join((text or "").split())
    return words.lower()


def speak_wav(text, voice=None, rate=0):
    """WAV bytes of the spoken text. Raises SpeechError with a plain message when it cannot."""
    spoken = clean_for_speech(text)
    if not spoken:
        raise SpeechError("There is no text to read aloud yet.")
    if len(spoken) > MAX_CHARS:
        spoken = spoken[:MAX_CHARS].rsplit(" ", 1)[0]
    return _synthesize(spoken, voice or "", max(-10, min(10, int(rate))))
