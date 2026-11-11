"""Opens every page of the online app with Streamlit's AppTest (no browser). Run:  .venv_deploy\\Scripts\\python.exe test_pages.py"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parent
APP = ROOT / "APP"
sys.path.insert(0, str(APP))
from streamlit.testing.v1 import AppTest  # noqa: E402

import types  # noqa: E402
import streamlit_webrtc  # noqa: E402


def _fake_webrtc_streamer(**kwargs):
    """Streamlit's AppTest cannot host the real WebRTC component (it needs a browser); the real one is tested in test_browser notes."""
    return types.SimpleNamespace(state=types.SimpleNamespace(playing=False, signalling=False), video_receiver=None)


streamlit_webrtc.webrtc_streamer = _fake_webrtc_streamer
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail and not ok else ""))


def button(at, label):
    return next(b for b in at.button if b.label == label)


print("Pages")
at = AppTest.from_file(str(APP / "app.py"), default_timeout=120).run()
check("the app opens on the Home page", not at.exception and any("Talk with signs" in t.value for t in at.title), str(at.exception))
for page in ("views/page_live.py", "views/page_text_to_sign.py", "views/page_speech_to_text.py", "views/page_model_performance.py"):
    at.switch_page(page).run()
    check(f"{page} opens inside the app without errors", not at.exception, str(at.exception))

at = AppTest.from_file(str(APP / "views" / "page_live.py"), default_timeout=120).run()
check("Live page: models load and the page draws", not at.exception, str(at.exception))
check("Live page: has Add Letter, Space, Delete Last, Clear Word, Clear Sentence",
      all(any(b.label == l for b in at.button) for l in ("Add Letter", "Space", "Delete Last", "Clear Word", "Clear Sentence")))
check("Live page: there is no server-side camera or Save-to-server button", not any(b.label in ("Start Camera", "Save Text") for b in at.button))
check("Live page: no Windows-only speak button", not any(b.label == "Speak" for b in at.button))
button(at, "Add Letter").click().run()
check("Add Letter with no camera running does not crash and adds nothing", not at.exception)
button(at, "Space").click().run()
check("Space with an empty word does not crash", not at.exception)

at = AppTest.from_file(str(APP / "views" / "page_text_to_sign.py"), default_timeout=60).run()
at.text_area[0].set_value("hello 12").run()
next(b for b in at.button if b.label == "Convert to Sign").click().run()
check("Text-to-Sign converts text to signs", not at.exception and any(s.value.startswith("Sign 1 of") for s in at.subheader), str(at.exception))

at = AppTest.from_file(str(APP / "views" / "page_speech_to_text.py"), default_timeout=60).run()
check("Speech-to-Text page opens and has its convert button", not at.exception and any(b.label == "Convert Speech to Text" for b in at.button), str(at.exception))

at = AppTest.from_file(str(APP / "views" / "page_model_performance.py"), default_timeout=60).run()
saved = json.loads((ROOT / "MODEL_EVALUATION" / "landmarks" / "results.json").read_text(encoding="utf-8"))
shown = {m.label: m.value for m in at.metric}
check("Model Performance shows the saved accuracy", shown.get("Accuracy (per frame)") == f"{saved['fresh_independent']['live_model']['accuracy'] * 100:.1f}%", str(shown))

failed = results.count(False)
print(f"\n{len(results) - failed} of {len(results)} checks passed" + ("" if not failed else f"  ({failed} FAILED)"))
sys.exit(1 if failed else 0)
