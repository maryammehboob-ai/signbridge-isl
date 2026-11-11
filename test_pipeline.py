"""Tests for the WebRTC live pipeline (no browser, no camera). Run:  .venv_deploy\\Scripts\\python.exe test_pipeline.py
Uses real frames from the fresh-test recording report folder and your real models."""
import ctypes
import glob
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "APP"))


def rss_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    info = PMC()
    info.cb = ctypes.sizeof(info)
    kernel, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb)
    return info.WorkingSetSize / 1e6


results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail and not ok else f"   {detail}" if detail else ""))


print("1. Packages import together, and memory")
before = rss_mb()
import av, aiortc, cv2, mediapipe, numpy, streamlit, streamlit_webrtc, tensorflow  # noqa: E402,F401
check("tensorflow, mediapipe, opencv, av, aiortc, streamlit-webrtc import together", True,
      f"tf {tensorflow.__version__}, mediapipe {mediapipe.__version__}, cv2 {cv2.__version__}, av {av.__version__}, webrtc {streamlit_webrtc.__version__}")
import core  # noqa: E402,F401
from core import webrtc_session as ws_  # noqa: E402
after_import = rss_mb()
shared = ws_.load_shared_models()
after_models = rss_mb()
print(f"       memory: start {before:.0f} MB, after imports {after_import:.0f} MB, after loading both models {after_models:.0f} MB")

print("2. Real frames: the analysis width does not change the model's answers (640x480 frames are not resized at all)")
import webcam_speech as wsp  # noqa: E402
frames = sorted(glob.glob(str(ROOT.parent / "isl_image_project" / "MODEL_EVALUATION" / "landmarks" / "fresh_test" / "reports" / "guided_*" / "*.jpg")))
print(f"       {len(frames)} saved frames found")
recognizer = ws_.make_recognizer(shared)
same = total = 0
mismatch = []
for path in frames:
    image = cv2.imdecode(numpy.fromfile(path, numpy.uint8), cv2.IMREAD_COLOR)
    if image is None:
        continue
    full = recognizer.read(image, "all")
    h, w = image.shape[:2]
    small = cv2.resize(image, (ws_.ANALYSIS_WIDTH, int(round(h * ws_.ANALYSIS_WIDTH / w))), interpolation=cv2.INTER_AREA) if w > ws_.ANALYSIS_WIDTH else image
    shrunk = recognizer.read(small, "all")
    total += 1
    a = None if full is None else full.label
    b = None if shrunk is None else shrunk.label
    if a == b:
        same += 1
    else:
        mismatch.append((os.path.basename(os.path.dirname(path))[-6:], os.path.basename(path), a, b))
check("same answer on the full frame and the frame as the session analyses it", total > 0 and same == total, f"{same} of {total} identical; differences: {mismatch[:6]}")

timing = []
for path in frames[:20]:
    image = cv2.imdecode(numpy.fromfile(path, numpy.uint8), cv2.IMREAD_COLOR)
    s = time.perf_counter(); recognizer.read(image, "all"); timing.append(time.perf_counter() - s)
timing = sorted(timing[3:])
print(f"       one recognition (hand detection + model) takes median {timing[len(timing)//2]*1000:.0f} ms, max {timing[-1]*1000:.0f} ms on this PC")
print("3. Session: frames in, newest frame analysed, rate limited, video returned fast")
session = ws_.get_session("test-a", shared)
check("a new session starts with the camera off", not session.camera_on)
image = cv2.imdecode(numpy.fromfile(frames[0], numpy.uint8), cv2.IMREAD_COLOR)
calls = {"n": 0}
real_analyse = session._analyse


def counting(frame, now):
    calls["n"] += 1
    return real_analyse(frame, now)


session._analyse = counting
durations = []
t0 = time.monotonic()
while time.monotonic() - t0 < 6.0:
    s = time.perf_counter()
    out = session.feed(image)
    durations.append(time.perf_counter() - s)
    time.sleep(1 / 30)
fed = len(durations)
check("feed() returns an image of the same size", out.shape == image.shape)
check("feed() is fast (median under 15 ms, so the video does not wait for recognition)", sorted(durations)[len(durations) // 2] < 0.015,
      f"median {sorted(durations)[len(durations)//2]*1000:.1f} ms, max {max(durations)*1000:.0f} ms")
limit = ws_.ANALYSIS_FPS * 6.0 + 4
check("recognition is rate limited, old frames are dropped", 5 < calls["n"] <= limit, f"{fed} frames in, {calls['n']} analysed in 6 s (limit about {limit:.0f})")
check("the stream counts as connected while frames arrive", session.camera_on)
snap = session.snapshot()
check("snapshot has a prediction and confidence from the real model", snap.label is not None and 0 < snap.confidence <= 1.0, f"{snap.label} {snap.confidence:.2f}")
check("engine is the hand-point model", snap.engine == "hand-point model")
time.sleep(ws_.CONNECTED_SECONDS + 0.5)
check("the stream counts as off a few seconds after the last frame", not session.camera_on)

print("4. Add Letter: one letter per press, no repeats")
for _ in range(25):
    session.feed(image)
    time.sleep(1 / 30)
before_text = session.text.text
a1 = session.perform("add")
a2 = session.perform("add")
check("two quick presses add only one letter", (a1, a2) == (True, False) or (a1, a2) == (False, False), f"{a1}, {a2}, text '{session.text.text}'")
check("the text is at most one letter longer", len(session.text.text) - len(before_text) <= 1)

print("5. Visitors are separate, and the live cap works")
other = ws_.get_session("test-b", shared)
check("another visitor has their own session and empty text", other is not session and other.text.text == "")
sessions = [ws_.get_session(f"cap-{i}", shared) for i in range(5)]
for _ in range(4):
    for s in sessions:
        s.feed(image)
        time.sleep(0.02)
busy = [s.busy for s in sessions]
check(f"at most {ws_.MAX_LIVE_VIEWERS} visitors stream at once; the rest are told the server is busy",
      sum(1 for b in busy if not b) <= ws_.MAX_LIVE_VIEWERS and any(busy), str(busy))
check("memory with 8 sessions stays reasonable", rss_mb() < after_models + 900, f"{rss_mb():.0f} MB")
for s in [session, other] + sessions:
    s.close()

failed = results.count(False)
print(f"\n{len(results) - failed} of {len(results)} checks passed" + ("" if not failed else f"  ({failed} FAILED)"))
sys.exit(1 if failed else 0)
