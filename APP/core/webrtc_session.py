"""Live Sign-to-Text over WebRTC (the camera is the VISITOR's browser, not the server).

How it works:
    browser camera --WebRTC--> feed(frame)            called for every video frame (fast: only draws)
                               worker thread          recognises the NEWEST frame at most ANALYSIS_FPS times a
                                                      second, older frames are dropped (no growing delay)
    page (Streamlit)           snapshot()/perform()   prediction, confidence, WORD and SENTENCE

It reuses your tested code and does not change it:
    sign_to_text.Recognizer      MediaPipe hand detection + isl_landmarks.keras (same settings, same features)
    sign_to_text.LetterCommitter confidence threshold, vote window, hold time, no repeated letter
    sign_to_text.TextBuffer      WORD / SENTENCE
    live_session.LiveSession     _analyse(), perform(), snapshot() ... (this class only replaces the camera part)

Every visitor gets their own session (own text, own hand detector); the models are loaded once and shared.
At most MAX_LIVE_VIEWERS people can stream at the same time (the free server is small); others see a notice.
"""
import threading
import time

import cv2
import numpy as np

from core import PROJECT_DIR  # noqa: F401  (also sets the import paths)
from core.live_session import LiveSession

ANALYSIS_FPS = 12.0          # recognitions per second at most (the returned video itself stays smooth)
ANALYSIS_WIDTH = 640         # larger frames (phones) are shrunk to this width; 640x480 camera frames are analysed as they are (same as the local app)
CONNECTED_SECONDS = 3.0      # no frame for this long = the camera counts as off
MAX_LIVE_VIEWERS = 3
VIEWER_IDLE_SECONDS = 8.0    # a viewer that sent no frame for this long gives the place back
SESSION_IDLE_SECONDS = 600.0 # sessions nobody used for 10 minutes are closed (memory)

_model_lock = threading.Lock()          # one prediction at a time through the shared Keras models
_registry_lock = threading.Lock()
_viewers = {}                           # session id -> time of its last frame
_sessions = {}                          # session id -> (session, time last used)


class _LockedModel:
    """The shared Keras model, called one visitor at a time (the model is tiny: this costs about a millisecond)."""

    def __init__(self, model):
        self._model = model

    def __call__(self, *args, **kwargs):
        with _model_lock:
            return self._model(*args, **kwargs)


def load_shared_models():
    """Load the models once per server (call through st.cache_resource). Same files and loader as your webcam script."""
    import webcam_speech as ws
    model, labels, image_size, _motion, landmark_model = ws.load_model_and_labels()
    if landmark_model is None:
        raise RuntimeError("MODELS/isl_landmarks.keras was not found, so live recognition cannot start.")
    return {"model": model, "labels": labels, "image_size": image_size, "landmark_model": landmark_model}


def make_recognizer(shared):
    """A Recognizer (your class) for ONE visitor: shared models, its own MediaPipe hand detector."""
    import mediapipe as mp
    import webcam_speech as ws
    from sign_to_text import Recognizer

    recognizer = Recognizer.__new__(Recognizer)
    recognizer.model, recognizer.labels, recognizer.image_size = shared["model"], shared["labels"], shared["image_size"]
    recognizer.landmark_model = _LockedModel(shared["landmark_model"])
    recognizer.use_landmarks = True
    recognizer.hands = mp.solutions.hands.Hands(            # same settings as Recognizer.__init__
        static_image_mode=True, max_num_hands=2, model_complexity=ws.HAND_MODEL_COMPLEXITY,
        min_detection_confidence=ws.HAND_DETECTION_CONFIDENCE, min_tracking_confidence=ws.HAND_TRACKING_CONFIDENCE)
    return recognizer


class WebRTCLiveSession(LiveSession):
    def __init__(self, recognizer, committer, text_buffer, session_id="local", **kwargs):
        super().__init__(recognizer, committer, text_buffer, camera_factory=lambda *a, **k: None, **kwargs)
        self.session_id = session_id
        self._inbox = None                   # newest raw frame (BGR)
        self._inbox_counter = 0
        self._last_feed = 0.0
        self._worker = None
        self.busy = False                    # True: this viewer was refused because the server is full
        self.connection_error = ""

    # ------------------------------------------------------------------ state of the stream
    @property
    def camera_on(self):
        return (time.monotonic() - self._last_feed) < CONNECTED_SECONDS and self._last_feed > 0 and not self.busy

    def snapshot(self, include_frame=False):
        if not self.camera_on and self._last_feed > 0 and self.state is not None and self.state.status != "Camera stopped":
            self.stop_camera()               # the stream ended by itself: clear the old prediction
        return super().snapshot(include_frame)

    def start_camera(self):                  # the WebRTC widget starts the stream; nothing to do here
        self.connection_error = ""

    def stop_camera(self, reason=""):
        with self._lock:
            self._last_feed = 0.0
            self._inbox = None
            self._frame_times.clear()
            self._reset_state()
        _release_viewer(self.session_id)

    # ------------------------------------------------------------------ frames from the browser
    def feed(self, frame_bgr):
        """Called for every video frame. Returns the frame to show (BGR) with the hand outline drawn on it.
        Fast on purpose: the recognition itself runs in the worker thread."""
        now = time.monotonic()
        if not _claim_viewer(self.session_id, now):
            self.busy = True
            out = frame_bgr.copy()
            cv2.putText(out, "Server busy - try again later", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255), 2, cv2.LINE_AA)
            return out
        self.busy = False
        self._last_feed = now
        self._frame_times.append(now)
        with self._cond:
            self._inbox = frame_bgr
            self._inbox_counter += 1
            self._cond.notify_all()
        self._ensure_worker()
        return self._overlay(frame_bgr, self._last_reading)

    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._stop.clear()
            self._worker = threading.Thread(target=self._work, name="isl-recognise", daemon=True)
            self._worker.start()

    def _work(self):
        seen, last_run = 0, 0.0
        gap = 1.0 / ANALYSIS_FPS
        while not self._stop.is_set():
            with self._cond:
                if self._inbox_counter == seen:
                    self._cond.wait(timeout=1.0)
                frame, counter = self._inbox, self._inbox_counter
            if frame is None or counter == seen:
                if time.monotonic() - self._last_feed > VIEWER_IDLE_SECONDS:
                    return                    # nobody is sending: the thread ends, feed() restarts it later
                continue
            wait = gap - (time.monotonic() - last_run)
            if wait > 0:
                time.sleep(wait)
                with self._cond:
                    frame, counter = self._inbox, self._inbox_counter    # take the newest one after waiting
            seen = counter
            last_run = time.monotonic()
            try:
                height, width = frame.shape[:2]
                small = frame if width <= ANALYSIS_WIDTH else cv2.resize(
                    frame, (ANALYSIS_WIDTH, int(round(height * ANALYSIS_WIDTH / width))), interpolation=cv2.INTER_AREA)
                reading = self._analyse(small, last_run)
                if reading is not None and reading.box is not None and small is not frame:
                    scale = width / small.shape[1]          # the hand box is in pixels of the shrunk frame
                    reading.box = tuple(v * scale for v in reading.box)
            except Exception as problem:      # one bad frame must not end the live view
                self.error = f"Recognition problem: {type(problem).__name__}: {problem}"
            else:
                self.error = ""

    def close(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        _release_viewer(self.session_id)
        close = getattr(self.recognizer, "close", None)
        if close:
            close()


# ------------------------------------------------------------------------------ visitors and sessions
def _claim_viewer(session_id, now):
    """True if this visitor may stream (a place is free or already theirs)."""
    with _registry_lock:
        for key in [k for k, t in _viewers.items() if now - t > VIEWER_IDLE_SECONDS]:
            del _viewers[key]
        if session_id not in _viewers and len(_viewers) >= MAX_LIVE_VIEWERS:
            return False
        _viewers[session_id] = now
        return True


def _release_viewer(session_id):
    with _registry_lock:
        _viewers.pop(session_id, None)


def live_viewers():
    now = time.monotonic()
    with _registry_lock:
        return sum(1 for t in _viewers.values() if now - t <= VIEWER_IDLE_SECONDS)


def get_session(session_id, shared):
    """The visitor's own session (created on first use). Old unused sessions are closed to free memory."""
    from sign_to_text import LetterCommitter, Settings, TextBuffer
    now = time.monotonic()
    with _registry_lock:
        old = [k for k, (_s, used) in _sessions.items() if now - used > SESSION_IDLE_SECONDS and k != session_id]
        dead = [_sessions.pop(k)[0] for k in old]
        entry = _sessions.get(session_id)
        if entry is None:
            settings = Settings()
            session = WebRTCLiveSession(
                make_recognizer(shared),
                LetterCommitter(threshold=settings.confidence_threshold, window_frames=settings.window_frames,
                                min_votes=settings.min_votes, hold_seconds=settings.hold_seconds,
                                release_seconds=settings.release_seconds),
                TextBuffer(settings.max_text_length), session_id=session_id, mode=settings.mode,
                capture_gap_seconds=settings.capture_gap_seconds)
        else:
            session = entry[0]
        _sessions[session_id] = (session, now)
    for stale in dead:
        stale.close()
    return session
