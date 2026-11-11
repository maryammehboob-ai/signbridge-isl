"""Live Sign-to-Text engine for the Streamlit app.

One LiveSession owns the camera, two background threads and the text (WORD and SENTENCE):
    capture thread      reads camera frames as fast as the camera delivers them, draws the hand
                        outline on them and publishes them as JPEG pictures
    recognition thread  recognises the newest frame (skipping old ones) and updates the prediction
A tiny local web server streams the JPEG pictures to the browser (MJPEG), so the video is smooth and
never waits for Streamlit page updates. The page only calls snapshot() for the prediction text and
perform()/set_*() for the buttons.

It reuses your tested code and does not change it:
    sign_to_text.Recognizer      (webcam_speech.py models, hand detection, pick_label)
    sign_to_text.TextBuffer      (WORD / SENTENCE)
    sign_to_text.LetterCommitter (stable prediction, optional Auto-add)

There is no Streamlit code here, so everything can be tested without a browser or a camera
(a fake camera and a fake recogniser can be passed in).

Camera notes (found by measuring this PC):
    * If another program already holds the camera, DirectShow still "opens" it but delivers BLACK
      frames at 1 per second. start_camera() checks the first frames and says so.
    * Media Foundation runs at about 29 fps here, DirectShow at 15 fps, so Media Foundation is
      tried first (with hardware transforms off: it then opens in 0.3 s instead of 1-2.5 s).
"""
import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # must be set before OpenCV opens a camera

import base64
import collections
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

from core import PROJECT_DIR  # noqa: F401  (also sets the import paths)

POLL_TIMEOUT_SECONDS = 8.0          # camera stops by itself if no page asks for pictures for this long
MAX_READ_FAILURES = 30              # camera frames that may fail in a row before "lost the camera"
MODES = ("letters", "numbers", "all")
NOTICE_SECONDS = 2.5
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
BACKENDS = ("msmf", "dshow")        # tried in this order
PROBE_SECONDS = 3.0                 # how long the first frames of a camera are checked
PROBE_GOOD_FRAMES = 3
BLACK_LEVEL = 2.0                   # average pixel value (0-255) below which a frame counts as black
BLACK_CHECKS_BEFORE_ERROR = 20      # consecutive black frame checks (every 5th frame) while running
JPEG_QUALITY = 80

CAMERA_BUSY_MESSAGE = ("The camera opened but is sending only black or very slow pictures. Another program is "
                       "probably using it: close the OpenCV windows (webcam_speech.py, Sign-to-Text), other "
                       "browser tabs or apps that use the camera, and any older copy of this app, then press "
                       "Start Camera again.")


@dataclass
class Snapshot:
    """Everything the page needs to draw one update."""
    camera_on: bool
    error: str
    hands: int
    label: Optional[str]            # the CURRENT PREDICTION (None: no hand or not sure)
    confidence: float
    status: str
    progress: float
    word: str
    sentence: str
    text: str
    notice: str
    mode: str
    threshold: float
    auto_add: bool
    engine: str
    stream_url: str                 # where the browser can watch the live video ("" until started)
    backend: str                    # which Windows camera driver is in use
    fps: float                      # camera pictures per second right now
    frame_rgb: Optional[np.ndarray] = None   # only filled when snapshot(include_frame=True)


def _hand_connections():
    import mediapipe as mp
    return sorted((int(a), int(b)) for a, b in mp.solutions.hands.HAND_CONNECTIONS)


_HAND_CONNECTIONS = _hand_connections()


# ------------------------------------------------------------------------------ cameras
def default_camera_factory(index=0, backend="msmf"):
    """Open the webcam with the chosen Windows driver."""
    code = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}[backend]
    camera = cv2.VideoCapture(index, code)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if backend == "dshow":
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


class ReplayCamera:
    """A 'camera' that shows one saved picture again and again (for testing the real running app).

    Used only when the environment variable ISL_TEST_REPLAY_IMAGE names a picture file.
    """

    def __init__(self, path):
        self.path, self.stamp, self.picture, self.last = path, None, None, 0.0
        self._reload()
        if self.picture is None:
            raise FileNotFoundError(path)

    def _reload(self):
        """Load the picture again whenever the file changes (a test can swap it to 'sign' another letter)."""
        try:
            stamp = os.stat(self.path).st_mtime_ns
            if stamp != self.stamp:
                picture = cv2.imread(self.path)
                if picture is not None:
                    self.picture, self.stamp = cv2.resize(picture, (CAMERA_WIDTH, CAMERA_HEIGHT)), stamp
        except OSError:
            pass

    def isOpened(self):
        return True

    def read(self):
        wait = 1 / 30 - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()
        self._reload()
        return True, self.picture.copy()

    def release(self):
        pass


def _camera_factory_from_environment(default):
    path = os.environ.get("ISL_TEST_REPLAY_IMAGE")
    if path:
        return lambda index=0, backend=None: ReplayCamera(path)
    return default


def frame_to_data_uri(frame_rgb, quality=75):
    """A video frame as a data: URI (kept for pages that need one picture without a stream)."""
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode the video frame")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


# ------------------------------------------------------------------------------ video stream
class _StreamHandler(BaseHTTPRequestHandler):
    """GET /stream/<token>  ->  an endless multipart/x-mixed-replace (MJPEG) video."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        session = self.server.session
        path = self.path.split("?")[0]
        if path == f"/jpeg/{session.token}":                    # one picture (handy for checks)
            counter, jpeg = session.latest_jpeg()
            if jpeg is None:
                self.send_error(404, "no picture yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)
            return
        if path != f"/stream/{session.token}":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        last = -1
        try:
            while session.camera_on:
                counter, jpeg = session.wait_for_jpeg(last, timeout=1.0)
                if jpeg is None:
                    continue
                last = counter
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass


# ------------------------------------------------------------------------------ the session
class LiveSession:
    def __init__(self, recognizer, committer, text_buffer, camera_factory=default_camera_factory,
                 camera_index=0, mode="letters", auto_add=False, capture_gap_seconds=0.6,
                 poll_timeout=POLL_TIMEOUT_SECONDS, backends=BACKENDS):
        self.recognizer = recognizer
        self.committer = committer
        self.text = text_buffer
        self.camera_factory = _camera_factory_from_environment(camera_factory)
        self.camera_index = camera_index
        self.mode = mode
        self.auto_add = auto_add
        self.capture_gap_seconds = capture_gap_seconds
        self.poll_timeout = poll_timeout
        self.backends = backends

        self._lock = threading.RLock()
        self._cond = threading.Condition()                     # signals "a new camera frame is ready"
        self._threads = []
        self._stop = threading.Event()
        self._camera = None
        self.backend = ""
        self.error = ""
        self.notice, self.notice_until = "", 0.0
        self.last_poll = time.monotonic()

        self._counter = 0
        self._raw = None                                       # (counter, BGR frame)
        self._jpeg = (0, None)                                 # (counter, JPEG bytes with the hand outline)
        self._frame_times = collections.deque(maxlen=30)
        self._last_reading = None

        self.token = secrets.token_urlsafe(9)
        self._server = None
        self.stream_url = ""

        self.hands_in_view = 0
        self.last_label, self.last_confidence = None, 0.0
        self.ambiguous = False
        self.state = None
        self.last_capture_time = -1e9
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self):
        from sign_to_text import CommitState
        self.state = CommitState(status="Camera stopped")
        self.hands_in_view = 0
        self.last_label, self.last_confidence, self.ambiguous = None, 0.0, False
        self._last_reading = None
        self.committer.reset()

    def say(self, message, seconds=NOTICE_SECONDS, now=None):
        now = time.monotonic() if now is None else now
        self.notice, self.notice_until = message, now + seconds

    @property
    def camera_on(self):
        return bool(self._threads) and not self._stop.is_set() and all(t.is_alive() for t in self._threads)

    # ------------------------------------------------------------------ camera
    def _probe(self, camera):
        """True when the camera delivers real (not black) pictures within a few seconds."""
        good, end = 0, time.monotonic() + PROBE_SECONDS
        while time.monotonic() < end:
            try:
                ok, frame = camera.read()
            except Exception:
                return False
            if ok and frame is not None and float(frame.mean()) > BLACK_LEVEL:
                good += 1
                if good >= PROBE_GOOD_FRAMES:
                    return True
        return False

    def start_camera(self):
        if self.camera_on:
            return True
        if self._threads:                                      # clean up anything half-started (outside the lock)
            self.stop_camera()
        with self._lock:
            if self.camera_on:
                return True
            self.error = ""
            opened_but_black, last_problem = False, ""
            for backend in self.backends:
                try:
                    camera = self.camera_factory(self.camera_index, backend)
                    opened = camera.isOpened()
                except Exception as problem:
                    camera, opened, last_problem = None, False, f"{type(problem).__name__}"
                    continue
                if not opened:
                    if camera is not None:
                        camera.release()
                    continue
                if self._probe(camera):
                    self._camera, self.backend = camera, backend
                    break
                camera.release()
                opened_but_black = True
            if self._camera is None:
                self.error = CAMERA_BUSY_MESSAGE if opened_but_black else (
                    "Could not open the webcam" + (f" ({last_problem})" if last_problem else "") +
                    ". Close other programs that use the camera and check Windows Settings > Privacy & security > Camera.")
                return False
            sys.setswitchinterval(0.001)                       # Python hands the interpreter to another thread every 1 ms
            self._ensure_server()                              # (default 5 ms makes a busy app slow the video down)
            self._stop.clear()
            self.last_poll = time.monotonic()
            self._counter, self._raw, self._jpeg = 0, None, (0, None)
            self._frame_times.clear()
            self._reset_state()
            self.state.status = "Show a hand to the camera"
            self._threads = [threading.Thread(target=self._capture_loop, name="isl-capture", daemon=True),
                             threading.Thread(target=self._recognize_loop, name="isl-recognize", daemon=True)]
            for thread in self._threads:
                thread.start()
            return True

    def stop_camera(self, reason=""):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        for thread in list(self._threads):
            if thread is not threading.current_thread():
                thread.join(timeout=3.0)
        with self._lock:
            if self._camera is not None:
                try:
                    self._camera.release()
                except Exception:
                    pass
                self._camera = None
            self._threads = []
            self._raw, self._jpeg = None, (0, None)
            self._reset_state()
            if reason:
                self.say(reason)

    def _fail(self, message):
        self.error = message
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def _capture_loop(self):
        failures, black_checks = 0, 0
        while not self._stop.is_set():
            if time.monotonic() - self.last_poll > self.poll_timeout:
                self.say("Camera stopped because the page was closed.")
                self._stop.set()
                break
            try:
                ok, frame = self._camera.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    self._fail("Lost the camera. Check the connection and press Start Camera again.")
                    break
                time.sleep(0.03)
                continue
            failures = 0
            with self._cond:
                self._counter += 1
                counter = self._counter
            self._frame_times.append(time.monotonic())
            if counter % 5 == 0:                               # the camera may start sending black pictures
                black_checks = black_checks + 1 if float(frame.mean()) <= BLACK_LEVEL else 0
                if black_checks >= BLACK_CHECKS_BEFORE_ERROR:
                    self._fail(CAMERA_BUSY_MESSAGE)
                    break
            ok, jpeg = cv2.imencode(".jpg", self._overlay(frame, self._last_reading), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with self._cond:
                self._raw = (counter, frame)
                if ok:
                    self._jpeg = (counter, jpeg.tobytes())
                self._cond.notify_all()
        with self._cond:
            self._cond.notify_all()
        with self._lock:                                       # release the camera when the loop ends
            if self._camera is not None:
                try:
                    self._camera.release()
                except Exception:
                    pass
                self._camera = None
            self.hands_in_view = 0
            self.state.status = "Camera stopped"
            self.last_label = None
            self._last_reading = None

    def _recognize_loop(self):
        last = -1
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: self._stop.is_set() or (self._raw is not None and self._raw[0] != last), timeout=0.5)
                if self._stop.is_set() or self._raw is None or self._raw[0] == last:
                    continue
                last, frame = self._raw
            try:
                self._analyse(frame, time.monotonic())
            except Exception as problem:                       # never let the thread die silently
                self._fail(f"Recognition error: {type(problem).__name__}: {problem}")
                break

    # ------------------------------------------------------------------ video for the browser
    def _ensure_server(self):
        if self._server is not None:
            return
        server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamHandler)
        server.daemon_threads = True
        server.session = self
        threading.Thread(target=server.serve_forever, name="isl-stream", daemon=True).start()
        self._server = server
        self.stream_url = f"http://127.0.0.1:{server.server_address[1]}/stream/{self.token}"

    def latest_jpeg(self):
        with self._cond:
            return self._jpeg

    def wait_for_jpeg(self, after, timeout=1.0):
        """The next picture newer than `after`, or (after, None) if none arrived in time."""
        with self._cond:
            self._cond.wait_for(lambda: self._stop.is_set() or self._jpeg[0] > after, timeout=timeout)
            if self._jpeg[0] > after and self._jpeg[1] is not None:
                return self._jpeg
            return after, None

    def _overlay(self, frame, reading):
        """The frame with the hand outline (BGR). A 'no hand' sign is drawn when no hand is known."""
        video = frame.copy()
        height, width = video.shape[:2]
        if reading is not None:
            segments = []
            for hand in reading.hand_list:
                points = np.array([(int(p.x * width), int(p.y * height)) for p in hand.landmark], np.int32)
                segments += [points[[a, b]] for a, b in _HAND_CONNECTIONS]
            if segments:
                cv2.polylines(video, segments, False, (255, 200, 0), 2)
            if reading.box is not None:
                x0, y0, x1, y1 = reading.box
                cv2.rectangle(video, (int(x0), int(y0)), (int(x1), int(y1)), (255, 180, 0), 2)
        else:
            cv2.putText(video, "No hand detected", (int(width * 0.2), 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (60, 60, 255), 3, cv2.LINE_AA)
        return video

    def _draw(self, frame, reading):
        return cv2.cvtColor(self._overlay(frame, reading), cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------- recognition
    def _analyse(self, frame, now):
        """Recognise one frame and update the prediction. The text changes only through
        capture_letter() (Add Letter) or, when Auto-add is ON, a held sign."""
        mode = self.mode
        reading = self.recognizer.read(frame, mode)           # slow: done outside the lock
        with self._lock:
            self._last_reading = reading
            self.hands_in_view = 0 if reading is None else reading.hands
            label, confidence = (None, 0.0) if reading is None else (reading.label, reading.confidence)
            self.ambiguous = label is not None and " or " in label     # look-alike pair (All mode)
            if self.ambiguous:
                confidence = 0.0
            self.last_label = None if (label is None or self.ambiguous) else label
            self.last_confidence = confidence
            self.state = self.committer.update(label, confidence, now, auto=self.auto_add)
            if self.ambiguous:
                self.state.status = "Look-alike sign - choose Letters or Numbers mode"
            if self.state.committed and self.text.add_letter(self.state.committed):
                self.last_capture_time = now
        return reading

    def process_frame(self, frame, now):
        """Analyse one frame and return the drawn RGB picture (used by tests and one-off checks)."""
        reading = self._analyse(frame, now)
        return self._draw(frame, reading)

    # ----------------------------------------------------------------- buttons
    def capture_letter(self, now=None):
        """Add the CURRENT prediction to the word, if it is safe to do so."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self.camera_on:
                self.say("Start the camera first", now=now)
                return False
            if self.hands_in_view == 0:
                self.say("No hand detected - nothing added", now=now)
                return False
            if self.ambiguous:
                self.say("Look-alike sign - choose Letters or Numbers mode", now=now)
                return False
            label = self.state.stable_label or self.last_label
            confidence = self.state.confidence or self.last_confidence
            if label is None:
                self.say("No clear sign yet - hold the sign steady", now=now)
                return False
            if confidence < self.committer.threshold:
                self.say(f"Confidence too low ({confidence:.0%}) - hold the sign steady", now=now)
                return False
            if now - self.last_capture_time < self.capture_gap_seconds:
                return False                                  # double click: one letter only
            if not self.text.add_letter(label):
                self.say("The text is full", now=now)
                return False
            self.last_capture_time = now
            self.committer.mark_committed(label)
            self.say(f"Added {label}", now=now)
            return True

    def perform(self, action):
        """add | space | delete | clear_word | clear_sentence. Returns True if something changed."""
        with self._lock:
            if action == "add":
                return self.capture_letter()
            if action == "space":
                done = self.text.add_space()
                self.say("Word finished" if done else "Nothing to finish: the word is empty")
                return done
            if action == "delete":
                done = self.text.delete_last()
                self.say("Deleted the last character" if done else "Nothing to delete")
                return done
            if action == "clear_word":
                done = self.text.clear_word()
                self.say("Word cleared" if done else "The word is already empty")
                return done
            if action == "clear_sentence":
                done = self.text.clear_sentence()
                self.say("Sentence cleared" if done else "The text is already empty")
                return done
        raise ValueError(f"Unknown action: {action}")

    def replace_word(self, new_word):
        """Replace the word being built (used by the spelling suggestions)."""
        with self._lock:
            new_word = str(new_word).upper().strip()
            if not new_word or " " in new_word:
                return False
            self.text.word = new_word
            self.committer.reset()
            self.say(f"Word changed to {new_word}")
            return True

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            if mode != self.mode:
                self.mode = mode
                self.committer.reset()

    def set_threshold(self, value):
        with self._lock:
            self.committer.threshold = float(min(0.99, max(0.3, value)))

    def set_auto_add(self, enabled):
        with self._lock:
            self.auto_add = bool(enabled)
            self.committer.reset()

    # ---------------------------------------------------------------- snapshot
    def _fps(self):
        times = list(self._frame_times)
        if len(times) < 2 or time.monotonic() - times[-1] > 1.5:
            return 0.0
        return (len(times) - 1) / max(1e-6, times[-1] - times[0])

    def snapshot(self, include_frame=False):
        """The latest numbers for the page. Also tells the camera that a page is still looking
        (the camera stops by itself when nobody asks for a while)."""
        now = time.monotonic()
        self.last_poll = now
        frame_rgb = None
        if include_frame:
            with self._cond:
                raw, reading = self._raw, self._last_reading
            if raw is not None:
                frame_rgb = self._draw(raw[1], reading)
        with self._lock:
            state = self.state
            label = self.last_label if self.hands_in_view else None
            return Snapshot(
                camera_on=self.camera_on, error=self.error,
                hands=self.hands_in_view, label=label,
                confidence=state.confidence if state and state.confidence else self.last_confidence,
                status=state.status if state else "", progress=state.progress if state else 0.0,
                word=self.text.current_word, sentence=self.text.sentence, text=self.text.text,
                notice=self.notice if now < self.notice_until else "",
                mode=self.mode, threshold=self.committer.threshold, auto_add=self.auto_add,
                engine=getattr(self.recognizer, "engine_name", "?"),
                stream_url=self.stream_url if self.camera_on else "",
                backend={"msmf": "Media Foundation", "dshow": "DirectShow"}.get(self.backend, self.backend),
                fps=self._fps(), frame_rgb=frame_rgb)

    def close(self):
        self.stop_camera()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        close = getattr(self.recognizer, "close", None)
        if close:
            close()


def create_default_session():
    """Build the real session: your models, your hand detector, an empty text."""
    from sign_to_text import LetterCommitter, Recognizer, Settings, TextBuffer
    settings = Settings()
    recognizer = Recognizer(engine=settings.engine)
    committer = LetterCommitter(threshold=settings.confidence_threshold, window_frames=settings.window_frames,
                                min_votes=settings.min_votes, hold_seconds=settings.hold_seconds,
                                release_seconds=settings.release_seconds)
    session = LiveSession(recognizer, committer, TextBuffer(settings.max_text_length),
                          camera_index=settings.camera_index, mode=settings.mode,
                          capture_gap_seconds=settings.capture_gap_seconds)
    # Streamlit runs Python's garbage collector after every page run. With TensorFlow and MediaPipe
    # loaded, one full collection takes about 0.2 s with every other thread paused, which cut the video
    # from 30 to 3 pictures per second (measured). Freezing the objects loaded so far makes later
    # collections skip them, so the video stays at the camera's full speed.
    import gc
    gc.collect()
    gc.freeze()
    return session
