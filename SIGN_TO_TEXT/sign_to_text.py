"""Sign-to-Text: build words and sentences from Indian Sign Language gestures.

You show a sign to the webcam. The program shows its CURRENT PREDICTION (the
letter or number it sees right now, with a confidence score). That prediction is
kept separate from the text you are writing:

    CURRENT PREDICTION   what the camera sees right now        e.g.  H   97%
    WORD                 the word you are building              e.g.  HEL
    SENTENCE             the finished words (after Space)       e.g.  HELLO WORLD

Nothing is added to the text unless you press ADD LETTER (or turn on Auto-add).
When the hand disappears, the text stays exactly as it was.

This file only ADDS a feature. It reuses the recognition code of your working
webcam script (SCRIPTS/webcam_speech.py) without changing it: the same models,
the same 35 labels (A-Z and 1-9), the same hand detection and the same
predictions.

Run (from the project folder):
    .venv\\Scripts\\python.exe SIGN_TO_TEXT\\sign_to_text.py

Options:
    --threshold 0.75     minimum confidence for a letter to count (0.5 - 0.99)
    --mode letters       letters (A-Z) or numbers (1-9)
    --engine auto        auto, landmarks (hand-point model) or image (MobileNetV2)
    --auto               start with Auto-add ON (hold a sign ~1 s to add it)
    --hold 1.0           seconds to hold a sign in Auto-add mode
    --no-autostart       open the window with the camera stopped

Keys (small or capital letters both work; click the window first):
    A or Enter   Add Letter           SPACE        finish the word (Space)
    D or Backspace   Delete Last      W   Clear Word       C   Clear Sentence
    T   toggle Auto-add               N / L   numbers / letters mode
    S   Start Camera   X   Stop Camera   F   Save text to a file
    + / -   change the confidence threshold       Q or Esc   Quit
The same actions are buttons under the video.
"""

import argparse
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "SCRIPTS"))

import cv2
import mediapipe as mp
import numpy as np

import webcam_speech as ws          # your existing, working recognition code (not modified)


# =============================================================== settings
@dataclass
class Settings:
    confidence_threshold: float = 0.75   # predictions below this are ignored
    window_frames: int = 8               # how many recent frames are voted on
    min_votes: int = 5                   # frames (of the window) that must agree
    capture_gap_seconds: float = 0.6     # minimum time between two added letters
    auto_add: bool = False               # False: letters are added only by Add Letter
    hold_seconds: float = 1.0            # Auto-add: how long a sign must be held
    release_seconds: float = 0.5         # Auto-add: hand away this long = letter may repeat
    mode: str = "letters"                # "letters" (A-Z) or "numbers" (1-9)
    engine: str = "auto"                 # "auto", "landmarks" or "image"
    camera_index: int = ws.CAMERA_INDEX
    max_text_length: int = 200
    autostart: bool = True


# =============================================================== the text
class TextBuffer:
    """The text, kept in two separate parts.

    done_words   words that were finished with Space  (the SENTENCE)
    word         the word being built right now       (the WORD)
    """

    def __init__(self, max_length=200):
        self.done_words = []
        self.word = ""
        self.max_length = max_length

    # -- read
    @property
    def sentence(self):
        """Only the finished words."""
        return " ".join(self.done_words)

    @property
    def current_word(self):
        return self.word

    @property
    def words(self):
        """Every word, including the one being built."""
        return self.done_words + ([self.word] if self.word else [])

    @property
    def text(self):
        """The whole text: finished words, then the word being built."""
        return " ".join(self.words)

    # -- change
    def add_letter(self, letter):
        if len(self.text) >= self.max_length:
            return False
        self.word += str(letter).upper()
        return True

    def add_space(self):
        """Finish the word: move it into the sentence. Empty words are ignored."""
        if not self.word:
            return False
        self.done_words.append(self.word)
        self.word = ""
        return True

    def delete_last(self):
        """Delete the last character. With an empty word, the previous word is
        re-opened and its last character removed."""
        if self.word:
            self.word = self.word[:-1]
            return True
        if self.done_words:
            self.word = self.done_words.pop()[:-1]
            return True
        return False

    def clear_word(self):
        had = bool(self.word)
        self.word = ""
        return had

    def clear_sentence(self):
        """Clear everything: the finished words and the word being built."""
        had = bool(self.done_words or self.word)
        self.done_words, self.word = [], ""
        return had

    clear = clear_sentence


# =============================================================== stable prediction / auto-add
@dataclass
class CommitState:
    stable_label: str = None      # the sign the recent frames agree on
    confidence: float = 0.0
    progress: float = 0.0         # 0..1 while a sign is being held (Auto-add only)
    status: str = ""
    committed: str = None         # Auto-add only: set on the one frame a letter is added


class LetterCommitter:
    """Turns noisy frame-by-frame predictions into one stable prediction.

    A frame counts only if its confidence is at least `threshold`. A sign is
    "stable" when `min_votes` of the last `window_frames` frames agree.

    With auto=True it also adds a letter once when a stable sign has been held
    for `hold_seconds`. The same letter can be added again only after the hand
    was away for `release_seconds`, or after a different sign was shown.
    """

    def __init__(self, threshold=0.75, window_frames=8, min_votes=5,
                 hold_seconds=1.0, release_seconds=0.5):
        self.threshold = threshold
        self.window_frames = window_frames
        self.min_votes = min_votes
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.reset()

    def reset(self):
        self.window = deque(maxlen=self.window_frames)
        self.hold_label = None
        self.hold_start = None
        self.last_committed = None
        self.armed = True
        self.hand_gone_since = None

    def mark_committed(self, label):
        """Call after a letter was added by hand so Auto-add does not add it again."""
        self.last_committed = label
        self.armed = False
        self.hold_label = None
        self.hold_start = None
        self.window.clear()

    def update(self, label, confidence, now, auto=True):
        """label is None when no hand is visible. Returns a CommitState."""
        if label is None:                                    # no hand
            if self.hand_gone_since is None:
                self.hand_gone_since = now
            if now - self.hand_gone_since >= self.release_seconds:
                self.armed = True
            self.window.clear()
            self.hold_label, self.hold_start = None, None
            return CommitState(status="No hand detected")
        self.hand_gone_since = None

        accepted = label if confidence >= self.threshold else None
        self.window.append((accepted, confidence))
        votes = Counter(item[0] for item in self.window if item[0] is not None)
        stable, count = votes.most_common(1)[0] if votes else (None, 0)

        if accepted is None and stable is None:
            self.hold_label, self.hold_start = None, None
            return CommitState(confidence=confidence,
                               status=f"Low confidence ({confidence:.0%}) - hold the sign steady")
        if stable is None or count < self.min_votes:
            self.hold_label, self.hold_start = None, None
            return CommitState(stable_label=stable, confidence=confidence, status="Detecting...")

        mean_conf = float(np.mean([c for l, c in self.window if l == stable]))
        if not auto:                                         # manual: wait for Add Letter
            return CommitState(stable, mean_conf, 0.0, f"Ready: {stable} - press Add Letter")

        if stable != self.last_committed:                    # a different sign re-arms
            self.armed = True
        if not self.armed:
            self.hold_label, self.hold_start = None, None
            return CommitState(stable, mean_conf, 0.0,
                               f"Added {stable} - lower your hand or change the sign for the next letter")
        if self.hold_label != stable:
            self.hold_label, self.hold_start = stable, now
        progress = min(1.0, (now - self.hold_start) / self.hold_seconds)
        if progress >= 1.0:                                  # held long enough: add it once
            self.mark_committed(stable)
            return CommitState(stable, mean_conf, 1.0, f"Added {stable}", committed=stable)
        return CommitState(stable, mean_conf, progress, f"Hold steady... {stable}")


# =============================================================== recognition (reused code)
@dataclass
class Reading:
    label: str
    confidence: float
    hands: int
    box: tuple = None
    hand_list: list = field(default_factory=list)


class Recognizer:
    """Predicts a sign from a webcam frame, using the code of webcam_speech.py."""

    def __init__(self, engine="auto"):
        model, labels, image_size, _motion, landmark_model = ws.load_model_and_labels()
        if engine == "landmarks" and landmark_model is None:
            raise RuntimeError("engine 'landmarks' needs MODELS/isl_landmarks.keras")
        self.model, self.labels, self.image_size = model, labels, image_size
        self.landmark_model = landmark_model
        self.use_landmarks = engine in ("auto", "landmarks") and landmark_model is not None
        # same hand detector settings as the webcam script
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=ws.HAND_MODEL_COMPLEXITY,
            min_detection_confidence=ws.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=ws.HAND_TRACKING_CONFIDENCE,
        )

    @property
    def engine_name(self):
        return "hand-point model" if self.use_landmarks else "MobileNetV2"

    def read(self, frame, mode):
        """Return a Reading, or None when no hand is found."""
        hand_list = ws.find_hands(self.hands, frame)
        if not hand_list:
            return None
        try:
            crop, box = ws.hand_crop(frame, hand_list)
        except ValueError:
            crop, box = None, None
        if self.use_landmarks:
            feature = ws.hands_to_static_features(hand_list, frame.shape)
            probabilities = self.landmark_model(feature[None, :], training=False).numpy()[0]
            label, confidence = ws.pick_label(probabilities, self.labels, mode)
        else:
            if crop is None:
                return None
            label, confidence = ws.predict_sign(crop, self.model, self.labels, self.image_size, mode)
        return Reading(label, float(confidence), len(hand_list), box, hand_list)

    def close(self):
        self.hands.close()


# =============================================================== the window
VIDEO_W, VIDEO_H = 640, 480
PANEL_W = 380
BUTTONS_H = 150
CANVAS_W, CANVAS_H = VIDEO_W + PANEL_W, VIDEO_H + BUTTONS_H
WINDOW_NAME = "ISL Sign-to-Text"
FONT = cv2.FONT_HERSHEY_SIMPLEX
BUTTON_ROWS = [                  # three rows of buttons under the video
    ["add", "space", "delete", "toggle_auto"],
    ["clear_word", "clear_sentence", "save", "toggle_mode"],
    ["start", "stop", "quit"],
]
BUTTON_TEXT = {
    "add": "Add Letter", "space": "Space", "delete": "Delete Last",
    "clear_word": "Clear Word", "clear_sentence": "Clear Sentence", "save": "Save Text",
    "start": "Start Camera", "stop": "Stop Camera", "quit": "Quit",
}
KEY_ACTIONS = {
    ord("a"): "add", 13: "add", 32: "space", 8: "delete", ord("d"): "delete",
    ord("w"): "clear_word", ord("c"): "clear_sentence", ord("t"): "toggle_auto",
    ord("s"): "start", ord("x"): "stop", ord("f"): "save",
    ord("n"): "numbers", ord("l"): "letters", ord("+"): "threshold_up",
    ord("="): "threshold_up", ord("-"): "threshold_down", ord("q"): "quit", 27: "quit",
}
GREEN, ORANGE, RED, GRAY, WHITE, YELLOW = (0, 200, 0), (0, 165, 255), (60, 60, 255), (150, 150, 150), (255, 255, 255), (0, 255, 255)


def put_text(image, text, org, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(image, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def wrap_text(text, max_width, scale=0.8, thickness=2):
    """Split text into lines that fit max_width pixels."""
    lines, line = [], ""
    for word in text.split(" "):
        trial = f"{line} {word}" if line else word
        if cv2.getTextSize(trial, FONT, scale, thickness)[0][0] <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


class SignToTextApp:
    def __init__(self, settings, recognizer, camera_factory=None):
        self.settings = settings
        self.recognizer = recognizer
        self.camera_factory = camera_factory or self._open_camera
        self.text = TextBuffer(settings.max_text_length)
        self.committer = LetterCommitter(
            settings.confidence_threshold, settings.window_frames, settings.min_votes,
            settings.hold_seconds, settings.release_seconds)
        self.camera = None
        self.hands_in_view = 0
        self.last_label, self.last_confidence = None, 0.0     # the current prediction
        self.ambiguous = False
        self.last_capture_time = -1e9
        self.state = CommitState(status="Camera stopped")
        self.quit = False
        self.clicked = None
        self.notice, self.notice_until = "", 0.0
        self.button_rects = self._button_rects()

    # ---------------------------------------------------------- camera
    def _open_camera(self):
        camera = cv2.VideoCapture(self.settings.camera_index, cv2.CAP_DSHOW)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)
        return camera

    def start_camera(self):
        if self.camera is not None:
            return
        camera = self.camera_factory()
        if not camera.isOpened():
            camera.release()
            self.say("Could not open the webcam - close other apps that use it")
            return
        self.camera = camera
        self.committer.reset()
        self.say("Camera started")

    def stop_camera(self):
        """Stop the camera. The text is kept."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.committer.reset()
        self.hands_in_view = 0
        self.last_label, self.last_confidence, self.ambiguous = None, 0.0, False
        self.state = CommitState(status="Camera stopped")

    # ---------------------------------------------------------- the text actions
    def say(self, message, seconds=2.5):
        self.notice, self.notice_until = message, time.monotonic() + seconds

    def capture_letter(self, now=None):
        """Add the CURRENT prediction to the word, if it is safe to do so."""
        now = time.monotonic() if now is None else now
        if self.camera is None:
            self.say("Start the camera first")
            return False
        if self.hands_in_view == 0:
            self.say("No hand detected - nothing added")
            return False
        if self.ambiguous:
            self.say("Look-alike sign - press L (letters) or N (numbers)")
            return False
        label = self.state.stable_label or self.last_label
        confidence = self.state.confidence or self.last_confidence
        if label is None:
            self.say("No clear sign yet - hold the sign steady")
            return False
        if confidence < self.committer.threshold:
            self.say(f"Confidence too low ({confidence:.0%}) - hold the sign steady")
            return False
        if now - self.last_capture_time < self.settings.capture_gap_seconds:
            return False                       # key repeat or double click: one letter only
        if not self.text.add_letter(label):
            self.say("The text is full")
            return False
        self.last_capture_time = now
        self.committer.mark_committed(label)
        self.say(f"Added {label}")
        return True

    def perform(self, action):
        if action == "start":
            self.start_camera()
        elif action == "stop":
            self.stop_camera()
            self.say("Camera stopped (your text is kept)")
        elif action == "add":
            self.capture_letter()
        elif action == "space":
            if self.text.add_space():
                self.say("Word finished - added to the sentence")
            else:
                self.say("Add a letter before pressing Space")
            self.committer.reset()
        elif action == "delete":
            if not self.text.delete_last():
                self.say("Nothing to delete")
            self.committer.reset()
        elif action == "clear_word":
            self.say("Word cleared" if self.text.clear_word() else "The word is already empty")
            self.committer.reset()
        elif action == "clear_sentence":
            self.say("Sentence cleared" if self.text.clear_sentence() else "Nothing to clear")
            self.committer.reset()
        elif action == "save":
            self.save_text()
        elif action == "toggle_auto":
            self.settings.auto_add = not self.settings.auto_add
            self.committer.reset()
            self.say("Auto-add ON: hold a sign to add it" if self.settings.auto_add
                     else "Auto-add OFF: press Add Letter")
        elif action == "toggle_mode":
            self.perform("numbers" if self.settings.mode == "letters" else "letters")
        elif action in ("letters", "numbers"):
            self.settings.mode = action
            self.committer.reset()
            self.say(f"Mode: {action.upper()}")
        elif action in ("threshold_up", "threshold_down"):
            step = 0.05 if action == "threshold_up" else -0.05
            value = round(min(0.99, max(0.50, self.committer.threshold + step)), 2)
            self.committer.threshold = self.settings.confidence_threshold = value
            self.say(f"Confidence threshold: {value:.0%}")
        elif action == "quit":
            self.quit = True

    def handle_key(self, key):
        if key == 255 or key < 0:
            return
        if 65 <= key <= 90:                  # capital letters (Caps Lock / Shift) work too
            key += 32
        action = KEY_ACTIONS.get(key)
        if action:
            self.perform(action)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, (x0, y0, x1, y1) in self.button_rects.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.clicked = action

    def save_text(self):
        if not self.text.text:
            self.say("Nothing to save yet")
            return
        path = Path(__file__).resolve().parent / "saved_text.txt"
        with open(path, "a", encoding="utf-8") as file:
            file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {self.text.text}\n")
        self.say(f"Saved to {path.name}")

    # ---------------------------------------------------------- one frame
    def step(self, frame, now):
        """Recognise one webcam frame and return the picture to show.

        Only the CURRENT PREDICTION changes here. The text changes only when
        a letter is captured (Add Letter) or, if Auto-add is ON, held long enough.
        """
        reading = self.recognizer.read(frame, self.settings.mode)
        self.hands_in_view = 0 if reading is None else reading.hands
        label, confidence = (None, 0.0) if reading is None else (reading.label, reading.confidence)
        self.ambiguous = label is not None and " or " in label   # look-alike pair (All mode only)
        if self.ambiguous:
            confidence = 0.0
        self.last_label = None if (label is None or self.ambiguous) else label
        self.last_confidence = confidence
        self.state = self.committer.update(label, confidence, now, auto=self.settings.auto_add)
        if self.ambiguous:
            self.state.status = "Look-alike sign - press L (letters) or N (numbers)"
        if self.state.committed:                          # Auto-add only
            if self.text.add_letter(self.state.committed):
                self.last_capture_time = now
        return self.render(frame, reading)

    # ---------------------------------------------------------- drawing
    def _button_rects(self):
        rects = {}
        gap, height = 8, 42
        for row_number, row in enumerate(BUTTON_ROWS):
            width = (CANVAS_W - gap * 5) // 4
            for column, action in enumerate(row):
                x0 = gap + column * (width + gap)
                y0 = VIDEO_H + 10 + row_number * (height + 6)
                rects[action] = (x0, y0, x0 + width, y0 + height)
        return rects

    def button_label(self, action):
        if action == "toggle_auto":
            return f"Auto-add: {'ON' if self.settings.auto_add else 'OFF'}"
        if action == "toggle_mode":
            return f"Mode: {self.settings.mode.upper()}"
        return BUTTON_TEXT[action]

    def render_video(self, frame, reading):
        if frame is None:
            video = np.full((VIDEO_H, VIDEO_W, 3), 40, np.uint8)
            put_text(video, "Camera stopped", (190, 225), 1.0, (200, 200, 200), 2)
            put_text(video, "Press S or click Start Camera", (165, 265), 0.7, (200, 200, 200))
            put_text(video, "(your text is kept)", (225, 300), 0.6, (150, 150, 150))
            return video
        video = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        if reading is not None:
            sx, sy = VIDEO_W / frame.shape[1], VIDEO_H / frame.shape[0]
            for hand in reading.hand_list:
                points = [(int(p.x * VIDEO_W), int(p.y * VIDEO_H)) for p in hand.landmark]
                for a, b in mp.solutions.hands.HAND_CONNECTIONS:
                    cv2.line(video, points[a], points[b], (255, 200, 0), 2)
            if reading.box is not None:
                x0, y0, x1, y1 = reading.box
                cv2.rectangle(video, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)),
                              GREEN if self.state.progress > 0 else (255, 180, 0), 2)
        else:
            put_text(video, "No hand detected", (150, 60), 1.2, RED, 3)
        if self.settings.auto_add:                                 # progress bar (Auto-add only)
            cv2.rectangle(video, (12, VIDEO_H - 36), (VIDEO_W - 12, VIDEO_H - 20), (60, 60, 60), -1)
            cv2.rectangle(video, (12, VIDEO_H - 36),
                          (12 + int((VIDEO_W - 24) * self.state.progress), VIDEO_H - 20), GREEN, -1)
        put_text(video, f"Mode: {self.settings.mode.upper()}   Threshold: {self.committer.threshold:.0%}   "
                        f"{self.recognizer.engine_name}", (12, VIDEO_H - 6), 0.5, (200, 200, 200))
        if time.monotonic() < self.notice_until:
            put_text(video, self.notice, (12, 24), 0.65, YELLOW, 2)
        return video

    def render_panel(self):
        panel = np.full((VIDEO_H, PANEL_W, 3), 30, np.uint8)
        # ---- 1. CURRENT PREDICTION
        put_text(panel, "CURRENT PREDICTION", (14, 24), 0.55, GRAY)
        camera_on = self.camera is not None
        has_hand = camera_on and self.hands_in_view > 0
        shown = (self.state.stable_label or self.last_label) if has_hand else None
        confidence = (self.state.confidence or self.last_confidence) if has_hand else 0.0
        accepted = has_hand and shown is not None and confidence >= self.committer.threshold
        color = GREEN if accepted else (ORANGE if has_hand else GRAY)
        put_text(panel, shown if shown else ("?" if (has_hand and self.ambiguous) else "-"),
                 (22, 120), 3.6, color, 6)
        put_text(panel, "Confidence", (178, 62), 0.55, GRAY)
        put_text(panel, f"{confidence:.0%}" if has_hand else "--", (178, 112), 1.6, color, 3)
        cv2.rectangle(panel, (14, 134), (PANEL_W - 14, 152), (70, 70, 70), -1)
        bar_end = 14 + int((PANEL_W - 28) * min(1.0, confidence))
        cv2.rectangle(panel, (14, 134), (bar_end, 152), color, -1)
        marker = 14 + int((PANEL_W - 28) * self.committer.threshold)
        cv2.line(panel, (marker, 130), (marker, 156), WHITE, 2)
        if not camera_on:
            status, status_color = "Camera stopped", GRAY
        elif not has_hand:
            status, status_color = "No hand detected", RED
        else:
            status, status_color = self.state.status, (WHITE if accepted else ORANGE)
        for number, line in enumerate(wrap_text(status, PANEL_W - 28, 0.55, 1)[:2]):
            put_text(panel, line, (14, 180 + number * 22), 0.55, status_color, 1)
        cv2.line(panel, (10, 215), (PANEL_W - 10, 215), (80, 80, 80), 1)
        # ---- 2. WORD (being built)
        put_text(panel, "WORD  (being built)", (14, 240), 0.55, GRAY)
        cursor = "|" if int(time.monotonic() * 2) % 2 == 0 else " "
        word_lines = wrap_text(self.text.word + cursor, PANEL_W - 28, 1.1, 2)
        put_text(panel, word_lines[-1], (14, 282), 1.1, YELLOW, 2)
        cv2.line(panel, (10, 305), (PANEL_W - 10, 305), (80, 80, 80), 1)
        # ---- 3. SENTENCE (finished words)
        count = len(self.text.done_words)
        put_text(panel, f"SENTENCE  ({count} word{'' if count == 1 else 's'})", (14, 330), 0.55, GRAY)
        lines = wrap_text(self.text.sentence, PANEL_W - 28, 0.7, 1) if self.text.sentence else [""]
        for number, line in enumerate(lines[-4:]):
            put_text(panel, line, (14, 362 + number * 30), 0.7, WHITE, 1)
        return panel

    def render(self, frame, reading):
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 20, np.uint8)
        canvas[:VIDEO_H, :VIDEO_W] = self.render_video(frame, reading)
        canvas[:VIDEO_H, VIDEO_W:] = self.render_panel()
        for row in BUTTON_ROWS:
            for action in row:
                x0, y0, x1, y1 = self.button_rects[action]
                camera_on = self.camera is not None
                disabled = ((action == "start" and camera_on) or (action in ("stop", "add") and not camera_on))
                fill = (70, 70, 70) if disabled else {"add": (60, 130, 40), "quit": (50, 50, 140)}.get(action, (110, 80, 40))
                cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)
                label = self.button_label(action)
                size = cv2.getTextSize(label, FONT, 0.6, 1)[0]
                put_text(canvas, label, (x0 + (x1 - x0 - size[0]) // 2, y0 + 28), 0.6,
                         GRAY if disabled else WHITE)
        return canvas

    # ---------------------------------------------------------- main loop
    def run(self):
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        if self.settings.autostart:
            self.start_camera()
        try:
            while not self.quit:
                frame = None
                if self.camera is not None:
                    ok, frame = self.camera.read()
                    if not ok:
                        self.say("Could not read a camera frame - camera stopped")
                        self.stop_camera()
                        frame = None
                if frame is not None:
                    canvas = self.step(frame, time.monotonic())
                else:
                    canvas = self.render(None, None)
                cv2.imshow(WINDOW_NAME, canvas)
                self.handle_key(cv2.waitKey(1) & 0xFF)
                if self.clicked:
                    self.perform(self.clicked)
                    self.clicked = None
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break                                    # window closed with the X button
        finally:
            self.stop_camera()
            self.recognizer.close()
            cv2.destroyAllWindows()


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description="ISL Sign-to-Text")
    parser.add_argument("--threshold", type=float, default=0.75, help="confidence needed (0.5 - 0.99)")
    parser.add_argument("--hold", type=float, default=1.0, help="Auto-add: seconds to hold a sign")
    parser.add_argument("--mode", choices=("letters", "numbers"), default="letters")
    parser.add_argument("--engine", choices=("auto", "landmarks", "image"), default="auto")
    parser.add_argument("--camera-index", type=int, default=ws.CAMERA_INDEX)
    parser.add_argument("--auto", action="store_true", help="start with Auto-add ON")
    parser.add_argument("--no-autostart", action="store_true", help="open with the camera stopped")
    args = parser.parse_args(arguments)
    return Settings(
        confidence_threshold=min(0.99, max(0.5, args.threshold)),
        hold_seconds=max(0.2, args.hold),
        mode=args.mode,
        engine=args.engine,
        camera_index=args.camera_index,
        auto_add=args.auto,
        autostart=not args.no_autostart,
    )


def main():
    settings = parse_arguments()
    print("Loading models (this can take up to a minute) ...", flush=True)
    try:
        recognizer = Recognizer(settings.engine)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return
    print(f"Using the {recognizer.engine_name}. Keys: see the top of sign_to_text.py", flush=True)
    SignToTextApp(settings, recognizer).run()


if __name__ == "__main__":
    main()
