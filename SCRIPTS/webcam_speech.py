
"""
ISL Webcam Recognition and Debugging Tool

N = Numbers mode (only 1-9)
L = Letters mode (only A-Z)
A = All signs (the app starts here; no need to pick a mode)
In All mode, signs with the same hand shape show as a pair: "1 or I" and
"2 or V". Press N or L to choose numbers or letters.
Green = steady and 95%+ confident. Orange "Uncertain" = lower confidence,
"Unsteady" = the last readings disagree (hold the sign still).
Q = Quit webcam
S = Save the frame. Then press the key of the sign you MEANT (1-9, A-Z) to
    label it (SPACE = skip). It goes to webcam_captures with the mode, what
    was shown and the confidence, and a line is added to labeled_log.csv.
The cyan "Top:" line shows the model's three best guesses for the current mode.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import logging
import re
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from motion_features import (
    FEATURE_SIZE, STATIC_FEATURE_SIZE, hands_to_static_features,
    landmarks_to_vector,
)


# ---------------- CONFIGURATION ----------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "MODELS" / "isl_mobilenetv2.keras"
LABELS_PATH = PROJECT_DIR / "MODELS" / "class_names.json"
MOTION_MODEL_PATH = PROJECT_DIR / "MODELS" / "isl_motion.keras"
LANDMARK_MODEL_PATH = PROJECT_DIR / "MODELS" / "isl_landmarks.keras"

IMAGE_SIZE = (224, 224)

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Predict once every 5 frames to reduce CPU workload.
INFERENCE_INTERVAL = 5

# Wrong answers are often confident (many wrong predictions score 90%+), while
# predictions at 99%+ are right about 99.6% of the time, so only a very
# confident, steady answer is shown in green.
CONFIDENCE_THRESHOLD = 0.95

# The shown sign is a vote over the last few predictions (about one second),
# so a single bad frame cannot flip the answer.
SMOOTHING_WINDOW = 5
MIN_AGREEMENT = 3

# The hand detector sometimes loses the hand for a frame or two (dark room,
# overlapping hands). Keep the last answer for this many frames instead of
# resetting the vote at every dropout.
MAX_MISSED_FRAMES = 8

# Hand detector settings. They match how the training photos were processed
# (lighter detector, 30% threshold). The earlier live settings (full detector,
# 55%) found different hands than training did and changed some predictions.
HAND_MODEL_COMPLEXITY = 0
HAND_DETECTION_CONFIDENCE = 0.30
HAND_TRACKING_CONFIDENCE = 0.30

# Pressing S saves frames here. Right after S, press the key of the sign you
# MEANT (1-9, A-Z); SPACE or waiting skips the label.
CAPTURE_DIR = PROJECT_DIR / "webcam_captures"
LABELED_LOG_NAME = "labeled_log.csv"
LABEL_KEYS = set("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LABEL_WAIT_SECONDS = 6.0

MOTION_CONFIDENCE_THRESHOLD = 0.85
MOTION_SEQUENCE_LENGTH = 30

WINDOW_NAME = "ISL Sign Recognition"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- LOAD MODEL ----------------

def load_model_and_labels():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

    # Keep CPU thread usage modest.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass

    logger.info("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    model_height, model_width = model.input_shape[1:3]
    if model_height is None or model_width is None:
        raise ValueError(f"Model has a dynamic image size: {model.input_shape}")
    image_size = (int(model_width), int(model_height))

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    if not isinstance(labels, list) or not labels:
        raise ValueError("The labels file is empty or invalid.")

    if model.output_shape[-1] != len(labels):
        raise ValueError(
            f"Model outputs {model.output_shape[-1]} classes, "
            f"but the labels file contains {len(labels)}."
        )

    logger.info("Model loaded successfully.")
    logger.info("Number of classes: %d", len(labels))
    logger.info("Input shape: %s", model.input_shape)

    motion_model = None
    if MOTION_MODEL_PATH.is_file():
        logger.info("Loading J/Z motion model...")
        motion_model = tf.keras.models.load_model(MOTION_MODEL_PATH)
        logger.info("J/Z motion model loaded.")

    landmark_model = None
    if LANDMARK_MODEL_PATH.is_file():
        logger.info("Loading hand-landmark classifier...")
        landmark_model = tf.keras.models.load_model(LANDMARK_MODEL_PATH)
        if landmark_model.input_shape[-1] != STATIC_FEATURE_SIZE:
            raise ValueError(
                "isl_landmarks.keras was trained with the old features. "
                "Run train_landmark_model.py again."
            )
        logger.info("Hand-landmark classifier loaded.")

    return model, labels, image_size, motion_model, landmark_model


# ---------------- PREDICTION ----------------

def hand_crop(frame, hands_landmarks, padding=0.35):
    """Square crop covering all detected hands, with padding."""
    height, width = frame.shape[:2]
    points = [point for hand in hands_landmarks for point in hand.landmark]
    xs = [point.x * width for point in points]
    ys = [point.y * height for point in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * padding)
    side = max(side, 2.0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = max(0, int(cx - side / 2))
    right = min(width, int(cx + side / 2))
    top = max(0, int(cy - side / 2))
    bottom = min(height, int(cy + side / 2))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Detected hand crop is empty")
    return crop, (left, top, right, bottom)


# Sign modes: some signs share a hand shape (e.g. "1" and "I" are both a
# raised index finger), so limiting the choice to numbers or letters
# removes that confusion.
MODES = {
    ord("n"): "numbers",
    ord("l"): "letters",
    ord("a"): "all",
}
MODE_TEXT = {"numbers": "NUMBERS", "letters": "LETTERS", "all": "ALL SIGNS"}


def allowed_mask(labels, mode):
    """True for the signs the current mode may choose from."""
    return np.array([
        mode == "all"
        or (mode == "numbers" and label.isdigit())
        or (mode == "letters" and label.isalpha())
        for label in labels
    ])


def fold_lookalikes(probabilities, labels, mode):
    """In Numbers or Letters mode, count a look-alike sign for its partner.

    1 and I (and 2 and V) have the same hand shape. When you choose Letters,
    a "2" reading means V; when you choose Numbers, a "V" reading means 2.
    """
    if mode not in ("numbers", "letters"):
        return probabilities
    folded = np.array(probabilities, dtype=np.float64, copy=True)
    for number, letter in LOOKALIKE_PAIRS:
        if number in labels and letter in labels:
            n, l = labels.index(number), labels.index(letter)
            if mode == "letters":
                folded[l] += folded[n]
            else:
                folded[n] += folded[l]
    return folded


def format_top_guesses(probabilities, labels, mode, count=3):
    """Text such as 'Top: 3 99% | 4 1% | V 0%' for the signs the mode allows."""
    masked = np.where(
        allowed_mask(labels, mode),
        fold_lookalikes(probabilities, labels, mode),
        0.0,
    )
    total = float(masked.sum())
    if total <= 0.0:
        return ""
    masked = masked / total
    best = np.argsort(masked)[::-1][:count]
    return "Top: " + " | ".join(
        f"{labels[i]} {masked[i]:.0%}" for i in best
    )


# Signs with the same hand shape in the training data. In All mode these are
# shown as a pair ("1 or I") because the model cannot tell them apart; press
# N or L to choose numbers or letters.
LOOKALIKE_PAIRS = (("1", "I"), ("2", "V"))
UNSTEADY_LABEL = "Unsteady"


def pick_label(probabilities, labels, mode):
    """Best label and confidence, only among the signs allowed by the mode.

    In All mode, a look-alike sign is returned as a pair such as "1 or I",
    with the pair's combined probability as the confidence.
    """
    allowed = allowed_mask(labels, mode)
    masked = np.where(
        allowed, fold_lookalikes(probabilities, labels, mode), 0.0
    )
    total = float(masked.sum())
    if total <= 0.0:
        return labels[int(np.argmax(probabilities))], 0.0
    masked = masked / total
    best = int(np.argmax(masked))
    label, confidence = labels[best], float(masked[best])
    if mode == "all":
        for first, second in LOOKALIKE_PAIRS:
            if label in (first, second) and first in labels and second in labels:
                confidence = float(
                    masked[labels.index(first)] + masked[labels.index(second)]
                )
                return f"{first} or {second}", confidence
    return label, confidence


def smooth_prediction(history):
    """Majority label of recent (label, confidence) predictions.

    The confidence is the average for that label. When fewer than
    MIN_AGREEMENT of the recent predictions agree, the result is "Unsteady"
    once there are enough readings to judge, and a scaled-down confidence
    before that.
    """
    votes = Counter(label for label, _ in history)
    label, count = votes.most_common(1)[0]
    confidence = float(np.mean([conf for name, conf in history if name == label]))
    if count < MIN_AGREEMENT:
        confidence *= count / MIN_AGREEMENT
        if len(history) >= MIN_AGREEMENT:
            return UNSTEADY_LABEL, confidence
    return label, confidence


def find_hands(hands, frame):
    """Hand landmarks found in a BGR frame, as a list (possibly empty).

    Overlapping hands often hide one hand from the detector, so a two-hand sign
    (H, M, Q, ...) is seen as one hand. When exactly one hand is found, its
    area is covered and the picture is searched again for the second hand.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    found = list(hands.process(rgb).multi_hand_landmarks or [])
    if len(found) != 1:
        return found
    height, width = frame.shape[:2]
    xs = [point.x * width for point in found[0].landmark]
    ys = [point.y * height for point in found[0].landmark]
    covered = rgb.copy()
    covered[max(0, int(min(ys))):int(max(ys)), max(0, int(min(xs))):int(max(xs))] = (
        int(rgb.mean())
    )
    second = hands.process(covered).multi_hand_landmarks or []
    return found + list(second[:1])


def save_capture(frame, mode, meant, shown, confidence, hands_found, top_text):
    """Save a frame with the sign the user meant, and add a line to the log."""
    CAPTURE_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    clean = re.sub(r"[^A-Za-z0-9]+", "_", shown).strip("_")
    name = (f"{stamp}_meant-{meant}_mode-{mode}_shown-{clean}"
            f"_conf-{confidence:.0%}.jpg")
    cv2.imwrite(str(CAPTURE_DIR / name), frame)
    with open(CAPTURE_DIR / LABELED_LOG_NAME, "a", encoding="utf-8") as log:
        if log.tell() == 0:
            log.write("file,meant,mode,shown,confidence,hands,top3\n")
        log.write(f"{name},{meant},{mode},{shown},{confidence:.3f},"
                  f"{hands_found},{top_text}\n")
    return name


def predict_sign(frame, model, labels, image_size, mode="all"):
    # OpenCV supplies BGR; the model was trained on RGB images.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized_frame = cv2.resize(
        rgb_frame,
        image_size,
        interpolation=cv2.INTER_LINEAR
    )

    # The model already contains Rescaling(1./255).
    # Do not divide by 255 again.
    image_batch = np.expand_dims(
        resized_frame.astype(np.float32),
        axis=0
    )

    predictions = model(image_batch, training=False).numpy()[0]

    return pick_label(predictions, labels, mode)


# ---------------- WEBCAM ----------------

def run_webcam():
    model, labels, image_size, motion_model, landmark_model = load_model_and_labels()

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions "
            "and close other applications using the camera."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    hands_api = mp.solutions.hands
    hands = hands_api.Hands(
        # Detect hands from scratch in every frame, like the training photos.
        # Video tracking gave different, noisier hand points for the same
        # frame (3 read as 4, phantom second hands) and was no faster.
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=HAND_MODEL_COMPLEXITY,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )

    predicted_label = "Show a sign"
    confidence = 0.0
    frame_count = 0
    motion_history = deque(maxlen=MOTION_SEQUENCE_LENGTH)
    missed_frames = 0
    hands_in_view = 0
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    top_text = ""
    pending_capture = None
    # Start in All mode: with the current model, number/letter mix-ups are
    # rare (about 0.05%), and picking the wrong mode caused most of the wrong
    # answers. Press N or L to force numbers or letters (e.g. to tell
    # 1 from I, or 2 from V).
    mode = "all"

    # This file will be created in the main project folder.
    save_path = PROJECT_DIR / "webcam_test.jpg"

    logger.info("Webcam started.")
    logger.info("Press N for numbers, L for letters, A for all signs.")
    logger.info("Press S to save an original frame.")
    logger.info("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                logger.warning("Could not read a webcam frame.")
                break

            # Keep an untouched copy BEFORE drawing prediction text.
            original_frame = frame.copy()

            frame_count += 1
            hand_list = find_hands(hands, frame)
            if hand_list:
                missed_frames = 0
                hands_in_view = len(hand_list)
                motion_history.append(
                    landmarks_to_vector(
                        hand_list, frame.shape
                    )
                )
                crop, box = hand_crop(
                    frame, hand_list
                )
                x0, y0, x1, y1 = box
                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 180, 0), 2)
                cv2.putText(frame, f"{len(hand_list)} hand(s) detected", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
            else:
                motion_history.append(np.zeros(FEATURE_SIZE, dtype=np.float32))
                crop = None
                hands_in_view = 0
                missed_frames += 1
                # Only forget the sign after the hand has been gone for a
                # while; a one-frame dropout keeps the current answer.
                if missed_frames > MAX_MISSED_FRAMES:
                    prediction_history.clear()
                    top_text = ""
                    predicted_label = "Show hand to camera"
                    confidence = 0.0

            if crop is not None and frame_count % INFERENCE_INTERVAL == 1:
                try:
                    if landmark_model is not None and hand_list:
                        feature = hands_to_static_features(
                            hand_list, frame.shape
                        )
                        probabilities = landmark_model(
                            feature[None, :], training=False
                        ).numpy()[0]
                        predicted_label, confidence = pick_label(
                            probabilities, labels, mode
                        )
                        top_text = format_top_guesses(probabilities, labels, mode)
                    else:
                        predicted_label, confidence = predict_sign(
                            crop, model, labels, image_size, mode
                        )
                        top_text = ""
                    # J and Z are letters, so skip them in numbers mode.
                    if (motion_model is not None
                            and mode != "numbers"
                            and len(motion_history) == MOTION_SEQUENCE_LENGTH):
                        motion_input = np.asarray(
                            [list(motion_history)], dtype=np.float32
                        )
                        motion_predictions = motion_model(
                            motion_input, training=False
                        ).numpy()[0]
                        motion_index = int(np.argmax(motion_predictions))
                        motion_label = ("J", "Z", "OTHER")[motion_index]
                        motion_confidence = float(motion_predictions[motion_index])
                        if (motion_label in {"J", "Z"}
                                and motion_confidence >= MOTION_CONFIDENCE_THRESHOLD):
                            predicted_label = motion_label
                            confidence = motion_confidence
                    prediction_history.append((predicted_label, confidence))
                    predicted_label, confidence = smooth_prediction(
                        prediction_history
                    )
                except Exception:
                    logger.exception("Prediction failed.")
                    prediction_history.clear()
                    predicted_label = "Prediction error"
                    confidence = 0.0

            if predicted_label == UNSTEADY_LABEL:
                display_text = "Unsteady - hold the sign still"
                text_color = (0, 165, 255)
            elif confidence >= CONFIDENCE_THRESHOLD:
                display_text = (
                    f"Sign: {predicted_label} "
                    f"| Confidence: {confidence:.1%}"
                )
                text_color = (0, 255, 0)
            else:
                display_text = (
                    f"Uncertain: {predicted_label} "
                    f"| Confidence: {confidence:.1%}"
                )
                text_color = (0, 165, 255)

            cv2.putText(
                frame,
                display_text,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_color,
                2,
                cv2.LINE_AA
            )
            if top_text:
                cv2.putText(
                    frame,
                    top_text,
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            frame_height, frame_width = frame.shape[:2]
            if hands_in_view == 1:
                cv2.putText(
                    frame,
                    "1 hand seen. Two-hand signs need BOTH hands in view.",
                    (10, frame_height - 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA
                )
            # A solid colored bar makes the current mode hard to miss.
            bar_color = {
                "all": (0, 140, 0),
                "numbers": (0, 120, 220),
                "letters": (170, 0, 150),
            }[mode]
            cv2.rectangle(
                frame, (0, frame_height - 36), (frame_width, frame_height),
                bar_color, -1
            )
            cv2.putText(
                frame,
                f"MODE: {MODE_TEXT[mode]}   (n numbers | l letters | a all)",
                (10, frame_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if pending_capture is not None:
                seconds_left = max(
                    0.0, pending_capture["deadline"] - time.monotonic())
                cv2.putText(frame, "SAVED. Press the sign you MEANT",
                            (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame,
                            f"(1-9 / A-Z, SPACE = skip, {seconds_left:.0f}s)",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)

            # Read the keyboard once per frame.
            key = cv2.waitKey(1) & 0xFF

            if pending_capture is not None:
                # While a capture waits for its label, keys are sign labels
                # (so N, L, A, S and Q mean those letters, not commands).
                label_key = chr(key).upper() if key != 255 else ""
                timed_out = time.monotonic() >= pending_capture["deadline"]
                if label_key in LABEL_KEYS or key == ord(" ") or timed_out:
                    meant = label_key if label_key in LABEL_KEYS else "unknown"
                    name = save_capture(meant=meant, **pending_capture["data"])
                    print(f"Saved {name}")
                    pending_capture = None

            elif key in MODES:
                mode = MODES[key]
                logger.info("Mode: %s", MODE_TEXT[mode])
                # Predict again on the next frame in the new mode, and drop
                # votes that were made in the old mode.
                frame_count = 0
                prediction_history.clear()

            elif key == ord("s"):
                saved = cv2.imwrite(
                    str(save_path),
                    original_frame
                )

                if saved:
                    print(f"Original webcam image saved: {save_path}")
                    # Keep a named copy so every misread frame can be
                    # inspected later. It is written once you press the key
                    # of the sign you meant (or skip / wait).
                    pending_capture = {
                        "deadline": time.monotonic() + LABEL_WAIT_SECONDS,
                        "data": dict(
                            frame=original_frame.copy(),
                            mode=mode,
                            shown=predicted_label,
                            confidence=confidence,
                            hands_found=len(
                                hand_list),
                            top_text=top_text,
                        ),
                    }
                else:
                    print("Could not save the webcam image.")

            elif key == ord("q"):
                break

    finally:
        if pending_capture is not None:
            # Closed while waiting for a label: keep the frame as unlabeled.
            save_capture(meant="unknown", **pending_capture["data"])
        camera.release()
        hands.close()
        cv2.destroyAllWindows()
        logger.info("Webcam closed.")


# ---------------- START ----------------

if __name__ == "__main__":
    try:
        run_webcam()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
    except Exception:
        logger.exception("Unexpected error.")
