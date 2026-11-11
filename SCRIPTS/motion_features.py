"""Shared MediaPipe hand-landmark features for dynamic sign recognition."""

import numpy as np


MAX_HANDS = 2
POINTS_PER_HAND = 21
FEATURES_PER_HAND = POINTS_PER_HAND * 3 + 3
FEATURE_SIZE = MAX_HANDS * FEATURES_PER_HAND


def landmarks_to_vector(hands_landmarks, frame_shape):
    """Convert up to two hands to a scale-normalized, fixed-size vector."""
    height, width = frame_shape[:2]
    hands = []
    for hand in hands_landmarks[:MAX_HANDS]:
        points = np.asarray(
            [[point.x * width, point.y * height, point.z * width]
             for point in hand.landmark],
            dtype=np.float32,
        )
        wrist = points[0].copy()
        scale = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
        points[:, 0] = (points[:, 0] - wrist[0]) / scale
        points[:, 1] = (points[:, 1] - wrist[1]) / scale
        points[:, 2] = (points[:, 2] - wrist[2]) / scale
        global_info = np.asarray(
            [wrist[0] / max(width, 1), wrist[1] / max(height, 1),
             scale / max(width, height, 1)],
            dtype=np.float32,
        )
        hands.append((wrist[0] / max(width, 1),
                      np.concatenate([points.reshape(-1), global_info])))

    hands.sort(key=lambda item: item[0])
    vector = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for slot, (_, values) in enumerate(hands):
        start = slot * FEATURES_PER_HAND
        vector[start:start + FEATURES_PER_HAND] = values
    return vector


# ---------------- STATIC (single-frame) SIGN FEATURES ----------------
# These features describe only the hand SHAPE and, for two-hand signs, where
# the hands are relative to each other. They do NOT include where the hand is
# on screen or how big it is, so camera framing cannot change the prediction.

STATIC_PAIR_FEATURES = 3
KEY = [0, 4, 8, 12, 16, 20]
EXTRA_FEATURES = 2 * 15 + 36
STATIC_FEATURE_SIZE = MAX_HANDS * POINTS_PER_HAND * 3 + STATIC_PAIR_FEATURES + EXTRA_FEATURES


def landmarks_to_points(hands_landmarks, frame_shape):
    """Return up to two hands as a (2, 21, 3) pixel array and the hand count."""
    height, width = frame_shape[:2]
    points = np.zeros((MAX_HANDS, POINTS_PER_HAND, 3), dtype=np.float32)
    count = 0
    for hand in hands_landmarks[:MAX_HANDS]:
        points[count] = [[p.x * width, p.y * height, p.z * width]
                         for p in hand.landmark]
        count += 1
    return points, count


def mirror_points(points, count):
    """Flip hands left-right, so a left-hand sign looks like a right-hand one."""
    mirrored = points.copy()
    mirrored[:count, :, 0] *= -1.0
    return mirrored


def static_features(points, count):
    """Shape-only feature vector from a (2, 21, 3) pixel array."""
    hands = sorted(points[:count], key=lambda hand: hand[0, 0])
    vector = np.zeros(STATIC_FEATURE_SIZE, dtype=np.float32)
    wrists, scales = [], []
    for slot, hand in enumerate(hands):
        wrist = hand[0]
        scale = max(float(np.ptp(hand[:, 0])), float(np.ptp(hand[:, 1])), 1.0)
        normalized = (hand - wrist) / scale
        start = slot * POINTS_PER_HAND * 3
        vector[start:start + POINTS_PER_HAND * 3] = normalized.reshape(-1)
        wrists.append(wrist)
        scales.append(scale)
    base = MAX_HANDS * POINTS_PER_HAND * 3
    if len(hands) == 2:
        mean_scale = (scales[0] + scales[1]) / 2.0
        offset = (wrists[1] - wrists[0]) / mean_scale
        vector[base:base + 3] = [offset[0], offset[1], np.log(scales[1] / scales[0])]
    extra = np.zeros(EXTRA_FEATURES, dtype=np.float32)
    iu = np.triu_indices(len(KEY), 1)
    for slot, hand in enumerate(hands):
        keys = hand[KEY]
        d = np.linalg.norm(keys[:, None] - keys[None], axis=2)[iu] / scales[slot]
        extra[slot * 15:(slot + 1) * 15] = d
    if len(hands) == 2:
        a, b = hands[0][KEY], hands[1][KEY]
        cross = np.linalg.norm(a[:, None] - b[None], axis=2).reshape(-1) / mean_scale
        extra[30:] = cross
    vector[base + 3:] = extra
    return vector


def hands_to_static_features(hands_landmarks, frame_shape):
    points, count = landmarks_to_points(hands_landmarks, frame_shape)
    return static_features(points, count)


def tilt_points(points, count, rng, max_roll=25.0, max_pitch=20.0, max_yaw=25.0):
    """Randomly rotate the hand(s) in 3D, as if held at a different angle.

    All hands in one sample are rotated together around their common centre.
    Angles are in degrees.
    """
    if count == 0:
        return points
    roll, pitch, yaw = np.radians(rng.uniform(
        [-max_roll, -max_pitch, -max_yaw], [max_roll, max_pitch, max_yaw]))
    rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    rotation = (rz @ rx @ ry).astype(np.float32)
    tilted = points.copy()
    centre = tilted[:count].reshape(-1, 3).mean(axis=0)
    tilted[:count] = (tilted[:count] - centre) @ rotation.T + centre
    return tilted
