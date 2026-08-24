import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.classifier import load_default
from backend.app.features import extract_features
from backend.app.pose_estimator import PoseEstimator
from backend.training.download_data import find_images

TORSO_KEYPOINT_INDICES = [11, 12, 23, 24]
MAX_SCAN_PER_POSE = 25
POSE_KEYS = [
    "tadasana", "adho_mukha_svanasana", "virabhadrasana_i", "virabhadrasana_ii",
    "vrksasana", "bhujangasana", "balasana", "marjaryasana",
]
DETECTION_KEYS = ["virabhadrasana_ii", "adho_mukha_svanasana", "vrksasana", "marjaryasana"]

POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
    [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
    [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
]


def torso_visibility(landmarks):
    return float(np.min(np.asarray(landmarks)[TORSO_KEYPOINT_INDICES, 3]))


def resize_to_width(image, max_width=640):
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def pick_best_sample(candidates, estimator, max_scan=MAX_SCAN_PER_POSE):
    best = None
    for path, _label in list(candidates)[:max_scan]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        landmarks = estimator.estimate(image)
        if landmarks is None:
            continue
        score = torso_visibility(landmarks)
        if best is None or score > best[0]:
            best = (score, image, landmarks, path)
    if best is None:
        return None
    _, image, landmarks, path = best
    return resize_to_width(image), landmarks, path


def pick_detection_sample(candidates, estimator, classifier, key,
                          max_scan=MAX_SCAN_PER_POSE):
    best_any = None
    best_correct = None
    for path, _label in list(candidates)[:max_scan]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        landmarks = estimator.estimate(image)
        if landmarks is None:
            continue
        score = torso_visibility(landmarks)
        feats, vis = extract_features(landmarks)
        result = classifier.predict(feats, vis)
        entry = (score, image, landmarks, result, path)
        if best_any is None or score > best_any[0]:
            best_any = entry
        if result.label == key and (best_correct is None or score > best_correct[0]):
            best_correct = entry
    chosen = best_correct or best_any
    if chosen is None:
        return None
    _, image, landmarks, result, _path = chosen
    return resize_to_width(image), landmarks, result


def draw_overlay(image, landmarks, label, confidence):
    out = image.copy()
    h, w = out.shape[:2]
    for a, b in POSE_CONNECTIONS:
        p1, p2 = landmarks[a], landmarks[b]
        cv2.line(out, (int(p1[0] * w), int(p1[1] * h)),
                 (int(p2[0] * w), int(p2[1] * h)), (120, 220, 80), 3)
    for lm in landmarks:
        cv2.circle(out, (int(lm[0] * w), int(lm[1] * h)), 4, (80, 220, 255), -1)
    text = f"{label}  {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(out, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    return out
