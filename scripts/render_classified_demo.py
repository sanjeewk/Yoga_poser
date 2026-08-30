"""Render a short landing-page demo with real pose-classifier output."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.classifier import load_default
from backend.app.features import extract_features
from backend.app.pose_estimator import PoseEstimator


PINK = (136, 95, 249)
LILAC = (230, 202, 214)
WHITE = (250, 246, 251)
PLUM = (55, 41, 48)


def _rounded_panel(frame: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, start, end, PLUM, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)


def _draw_skeleton(frame: np.ndarray, landmarks: np.ndarray) -> None:
    height, width = frame.shape[:2]
    points = [(int(point[0] * width), int(point[1] * height)) for point in landmarks]

    for start, end in mp.solutions.pose.POSE_CONNECTIONS:
        if landmarks[start, 3] < 0.55 or landmarks[end, 3] < 0.55:
            continue
        cv2.line(frame, points[start], points[end], PLUM, 8, cv2.LINE_AA)
        cv2.line(frame, points[start], points[end], PINK, 4, cv2.LINE_AA)

    for index, point in enumerate(points):
        if landmarks[index, 3] < 0.55:
            continue
        cv2.circle(frame, point, 6, PLUM, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 3, LILAC, -1, cv2.LINE_AA)


def _draw_status(frame: np.ndarray, label: str, confidence: float) -> None:
    _rounded_panel(frame, (38, 34), (460, 235))
    cv2.circle(frame, (66, 65), 8, PINK, -1, cv2.LINE_AA)
    cv2.putText(frame, "LIVE CLASSIFIER", (88, 73), cv2.FONT_HERSHEY_DUPLEX,
                0.68, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (58, 145), cv2.FONT_HERSHEY_DUPLEX,
                1.45, WHITE, 3, cv2.LINE_AA)
    cv2.putText(frame, f"CONFIDENCE  {confidence:.0%}", (58, 184),
                cv2.FONT_HERSHEY_DUPLEX, 0.66, LILAC, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (58, 205), (425, 217), (88, 67, 95), -1, cv2.LINE_AA)
    bar_end = 58 + round(367 * min(max(confidence, 0.0), 1.0))
    cv2.rectangle(frame, (58, 205), (bar_end, 217), PINK, -1, cv2.LINE_AA)

    _rounded_panel(frame, (38, frame.shape[0] - 85), (370, frame.shape[0] - 30))
    cv2.putText(frame, "33 LANDMARKS TRACKED", (58, frame.shape[0] - 49),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, WHITE, 1, cv2.LINE_AA)


def render(input_path: Path, output_path: Path, start: float, duration: float,
           width: int) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {input_path}")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = round(width * source_height / source_width)
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_limit = round(duration * fps)
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    estimator = PoseEstimator(static_image_mode=False)
    classifier = load_default()
    if classifier is None:
        raise RuntimeError("The default classifier model is unavailable")

    smoothed: dict[str, float] | None = None
    written = 0
    try:
        while written < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            landmarks = estimator.estimate(frame)
            if landmarks is not None:
                features, visibility = extract_features(landmarks)
                prediction = classifier.predict(features, visibility)
                probabilities = prediction.probabilities
                if smoothed is None:
                    smoothed = probabilities.copy()
                else:
                    smoothed = {
                        name: 0.86 * smoothed.get(name, 0.0) + 0.14 * probability
                        for name, probability in probabilities.items()
                    }
                top_label, confidence = max(smoothed.items(), key=lambda item: item[1])
                label = "TREE POSE" if top_label == "vrksasana" and confidence >= 0.45 else "FINDING POSE"
                _draw_skeleton(frame, landmarks)
            else:
                label, confidence = "FINDING POSE", 0.0

            _draw_status(frame, label, confidence)
            assert encoder.stdin is not None
            encoder.stdin.write(frame.tobytes())
            written += 1
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    if written == 0:
        raise RuntimeError("No frames were rendered")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=2.5)
    parser.add_argument("--duration", type=float, default=9.0)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()
    render(args.input, args.output, args.start, args.duration, args.width)


if __name__ == "__main__":
    main()
