import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from scripts.generate_readme_images import (
    torso_visibility, resize_to_width, pick_best_sample,
    pick_detection_sample, draw_overlay,
)


def _landmarks(vis=0.9):
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 0] = 0.2 + 0.6 * (np.arange(33) % 5) / 4.0
    lm[:, 1] = np.linspace(0.1, 0.9, 33)
    lm[:, 3] = vis
    return lm


class FakeEstimator:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def estimate(self, image):
        r = self.results[self.calls]
        self.calls += 1
        return r


class FakeClassifier:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def predict(self, features, visibility):
        r = self.results[self.calls]
        self.calls += 1
        return r


def _write_image(path, width=120, height=90):
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def test_torso_visibility_is_min_over_torso_indices():
    lm = _landmarks(vis=0.9)
    lm[24, 3] = 0.4
    assert torso_visibility(lm) == pytest.approx(0.4)


def test_resize_to_width_scales_only_wide_images():
    img = np.zeros((90, 1280, 3), dtype=np.uint8)
    out = resize_to_width(img, max_width=640)
    assert out.shape == (45, 640, 3)
    narrow = np.zeros((90, 300, 3), dtype=np.uint8)
    assert resize_to_width(narrow, max_width=640) is narrow


def test_pick_best_sample_prefers_highest_visibility(tmp_path):
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b.jpg")
    low = _landmarks(vis=0.5)
    high = _landmarks(vis=0.95)
    est = FakeEstimator([low, high])
    picked = pick_best_sample([(tmp_path / "a.jpg", "x"), (tmp_path / "b.jpg", "x")], est)
    assert picked is not None
    image, landmarks, path = picked
    assert path == tmp_path / "b.jpg"
    assert torso_visibility(landmarks) == pytest.approx(0.95)
    assert image.shape == (90, 120, 3)


def test_pick_best_sample_returns_none_when_no_detection(tmp_path):
    _write_image(tmp_path / "a.jpg")
    est = FakeEstimator([None])
    assert pick_best_sample([(tmp_path / "a.jpg", "x")], est) is None


def test_pick_detection_sample_prefers_correct_label(tmp_path):
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b.jpg")
    wrong = SimpleNamespace(label="balasana", confidence=0.9)
    right = SimpleNamespace(label="marjaryasana", confidence=0.8)
    est = FakeEstimator([_landmarks(vis=0.95), _landmarks(vis=0.6)])
    clf = FakeClassifier([wrong, right])
    picked = pick_detection_sample(
        [(tmp_path / "a.jpg", "x"), (tmp_path / "b.jpg", "x")], est, clf, "marjaryasana",
    )
    assert picked is not None
    _, _, result = picked
    assert result.label == "marjaryasana"


def test_pick_detection_sample_falls_back_to_best_visibility(tmp_path):
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b.jpg")
    wrong_a = SimpleNamespace(label="balasana", confidence=0.9)
    wrong_b = SimpleNamespace(label="tadasana", confidence=0.7)
    est = FakeEstimator([_landmarks(vis=0.5), _landmarks(vis=0.9)])
    clf = FakeClassifier([wrong_a, wrong_b])
    picked = pick_detection_sample(
        [(tmp_path / "a.jpg", "x"), (tmp_path / "b.jpg", "x")], est, clf, "marjaryasana",
    )
    assert picked is not None
    image, landmarks, result = picked
    assert torso_visibility(landmarks) == pytest.approx(0.9)
    assert result.label == "tadasana"


def test_draw_overlay_draws_skeleton_and_banner():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lm = _landmarks(vis=0.9)
    out = draw_overlay(img, lm, "tadasana", 0.93)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    joint = np.all(out == (80, 220, 255), axis=-1)
    line = np.all(out == (120, 220, 80), axis=-1)
    assert joint.sum() > 0
    assert line.sum() > 0
    banner = out[0:25, 0:100]
    assert np.any(np.all(banner == (255, 255, 255), axis=-1))
