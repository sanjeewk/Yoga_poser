from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from backend.training.extract_features import process_images


def _fake_landmarks():
    rng = np.random.default_rng(0)
    lm = rng.uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    lm[11:18, 3] = 0.95
    lm[23:30, 3] = 0.95
    return lm


def _write_tiny_image(path):
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))


def test_process_images_filters_low_visibility(tmp_path):
    _write_tiny_image(tmp_path / "a.jpg")
    _write_tiny_image(tmp_path / "b.jpg")
    paths = [(tmp_path / "a.jpg", "tadasana"), (tmp_path / "b.jpg", "vrksasana")]
    with patch("backend.training.extract_features.PoseEstimator") as Est:
        est = Est.return_value
        est.estimate.side_effect = [_fake_landmarks(), None]
        df = process_images(paths)
    assert len(df) == 1
    assert df.iloc[0]["label"] == "tadasana"
    assert all(f"f{i}" in df.columns for i in range(16))
    assert all(f"v{i}" in df.columns for i in range(33))


def test_process_images_handles_empty():
    with patch("backend.training.extract_features.PoseEstimator"):
        df = process_images([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
