import numpy as np
from unittest.mock import patch, MagicMock
from backend.app.pose_estimator import PoseEstimator


def _fake_results(landmarks_array):
    lm = MagicMock()
    pairs = []
    for row in landmarks_array:
        p = MagicMock()
        p.x, p.y, p.z, p.visibility = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        pairs.append(p)
    lm.landmark = pairs
    res = MagicMock()
    res.pose_landmarks = lm
    return res


def test_estimate_returns_array_shape_when_person_detected():
    est = PoseEstimator()
    fake = np.random.default_rng(0).uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    with patch.object(est._pose, 'process', return_value=_fake_results(fake)):
        out = est.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
    assert out is not None
    assert out.shape == (33, 4)
    np.testing.assert_allclose(out, fake, atol=1e-5)


def test_estimate_returns_none_when_no_person():
    est = PoseEstimator()
    res = MagicMock()
    res.pose_landmarks = None
    with patch.object(est._pose, 'process', return_value=res):
        out = est.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
    assert out is None
