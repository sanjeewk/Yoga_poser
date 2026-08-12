import numpy as np
from backend.app.features import extract_features, FEATURE_NAMES


def test_features_shape():
    rng = np.random.default_rng(0)
    lm = rng.uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    lm[:, 3] = 0.95
    feats, vis = extract_features(lm)
    assert feats.shape == (16,)
    assert vis.shape == (33,)
    assert len(FEATURE_NAMES) == 16


def test_translation_invariance():
    rng = np.random.default_rng(1)
    lm = rng.uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    lm[:, 3] = 0.95
    f1, _ = extract_features(lm)
    shifted = lm.copy()
    shifted[:, :3] += 0.1
    f2, _ = extract_features(shifted)
    assert np.allclose(f1, f2, atol=1e-5)


def test_scale_invariance():
    rng = np.random.default_rng(2)
    lm = rng.uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    lm[:, 3] = 0.95
    f1, _ = extract_features(lm)
    scaled = lm.copy()
    scaled[:, :3] *= 2.0
    f2, _ = extract_features(scaled)
    assert np.allclose(f1, f2, atol=1e-5)


def test_angle_in_known_range():
    rng = np.random.default_rng(3)
    lm = rng.uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    lm[:, 3] = 0.95
    feats, _ = extract_features(lm)
    assert np.all(feats[:12] >= 0.0) and np.all(feats[:12] <= 180.0)
    assert feats[12] >= 0.0 and feats[13] >= 0.0 and feats[15] >= 0.0
    assert np.isfinite(feats[14])


def test_features_14_and_15_are_distinct():
    rng = np.random.default_rng(4)
    lm = rng.uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    lm[:, 3] = 0.95
    feats, _ = extract_features(lm)
    assert not np.isclose(feats[14], feats[15])
