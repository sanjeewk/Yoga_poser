import numpy as np
from backend.app.feedback import FeedbackEngine


TEMPLATES = {
    "tadasana": {
        "left_knee": 175.0, "right_knee": 175.0,
        "left_elbow": 180.0, "right_elbow": 180.0,
        "torso_lean": 0.0, "spine_arch": 0.0,
    }
}


def test_no_feedback_when_angles_match_template():
    feats = np.zeros(16, dtype=np.float32)
    feats[4] = 175.0  # left_knee
    feats[5] = 175.0  # right_knee
    feats[0] = 180.0  # left_elbow
    feats[1] = 180.0  # right_elbow
    feats[8] = 0.0    # torso_lean
    feats[9] = 0.0    # spine_arch
    eng = FeedbackEngine(TEMPLATES)
    hints = eng.get_feedback("tadasana", feats)
    assert hints == []


def test_emits_major_hint_when_knee_far_from_template():
    feats = np.zeros(16, dtype=np.float32)
    feats[4] = 120.0  # left_knee (template 175, delta 55)
    feats[5] = 120.0  # right_knee
    feats[0] = 180.0
    feats[1] = 180.0
    feats[8] = 0.0
    feats[9] = 0.0
    eng = FeedbackEngine(TEMPLATES)
    hints = eng.get_feedback("tadasana", feats)
    assert len(hints) >= 1
    assert any("knee" in h.joint for h in hints)
    assert all(h.severity in ("minor", "major") for h in hints)


def test_at_most_two_hints_returned():
    feats = np.full(16, 100.0, dtype=np.float32)
    eng = FeedbackEngine(TEMPLATES)
    hints = eng.get_feedback("tadasana", feats)
    assert len(hints) <= 2
