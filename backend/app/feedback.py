import numpy as np
from backend.app.features import FEATURE_NAMES
from backend.app.schemas import FeedbackHint

EMIT_THRESHOLD = 20.0
CLEAR_THRESHOLD = 10.0

PRIORITY = {
    "spine_arch": 4, "torso_lean": 4,
    "left_hip": 3, "right_hip": 3,
    "left_knee": 3, "right_knee": 3,
    "left_shoulder": 2, "right_shoulder": 2,
    "left_elbow": 1, "right_elbow": 1,
    "left_wrist_shoulder_closure": 1, "right_wrist_shoulder_closure": 1,
}

JOINT_TO_CUE = {
    "left_knee": ("Straighten your front knee", "Bend your front knee more", "major"),
    "right_knee": ("Straighten your back knee", "Bend your back knee more", "major"),
    "left_hip": ("Open your hip more", "Sink deeper into the hip", "major"),
    "right_hip": ("Open your hip more", "Sink deeper into the hip", "major"),
    "left_shoulder": ("Raise your arm higher", "Lower your arm slightly", "minor"),
    "right_shoulder": ("Raise your arm higher", "Lower your arm slightly", "minor"),
    "left_elbow": ("Straighten your arm", "Bend your elbow more", "minor"),
    "right_elbow": ("Straighten your arm", "Bend your elbow more", "minor"),
    "torso_lean": ("Lean further forward", "Stand more upright", "major"),
    "spine_arch": ("Arch your spine more", "Round your spine less", "minor"),
    "left_wrist_shoulder_closure": ("Reach your arm further", "Bring your arm closer", "minor"),
    "right_wrist_shoulder_closure": ("Reach your arm further", "Bring your arm closer", "minor"),
}


class FeedbackEngine:
    def __init__(self, templates: dict):
        self.templates = templates

    def get_feedback(self, pose_key: str, features: np.ndarray) -> list[FeedbackHint]:
        template = self.templates.get(pose_key)
        if template is None:
            return []
        candidates = []
        for name, ideal in template.items():
            if name not in FEATURE_NAMES:
                continue
            idx = FEATURE_NAMES.index(name)
            actual = float(features[idx])
            delta = actual - float(ideal)
            if abs(delta) <= EMIT_THRESHOLD:
                continue
            cue_map = JOINT_TO_CUE.get(name)
            if cue_map is None:
                continue
            too_low, too_high, severity = cue_map
            cue = too_low if delta < 0 else too_high
            candidates.append((PRIORITY.get(name, 1), abs(delta), name, cue, severity))
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        return [FeedbackHint(joint=n, cue=c, severity=s) for _, _, n, c, s in candidates[:2]]
