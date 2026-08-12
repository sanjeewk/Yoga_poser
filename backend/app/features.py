import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
NOSE = 0

FEATURE_NAMES = [
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_knee", "right_knee",
    "left_hip", "right_hip",
    "torso_lean", "spine_arch",
    "left_wrist_shoulder_closure", "right_wrist_shoulder_closure",
    "wrist_to_wrist", "ankle_to_ankle", "wrist_height", "hip_wrist_vertical",
]
ANGLE_FEATURE_INDICES = list(range(12))
DISTANCE_FEATURE_INDICES = list(range(12, 16))


def _angle_at_vertex(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _vector_angle(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_features(landmarks):
    lm = np.asarray(landmarks, dtype=np.float32)
    coords = lm[:, :3]
    visibility = lm[:, 3]

    mid_shoulder = (coords[L_SHOULDER] + coords[R_SHOULDER]) / 2.0
    mid_hip = (coords[L_HIP] + coords[R_HIP]) / 2.0
    torso_vec = mid_shoulder - mid_hip
    torso_len = float(np.linalg.norm(torso_vec))

    features = np.zeros(16, dtype=np.float32)
    if torso_len < 1e-6:
        return features, visibility

    angles = [
        _angle_at_vertex(coords[L_SHOULDER], coords[L_ELBOW], coords[L_WRIST]),
        _angle_at_vertex(coords[R_SHOULDER], coords[R_ELBOW], coords[R_WRIST]),
        _angle_at_vertex(coords[L_ELBOW], coords[L_SHOULDER], coords[L_HIP]),
        _angle_at_vertex(coords[R_ELBOW], coords[R_SHOULDER], coords[R_HIP]),
        _angle_at_vertex(coords[L_HIP], coords[L_KNEE], coords[L_ANKLE]),
        _angle_at_vertex(coords[R_HIP], coords[R_KNEE], coords[R_ANKLE]),
        _angle_at_vertex(coords[L_KNEE], coords[L_HIP], coords[L_SHOULDER]),
        _angle_at_vertex(coords[R_KNEE], coords[R_HIP], coords[R_SHOULDER]),
        _vector_angle(torso_vec, np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        _angle_at_vertex(mid_hip, mid_shoulder, coords[NOSE]),
        _angle_at_vertex(coords[L_WRIST], coords[L_SHOULDER], coords[L_ELBOW]),
        _angle_at_vertex(coords[R_WRIST], coords[R_SHOULDER], coords[R_ELBOW]),
    ]
    for i, a in enumerate(angles):
        features[i] = a

    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    features[12] = float(np.linalg.norm(coords[L_WRIST] - coords[R_WRIST])) / torso_len
    features[13] = float(np.linalg.norm(coords[L_ANKLE] - coords[R_ANKLE])) / torso_len
    features[14] = float(np.dot(coords[L_WRIST] + coords[R_WRIST] - 2 * mid_hip, up)) / (2.0 * torso_len)
    features[15] = float(
        (np.linalg.norm(coords[L_WRIST] - mid_hip)
         + np.linalg.norm(coords[R_WRIST] - mid_hip)) / 2.0
    ) / torso_len

    return features, visibility
