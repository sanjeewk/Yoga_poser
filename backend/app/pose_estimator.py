import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    def __init__(self, static_image_mode: bool = False):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, image):
        if isinstance(image, (bytes, bytearray)):
            arr = np.frombuffer(image, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            frame = image
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if result.pose_landmarks is None:
            return None
        out = np.zeros((33, 4), dtype=np.float32)
        for i, p in enumerate(result.pose_landmarks.landmark):
            out[i] = (p.x, p.y, p.z, p.visibility)
        return out


_estimator = None


def get_estimator():
    global _estimator
    if _estimator is None:
        _estimator = PoseEstimator(static_image_mode=True)
    return _estimator
