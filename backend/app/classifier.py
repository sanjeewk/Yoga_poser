import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    low_confidence_reason: Optional[str]


class PoseClassifier:
    def __init__(self, model_path: str, label_encoder_path: str,
                 confidence_threshold: float = 0.45):
        self._model = joblib.load(model_path)
        self._label_encoder = joblib.load(label_encoder_path)
        self._threshold = confidence_threshold

    def predict(self, features: np.ndarray, visibility: np.ndarray) -> PredictionResult:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        classes = self._label_encoder.inverse_transform(np.arange(len(proba)))
        probs = {str(c): float(p) for c, p in zip(classes, proba)}
        top_idx = int(np.argmax(proba))
        confidence = float(proba[top_idx])
        label = str(classes[top_idx])
        reason = None
        if confidence < self._threshold:
            label = "Unknown"
            reason = f"top-1 confidence {confidence:.2f} below threshold {self._threshold}"
        return PredictionResult(label=label, confidence=confidence,
                                probabilities=probs, low_confidence_reason=reason)


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models",
                                  "pose_classifier.joblib")
DEFAULT_LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models",
                                          "label_encoder.pkl")


def load_default() -> Optional[PoseClassifier]:
    if os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(DEFAULT_LABEL_ENCODER_PATH):
        return PoseClassifier(DEFAULT_MODEL_PATH, DEFAULT_LABEL_ENCODER_PATH)
    return None
