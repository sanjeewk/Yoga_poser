from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app


def test_end_to_end_predict_with_browser_landmarks_and_stubbed_classifier():
    client = TestClient(app)
    fake_landmarks = np.random.default_rng(0).uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    fake_landmarks[:, 3] = 0.95
    with patch("backend.app.main._classifier") as clf, \
         patch("backend.app.main._feedback_engine") as fe:
        proba = clf.predict.return_value
        proba.label = "tadasana"
        proba.confidence = 0.9
        proba.probabilities = {"tadasana": 0.9}
        proba.low_confidence_reason = None
        fe.get_feedback.return_value = []
        start = client.post("/api/session/start", json={})
        sid = start.json()["session_id"]
        r = client.post("/api/predict", json={
            "session_id": sid, "landmarks": fake_landmarks.tolist(),
        })
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "tadasana"
    assert body["confidence"] == 0.9
    assert isinstance(body["landmarks"], list)
    assert len(body["landmarks"]) == 33


def test_health_after_lifespan_boot():
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["landmark_runtime"] == "browser"
