import io
import json
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "mediapipe_version" in body


def test_poses_returns_catalog(client):
    r = client.get("/api/poses")
    assert r.status_code == 200
    body = r.json()
    assert len(body["poses"]) == 8
    keys = {p["key"] for p in body["poses"]}
    assert "tadasana" in keys


def test_session_lifecycle(client):
    r = client.post("/api/session/start", json={})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    r2 = client.get(f"/api/session/{sid}")
    assert r2.status_code == 200
    r3 = client.post(f"/api/session/{sid}/reset")
    assert r3.status_code == 200


def test_predict_without_model_returns_unknown(client):
    with patch("backend.app.main._classifier", None), \
         patch("backend.app.main._estimator") as est, \
         patch("backend.app.main.get_session_store") as gss:
        est.estimate.return_value = np.zeros((33, 4), dtype=np.float32)
        ss = gss.return_value
        ss.get.return_value = None
        files = {"image": ("f.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")}
        data = {"session_id": "nonsense"}
        r = client.post("/api/predict", files=files, data=data)
        assert r.status_code == 200
        body = r.json()
        assert body["label"] == "Unknown"


def test_predict_no_person_returns_unknown(client):
    with patch("backend.app.main._estimator") as est, \
         patch("backend.app.main._classifier", None):
        est.estimate.return_value = None
        files = {"image": ("f.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")}
        data = {"session_id": "nonsense"}
        r = client.post("/api/predict", files=files, data=data)
        assert r.status_code == 200
        assert r.json()["label"] == "Unknown"
