import json
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.classifier import load_default
from backend.app.features import extract_features
from backend.app.feedback import FeedbackEngine
from backend.app.pose_estimator import get_estimator
from backend.app.schemas import (
    POSE_CATALOG, HealthResponse, PoseCatalogEntry, PredictionResponse,
    SessionStartRequest, SessionStartResponse, SessionStatusResponse, mediapipe_version,
)
from backend.app.session import SessionStore

_classifier = None
_estimator = None
_feedback_engine = None
_session_store = SessionStore()


def _load_templates():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "pose_templates.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _init_state():
    global _classifier, _estimator, _feedback_engine
    _classifier = load_default()
    _estimator = get_estimator()
    _feedback_engine = FeedbackEngine(_load_templates())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_state()
    yield


app = FastAPI(title="Yoga Poser API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session_store() -> SessionStore:
    return _session_store


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_classifier is not None,
        mediapipe_version=mediapipe_version(),
    )


@app.get("/api/poses")
def poses():
    return {"poses": [p.model_dump() for p in POSE_CATALOG]}


@app.post("/api/session/start", response_model=SessionStartResponse)
def session_start(payload: SessionStartRequest):
    targets = payload.target_poses or [p.key for p in POSE_CATALOG]
    s = _session_store.start(targets)
    return SessionStartResponse(session_id=s.session_id, target_poses=targets)


@app.get("/api/session/{session_id}", response_model=SessionStatusResponse)
def session_status(session_id: str):
    s = _session_store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return s.snapshot()


@app.post("/api/session/{session_id}/reset")
def session_reset(session_id: str):
    s = _session_store.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    s.reset()
    return {"ok": True}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...), session_id: str = Form(...)):
    raw = await image.read()
    landmarks = _estimator.estimate(raw)
    if landmarks is None:
        return PredictionResponse(label="Unknown", confidence=0.0,
                                  landmarks=None, feedback=[],
                                  hold_seconds=0.0, rep_count=0)
    features, visibility = extract_features(landmarks)
    label = "Unknown"
    confidence = 0.0
    feedback = []
    has_major = False
    if _classifier is not None:
        result = _classifier.predict(features, visibility)
        label = result.label
        confidence = result.confidence
        if label != "Unknown" and _feedback_engine is not None:
            hints = _feedback_engine.get_feedback(label, features)
            feedback = hints
            has_major = any(h.severity == "major" for h in hints)
    now = time.time()
    s = get_session_store().get(session_id)
    hold_seconds = 0.0
    rep_count = 0
    if s is not None:
        s.update(label, confidence, has_major, now)
        snap = s.snapshot()
        hold_seconds = snap.hold_seconds
        rep_count = snap.rep_count_per_pose.get(label, 0) if label != "Unknown" else 0
    return PredictionResponse(
        label=label,
        confidence=round(confidence, 4),
        landmarks=landmarks.tolist(),
        feedback=feedback,
        hold_seconds=hold_seconds,
        rep_count=rep_count,
    )


static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
