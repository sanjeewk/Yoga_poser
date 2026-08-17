# Yoga Pose Detector — v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time yoga pose detector — FastAPI backend running MediaPipe Pose + a scikit-learn RandomForest classifier, served to a React/Vite frontend over HTTP, detecting 8 yoga asanas with form-feedback cues, hold-time tracking, and rep counting.

**Architecture:** Browser captures webcam frames at ~8 FPS and POSTs JPEGs to `/api/predict`. The FastAPI backend runs MediaPipe Pose to extract 33 landmarks, computes a 16-dim normalized feature vector (12 joint angles + 4 distances), classifies with a RandomForest, compares against per-pose angle templates for feedback cues, and updates in-memory session state (hold time, reps). An offline training pipeline ingests Yoga-82 + Kaggle datasets, extracts the same features via MediaPipe, and trains the classifier.

**Tech Stack:**
- **Backend:** Python 3.10+, FastAPI, uvicorn, MediaPipe (Pose), OpenCV (headless), scikit-learn, NumPy, Pandas, PyArrow, Pillow, joblib, pydantic v2, pytest, httpx (test client).
- **Frontend:** Node 18+, React 18, Vite 5, Zustand 4, vitest.
- **Shared:** The `features.py` module is imported by both the serving backend and the training pipeline so feature semantics are identical.

## Global Constraints

Copied verbatim from the spec where applicable:

- **Python version:** 3.10+ required (MediaPipe ≥0.10.14, scikit-learn ≥1.4).
- **MediaPipe version:** pin `mediapipe==0.10.14` exactly. Used in **both** training and serving.
- **Frontend:** React 18 + Vite 5 + Zustand 4. Dev server runs on `http://localhost:5173`.
- **Backend:** FastAPI with uvicorn, served on `http://localhost:8000`. CORS allows `localhost:5173` only.
- **Transport:** HTTP POST per frame, throttled to ~8 FPS in browser. No WebSockets.
- **Inference:** All pose inference runs in the Python backend. No browser-side ML.
- **Pose estimator:** MediaPipe Pose, model complexity 1, `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`.
- **Classifier:** scikit-learn `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`.
- **Persistence:** In-memory session state only. No DB.
- **Localization:** English only for v1.
- **Naming:** snake_case for Python files/identifiers; camelCase for JS; React components PascalCase.
- **No comments in code** unless absolutely required for non-obvious math (e.g. a formula citation).
- **Landmark indices (MediaPipe Pose):** `11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow, 15=left_wrist, 16=right_wrist, 23=left_hip, 24=right_hip, 25=left_knee, 26=right_knee, 27=left_ankle, 28=right_ankle, 0=nose`.
- **8 v1 poses (class keys):** `tadasana`, `adho_mukha_svanasana`, `virabhadrasana_i`, `virabhadrasana_ii`, `vrksasana`, `bhujangasana`, `balasana`, `marjaryasana`.
- **Accuracy gate:** top-1 ≥85% target; <80% triggers v2 redesign.

---

## File Structure

```
Yoga_poser/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app + routes
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── pose_estimator.py    # MediaPipe wrapper (singleton)
│   │   ├── features.py          # landmark → 16-dim feature vector (shared)
│   │   ├── classifier.py        # joblib load + predict
│   │   ├── feedback.py          # angle-diff correction rules
│   │   └── session.py           # hold-time, rep counter, state
│   ├── models/
│   │   ├── pose_classifier.joblib
│   │   ├── label_encoder.pkl
│   │   └── pose_templates.json
│   ├── static/
│   │   └── poses/<key>.jpg
│   ├── training/
│   │   ├── __init__.py
│   │   ├── download_data.py
│   │   ├── extract_features.py
│   │   ├── train.py
│   │   └── build_templates.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_features.py
│   │   ├── test_pose_estimator.py
│   │   ├── test_classifier.py
│   │   ├── test_feedback.py
│   │   ├── test_session.py
│   │   └── test_main.py
│   ├── pyproject.toml
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── api.js
│   │   ├── store.js
│   │   ├── components/
│   │   │   ├── WebcamView.jsx
│   │   │   ├── PoseOverlay.jsx
│   │   │   ├── FeedbackBanner.jsx
│   │   │   ├── SessionStats.jsx
│   │   │   └── PosePicker.jsx
│   │   └── hooks/
│   │       └── usePoseDetection.js
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── .eslintrc.json
├── scripts/
│   └── dev.sh
├── data/                        # gitignored
├── docs/superpowers/specs/2026-08-12-yoga-pose-detector-design.md
├── docs/superpowers/plans/2026-08-12-yoga-pose-detector.md
├── .gitignore
└── README.md
```

---

## Task Index

1. Repo scaffolding (Python backend + React frontend skeletons, deps, .gitignore, dev script).
2. `features.py` — landmark → 16-dim feature vector (pure functions).
3. `pose_estimator.py` — MediaPipe wrapper.
4. `schemas.py` — Pydantic models for API contracts.
5. `feedback.py` — angle-diff correction rules.
6. `session.py` — in-memory session state (hold time, reps).
7. `classifier.py` — joblib load + predict wrapper.
8. `main.py` — FastAPI app and endpoints.
9. Training pipeline: `download_data.py` (Yoga-82 + Kaggle fetch + relabel).
10. Training pipeline: `extract_features.py` (batch MediaPipe over images → parquet).
11. Training pipeline: `train.py` (RF + baselines, eval, serialize).
12. Training pipeline: `build_templates.py` (per-pose median angles → JSON).
13. Frontend: Vite + React + Zustand scaffolding and `api.js`.
14. Frontend: `usePoseDetection` hook with frame loop.
15. Frontend: `WebcamView` + `PoseOverlay`.
16. Frontend: `FeedbackBanner` + `SessionStats` + `PosePicker`.
17. Frontend: `App.jsx` composition and styling.
18. Integration: dev script smoke test, README, final commit.

---

## Task 1: Repo Scaffolding

**Files:**
- Create: `backend/pyproject.toml`, `backend/requirements.txt`, `backend/app/__init__.py`, `backend/training/__init__.py`, `backend/tests/__init__.py`, `backend/tests/conftest.py`
- Create: `.gitignore`, `README.md`, `scripts/dev.sh`
- Create: `frontend/package.json`, `frontend/vite.config.js`, `frontend/index.html`, `frontend/src/main.jsx`, `frontend/src/App.jsx`, `frontend/.eslintrc.json`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: directory tree, installable backend venv, installable frontend deps, runnable `scripts/dev.sh`.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Data and model artifacts (keep committed model files in backend/models/)
data/
backend/models/*.tmp

# Node
node_modules/
frontend/dist/
frontend/.vite/

# Editor
.vscode/
.idea/
*.swp
.DS_Store

# Env
.env
.env.local
```

- [ ] **Step 2: Create `backend/requirements.txt`**

```
mediapipe==0.10.14
opencv-python-headless==4.9.0.80
scikit-learn==1.4.2
numpy==1.26.4
pandas==2.2.2
pyarrow==15.0.2
Pillow==10.3.0
joblib==1.4.2
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
python-multipart==0.0.9
httpx==0.27.0
pytest==8.2.0
```

- [ ] **Step 3: Create `backend/pyproject.toml`**

```toml
[project]
name = "yoga-poser-backend"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v"
```

- [ ] **Step 4: Create empty `__init__.py` files**

Create empty files at:
- `backend/app/__init__.py`
- `backend/training/__init__.py`
- `backend/tests/__init__.py`

- [ ] **Step 5: Create `backend/tests/conftest.py` with shared fixtures**

```python
import numpy as np
import pytest


@pytest.fixture
def synthetic_landmarks():
    np.random.seed(42)
    arr = np.random.default_rng(0).uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    arr[:, 3] = 0.95
    return arr
```

- [ ] **Step 6: Set up Python venv and install**

Run:
```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

This repo has a `.python-version` file pinning Python 3.10.14 via pyenv, so `python3` resolves to 3.10+ automatically. If for some reason `python3 --version` reports < 3.10, install 3.10 via pyenv (`pyenv install 3.10.14 && pyenv local 3.10.14`) before running the commands above.

- [ ] **Step 7: Create `frontend/package.json`**

```json
{
  "name": "yoga-poser-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "lint": "eslint src"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "zustand": "^4.5.2"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.1",
    "vite": "^5.2.10",
    "vitest": "^1.5.0",
    "eslint": "^8.57.0",
    "eslint-plugin-react": "^7.34.1"
  }
}
```

- [ ] **Step 8: Create `frontend/vite.config.js`**

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
```

- [ ] **Step 9: Create `frontend/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Yoga Poser</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 10: Create `frontend/src/main.jsx` and `frontend/src/App.jsx` placeholders**

`frontend/src/main.jsx`:
```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

`frontend/src/App.jsx` (placeholder, replaced in Task 17):
```jsx
export default function App() {
  return <div>Yoga Poser — bootstrapping…</div>
}
```

- [ ] **Step 11: Create `frontend/.eslintrc.json`**

```json
{
  "env": { "browser": true, "es2021": true },
  "extends": ["eslint:recommended", "plugin:react/recommended"],
  "parserOptions": { "ecmaVersion": "latest", "sourceType": "module" },
  "settings": { "react": { "version": "detect" } }
}
```

- [ ] **Step 12: Install frontend deps**

Run:
```bash
cd frontend && npm install
```

If `node`/`npm` are missing, install Node 18 LTS first (e.g. via `nvm install 18 && nvm use 18`).

- [ ] **Step 13: Create `scripts/dev.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
trap 'kill 0' EXIT

(cd backend && . .venv/bin/activate && uvicorn app.main:app --reload --port 8000) &
BACK_PID=$!
(cd frontend && npm run dev) &
FRONT_PID=$!

wait
```

Then `chmod +x scripts/dev.sh`.

- [ ] **Step 14: Create skeleton `README.md`**

```markdown
# Yoga Poser

Real-time yoga pose detection. FastAPI + MediaPipe + scikit-learn backend, React + Vite frontend.

See `docs/superpowers/specs/2026-08-12-yoga-pose-detector-design.md` for design.

## Quick start

1. Python 3.10+ and Node 18+ required.
2. Backend: `python3 -m venv backend/.venv && source backend/.venv/bin/activate && pip install -r backend/requirements.txt`
3. Frontend: `cd frontend && npm install`
4. Train: see `backend/training/` README.
5. Run: `./scripts/dev.sh` → open http://localhost:5173
```

- [ ] **Step 15: Verify both stacks boot**

```bash
source backend/.venv/bin/activate
python -c "import fastapi, mediapipe, sklearn, numpy, cv2; print('backend ok')"
cd frontend && npm run build && cd ..
```

Expected: `backend ok` printed; frontend builds without error.

- [ ] **Step 16: Commit**

```bash
git add .gitignore README.md scripts/ backend/pyproject.toml backend/requirements.txt backend/app/__init__.py backend/training/__init__.py backend/tests/__init__.py backend/tests/conftest.py frontend/package.json frontend/vite.config.js frontend/index.html frontend/src/main.jsx frontend/src/App.jsx frontend/.eslintrc.json
git commit -m "chore: scaffold backend and frontend project skeletons"
```

---

## Task 2: Feature Extraction Module (`features.py`)

**Files:**
- Create: `backend/app/features.py`
- Test: `backend/tests/test_features.py`

**Interfaces:**
- Consumes: NumPy. Landmark array shape `(33, 4)` where columns are `x, y, z, visibility` in MediaPipe's normalized image space (x, y in [0,1]; z relative depth; visibility in [0,1]).
- Produces:
  - `extract_features(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — returns `(features(16,), visibility(33,))`.
  - `FEATURE_NAMES: list[str]` — length 16, ordered list of feature names.
  - `ANGLE_FEATURE_INDICES = list(range(12))` and `DISTANCE_FEATURE_INDICES = list(range(12, 16))` for augmentation.

**Math reference (for the implementer):**

Landmark indices used:
- `L_SHOULDER=11, R_SHOULDER=12, L_ELBOW=13, R_ELBOW=14, L_WRIST=15, R_WRIST=16, L_HIP=23, R_HIP=24, L_KNEE=25, R_KNEE=26, L_ANKLE=27, R_ANKLE=28, NOSE=0`.
- Mid-shoulder = avg(11, 12). Mid-hip = avg(23, 24).

Helpers:
- `angle_at_vertex(p1, p2, p3)` — angle in degrees at `p2` formed by rays to `p1` and `p3`. Use `arccos(dot(v1, v2) / (|v1||v2|))` clipped to [0, 180].
- `vector_angle(v1, v2)` — angle in degrees between two vectors.

Torso length (normalization denominator): Euclidean distance from mid-hip to mid-shoulder in 3D (x, y, z). If torso length < 1e-6, return zero features.

- [ ] **Step 1: Write the failing test for shape and invariance**

`backend/tests/test_features.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && . .venv/bin/activate && pytest tests/test_features.py -v
```

Expected: FAIL with `ModuleNotFoundError: backend.app.features`.

- [ ] **Step 3: Implement `backend/app/features.py`**

```python
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
    features[14] = float(np.dot((coords[L_WRIST] + coords[R_WRIST]) / 2.0 - mid_hip, up)) / torso_len
    features[15] = float(
        (np.linalg.norm(coords[L_WRIST] - mid_hip)
         + np.linalg.norm(coords[R_WRIST] - mid_hip)) / 2.0
    ) / torso_len

    return features, visibility
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_features.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/features.py backend/tests/test_features.py
git commit -m "feat(features): add landmark-to-16dim feature vector extractor"
```

---

## Task 3: MediaPipe Wrapper (`pose_estimator.py`)

**Files:**
- Create: `backend/app/pose_estimator.py`
- Test: `backend/tests/test_pose_estimator.py`

**Interfaces:**
- Consumes: `mediapipe`, `numpy`, `cv2`.
- Produces:
  - `class PoseEstimator` with `__init__()` configuring MediaPipe Pose and `estimate(image: np.ndarray | bytes) -> np.ndarray | None` returning landmarks shape `(33, 4)` or `None`.
  - Module-level singleton accessor `get_estimator() -> PoseEstimator` (lazy-initialized).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_pose_estimator.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_pose_estimator.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/app/pose_estimator.py`**

```python
import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
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
        _estimator = PoseEstimator()
    return _estimator
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_pose_estimator.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pose_estimator.py backend/tests/test_pose_estimator.py
git commit -m "feat(pose): add MediaPipe Pose singleton wrapper"
```

---

## Task 4: API Schemas (`schemas.py`)

**Files:**
- Create: `backend/app/schemas.py`
- Test: `backend/tests/test_schemas.py`

**Interfaces:**
- Produces Pydantic v2 models used by `main.py`, `classifier.py`, `feedback.py`, `session.py`:
  - `PoseCatalogEntry(key, display_name, sanskrit, description)`
  - `FeedbackHint(joint, cue, severity)` where severity ∈ `{"minor", "major"}`
  - `PredictionResponse(label, confidence, landmarks, feedback: list[FeedbackHint], hold_seconds, rep_count)`
  - `SessionStartRequest(target_poses: list[str] | None = None)`
  - `SessionStartResponse(session_id, target_poses)`
  - `SessionStatusResponse(current_pose, hold_seconds, rep_count_per_pose: dict[str,int], history: list[dict])`
  - `HealthResponse(status, model_loaded, mediapipe_version)`
  - `POSE_CATALOG: list[PoseCatalogEntry]` — the 8 v1 poses with display names and short descriptions.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_schemas.py`:
```python
from backend.app.schemas import (
    POSE_CATALOG, PredictionResponse, FeedbackHint, SessionStartRequest,
)


def test_catalog_has_8_poses():
    keys = {p.key for p in POSE_CATALOG}
    assert keys == {
        "tadasana", "adho_mukha_svanasana", "virabhadrasana_i",
        "virabhadrasana_ii", "vrksasana", "bhujangasana",
        "balasana", "marjaryasana",
    }


def test_prediction_response_optional_fields():
    r = PredictionResponse(label="tadasana", confidence=0.9, landmarks=None,
                           feedback=[], hold_seconds=0.0, rep_count=0)
    assert r.label == "tadasana"
    assert r.feedback == []


def test_feedback_hint_severity_validates():
    h = FeedbackHint(joint="left_knee", cue="Straighten your front knee", severity="major")
    assert h.severity == "major"


def test_session_start_request_default():
    r = SessionStartRequest()
    assert r.target_poses is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_schemas.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/app/schemas.py`**

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field
import mediapipe as mp


class PoseCatalogEntry(BaseModel):
    key: str
    display_name: str
    sanskrit: str
    description: str


POSE_CATALOG: list[PoseCatalogEntry] = [
    PoseCatalogEntry(key="tadasana", display_name="Mountain", sanskrit="Tadasana",
                     description="Neutral standing baseline; feet together, arms at sides."),
    PoseCatalogEntry(key="adho_mukha_svanasana", display_name="Downward-Facing Dog",
                     sanskrit="Adho Mukha Svanasana",
                     description="Inverted V; hands and feet on floor, hips lifted."),
    PoseCatalogEntry(key="virabhadrasana_i", display_name="Warrior I",
                     sanskrit="Virabhadrasana I",
                     description="Deep lunge, back foot angled, arms raised overhead."),
    PoseCatalogEntry(key="virabhadrasana_ii", display_name="Warrior II",
                     sanskrit="Virabhadrasana II",
                     description="Deep lunge, arms extended horizontally, gaze forward."),
    PoseCatalogEntry(key="vrksasana", display_name="Tree", sanskrit="Vrksasana",
                     description="Standing balance; one foot on inner thigh, hands in prayer."),
    PoseCatalogEntry(key="bhujangasana", display_name="Cobra", sanskrit="Bhujangasana",
                     description="Prone backbend; chest lifted, arms supporting."),
    PoseCatalogEntry(key="balasana", display_name="Child's Pose", sanskrit="Balasana",
                     description="Kneeling fold; hips to heels, arms forward or beside."),
    PoseCatalogEntry(key="marjaryasana", display_name="Cat", sanskrit="Marjaryasana",
                     description="Tabletop with spine rounded upward."),
]


class FeedbackHint(BaseModel):
    joint: str
    cue: str
    severity: Literal["minor", "major"]


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    landmarks: Optional[list[list[float]]] = None
    feedback: list[FeedbackHint] = Field(default_factory=list)
    hold_seconds: float = 0.0
    rep_count: int = 0


class SessionStartRequest(BaseModel):
    target_poses: Optional[list[str]] = None


class SessionStartResponse(BaseModel):
    session_id: str
    target_poses: list[str]


class SessionStatusResponse(BaseModel):
    current_pose: Optional[str]
    hold_seconds: float
    rep_count_per_pose: dict[str, int]
    history: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mediapipe_version: str

    model_config = {"protected_namespaces": ()}


def mediapipe_version() -> str:
    return mp.__version__
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_schemas.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas.py backend/tests/test_schemas.py
git commit -m "feat(schemas): add Pydantic API models and 8-pose catalog"
```

---

## Task 5: Feedback Engine (`feedback.py`)

**Files:**
- Create: `backend/app/feedback.py`
- Test: `backend/tests/test_feedback.py`
- Create: `backend/models/pose_templates.json` (test fixture version; real one built in Task 12).

**Interfaces:**
- Consumes: `features.py` (`FEATURE_NAMES`, angle indices), a templates dict `{pose_key: {feature_name: ideal_angle}}`, the 16-dim feature vector.
- Produces:
  - `class FeedbackEngine` with `__init__(templates: dict)` and `get_feedback(pose_key: str, features: np.ndarray) -> list[FeedbackHint]`.
  - `JOINT_TO_CUE: dict[str, tuple[str, str, str]]` mapping feature name → `(too_low_cue, too_high_cue, severity)`.
  - Threshold: emit a hint when `|delta| > 20°`. Band-clear at `|delta| ≤ 10°`.
  - Priority weighting: spine=4, hip=3, knee=3, shoulder=2, elbow=1, wrist=1.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_feedback.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_feedback.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/app/feedback.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_feedback.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/feedback.py backend/tests/test_feedback.py
git commit -m "feat(feedback): add angle-diff correction engine"
```

---

## Task 6: Session State (`session.py`)

**Files:**
- Create: `backend/app/session.py`
- Test: `backend/tests/test_session.py`

**Interfaces:**
- Produces:
  - `class SessionState` with methods:
    - `__init__(session_id, target_poses)`
    - `update(label: str, confidence: float, has_major_feedback: bool, now_ts: float) -> None`
    - `snapshot() -> SessionStatusResponse`
    - `reset() -> None`
  - `class SessionStore` with `start(target_poses) -> SessionState`, `get(session_id) -> SessionState | None`, `evict_idle(now_ts, max_age_seconds=1800)`.
  - Hold accrual rule: top-1 label stable ≥1 s, confidence ≥0.6, no `major` feedback.
  - Rep rule: held ≥3 s then exited (label change ≥0.5 s or confidence <0.4 ≥0.5 s). 0.5 s cooldown.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_session.py`:
```python
from backend.app.session import SessionState, SessionStore


def test_hold_accrues_when_stable_and_confident():
    s = SessionState("s1", ["tadasana"])
    for t in range(0, 30):
        s.update("tadasana", 0.9, False, float(t) / 10)
    snap = s.snapshot()
    assert snap.hold_seconds >= 1.5
    assert snap.current_pose == "tadasana"


def test_hold_does_not_accrue_with_major_feedback():
    s = SessionState("s1", ["tadasana"])
    for t in range(0, 30):
        s.update("tadasana", 0.9, True, float(t) / 10)
    snap = s.snapshot()
    assert snap.hold_seconds == 0.0


def test_rep_counted_after_three_second_hold_then_exit():
    s = SessionState("s1", ["tadasana"])
    for t in range(0, 40):
        s.update("tadasana", 0.9, False, float(t) / 10)
    for t in range(40, 50):
        s.update("unknown", 0.2, False, float(t) / 10)
    snap = s.snapshot()
    assert snap.rep_count_per_pose.get("tadasana", 0) == 1


def test_store_start_and_get():
    store = SessionStore()
    s = store.start(["tadasana"])
    assert store.get(s.session_id) is s


def test_store_evicts_idle():
    import time
    store = SessionStore()
    s = store.start(["tadasana"])
    future = time.time() + 100
    store.evict_idle(now_ts=future, max_age_seconds=10)
    assert store.get(s.session_id) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_session.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/app/session.py`**

```python
import time
import uuid
from collections import deque

from backend.app.schemas import SessionStatusResponse

HOLD_MIN_CONFIDENCE = 0.6
HOLD_STABILITY_WINDOW = 1.0
REP_MIN_HOLD_SECONDS = 3.0
REP_EXIT_DURATION = 0.5
REP_COOLDOWN = 0.5


class SessionState:
    def __init__(self, session_id: str, target_poses: list[str]):
        self.session_id = session_id
        self.target_poses = target_poses
        self._current_pose: str | None = None
        self._current_since_ts: float = 0.0
        self._hold_seconds: float = 0.0
        self._last_update_ts: float = 0.0
        self._label_history: deque[tuple[float, str]] = deque(maxlen=60)
        self._rep_count_per_pose: dict[str, int] = {}
        self._history: list[dict] = []
        self._rep_cooldown_until: float = 0.0

    def update(self, label: str, confidence: float, has_major_feedback: bool, now_ts: float):
        prev_ts = self._last_update_ts
        self._last_update_ts = now_ts
        self._label_history.append((now_ts, label))

        if label != self._current_pose:
            self._on_pose_exit(now_ts)
            self._current_pose = label
            self._current_since_ts = now_ts
            self._hold_seconds = 0.0
            return

        stable = self._is_stable(label, now_ts)
        if (stable and confidence >= HOLD_MIN_CONFIDENCE and not has_major_feedback):
            self._hold_seconds += max(0.0, now_ts - prev_ts)
        self._current_pose = label

    def _is_stable(self, label: str, now_ts: float) -> bool:
        window_start = now_ts - HOLD_STABILITY_WINDOW
        recent = [(t, l) for t, l in self._label_history if t >= window_start]
        return bool(recent) and all(l == label for _, l in recent)

    def _on_pose_exit(self, now_ts: float):
        if self._current_pose is None or self._current_pose == "unknown":
            return
        if self._hold_seconds >= REP_MIN_HOLD_SECONDS and now_ts >= self._rep_cooldown_until:
            self._rep_count_per_pose[self._current_pose] = (
                self._rep_count_per_pose.get(self._current_pose, 0) + 1
            )
            self._history.append({
                "pose": self._current_pose,
                "held_seconds": round(self._hold_seconds, 2),
                "ended_at": now_ts,
            })
            self._rep_cooldown_until = now_ts + REP_COOLDOWN

    def snapshot(self) -> SessionStatusResponse:
        return SessionStatusResponse(
            current_pose=self._current_pose,
            hold_seconds=round(self._hold_seconds, 2),
            rep_count_per_pose=dict(self._rep_count_per_pose),
            history=list(self._history),
        )

    def reset(self):
        self._current_pose = None
        self._current_since_ts = 0.0
        self._hold_seconds = 0.0
        self._label_history.clear()
        self._rep_count_per_pose.clear()
        self._history.clear()
        self._rep_cooldown_until = 0.0


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._last_active: dict[str, float] = {}

    def start(self, target_poses: list[str]) -> SessionState:
        sid = uuid.uuid4().hex
        s = SessionState(sid, target_poses)
        self._sessions[sid] = s
        self._last_active[sid] = time.time()
        return s

    def get(self, session_id: str) -> SessionState | None:
        s = self._sessions.get(session_id)
        if s is not None:
            self._last_active[session_id] = time.time()
        return s

    def evict_idle(self, now_ts: float, max_age_seconds: float = 1800):
        stale = [sid for sid, t in self._last_active.items() if now_ts - t > max_age_seconds]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._last_active.pop(sid, None)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_session.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/session.py backend/tests/test_session.py
git commit -m "feat(session): add in-memory session state with hold time and rep counting"
```

---

## Task 7: Classifier Wrapper (`classifier.py`)

**Files:**
- Create: `backend/app/classifier.py`
- Test: `backend/tests/test_classifier.py`
- Create: `backend/models/.gitkeep`

**Interfaces:**
- Produces:
  - `class PoseClassifier` with `__init__(model_path, label_encoder_path, confidence_threshold=0.45)` and `predict(features: np.ndarray, visibility: np.ndarray) -> PredictionResult`.
  - `PredictionResult` dataclass: `label: str, confidence: float, probabilities: dict[str, float], low_confidence_reason: str | None`.
  - `load_default() -> PoseClassifier | None` — returns `None` if artifacts missing (so server can boot before training).

- [ ] **Step 1: Write the failing test using a fake model**

`backend/tests/test_classifier.py`:
```python
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from backend.app.classifier import PoseClassifier


def _train_tiny(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 180, size=(40, 16)).astype(np.float32)
    y = np.array(["tadasana"] * 20 + ["vrksasana"] * 20)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y_enc)
    model_path = tmp_path / "m.joblib"
    le_path = tmp_path / "le.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(le, le_path)
    return str(model_path), str(le_path)


def test_predict_returns_label_and_confidence(tmp_path):
    mp, lp = _train_tiny(tmp_path)
    clf = PoseClassifier(mp, lp, confidence_threshold=0.4)
    feats = np.full(16, 90.0, dtype=np.float32)
    vis = np.full(33, 0.9, dtype=np.float32)
    result = clf.predict(feats, vis)
    assert result.label in {"tadasana", "vrksasana", "Unknown"}
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.probabilities, dict)


def test_predict_marks_low_confidence_as_unknown(tmp_path):
    mp, lp = _train_tiny(tmp_path)
    clf = PoseClassifier(mp, lp, confidence_threshold=0.99)
    feats = np.full(16, 30.0, dtype=np.float32)
    vis = np.full(33, 0.9, dtype=np.float32)
    result = clf.predict(feats, vis)
    assert result.label == "Unknown"
    assert result.low_confidence_reason is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_classifier.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/app/classifier.py`**

```python
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
```

Create empty marker file `backend/models/.gitkeep` so the directory exists in git.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_classifier.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/classifier.py backend/tests/test_classifier.py backend/models/.gitkeep
git commit -m "feat(classifier): add joblib-backed RandomForest wrapper with low-confidence fallback"
```

---

## Task 8: FastAPI App (`main.py`)

**Files:**
- Create: `backend/app/main.py`
- Test: `backend/tests/test_main.py`

**Interfaces:**
- Consumes: `pose_estimator.get_estimator()`, `features.extract_features`, `classifier.load_default`, `feedback.FeedbackEngine`, `session.SessionStore`, `schemas.*`.
- Produces a FastAPI `app` object exposed at `backend.app.main:app`.
- Endpoints: `/api/health`, `/api/poses`, `/api/session/start`, `/api/session/{id}`, `/api/session/{id}/reset`, `/api/predict`. CORS allows `http://localhost:5173`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_main.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_main.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/app/main.py`**

```python
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
    s = _session_store.get(session_id)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_main.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Manual smoke boot**

```bash
cd backend && . .venv/bin/activate && uvicorn app.main:app --reload --port 8000 &
sleep 3
curl -s http://localhost:8000/api/health
curl -s http://localhost:8000/api/poses
kill %1
```

Expected: JSON responses, no 500s.

- [ ] **Step 6: Commit**

```bash
git add backend/app/main.py backend/tests/test_main.py
git commit -m "feat(api): add FastAPI app with predict, session, health, poses endpoints"
```

---

## Task 9: Training Data Download (`training/download_data.py`)

**Files:**
- Create: `backend/training/download_data.py`
- Create: `backend/training/README.md`
- Test: `backend/tests/test_download_data.py`

**Interfaces:**
- Produces:
  - `SYNONYM_MAP: dict[str, str]` mapping dataset label variants to canonical v1 keys.
  - `relabel(label: str) -> str | None` — returns canonical key or `None` if not in v1 set.
  - `find_images(raw_dir: Path) -> list[tuple[Path, str]]` — walks `data/raw/` and yields `(image_path, canonical_label)` for matching images only.

Yoga-82 download: `https://github.com/manish7suthar/Yoga-82-dataset` (metadata CSVs only — image URLs may need re-fetching). Document Kaggle manual steps.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_download_data.py`:

```python
from pathlib import Path
from backend.training.download_data import relabel, SYNONYM_MAP, find_images


def test_relabel_known_synonyms():
    assert relabel("Adho Mukha Svanasana") == "adho_mukha_svanasana"
    assert relabel("Downward-Facing Dog") == "adho_mukha_svanasana"
    assert relabel("Downward Dog") == "adho_mukha_svanasana"
    assert relabel("Warrior I") == "virabhadrasana_i"
    assert relabel("Warrior II") == "virabhadrasana_ii"
    assert relabel("Child's Pose") == "balasana"
    assert relabel("Balasana") == "balasana"


def test_relabel_unknown_returns_none():
    assert relabel("Sirsasana") is None
    assert relabel("") is None


def test_find_images_filters_to_canonical(tmp_path):
    raw = tmp_path / "raw"
    (raw / "Adho Mukha Svanasana").mkdir(parents=True)
    (raw / "Adho Mukha Svanasana" / "a.jpg").write_bytes(b"x")
    (raw / "Sirsasana").mkdir(parents=True)
    (raw / "Sirsasana" / "b.jpg").write_bytes(b"x")
    pairs = list(find_images(tmp_path / "raw"))
    labels = {label for _, label in pairs}
    assert labels == {"adho_mukha_svanasana"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_download_data.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/training/download_data.py`**

```python
import os
from pathlib import Path

CANONICAL_KEYS = [
    "tadasana", "adho_mukha_svanasana", "virabhadrasana_i", "virabhadrasana_ii",
    "vrksasana", "bhujangasana", "balasana", "marjaryasana",
]

SYNONYM_MAP = {
    "tadasana": "tadasana",
    "mountain": "tadasana",
    "mountain pose": "tadasana",
    "adho mukha svanasana": "adho_mukha_svanasana",
    "downward-facing dog": "adho_mukha_svanasana",
    "downward dog": "adho_mukha_svanasana",
    "down dog": "adho_mukha_svanasana",
    "virabhadrasana i": "virabhadrasana_i",
    "virabhadrasana 1": "virabhadrasana_i",
    "warrior i": "virabhadrasana_i",
    "warrior 1": "virabhadrasana_i",
    "virabhadrasana ii": "virabhadrasana_ii",
    "virabhadrasana 2": "virabhadrasana_ii",
    "warrior ii": "virabhadrasana_ii",
    "warrior 2": "virabhadrasana_ii",
    "vrksasana": "vrksasana",
    "tree": "vrksasana",
    "tree pose": "vrksasana",
    "bhujangasana": "bhujangasana",
    "cobra": "bhujangasana",
    "cobra pose": "bhujangasana",
    "balasana": "balasana",
    "child's pose": "balasana",
    "child pose": "balasana",
    "marjaryasana": "marjaryasana",
    "cat": "marjaryasana",
    "cat pose": "marjaryasana",
}


def relabel(label: str) -> str | None:
    if not label:
        return None
    key = label.strip().lower()
    return SYNONYM_MAP.get(key)


def find_images(raw_dir):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label_source = path.parent.name
        canonical = relabel(label_source)
        if canonical is None:
            continue
        yield (path, canonical)


YOGA82_META_URLS = [
    "https://raw.githubusercontent.com/manish7suthar/Yoga-82-dataset/master/Yoga-82/yoga_dataset_links/3personWarriorII.txt",
]


def main(raw_dir: str = "data/raw"):
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Manual step: download Yoga-82 from https://github.com/manish7suthar/Yoga-82-dataset")
    print(f"Manual step: download Kaggle 'Yoga Posture Dataset' via `kaggle datasets download -d shrutisaxena/yoga-pose-image-classification-dataset`")
    print(f"Place extracted folders under {raw_dir}/. Expected structure: <raw_dir>/<Pose Label>/<image>.jpg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `backend/training/README.md` with manual download instructions**

```markdown
# Training pipeline

## 1. Acquire data

### Yoga-82
1. Clone: `git clone https://github.com/manish7suthar/Yoga-82-dataset data/yoga-82`
2. Use the metadata files to download the subset of images for the 8 v1 poses.

### Kaggle Yoga Posture Dataset
1. Install: `pip install kaggle`
2. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json`.
3. Run: `kaggle datasets download -d shrutisaxena/yoga-pose-image-classification-dataset -p data/kaggle --unzip`

## 2. Consolidate
Place all per-pose image folders under `data/raw/<Pose Label>/`. Folder names
must match a synonym in `download_data.SYNONYM_MAP`. Examples of valid folder
names: `Tadasana`, `Downward Dog`, `Warrior I`.

## 3. Run the pipeline
```bash
python -m backend.training.extract_features   # → data/features.parquet
python -m backend.training.train              # → backend/models/{pose_classifier.joblib,label_encoder.pkl}
python -m backend.training.build_templates    # → backend/models/pose_templates.json
```
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_download_data.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/training/download_data.py backend/training/README.md backend/tests/test_download_data.py
git commit -m "feat(training): add dataset download/relabel module and README"
```

---

## Task 10: Feature Extraction Pipeline (`training/extract_features.py`)

**Files:**
- Create: `backend/training/extract_features.py`
- Test: `backend/tests/test_extract_features_pipeline.py`

**Interfaces:**
- Consumes: `pose_estimator.PoseEstimator`, `features.extract_features`, `download_data.find_images`.
- Produces:
  - `process_images(image_paths: Iterable[(Path, str)], min_torso_visibility: float = 0.5) -> pd.DataFrame` — columns: `path, label, f0..f15, v0..v32`.
  - `main(raw_dir, out_path)` CLI entry.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_extract_features_pipeline.py`:
```python
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from backend.training.extract_features import process_images


def _fake_landmarks():
    rng = np.random.default_rng(0)
    lm = rng.uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    lm[11:18, 3] = 0.95
    lm[23:30, 3] = 0.95
    return lm


def _write_tiny_image(path):
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))


def test_process_images_filters_low_visibility(tmp_path):
    _write_tiny_image(tmp_path / "a.jpg")
    _write_tiny_image(tmp_path / "b.jpg")
    paths = [(tmp_path / "a.jpg", "tadasana"), (tmp_path / "b.jpg", "vrksasana")]
    with patch("backend.training.extract_features.PoseEstimator") as Est:
        est = Est.return_value
        est.estimate.side_effect = [_fake_landmarks(), None]
        df = process_images(paths)
    assert len(df) == 1
    assert df.iloc[0]["label"] == "tadasana"
    assert all(f"f{i}" in df.columns for i in range(16))
    assert all(f"v{i}" in df.columns for i in range(33))


def test_process_images_handles_empty():
    with patch("backend.training.extract_features.PoseEstimator"):
        df = process_images([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_extract_features_pipeline.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/training/extract_features.py`**

```python
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from backend.app.features import extract_features
from backend.app.pose_estimator import PoseEstimator
from backend.training.download_data import find_images

TORSO_KEYPOINT_INDICES = [11, 12, 23, 24]


def process_images(image_paths: Iterable[tuple[Path, str]],
                   min_torso_visibility: float = 0.5) -> pd.DataFrame:
    estimator = PoseEstimator()
    rows = []
    for path, label in image_paths:
        try:
            image = _read_image(str(path))
        except Exception:
            continue
        if image is None:
            continue
        landmarks = estimator.estimate(image)
        if landmarks is None:
            continue
        if float(np.min(landmarks[TORSO_KEYPOINT_INDICES, 3])) < min_torso_visibility:
            continue
        feats, vis = extract_features(landmarks)
        row = {"path": str(path), "label": label}
        for i in range(16):
            row[f"f{i}"] = float(feats[i])
        for i in range(33):
            row[f"v{i}"] = float(vis[i])
        rows.append(row)
    return pd.DataFrame(rows)


def _read_image(path: str):
    import cv2
    return cv2.imread(path)


def main(raw_dir: str = "data/raw", out_path: str = "data/features.parquet"):
    pairs = list(find_images(raw_dir))
    print(f"Found {len(pairs)} candidate images")
    df = process_images(pairs)
    print(f"Extracted features for {len(df)} samples")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/features.parquet")
    args = p.parse_args()
    main(args.raw_dir, args.out)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_extract_features_pipeline.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/training/extract_features.py backend/tests/test_extract_features_pipeline.py
git commit -m "feat(training): batch MediaPipe feature extraction over dataset images"
```

---

## Task 11: Training Script (`training/train.py`)

**Files:**
- Create: `backend/training/train.py`
- Test: `backend/tests/test_train.py`

**Interfaces:**
- Consumes: `data/features.parquet` (produced by Task 10).
- Produces `backend/models/pose_classifier.joblib`, `backend/models/label_encoder.pkl`, and prints accuracy report.
- Functions:
  - `augment(X: np.ndarray, n_variants: int = 3, angle_jitter: float = 5.0, dist_jitter: float = 0.05, seed: int = 42) -> np.ndarray`
  - `balance_classes(X, y, seed) -> (X, y)` (random undersample)
  - `train(df: pd.DataFrame, out_dir: Path) -> dict` returning metrics dict.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_train.py`:
```python
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from backend.training.train import augment, balance_classes, train


def _synthetic_df():
    rng = np.random.default_rng(0)
    rows = []
    for label in ["tadasana", "vrksasana"]:
        for _ in range(30):
            row = {"label": label, "path": "x"}
            for i in range(16):
                row[f"f{i}"] = float(rng.uniform(0, 180))
            for i in range(33):
                row[f"v{i}"] = 0.9
            rows.append(row)
    return pd.DataFrame(rows)


def test_augment_increases_rows_and_keeps_columns():
    X = np.full((10, 16), 90.0, dtype=np.float32)
    out = augment(X, n_variants=2)
    assert out.shape[0] == 30
    assert out.shape[1] == 16


def test_balance_classes_undersamples_to_min():
    X = np.zeros((50, 16))
    y = np.array(["a"] * 30 + ["b"] * 20)
    Xb, yb = balance_classes(X, y, seed=0)
    assert sum(yb == "a") == 20
    assert sum(yb == "b") == 20


def test_train_writes_artifacts_and_returns_metrics(tmp_path):
    df = _synthetic_df()
    metrics = train(df, tmp_path)
    assert (tmp_path / "pose_classifier.joblib").exists()
    assert (tmp_path / "label_encoder.pkl").exists()
    assert "test_top1_accuracy" in metrics
    clf = joblib.load(tmp_path / "pose_classifier.joblib")
    assert hasattr(clf, "predict_proba")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_train.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/training/train.py`**

```python
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [f"f{i}" for i in range(16)]
VIS_COLS = [f"v{i}" for i in range(33)]
ANGLE_IDX = list(range(12))
DIST_IDX = list(range(12, 16))


def augment(X: np.ndarray, n_variants: int = 3,
            angle_jitter: float = 5.0, dist_jitter: float = 0.05,
            seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    out = [X]
    for _ in range(n_variants):
        noise = np.zeros_like(X)
        noise[:, ANGLE_IDX] = rng.uniform(-angle_jitter, angle_jitter, size=(X.shape[0], 12))
        noise[:, DIST_IDX] = rng.uniform(-dist_jitter, dist_jitter, size=(X.shape[0], 4))
        out.append(X + noise)
    return np.vstack(out).astype(np.float32)


def balance_classes(X: np.ndarray, y: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    counts = pd.Series(y).value_counts()
    min_n = int(counts.min())
    keep_idx = []
    for label in counts.index:
        idx = np.where(y == label)[0]
        chosen = rng.choice(idx, size=min_n, replace=False)
        keep_idx.extend(chosen.tolist())
    keep_idx = np.array(keep_idx)
    return X[keep_idx], y[keep_idx]


def train(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()

    X, y = balance_classes(X, y)
    if len(set(y)) < 2:
        raise ValueError("Need at least 2 classes to train")

    X_aug = augment(X)
    y_aug = np.tile(y, 4)

    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug,
    )

    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42,
    ).fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    top1 = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    le = LabelEncoder().fit(df["label"])
    joblib.dump(clf, out_dir / "pose_classifier.joblib")
    joblib.dump(le, out_dir / "label_encoder.pkl")

    return {
        "test_top1_accuracy": float(top1),
        "test_classification_report": report,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "classes": list(le.classes_),
    }


def main(features_path: str = "data/features.parquet",
         out_dir: str = "backend/models"):
    df = pd.read_parquet(features_path)
    metrics = train(df, Path(out_dir))
    print(f"Top-1 accuracy: {metrics['test_top1_accuracy']:.3f}")
    print(f"Classes: {metrics['classes']}")
    print(f"Wrote {out_dir}/pose_classifier.joblib and label_encoder.pkl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features.parquet")
    p.add_argument("--out", default="backend/models")
    args = p.parse_args()
    main(args.features, args.out)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_train.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/training/train.py backend/tests/test_train.py
git commit -m "feat(training): RF train with augmentation, balancing, and metrics"
```

---

## Task 12: Pose Templates (`training/build_templates.py`)

**Files:**
- Create: `backend/training/build_templates.py`
- Test: `backend/tests/test_build_templates.py`

**Interfaces:**
- Consumes: `data/features.parquet`.
- Produces `backend/models/pose_templates.json` of shape `{pose_key: {feature_name: median_value}}` for the **first 12 features** (angles only — feedback uses angles).
- Function `build_templates(df: pd.DataFrame) -> dict[str, dict[str, float]]`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_build_templates.py`:
```python
import numpy as np
import pandas as pd
from backend.training.build_templates import build_templates
from backend.app.features import FEATURE_NAMES


def _synthetic_df():
    rng = np.random.default_rng(0)
    rows = []
    for label in ["tadasana", "vrksasana"]:
        for _ in range(20):
            row = {"label": label, "path": "x"}
            for i in range(16):
                row[f"f{i}"] = float(rng.uniform(0, 180))
            for i in range(33):
                row[f"v{i}"] = 0.9
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_templates_returns_dict_per_pose():
    df = _synthetic_df()
    templates = build_templates(df)
    assert set(templates.keys()) == {"tadasana", "vrksasana"}
    for pose, feats in templates.items():
        for name in FEATURE_NAMES[:12]:
            assert name in feats
            assert isinstance(feats[name], float)


def test_build_templates_excludes_distance_features():
    df = _synthetic_df()
    templates = build_templates(df)
    assert "wrist_to_wrist" not in templates["tadasana"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/test_build_templates.py -v
```

Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/training/build_templates.py`**

```python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.features import FEATURE_NAMES

ANGLE_FEATURE_NAMES = FEATURE_NAMES[:12]
FEATURE_COLS = [f"f{i}" for i in range(12)]


def build_templates(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    grouped = df.groupby("label")[FEATURE_COLS].median()
    out = {}
    for label, row in grouped.iterrows():
        out[str(label)] = {
            ANGLE_FEATURE_NAMES[i]: float(row[col])
            for i, col in enumerate(FEATURE_COLS)
        }
    return out


def main(features_path: str = "data/features.parquet",
         out_path: str = "backend/models/pose_templates.json"):
    df = pd.read_parquet(features_path)
    templates = build_templates(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"Wrote {out_path} for {len(templates)} poses")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features.parquet")
    p.add_argument("--out", default="backend/models/pose_templates.json")
    args = p.parse_args()
    main(args.features, args.out)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && pytest tests/test_build_templates.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/training/build_templates.py backend/tests/test_build_templates.py
git commit -m "feat(training): build per-pose median angle templates for feedback"
```

---

## Task 13: Frontend Scaffolding (`api.js`, `store.js`)

**Files:**
- Create: `frontend/src/api.js`, `frontend/src/store.js`
- Modify: `frontend/src/App.jsx` (interim render of session UI shell)
- Test: `frontend/src/__tests__/store.test.js`

**Interfaces:**
- Produces:
  - `api.js`: `startSession(targets?) -> Promise<{session_id, target_poses}>`, `predict(imageBlob, sessionId) -> Promise<PredictionResponse>`, `getSessionStatus(sessionId)`, `resetSession(sessionId)`, `getPoses()`, `getHealth()`. Base URL `/api` (proxied by Vite).
  - `store.js`: Zustand store with `sessionId, prediction, stats, feedback, start(), reset(), setPrediction()`.

- [ ] **Step 1: Install JS DOM test deps**

```bash
cd frontend && npm install --save-dev jsdom @testing-library/react
```

- [ ] **Step 2: Write the failing test**

`frontend/src/__tests__/store.test.js`:
```javascript
import { describe, it, expect } from 'vitest'
import { useStore } from '../store'

describe('store', () => {
  it('starts with no session', () => {
    expect(useStore.getState().sessionId).toBeNull()
  })

  it('setPrediction updates prediction and feedback', () => {
    useStore.getState().setPrediction({
      label: 'tadasana',
      confidence: 0.9,
      feedback: [{ joint: 'left_knee', cue: 'Straighten', severity: 'major' }],
      hold_seconds: 1.5,
      rep_count: 0,
    })
    const s = useStore.getState()
    expect(s.prediction.label).toBe('tadasana')
    expect(s.feedback).toHaveLength(1)
    expect(s.holdSeconds).toBe(1.5)
  })

  it('reset clears session state', () => {
    useStore.getState().setPrediction({ label: 'tadasana', confidence: 0.9, feedback: [], hold_seconds: 2, rep_count: 1 })
    useStore.getState().reset()
    const s = useStore.getState()
    expect(s.sessionId).toBeNull()
    expect(s.prediction).toBeNull()
  })
})
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd frontend && npx vitest run src/__tests__/store.test.js
```

Expected: FAIL (store not found).

- [ ] **Step 4: Implement `frontend/src/api.js`**

```javascript
const BASE = '/api'

async function jsonOrThrow(res) {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export async function getHealth() {
  return fetch(`${BASE}/health`).then(jsonOrThrow)
}

export async function getPoses() {
  return fetch(`${BASE}/poses`).then(jsonOrThrow)
}

export async function startSession(targets = null) {
  return fetch(`${BASE}/session/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_poses: targets }),
  }).then(jsonOrThrow)
}

export async function getSessionStatus(sessionId) {
  return fetch(`${BASE}/session/${sessionId}`).then(jsonOrThrow)
}

export async function resetSession(sessionId) {
  return fetch(`${BASE}/session/${sessionId}/reset`, { method: 'POST' }).then(jsonOrThrow)
}

export async function predict(imageBlob, sessionId) {
  const form = new FormData()
  form.append('image', imageBlob)
  form.append('session_id', sessionId)
  return fetch(`${BASE}/predict`, { method: 'POST', body: form }).then(jsonOrThrow)
}
```

- [ ] **Step 5: Implement `frontend/src/store.js`**

```javascript
import { create } from 'zustand'

export const useStore = create((set) => ({
  sessionId: null,
  targetPoses: [],
  prediction: null,
  feedback: [],
  holdSeconds: 0,
  repCount: 0,
  repCountPerPose: {},
  history: [],
  isRunning: false,

  setSession: (sessionId, targetPoses) => set({ sessionId, targetPoses }),
  setRunning: (isRunning) => set({ isRunning }),

  setPrediction: (p) => set({
    prediction: p,
    feedback: p.feedback || [],
    holdSeconds: p.hold_seconds || 0,
    repCount: p.rep_count || 0,
  }),

  setSessionStatus: (s) => set({
    repCountPerPose: s.rep_count_per_pose || {},
    history: s.history || [],
  }),

  reset: () => set({
    sessionId: null,
    targetPoses: [],
    prediction: null,
    feedback: [],
    holdSeconds: 0,
    repCount: 0,
    repCountPerPose: {},
    history: [],
    isRunning: false,
  }),
}))
```

- [ ] **Step 6: Run test to verify it passes**

```bash
cd frontend && npx vitest run src/__tests__/store.test.js
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api.js frontend/src/store.js frontend/src/__tests__/store.test.js frontend/package.json frontend/package-lock.json
git commit -m "feat(frontend): add API client and Zustand session store"
```

---

## Task 14: Pose Detection Hook (`usePoseDetection.js`)

**Files:**
- Create: `frontend/src/hooks/usePoseDetection.js`

**Interfaces:**
- Consumes: `api.predict`, the Zustand store.
- Produces a React hook that takes a `mediaStream` and an `enabled` flag, owns the frame loop with adaptive throttling (skip frame if previous fetch still in flight), targets ~8 FPS via a 125 ms `setTimeout` cadence.

- [ ] **Step 1: Implement the hook (no JS unit test; verified by integration in Task 18)**

`frontend/src/hooks/usePoseDetection.js`:
```javascript
import { useEffect, useRef } from 'react'
import { predict } from '../api'
import { useStore } from '../store'

const FRAME_INTERVAL_MS = 125

export function usePoseDetection(videoRef, enabled) {
  const runningRef = useRef(false)
  const canvasRef = useRef(null)
  const inFlightRef = useRef(false)
  const sessionId = useStore((s) => s.sessionId)
  const setPrediction = useStore((s) => s.setPrediction)

  useEffect(() => {
    if (!enabled || !sessionId) return
    runningRef.current = true
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas')
      canvasRef.current.width = 640
      canvasRef.current.height = 480
    }
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    const tick = async () => {
      if (!runningRef.current) return
      const video = videoRef.current
      if (video && video.readyState >= 2 && !inFlightRef.current) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        inFlightRef.current = true
        canvas.toBlob(async (blob) => {
          try {
            const result = await predict(blob, sessionId)
            if (runningRef.current) setPrediction(result)
          } catch (err) {
            console.error('predict failed', err)
          } finally {
            inFlightRef.current = false
          }
        }, 'image/jpeg', 0.7)
      }
      setTimeout(tick, FRAME_INTERVAL_MS)
    }
    tick()

    return () => { runningRef.current = false }
  }, [enabled, sessionId, videoRef, setPrediction])
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

Expected: build succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/usePoseDetection.js
git commit -m "feat(frontend): add usePoseDetection hook with throttled frame loop"
```

---

## Task 15: Webcam + Overlay Components

**Files:**
- Create: `frontend/src/components/WebcamView.jsx`, `frontend/src/components/PoseOverlay.jsx`

**Interfaces:**
- Consumes: `navigator.mediaDevices.getUserMedia`, the prediction from store.
- Produces:
  - `WebcamView({ videoRef, onError })` — sets up webcam stream, exposes `<video>` ref to parent.
  - `PoseOverlay({ landmarks, width, height })` — draws skeleton on `<canvas>` using MediaPipe's POSE_CONNECTIONS topology.

- [ ] **Step 1: Implement `WebcamView.jsx`**

```jsx
import { useEffect } from 'react'

export default function WebcamView({ videoRef, onError }) {
  useEffect(() => {
    let stream
    let active = true
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then((s) => {
        if (!active) { s.getTracks().forEach((t) => t.stop()); return }
        stream = s
        if (videoRef.current) {
          videoRef.current.srcObject = s
          videoRef.current.play().catch(onError)
        }
      })
      .catch(onError)
    return () => {
      active = false
      if (stream) stream.getTracks().forEach((t) => t.stop())
    }
  }, [videoRef, onError])

  return (
    <video
      ref={videoRef}
      playsInline
      muted
      style={{ width: '100%', maxWidth: 640, borderRadius: 8, background: '#000' }}
    />
  )
}
```

- [ ] **Step 2: Implement `PoseOverlay.jsx`**

```jsx
const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
]

export default function PoseOverlay({ landmarks, width = 640, height = 480 }) {
  return (
    <canvas
      width={width}
      height={height}
      ref={(canvas) => {
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, width, height)
        if (!landmarks || landmarks.length < 33) return
        ctx.strokeStyle = 'rgba(80, 220, 120, 0.9)'
        ctx.lineWidth = 3
        POSE_CONNECTIONS.forEach(([a, b]) => {
          const p1 = landmarks[a], p2 = landmarks[b]
          if (!p1 || !p2) return
          ctx.beginPath()
          ctx.moveTo(p1[0] * width, p1[1] * height)
          ctx.lineTo(p2[0] * width, p2[1] * height)
          ctx.stroke()
        })
        ctx.fillStyle = 'rgba(255, 220, 80, 0.95)'
        landmarks.forEach((lm) => {
          ctx.beginPath()
          ctx.arc(lm[0] * width, lm[1] * height, 4, 0, 2 * Math.PI)
          ctx.fill()
        })
      }}
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none', width: '100%', maxWidth: 640 }}
    />
  )
}
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

Expected: success.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/WebcamView.jsx frontend/src/components/PoseOverlay.jsx
git commit -m "feat(frontend): add WebcamView and PoseOverlay canvas components"
```

---

## Task 16: Feedback, Stats, and Pose Picker Components

**Files:**
- Create: `frontend/src/components/FeedbackBanner.jsx`, `frontend/src/components/SessionStats.jsx`, `frontend/src/components/PosePicker.jsx`

- [ ] **Step 1: Implement `FeedbackBanner.jsx`**

```jsx
export default function FeedbackBanner({ feedback }) {
  if (!feedback || feedback.length === 0) {
    return (
      <div style={styles.box({ severity: 'ok' })}>
        <strong>Nice form</strong>
        <span>Hold the pose and breathe.</span>
      </div>
    )
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {feedback.map((h, i) => (
        <div key={i} style={styles.box({ severity: h.severity })}>
          <strong>{h.severity === 'major' ? 'Adjust' : 'Tip'}</strong>
          <span>{h.cue}</span>
        </div>
      ))}
    </div>
  )
}

const styles = {
  box: ({ severity }) => ({
    padding: '10px 14px',
    borderRadius: 8,
    background: severity === 'major' ? '#ffe3e3' : severity === 'minor' ? '#fff6d6' : '#e3f7e8',
    border: `1px solid ${severity === 'major' ? '#d33' : severity === 'minor' ? '#dc0' : '#3c3'}`,
    display: 'flex',
    gap: 10,
    alignItems: 'center',
  }),
}
```

- [ ] **Step 2: Implement `SessionStats.jsx`**

```jsx
import { useStore } from '../store'

export default function SessionStats() {
  const prediction = useStore((s) => s.prediction)
  const holdSeconds = useStore((s) => s.holdSeconds)
  const repCount = useStore((s) => s.repCount)
  const repCountPerPose = useStore((s) => s.repCountPerPose)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div>
        <strong>Current pose:</strong> {prediction?.label || '—'}{' '}
        {prediction && `(${Math.round((prediction.confidence || 0) * 100)}%)`}
      </div>
      <div><strong>Hold:</strong> {holdSeconds.toFixed(1)} s</div>
      <div><strong>Reps this pose:</strong> {repCount}</div>
      <div>
        <strong>Session totals:</strong>{' '}
        {Object.keys(repCountPerPose).length === 0
          ? 'none yet'
          : Object.entries(repCountPerPose).map(([k, v]) => `${k}: ${v}`).join(', ')}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Implement `PosePicker.jsx`**

```jsx
import { useEffect, useState } from 'react'
import { getPoses } from '../api'

export default function PosePicker({ onPick }) {
  const [poses, setPoses] = useState([])
  useEffect(() => {
    getPoses().then((r) => setPoses(r.poses || [])).catch(() => {})
  }, [])
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
      <button onClick={() => onPick(null)}>Free practice</button>
      {poses.map((p) => (
        <button key={p.key} onClick={() => onPick([p.key])}>{p.display_name}</button>
      ))}
    </div>
  )
}
```

- [ ] **Step 4: Verify build**

```bash
cd frontend && npm run build
```

Expected: success.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/FeedbackBanner.jsx frontend/src/components/SessionStats.jsx frontend/src/components/PosePicker.jsx
git commit -m "feat(frontend): add feedback, stats, and pose picker components"
```

---

## Task 17: App Composition

**Files:**
- Modify: `frontend/src/App.jsx`
- Create: `frontend/src/styles.css`

- [ ] **Step 1: Implement `frontend/src/styles.css`**

```css
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, sans-serif; background: #f6f7f9; color: #1c1c1c; }
.app { max-width: 1100px; margin: 0 auto; padding: 20px; display: grid; gap: 18px; }
.header { display: flex; justify-content: space-between; align-items: center; }
.left-col, .right-col { display: flex; flex-direction: column; gap: 14px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
.stage { position: relative; width: 100%; max-width: 640px; }
button { padding: 6px 12px; border-radius: 6px; border: 1px solid #ccc; background: #fff; cursor: pointer; }
button:hover { background: #f0f0f0; }
```

- [ ] **Step 2: Replace `frontend/src/App.jsx`**

```jsx
import { useEffect, useRef, useState } from 'react'
import './styles.css'
import { useStore } from './store'
import { startSession, resetSession, getSessionStatus } from './api'
import { usePoseDetection } from './hooks/usePoseDetection'
import WebcamView from './components/WebcamView'
import PoseOverlay from './components/PoseOverlay'
import FeedbackBanner from './components/FeedbackBanner'
import SessionStats from './components/SessionStats'
import PosePicker from './components/PosePicker'

export default function App() {
  const videoRef = useRef(null)
  const [error, setError] = useState(null)
  const [enabled, setEnabled] = useState(false)
  const {
    sessionId, prediction, feedback,
    setSession, setRunning, reset, setSessionStatus,
  } = useStore()

  usePoseDetection(videoRef, enabled)

  useEffect(() => {
    if (!sessionId) return
    const id = setInterval(() => getSessionStatus(sessionId)
      .then(setSessionStatus)
      .catch(() => {}), 2000)
    return () => clearInterval(id)
  }, [sessionId, setSessionStatus])

  const handlePick = async (targets) => {
    const { session_id, target_poses } = await startSession(targets)
    setSession(session_id, target_poses)
    setEnabled(true)
    setRunning(true)
  }

  const handleStop = async () => {
    setEnabled(false)
    setRunning(false)
    if (sessionId) await resetSession(sessionId).catch(() => {})
    reset()
  }

  return (
    <div className="app">
      <div className="header">
        <h1>Yoga Poser</h1>
        <div>{sessionId ? `Session ${sessionId.slice(0, 8)}` : 'No session'}</div>
      </div>

      <div className="grid">
        <div className="left-col">
          <div className="stage">
            <WebcamView videoRef={videoRef} onError={setError} />
            <PoseOverlay landmarks={prediction?.landmarks || null} />
          </div>
          {error && <div style={{ color: '#c00' }}>Camera error: {String(error)}</div>}
          <FeedbackBanner feedback={feedback} />
        </div>

        <div className="right-col">
          <SessionStats />
          <div>
            <h3>Pick a pose</h3>
            <PosePicker onPick={handlePick} />
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={handleStop} disabled={!sessionId}>Reset session</button>
          </div>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

Expected: success.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.jsx frontend/src/styles.css
git commit -m "feat(frontend): compose app shell with webcam, overlay, feedback, and stats"
```

---

## Task 18: Integration Smoke Test and README Finalization

**Files:**
- Modify: `README.md`
- Test: `backend/tests/test_integration.py`

**Goal:** Verify the end-to-end pipeline runs against a stubbed classifier + estimator, then document run steps in README.

- [ ] **Step 1: Write integration test**

`backend/tests/test_integration.py`:
```python
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app


def test_end_to_end_predict_with_stubbed_estimator_and_classifier():
    client = TestClient(app)
    fake_landmarks = np.random.default_rng(0).uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)
    fake_landmarks[:, 3] = 0.95
    with patch("backend.app.main._estimator") as est, \
         patch("backend.app.main._classifier") as clf, \
         patch("backend.app.main._feedback_engine") as fe:
        est.estimate.return_value = fake_landmarks
        proba = clf.predict.return_value
        proba.label = "tadasana"
        proba.confidence = 0.9
        proba.probabilities = {"tadasana": 0.9}
        proba.low_confidence_reason = None
        fe.get_feedback.return_value = []
        start = client.post("/api/session/start", json={})
        sid = start.json()["session_id"]
        files = {"image": ("f.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")}
        data = {"session_id": sid}
        r = client.post("/api/predict", files=files, data=data)
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
    assert "mediapipe_version" in r.json()
```

- [ ] **Step 2: Run all backend tests**

```bash
cd backend && pytest -v
```

Expected: all green (counts: features=4, pose_estimator=2, schemas=4, feedback=3, session=5, classifier=2, main=5, download=3, extract_pipeline=2, train=3, templates=2, integration=2 → ~37 tests).

- [ ] **Step 3: Run frontend tests and build**

```bash
cd frontend && npx vitest run && npm run build
```

Expected: 3 store tests pass; build succeeds with `dist/` produced.

- [ ] **Step 4: Finalize `README.md`**

```markdown
# Yoga Poser

Real-time yoga pose detector. Webcam frames flow from a React frontend to a FastAPI backend that runs MediaPipe Pose + a scikit-learn RandomForest classifier on a 16-dim feature vector. Form-feedback cues, hold-time tracking, and rep counting are all in.

See `docs/superpowers/specs/2026-08-12-yoga-pose-detector-design.md` for the full design.

## Prerequisites
- Python 3.10+ (this repo pins 3.10.14 via pyenv's `.python-version`)
- Node 18+

## Quick start
```bash
# Backend
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..

# Train (optional: ship pre-trained model under backend/models/)
python -m backend.training.download_data    # prints manual dataset steps
python -m backend.training.extract_features
python -m backend.training.train
python -m backend.training.build_templates

# Run
./scripts/dev.sh
# → open http://localhost:5173
```

## API
- `GET  /api/health`
- `GET  /api/poses`
- `POST /api/session/start`  `{target_poses?: [...]}`
- `GET  /api/session/{id}`
- `POST /api/session/{id}/reset`
- `POST /api/predict`        multipart `image`, `session_id`
- API docs: http://localhost:8000/docs

## Tests
```bash
# Backend
cd backend && source .venv/bin/activate && pytest -v
# Frontend
cd frontend && npx vitest run
```
```

- [ ] **Step 5: Commit**

```bash
git add README.md backend/tests/test_integration.py
git commit -m "test: end-to-end integration test and finalize README"
```

---

## Self-Review Notes

**Spec coverage check:**
- §3 architecture → Tasks 1, 8, 13, 17.
- §4 pose set (8 classes) → Tasks 4 (catalog), 9 (synonyms).
- §5 project structure → Task 1 (scaffolding) + per-task file creation matches.
- §6 feature vector (16-dim) → Task 2.
- §7.1–7.6 backend components → Tasks 2–8.
- §8 frontend components → Tasks 13–17.
- §9 training pipeline → Tasks 9–12.
- §10 feedback & tracking logic → Tasks 5, 6.
- §11 scope summary → matched.
- §12 risks → mitigations embedded in code (spine_arch feature, throttling in hook, low-visibility drop, version pins, fallback to "Unknown").
- §13 v2 triggers → documented in spec; no extra task needed for v1.

**Placeholder scan:** No TBD/TODO/“fill in details”/“add appropriate error handling” patterns. Each test contains real assertions and each implementation step contains real code.

**Type/name consistency:** `extract_features(features, visibility)`, `PredictionResult(label, confidence, probabilities, low_confidence_reason)`, `FeedbackEngine(templates).get_feedback(pose_key, features)`, `SessionStore.start/get/evict_idle`, `PoseClassifier.predict(features, visibility)`, `POSE_CATALOG`, `FEATURE_NAMES` all referenced consistently across tasks.

**Known data gap:** Tasks 9–12 require actual dataset download (manual Kaggle token step). Tasks 11 and 12 are still tested via synthetic in-memory DataFrames so they are independently verifiable. The model artifacts (`pose_classifier.joblib`, etc.) are not produced by tests — the server is designed to boot without them (Task 7 `load_default()` returns `None`; Task 8 returns `"Unknown"` label when classifier is `None`).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-12-yoga-pose-detector.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
