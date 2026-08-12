# Yoga Pose Detector — v1 Design

**Status:** Approved
**Date:** 2026-08-12
**Authors:** brainstorming session (sanjeew + opencode)

## 1. Goal

Build a real-time yoga pose detector: a webcam-enabled web app that identifies which of 8 yoga asanas the user is performing, gives form-correction cues, and tracks hold time and reps during a practice session.

**v1 success criteria:**
- Detects the 8 target poses at >=85% top-1 accuracy on a held-out test set.
- Runs at >=5 FPS on a laptop with CPU only.
- Produces at least one actionable form-correction cue per misaligned pose.
- Tracks hold time (seconds) and reps per pose within a session.

## 2. Non-goals (deferred to v2+)

- Multi-person / multi-user-in-frame detection.
- Pose flow sequencing and overall session scoring.
- Voice cues (text-to-speech).
- Persisted session history across server restarts (v1 is in-memory).
- Authentication and accounts.
- Mobile-native apps.
- CNN-based image classifier (see "v2 upgrade triggers" below).

## 3. Architecture

```
┌─────────────────────────┐         ┌──────────────────────────────────────┐
│  React + Vite frontend  │  HTTP   │  FastAPI backend (Python)            │
│                         │  POST   │                                      │
│  Webcam  → canvas       │ ──────▶ │  /api/predict  (image → pose label)  │
│  ~8 FPS frame capture   │  JSON   │    ├─ MediaPipe Pose (33 landmarks)  │
│  Pose overlay + label   │ ◀────── │    ├─ Feature extractor (angles)     │
│  Feedback banner        │         │    ├─ Classifier (Random Forest)     │
│  Hold timer / reps      │         │    ├─ Feedback engine (angle diffs)  │
│  Session summary        │         │    └─ Session state (in-memory)      │
└─────────────────────────┘         │                                      │
                                     │  /api/session/start|status|reset     │
                                     │  /api/health, /api/poses             │
                                     └──────────────────────────────────────┘
                       ┌─────────────────────────────┐
                       │ Offline training pipeline   │
                       │ Yoga-82 + Kaggle images     │
                       │   → MediaPipe               │
                       │   → angle features          │
                       │   → RandomForest fit        │
                       │   → model.joblib            │
                       └─────────────────────────────┘
```

**Key decisions:**
- All pose inference runs in the Python backend. The frontend streams JPEG frames over HTTP POST and renders results.
- MediaPipe Pose is the pose estimator (33 landmarks, 3D, CPU-friendly, single-person focus).
- A scikit-learn `RandomForestClassifier` runs on a 16-dimensional hand-crafted feature vector derived from joint angles and normalized distances.
- MediaPipe runs in **both** the training pipeline (over dataset images) and the serving backend, pinned to the same version so features match.

## 4. v1 Pose Set (8 classes)

| Class key | Display name | Sanskrit | Notes |
|---|---|---|---|
| `tadasana` | Mountain | Tadasana | Neutral standing baseline; rest state. |
| `adho_mukha_svanasana` | Downward-Facing Dog | Adho Mukha Svanasana | Inverted V; visually distinctive. |
| `virabhadrasana_i` | Warrior I | Virabhadrasana I | Lunge, arms raised overhead. |
| `virabhadrasana_ii` | Warrior II | Virabhadrasana II | Lunge, arms extended sideways. |
| `vrksasana` | Tree | Vrksasana | Standing balance, one foot on inner thigh. |
| `bhujangasana` | Cobra | Bhujangasana | Prone backbend, chest lifted. |
| `balasana` | Child's Pose | Balasana | Kneeling fold. **Visually similar to Cat.** |
| `marjaryasana` | Cat | Marjaryasana | Tabletop, spine rounded. **Visually similar to Child's.** |

**Known risk:** Cat (`marjaryasana`) and Child's Pose (`balasana`) are both kneeling/tabletop poses. Mitigation: add a spine-arch feature in the feature vector (see §6) and inspect the confusion matrix; if confusion is unresolvable, document and consider merging in a later iteration.

## 5. Project Structure

```
Yoga_poser/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + routes
│   │   ├── pose_estimator.py    # MediaPipe wrapper (singleton)
│   │   ├── features.py          # landmark → 16-dim feature vector
│   │   ├── classifier.py        # joblib load + predict
│   │   ├── feedback.py          # angle-diff correction rules
│   │   ├── session.py           # hold-time, rep counter, state
│   │   └── schemas.py           # Pydantic request/response models
│   ├── models/
│   │   ├── pose_classifier.joblib
│   │   ├── label_encoder.pkl
│   │   └── pose_templates.json  # ideal-angle reference per pose
│   ├── training/
│   │   ├── download_data.py
│   │   ├── extract_features.py  # batch MediaPipe over images
│   │   ├── train.py             # sklearn RF, cross-val, metrics
│   │   ├── build_templates.py   # median angles per pose → JSON
│   │   └── notebooks/
│   ├── tests/
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── WebcamView.jsx
│   │   │   ├── PoseOverlay.jsx
│   │   │   ├── FeedbackBanner.jsx
│   │   │   └── SessionStats.jsx
│   │   ├── hooks/
│   │   │   └── usePoseDetection.js   # frame loop + fetch
│   │   ├── store.js                  # Zustand store
│   │   └── api.js
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── dev.sh                   # boot backend + frontend together
├── data/                        # gitignored: raw datasets
├── docs/superpowers/specs/2026-08-12-yoga-pose-detector-design.md
├── .gitignore
└── README.md
```

## 6. Feature Vector (16 dimensions)

The classifier input is a 16-element `float32` vector, scale- and translation-invariant (normalized against torso length, defined as the Euclidean distance from the hip center to the shoulder center).

### 12 joint angles (degrees, 0-180)

Computed via the cosine rule on landmark triples:

| # | Angle | Landmarks used |
|---|---|---|
| 1 | Left elbow | shoulder-elbow-wrist (L) |
| 2 | Right elbow | shoulder-elbow-wrist (R) |
| 3 | Left shoulder | elbow-shoulder-hip (L) |
| 4 | Right shoulder | elbow-shoulder-hip (R) |
| 5 | Left knee | hip-knee-ankle (L) |
| 6 | Right knee | hip-knee-ankle (R) |
| 7 | Left hip | knee-hip-shoulder (L) |
| 8 | Right hip | knee-hip-shoulder (R) |
| 9 | Torso lean (sagittal) | Angle at mid-hip between (mid-shoulder → mid-hip) and (mid-hip → vertical-up). 0° = upright; positive = leaning forward. |
| 10 | Spine arch (sagittal) | Angle at mid-shoulder between (mid-hip → mid-shoulder) and (mid-shoulder → mid-head). 0° = neutral; positive = arched back (Cat); negative = rounded forward (Child's). |
| 11 | Left wrist-to-shoulder closure | wrist-shoulder-elbow (L) |
| 12 | Right wrist-to-shoulder closure | wrist-shoulder-elbow (R) |

Features #9 and #10 are added to disambiguate visually similar poses (Cat vs Child's, Cobra vs upward variations).

### 4 normalized distances (÷ torso length)

| # | Distance | Why |
|---|---|---|
| 13 | Wrist-to-wrist horizontal | Arm spread: separates Warrior II (wide) from Warrior I (overhead). |
| 14 | Ankle-to-ankle horizontal | Stance width: separates standing poses from one-leg balances. |
| 15 | Wrist height (relative to hip) | Arms up (Warrior I, Tree, Dog) vs arms out. |
| 16 | Hip-to-wrist vertical reach | Distinguishes reaching poses (Dog, Triangle). |

### Visibility handling

Each sample also carries a 33-element visibility mask. During training, samples with torso-keypoint visibility < 0.5 are dropped. During inference, if the predicted-pose's critical landmarks (per pose) have low visibility, the response includes a `low_confidence_reason` hint.

## 7. Backend Components

### 7.1 `pose_estimator.py`
- Wraps `mediapipe.solutions.pose.Pose` as a module-level singleton, model complexity = 1, `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`.
- Method: `estimate(image_bytes_or_array) -> Optional[Landmarks]` where `Landmarks` is a dataclass holding `np.ndarray` of shape `(33, 4)` (x, y, z, visibility).
- Returns `None` on no-detection or decode failure; never raises to the API layer.

### 7.2 `features.py`
- Pure function: `extract_features(landmarks: Landmarks) -> tuple[np.ndarray(16,), np.ndarray(33,)]` returning the feature vector and visibility mask.
- Shared between training pipeline and serving backend to guarantee identical feature semantics.

### 7.3 `classifier.py`
- Loads `models/pose_classifier.joblib` and `models/label_encoder.pkl` at startup.
- Method: `predict(features, visibility) -> PredictionResult(label, confidence, probabilities, low_confidence_reason)`.
- Confidence threshold: top-1 probability < 0.45 → label is `"Unknown"`.

### 7.4 `feedback.py`
- Loads `models/pose_templates.json` containing the 12 ideal median angles per pose (computed by `build_templates.py`).
- For a given predicted pose, computes signed deltas vs template for each of the 12 angles.
- If `|delta| > 20°` for any angle, emits a correction hint via a lookup table mapping joint-delta-signature to a human cue (e.g. `front_knee_too_bent` → `"Straighten your front knee"`).
- Returns at most 2 hints, sorted by magnitude. Hints clear automatically when the user's angle enters the ±10° band.

### 7.5 `session.py`
- In-memory `dict[session_id -> SessionState]`, evicted after 30 min idle.
- `SessionState` tracks:
  - `current_pose`, `current_since_ts`
  - `hold_seconds` (only accrues when label is stable >=1 s and confidence >=0.6 and no severe feedback hint)
  - `rep_count_per_pose: dict[str, int]`
  - `history: list[SessionEvent]`
- A rep is registered when a pose is held >=3 s and then exited (label change or confidence drop). 0.5 s cooldown prevents double-counting.

### 7.6 API Endpoints

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/api/health` | GET | — | `{status, model_loaded, mediapipe_version}` |
| `/api/poses` | GET | — | List of 8 pose descriptors (key, display name, description, `ideal_image_url`). Ideal images are committed under `backend/static/poses/<key>.jpg` and sourced from the training dataset's highest-visibility, highest-confidence sample per class. |
| `/api/session/start` | POST | `{target_poses?: [keys]}` | `{session_id, target_poses}` |
| `/api/session/{id}` | GET | — | `{current_pose, hold_seconds, reps, history}` |
| `/api/session/{id}/reset` | POST | — | `{ok: true}` |
| `/api/predict` | POST | `multipart: image (JPEG), session_id` | `{label, confidence, landmarks, feedback[], hold_seconds, rep_count}` |
| `/` | GET | — | Serves built React bundle from `frontend/dist` (production only) |

CORS allows `http://localhost:5173` in dev.

## 8. Frontend (React + Vite)

- **`WebcamView`** — `getUserMedia({video: true})`, draws video to a hidden `<video>` element, samples to a `<canvas>` at ~8 FPS using `requestAnimationFrame` with a 125 ms throttle.
- **`usePoseDetection` hook** — owns the frame loop, POSTs JPEG blob to `/api/predict` via `fetch`, exposes `{prediction, isLoading, error}`. Adaptive throttling: if a previous request is still in flight, the next frame is skipped.
- **`PoseOverlay`** — draws the 33 returned landmarks + skeleton connections on a canvas above the video. Uses a port of MediaPipe's `POSE_CONNECTIONS` to JS.
- **`FeedbackBanner`** — colored card showing correction text and a severity icon. Clears automatically when backend returns no hints.
- **`SessionStats`** — current pose, live hold timer, rep count per pose, session history.
- **`PosePicker`** — pick a target pose or "Free practice" mode.
- **State** — Zustand store (`store.js`) holds session id, predictions, feedback, and stats. No Redux.

## 9. Training Pipeline

Run via `python -m backend.training.<script>`. Steps:

1. **Download datasets** — `download_data.py` fetches Yoga-82 and the Kaggle Yoga Posture Dataset into `data/raw/`. Document manual Kaggle API token setup.
2. **Filter & relabel** — Keep only the 8 v1 classes, mapping synonyms (e.g. "Downward-Facing Dog", "Adho Mukha Svanasana", "Down Dog" → `adho_mukha_svanasana`).
3. **Extract landmarks** — `extract_features.py` runs MediaPipe over every image; drops samples where torso-keypoint visibility < 0.5. Writes `data/features.parquet` with columns `path, label, f0..f15, v0..v32`.
4. **Augment** — For each sample, generate up to 3 augmented variants by perturbing each of the **12 angle features** by an independent uniform random ±5° and each of the **4 distance features** by ±0.05 (normalized units). Visibility mask is copied unchanged. This improves generalization against dataset photo style.
5. **Balance** — Random undersample to the smallest per-class N. Target: at least 80 samples per class after balancing.
6. **Split** — Stratified 70/15/15 train/val/test.
7. **Train** — `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`. Also train `SVC(rbf)` and `GradientBoostingClassifier` as baselines; report all three.
8. **Evaluate** — Per-class precision/recall/F1, top-1 and top-2 accuracy, confusion matrix (saved as PNG to `data/confusion_matrix.png`).
9. **Build templates** — `build_templates.py` computes the median of each of the 12 angles per pose from the training set and writes `models/pose_templates.json`.
10. **Serialize** — `model.joblib` and `label_encoder.pkl` written to `backend/models/`. Committed to git (expected size <1 MB).

**Success gate:** top-1 accuracy >=85% on the held-out test set across the 8 classes. The 80% figure referenced in §13 is the *floor* — anything between 80% and 85% is acceptable for v1 ship but flagged for improvement; below 80% triggers the v2 redesign.

## 10. Feedback & Tracking Logic

### Hold-time accrual
Hold seconds accrue when **all** are true:
- Top-1 label has been stable for >=1 s (rolling window over last 5 predictions).
- Top-1 confidence >=0.6.
- No feedback hint of severity `major` is currently active.

If any condition breaks, the timer pauses (does not reset). It resets only when the user's predicted pose changes.

### Rep counting
A rep is registered when:
- A pose was held >=3 s, AND
- The user exits the pose (label change for >=0.5 s, or confidence drops below 0.4 for >=0.5 s).

A 0.5 s cooldown after a rep prevents double-counting from jitter.

### Feedback priority
Joint priority order for surfacing hints when more than 2 are active: spine > hips > knees > shoulders > elbows > wrists. Top 2 by `(magnitude × priority_weight)` are shown.

## 11. Scope Summary

**In scope (v1):**
- 8 poses, real-time webcam detection
- Pose label + confidence + landmark overlay
- Angle-based feedback (up to 2 cues at a time)
- Hold-time tracking and rep counting per pose
- Session start, reset, history (in-memory)
- Reproducible training pipeline from public datasets
- Local-only deployment, served via FastAPI + Vite dev server

**Out of scope (v1):**
- See §2.

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Cat ↔ Child's Pose confusion | Spine-arch feature (#10) added; inspect confusion matrix; if F1 on either class <0.7, consider adding curvature feature or merging classes in a later iteration. |
| HTTP POST latency spikes at 8 FPS | Frontend adaptive throttling (skip frame if previous request in flight); server-side prediction timeout of 2 s. |
| Public dataset label noise / MediaPipe failures | Drop low-visibility samples; require min per-class N=80; fall back to self-recording to top up small classes. |
| Classifier overfits to dataset photo style | Aggressive feature normalization; ±5° augmentation; report val vs test gap to detect. |
| MediaPipe version drift between training and serving | Pin `mediapipe==<exact>` in `requirements.txt`; CI check that `extract_features` is imported from a single shared module. |
| Browser webcam permission denied | Frontend shows a friendly fallback with a "Upload image" mode hitting the same `/api/predict`. |

## 13. v2 Upgrade Triggers

Documented conditions that should trigger revisiting the classifier choice (likely to a CNN or hybrid CNN+MediaPipe):

- v1 top-1 accuracy lands below 80% on the 8-pose test set after feature engineering and augmentation.
- Pose count expands beyond ~20 classes, where hand-crafted features become ambiguous.
- New rare poses added where critical joints are occluded (arm binds, inversions, headstand).

In v2, the recommended path is a hybrid: fine-tuned EfficientNet-B0 for classification + MediaPipe retained for overlay and feedback. Expected to roughly double per-frame latency.

## 14. Open Items Resolved

- **Augmentation:** Yes, ±5° landmark perturbation, up to 3 variants per sample. (Was open item #1.)
- **Model file storage:** Commit `model.joblib` and `label_encoder.pkl` in git under `backend/models/`. (Was open item #2.)
- **Free-practice vs guided:** Both supported in v1 via `PosePicker`. `target_poses` is optional in `/api/session/start`. (Was open item #3.)
- **CNN for classification:** Rejected for v1; landmark + RF chosen. Documented as v2 upgrade path. (User-confirmed.)

## 15. Glossary

- **Landmark** — A 3D body keypoint returned by MediaPipe Pose (33 per person).
- **Feature vector** — The 16-dim normalized input to the classifier.
- **Template** — The reference pose: median of each of the 12 joint angles across training samples for a class.
- **Hold seconds** — Continuous seconds the user has been in a stable, confident prediction of one pose.
- **Rep** — One completed cycle: held a pose >=3 s, then exited.
