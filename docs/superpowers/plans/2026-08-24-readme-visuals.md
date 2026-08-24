# README Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add committed, script-generated yoga pose pictures and annotated detection examples to the root `README.md`.

**Architecture:** A committed script `scripts/generate_readme_images.py` walks `data/raw/` via the existing `find_images` helper, picks the highest-visibility sample per pose with `PoseEstimator`, runs the committed classifier for detection examples, and draws the same skeleton overlay the frontend renders. `README.md` gains two sections referencing the generated JPGs under `docs/images/`.

**Tech Stack:** Python 3.10 (backend venv), OpenCV (`cv2`), NumPy, existing `backend.app` modules. No new dependencies.

## Global Constraints

- Script must be committed and re-runnable: `backend/.venv/bin/python scripts/generate_readme_images.py` from the repo root.
- Script self-inserts the repo root into `sys.path` so `backend.*` imports resolve when run as a plain file.
- Skeleton topology and colors must match `frontend/src/components/PoseOverlay.jsx`: connections list identical; lines RGB(80,220,120), joints RGB(255,220,80) (convert to BGR for OpenCV).
- Generated JPGs are committed under `docs/images/` (12 files: 8 poses + 4 detection).
- No changes to `backend/app/**`, `backend/training/**`, or existing tests.
- No comments in code unless absolutely required.
- Detection keys (from spec): `virabhadrasana_ii`, `adho_mukha_svanasana`, `vrksasana`, `marjaryasana`.
- If model artifacts are missing, the poses grid still generates, then the script exits non-zero with a clear message before detection images.
- Scan cap: at most 25 candidate images per pose pass through MediaPipe (keeps runtime a few minutes).
- All backend tests run from the repo root: `backend/.venv/bin/python -m pytest backend/tests -v` (CWD on `sys.path` is what makes `backend.*` and `scripts.*` importable).

---

## Task 1: Image Selection and Overlay Helpers

**Files:**
- Create: `scripts/generate_readme_images.py`
- Test: `backend/tests/test_generate_readme_images.py`

**Interfaces:**
- Consumes (runtime only, imported after `sys.path` fixup): `backend.app.pose_estimator.PoseEstimator`, `backend.app.classifier.load_default` and `PredictionResult`, `backend.app.features.extract_features`, `backend.training.download_data.find_images`.
- Produces (used by Task 2 and tests):
  - `torso_visibility(landmarks: np.ndarray) -> float` — min visibility over indices `[11, 12, 23, 24]`.
  - `resize_to_width(image: np.ndarray, max_width: int = 640) -> np.ndarray`
  - `pick_best_sample(candidates, estimator, max_scan: int = 25) -> tuple[np.ndarray, np.ndarray, Path] | None` — `(image, landmarks, path)` of the highest-visibility candidate.
  - `pick_detection_sample(candidates, estimator, classifier, key: str, max_scan: int = 25) -> tuple[np.ndarray, np.ndarray, PredictionResult] | None` — prefers the highest-visibility candidate whose predicted label equals `key`; falls back to the best-visibility candidate with its actual prediction.
  - `draw_overlay(image: np.ndarray, landmarks: np.ndarray, label: str, confidence: float) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_generate_readme_images.py`:

```python
import cv2
import numpy as np
from types import SimpleNamespace

from scripts.generate_readme_images import (
    torso_visibility, resize_to_width, pick_best_sample,
    pick_detection_sample, draw_overlay,
)


def _landmarks(vis=0.9):
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 0] = 0.5
    lm[:, 1] = np.linspace(0.1, 0.9, 33)
    lm[:, 3] = vis
    return lm


class FakeEstimator:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def estimate(self, image):
        r = self.results[self.calls]
        self.calls += 1
        return r


class FakeClassifier:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def predict(self, features, visibility):
        r = self.results[self.calls]
        self.calls += 1
        return r


def _write_image(path, width=120, height=90):
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def test_torso_visibility_is_min_over_torso_indices():
    lm = _landmarks(vis=0.9)
    lm[24, 3] = 0.4
    assert torso_visibility(lm) == 0.4


def test_resize_to_width_scales_only_wide_images():
    img = np.zeros((90, 1280, 3), dtype=np.uint8)
    out = resize_to_width(img, max_width=640)
    assert out.shape == (45, 640, 3)
    narrow = np.zeros((90, 300, 3), dtype=np.uint8)
    assert resize_to_width(narrow, max_width=640) is narrow


def test_pick_best_sample_prefers_highest_visibility(tmp_path):
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b.jpg")
    low = _landmarks(vis=0.5)
    high = _landmarks(vis=0.95)
    est = FakeEstimator([low, high])
    picked = pick_best_sample([(tmp_path / "a.jpg", "x"), (tmp_path / "b.jpg", "x")], est)
    assert picked is not None
    image, landmarks, path = picked
    assert path == tmp_path / "b.jpg"
    assert torso_visibility(landmarks) == 0.95
    assert image.shape == (90, 120, 3)


def test_pick_best_sample_returns_none_when_no_detection(tmp_path):
    _write_image(tmp_path / "a.jpg")
    est = FakeEstimator([None])
    assert pick_best_sample([(tmp_path / "a.jpg", "x")], est) is None


def test_pick_detection_sample_prefers_correct_label(tmp_path):
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b.jpg")
    wrong = SimpleNamespace(label="balasana", confidence=0.9)
    right = SimpleNamespace(label="marjaryasana", confidence=0.8)
    est = FakeEstimator([_landmarks(vis=0.95), _landmarks(vis=0.6)])
    clf = FakeClassifier([wrong, right])
    picked = pick_detection_sample(
        [(tmp_path / "a.jpg", "x"), (tmp_path / "b.jpg", "x")], est, clf, "marjaryasana",
    )
    assert picked is not None
    _, _, result = picked
    assert result.label == "marjaryasana"


def test_pick_detection_sample_falls_back_to_best_visibility(tmp_path):
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b.jpg")
    wrong_a = SimpleNamespace(label="balasana", confidence=0.9)
    wrong_b = SimpleNamespace(label="tadasana", confidence=0.7)
    est = FakeEstimator([_landmarks(vis=0.5), _landmarks(vis=0.9)])
    clf = FakeClassifier([wrong_a, wrong_b])
    picked = pick_detection_sample(
        [(tmp_path / "a.jpg", "x"), (tmp_path / "b.jpg", "x")], est, clf, "marjaryasana",
    )
    assert picked is not None
    image, landmarks, result = picked
    assert torso_visibility(landmarks) == 0.9
    assert result.label == "tadasana"


def test_draw_overlay_draws_skeleton_and_banner():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lm = _landmarks(vis=0.9)
    out = draw_overlay(img, lm, "tadasana", 0.93)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    joint = np.all(out == (80, 220, 255), axis=-1)
    line = np.all(out == (120, 220, 80), axis=-1)
    assert joint.sum() > 0
    assert line.sum() > 0
    banner = out[0:25, 0:100]
    assert np.any(np.all(banner == (255, 255, 255), axis=-1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_generate_readme_images.py -v` (from repo root)
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.generate_readme_images'`

- [ ] **Step 3: Implement the helpers**

`scripts/generate_readme_images.py`:

```python
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.classifier import load_default
from backend.app.features import extract_features
from backend.app.pose_estimator import PoseEstimator
from backend.training.download_data import find_images

TORSO_KEYPOINT_INDICES = [11, 12, 23, 24]
MAX_SCAN_PER_POSE = 25
POSE_KEYS = [
    "tadasana", "adho_mukha_svanasana", "virabhadrasana_i", "virabhadrasana_ii",
    "vrksasana", "bhujangasana", "balasana", "marjaryasana",
]
DETECTION_KEYS = ["virabhadrasana_ii", "adho_mukha_svanasana", "vrksasana", "marjaryasana"]

POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
    [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
    [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
]


def torso_visibility(landmarks):
    return float(np.min(np.asarray(landmarks)[TORSO_KEYPOINT_INDICES, 3]))


def resize_to_width(image, max_width=640):
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def pick_best_sample(candidates, estimator, max_scan=MAX_SCAN_PER_POSE):
    best = None
    for path, _label in list(candidates)[:max_scan]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        landmarks = estimator.estimate(image)
        if landmarks is None:
            continue
        score = torso_visibility(landmarks)
        if best is None or score > best[0]:
            best = (score, image, landmarks, path)
    if best is None:
        return None
    _, image, landmarks, path = best
    return resize_to_width(image), landmarks, path


def pick_detection_sample(candidates, estimator, classifier, key,
                          max_scan=MAX_SCAN_PER_POSE):
    best_any = None
    best_correct = None
    for path, _label in list(candidates)[:max_scan]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        landmarks = estimator.estimate(image)
        if landmarks is None:
            continue
        score = torso_visibility(landmarks)
        feats, vis = extract_features(landmarks)
        result = classifier.predict(feats, vis)
        entry = (score, image, landmarks, result, path)
        if best_any is None or score > best_any[0]:
            best_any = entry
        if result.label == key and (best_correct is None or score > best_correct[0]):
            best_correct = entry
    chosen = best_correct or best_any
    if chosen is None:
        return None
    _, image, landmarks, result, _path = chosen
    return resize_to_width(image), landmarks, result


def draw_overlay(image, landmarks, label, confidence):
    out = image.copy()
    h, w = out.shape[:2]
    for a, b in POSE_CONNECTIONS:
        p1, p2 = landmarks[a], landmarks[b]
        cv2.line(out, (int(p1[0] * w), int(p1[1] * h)),
                 (int(p2[0] * w), int(p2[1] * h)), (120, 220, 80), 3)
    for lm in landmarks:
        cv2.circle(out, (int(lm[0] * w), int(lm[1] * h)), 4, (80, 220, 255), -1)
    text = f"{label}  {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(out, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `backend/.venv/bin/python -m pytest backend/tests/test_generate_readme_images.py -v` (from repo root)
Expected: 7 passed

- [ ] **Step 5: Run the full backend suite to check for regressions**

Run: `backend/.venv/bin/python -m pytest backend/tests -v` (from repo root)
Expected: 47 passed (40 existing + 7 new)

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_readme_images.py backend/tests/test_generate_readme_images.py
git commit -m "feat(scripts): add README image selection and overlay helpers"
```

---

## Task 2: CLI Entry Point and Image Generation

**Files:**
- Modify: `scripts/generate_readme_images.py` (append `main()` and `__main__` guard)
- Create (generated, committed): `docs/images/poses/<key>.jpg` ×8, `docs/images/detection/<key>.jpg` ×4

**Interfaces:**
- Consumes: Task 1 helpers, `find_images("data/raw")`, `PoseEstimator(static_image_mode=True)`, `load_default()`.
- Produces: `main(raw_dir="data/raw", out_dir="docs/images")` writing the 12 JPGs at quality 85; exit code 0 on full success, non-zero with message when data or model artifacts are missing (poses grid still written before the model check fails).

- [ ] **Step 1: Append `main()` to `scripts/generate_readme_images.py`**

```python
def main(raw_dir="data/raw", out_dir="docs/images"):
    pairs_by_pose = {}
    for path, label in find_images(raw_dir):
        pairs_by_pose.setdefault(label, []).append((path, label))
    missing = [k for k in POSE_KEYS if not pairs_by_pose.get(k)]
    if missing:
        raise SystemExit(f"No training images found under {raw_dir} for: {missing}")

    estimator = PoseEstimator(static_image_mode=True)
    poses_dir = Path(out_dir) / "poses"
    detection_dir = Path(out_dir) / "detection"
    poses_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)

    for key in POSE_KEYS:
        picked = pick_best_sample(pairs_by_pose[key], estimator)
        if picked is None:
            raise SystemExit(f"MediaPipe found no usable sample for {key}")
        image, _landmarks, path = picked
        cv2.imwrite(str(poses_dir / f"{key}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"poses/{key}.jpg  <-  {path}")

    classifier = load_default()
    if classifier is None:
        raise SystemExit(
            "Model artifacts missing under backend/models/ "
            "(pose_classifier.joblib / label_encoder.pkl) — cannot generate detection images."
        )

    for key in DETECTION_KEYS:
        picked = pick_detection_sample(pairs_by_pose[key], estimator, classifier, key)
        if picked is None:
            raise SystemExit(f"MediaPipe found no usable sample for {key}")
        image, landmarks, result = picked
        if result.label != key:
            print(f"warning: best sample for {key} predicted {result.label}")
        annotated = draw_overlay(image, landmarks, result.label, result.confidence)
        cv2.imwrite(str(detection_dir / f"{key}.jpg"), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"detection/{key}.jpg  predicted={result.label} conf={result.confidence:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script to generate all 12 images**

Run: `backend/.venv/bin/python scripts/generate_readme_images.py` (from repo root; allow 2–5 minutes for ~300 MediaPipe calls)
Expected: 8 `poses/*.jpg` lines then 4 `detection/*.jpg` lines, exit 0.

- [ ] **Step 3: Verify the outputs**

```bash
ls docs/images/poses docs/images/detection
```

Expected: 8 files in `poses/`, 4 in `detection/`, each a non-empty JPEG. Visually spot-check that skeletons align with bodies and the banner text shows label + confidence.

- [ ] **Step 4: Re-run to verify idempotency**

Run the script again; expected: same selections (deterministic — `find_images` yields sorted paths), files overwritten, exit 0, no errors.

- [ ] **Step 5: Run full backend suite**

Run: `backend/.venv/bin/python -m pytest backend/tests -v` (from repo root)
Expected: 47 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_readme_images.py docs/images
git commit -m "feat(docs): generate pose catalog and detection overlay images"
```

---

## Task 3: README Sections

**Files:**
- Modify: `README.md` (insert two sections after the design-doc link line, before `## Prerequisites`)

**Interfaces:**
- Consumes: the 12 committed images from Task 2 (relative paths `docs/images/...`).
- Produces: rendered "The 8 poses" grid and "Detection in action" sections on GitHub.

- [ ] **Step 1: Edit `README.md`**

Insert after the line `See \`docs/superpowers/specs/2026-08-12-yoga-pose-detector-design.md\` for the full design.` and before `## Prerequisites`:

```markdown
## The 8 poses

| | | | |
|---|---|---|---|
| **Mountain**<br>Tadasana | **Downward-Facing Dog**<br>Adho Mukha Svanasana | **Warrior I**<br>Virabhadrasana I | **Warrior II**<br>Virabhadrasana II |
| ![](docs/images/poses/tadasana.jpg) | ![](docs/images/poses/adho_mukha_svanasana.jpg) | ![](docs/images/poses/virabhadrasana_i.jpg) | ![](docs/images/poses/virabhadrasana_ii.jpg) |
| **Tree**<br>Vrksasana | **Cobra**<br>Bhujangasana | **Child's Pose**<br>Balasana | **Cat**<br>Marjaryasana |
| ![](docs/images/poses/vrksasana.jpg) | ![](docs/images/poses/bhujangasana.jpg) | ![](docs/images/poses/balasana.jpg) | ![](docs/images/poses/marjaryasana.jpg) |

## Detection in action

MediaPipe Pose landmarks (green skeleton, yellow joints) with the RandomForest
classifier's prediction — the same pipeline `/api/predict` runs on every webcam
frame.

| | | |
|---|---|---|
| ![](docs/images/detection/virabhadrasana_ii.jpg) | ![](docs/images/detection/adho_mukha_svanasana.jpg) | ![](docs/images/detection/vrksasana.jpg) |
| ![](docs/images/detection/marjaryasana.jpg) | | |

Regenerate all images with:

```bash
backend/.venv/bin/python scripts/generate_readme_images.py
```
```

- [ ] **Step 2: Verify links resolve**

```bash
grep -o 'docs/images/[a-z]*/[a-z_]*\.jpg' README.md | sort -u | while read f; do test -f "$f" || echo "MISSING: $f"; done
```

Expected: no output (all 12 referenced files exist).

- [ ] **Step 3: Verify the full test suite one final time**

Run: `backend/.venv/bin/python -m pytest backend/tests -v` (from repo root)
Expected: 47 passed

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): add pose catalog and detection images"
```

---

## Self-Review Notes

**Spec coverage check:**
- §2 assets (`docs/images/poses/<key>.jpg` ×8, `docs/images/detection/<key>.jpg` ×4) → Tasks 2–3.
- §2 script behaviors (visibility-based selection, 640 px cap, same POSE_CONNECTIONS as frontend, label+confidence burned in, repo-root execution) → Task 1 (helpers, tested) + Task 2 (main).
- §2 README sections after intro → Task 3.
- §3 constraints: committed script (Task 1–2 commits), committed JPGs (Task 2 commit), clear error when model missing after poses grid (Task 2 `main` order), no app-code changes (no task touches `backend/app/**`).
- §4 testing: script run + idempotent re-run (Task 2 steps 2–4), file existence check (Task 3 step 2), README renders with relative paths to committed files (Task 3 step 2 + commit).
- Detection keys match spec verbatim.

**Placeholder scan:** No TBD/TODO/"add error handling" patterns; every step has runnable code or commands.

**Type/name consistency:** `torso_visibility`, `resize_to_width`, `pick_best_sample`, `pick_detection_sample`, `draw_overlay`, `main` used identically across Tasks 1–2; test imports match; `PredictionResult` fields (`label`, `confidence`) match `backend/app/classifier.py:24-37`; `estimate` returns `(33, 4)` array or `None` per `backend/app/pose_estimator.py:15-30`.

**Known risk:** A detection sample may classify as a different pose than its folder label; `pick_detection_sample` prefers correct predictions and `main` prints a warning on fallback, so a wrong-label showcase image is visible at generation time and can be re-picked by raising `MAX_SCAN_PER_POSE`.
