# README Visuals — Design

**Status:** Approved
**Date:** 2026-08-24
**Authors:** brainstorming session (sanjeew + opencode)

## 1. Goal

Add pictures to `README.md` showing (a) each of the 8 supported yoga poses and (b) pose detection in action — MediaPipe skeleton overlay plus the classifier's predicted label and confidence rendered onto sample images.

## 2. Approach

A committed, re-runnable script generates all images from the local training data; no manual screenshotting.

### Assets

New directory `docs/images/`:

- `docs/images/poses/<key>.jpg` — one representative sample per pose (8 total), selected from `data/raw/` by highest torso landmark visibility.
- `docs/images/detection/<key>.jpg` — 4 of the 8 poses annotated with:
  - green skeleton drawn using the same `POSE_CONNECTIONS` topology as `PoseOverlay.jsx`
  - the predicted label + confidence (e.g. `virabhadrasana_ii 0.93`) burned in as text at the top-left
  - produced by running `PoseEstimator` → `extract_features` → `PoseClassifier` (the committed model) → `cv2` drawing

### Script

`scripts/generate_readme_images.py` (committed):

1. Walk `data/raw/` via `backend.training.download_data.find_images`
2. For each of the 8 canonical keys, score candidates with MediaPipe and keep the highest-visibility frame (resize to max 640 px wide for a compact repo)
3. Copy the 8 winners to `docs/images/poses/`
4. Run the full predict path on 4 chosen keys and save annotated copies to `docs/images/detection/`

Detection keys for v1: `virabhadrasana_ii`, `adho_mukha_svanasana`, `vrksasana`, `marjaryasana` (one from each visual family: lunge, inversion, balance, kneeling).

### README changes

Two new sections inserted after the intro paragraph:

- **The 8 poses** — markdown table/grid of pose images with display names
- **Detection in action** — annotated overlay images

## 3. Constraints

- Generated JPGs are committed (small; one per pose + 4 annotated).
- Script imports `backend.app` modules, so it must run from the repo root (same as training commands).
- The classifier must load from `backend/models/pose_classifier.joblib`; if missing, the script errors with a clear message (poses grid can still generate, detection images cannot).
- No app code changes; existing tests untouched.

## 4. Testing

- Run `python scripts/generate_readme_images.py` from repo root; verify 12 files created.
- Re-run; verify idempotent (same selection, files overwritten cleanly).
- Verify README renders on GitHub (relative paths, images committed).
