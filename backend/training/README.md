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

## Committed model provenance

The committed model covers **all 8 v1 poses**, trained from two public
HuggingFace datasets (no Kaggle token required):

1. `rotemvahava/yoga-poses-107` (~1.1GB parquet; mirror of the 107-class
   Kaggle yoga dataset) — extracted via `backend/training/hf107_extract.py`,
   keeping only the 8 exact Sanskrit labels (488 images).
2. `AdityasArsenal/Yoga-pose-Data-Set` (5-class) — extracted via
   `backend/training/hf_extract.py`, keeping downdog/tree/warrior2
   (1434 images).

Folders from both sources land in `data/raw/<label>/` and are unified by
`download_data.SYNONYM_MAP` (note: dataset spells tree pose "vriksasana";
canonical key is `vrksasana`).

- 1922 candidate images → 1791 passed MediaPipe visibility filtering
- Selected model: random_forest (val 0.889; GBM 0.870, SVC 0.852)
- Test top-1 0.907, top-2 0.944 (≥0.85 gate cleared)
- Batch static-mode verification: 113/125 = 0.904

To retrain from scratch, re-download the parquet files (URLs in the two
extractor scripts) and run the three pipeline commands above.
Training uses the Python BlazePose full model through
`PoseEstimator(static_image_mode=True)`. Live inference uses the equivalent
MediaPipe Pose Landmarker full model in a browser worker and sends only its
landmarks to the API. Keep both on the full model variant so the classifier's
feature distribution remains aligned.
