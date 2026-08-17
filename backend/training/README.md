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

## Interim model provenance (committed artifacts)

The currently committed model is an **interim 3-pose model** trained from
`AdityasArsenal/Yoga-pose-Data-Set` on HuggingFace (public mirror of a 5-class
Kaggle yoga dataset; classes: downdog, goddess, plank, tree, warrior2). Only the
3 classes matching v1 poses were kept, filtered into:

```
data/raw/Downward Dog/    → adho_mukha_svanasana
data/raw/Tree/            → vrksasana
data/raw/Warrior II/      → virabhadrasana_ii
```

- 214 images → 195 passed MediaPipe visibility filtering
- Val accuracy: 0.947 (all three models; RF selected on tie-break)
- Retrain with the full 8-pose Kaggle dataset (107 classes) once
  `~/.kaggle/kaggle.json` is available.
- Note: when evaluating with unrelated still images, use
  `PoseEstimator(static_image_mode=True)`; the serving singleton uses tracking
  mode (`False`), which assumes temporally related frames (webcam video).
