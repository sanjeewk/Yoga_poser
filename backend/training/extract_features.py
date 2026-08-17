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
    estimator = PoseEstimator(static_image_mode=True)
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
