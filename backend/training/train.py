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
