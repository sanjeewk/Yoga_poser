import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

FEATURE_COLS = [f"f{i}" for i in range(16)]
VIS_COLS = [f"v{i}" for i in range(33)]
ANGLE_IDX = list(range(12))
DIST_IDX = list(range(12, 16))
TOP1_TARGET = 0.85


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


def _top2_accuracy(clf, X: np.ndarray, y: np.ndarray) -> float:
    proba = clf.predict_proba(X)
    top2_idx = np.argsort(proba, axis=1)[:, -2:]
    top2_labels = clf.classes_[top2_idx]
    return float(np.mean(np.any(top2_labels == y[:, None], axis=1)))


def _save_confusion_matrix(y_true, y_pred, classes, plot_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    return True


def train(df: pd.DataFrame, out_dir: Path, plot_path: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()

    X, y = balance_classes(X, y)
    if len(set(y)) < 2:
        raise ValueError("Need at least 2 classes to train")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15 / 0.85, random_state=42, stratify=y_dev,
    )

    X_train_aug = augment(X_train)
    y_train_aug = np.tile(y_train, 4)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42,
        ),
        "svc_rbf": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, random_state=42),
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    val_accuracies = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_aug, y_train_aug)
        fitted[name] = model
        val_accuracies[name] = float(accuracy_score(y_val, model.predict(X_val)))

    selected_name = None
    for name in models:
        if selected_name is None or val_accuracies[name] > val_accuracies[selected_name]:
            selected_name = name
    clf = fitted[selected_name]

    y_pred = clf.predict(X_test)
    top1 = accuracy_score(y_test, y_pred)
    top2 = _top2_accuracy(clf, X_test, y_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    if plot_path is not None:
        _save_confusion_matrix(y_test, y_pred, list(clf.classes_), Path(plot_path))

    if top1 < TOP1_TARGET:
        print(f"WARNING: top-1 accuracy {top1:.3f} below 0.85 target")

    le = LabelEncoder().fit(df["label"])
    joblib.dump(clf, out_dir / "pose_classifier.joblib")
    joblib.dump(le, out_dir / "label_encoder.pkl")

    return {
        "val_accuracies": val_accuracies,
        "selected_model": selected_name,
        "test_top1_accuracy": float(top1),
        "test_top2_accuracy": float(top2),
        "test_classification_report": report,
        "n_train_samples": int(len(X_train_aug)),
        "n_test_samples": int(len(X_test)),
        "classes": list(le.classes_),
    }


def main(features_path: str = "data/features.parquet",
         out_dir: str = "backend/models"):
    df = pd.read_parquet(features_path)
    Path("data").mkdir(exist_ok=True)
    metrics = train(df, Path(out_dir), Path("data/confusion_matrix.png"))
    print(f"Selected model: {metrics['selected_model']}")
    for name, acc in metrics["val_accuracies"].items():
        print(f"Val accuracy ({name}): {acc:.3f}")
    print(f"Top-1 accuracy: {metrics['test_top1_accuracy']:.3f}")
    print(f"Top-2 accuracy: {metrics['test_top2_accuracy']:.3f}")
    print(f"Classes: {metrics['classes']}")
    print(f"Wrote {out_dir}/pose_classifier.joblib and label_encoder.pkl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features.parquet")
    p.add_argument("--out", default="backend/models")
    args = p.parse_args()
    main(args.features, args.out)
