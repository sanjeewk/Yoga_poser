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
