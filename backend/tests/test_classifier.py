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
    feats = np.full(16, 30.0, dtype=np.float32)
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
