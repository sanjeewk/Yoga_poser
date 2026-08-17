import numpy as np
import pandas as pd
from backend.training.build_templates import build_templates
from backend.app.features import FEATURE_NAMES


def _synthetic_df():
    rng = np.random.default_rng(0)
    rows = []
    for label in ["tadasana", "vrksasana"]:
        for _ in range(20):
            row = {"label": label, "path": "x"}
            for i in range(16):
                row[f"f{i}"] = float(rng.uniform(0, 180))
            for i in range(33):
                row[f"v{i}"] = 0.9
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_templates_returns_dict_per_pose():
    df = _synthetic_df()
    templates = build_templates(df)
    assert set(templates.keys()) == {"tadasana", "vrksasana"}
    for pose, feats in templates.items():
        for name in FEATURE_NAMES[:12]:
            assert name in feats
            assert isinstance(feats[name], float)


def test_build_templates_excludes_distance_features():
    df = _synthetic_df()
    templates = build_templates(df)
    assert "wrist_to_wrist" not in templates["tadasana"]
