import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.features import FEATURE_NAMES

ANGLE_FEATURE_NAMES = FEATURE_NAMES[:12]
FEATURE_COLS = [f"f{i}" for i in range(12)]


def build_templates(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    grouped = df.groupby("label")[FEATURE_COLS].median()
    out = {}
    for label, row in grouped.iterrows():
        out[str(label)] = {
            ANGLE_FEATURE_NAMES[i]: float(row[col])
            for i, col in enumerate(FEATURE_COLS)
        }
    return out


def main(features_path: str = "data/features.parquet",
         out_path: str = "backend/models/pose_templates.json"):
    df = pd.read_parquet(features_path)
    templates = build_templates(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"Wrote {out_path} for {len(templates)} poses")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features.parquet")
    p.add_argument("--out", default="backend/models/pose_templates.json")
    args = p.parse_args()
    main(args.features, args.out)
