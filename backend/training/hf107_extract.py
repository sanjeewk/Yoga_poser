import argparse
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

LABEL_NAMES_URL = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=rotemvahava%2Fyoga-poses-107&config=default&split=train"
)
TARGET_LABELS = [
    "tadasana", "adho mukha svanasana", "virabhadrasana i",
    "virabhadrasana ii", "vriksasana", "bhujangasana",
    "balasana", "marjaryasana",
]
EXPECTED_SIZES = {
    "train_0000.parquet": 345477450,
    "train_0001.parquet": 342366517,
    "train_0002.parquet": 409883099,
}


def load_label_names(cache_path: Path) -> list[str]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    with urllib.request.urlopen(LABEL_NAMES_URL) as r:
        d = json.load(r)
    label_feat = [f for f in d["features"] if f["name"] == "label"][0]
    names = label_feat["type"]["names"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(names))
    return names


def extract(parquet_path: Path, out_root: Path, keep_indices: dict[int, str]) -> int:
    table = pq.read_table(parquet_path)
    written = 0
    labels = table.column("label").to_pylist()
    images = table.column("image")
    for i in range(table.num_rows):
        name = keep_indices.get(labels[i])
        if name is None:
            continue
        img = images[i].as_py()
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = img.get("path") or f"{name}{i}"
        out = out_dir / f"{parquet_path.stem}_{i:05d}_{Path(stem).name}"
        if out.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            out = out.with_suffix(".jpg")
        out.write_bytes(img["bytes"])
        written += 1
    return written


def main(parquet_dir: str, out_dir: str, names_cache: str, only_complete: bool = True):
    names = load_label_names(Path(names_cache))
    keep_indices = {names.index(n): n for n in TARGET_LABELS}
    total = 0
    for pq_file in sorted(Path(parquet_dir).glob("*.parquet")):
        expected = EXPECTED_SIZES.get(pq_file.name)
        if only_complete and expected and pq_file.stat().st_size != expected:
            print(f"skipping {pq_file.name}: incomplete "
                  f"({pq_file.stat().st_size} != {expected})")
            continue
        n = extract(pq_file, Path(out_dir), keep_indices)
        print(f"{pq_file.name}: wrote {n} images")
        total += n
    print(f"total written: {total}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default="data/hf107_parquet")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--names-cache", default="data/hf107_label_names.json")
    p.add_argument("--include-incomplete", action="store_true")
    args = p.parse_args()
    main(args.parquet_dir, args.out, args.names_cache, not args.include_incomplete)
