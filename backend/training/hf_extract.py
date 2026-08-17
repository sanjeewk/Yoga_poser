import argparse
from pathlib import Path

import pyarrow.parquet as pq

LABEL_NAMES = {0: "downdog", 1: "goddess", 2: "plank", 3: "tree", 4: "warrior2"}
KEEP = {"downdog": "Downward Dog", "tree": "Tree", "warrior2": "Warrior II"}

EXPECTED_SIZES = {
    "test_0000.parquet": 79578584,
    "train_0000.parquet": 292454872,
    "train_0001.parquet": 158821163,
}


def extract(parquet_path: Path, out_root: Path, prefix: str) -> int:
    table = pq.read_table(parquet_path)
    written = 0
    for i in range(table.num_rows):
        label = LABEL_NAMES[table.column("label")[i].as_py()]
        if label not in KEEP:
            continue
        img = table.column("image")[i].as_py()
        out_dir = out_root / KEEP[label]
        out_dir.mkdir(parents=True, exist_ok=True)
        name = img.get("path") or f"{label}{i}.jpg"
        out = out_dir / f"{prefix}{i:04d}_{Path(name).name}"
        if not out.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            out = out.with_suffix(".jpg")
        out.write_bytes(img["bytes"])
        written += 1
    return written


def main(parquet_dir: str, out_dir: str, only_complete: bool = True):
    parquet_dir = Path(parquet_dir)
    out_root = Path(out_dir)
    total = 0
    for pq_file in sorted(parquet_dir.glob("*.parquet")):
        expected = EXPECTED_SIZES.get(pq_file.name)
        if only_complete and expected and pq_file.stat().st_size != expected:
            print(f"skipping {pq_file.name}: {pq_file.stat().st_size} bytes "
                  f"!= expected {expected} (incomplete download)")
            continue
        prefix = pq_file.name.split("_")[0][:2]
        n = extract(pq_file, out_root, prefix)
        print(f"{pq_file.name}: wrote {n} images")
        total += n
    print(f"total written: {total}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default="data/hf_parquet")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--include-incomplete", action="store_true")
    args = p.parse_args()
    main(args.parquet_dir, args.out, not args.include_incomplete)
