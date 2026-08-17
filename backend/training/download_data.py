import os
from pathlib import Path

CANONICAL_KEYS = [
    "tadasana", "adho_mukha_svanasana", "virabhadrasana_i", "virabhadrasana_ii",
    "vrksasana", "bhujangasana", "balasana", "marjaryasana",
]

SYNONYM_MAP = {
    "tadasana": "tadasana",
    "mountain": "tadasana",
    "mountain pose": "tadasana",
    "adho mukha svanasana": "adho_mukha_svanasana",
    "downward-facing dog": "adho_mukha_svanasana",
    "downward dog": "adho_mukha_svanasana",
    "down dog": "adho_mukha_svanasana",
    "virabhadrasana i": "virabhadrasana_i",
    "virabhadrasana 1": "virabhadrasana_i",
    "warrior i": "virabhadrasana_i",
    "warrior 1": "virabhadrasana_i",
    "virabhadrasana ii": "virabhadrasana_ii",
    "virabhadrasana 2": "virabhadrasana_ii",
    "warrior ii": "virabhadrasana_ii",
    "warrior 2": "virabhadrasana_ii",
    "vrksasana": "vrksasana",
    "tree": "vrksasana",
    "tree pose": "vrksasana",
    "bhujangasana": "bhujangasana",
    "cobra": "bhujangasana",
    "cobra pose": "bhujangasana",
    "balasana": "balasana",
    "child's pose": "balasana",
    "child pose": "balasana",
    "marjaryasana": "marjaryasana",
    "cat": "marjaryasana",
    "cat pose": "marjaryasana",
}


def relabel(label: str) -> str | None:
    if not label:
        return None
    key = label.strip().lower()
    return SYNONYM_MAP.get(key)


def find_images(raw_dir):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label_source = path.parent.name
        canonical = relabel(label_source)
        if canonical is None:
            continue
        yield (path, canonical)


YOGA82_META_URLS = [
    "https://raw.githubusercontent.com/manish7suthar/Yoga-82-dataset/master/Yoga-82/yoga_dataset_links/3personWarriorII.txt",
]


def main(raw_dir: str = "data/raw"):
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Manual step: download Yoga-82 from https://github.com/manish7suthar/Yoga-82-dataset")
    print(f"Manual step: download Kaggle 'Yoga Posture Dataset' via `kaggle datasets download -d shrutisaxena/yoga-pose-image-classification-dataset`")
    print(f"Place extracted folders under {raw_dir}/. Expected structure: <raw_dir>/<Pose Label>/<image>.jpg")


if __name__ == "__main__":
    main()
