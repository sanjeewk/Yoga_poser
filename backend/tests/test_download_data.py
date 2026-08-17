from pathlib import Path
from backend.training.download_data import relabel, SYNONYM_MAP, find_images


def test_relabel_known_synonyms():
    assert relabel("Adho Mukha Svanasana") == "adho_mukha_svanasana"
    assert relabel("Downward-Facing Dog") == "adho_mukha_svanasana"
    assert relabel("Downward Dog") == "adho_mukha_svanasana"
    assert relabel("Warrior I") == "virabhadrasana_i"
    assert relabel("Warrior II") == "virabhadrasana_ii"
    assert relabel("Child's Pose") == "balasana"
    assert relabel("Balasana") == "balasana"


def test_relabel_unknown_returns_none():
    assert relabel("Sirsasana") is None
    assert relabel("") is None


def test_find_images_filters_to_canonical(tmp_path):
    raw = tmp_path / "raw"
    (raw / "Adho Mukha Svanasana").mkdir(parents=True)
    (raw / "Adho Mukha Svanasana" / "a.jpg").write_bytes(b"x")
    (raw / "Sirsasana").mkdir(parents=True)
    (raw / "Sirsasana" / "b.jpg").write_bytes(b"x")
    pairs = list(find_images(tmp_path / "raw"))
    labels = {label for _, label in pairs}
    assert labels == {"adho_mukha_svanasana"}
