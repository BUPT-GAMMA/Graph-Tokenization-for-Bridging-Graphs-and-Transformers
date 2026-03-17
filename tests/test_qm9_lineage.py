import json
from pathlib import Path


def test_qm9test_original_indices_come_from_qm9_non_train_partition():
    qm9_dir = Path("data/qm9")
    qm9test_meta = json.loads((Path("data/qm9test") / "metadata.json").read_text(encoding="utf-8"))

    qm9_val = set(json.loads((qm9_dir / "val_index.json").read_text(encoding="utf-8")))
    qm9_test = set(json.loads((qm9_dir / "test_index.json").read_text(encoding="utf-8")))
    qm9_train = set(json.loads((qm9_dir / "train_index.json").read_text(encoding="utf-8")))
    selected = set(qm9test_meta["original_indices"])

    assert selected
    assert selected.isdisjoint(qm9_train)
    assert selected == (selected & (qm9_val | qm9_test))


def test_qm9_secondary_script_writes_to_qm9_processed_not_current_baseline_dir():
    script_text = (Path("data/qm9") / "process_qm9_dataset.py").read_text(encoding="utf-8")
    assert 'output_dir = "data/qm9_processed"' in script_text
