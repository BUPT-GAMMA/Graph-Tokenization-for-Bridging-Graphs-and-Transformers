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


def test_qm9test_script_supports_explicit_original_indices_override():
    script_text = (Path("data/qm9test") / "create_qm9test_dataset.py").read_text(encoding="utf-8")
    assert "original_indices_path" in script_text
    assert "source_dir" in script_text
    assert "src.data.qm9_loader" not in script_text


def test_qm9test_existing_metadata_is_sufficient_to_replay_current_subset_splits():
    meta = json.loads((Path("data/qm9test") / "metadata.json").read_text(encoding="utf-8"))
    selected = meta["original_indices"]
    from sklearn.model_selection import train_test_split

    train_idx, temp_idx = train_test_split(
        selected,
        train_size=meta["train_samples"],
        random_state=meta["random_state"],
        shuffle=True,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=0.5,
        random_state=meta["random_state"],
        shuffle=True,
    )
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)

    replay = {
        "train": [selected.index(i) for i in train_idx],
        "val": [selected.index(i) for i in val_idx],
        "test": [selected.index(i) for i in test_idx],
    }
    for split in ["train", "val", "test"]:
        current = json.loads((Path("data/qm9test") / f"{split}_index.json").read_text(encoding="utf-8"))
        assert current == replay[split]
