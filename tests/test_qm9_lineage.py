from pathlib import Path


def test_release_docs_state_that_qm9test_is_derived_from_qm9():
    readme = Path("README.md").read_text(encoding="utf-8")
    guide = Path("docs/reproducibility/paper-dataset-cold-start-guide.md").read_text(encoding="utf-8")

    assert "derived from `qm9`" in readme
    assert "`qm9test`" in guide
    assert "derived from `qm9`" in guide


def test_qm9test_script_supports_explicit_original_indices_override():
    script_text = (Path("data/qm9test") / "create_qm9test_dataset.py").read_text(encoding="utf-8")
    assert "original_indices_path" in script_text
    assert "source_dir" in script_text
    assert "src.data.qm9_loader" not in script_text


def test_qm9test_script_writes_metadata_and_split_indices():
    script_text = (Path("data/qm9test") / "create_qm9test_dataset.py").read_text(encoding="utf-8")
    assert "metadata.json" in script_text
    assert "train_index.json" in script_text
    assert "val_index.json" in script_text
    assert "test_index.json" in script_text
