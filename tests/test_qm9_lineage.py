from pathlib import Path


def test_qm9_release_entrypoint_and_dev_secondary_helper_are_both_documented():
    readme = Path("scripts/dataset_conversion/README.md").read_text(encoding="utf-8")
    assert "data/qm9/process_qm9_dataset.py" in readme
    assert "`qm9`" in readme
    assert "secondary helpers" in readme


def test_qm9_secondary_script_writes_to_qm9_processed_not_current_baseline_dir():
    script_text = (Path("data/qm9") / "process_qm9_dataset.py").read_text(encoding="utf-8")
    assert 'output_dir = "data/qm9_processed"' in script_text


def test_qm9test_script_supports_explicit_original_indices_override():
    script_text = (Path("data/qm9test") / "create_qm9test_dataset.py").read_text(encoding="utf-8")
    assert "original_indices_path" in script_text
    assert "source_dir" in script_text
    assert "src.data.qm9_loader" not in script_text


def test_qm9test_docs_state_that_it_is_derived_from_qm9():
    readme = Path("README.md").read_text(encoding="utf-8")
    guide = Path("docs/datasets_overview.md").read_text(encoding="utf-8")
    assert "derived from `qm9`" in readme
    assert "## 13. `qm9test`" in guide
    assert "派生自 `qm9`" in guide
