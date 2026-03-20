from pathlib import Path


def test_qm9_release_entrypoint_and_dev_secondary_helper_are_both_documented():
    audit = Path("docs/reproducibility/dataset-cold-start-audit.md").read_text(encoding="utf-8")
    assert "data/qm9/prepare_qm9_raw.py" in audit
    assert "data/qm9/process_qm9_dataset.py" in audit
    assert "secondary script" in audit


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
    guide = Path("docs/reproducibility/paper-dataset-cold-start-guide.md").read_text(encoding="utf-8")
    assert "derived from `qm9`" in readme
    assert "`qm9test`" in guide
