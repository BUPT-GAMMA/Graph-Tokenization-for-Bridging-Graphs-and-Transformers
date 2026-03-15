from pathlib import Path
import subprocess


MISSING_EXPORT_SCRIPT_PATTERNS = [
    "export_qm9.py",
    "export_zinc.py",
    "export_molhiv.py",
]


def test_cold_start_audit_marks_qm9_and_qm9test_as_secondary_not_raw_cold_start():
    audit = Path("docs/reproducibility/dataset-cold-start-audit.md").read_text(encoding="utf-8")
    assert "secondary script" in audit
    assert "`qm9`" in audit
    assert "`qm9test`" in audit
    assert "not yet a public-raw cold-start pipeline" in audit


def test_export_docs_do_not_advertise_missing_export_scripts_as_existing_tools():
    doc_paths = [
        Path("docs/guides/dataset_export_guide.md"),
        Path("docs/guides/simple_export_guide.md"),
        Path("export_system/DATASET_EXPORT_GUIDE.md"),
        Path("export_system/SIMPLE_EXPORT_GUIDE.md"),
    ]
    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8")
        for pattern in MISSING_EXPORT_SCRIPT_PATTERNS:
            assert pattern not in text, f"{doc_path} still advertises missing script {pattern}"


def test_data_preprocess_scripts_are_not_gitignored():
    script_paths = [
        "data/molhiv/preprocess_molhiv.py",
        "data/mnist_raw/prepare.py",
        "data/qm9/process_qm9_dataset.py",
        "data/qm9test/create_qm9test_dataset.py",
        "data/coildel/preprocess_coil_del.py",
    ]
    for script_path in script_paths:
        result = subprocess.run(
            ["git", "check-ignore", script_path],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, f"{script_path} is still ignored by .gitignore"
