from pathlib import Path
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


MISSING_EXPORT_SCRIPT_PATTERNS = [
    "export_qm9.py",
    "export_zinc.py",
    "export_molhiv.py",
]


def test_cold_start_audit_marks_zinc_and_aqsol_as_out_of_current_formal_scope():
    audit = Path("docs/reproducibility/dataset-cold-start-audit.md").read_text(encoding="utf-8")
    assert "`zinc`" in audit
    assert "`aqsol`" in audit
    assert "Out of current formal scope" in audit
    assert "experimental draft" in audit


def test_paper_scope_guide_lists_formal_and_excluded_datasets():
    guide = Path("docs/reproducibility/paper-dataset-cold-start-guide.md").read_text(encoding="utf-8")
    assert "`mnist_raw`" in guide
    assert "`qm9`" in guide
    assert "`qm9test`" in guide
    assert "`zinc`" in guide
    assert "`aqsol`" in guide
    assert "当前不纳入正式保证范围" in guide


def test_paper_scope_guide_points_to_explicit_environment_setup():
    guide = Path("docs/reproducibility/paper-dataset-cold-start-guide.md").read_text(encoding="utf-8")
    assert "environment-setup.md" in guide
    assert Path("docs/reproducibility/environment-setup.md").exists()


def test_environment_setup_doc_avoids_machine_specific_paths():
    guide = Path("docs/reproducibility/environment-setup.md").read_text(encoding="utf-8")
    assert "/home/gzy/" not in guide
    assert "/tmp/" not in guide


def test_readme_explains_that_qm9test_is_derived_not_checked_in():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "create_qm9test_dataset.py" in readme
    assert "derived from `qm9`" in readme


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


def test_build_system_declares_pybind11_for_editable_install():
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "Cold-start install needs pyproject.toml build metadata"

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    build_system = pyproject.get("build-system", {})
    requires = build_system.get("requires", [])

    assert build_system.get("build-backend") == "setuptools.build_meta"
    assert any(req.startswith("pybind11") for req in requires), (
        "Editable install must declare pybind11 in build-system.requires"
    )


def test_editable_install_egg_info_is_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "tokenizerGraph_cpp_ext.egg-info"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Editable-install egg-info directory should be ignored"


def test_internal_agent_process_docs_are_not_committed():
    assert not Path("docs/superpowers/plans/2026-03-20-repo-sync-and-repro.md").exists()
    assert not Path("docs/reproducibility/2026-03-20-repo-sync-and-repro-log.md").exists()
