from pathlib import Path


COLD_START_SCRIPT_PATHS = [
    "data/code2/preprocess_code2.py",
    "data/coildel/preprocess_coil_del.py",
    "data/colors3/preprocess_colors3.py",
    "data/dblp/preprocess_dblp_v1.py",
    "data/dd/preprocess_dd.py",
    "data/mnist_raw/prepare.py",
    "data/molhiv/preprocess_molhiv.py",
    "data/mutagenicity/preprocess_mutagenicity.py",
    "data/peptides_func/prepare_lrgb_data.py",
    "data/peptides_struct/prepare_lrgb_data.py",
    "data/proteins/preprocess_proteins.py",
    "data/synthetic/preprocess_synthetic.py",
    "data/twitter/preprocess_twitter_real_graph_partial.py",
]

SECONDARY_SCRIPT_PATHS = [
    "data/qm9/process_qm9_dataset.py",
    "data/qm9test/create_qm9test_dataset.py",
]

PARTIAL_SCRIPT_PATHS = [
    "data/mnist/convert_mnist_to_dgl.py",
]


def test_expected_cold_start_preprocess_scripts_exist():
    for script_path in COLD_START_SCRIPT_PATHS:
        assert Path(script_path).exists(), f"missing cold-start script: {script_path}"


def test_secondary_and_partial_scripts_exist_for_documented_gaps():
    for script_path in SECONDARY_SCRIPT_PATHS + PARTIAL_SCRIPT_PATHS:
        assert Path(script_path).exists(), f"missing documented non-primary script: {script_path}"


def test_audit_doc_mentions_secondary_and_partial_statuses():
    audit = Path("docs/reproducibility/dataset-cold-start-audit.md").read_text(encoding="utf-8")
    assert "secondary scripts" in audit
    assert "Partially traceable" in audit
    assert "qm9test" in audit
    assert "mnist" in audit
