from pathlib import Path


def test_qm9_raw_script_exists_and_targets_current_baseline_layout():
    script_path = Path("data/qm9/prepare_qm9_raw.py")
    assert script_path.exists(), "qm9 raw cold-start script is missing"
    text = script_path.read_text(encoding="utf-8")
    assert 'data/qm9' in text or 'Path("data") / "qm9"' in text
    assert "train_index.json" in text
    assert "val_index.json" in text
    assert "test_index.json" in text
    assert "smiles_1_direct.txt" in text
    assert "smiles_4_addhs_explicit_h.txt" in text
    assert "attr" in text
    assert "edge_attr" in text
    assert "pos" in text


def test_qm9_raw_script_is_not_secondary_qm9_processed_export():
    text = Path("data/qm9/prepare_qm9_raw.py").read_text(encoding="utf-8")
    assert 'data/qm9_processed' not in text


def test_qm9_raw_script_encodes_current_attr_contract():
    text = Path("data/qm9/prepare_qm9_raw.py").read_text(encoding="utf-8")
    assert "ATOM_ORDER = ['H', 'C', 'N', 'O', 'F']" in text or 'ATOM_ORDER = ["H", "C", "N", "O", "F"]' in text
    assert "11" in text
    assert "4" in text
