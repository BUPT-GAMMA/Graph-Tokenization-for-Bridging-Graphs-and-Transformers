from pathlib import Path


def _assert_common_molecule_raw_script(script_path: Path) -> None:
    text = script_path.read_text(encoding="utf-8")
    assert 'data") / "zinc"' in text or "data') / 'zinc'" in text or 'data") / "aqsol"' in text or "data') / 'aqsol'" in text or "output_dir" in text
    assert "data.pkl" in text
    assert "train_index.json" in text or 'f"{split}_index.json"' in text
    assert "val_index.json" in text or 'f"{split}_index.json"' in text
    assert "test_index.json" in text or 'f"{split}_index.json"' in text
    assert "smiles_1_direct.txt" in text
    assert "smiles_4_addhs_explicit_h.txt" in text
    assert "use_env_proxy" in text
    assert "output_dir" in text


def test_zinc_raw_script_exists_and_targets_current_layout():
    script_path = Path("data/zinc/prepare_zinc_raw.py")
    assert script_path.exists(), "zinc raw cold-start script is missing"
    _assert_common_molecule_raw_script(script_path)


def test_aqsol_raw_script_exists_and_targets_current_layout():
    script_path = Path("data/aqsol/prepare_aqsol_raw.py")
    assert script_path.exists(), "aqsol raw cold-start script is missing"
    _assert_common_molecule_raw_script(script_path)
