from pathlib import Path

import dgl
import torch

from data.qm9.prepare_qm9_raw import _ordered_graph_signature, _reorder_to_reference


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
    assert "split_source_dir" in text
    assert "use_env_proxy" in text
    assert "output_dir" in text
    assert "reference_data_pkl" in text
    assert "reference_smiles_dir" in text


def test_qm9_raw_script_is_not_secondary_qm9_processed_export():
    text = Path("data/qm9/prepare_qm9_raw.py").read_text(encoding="utf-8")
    assert 'data/qm9_processed' not in text


def test_qm9_raw_script_encodes_current_attr_contract():
    text = Path("data/qm9/prepare_qm9_raw.py").read_text(encoding="utf-8")
    assert "QM9EdgeDataset" in text
    assert "ATTR_DIM = 11" in text
    assert "EDGE_ATTR_DIM = 4" in text


def _make_graph(node_rows: list[list[float]], pos_rows: list[list[float]]) -> dgl.DGLGraph:
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    graph.ndata["attr"] = torch.tensor(node_rows, dtype=torch.float32)
    graph.ndata["pos"] = torch.tensor(pos_rows, dtype=torch.float32)
    graph.edata["edge_attr"] = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
    return graph


def test_ordered_graph_signature_distinguishes_duplicate_property_candidates():
    g1 = _make_graph([[0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], [[0, 0, 0], [1, 0, 0]])
    g2 = _make_graph([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 1]], [[1, 0, 0], [0, 0, 0]])
    assert _ordered_graph_signature(g1) != _ordered_graph_signature(g2)


def test_reorder_to_reference_prefers_exact_graph_signature_for_duplicate_properties(tmp_path):
    props = {
        "mu": 1.0,
        "alpha": 2.0,
        "homo": 3.0,
        "lumo": 4.0,
        "gap": 5.0,
        "r2": 6.0,
        "zpve": 7.0,
        "u0": 8.0,
        "u298": 9.0,
        "h298": 10.0,
        "g298": 11.0,
        "cv": 12.0,
        "u0_atom": 13.0,
        "u298_atom": 14.0,
        "h298_atom": 15.0,
        "g298_atom": 16.0,
    }
    reference_graph = _make_graph(
        [[0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
        [[0, 0, 0], [1, 0, 0]],
    )
    swapped_graph = _make_graph(
        [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 1]],
        [[1, 0, 0], [0, 0, 0]],
    )
    reference_path = tmp_path / "reference.pkl"
    import pickle

    with reference_path.open("wb") as f:
        pickle.dump([(reference_graph, props)], f, protocol=pickle.HIGHEST_PROTOCOL)

    reordered_samples, _ = _reorder_to_reference(
        [(swapped_graph, props), (reference_graph, props)],
        {"smiles_1": ["a", "b"], "smiles_2": ["a", "b"], "smiles_3": ["a", "b"], "smiles_4": ["a", "b"]},
        reference_path,
        reorder_smiles=True,
    )
    assert torch.equal(reordered_samples[0][0].ndata["attr"], reference_graph.ndata["attr"])
    assert torch.equal(reordered_samples[0][0].ndata["pos"], reference_graph.ndata["pos"])
