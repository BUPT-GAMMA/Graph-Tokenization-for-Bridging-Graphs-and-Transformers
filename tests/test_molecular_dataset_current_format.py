import pickle
from pathlib import Path

import torch


QM9_KEYS = {
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
    "u0_atom",
    "u298_atom",
    "h298_atom",
    "g298_atom",
}


def _load_first_sample(dataset_name: str):
    data_path = Path("data") / dataset_name / "data.pkl"
    with data_path.open("rb") as f:
        data = pickle.load(f)
    return data[0]


def test_qm9_current_format_matches_documented_baseline():
    graph, label = _load_first_sample("qm9")
    assert isinstance(label, dict)
    assert set(label.keys()) == QM9_KEYS
    assert "pos" in graph.ndata
    assert "attr" in graph.ndata
    assert "edge_attr" in graph.edata
    assert tuple(graph.ndata["pos"].shape[1:]) == (3,)
    assert tuple(graph.ndata["attr"].shape[1:]) == (11,)
    assert tuple(graph.edata["edge_attr"].shape[1:]) == (4,)


def test_zinc_current_format_matches_documented_baseline():
    graph, label = _load_first_sample("zinc")
    assert isinstance(label, torch.Tensor)
    assert label.shape == (1,)
    assert graph.ndata["feat"].dtype == torch.int64
    assert graph.edata["feat"].dtype == torch.int64
    assert graph.ndata["feat"].dim() == 1
    assert graph.edata["feat"].dim() == 1


def test_aqsol_current_format_matches_documented_baseline():
    graph, label = _load_first_sample("aqsol")
    assert isinstance(label, float)
    assert graph.ndata["feat"].dtype == torch.int64
    assert graph.edata["feat"].dtype == torch.int64
    assert graph.ndata["feat"].dim() == 1
    assert graph.edata["feat"].dim() == 1
