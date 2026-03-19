#!/usr/bin/env python3
from __future__ import annotations

"""
QM9 raw cold-start preparation
==============================

使用 DGL 官方 `QM9EdgeDataset` 作为原始图来源，生成当前项目
`data/qm9` baseline 所需结构：

- `data.pkl`
- `train_index.json`
- `val_index.json`
- `test_index.json`
- 四份 SMILES side files

说明：
- 当前 baseline 的 graph 字段正是 `pos / attr / edge_attr` 结构，因此优先复用
  `QM9EdgeDataset`，而不是手写 CSV + NPZ 拼接逻辑。
- 若提供 `reference_data_pkl`，脚本会尝试按当前 baseline 的样本语义顺序重排。
- 若提供 `split_source_dir`，脚本会直接复用该目录下的三份 split 文件。
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from dgl.data import QM9EdgeDataset
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


QM9_EDGE_URL = "https://data.dgl.ai/dataset/qm9_edge.npz"
QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
ATTR_DIM = 11
EDGE_ATTR_DIM = 4
PROPERTY_KEYS = [
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
]
QM9EDGE_LABEL_KEYS = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atom",
    "U_atom",
    "H_atom",
    "G_atom",
    "A",
    "B",
    "C",
]
QM9EDGE_TO_BASELINE = {
    "mu": "mu",
    "alpha": "alpha",
    "homo": "homo",
    "lumo": "lumo",
    "gap": "gap",
    "r2": "r2",
    "zpve": "zpve",
    "U0": "u0",
    "U": "u298",
    "H": "h298",
    "G": "g298",
    "Cv": "cv",
    "U0_atom": "u0_atom",
    "U_atom": "u298_atom",
    "H_atom": "h298_atom",
    "G_atom": "g298_atom",
}


def dgl_graph_to_mol(graph: dgl.DGLGraph) -> Chem.Mol | None:
    try:
        u, v = graph.edges()
        node_features = graph.ndata["attr"]
        atomic_nums = [int(node_features[i][5].item()) for i in range(graph.num_nodes())]

        mol = Chem.RWMol()
        for atomic_num in atomic_nums:
            mol.AddAtom(Chem.Atom(int(atomic_num)))

        bond_map = {
            0: Chem.BondType.SINGLE,
            1: Chem.BondType.DOUBLE,
            2: Chem.BondType.TRIPLE,
            3: Chem.BondType.AROMATIC,
        }
        processed_bonds = set()
        edge_features = graph.edata["edge_attr"]
        for i in range(graph.num_edges()):
            src_node, dst_node = int(u[i].item()), int(v[i].item())
            edge_key = tuple(sorted([src_node, dst_node]))
            if edge_key in processed_bonds:
                continue
            bond_type_idx = int(torch.argmax(edge_features[i][:4]).item())
            mol.AddBond(src_node, dst_node, bond_map[bond_type_idx])
            processed_bonds.add(edge_key)

        final_mol = mol.GetMol()
        Chem.SanitizeMol(final_mol)
        return final_mol
    except Exception:
        return None


def generate_four_smiles_formats(mol: Chem.Mol) -> dict[str, str]:
    smiles_1 = Chem.MolToSmiles(mol)
    smiles_2 = Chem.MolToSmiles(mol, allHsExplicit=True)
    smiles_3 = Chem.MolToSmiles(Chem.AddHs(mol))
    smiles_4 = Chem.MolToSmiles(Chem.AddHs(mol), allHsExplicit=True)
    return {
        "smiles_1": smiles_1,
        "smiles_2": smiles_2,
        "smiles_3": smiles_3,
        "smiles_4": smiles_4,
    }


def _download_bytes(url: str, *, verify: bool = True, use_env_proxy: bool = False) -> bytes:
    session = requests.Session()
    session.trust_env = use_env_proxy
    response = session.get(url, timeout=120, verify=verify)
    response.raise_for_status()
    return response.content


def _ensure_qm9_csv(raw_dir: Path, *, verify_ssl: bool = True, use_env_proxy: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "qm9.csv"
    if not csv_path.exists():
        csv_bytes = _download_bytes(QM9_CSV_URL, verify=verify_ssl, use_env_proxy=use_env_proxy)
        csv_path.write_bytes(csv_bytes)
    return csv_path


def _ensure_qm9edge_npz(raw_dir: Path, *, verify_ssl: bool = True, use_env_proxy: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    npz_path = raw_dir / "qm9_edge.npz"
    if not npz_path.exists():
        npz_bytes = _download_bytes(QM9_EDGE_URL, verify=verify_ssl, use_env_proxy=use_env_proxy)
        npz_path.write_bytes(npz_bytes)
    return npz_path


def _sample_signature(props: dict[str, float]) -> tuple:
    return tuple(np.float32(props[key]).item() for key in PROPERTY_KEYS)


def _graph_wl_signature(graph) -> str:
    g = nx.Graph()
    atom_nums = graph.ndata["attr"][:, 5].detach().cpu().numpy().astype(int)
    for idx, atomic_num in enumerate(atom_nums.tolist()):
        g.add_node(idx, label=str(atomic_num))

    src, dst = graph.edges()
    edge_types = np.argmax(graph.edata["edge_attr"].detach().cpu().numpy(), axis=1).astype(int)
    seen = set()
    for s, d, e in zip(src.tolist(), dst.tolist(), edge_types.tolist()):
        key = tuple(sorted((int(s), int(d))))
        if key in seen:
            continue
        seen.add(key)
        g.add_edge(key[0], key[1], label=str(e))

    return weisfeiler_lehman_graph_hash(g, node_attr="label", edge_attr="label")


def _ordered_graph_signature(graph) -> str:
    digest = hashlib.sha256()
    digest.update(f"nodes:{graph.num_nodes()}".encode())
    digest.update(f"edges:{graph.num_edges()}".encode())
    for key in sorted(graph.ndata.keys()):
        tensor = graph.ndata[key].detach().cpu().contiguous()
        digest.update(f"ndata:{key}".encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    for key in sorted(graph.edata.keys()):
        tensor = graph.edata[key].detach().cpu().contiguous()
        digest.update(f"edata:{key}".encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _reorder_to_reference(
    samples: list[tuple],
    smiles_buffers: dict[str, list[str]],
    reference_data_pkl: Path,
    *,
    reorder_smiles: bool = True,
) -> tuple[list[tuple], dict[str, list[str]]]:
    with reference_data_pkl.open("rb") as f:
        reference_samples = pickle.load(f)

    signature_to_candidate_indices: dict[str, list[int]] = {}
    for idx, (_graph, props) in enumerate(samples):
        signature = _property_signature(props)
        signature_to_candidate_indices.setdefault(signature, []).append(idx)

    duplicate_signatures = {sig for sig, bucket in signature_to_candidate_indices.items() if len(bucket) > 1}
    duplicate_candidate_maps: dict[tuple, dict[str, dict[str, list[int]]]] = {}
    for signature in duplicate_signatures:
        exact_to_indices: dict[str, list[int]] = {}
        wl_to_indices: dict[str, list[int]] = {}
        for idx in signature_to_candidate_indices[signature]:
            graph, _ = samples[idx]
            exact_to_indices.setdefault(_ordered_graph_signature(graph), []).append(idx)
            wl_to_indices.setdefault(_graph_wl_signature(graph), []).append(idx)
        duplicate_candidate_maps[signature] = {
            "exact": exact_to_indices,
            "wl": wl_to_indices,
        }

    ordered_indices: list[int] = []
    for reference_graph, props in reference_samples:
        signature = _property_signature(props)
        if signature in duplicate_candidate_maps:
            exact_signature = _ordered_graph_signature(reference_graph)
            exact_bucket = duplicate_candidate_maps[signature]["exact"].get(exact_signature)
            if exact_bucket:
                ordered_indices.append(exact_bucket.pop())
                continue

            wl_signature = _graph_wl_signature(reference_graph)
            wl_bucket = duplicate_candidate_maps[signature]["wl"].get(wl_signature)
            if not wl_bucket:
                raise KeyError("Failed to align QM9 duplicate-signature sample to reference order")
            ordered_indices.append(wl_bucket.pop())
            continue
        bucket = signature_to_candidate_indices.get(signature)
        if not bucket:
            raise KeyError("Failed to align QM9 candidate sample to reference order")
        ordered_indices.append(bucket.pop())

    reordered_samples = [samples[i] for i in ordered_indices]
    if reorder_smiles:
        reordered_smiles = {key: [values[i] for i in ordered_indices] for key, values in smiles_buffers.items()}
    else:
        reordered_smiles = smiles_buffers
    return reordered_samples, reordered_smiles


def _reuse_existing_splits(dataset_dir: Path, split_source_dir: Path | None) -> None:
    source_dir = split_source_dir or dataset_dir
    required = ["train_index.json", "val_index.json", "test_index.json"]
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "QM9 baseline split rule has not been recovered yet. "
            f"Missing files: {missing}. "
            "Current script refuses to invent a replacement split."
        )
    if source_dir != dataset_dir:
        for name in required:
            (dataset_dir / name).write_text((source_dir / name).read_text(encoding="utf-8"), encoding="utf-8")


def _convert_label_tensor(label_tensor: torch.Tensor) -> dict[str, float]:
    label_tensor = label_tensor.detach().cpu().view(-1)
    props = {}
    for idx, edge_key in enumerate(QM9EDGE_LABEL_KEYS):
        if edge_key not in QM9EDGE_TO_BASELINE:
            continue
        props[QM9EDGE_TO_BASELINE[edge_key]] = float(np.float32(label_tensor[idx].item()))
    return props


def _canonicalize_smiles(smiles: str) -> dict[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid QM9 SMILES from CSV lookup")
    return generate_four_smiles_formats(mol)


def _property_signature(props: dict[str, float]) -> tuple:
    return tuple(np.float32(props[key]).item() for key in PROPERTY_KEYS)


def _build_smiles_lookup(csv_path: Path) -> dict[tuple, str]:
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        props = {}
        for key in PROPERTY_KEYS:
            candidate = {
                "u0": "U0",
                "u298": "U",
                "h298": "H",
                "g298": "G",
                "cv": "Cv",
                "u0_atom": "U0_atom",
                "u298_atom": "U_atom",
                "h298_atom": "H_atom",
                "g298_atom": "G_atom",
            }.get(key, key)
            if candidate not in row or pd.isna(row[candidate]):
                props = None
                break
            props[key] = float(np.float32(row[candidate]))
        if props is None:
            continue
        lookup.setdefault(_property_signature(props), row["smiles"])
    return lookup


def _write_lines(path: Path, values: list[str]) -> None:
    payload = "\n".join(values)
    if values:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def prepare_qm9_raw(
    *,
    verify_ssl: bool = True,
    use_env_proxy: bool = False,
    split_source_dir: str | None = None,
    output_dir: str | None = None,
    reference_data_pkl: str | None = None,
    reference_smiles_dir: str | None = None,
) -> None:
    dataset_dir = Path(output_dir).resolve() if output_dir else (REPO_ROOT / "data" / "qm9")
    raw_dir = dataset_dir / "raw_sources"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    split_dir_path = Path(split_source_dir).resolve() if split_source_dir else None
    _reuse_existing_splits(dataset_dir, split_dir_path)

    _ensure_qm9edge_npz(raw_dir, verify_ssl=verify_ssl, use_env_proxy=use_env_proxy)
    csv_path = _ensure_qm9_csv(raw_dir, verify_ssl=verify_ssl, use_env_proxy=use_env_proxy)
    smiles_lookup = _build_smiles_lookup(csv_path)
    ds = QM9EdgeDataset(label_keys=QM9EDGE_LABEL_KEYS, raw_dir=str(raw_dir), force_reload=False, verbose=False)

    all_samples: list[tuple] = []
    smiles_buffers = {"smiles_1": [], "smiles_2": [], "smiles_3": [], "smiles_4": []}

    for idx in range(len(ds)):
        graph, label = ds[idx]
        assert graph.ndata["attr"].shape[1] == ATTR_DIM
        assert graph.edata["edge_attr"].shape[1] == EDGE_ATTR_DIM
        props = _convert_label_tensor(label)
        all_samples.append((graph, props))
        if reference_smiles_dir is None:
            smiles = smiles_lookup.get(_property_signature(props))
            if smiles is None:
                raise KeyError("Failed to match QM9EdgeDataset sample to CSV SMILES by property signature")
            smiles_dict = _canonicalize_smiles(smiles)
            for key, value in smiles_dict.items():
                smiles_buffers[key].append(value)

    if reference_smiles_dir is not None:
        ref_dir = Path(reference_smiles_dir).resolve()
        smiles_buffers = {
            "smiles_1": (ref_dir / "smiles_1_direct.txt").read_text(encoding="utf-8").splitlines(),
            "smiles_2": (ref_dir / "smiles_2_explicit_h.txt").read_text(encoding="utf-8").splitlines(),
            "smiles_3": (ref_dir / "smiles_3_addhs.txt").read_text(encoding="utf-8").splitlines(),
            "smiles_4": (ref_dir / "smiles_4_addhs_explicit_h.txt").read_text(encoding="utf-8").splitlines(),
        }
        if any(len(v) != len(all_samples) for v in smiles_buffers.values()):
            raise ValueError("reference_smiles_dir does not match QM9 sample count")

    if reference_data_pkl:
        all_samples, smiles_buffers = _reorder_to_reference(
            all_samples,
            smiles_buffers,
            Path(reference_data_pkl).resolve(),
            reorder_smiles=reference_smiles_dir is None,
        )

    with (dataset_dir / "data.pkl").open("wb") as f:
        pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    _write_lines(dataset_dir / "smiles_1_direct.txt", smiles_buffers["smiles_1"])
    _write_lines(dataset_dir / "smiles_2_explicit_h.txt", smiles_buffers["smiles_2"])
    _write_lines(dataset_dir / "smiles_3_addhs.txt", smiles_buffers["smiles_3"])
    _write_lines(dataset_dir / "smiles_4_addhs_explicit_h.txt", smiles_buffers["smiles_4"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare QM9 from public raw sources into current project layout.")
    parser.add_argument("--split-source-dir", type=str, default=None, help="Optional directory containing canonical train/val/test index JSON files.")
    parser.add_argument("--verify-ssl", action="store_true", help="Enable SSL verification for source downloads.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Use environment proxy variables for source downloads.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory. Defaults to data/qm9 in the current repository.")
    parser.add_argument("--reference-data-pkl", type=str, default=None, help="Optional reference data.pkl used to replay the exact current baseline order.")
    parser.add_argument("--reference-smiles-dir", type=str, default=None, help="Optional directory containing canonical SMILES side files to replay exactly.")
    args = parser.parse_args()
    prepare_qm9_raw(
        verify_ssl=args.verify_ssl,
        use_env_proxy=args.use_env_proxy,
        split_source_dir=args.split_source_dir,
        output_dir=args.output_dir,
        reference_data_pkl=args.reference_data_pkl,
        reference_smiles_dir=args.reference_smiles_dir,
    )
