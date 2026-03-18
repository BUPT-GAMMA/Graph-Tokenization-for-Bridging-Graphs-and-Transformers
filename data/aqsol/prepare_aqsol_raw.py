#!/usr/bin/env python3
from __future__ import annotations

"""
AQSOL raw cold-start preparation
================================

目标：
- 从公开原始 AqSol 图数据恢复当前 `data/aqsol` baseline 结构
- 生成：
  - `data.pkl`
  - `train_index.json`
  - `val_index.json`
  - `test_index.json`
  - 四份 SMILES side files

说明：
- 当前 baseline split 与简单 `train_test_split(random_state=42)` 一致
- 原始输入格式来自历史 `aqsol_graph_raw.zip`
"""

import json
import pickle
import sys
import zipfile
from pathlib import Path

import requests
import torch
import dgl
from rdkit import Chem
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.zinc.molecular_graph_utils import generate_four_smiles_formats


AQSOL_URL = "https://www.dropbox.com/s/lzu9lmukwov12kt/aqsol_graph_raw.zip?dl=1"


def _download_archive(out_path: Path, *, verify_ssl: bool = True) -> None:
    response = requests.get(AQSOL_URL, timeout=120, verify=verify_ssl)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def _ensure_raw_archive(raw_dir: Path, *, verify_ssl: bool = True) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "aqsol_graph_raw.zip"
    if not archive_path.exists():
        _download_archive(archive_path, verify_ssl=verify_ssl)
    return archive_path


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    if (target_dir / "asqol_graph_raw").exists():
        return
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)


def _load_split_payload(raw_root: Path, split: str):
    with open(raw_root / f"{split}.pickle", "rb") as f:
        return pickle.load(f)


def _build_graph(sample) -> dgl.DGLGraph:
    node_features = torch.LongTensor(sample[0])
    edge_features = torch.LongTensor(sample[1])
    g = dgl.graph((sample[2][0], sample[2][1]))
    if g.num_nodes() == 0 or g.num_nodes() != len(node_features):
        raise ValueError("Invalid AqSol raw graph sample")
    g.ndata["feat"] = node_features
    g.edata["feat"] = edge_features
    return g


def _graph_to_smiles(graph: dgl.DGLGraph) -> dict[str, str]:
    atom_numbers = graph.ndata["feat"].view(-1).tolist()
    src, dst = graph.edges()
    edge_types = graph.edata["feat"].view(-1).tolist()
    mol = Chem.RWMol()
    for atomic_num in atom_numbers:
        mol.AddAtom(Chem.Atom(int(atomic_num)))
    seen = set()
    for s, d, e in zip(src.tolist(), dst.tolist(), edge_types):
        key = tuple(sorted((int(s), int(d))))
        if key in seen:
            continue
        seen.add(key)
        bond_type = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
        }.get(int(e), Chem.BondType.SINGLE)
        mol.AddBond(int(s), int(d), bond_type)
    final = mol.GetMol()
    Chem.SanitizeMol(final)
    return generate_four_smiles_formats(final)


def prepare_aqsol_raw(*, verify_ssl: bool = True) -> None:
    dataset_dir = REPO_ROOT / Path("data") / "aqsol"
    raw_root = dataset_dir / "raw_sources"
    archive_path = _ensure_raw_archive(raw_root, verify_ssl=verify_ssl)
    _extract_archive(archive_path, raw_root)
    raw_graph_root = raw_root / "asqol_graph_raw"

    all_samples = []
    smiles_buffers = {"smiles_1": [], "smiles_2": [], "smiles_3": [], "smiles_4": []}

    for split in ["train", "val", "test"]:
        payload = _load_split_payload(raw_graph_root, split)
        for sample in payload:
            graph = _build_graph(sample)
            label = float(sample[3])
            all_samples.append((graph, label))
            smiles_dict = _graph_to_smiles(graph)
            for key, value in smiles_dict.items():
                smiles_buffers[key].append(value)

    with open(dataset_dir / "data.pkl", "wb") as f:
        pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    idx = list(range(len(all_samples)))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
    test_idx, val_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)

    (dataset_dir / "train_index.json").write_text(json.dumps(train_idx), encoding="utf-8")
    (dataset_dir / "val_index.json").write_text(json.dumps(val_idx), encoding="utf-8")
    (dataset_dir / "test_index.json").write_text(json.dumps(test_idx), encoding="utf-8")

    (dataset_dir / "smiles_1_direct.txt").write_text("\n".join(smiles_buffers["smiles_1"]), encoding="utf-8")
    (dataset_dir / "smiles_2_explicit_h.txt").write_text("\n".join(smiles_buffers["smiles_2"]), encoding="utf-8")
    (dataset_dir / "smiles_3_addhs.txt").write_text("\n".join(smiles_buffers["smiles_3"]), encoding="utf-8")
    (dataset_dir / "smiles_4_addhs_explicit_h.txt").write_text("\n".join(smiles_buffers["smiles_4"]), encoding="utf-8")


if __name__ == "__main__":
    prepare_aqsol_raw()
