#!/usr/bin/env python3
from __future__ import annotations

"""
ZINC raw cold-start preparation
===============================

目标：
- 从公开原始分子数据恢复当前 `data/zinc` baseline 结构
- 生成：
  - `data.pkl`
  - `train_index.json`
  - `val_index.json`
  - `test_index.json`
  - 四份 SMILES side files

说明：
- 该脚本使用历史 molecules 格式中的官方 sampled split index
- 具体 raw source 为 `molecules.zip`
"""

import csv
import io
import json
import pickle
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

import requests
import torch
import dgl
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.zinc.molecular_graph_utils import generate_four_smiles_formats


MOLECULES_URL = "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1"


def _download_archive(out_path: Path, *, verify_ssl: bool = True) -> None:
    response = requests.get(MOLECULES_URL, timeout=120, verify=verify_ssl)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def _ensure_raw_archive(raw_dir: Path, *, verify_ssl: bool = True) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "molecules.zip"
    if not archive_path.exists():
        _download_archive(archive_path, verify_ssl=verify_ssl)
    return archive_path


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    if (target_dir / "molecules").exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)
    maybe_tar = target_dir / "molecules.tar"
    if maybe_tar.exists():
        with tarfile.open(maybe_tar, "r") as tf:
            tf.extractall(target_dir)


def _load_split_indices(raw_molecules_dir: Path, split: str) -> list[int]:
    with open(raw_molecules_dir / f"{split}.index", "r", encoding="utf-8") as f:
        rows = [list(map(int, row)) for row in csv.reader(f)]
    return rows[0]


def _load_split_payload(raw_molecules_dir: Path, split: str):
    with open(raw_molecules_dir / f"{split}.pickle", "rb") as f:
        return pickle.load(f)


def _bond_adj_to_graph(molecule: dict) -> dgl.DGLGraph:
    adj = molecule["bond_type"]
    edge_list = (adj != 0).nonzero()
    edge_features = adj[edge_list.split(1, dim=1)].reshape(-1).long()
    g = dgl.graph((edge_list[:, 0], edge_list[:, 1]), num_nodes=int(molecule["num_atom"]))
    g.ndata["feat"] = molecule["atom_type"].long()
    g.edata["feat"] = edge_features.long()
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


def prepare_zinc_raw(*, verify_ssl: bool = True) -> None:
    dataset_dir = REPO_ROOT / Path("data") / "zinc"
    raw_root = dataset_dir / "raw_sources"
    archive_path = _ensure_raw_archive(raw_root, verify_ssl=verify_ssl)
    _extract_archive(archive_path, raw_root)
    raw_molecules_dir = raw_root / "molecules"

    ordered_graphs = []
    split_indices: dict[str, list[int]] = {}
    smiles_buffers = {"smiles_1": [], "smiles_2": [], "smiles_3": [], "smiles_4": []}

    cursor = 0
    for split in ["train", "val", "test"]:
        payload = _load_split_payload(raw_molecules_dir, split)
        indices = _load_split_indices(raw_molecules_dir, split)
        selected = [payload[i] for i in indices]
        local_indices = list(range(cursor, cursor + len(selected)))
        split_indices[split] = local_indices
        cursor += len(selected)

        for molecule in selected:
            graph = _bond_adj_to_graph(molecule)
            label = torch.tensor([float(molecule["logP_SA_cycle_normalized"])], dtype=torch.float32)
            ordered_graphs.append((graph, label))
            smiles_dict = _graph_to_smiles(graph)
            for key, value in smiles_dict.items():
                smiles_buffers[key].append(value)

    with open(dataset_dir / "data.pkl", "wb") as f:
        pickle.dump(ordered_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    for split in ["train", "val", "test"]:
        (dataset_dir / f"{split}_index.json").write_text(json.dumps(split_indices[split]), encoding="utf-8")

    (dataset_dir / "smiles_1_direct.txt").write_text("\n".join(smiles_buffers["smiles_1"]), encoding="utf-8")
    (dataset_dir / "smiles_2_explicit_h.txt").write_text("\n".join(smiles_buffers["smiles_2"]), encoding="utf-8")
    (dataset_dir / "smiles_3_addhs.txt").write_text("\n".join(smiles_buffers["smiles_3"]), encoding="utf-8")
    (dataset_dir / "smiles_4_addhs_explicit_h.txt").write_text("\n".join(smiles_buffers["smiles_4"]), encoding="utf-8")


if __name__ == "__main__":
    prepare_zinc_raw()
