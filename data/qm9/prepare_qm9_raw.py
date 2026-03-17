#!/usr/bin/env python3
from __future__ import annotations

"""
QM9 raw cold-start preparation
==============================

目标：
- 从公开原始数据源构建当前项目 `data/qm9/` 期望的数据结构
- 产出 `data.pkl` 与四份 SMILES side files
- 若当前目录下已存在 baseline split 文件，则复用它们
- 若不存在 baseline split 文件，则明确失败，而不是猜测一个“看起来差不多”的 split

说明：
- 当前运行环境下，QM9 两个已知公开源都可能受到 SSL / 代理链路影响。
- 因此脚本实现了完整逻辑，但在源不可达时会 fail-fast。
"""

import io
import json
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import torch
import dgl
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
NPZ_URL = "https://data.dgl.ai/dataset/qm9_eV.npz"
ATOM_ORDER = ["H", "C", "N", "O", "F"]
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
BOND_TYPE_TO_INDEX = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def _download_bytes(url: str, *, verify: bool = True) -> bytes:
    response = requests.get(url, timeout=60, verify=verify)
    response.raise_for_status()
    return response.content


def _load_qm9_csv(raw_dir: Path, *, verify_ssl: bool = True) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "qm9.csv"
    if not csv_path.exists():
        csv_bytes = _download_bytes(CSV_URL, verify=verify_ssl)
        csv_path.write_bytes(csv_bytes)
    return pd.read_csv(csv_path)


def _load_qm9_npz(raw_dir: Path, *, verify_ssl: bool = True) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    npz_path = raw_dir / "qm9_eV.npz"
    if not npz_path.exists():
        npz_bytes = _download_bytes(NPZ_URL, verify=verify_ssl)
        npz_path.write_bytes(npz_bytes)
    return dict(np.load(npz_path, allow_pickle=True))


def _extract_coordinates(npz_payload: dict) -> list[np.ndarray]:
    counts = npz_payload["N"]
    coords = npz_payload["R"]
    cumsum = np.concatenate([[0], np.cumsum(counts)])
    return [coords[cumsum[i] : cumsum[i + 1]] for i in range(len(counts))]


def _build_node_attr(mol: Chem.Mol) -> torch.Tensor:
    rows = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ATOM_ORDER:
            raise ValueError(f"Unsupported QM9 atom symbol: {symbol}")
        one_hot = [1.0 if symbol == item else 0.0 for item in ATOM_ORDER]
        atomic_num = float(atom.GetAtomicNum())
        degree = float(atom.GetDegree())
        row = one_hot + [atomic_num, 0.0, 0.0, 0.0, 0.0, degree]
        assert len(row) == ATTR_DIM
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def _build_edge_attr(mol: Chem.Mol) -> tuple[list[int], list[int], torch.Tensor]:
    src: list[int] = []
    dst: list[int] = []
    features: list[list[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_index = BOND_TYPE_TO_INDEX.get(bond.GetBondType())
        if bond_index is None:
            raise ValueError(f"Unsupported QM9 bond type: {bond.GetBondType()}")
        one_hot = [1.0 if i == bond_index else 0.0 for i in range(EDGE_ATTR_DIM)]
        src.extend([begin, end])
        dst.extend([end, begin])
        features.extend([one_hot, one_hot])
    return src, dst, torch.tensor(features, dtype=torch.float32)


def _canonicalize_smiles(smiles: str) -> dict[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid QM9 SMILES: {smiles}")
    return {
        "smiles_1": Chem.MolToSmiles(mol),
        "smiles_2": Chem.MolToSmiles(mol, allHsExplicit=True),
        "smiles_3": Chem.MolToSmiles(Chem.AddHs(mol)),
        "smiles_4": Chem.MolToSmiles(Chem.AddHs(mol), allHsExplicit=True),
    }


def _extract_property_dict(row: pd.Series) -> dict[str, float]:
    aliases = {
        "u0": ["u0", "U0"],
        "u298": ["u298", "U"],
        "h298": ["h298", "H"],
        "g298": ["g298", "G"],
        "cv": ["cv", "Cv"],
        "u0_atom": ["u0_atom"],
        "u298_atom": ["u298_atom", "U_atom"],
        "h298_atom": ["h298_atom", "H_atom"],
        "g298_atom": ["g298_atom", "G_atom"],
    }
    props: dict[str, float] = {}
    for key in PROPERTY_KEYS:
        candidate_keys = aliases.get(key, [key])
        value = None
        for candidate in candidate_keys:
            if candidate in row and pd.notna(row[candidate]):
                value = float(row[candidate])
                break
        if value is None:
            raise KeyError(f"Missing QM9 property for key {key}, checked {candidate_keys}")
        props[key] = value
    return props


def _reuse_existing_splits(dataset_dir: Path) -> None:
    required = ["train_index.json", "val_index.json", "test_index.json"]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "QM9 baseline split rule has not been recovered yet. "
            f"Missing files: {missing}. "
            "Current script refuses to invent a replacement split."
        )


def prepare_qm9_raw(*, verify_ssl: bool = True) -> None:
    dataset_dir = REPO_ROOT / "data" / "qm9"
    raw_dir = dataset_dir / "raw_sources"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    _reuse_existing_splits(dataset_dir)

    df = _load_qm9_csv(raw_dir, verify_ssl=verify_ssl)
    npz_payload = _load_qm9_npz(raw_dir, verify_ssl=verify_ssl)
    coords_per_graph = _extract_coordinates(npz_payload)
    assert len(df) == len(coords_per_graph), "QM9 CSV rows and QM9 NPZ molecules are misaligned"

    all_samples = []
    smiles_1: list[str] = []
    smiles_2: list[str] = []
    smiles_3: list[str] = []
    smiles_4: list[str] = []

    for idx, row in df.iterrows():
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid QM9 SMILES at row {idx}: {smiles}")

        coords = coords_per_graph[idx]
        if mol.GetNumAtoms() != len(coords):
            raise ValueError(
                f"QM9 coordinate count mismatch at row {idx}: "
                f"{mol.GetNumAtoms()} atoms vs {len(coords)} coordinates"
            )

        src, dst, edge_attr = _build_edge_attr(mol)
        graph = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
        graph.ndata["pos"] = torch.tensor(coords, dtype=torch.float32)
        graph.ndata["attr"] = _build_node_attr(mol)
        graph.edata["edge_attr"] = edge_attr

        props = _extract_property_dict(row)
        all_samples.append((graph, props))

        smiles_dict = _canonicalize_smiles(smiles)
        smiles_1.append(smiles_dict["smiles_1"])
        smiles_2.append(smiles_dict["smiles_2"])
        smiles_3.append(smiles_dict["smiles_3"])
        smiles_4.append(smiles_dict["smiles_4"])

    with (dataset_dir / "data.pkl").open("wb") as f:
        pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    (dataset_dir / "smiles_1_direct.txt").write_text("\n".join(smiles_1), encoding="utf-8")
    (dataset_dir / "smiles_2_explicit_h.txt").write_text("\n".join(smiles_2), encoding="utf-8")
    (dataset_dir / "smiles_3_addhs.txt").write_text("\n".join(smiles_3), encoding="utf-8")
    (dataset_dir / "smiles_4_addhs_explicit_h.txt").write_text("\n".join(smiles_4), encoding="utf-8")


if __name__ == "__main__":
    prepare_qm9_raw()
