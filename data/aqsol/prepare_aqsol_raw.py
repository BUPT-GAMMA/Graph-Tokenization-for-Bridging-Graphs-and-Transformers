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
import argparse
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
import dgl
from rdkit import Chem
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.zinc.molecular_graph_utils import generate_four_smiles_formats, mol_to_explicit_h_graph, parse_atom_feature


AQSOL_URL = "https://www.dropbox.com/s/lzu9lmukwov12kt/aqsol_graph_raw.zip?dl=1"


class _CompatDictionary:
    pass


def _download_archive(out_path: Path, *, verify_ssl: bool = True, use_env_proxy: bool = False) -> None:
    session = requests.Session()
    session.trust_env = use_env_proxy
    response = session.get(AQSOL_URL, timeout=120, verify=verify_ssl)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def _ensure_raw_archive(raw_dir: Path, *, verify_ssl: bool = True, use_env_proxy: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "aqsol_graph_raw.zip"
    if not archive_path.exists():
        _download_archive(archive_path, verify_ssl=verify_ssl, use_env_proxy=use_env_proxy)
    return archive_path


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    if (target_dir / "asqol_graph_raw").exists():
        return
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)


def _load_split_payload(raw_root: Path, split: str):
    with open(raw_root / f"{split}.pickle", "rb") as f:
        return pickle.load(f)


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Dictionary":
            return _CompatDictionary
        return super().find_class(module, name)


def _load_dictionary(path: Path) -> _CompatDictionary:
    with path.open("rb") as f:
        return _CompatUnpickler(f).load()


def _build_graph(sample) -> dgl.DGLGraph:
    node_features = torch.LongTensor(sample[0])
    edge_features = torch.LongTensor(sample[1])
    g = dgl.graph((sample[2][0], sample[2][1]))
    if g.num_nodes() == 0 or g.num_nodes() != len(node_features):
        return None
    g.ndata["feat"] = node_features
    g.edata["feat"] = edge_features
    return g


def _graph_to_mol(graph: dgl.DGLGraph, atom_dict: _CompatDictionary, bond_dict: _CompatDictionary) -> Chem.Mol:
    atom_features = graph.ndata["feat"].view(-1).tolist()
    src, dst = graph.edges()
    edge_types = graph.edata["feat"].view(-1).tolist()
    mol = Chem.RWMol()
    for feature_idx in atom_features:
        feature_str = atom_dict.idx2word[int(feature_idx)]
        symbol, _, charge = parse_atom_feature(feature_str)
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(charge)
        mol.AddAtom(atom)
    seen = set()
    for s, d, e in zip(src.tolist(), dst.tolist(), edge_types):
        key = tuple(sorted((int(s), int(d))))
        if key in seen:
            continue
        seen.add(key)
        bond_str = bond_dict.idx2word[int(e)]
        if bond_str == "NONE":
            continue
        bond_type = {
            "SINGLE": Chem.BondType.SINGLE,
            "DOUBLE": Chem.BondType.DOUBLE,
            "TRIPLE": Chem.BondType.TRIPLE,
            "AROMATIC": Chem.BondType.AROMATIC,
        }[bond_str]
        mol.AddBond(int(s), int(d), bond_type)
    final = mol.GetMol()
    final.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(final, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_KEKULIZE)
    return final


def _write_lines(path: Path, values: list[str]) -> None:
    payload = "\n".join(values)
    if values:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def prepare_aqsol_raw(*, verify_ssl: bool = True, use_env_proxy: bool = False, output_dir: str | None = None) -> None:
    dataset_dir = Path(output_dir).resolve() if output_dir else (REPO_ROOT / "data" / "aqsol")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_root = dataset_dir / "raw_sources"
    archive_path = _ensure_raw_archive(raw_root, verify_ssl=verify_ssl, use_env_proxy=use_env_proxy)
    _extract_archive(archive_path, raw_root)
    raw_graph_root = raw_root / "asqol_graph_raw"
    atom_dict = _load_dictionary(raw_graph_root / "atom_dict.pickle")
    bond_dict = _load_dictionary(raw_graph_root / "bond_dict.pickle")

    all_samples = []
    smiles_buffers = {"smiles_1": [], "smiles_2": [], "smiles_3": [], "smiles_4": []}

    for split in ["train", "val", "test"]:
        payload = _load_split_payload(raw_graph_root, split)
        for sample in payload:
            raw_graph = _build_graph(sample)
            if raw_graph is None:
                continue
            label = float(np.float32(sample[3]))
            mol = _graph_to_mol(raw_graph, atom_dict, bond_dict)
            graph = mol_to_explicit_h_graph(mol)
            all_samples.append((graph, label))
            smiles_dict = generate_four_smiles_formats(mol)
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

    (dataset_dir / "train_index.json").write_text(json.dumps(train_idx, indent=2), encoding="utf-8")
    (dataset_dir / "val_index.json").write_text(json.dumps(val_idx, indent=2), encoding="utf-8")
    (dataset_dir / "test_index.json").write_text(json.dumps(test_idx, indent=2), encoding="utf-8")

    _write_lines(dataset_dir / "smiles_1_direct.txt", smiles_buffers["smiles_1"])
    _write_lines(dataset_dir / "smiles_2_explicit_h.txt", smiles_buffers["smiles_2"])
    _write_lines(dataset_dir / "smiles_3_addhs.txt", smiles_buffers["smiles_3"])
    _write_lines(dataset_dir / "smiles_4_addhs_explicit_h.txt", smiles_buffers["smiles_4"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AqSol from raw public sources into current project layout.")
    parser.add_argument("--verify-ssl", action="store_true", help="Enable SSL verification for source downloads.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Use environment proxy variables for source downloads.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory. Defaults to data/aqsol.")
    args = parser.parse_args()
    prepare_aqsol_raw(verify_ssl=args.verify_ssl, use_env_proxy=args.use_env_proxy, output_dir=args.output_dir)
