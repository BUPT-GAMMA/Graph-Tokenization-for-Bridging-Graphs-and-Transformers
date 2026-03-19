#!/usr/bin/env python3
from __future__ import annotations

"""
ZINC cold-start preparation from public benchmarking-gnns package.

Uses the public `ZINC.pkl` artifact as the reproducible source of the sampled
10k/1k/1k split that matches the current project baseline more closely than the
full `molecules.zip` archive.
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import dgl
import requests
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.zinc.molecular_graph_utils import dgl_graph_to_mol, generate_four_smiles_formats, mol_to_explicit_h_graph


ZINC_PKL_URL = "https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl"


class _CompatMoleculeDGL:
    pass


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "dgl.heterograph" and name == "DGLHeteroGraph":
            return dgl.DGLGraph
        if module == "data.molecules" and name == "MoleculeDGL":
            return _CompatMoleculeDGL
        return super().find_class(module, name)


def _download_bytes(url: str, *, verify_ssl: bool = True, use_env_proxy: bool = False) -> bytes:
    session = requests.Session()
    session.trust_env = use_env_proxy
    response = session.get(url, timeout=300, verify=verify_ssl)
    response.raise_for_status()
    return response.content


def _ensure_public_pkl(raw_dir: Path, *, verify_ssl: bool = True, use_env_proxy: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = raw_dir / "ZINC.pkl"
    if not pkl_path.exists():
        pkl_path.write_bytes(_download_bytes(ZINC_PKL_URL, verify_ssl=verify_ssl, use_env_proxy=use_env_proxy))
    return pkl_path


def _load_public_payload(path: Path):
    with path.open("rb") as f:
        payload = _CompatUnpickler(f).load()
    train_set, val_set, test_set = payload[:3]
    return train_set, val_set, test_set


def _write_lines(path: Path, values: list[str]) -> None:
    payload = "\n".join(values)
    if values:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _graph_label_signature(graph, label) -> str:
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
    label_tensor = label.detach().cpu().contiguous() if torch.is_tensor(label) else torch.tensor([float(label)], dtype=torch.float32)
    digest.update(label_tensor.numpy().tobytes())
    return digest.hexdigest()


def _reorder_to_reference(samples: list[tuple], smiles_buffers: dict[str, list[str]], reference_data_pkl: Path) -> tuple[list[tuple], dict[str, list[str]]]:
    with reference_data_pkl.open("rb") as f:
        reference_samples = pickle.load(f)
    lookup: dict[str, list[int]] = {}
    for idx, (graph, label) in enumerate(samples):
        lookup.setdefault(_graph_label_signature(graph, label), []).append(idx)
    ordered_indices = []
    for graph, label in reference_samples:
        bucket = lookup.get(_graph_label_signature(graph, label))
        if not bucket:
            raise KeyError("Failed to align ZINC candidate sample to reference order")
        ordered_indices.append(bucket.pop())
    reordered_samples = [samples[i] for i in ordered_indices]
    reordered_smiles = {key: [values[i] for i in ordered_indices] for key, values in smiles_buffers.items()}
    return reordered_samples, reordered_smiles


def _reuse_existing_splits(dataset_dir: Path, split_source_dir: Path | None) -> None:
    source_dir = split_source_dir or dataset_dir
    for name in ["train_index.json", "val_index.json", "test_index.json"]:
        if not (source_dir / name).exists():
            raise FileNotFoundError(f"Missing ZINC split file: {source_dir / name}")
        if source_dir != dataset_dir:
            (dataset_dir / name).write_text((source_dir / name).read_text(encoding="utf-8"), encoding="utf-8")


def prepare_zinc_raw(
    *,
    verify_ssl: bool = True,
    use_env_proxy: bool = False,
    output_dir: str | None = None,
    split_source_dir: str | None = None,
    reference_data_pkl: str | None = None,
) -> None:
    dataset_dir = Path(output_dir).resolve() if output_dir else (REPO_ROOT / "data" / "zinc")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_root = dataset_dir / "raw_sources"
    public_pkl = _ensure_public_pkl(raw_root, verify_ssl=verify_ssl, use_env_proxy=use_env_proxy)
    train_set, val_set, test_set = _load_public_payload(public_pkl)
    split_dir_path = Path(split_source_dir).resolve() if split_source_dir else None
    _reuse_existing_splits(dataset_dir, split_dir_path)

    ordered_graphs = []
    smiles_buffers = {"smiles_1": [], "smiles_2": [], "smiles_3": [], "smiles_4": []}
    for _split, dataset in [("train", train_set), ("val", val_set), ("test", test_set)]:
        for graph, label in zip(dataset.graph_lists, dataset.graph_labels):
            mol = dgl_graph_to_mol(graph)
            if mol is None:
                raise ValueError("Failed to convert ZINC graph back to RDKit molecule")
            simplified_graph = mol_to_explicit_h_graph(mol)
            label_tensor = label if torch.is_tensor(label) else torch.tensor([float(label)], dtype=torch.float32)
            ordered_graphs.append((simplified_graph, label_tensor))
            smiles_dict = generate_four_smiles_formats(mol)
            for key, value in smiles_dict.items():
                smiles_buffers[key].append(value)

    if reference_data_pkl:
        ordered_graphs, smiles_buffers = _reorder_to_reference(
            ordered_graphs,
            smiles_buffers,
            Path(reference_data_pkl).resolve(),
        )

    with (dataset_dir / "data.pkl").open("wb") as f:
        pickle.dump(ordered_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    _write_lines(dataset_dir / "smiles_1_direct.txt", smiles_buffers["smiles_1"])
    _write_lines(dataset_dir / "smiles_2_explicit_h.txt", smiles_buffers["smiles_2"])
    _write_lines(dataset_dir / "smiles_3_addhs.txt", smiles_buffers["smiles_3"])
    _write_lines(dataset_dir / "smiles_4_addhs_explicit_h.txt", smiles_buffers["smiles_4"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ZINC from public sources into current project layout.")
    parser.add_argument("--verify-ssl", action="store_true", help="Enable SSL verification for source downloads.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Use environment proxy variables for source downloads.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory. Defaults to data/zinc.")
    parser.add_argument("--split-source-dir", type=str, default=None, help="Optional directory containing canonical split JSON files.")
    parser.add_argument("--reference-data-pkl", type=str, default=None, help="Optional reference data.pkl used to replay the exact current baseline order.")
    args = parser.parse_args()
    prepare_zinc_raw(
        verify_ssl=args.verify_ssl,
        use_env_proxy=args.use_env_proxy,
        output_dir=args.output_dir,
        split_source_dir=args.split_source_dir,
        reference_data_pkl=args.reference_data_pkl,
    )
