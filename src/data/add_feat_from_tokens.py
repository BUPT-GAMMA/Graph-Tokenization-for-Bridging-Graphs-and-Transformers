#!/usr/bin/env python3
"""
Write token IDs into standard feature keys (g.ndata['feat'] / g.edata['feat'])
for all (or specified) datasets and re-save data.pkl.

Rules:
- Node feat = node_token_ids (fallback: node_type_id.view(-1,1); error if missing)
- Edge feat = edge_token_ids (fallback: edge_type_id.view(-1,1); zeros if missing)

Save strategy:
- Default: write to data/gnn_use/<dataset>/data.pkl; copy split JSONs
- --inplace: overwrite original data.pkl (optionally --backup for .bak)

Examples:
  python src/data/add_feat_from_tokens.py --datasets "proteins,mutagenicity"
  python src/data/add_feat_from_tokens.py --inplace --backup
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import torch
import shutil

from src.data.unified_data_factory import list_available_datasets, get_dataloader
from config import ProjectConfig
import json

# Early datasets: use loader's get_graph_*_token_ids interface
_EARLY_DATASETS = {"qm9", "qm9test", "zinc", "aqsol"}


def _ensure_long_2d(t: torch.Tensor, num_rows: int) -> torch.Tensor:
    t = t.long()
    if t.dim() == 1:
        t = t.view(-1, 1)
    assert t.shape[0] == num_rows and t.shape[1] == 1, f"feat shape mismatch: expected [{num_rows},1], got {tuple(t.shape)}"
    return t


def _add_feat_to_graph(g, loader=None) -> None:
    # Nodes
    n = int(g.num_nodes())
    if loader is not None:
        node_tok = loader.get_graph_node_token_ids(g)
    elif 'node_token_ids' in g.ndata:
        node_tok = g.ndata['node_token_ids']
    elif 'node_type_id' in g.ndata:
        node_tok = g.ndata['node_type_id'].view(-1, 1)
    else:
        raise KeyError("Graph missing node_token_ids and node_type_id; cannot build node feat")
    node_tok = _ensure_long_2d(node_tok, n)
    # Ensure node tokens are in odd domain (offset from edges); remap if needed
    if node_tok.numel() > 0:
        if torch.any((node_tok.view(-1) % 2) == 0):
            node_tok = (node_tok * 2 + 1).long()
    g.ndata['feat'] = node_tok

    # Edges
    e = int(g.num_edges())
    if e == 0:
        g.edata['feat'] = torch.empty((0, 1), dtype=torch.long)
    else:
        if loader is not None:
            edge_tok = loader.get_graph_edge_token_ids(g)
        elif 'edge_token_ids' in g.edata:
            edge_tok = g.edata['edge_token_ids']
        elif 'edge_type_id' in g.edata:
            edge_tok = g.edata['edge_type_id'].view(-1, 1)
        else:
            edge_tok = torch.zeros((e, 1), dtype=torch.long)
        edge_tok = _ensure_long_2d(edge_tok, e)
        # Ensure edge tokens are in even domain; remap if needed
        if edge_tok.numel() > 0:
            if torch.any((edge_tok.view(-1) % 2) != 0):
                edge_tok = (edge_tok * 2).long()
        g.edata['feat'] = edge_tok


def _strip_to_feat_only(g) -> None:
    # Keep only feat; remove other ndata/edata keys to reduce size
    for k in list(g.ndata.keys()):
        if k != 'feat':
            del g.ndata[k]
    for k in list(g.edata.keys()):
        if k != 'feat':
            del g.edata[k]


def _print_feat_summary(ds_name: str, payload: List[Any], limit: int = 2) -> None:
    print(f"--- Dataset: {ds_name} feat overview (up to {limit} samples) ---")
    shown = 0
    for item in payload:
        if shown >= limit:
            break
        if isinstance(item, tuple) and len(item) == 2:
            g, _ = item
        elif isinstance(item, dict):
            g = item.get('dgl_graph')
        else:
            continue
        n, e = int(g.num_nodes()), int(g.num_edges())
        nf = g.ndata['feat'] if 'feat' in g.ndata else None
        ef = g.edata['feat'] if 'feat' in g.edata else None
        nf_shape = tuple(nf.shape) if nf is not None else None
        ef_shape = tuple(ef.shape) if ef is not None else None
        nf_dtype = str(nf.dtype) if nf is not None else None
        ef_dtype = str(ef.dtype) if ef is not None else None
        nf_head = nf.view(-1)[:5].tolist() if nf is not None and nf.numel() > 0 else []
        ef_head = ef.view(-1)[:5].tolist() if ef is not None and ef.numel() > 0 else []
        print(f"Sample {shown}: N={n}, E={e}; node_feat={nf_shape}/{nf_dtype} e.g.: {nf_head}; edge_feat={ef_shape}/{ef_dtype} e.g.: {ef_head}")
        shown += 1
    if shown == 0:
        print("No samples to display")


def _process_dataset(ds_name: str, inplace: bool, backup: bool, cfg: ProjectConfig) -> Tuple[str, int]:
    data_dir = Path('data') / ds_name
    src_file = data_dir / 'data.pkl'
    if not src_file.exists():
        return '', 0

    # Early datasets: load via loader and reassemble in original order
    if ds_name in _EARLY_DATASETS:
        try:
            loader = get_dataloader(ds_name, cfg)
            train_data, val_data, test_data, _, _, _ = loader.load_data()
            # Populate tokens on graphs
            try:
                all_samples = train_data + val_data + test_data
                if hasattr(loader, '_build_attribute_cache'):
                    loader._build_attribute_cache(all_samples)
            except Exception as e:
                print(f"[WARN] Failed to populate tokens for early dataset ({ds_name}): {e}")
            # Read original indices
            with (data_dir / 'train_index.json').open('r') as f:
                train_idx = json.load(f)
            with (data_dir / 'val_index.json').open('r') as f:
                val_idx = json.load(f)
            with (data_dir / 'test_index.json').open('r') as f:
                test_idx = json.load(f)
            total_len = max(train_idx + val_idx + test_idx) + 1 if (train_idx or val_idx or test_idx) else (len(train_data) + len(val_data) + len(test_data))
            assembled: List[Any] = [None] * total_len
            # Place back at original positions
            for pos, s in zip(train_idx, train_data):
                g = s['dgl_graph']
                _add_feat_to_graph(g, loader=loader)
                assembled[pos] = (g, s.get('properties', {}))
            for pos, s in zip(val_idx, val_data):
                g = s['dgl_graph']
                _add_feat_to_graph(g, loader=loader)
                assembled[pos] = (g, s.get('properties', {}))
            for pos, s in zip(test_idx, test_data):
                g = s['dgl_graph']
                _add_feat_to_graph(g, loader=loader)
                assembled[pos] = (g, s.get('properties', {}))
            # Remove possible None entries (non-contiguous indices)
            payload = [x for x in assembled if x is not None]
            updated = len(payload)
        except Exception as e:
            print(f"[WARN] Loader processing failed for early dataset ({ds_name}), falling back to raw fields: {e}")
            with src_file.open('rb') as f:
                payload = pickle.load(f)
            updated = 0
            if isinstance(payload, list) and payload:
                sample0 = payload[0]
                if isinstance(sample0, tuple) and len(sample0) == 2:
                    new_list: List[Any] = []
                    for g, y in payload:
                        _add_feat_to_graph(g, loader=None)
                        new_list.append((g, y))
                        updated += 1
                    payload = new_list
                elif isinstance(sample0, dict):
                    for s in payload:
                        g = s.get('dgl_graph')
                        assert g is not None, 'Sample missing dgl_graph'
                        _add_feat_to_graph(g, loader=None)
                        updated += 1
                else:
                    raise TypeError('Unsupported data.pkl element type')
    else:
        # Regular datasets: read and write feat directly
        with src_file.open('rb') as f:
            payload = pickle.load(f)
        updated = 0
        if isinstance(payload, list) and payload:
            sample0 = payload[0]
            if isinstance(sample0, tuple) and len(sample0) == 2:
                new_list: List[Any] = []
                for g, y in payload:
                    _add_feat_to_graph(g, loader=None)
                    new_list.append((g, y))
                    updated += 1
                payload = new_list
            elif isinstance(sample0, dict):
                for s in payload:
                    g = s.get('dgl_graph')
                    assert g is not None, 'Sample missing dgl_graph'
                    _add_feat_to_graph(g, loader=None)
                    updated += 1
            else:
                raise TypeError('Unsupported data.pkl element type')

    # Target directory and save
    if inplace:
        if backup:
            bak = src_file.with_suffix('.pkl.bak')
            if not bak.exists():
                bak.write_bytes(src_file.read_bytes())
        out_dir = data_dir
        out_file = src_file
    else:
        out_dir = Path('data') / 'gnn_use' / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / 'data.pkl'

    # If writing to new directory (default), keep only feat to reduce size
    if not inplace:
        if isinstance(payload, list) and payload:
            sample0 = payload[0]
            if isinstance(sample0, tuple) and len(sample0) == 2:
                minimized: List[Any] = []
                for g, y in payload:
                    _strip_to_feat_only(g)
                    minimized.append((g, y))
                payload = minimized
            elif isinstance(sample0, dict):
                for s in payload:
                    g = s.get('dgl_graph')
                    _strip_to_feat_only(g)

    # Print feat summary before saving
    try:
        _print_feat_summary(ds_name, payload, limit=2)
    except Exception as e:
        print(f"[WARN] Failed to print feat summary: {e}")

    with out_file.open('wb') as f:
        pickle.dump(payload, f)

    # Copy split JSONs to target directory
    for name in ['train_index.json', 'val_index.json', 'test_index.json']:
        src = data_dir / name
        if src.exists():
            dst = out_dir / name
            shutil.copy2(src, dst)

    return str(out_file), updated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default=None, help='Comma-separated dataset names; processes all if omitted')
    ap.add_argument('--inplace', action='store_true', help='Overwrite data.pkl in place')
    ap.add_argument('--backup', action='store_true', help='Backup as data.pkl.bak when using --inplace')
    args = ap.parse_args()

    if args.datasets:
        datasets = [x.strip() for x in args.datasets.split(',') if x.strip()]
    else:
        datasets = list_available_datasets()

    cfg = ProjectConfig()

    total = 0
    for ds in datasets:
        out_file, cnt = _process_dataset(ds, inplace=args.inplace, backup=args.backup, cfg=cfg)
        if out_file:
            print(f"[{ds}] Written: {out_file} (graphs updated: {cnt})")
        else:
            print(f"[{ds}] Skipped: data/{ds}/data.pkl not found")
        total += cnt
    print(f"Done. {total} graphs updated.")


if __name__ == '__main__':
    main()


