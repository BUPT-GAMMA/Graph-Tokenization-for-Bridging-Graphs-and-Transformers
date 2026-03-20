from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import dgl
import numpy as np
from dgl.data import TUDataset


from sklearn.model_selection import train_test_split


def build_splits(n: int, labels):
    idx = np.arange(n)
    labels = np.array(labels)
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.2, random_state=42, shuffle=True, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, shuffle=True, stratify=labels[temp_idx]
    )
    return sorted(train_idx.tolist()), sorted(val_idx.tolist()), sorted(test_idx.tolist())


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = TUDataset(name='DBLP_v1')
    raw: List[Tuple[dgl.DGLGraph, int]] = []

    for i in range(len(ds)):
        g, y = ds[i]
        # 节点 token: 使用 node_labels
        if 'node_labels' not in g.ndata:
            raise RuntimeError('DBLP_v1 缺少 node_labels')
        node_labels = g.ndata['node_labels'].view(-1).long()
        g.ndata['node_token_ids'] = node_labels.view(-1, 1)
        g.ndata['node_type_id'] = node_labels

        # 边 token: 使用 edge_labels/edge_label；若缺失则 0
        if 'edge_labels' in g.edata:
            edge_labels = g.edata['edge_labels'].view(-1).long()
        elif 'edge_label' in g.edata:
            edge_labels = g.edata['edge_label'].view(-1).long()
        else:
            raise RuntimeError('DBLP_v1 缺少 edge_labels/edge_label')
        g.edata['edge_token_ids'] = edge_labels.view(-1, 1)
        g.edata['edge_type_id'] = edge_labels

        raw.append((g, int(y)))

    with open(out_dir / 'data.pkl', 'wb') as f:
        pickle.dump(raw, f)

    labels = [y for _, y in raw]
    train, val, test = build_splits(len(ds), labels)
    (out_dir / 'train_index.json').write_text(json.dumps(train), encoding='utf-8')
    (out_dir / 'val_index.json').write_text(json.dumps(val), encoding='utf-8')
    (out_dir / 'test_index.json').write_text(json.dumps(test), encoding='utf-8')
    # 写入汇总统计（全图）
    import torch
    node_uniques, edge_uniques = set(), set()
    for g, _ in raw:
        if g.ndata['node_token_ids'].numel() > 0:
            node_uniques.update(torch.unique(g.ndata['node_token_ids'].view(-1)).tolist())
        if g.edata['edge_token_ids'].numel() > 0:
            edge_uniques.update(torch.unique(g.edata['edge_token_ids'].view(-1)).tolist())
    summary = {
        'num_graphs': len(raw),
        'node_token_unique_count': len({int(v) for v in node_uniques}),
        'edge_token_unique_count': len({int(v) for v in edge_uniques}),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done. Saved to', out_dir)


if __name__ == '__main__':
    main()


