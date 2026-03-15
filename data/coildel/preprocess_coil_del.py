from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import dgl
import numpy as np
import torch
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

    ds = TUDataset(name='COIL-DEL')
    raw: List[Tuple[dgl.DGLGraph, int]] = []

    for i in range(len(ds)):
        g, y = ds[i]
        # 节点 token: 两列整数化后取积
        if 'node_attr' not in g.ndata:
            raise RuntimeError('COIL-DEL 缺少 node_attr')
        a = g.ndata['node_attr'].numpy()
        if a.ndim != 2 or a.shape[1] < 2:
            raise RuntimeError('COIL-DEL node_attr 维度异常')
        ai = a.astype(np.int64, copy=False)
        prod = (ai[:, 0] * ai[:, 1]).astype(np.int64)
        node_ids_t = torch.from_numpy(prod).long().view(-1, 1)
        g.ndata['node_token_ids'] = node_ids_t
        g.ndata['node_type_id'] = node_ids_t.view(-1)

        # 边 token: 使用 edge_labels（若缺失则 0）
        if 'edge_labels' in g.edata:
            edge_labels = g.edata['edge_labels'].view(-1).long()
        else:
            edge_labels = torch.zeros(g.num_edges(), dtype=torch.long)
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
    print('Done. Saved to', out_dir)


if __name__ == '__main__':
    main()


