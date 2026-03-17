from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import dgl
import numpy as np
import torch
from dgl.data import TUDataset
def multihot_row_to_int(row: np.ndarray) -> int:
    # 将二值多热向量编码为整数（bit 编码）
    val = 0
    for i, b in enumerate(row.astype(np.int64).tolist()):
        if int(b) != 0:
            val |= (1 << i)
    return int(val)


def build_splits(n: int, labels):
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n))
    return train_idx, val_idx, test_idx


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = TUDataset(name='COLORS-3')
    raw: List[Tuple[dgl.DGLGraph, int]] = []

    for i in range(len(ds)):
        g, y = ds[i]
        # 节点多/one-hot -> 单一整数
        if 'node_attr' not in g.ndata:
            raise RuntimeError('COLORS-3 缺少 node_attr')
        na = g.ndata['node_attr'].numpy()
        # 允许 one-hot 或 multihot，统一按 bit 编码
        node_ids = np.array([multihot_row_to_int(row) for row in na], dtype=np.int64)
        node_ids_t = torch.from_numpy(node_ids).long().view(-1, 1)
        g.ndata['node_token_ids'] = node_ids_t
        g.ndata['node_type_id'] = node_ids_t.view(-1)

        # 边：无属性，统一置 0
        ecount = g.num_edges()
        edge_tok = torch.zeros(ecount, dtype=torch.long).view(-1, 1)
        g.edata['edge_token_ids'] = edge_tok
        g.edata['edge_type_id'] = edge_tok.view(-1)

        raw.append((g, int(y)))

    # 保存统一数据文件
    data_path = out_dir / 'data.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(raw, f)

    # 构建并保存索引划分
    labels = [y for _, y in raw]
    train, val, test = build_splits(len(ds), labels)
    (out_dir / 'train_index.json').write_text(json.dumps(train), encoding='utf-8')
    (out_dir / 'val_index.json').write_text(json.dumps(val), encoding='utf-8')
    (out_dir / 'test_index.json').write_text(json.dumps(test), encoding='utf-8')

    print('Done. Saved to', out_dir)


if __name__ == '__main__':
    main()

