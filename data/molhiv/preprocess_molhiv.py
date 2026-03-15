#!/usr/bin/env python3
from __future__ import annotations

"""
预处理 ogbg-molhiv：
- 从 OGB 读取图与标签
- 节点：仅保留第一维（原脚本统计为 atomic_num-1），转为原子序数 Z=val+1，节点token=2*Z+1
- 边：仅保留第一维（边类型，0..3对应 SINGLE/DOUBLE/TRIPLE/AROMATIC；其它映射为 0: NONE/misc），边token=2*etype
- 将 token 写入 g.ndata['node_token_ids'] / g.edata['edge_token_ids']，并复制到 g.ndata['feat'] / g.edata['feat']
- 生成 data.pkl 及 train/val/test 三份索引 JSON
- 删除原始多维属性以减小体积
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import torch
import dgl
from ogb.graphproppred import PygGraphPropPredDataset


BOND_TYPES_OGB_TO_STD = {
    0: 1,  # SINGLE
    1: 2,  # DOUBLE
    2: 3,  # TRIPLE
    3: 4,  # AROMATIC
}


def _to_dgl(pygeo_data) -> dgl.DGLGraph:
    import torch as _torch
    edge_index = pygeo_data.edge_index  # [2, E]
    src = edge_index[0].to(_torch.long)
    dst = edge_index[1].to(_torch.long)
    g = dgl.graph((src, dst), num_nodes=int(pygeo_data.num_nodes))
    return g


def main() -> None:
    out_dir = Path("data") / "molhiv"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(Path("data/ogb")))
    split_idx = ds.get_idx_split()

    all_samples: List[Tuple[dgl.DGLGraph, int]] = []
    for i in range(len(ds)):
        pyg = ds[i]
        g = _to_dgl(pyg)
        # 节点第一维：atomic_num-1 → +1 得 Z
        assert hasattr(pyg, 'x') and pyg.x is not None
        x = pyg.x
        if x.dim() == 1:
            atomic_minus1 = x.long()
        else:
            atomic_minus1 = x[:, 0].long()
        atomic_num = (atomic_minus1 + 1).view(-1, 1)
        node_token = (atomic_num * 2 + 1).long()
        g.ndata['node_token_ids'] = node_token
        g.ndata['node_type_id'] = atomic_num.view(-1)
        g.ndata['feat'] = node_token

        # 边第一维：边类型（0..3），映射到标准 1..4；其他→0
        edge_attr = getattr(pyg, 'edge_attr', None)
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                raw = edge_attr.long().view(-1)
            else:
                raw = edge_attr[:, 0].long().view(-1)
            mapped = torch.zeros_like(raw)
            for k, v in BOND_TYPES_OGB_TO_STD.items():
                mapped[raw == k] = v
            edge_type = mapped.view(-1)
        else:
            edge_type = torch.zeros(g.num_edges(), dtype=torch.long)
        edge_token = (edge_type * 2).view(-1, 1)
        g.edata['edge_token_ids'] = edge_token
        g.edata['edge_type_id'] = edge_type
        g.edata['feat'] = edge_token

        # 清理其他属性以减小体积
        for k in list(g.ndata.keys()):
            if k not in ('node_token_ids', 'node_type_id', 'feat'):
                del g.ndata[k]
        for k in list(g.edata.keys()):
            if k not in ('edge_token_ids', 'edge_type_id', 'feat'):
                del g.edata[k]

        # 标签（binary）
        y = pyg.y
        label = int(y.view(-1)[0].item())
        all_samples.append((g, label))

    with (out_dir / 'data.pkl').open('wb') as f:
        pickle.dump(all_samples, f)

    # 保存划分索引
    for split_name in ['train', 'val', 'test']:
        key = 'valid' if split_name == 'val' else split_name
        idxs = split_idx[key].tolist()
        with (out_dir / f'{split_name}_index.json').open('w') as f:
            json.dump([int(i) for i in idxs], f)

    print(f"✅ 预处理完成: {out_dir}")


if __name__ == '__main__':
    main()


