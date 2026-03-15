#!/usr/bin/env python3
from __future__ import annotations

"""
预处理 ogbg-code2：
- 从 OGB 读取图与标签（序列）
- 节点：两维离散特征分别作为两个节点token；第一维 token = 2*v1 + 1；第二维 token = 2*v2 + 1 + bias
- 边：数据集无边属性 → 边token=0
- 将 token 写入 g.ndata['node_token_ids'] / g.edata['edge_token_ids']，并复制到 feat
- 生成 data.pkl 与 train/val/test 索引 JSON
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple
import torch
import dgl
from ogb.graphproppred import DglGraphPropPredDataset


SECOND_CHANNEL_BIAS = 10_000_000


def main() -> None:
    out_dir = Path("data") / "code2"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = DglGraphPropPredDataset(name="ogbg-code2", root=str(Path("data/ogb")))
    split_idx = ds.get_idx_split()

    all_samples: List[Tuple[dgl.DGLGraph, List[str]]] = []
    for i in range(len(ds)):
        g, y = ds[i]
        # 节点两维离散特征（DGL接口下常用键可能为 'feat' 或 'x' 或 'node_feat'）
        x = None
        for k in ('feat', 'x', 'node_feat'):
            if k in g.ndata:
                x = g.ndata[k]
                break
        assert x is not None and x.dim() == 2 and x.size(1) >= 2, "code2节点特征应为 [N,2] 整型"
        v1 = x[:, 0].long().view(-1, 1)
        v2 = x[:, 1].long().view(-1, 1)
        tok1 = v1 * 2 + 1
        tok2 = v2 * 2 + 1 + SECOND_CHANNEL_BIAS
        node_token = torch.cat([tok1, tok2], dim=1).long()
        e = g.num_edges()
        edge_token = torch.zeros((e, 1), dtype=torch.long)

        # 将token直接写回到图
        g.ndata['node_token_ids'] = node_token
        g.ndata['node_type_id'] = v1.view(-1)
        g.ndata['feat'] = node_token
        g.edata['edge_token_ids'] = edge_token
        g.edata['edge_type_id'] = torch.zeros(e, dtype=torch.long)
        g.edata['feat'] = edge_token

        # 清理无关原始键，减少体积（仅保留上述标准键）
        for k in list(g.ndata.keys()):
            if k not in ('node_token_ids', 'node_type_id', 'feat'):
                del g.ndata[k]
        for k in list(g.edata.keys()):
            if k not in ('edge_token_ids', 'edge_type_id', 'feat'):
                del g.edata[k]

        # 标签（序列）
        if isinstance(y, (list, tuple)):
            label = list(y[0]) if len(y) > 0 and isinstance(y[0], (list, tuple)) else list(y)
        elif torch.is_tensor(y):
            # 一些实现中可能返回张量，尽量转成 python 列表
            label = y.detach().cpu().tolist()
        else:
            raise ValueError(f"标签类型不支持: {type(y)}")

        all_samples.append((g, label))

    with (out_dir / 'data.pkl').open('wb') as f:
        pickle.dump(all_samples, f)
    # 不再单独保存 attrs

    # 保存划分索引（使用官方 split）
    for split_name in ['train', 'val', 'test']:
        key = 'valid' if split_name == 'val' else split_name
        idxs = split_idx[key].tolist()
        with (out_dir / f'{split_name}_index.json').open('w') as f:
            json.dump([int(i) for i in idxs], f)

    print(f"✅ 预处理完成: {out_dir}")


if __name__ == '__main__':
    main()


