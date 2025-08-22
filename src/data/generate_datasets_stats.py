#!/usr/bin/env python3
"""
生成数据层数据集统计文档
======================

输出文件：src/data/DATASETS_STATS.md

内容：
- 每个已注册且已完成预处理的数据集：
  - 图数量
  - 节点数/边数的均值、标准差、最小、最大
  - 节点/边 token 维度（Dn/De）
  - （可选）节点/边 token 的唯一值数量（全量统计，若规模很大仍会计算，可能较慢）

注意：仅统计已存在预处理产物的数据集（data/<name>/ 下存在 data.pkl 与三份索引文件）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import sys
import numpy as np
import torch

# 确保项目根目录在 sys.path 中（src/data/ -> src -> project root）
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader, list_available_datasets


@dataclass
class DatasetStats:
    name: str
    total_graphs: int
    nodes_mean: float
    nodes_std: float
    nodes_min: int
    nodes_max: int
    edges_mean: float
    edges_std: float
    edges_min: int
    edges_max: int
    node_token_dim: int
    edge_token_dim: int
    node_token_unique: int | None = None
    edge_token_unique: int | None = None


def _compute_graph_stats(graphs: List[Dict[str, Any]]) -> Tuple[float, float, int, int, float, float, int, int]:
    num_nodes = [int(s['dgl_graph'].num_nodes()) for s in graphs]
    num_edges = [int(s['dgl_graph'].num_edges()) for s in graphs]
    n_mean = float(np.mean(num_nodes)) if num_nodes else 0.0
    n_std = float(np.std(num_nodes)) if num_nodes else 0.0
    n_min = int(np.min(num_nodes)) if num_nodes else 0
    n_max = int(np.max(num_nodes)) if num_nodes else 0
    e_mean = float(np.mean(num_edges)) if num_edges else 0.0
    e_std = float(np.std(num_edges)) if num_edges else 0.0
    e_min = int(np.min(num_edges)) if num_edges else 0
    e_max = int(np.max(num_edges)) if num_edges else 0
    return n_mean, n_std, n_min, n_max, e_mean, e_std, e_min, e_max


def _infer_token_dims(sample_graph) -> Tuple[int, int]:
    g = sample_graph['dgl_graph']
    dn = int(g.ndata['node_token_ids'].shape[1]) if 'node_token_ids' in g.ndata else 0
    de = int(g.edata['edge_token_ids'].shape[1]) if 'edge_token_ids' in g.edata else 0
    return dn, de


def _count_token_uniques(graphs: List[Dict[str, Any]]) -> Tuple[int, int]:
    node_uniques: set[int] = set()
    edge_uniques: set[int] = set()
    for s in graphs:
        g = s['dgl_graph']
        if 'node_token_ids' in g.ndata:
            vals = torch.unique(g.ndata['node_token_ids'].view(-1)).tolist()
            node_uniques.update(int(v) for v in vals)
        if 'edge_token_ids' in g.edata and g.num_edges() > 0:
            vals = torch.unique(g.edata['edge_token_ids'].view(-1)).tolist()
            edge_uniques.update(int(v) for v in vals)
    return len(node_uniques) if node_uniques else 0, len(edge_uniques) if edge_uniques else 0


def main() -> None:
    cfg = ProjectConfig()
    out_path = Path(__file__).resolve().parent / 'DATASETS_STATS.md'

    datasets = list_available_datasets()
    rows: List[DatasetStats] = []

    for ds in datasets:
        try:
            loader = get_dataloader(ds, cfg)
            train_data, val_data, test_data, y_tr, y_va, y_te = loader.load_data()
            graphs = train_data + val_data + test_data
            if not graphs:
                continue

            n_mean, n_std, n_min, n_max, e_mean, e_std, e_min, e_max = _compute_graph_stats(graphs)
            dn, de = _infer_token_dims(graphs[0])
            nu, eu = _count_token_uniques(graphs)

            rows.append(DatasetStats(
                name=loader.dataset_name,
                total_graphs=len(graphs),
                nodes_mean=n_mean,
                nodes_std=n_std,
                nodes_min=n_min,
                nodes_max=n_max,
                edges_mean=e_mean,
                edges_std=e_std,
                edges_min=e_min,
                edges_max=e_max,
                node_token_dim=dn,
                edge_token_dim=de,
                node_token_unique=nu,
                edge_token_unique=eu,
            ))
        except FileNotFoundError:
            # 预处理产物缺失则跳过
            continue
        except Exception as e:
            # 其他异常仅记录并跳过
            print(f"[WARN] 统计失败: {ds}: {e}")
            continue

    # 生成 Markdown
    lines: List[str] = []
    lines.append('# 数据集统计总览\n')
    lines.append('本表统计当前已注册且已完成预处理的数据集规模与特征维度：')
    lines.append('')
    header = (
        '| 数据集 | 图数量 | 节点数均值 | 节点数Std | 节点min | 节点max | 边数均值 | 边数Std | 边min | 边max | 节点Token维度Dn | 边Token维度De | 节点Token唯一 | 边Token唯一 |'
    )
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append(
            f"| {r.name} | {r.total_graphs} | "
            f"{r.nodes_mean:.2f} | {r.nodes_std:.2f} | {r.nodes_min} | {r.nodes_max} | "
            f"{r.edges_mean:.2f} | {r.edges_std:.2f} | {r.edges_min} | {r.edges_max} | "
            f"{r.node_token_dim} | {r.edge_token_dim} | {r.node_token_unique or 0} | {r.edge_token_unique or 0} |"
        )

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"✅ 数据集统计已生成: {out_path}")


if __name__ == '__main__':
    main()


