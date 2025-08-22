#!/usr/bin/env python3
"""
为所有（或指定）数据集，将 token 写入标准特征键：g.ndata['feat'] / g.edata['feat']，并重新保存 data.pkl。

规则：
- 节点特征 feat = node_token_ids（若缺失则优先 node_type_id.view(-1,1)，仍缺失则报错）
- 边特征   feat = edge_token_ids（若缺失则优先 edge_type_id.view(-1,1)，仍缺失则 E=0 则空，否则全 0）

保存策略：
- 默认非覆盖：写到 data/gnn_use/<dataset>/data.pkl；同步复制三份划分 JSON
- 可用 --inplace 覆盖原 data.pkl（可选 --backup 生成 .bak）

示例：
  python src/data/add_feat_from_tokens.py --datasets "proteins,mutagenicity"
  python src/data/add_feat_from_tokens.py --inplace --backup   # 处理全部可用数据集
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

# 早期数据集：使用其 loader 的 get_graph_*_token_ids 接口获取 token
_EARLY_DATASETS = {"qm9", "qm9test", "zinc", "aqsol"}


def _ensure_long_2d(t: torch.Tensor, num_rows: int) -> torch.Tensor:
    t = t.long()
    if t.dim() == 1:
        t = t.view(-1, 1)
    assert t.shape[0] == num_rows and t.shape[1] == 1, f"feat 形状不符，期望 [{num_rows},1]，得到 {tuple(t.shape)}"
    return t


def _add_feat_to_graph(g, loader=None) -> None:
    # 节点
    n = int(g.num_nodes())
    if loader is not None:
        node_tok = loader.get_graph_node_token_ids(g)
    elif 'node_token_ids' in g.ndata:
        node_tok = g.ndata['node_token_ids']
    elif 'node_type_id' in g.ndata:
        node_tok = g.ndata['node_type_id'].view(-1, 1)
    else:
        raise KeyError("图缺少 node_token_ids 与 node_type_id，无法构建节点 feat")
    node_tok = _ensure_long_2d(node_tok, n)
    # 统一保证：节点token为奇数空间（与边错位），若不是则映射为 2*x+1
    if node_tok.numel() > 0:
        if torch.any((node_tok.view(-1) % 2) == 0):
            node_tok = (node_tok * 2 + 1).long()
    g.ndata['feat'] = node_tok

    # 边
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
        # 统一保证：边token为偶数空间（与节点错位），若不是则映射为 2*x
        if edge_tok.numel() > 0:
            if torch.any((edge_tok.view(-1) % 2) != 0):
                edge_tok = (edge_tok * 2).long()
        g.edata['feat'] = edge_tok


def _strip_to_feat_only(g) -> None:
    # 仅保留 feat，删除其它 ndata/edata 键以减小体积
    for k in list(g.ndata.keys()):
        if k != 'feat':
            del g.ndata[k]
    for k in list(g.edata.keys()):
        if k != 'feat':
            del g.edata[k]


def _print_feat_summary(ds_name: str, payload: List[Any], limit: int = 2) -> None:
    print(f"--- 数据集: {ds_name} 的 feat 概览（最多显示 {limit} 个样本） ---")
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
        print(f"样本{shown}: N={n}, E={e}; node_feat={nf_shape}/{nf_dtype} 例: {nf_head}; edge_feat={ef_shape}/{ef_dtype} 例: {ef_head}")
        shown += 1
    if shown == 0:
        print("无样本可展示")


def _process_dataset(ds_name: str, inplace: bool, backup: bool, cfg: ProjectConfig) -> Tuple[str, int]:
    data_dir = Path('data') / ds_name
    src_file = data_dir / 'data.pkl'
    if not src_file.exists():
        return '', 0

    # 早期数据集：用 loader 加载并按 split 重建原顺序
    if ds_name in _EARLY_DATASETS:
        try:
            loader = get_dataloader(ds_name, cfg)
            train_data, val_data, test_data, _, _, _ = loader.load_data()
            # 统一补齐 token 到图上
            try:
                all_samples = train_data + val_data + test_data
                if hasattr(loader, '_build_attribute_cache'):
                    loader._build_attribute_cache(all_samples)  # 写入 node/edge_type_id 与 node/edge_token_ids
            except Exception as e:
                print(f"[WARN] 早期数据集补齐 token 失败（{ds_name}）：{e}")
            # 读取原始索引
            with (data_dir / 'train_index.json').open('r') as f:
                train_idx = json.load(f)
            with (data_dir / 'val_index.json').open('r') as f:
                val_idx = json.load(f)
            with (data_dir / 'test_index.json').open('r') as f:
                test_idx = json.load(f)
            total_len = max(train_idx + val_idx + test_idx) + 1 if (train_idx or val_idx or test_idx) else (len(train_data) + len(val_data) + len(test_data))
            assembled: List[Any] = [None] * total_len
            # 放回原位置：样本 -> (graph, properties)
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
            # 去除可能的 None（若索引不连续）
            payload = [x for x in assembled if x is not None]
            updated = len(payload)
        except Exception as e:
            print(f"[WARN] 早期数据集经 loader 处理失败，回退直接读图字段（{ds_name}）：{e}")
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
                        assert g is not None, '样本缺少 dgl_graph'
                        _add_feat_to_graph(g, loader=None)
                        updated += 1
                else:
                    raise TypeError('不支持的 data.pkl 元素类型')
    else:
        # 常规数据集：直接读取并写入 feat
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
                    assert g is not None, '样本缺少 dgl_graph'
                    _add_feat_to_graph(g, loader=None)
                    updated += 1
            else:
                raise TypeError('不支持的 data.pkl 元素类型')

    # 目标目录与保存
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

    # 若写入新目录（默认），仅保留 feat 以减少体积
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

    # 打印处理后 feat 情况（保存前）
    try:
        _print_feat_summary(ds_name, payload, limit=2)
    except Exception as e:
        print(f"[WARN] 打印 feat 概览失败: {e}")

    with out_file.open('wb') as f:
        pickle.dump(payload, f)

    # 复制划分 JSON 到目标目录
    for name in ['train_index.json', 'val_index.json', 'test_index.json']:
        src = data_dir / name
        if src.exists():
            dst = out_dir / name
            shutil.copy2(src, dst)

    return str(out_file), updated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default=None, help='逗号分隔数据集名称；未提供则处理全部可用数据集')
    ap.add_argument('--inplace', action='store_true', help='原地覆盖 data.pkl')
    ap.add_argument('--backup', action='store_true', help='inplace 时备份为 data.pkl.bak')
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
            print(f"[{ds}] 写入: {out_file} (更新图数: {cnt})")
        else:
            print(f"[{ds}] 跳过：未找到 data/{ds}/data.pkl")
        total += cnt
    print(f"✅ 完成。共更新 {total} 个图。")


if __name__ == '__main__':
    main()


