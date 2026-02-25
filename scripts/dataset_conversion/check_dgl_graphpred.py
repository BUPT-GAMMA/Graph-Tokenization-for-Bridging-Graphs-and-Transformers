from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import dgl
from dgl.data import TUDataset, BA2MotifDataset


@dataclass
class Quantiles:
    p50: float
    p90: float
    p95: float
    p99: float


@dataclass
class ScalarStats:
    minimum: float
    maximum: float
    mean: float
    std: float
    quantiles: Quantiles


@dataclass
class FeatureInfo:
    exists: bool
    keys: List[str]
    main_key: Optional[str]
    dim: Optional[int]
    dtype: Optional[str]
    is_discrete_like: Optional[bool]
    is_one_hot_like: Optional[bool]
    example_values: Optional[List[List[float]]]


@dataclass
class LabelSummary:
    shape: List[int]
    dtype: str
    task_type: str
    details: Dict[str, Any]


@dataclass
class DatasetReport:
    name: str
    num_graphs: int
    node_count: ScalarStats
    edge_count: ScalarStats
    node_features: FeatureInfo
    edge_features: FeatureInfo
    graph_label_shape: List[int]
    graph_label_dtype: str
    # legacy single-source summary
    node_token_key: Optional[str]
    node_token_unique_count: Optional[int]
    edge_token_key: Optional[str]
    edge_token_unique_count: Optional[int]
    node_token_source: Optional[str]
    edge_token_source: Optional[str]
    node_token_notes: Optional[str]
    edge_token_notes: Optional[str]
    # dual-source breakdown
    node_label_token_key: Optional[str]
    node_label_token_unique_count: Optional[int]
    node_label_token_mode: Optional[str]
    node_label_token_notes: Optional[str]
    node_attr_token_key: Optional[str]
    node_attr_token_unique_count: Optional[int]
    node_attr_token_mode: Optional[str]
    node_attr_token_notes: Optional[str]
    edge_label_token_key: Optional[str]
    edge_label_token_unique_count: Optional[int]
    edge_label_token_mode: Optional[str]
    edge_label_token_notes: Optional[str]
    edge_attr_token_key: Optional[str]
    edge_attr_token_unique_count: Optional[int]
    edge_attr_token_mode: Optional[str]
    edge_attr_token_notes: Optional[str]


def compute_scalar_stats(values: List[int]) -> ScalarStats:
    if not values:
        raise ValueError("values 为空，无法计算统计量")
    values_sorted = sorted(values)
    minimum = float(values_sorted[0])
    maximum = float(values_sorted[-1])
    mean = float(sum(values_sorted) / len(values_sorted))
    std = float(statistics.pstdev(values_sorted)) if len(values_sorted) > 1 else 0.0

    def percentile(vs: List[int], p: float) -> float:
        if not vs:
            return float("nan")
        k = (len(vs) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(vs[int(k)])
        d0 = vs[f] * (c - k)
        d1 = vs[c] * (k - f)
        return float(d0 + d1)

    quantiles = Quantiles(
        p50=percentile(values_sorted, 0.50),
        p90=percentile(values_sorted, 0.90),
        p95=percentile(values_sorted, 0.95),
        p99=percentile(values_sorted, 0.99),
    )
    return ScalarStats(minimum=minimum, maximum=maximum, mean=mean, std=std, quantiles=quantiles)


def _pick_ndata_key(g: dgl.DGLGraph) -> Tuple[Optional[str], List[str]]:
    keys = list(g.ndata.keys())
    main_key: Optional[str] = None
    # 优先使用 TU/DGL 常见键名
    for k in (
        "node_attr",  # 优先节点属性
        "node_labels",
        "node_label",
        "feat",
        "label",
        "labels",
        "attr",
    ):
        if k in g.ndata:
            main_key = k
            break
    return main_key, keys


def _pick_edata_key(g: dgl.DGLGraph) -> Tuple[Optional[str], List[str]]:
    keys = list(g.edata.keys())
    main_key: Optional[str] = None
    for k in (
        "edge_attr",  # 优先边属性
        "edge_labels",
        "edge_label",
        "feat",
        "label",
        "labels",
        "attr",
    ):
        if k in g.edata:
            main_key = k
            break
    return main_key, keys


def _infer_discrete_and_onehot(x: np.ndarray) -> Tuple[bool, bool]:
    if x.ndim == 1:
        uniq = np.unique(x)
        if np.issubdtype(x.dtype, np.integer):
            return True, False
        if np.all(np.isin(uniq, [0.0, 1.0])):
            return True, False
        return False, False
    if x.ndim == 2:
        if x.shape[1] <= 64:
            vals = x
        else:
            vals = x[:, :64]
        uniq = np.unique(vals)
        if np.all(np.isin(uniq, [0.0, 1.0])):
            row_sums = vals.sum(axis=1)
            one_hot_like = np.all((row_sums == 1) | (row_sums == 0))
            return True, bool(one_hot_like)
        if np.issubdtype(vals.dtype, np.integer):
            return True, False
        return False, False
    return False, False


def summarize_feature(g: dgl.DGLGraph, is_node: bool) -> FeatureInfo:
    main_key, keys = _pick_ndata_key(g) if is_node else _pick_edata_key(g)
    if main_key is None:
        return FeatureInfo(False, keys=keys, main_key=None, dim=None, dtype=None, is_discrete_like=None, is_one_hot_like=None, example_values=None)
    arr = g.ndata[main_key] if is_node else g.edata[main_key]
    if hasattr(arr, "numpy"):
        a = arr.numpy()
    else:
        a = np.asarray(arr)
    if a.ndim == 1:
        dim = 1
        sample = a[: min(8, a.shape[0])].reshape(-1, 1)
    elif a.ndim == 2:
        dim = int(a.shape[1])
        sample = a[: min(3, a.shape[0]), : min(8, a.shape[1])]
    else:
        dim = None
        sample = None
    is_discrete_like, is_one_hot_like = _infer_discrete_and_onehot(a)
    dtype = str(a.dtype)
    example_values = sample.tolist() if sample is not None else None
    return FeatureInfo(True, keys=keys, main_key=main_key, dim=dim, dtype=dtype, is_discrete_like=is_discrete_like, is_one_hot_like=is_one_hot_like, example_values=example_values)


def _pick_token_key_and_mode(g: dgl.DGLGraph, is_node: bool, dataset_name: str) -> Tuple[Optional[str], Optional[str]]:
    # 优先离散标签，然后属性（支持 onehot/multihot/int），并特殊处理 COIL-DEL 的二维连续坐标
    store = g.ndata if is_node else g.edata
    label_keys = ["node_labels", "label", "labels"] if is_node else ["edge_labels", "label", "labels"]
    attr_keys = ["node_attr", "feat", "attr"] if is_node else ["edge_attr", "feat", "attr"]

    # 先尝试标签（1D 整型）
    for k in label_keys:
        if k in store:
            a = store[k].numpy() if hasattr(store[k], "numpy") else np.asarray(store[k])
            if a.ndim == 2 and a.shape[1] == 1:
                a = a.reshape(-1)
            if a.ndim == 1 and np.issubdtype(a.dtype, np.integer):
                return k, "int"

    # 再尝试属性
    for k in attr_keys:
        if k not in store:
            continue
        a = store[k].numpy() if hasattr(store[k], "numpy") else np.asarray(store[k])
        if a.ndim == 2 and a.shape[1] == 1:
            a = a.reshape(-1)
        # 1D int 直接可用
        if a.ndim == 1 and np.issubdtype(a.dtype, np.integer):
            return k, "int"
        # 二维二值矩阵：区分 onehot 与 multihot
        if a.ndim == 2:
            probe = a if a.shape[1] <= 128 else a[:, :128]
            uniq = np.unique(probe)
            if np.all(np.isin(uniq, [0.0, 1.0])):
                row_sums = a.sum(axis=1)
                if np.all((row_sums == 1) | (row_sums == 0)):
                    return k, "onehot"
                else:
                    return k, "multihot"
        # COIL-DEL 特例：二维连续（float）坐标样式
        if is_node and dataset_name == "COIL-DEL" and a.ndim == 2 and a.shape[1] == 2 and np.issubdtype(a.dtype, np.floating):
            return k, "coil_del_2col_float"
    return None, None


def _accumulate_token_uniques(ds, is_node: bool, limit_graphs: Optional[int]) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    uniques: Optional[set] = None
    chosen_key: Optional[str] = None
    notes: Optional[str] = None
    total = len(ds) if limit_graphs is None else min(limit_graphs, len(ds))
    for i in range(total):
        g, _ = ds[i]
        key, mode = _pick_token_key_and_mode(g, is_node=is_node, dataset_name=getattr(ds, 'name', getattr(ds, '_name', '')))
        if key is None or mode is None:
            continue
        store = g.ndata if is_node else g.edata
        arr = store[key]
        a = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a.reshape(-1)
        tokens: Optional[np.ndarray] = None
        if mode == "int":
            tokens = a.astype(np.int64, copy=False) if a.ndim == 1 else None
        elif mode == "onehot":
            row_sums = a.sum(axis=1)
            mask = row_sums > 0
            if mask.any():
                tokens = np.argmax(a[mask], axis=1).astype(np.int64, copy=False)
        elif mode == "multihot":
            if uniques is None:
                uniques = set()
                chosen_key = key
            for row in a:
                uniques.add(tuple(row.tolist()))
            tokens = None
        elif mode == "coil_del_2col_float":
            col0 = np.unique(a[:, 0]).tolist()
            col1 = np.unique(a[:, 1]).tolist()
            if uniques is None:
                uniques = set()
                chosen_key = key
            for v in col0:
                uniques.add(("col0", float(v)))
            for v in col1:
                uniques.add(("col1", float(v)))
            notes = "sum(unique(col0), unique(col1))"
            tokens = None
        if tokens is None:
            pass
        else:
            if uniques is None:
                uniques = set()
                chosen_key = key
            if tokens.ndim == 1:
                uniques.update(tokens.tolist())
            else:
                uniques.update(tokens.reshape(-1).tolist())
    if uniques is None:
        return None, None, notes
    return chosen_key, len(uniques), notes


def _detect_key_mode(store: Any, key: str, is_node: bool, dataset_name: str) -> Tuple[Optional[str], Optional[str]]:
    if key not in store:
        return None, None
    a = store[key].numpy() if hasattr(store[key], "numpy") else np.asarray(store[key])
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.reshape(-1)
    # label-like: expect 1D int
    if key.endswith("labels") or key.endswith("label") or key in ("label", "labels"):
        if a.ndim == 1 and np.issubdtype(a.dtype, np.integer):
            return key, "int"
        return None, None
    # attr-like
    if a.ndim == 1 and np.issubdtype(a.dtype, np.integer):
        return key, "int"
    if a.ndim == 2:
        probe = a if a.shape[1] <= 128 else a[:, :128]
        uniq = np.unique(probe)
        if np.all(np.isin(uniq, [0.0, 1.0])):
            row_sums = a.sum(axis=1)
            if np.all((row_sums == 1) | (row_sums == 0)):
                return key, "onehot"
            else:
                return key, "multihot"
    # 连续情况也要展示（便于人工后续决策）
    if a.ndim == 1 and np.issubdtype(a.dtype, np.floating):
        return key, "cont1d"
    if a.ndim == 2 and np.issubdtype(a.dtype, np.floating):
        if is_node and dataset_name == "COIL-DEL" and a.shape[1] == 2:
            return key, "coil_del_2col_float"
        return key, "cont2d"
    return None, None


def _accumulate_for_keys(ds, is_node: bool, limit_graphs: Optional[int], keys: List[str]) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    uniques: Optional[set] = None
    chosen_key: Optional[str] = None
    chosen_mode: Optional[str] = None
    notes: Optional[str] = None
    # For COIL-DEL aggregate stats across graphs when encountering 2-col float
    col0_all: set = set()
    col1_all: set = set()
    pairs_all: set = set()
    total = len(ds) if limit_graphs is None else min(limit_graphs, len(ds))
    dataset_name = getattr(ds, 'name', getattr(ds, '_name', ''))
    for i in range(total):
        g, _ = ds[i]
        store = g.ndata if is_node else g.edata
        # pick the first usable key according to priority list
        km: Optional[Tuple[str, str]] = None
        for k in keys:
            k2, m2 = _detect_key_mode(store, k, is_node=is_node, dataset_name=dataset_name)
            if k2 is not None and m2 is not None:
                km = (k2, m2)
                break
        if km is None:
            continue
        key, mode = km
        arr = store[key]
        a = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a.reshape(-1)
        tokens: Optional[np.ndarray] = None
        if mode == "int":
            tokens = a.astype(np.int64, copy=False) if a.ndim == 1 else None
        elif mode == "onehot":
            row_sums = a.sum(axis=1)
            mask = row_sums > 0
            if mask.any():
                tokens = np.argmax(a[mask], axis=1).astype(np.int64, copy=False)
        elif mode == "multihot":
            if uniques is None:
                uniques = set()
                chosen_key = key
                chosen_mode = mode
            for row in a:
                uniques.add(tuple(row.tolist()))
            tokens = None
        elif mode == "coil_del_2col_float":
            col0 = np.unique(a[:, 0]).tolist()
            col1 = np.unique(a[:, 1]).tolist()
            if uniques is None:
                uniques = set()
                chosen_key = key
                chosen_mode = mode
            for v in col0:
                v = float(v)
                uniques.add(("col0", v))
                col0_all.add(v)
            for v in col1:
                v = float(v)
                uniques.add(("col1", v))
                col1_all.add(v)
            # 额外：成对唯一数量（全局累积）
            for r in a:
                pairs_all.add((float(r[0]), float(r[1])))
            tokens = None
        elif mode == "cont1d":
            # 连续一维：统计原始唯一值数量
            vals = np.unique(a).tolist()
            if uniques is None:
                uniques = set()
                chosen_key = key
                chosen_mode = mode
            # 用特殊前缀避免与其它模式冲突
            for v in vals:
                uniques.add(("cont1d", float(v)))
            tokens = None
        elif mode == "cont2d":
            # 连续二维：按行去重计数
            if uniques is None:
                uniques = set()
                chosen_key = key
                chosen_mode = mode
            for r in a:
                uniques.add(tuple([float(x) for x in r.tolist()]))
            tokens = None
        if tokens is None:
            pass
        else:
            if uniques is None:
                uniques = set()
                chosen_key = key
                chosen_mode = mode
            if tokens.ndim == 1:
                uniques.update(tokens.tolist())
            else:
                uniques.update(tokens.reshape(-1).tolist())
    if uniques is None:
        return None, None, None, notes
    # 汇总 COIL-DEL 的聚合备注
    if chosen_mode == "coil_del_2col_float":
        notes = f"sum_unique_cols={len(col0_all)+len(col1_all)}; pair_unique={len(pairs_all)}"
    return chosen_key, len(uniques), chosen_mode, notes


def _accumulate_label_and_attr_uniques(ds, is_node: bool, limit_graphs: Optional[int]) -> Tuple[Tuple[Optional[str], Optional[int], Optional[str], Optional[str]], Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]]:
    label_keys = ["node_labels", "label", "labels"] if is_node else ["edge_labels", "label", "labels"]
    attr_keys = ["node_attr", "feat", "attr"] if is_node else ["edge_attr", "feat", "attr"]
    label_res = _accumulate_for_keys(ds, is_node=is_node, limit_graphs=limit_graphs, keys=label_keys)
    attr_res = _accumulate_for_keys(ds, is_node=is_node, limit_graphs=limit_graphs, keys=attr_keys)
    return label_res, attr_res


def analyze_tu(name: str, limit_graphs: Optional[int]) -> DatasetReport:
    ds = TUDataset(name=name)
    if limit_graphs is not None:
        n = min(limit_graphs, len(ds))
        indices = list(range(n))
    else:
        indices = list(range(len(ds)))

    node_counts: List[int] = []
    edge_counts: List[int] = []
    first_g: Optional[dgl.DGLGraph] = None
    first_y_shape: List[int] = []
    first_y_dtype: str = "unknown"

    for i in indices:
        g, y = ds[i]
        if first_g is None:
            first_g = g
            y_arr = np.array(y)
            first_y_shape = list(y_arr.shape)
            first_y_dtype = str(y_arr.dtype)
        node_counts.append(int(g.num_nodes()))
        edge_counts.append(int(g.num_edges()))

    node_stats = compute_scalar_stats(node_counts)
    edge_stats = compute_scalar_stats(edge_counts)
    nf = summarize_feature(first_g, is_node=True) if first_g is not None else FeatureInfo(False, [], None, None, None, None, None, None)
    ef = summarize_feature(first_g, is_node=False) if first_g is not None else FeatureInfo(False, [], None, None, None, None, None, None)

    node_token_key, node_unique, node_notes = _accumulate_token_uniques(ds, is_node=True, limit_graphs=limit_graphs)
    edge_token_key, edge_unique, edge_notes = _accumulate_token_uniques(ds, is_node=False, limit_graphs=limit_graphs)
    # Dual path accumulation (label vs attr), to display both if present
    (n_lbl_key, n_lbl_unique, n_lbl_mode, n_lbl_notes), (n_attr_key, n_attr_unique, n_attr_mode, n_attr_notes) = _accumulate_label_and_attr_uniques(ds, is_node=True, limit_graphs=limit_graphs)
    (e_lbl_key, e_lbl_unique, e_lbl_mode, e_lbl_notes), (e_attr_key, e_attr_unique, e_attr_mode, e_attr_notes) = _accumulate_label_and_attr_uniques(ds, is_node=False, limit_graphs=limit_graphs)
    label_like = {"node_labels", "label", "labels"}
    edge_label_like = {"edge_labels", "label", "labels"}
    node_source = ("label" if (node_token_key in label_like) else ("attr" if node_token_key is not None else None))
    edge_source = ("label" if (edge_token_key in edge_label_like) else ("attr" if edge_token_key is not None else None))

    return DatasetReport(
        name=name,
        num_graphs=len(ds),
        node_count=node_stats,
        edge_count=edge_stats,
        node_features=nf,
        edge_features=ef,
        graph_label_shape=first_y_shape,
        graph_label_dtype=first_y_dtype,
        node_token_key=node_token_key,
        node_token_unique_count=node_unique,
        edge_token_key=edge_token_key,
        edge_token_unique_count=edge_unique,
        node_token_source=node_source,
        edge_token_source=edge_source,
        node_token_notes=node_notes,
        edge_token_notes=edge_notes,
        node_label_token_key=n_lbl_key,
        node_label_token_unique_count=n_lbl_unique,
        node_label_token_mode=n_lbl_mode,
        node_label_token_notes=n_lbl_notes,
        node_attr_token_key=n_attr_key,
        node_attr_token_unique_count=n_attr_unique,
        node_attr_token_mode=n_attr_mode,
        node_attr_token_notes=n_attr_notes,
        edge_label_token_key=e_lbl_key,
        edge_label_token_unique_count=e_lbl_unique,
        edge_label_token_mode=e_lbl_mode,
        edge_label_token_notes=e_lbl_notes,
        edge_attr_token_key=e_attr_key,
        edge_attr_token_unique_count=e_attr_unique,
        edge_attr_token_mode=e_attr_mode,
        edge_attr_token_notes=e_attr_notes,
    )


def analyze_ba2motif(n_graphs: int, seed: int) -> DatasetReport:
    ds = BA2MotifDataset(n_graphs=n_graphs, seed=seed)
    node_counts: List[int] = []
    edge_counts: List[int] = []
    first_g: Optional[dgl.DGLGraph] = None
    first_y_shape: List[int] = []
    first_y_dtype: str = "unknown"

    for i in range(len(ds)):
        g, y = ds[i]
        if first_g is None:
            first_g = g
            y_arr = np.array(y)
            first_y_shape = list(y_arr.shape)
            first_y_dtype = str(y_arr.dtype)
        node_counts.append(int(g.num_nodes()))
        edge_counts.append(int(g.num_edges()))

    node_stats = compute_scalar_stats(node_counts)
    edge_stats = compute_scalar_stats(edge_counts)
    nf = summarize_feature(first_g, is_node=True) if first_g is not None else FeatureInfo(False, [], None, None, None, None, None, None)
    ef = summarize_feature(first_g, is_node=False) if first_g is not None else FeatureInfo(False, [], None, None, None, None, None, None)

    node_token_key, node_unique, node_notes = _accumulate_token_uniques(ds, is_node=True, limit_graphs=None)
    edge_token_key, edge_unique, edge_notes = _accumulate_token_uniques(ds, is_node=False, limit_graphs=None)
    (n_lbl_key, n_lbl_unique, n_lbl_mode, n_lbl_notes), (n_attr_key, n_attr_unique, n_attr_mode, n_attr_notes) = _accumulate_label_and_attr_uniques(ds, is_node=True, limit_graphs=None)
    (e_lbl_key, e_lbl_unique, e_lbl_mode, e_lbl_notes), (e_attr_key, e_attr_unique, e_attr_mode, e_attr_notes) = _accumulate_label_and_attr_uniques(ds, is_node=False, limit_graphs=None)
    label_like = {"node_labels", "label", "labels"}
    edge_label_like = {"edge_labels", "label", "labels"}
    node_source = ("label" if (node_token_key in label_like) else ("attr" if node_token_key is not None else None))
    edge_source = ("label" if (edge_token_key in edge_label_like) else ("attr" if edge_token_key is not None else None))

    return DatasetReport(
        name=f"BA2MotifDataset(n_graphs={n_graphs},seed={seed})",
        num_graphs=len(ds),
        node_count=node_stats,
        edge_count=edge_stats,
        node_features=nf,
        edge_features=ef,
        graph_label_shape=first_y_shape,
        graph_label_dtype=first_y_dtype,
        node_token_key=node_token_key,
        node_token_unique_count=node_unique,
        edge_token_key=edge_token_key,
        edge_token_unique_count=edge_unique,
        node_token_source=node_source,
        edge_token_source=edge_source,
        node_token_notes=node_notes,
        edge_token_notes=edge_notes,
        node_label_token_key=n_lbl_key,
        node_label_token_unique_count=n_lbl_unique,
        node_label_token_mode=n_lbl_mode,
        node_label_token_notes=n_lbl_notes,
        node_attr_token_key=n_attr_key,
        node_attr_token_unique_count=n_attr_unique,
        node_attr_token_mode=n_attr_mode,
        node_attr_token_notes=n_attr_notes,
        edge_label_token_key=e_lbl_key,
        edge_label_token_unique_count=e_lbl_unique,
        edge_label_token_mode=e_lbl_mode,
        edge_label_token_notes=e_lbl_notes,
        edge_attr_token_key=e_attr_key,
        edge_attr_token_unique_count=e_attr_unique,
        edge_attr_token_mode=e_attr_mode,
        edge_attr_token_notes=e_attr_notes,
    )


def print_report(report: DatasetReport) -> None:
    print("=" * 80)
    print(f"数据集: {report.name}")
    print(f"图数量: {report.num_graphs}")

    def _fmt_stats(stats: ScalarStats, unit: str) -> None:
        print(f"{unit}统计: min={stats.minimum:.0f}, max={stats.maximum:.0f}, mean={stats.mean:.2f}, std={stats.std:.2f}")
        q = stats.quantiles
        print(f"{unit}分位数: p50={q.p50:.0f}, p90={q.p90:.0f}, p95={q.p95:.0f}, p99={q.p99:.0f}")

    _fmt_stats(report.node_count, "节点数")
    _fmt_stats(report.edge_count, "边数")

    nf = report.node_features
    print(f"节点特征: exists={nf.exists}, keys={nf.keys}, main_key={nf.main_key}, dim={nf.dim}, dtype={nf.dtype}, 离散样式={nf.is_discrete_like}, onehot样式={nf.is_one_hot_like}")
    if nf.example_values is not None:
        print("节点特征示例(前3行x前8列):")
        for row in nf.example_values:
            print(f"  {row}")

    ef = report.edge_features
    print(f"边特征: exists={ef.exists}, keys={ef.keys}, main_key={ef.main_key}, dim={ef.dim}, dtype={ef.dtype}, 离散样式={ef.is_discrete_like}, onehot样式={ef.is_one_hot_like}")
    if ef.example_values is not None:
        print("边特征示例(前3行x前8列):")
        for row in ef.example_values:
            print(f"  {row}")

    print(f"图标签: shape={report.graph_label_shape}, dtype={report.graph_label_dtype}")
    print(f"节点 token 源(自动选择): source={report.node_token_source}, key={report.node_token_key}, unique_tokens={report.node_token_unique_count}, notes={report.node_token_notes}")
    print(f"  节点 label 路径: key={report.node_label_token_key}, unique_tokens={report.node_label_token_unique_count}, mode={report.node_label_token_mode}, notes={report.node_label_token_notes}")
    print(f"  节点 attr  路径: key={report.node_attr_token_key}, unique_tokens={report.node_attr_token_unique_count}, mode={report.node_attr_token_mode}, notes={report.node_attr_token_notes}")
    print(f"边 token 源(自动选择): source={report.edge_token_source}, key={report.edge_token_key}, unique_tokens={report.edge_token_unique_count}, notes={report.edge_token_notes}")
    print(f"  边   label 路径: key={report.edge_label_token_key}, unique_tokens={report.edge_label_token_unique_count}, mode={report.edge_label_token_mode}, notes={report.edge_label_token_notes}")
    print(f"  边   attr  路径: key={report.edge_attr_token_key}, unique_tokens={report.edge_attr_token_unique_count}, mode={report.edge_attr_token_mode}, notes={report.edge_attr_token_notes}")

    # 按用户要求：不输出任何“建议”类文案


def save_report_json(report: DatasetReport, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{report.name.replace('/', '_')}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    return path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查并汇总 DGL 图级任务数据集（TU/BA2Motif 等）")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "PROTEINS",
            "COLORS-3",
            "SYNTHETIC",
            "Mutagenicity",
            "COIL-DEL",
            "DBLP_v1",
            "DD",
            "TWITTER-Real-Graph-Partial",
        ],
        help="需要检查的 TUDataset 名称列表；BA2Motif 用 --ba2motif 指定",
    )
    parser.add_argument(
        "--limit-graphs",
        type=int,
        default=None,
        help="限制统计的图数量（默认: None=全部）",
    )
    parser.add_argument(
        "--ba2motif",
        action="store_true",
        help="是否同时检查 BA2MotifDataset",
    )
    parser.add_argument(
        "--ba2motif-n",
        type=int,
        default=200,
        help="BA2MotifDataset 的图数量",
    )
    parser.add_argument(
        "--ba2motif-seed",
        type=int,
        default=0,
        help="BA2MotifDataset 的随机种子",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="若提供，将保存各数据集 JSON 摘要到该目录",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    for name in args.datasets:
        print(f"开始分析数据集：{name} ...")
        report = analyze_tu(name=name, limit_graphs=args.limit_graphs)
        print_report(report)
        if args.output_dir:
            out_path = save_report_json(report, args.output_dir)
            print(f"JSON 摘要已保存：{out_path}")

    if args.ba2motif:
        print(f"开始分析数据集：BA2MotifDataset(n={args.ba2motif_n}, seed={args.ba2motif_seed}) ...")
        report = analyze_ba2motif(n_graphs=args.ba2motif_n, seed=args.ba2motif_seed)
        print_report(report)
        if args.output_dir:
            out_path = save_report_json(report, args.output_dir)
            print(f"JSON 摘要已保存：{out_path}")

    print("=" * 80)
    print("全部数据集分析完成。")


if __name__ == "__main__":
    main()


