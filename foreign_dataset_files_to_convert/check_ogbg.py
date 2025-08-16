"""
检查 OGB Graph Property 任务的若干数据集，默认涵盖：
  - ogbg-molhiv（二分类，图级）
  - ogbg-molpcba（多标签分类，图级）
  - ogbg-ppa（多类分类，图级）

输出：
  - 控制台可读报告
  - 可选保存 JSON 摘要到 --output-dir

检查范围（逐图统计并汇总）：
  - 图规模：节点数、边数（min/max/mean/std/分位数）
  - 节点特征：是否存在、维度、dtype、是否连续、示例值
  - 边特征：是否存在、维度、dtype、是否连续、示例值
  - 标签：形状、dtype、任务类型推断、分布/统计（含缺失值处理）
  - 官方划分规模：train/valid/test 数量

用法示例：
  python foreign_dataset_files_to_convert/check_ogbg.py \
      --datasets ogbg-molhiv ogbg-molpcba ogbg-ppa \
      --root ./data/ogb --output-dir ./outputs/ogbg_checks

依赖：ogb, torch, torch_geometric
官方文档： https://ogb.stanford.edu/docs/graphprop/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

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
    dim: Optional[int]
    dtype: Optional[str]
    is_continuous: Optional[bool]
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
    split_sizes: Dict[str, int]
    node_count: ScalarStats
    edge_count: ScalarStats
    node_features: FeatureInfo
    edge_features: FeatureInfo
    labels: LabelSummary


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
    return ScalarStats(
        minimum=minimum, maximum=maximum, mean=mean, std=std, quantiles=quantiles
    )


def infer_feature_info(batch_first_data) -> FeatureInfo:
    if batch_first_data is None:
        return FeatureInfo(False, None, None, None, None)
    if batch_first_data.x is None:
        return FeatureInfo(False, None, None, None, None)
    x = batch_first_data.x
    if not isinstance(x, torch.Tensor):
        return FeatureInfo(False, None, None, None, None)
    dim = int(x.size(-1)) if x.dim() == 2 else None
    dtype = str(x.dtype)
    is_cont = x.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    ex_rows = min(3, x.size(0))
    ex_cols = min(8, x.size(1)) if x.dim() == 2 else 0
    example = x[:ex_rows, :ex_cols].detach().cpu().tolist() if x.dim() == 2 else None
    return FeatureInfo(True, dim, dtype, is_cont, example)


def infer_edge_feature_info(batch_first_data) -> FeatureInfo:
    if batch_first_data is None:
        return FeatureInfo(False, None, None, None, None)
    if getattr(batch_first_data, "edge_attr", None) is None:
        return FeatureInfo(False, None, None, None, None)
    e = batch_first_data.edge_attr
    if not isinstance(e, torch.Tensor):
        return FeatureInfo(False, None, None, None, None)
    dim = int(e.size(-1)) if e.dim() == 2 else None
    dtype = str(e.dtype)
    is_cont = e.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    ex_rows = min(3, e.size(0))
    ex_cols = min(8, e.size(1)) if e.dim() == 2 else 0
    example = e[:ex_rows, :ex_cols].detach().cpu().tolist() if e.dim() == 2 else None
    return FeatureInfo(True, dim, dtype, is_cont, example)


def _flatten_y_tensor(y: torch.Tensor) -> torch.Tensor:
    if y is None:
        return torch.tensor([])
    y = y.detach().cpu()
    return y.view(-1)


def summarize_labels(sample_data, loader: 'DataLoader') -> LabelSummary:
    if sample_data is None or getattr(sample_data, "y", None) is None:
        return LabelSummary(shape=[], dtype="unknown", task_type="unknown", details={})

    y0 = sample_data.y

    # 情况A：张量标签（分类/回归常见）
    if isinstance(y0, torch.Tensor):
        y0_shape = list(y0.size())
        y_dtype = str(y0.dtype)

        all_y: List[float] = []
        int_like: bool = True
        float_like: bool = True
        nan_count: int = 0

        for batch in loader:
            y = getattr(batch, "y", None)
            if y is None:
                continue
            yf = _flatten_y_tensor(y)
            if yf.numel() == 0:
                continue
            if yf.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                int_like = False
            if yf.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                float_like = False

            if torch.is_floating_point(yf):
                nan_mask = torch.isnan(yf)
                nan_count += int(nan_mask.sum().item())
                yf = yf[~nan_mask]

            all_y.extend(yf.tolist())

        details: Dict[str, Any] = {"num_total_labels_after_nan_removed": len(all_y), "num_nan": nan_count}

        if len(all_y) == 0:
            return LabelSummary(shape=y0_shape, dtype=y_dtype, task_type="unknown", details=details)

        unique_vals = sorted(set(all_y))
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0})

        if is_binary:
            pos = sum(1 for v in all_y if v == 1.0)
            neg = sum(1 for v in all_y if v == 0.0)
            details.update({"binary_counts": {"neg": int(neg), "pos": int(pos)}, "pos_ratio": (pos / (pos + neg) if (pos + neg) > 0 else None)})
            task_type = "binary_classification" if (len(y0_shape) == 2 and y0_shape[-1] == 1) else "multilabel_classification"
            return LabelSummary(shape=y0_shape, dtype=y_dtype, task_type=task_type, details=details)

        if int_like and not float_like:
            from collections import Counter

            counter = Counter(int(v) for v in all_y)
            top5 = counter.most_common(5)
            details.update({
                "num_classes": len(counter),
                "top5_counts": [[int(k), int(v)] for k, v in top5],
            })
            return LabelSummary(shape=y0_shape, dtype=y_dtype, task_type="multiclass_classification", details=details)

        y_stats = compute_scalar_stats([float(v) for v in all_y])
        details.update({"regression_stats": asdict(y_stats)})
        return LabelSummary(shape=y0_shape, dtype=y_dtype, task_type="regression", details=details)

    # 情况B：序列标签（如 ogbg-code2，y 为 Python 列表/元组）
    if isinstance(y0, (list, tuple)):
        def _extract_seq(obj):
            if isinstance(obj, (list, tuple)):
                if len(obj) > 0 and isinstance(obj[0], (list, tuple)):
                    return list(obj[0])
                return list(obj)
        return None

        seq_example = _extract_seq(y0) or []
        elem_dtype = type(seq_example[0]).__name__ if len(seq_example) > 0 else "unknown"
        y_dtype = f"list[{elem_dtype}]"
        y0_shape = [len(seq_example)]

        lengths: List[int] = []
        # 收集长度分布
        for batch in loader:
            y = getattr(batch, "y", None)
            if isinstance(y, (list, tuple)):
                seq_b = _extract_seq(y) or []
                lengths.append(len(seq_b))
        if not lengths:
            return LabelSummary(shape=y0_shape, dtype=y_dtype, task_type="sequence_prediction", details={})

        length_stats = compute_scalar_stats([int(length_val) for length_val in lengths])
        details = {
            "num_sequences": len(lengths),
            "length_stats": asdict(length_stats),
            "example_first_10_tokens": seq_example[:10],
        }
        return LabelSummary(shape=y0_shape, dtype=y_dtype, task_type="sequence_prediction", details=details)

    # 其他未知类型
    return LabelSummary(shape=[], dtype=str(type(y0)), task_type="unknown", details={})


def analyze_dataset(name: str, root: str, batch_size: int, num_workers: int, limit_graphs: Optional[int] = None) -> DatasetReport:
    dataset = PygGraphPropPredDataset(name=name, root=root)
    num_graphs_total = len(dataset)
    if limit_graphs is not None:
        num_graphs = min(limit_graphs, num_graphs_total)
        dataset = dataset[:num_graphs]
    else:
        num_graphs = num_graphs_total

    split_idx = dataset.get_idx_split()
    split_sizes = {k: int(len(v)) for k, v in split_idx.items()}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    node_counts: List[int] = []
    edge_counts: List[int] = []

    first_sample = None
    for data in loader:
        if first_sample is None:
            if hasattr(data, "to_data_list"):
                dl = data.to_data_list()
                first_sample = dl[0] if len(dl) > 0 else None
            else:
                first_sample = data

        if hasattr(data, "to_data_list"):
            for g in data.to_data_list():
                node_counts.append(int(g.num_nodes))
                edge_counts.append(int(g.num_edges))
        else:
        node_counts.append(int(data.num_nodes))
            edge_counts.append(int(data.num_edges))

    node_stats = compute_scalar_stats(node_counts)
    edge_stats = compute_scalar_stats(edge_counts)

    node_feat_info = infer_feature_info(first_sample)
    edge_feat_info = infer_edge_feature_info(first_sample)

    single_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    label_summary = summarize_labels(first_sample, single_loader)

    return DatasetReport(
        name=name,
        num_graphs=num_graphs,
        split_sizes=split_sizes,
        node_count=node_stats,
        edge_count=edge_stats,
        node_features=node_feat_info,
        edge_features=edge_feat_info,
        labels=label_summary,
    )


def print_report(report: DatasetReport) -> None:
    print("=" * 80)
    print(f"数据集: {report.name}")
    print(f"图数量: {report.num_graphs}")
    if report.split_sizes:
        print(f"官方划分: train={report.split_sizes.get('train', 0)}, valid={report.split_sizes.get('valid', 0)}, test={report.split_sizes.get('test', 0)}")

    def _fmt_stats(stats: ScalarStats, unit: str) -> None:
        print(f"{unit}统计: min={stats.minimum:.0f}, max={stats.maximum:.0f}, mean={stats.mean:.2f}, std={stats.std:.2f}")
        q = stats.quantiles
        print(f"{unit}分位数: p50={q.p50:.0f}, p90={q.p90:.0f}, p95={q.p95:.0f}, p99={q.p99:.0f}")

    _fmt_stats(report.node_count, "节点数")
    _fmt_stats(report.edge_count, "边数")

    nf = report.node_features
    print(f"节点特征: exists={nf.exists}, dim={nf.dim}, dtype={nf.dtype}, 连续={nf.is_continuous}")
    if nf.example_values is not None:
        print("节点特征示例(前3行x前8列):")
        for row in nf.example_values:
            print(f"  {row}")

    ef = report.edge_features
    print(f"边特征: exists={ef.exists}, dim={ef.dim}, dtype={ef.dtype}, 连续={ef.is_continuous}")
    if ef.example_values is not None:
        print("边特征示例(前3行x前8列):")
        for row in ef.example_values:
            print(f"  {row}")

    ls = report.labels
    print(f"标签: shape={ls.shape}, dtype={ls.dtype}, 任务类型={ls.task_type}")
    if ls.details:
        print("标签细节:")
        for k, v in ls.details.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")

    print("-" * 80)
    suggestions: List[str] = []
    typical_nodes = max(32, int(report.node_count.quantiles.p95))
    suggestions.append(f"建议下游模型的最大节点相关序列长度至少覆盖 p95 节点数 ≈ {typical_nodes}。")
    if not nf.exists:
        suggestions.append("数据集无节点特征，需考虑使用结构特征或可学习嵌入。")
    if not ef.exists:
        suggestions.append("数据集无边特征，可按需要构造边类型或距离等辅助特征。")
    if ls.task_type in ("binary_classification", "multilabel_classification"):
        bin_info = ls.details.get("binary_counts")
        pos_ratio = ls.details.get("pos_ratio")
        if bin_info is not None:
            neg = bin_info.get("neg", 0)
            pos = bin_info.get("pos", 0)
            if (neg + pos) > 0:
                imbalance = max(neg, pos) / max(1, min(neg, pos))
                if imbalance >= 5:
                    suggestions.append(f"标签严重不均衡（最大/最小≈{imbalance:.1f}），建议使用加权损失或采样策略；pos_ratio≈{pos_ratio:.3f}。")
    if suggestions:
        print("分析建议：")
        for s in suggestions:
            print(f"  - {s}")


def save_report_json(report: DatasetReport, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{report.name.replace('/', '_')}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    return path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查并汇总 OGB Graph Property 数据集特征与统计")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[ "ogbg-code2"],
        help="需要检查的数据集名称列表（默认: ogbg-molhiv ogbg-molpcba ogbg-ppa）",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(".", "data", "ogb"),
        help="OGB 数据根目录（默认: ./data/ogb）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="统计时的批大小（默认: 64）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 工作线程数（默认: 0，设置更高可加速）",
    )
    parser.add_argument(
        "--limit-graphs",
        type=int,
        default=None,
        help="仅用于快速测试：限制统计的图数量（默认: None=全部）",
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
    reports: List[DatasetReport] = []

        for name in args.datasets:
        print(f"开始分析数据集：{name} ...")
        report = analyze_dataset(
            name=name,
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit_graphs=args.limit_graphs,
        )
        print_report(report)
        if args.output_dir:
            out_path = save_report_json(report, args.output_dir)
            print(f"JSON 摘要已保存：{out_path}")
        reports.append(report)

    print("=" * 80)
    print("全部数据集分析完成。")


if __name__ == "__main__":
    main()

