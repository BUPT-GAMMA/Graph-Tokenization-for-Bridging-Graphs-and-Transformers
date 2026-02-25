#!/usr/bin/env python3
"""
分类数据集类别分布与均衡性检查
================================

功能：
- 遍历已注册的数据集，跳过 {qm9, mnist, zinc, aqsol, qm9test}
- 仅对任务类型为 classification 的数据集：
  - 统计 train/val/test 及 overall 的类别分布
  - 计算均衡性指标：
    - 归一化熵 (0~1，越大越均匀)
    - 归一化 Gini (0~1，越大越均匀)
    - 最大/最小样本数比 (越小越均匀)
    - 变异系数 Coefficient of Variation, CV (越小越均匀)
    - KL(p || Uniform) 散度 (越小越均匀)
- 将结果打印到控制台，并写入 src/data/CLASS_BALANCE_REPORT.md；
- 追加全局总结：各数据集 overall 指标对比与不平衡标记，以及最不平衡 split 概览。

备注：
- 仅统计已经存在预处理产物的数据集（data/<name>/ 下应存在 data.pkl 与三份索引文件）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import math
import sys
import pickle

import numpy as np


# 确保项目根目录在 sys.path 中（src/data/ -> src -> project root）
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ProjectConfig
from src.data.unified_data_factory import list_available_datasets, get_dataloader


EXCLUDE_DATASETS = {"qm9", "mnist", "zinc", "aqsol", "qm9test"}


@dataclass
class SplitStats:
    counts: Dict[int, int]
    proportions: Dict[int, float]
    num_samples: int
    metrics: Dict[str, float]


def _safe_div(a: float, b: float) -> float:
    return a / b if b not in (0, 0.0) else float("inf")


def _complete_class_counts(raw_counts: Dict[int, int], num_classes: int | None) -> Dict[int, int]:
    """
    若已知类别总数 K，则补齐缺失类别为0计数；否则按观察到的标签集合返回。
    """
    if num_classes is not None and num_classes > 0 and num_classes <= 10000:
        return {c: int(raw_counts.get(c, 0)) for c in range(int(num_classes))}
    # Fallback：仅使用观测到的类别
    return {int(c): int(n) for c, n in raw_counts.items()}


def _counts_to_props(counts: Dict[int, int]) -> Dict[int, float]:
    total = int(sum(counts.values()))
    if total <= 0:
        return {c: 0.0 for c in counts.keys()}
    return {c: float(n) / float(total) for c, n in counts.items()}


def _normalized_entropy(proportions: Dict[int, float]) -> float:
    ps = [max(1e-12, p) for p in proportions.values()]
    k = len(ps)
    h = -sum(p * math.log(p) for p in ps)
    denom = math.log(k) if k > 1 else 1.0
    return float(h / denom)


def _normalized_gini(proportions: Dict[int, float]) -> float:
    ps = list(proportions.values())
    k = len(ps)
    gini = 1.0 - float(sum(p * p for p in ps))
    max_gini = 1.0 - 1.0 / float(k) if k > 0 else 1.0
    if max_gini <= 0:
        return 0.0
    return float(gini / max_gini)


def _imbalance_ratio(counts: Dict[int, int]) -> float:
    vals = [int(v) for v in counts.values() if int(v) >= 0]
    if not vals:
        return float("inf")
    mn = min(vals)
    mx = max(vals)
    if mn == 0:
        return float("inf")
    return float(mx) / float(mn)


def _coefficient_of_variation(counts: Dict[int, int]) -> float:
    arr = np.asarray([int(v) for v in counts.values()], dtype=np.float64)
    mean = float(arr.mean()) if arr.size > 0 else 0.0
    std = float(arr.std(ddof=0)) if arr.size > 0 else 0.0
    return _safe_div(std, mean) if mean > 0 else float("inf")


def _kl_to_uniform(proportions: Dict[int, float]) -> float:
    ps = [max(1e-12, p) for p in proportions.values()]
    k = len(ps)
    if k <= 0:
        return 0.0
    u = 1.0 / float(k)
    return float(sum(p * math.log(p / u) for p in ps))


def _compute_metrics(counts: Dict[int, int]) -> Dict[str, float]:
    props = _counts_to_props(counts)
    return {
        "normalized_entropy": _normalized_entropy(props),
        "normalized_gini": _normalized_gini(props),
        "imbalance_ratio_max_min": _imbalance_ratio(counts),
        "cv_counts": _coefficient_of_variation(counts),
        "kl_to_uniform": _kl_to_uniform(props),
    }


def _summarize_split(labels: List[int], num_classes: int | None) -> SplitStats:
    raw_counts: Dict[int, int] = {}
    for y in labels:
        raw_counts[int(y)] = raw_counts.get(int(y), 0) + 1
    counts = _complete_class_counts(raw_counts, num_classes)
    return SplitStats(
        counts=counts,
        proportions=_counts_to_props(counts),
        num_samples=sum(counts.values()),
        metrics=_compute_metrics(counts),
    )


def _format_counts(counts: Dict[int, int]) -> str:
    # 以标签升序稳定输出
    items = sorted(counts.items(), key=lambda kv: kv[0])
    return ", ".join([f"{lbl}: {cnt}" for lbl, cnt in items])


def _format_props(props: Dict[int, float]) -> str:
    items = sorted(props.items(), key=lambda kv: kv[0])
    return ", ".join([f"{lbl}: {p:.4f}" for lbl, p in items])


def main() -> None:
    cfg = ProjectConfig()
    out_path = Path(__file__).resolve().parent / "CLASS_BALANCE_REPORT.md"

    datasets = list_available_datasets()
    lines: List[str] = []
    lines.append("# 分类数据集类别分布与均衡性检查\n")
    lines.append("本文档统计当前已注册且为分类任务、且已完成预处理的数据集的类别分布与均衡性指标。\n")

    # 汇总数据，用于全局总结
    summary_entries: List[Dict[str, Any]] = []

    for ds in datasets:
        if ds in EXCLUDE_DATASETS:
            continue
        try:
            loader = get_dataloader(ds, cfg)
            task_type = str(loader.get_dataset_task_type()).lower()
            if task_type != "classification":
                continue

            train_data, val_data, test_data, y_tr, y_va, y_te = loader.load_data()
            # 期望类别数（若子类未覆盖或偏小，按观测替代）
            k_expected = int(loader.get_num_classes()) if hasattr(loader, "get_num_classes") else None
            unique_overall = sorted(set(int(y) for y in (y_tr + y_va + y_te)))
            if (k_expected is None) or (len(unique_overall) > (k_expected or 0)):
                k_expected = len(unique_overall)

            tr_stats = _summarize_split([int(y) for y in y_tr], k_expected)
            va_stats = _summarize_split([int(y) for y in y_va], k_expected)
            te_stats = _summarize_split([int(y) for y in y_te], k_expected)
            all_labels = [int(y) for y in (y_tr + y_va + y_te)]
            all_stats = _summarize_split(all_labels, k_expected)

            # 直接从 data.pkl 读取全量标签（不依赖 split），用于“Raw-All”统计
            raw_all_stats: SplitStats | None = None
            try:
                data_pkl = loader.data_dir / "data.pkl"
                with open(data_pkl, "rb") as f:
                    raw_data = pickle.load(f)
                # 适配两种常见结构：[(graph, label)] 或 [dict]
                raw_labels: List[int] = []
                if isinstance(raw_data, list) and raw_data:
                    first = raw_data[0]
                    if isinstance(first, tuple) and len(first) >= 2:
                        for it in raw_data:
                            raw_labels.append(int(it[1]))
                    elif isinstance(first, dict):
                        for it in raw_data:
                            lab = it.get("properties", {}).get("label")
                            if lab is None and "label" in it:
                                lab = it["label"]
                            raw_labels.append(int(lab))
                # 若能解析到原始标签，则进行统计（类别数沿用 k_expected 或观测更新）
                if raw_labels:
                    k_raw = max(k_expected, len(set(raw_labels))) if k_expected is not None else len(set(raw_labels))
                    raw_all_stats = _summarize_split(raw_labels, k_raw)
            except Exception:
                raw_all_stats = None

            # 控制台输出
            print(f"=== {loader.dataset_name} (K={k_expected}) ===")
            for split_name, stats in ("train", tr_stats), ("val", va_stats), ("test", te_stats), ("overall", all_stats):
                print(f"[{split_name}] N={stats.num_samples}")
                print("  counts     :", _format_counts(stats.counts))
                print("  proportions:", _format_props(stats.proportions))
                print(
                    "  metrics   :",
                    {
                        "H_norm": round(stats.metrics["normalized_entropy"], 4),
                        "Gini_norm": round(stats.metrics["normalized_gini"], 4),
                        "IR": round(stats.metrics["imbalance_ratio_max_min"], 4)
                        if math.isfinite(stats.metrics["imbalance_ratio_max_min"]) else float("inf"),
                        "CV": round(stats.metrics["cv_counts"], 4)
                        if math.isfinite(stats.metrics["cv_counts"]) else float("inf"),
                        "KL": round(stats.metrics["kl_to_uniform"], 4),
                    },
                )
            print()

            # 写入 Markdown
            lines.append(f"## {loader.dataset_name} (K={k_expected})\n")
            lines.append("")
            for title, stats in (
                ("Train", tr_stats),
                ("Val", va_stats),
                ("Test", te_stats),
                ("Overall", all_stats),
            ):
                lines.append(f"- **{title}**: N={stats.num_samples}")
                lines.append(f"  - **Counts**: {_format_counts(stats.counts)}")
                lines.append(f"  - **Proportions**: {_format_props(stats.proportions)}")
                lines.append(
                    "  - **Metrics**: "
                    f"H_norm={stats.metrics['normalized_entropy']:.4f}, "
                    f"Gini_norm={stats.metrics['normalized_gini']:.4f}, "
                    f"IR={stats.metrics['imbalance_ratio_max_min'] if math.isfinite(stats.metrics['imbalance_ratio_max_min']) else 'inf'}, "
                    f"CV={stats.metrics['cv_counts'] if math.isfinite(stats.metrics['cv_counts']) else 'inf'}, "
                    f"KL={stats.metrics['kl_to_uniform']:.4f}"
                )
            lines.append("")

            # 追加 Raw-All 段（如可用）
            if raw_all_stats is not None:
                lines.append(f"- **Raw-All (from data.pkl)**: N={raw_all_stats.num_samples}")
                lines.append(f"  - **Counts**: {_format_counts(raw_all_stats.counts)}")
                lines.append(f"  - **Proportions**: {_format_props(raw_all_stats.proportions)}")
                lines.append(
                    "  - **Metrics**: "
                    f"H_norm={raw_all_stats.metrics['normalized_entropy']:.4f}, "
                    f"Gini_norm={raw_all_stats.metrics['normalized_gini']:.4f}, "
                    f"IR={raw_all_stats.metrics['imbalance_ratio_max_min'] if math.isfinite(raw_all_stats.metrics['imbalance_ratio_max_min']) else 'inf'}, "
                    f"CV={raw_all_stats.metrics['cv_counts'] if math.isfinite(raw_all_stats.metrics['cv_counts']) else 'inf'}, "
                    f"KL={raw_all_stats.metrics['kl_to_uniform']:.4f}"
                )
                lines.append("")

            # 收集总结信息（以 overall 为主，并记录最不平衡 split）
            def _imbalance_flag_overall(st: SplitStats) -> bool:
                # 经验阈值：任一满足即判定整体不平衡
                ir = st.metrics["imbalance_ratio_max_min"]
                cv = st.metrics["cv_counts"]
                h = st.metrics["normalized_entropy"]
                return (
                    (math.isfinite(ir) and ir >= 1.5)
                    or (math.isfinite(cv) and cv >= 0.2)
                    or (h <= 0.98)
                )

            def _split_severity(st: SplitStats) -> float:
                # 用 IR 为主的严重度评分；若无穷大直接最大
                ir = st.metrics["imbalance_ratio_max_min"]
                if not math.isfinite(ir):
                    return float("inf")
                return float(ir)

            split_map = {
                "train": tr_stats,
                "val": va_stats,
                "test": te_stats,
            }
            worst_split_name, worst_split_stats = max(split_map.items(), key=lambda kv: _split_severity(kv[1]))
            # 全局总结优先使用 Raw-All（若存在），否则使用 Overall
            overall_for_summary = raw_all_stats if raw_all_stats is not None else all_stats
            summary_entries.append({
                "name": loader.dataset_name,
                "k": int(k_expected),
                "n_total": int(overall_for_summary.num_samples),
                "overall_ir": float(overall_for_summary.metrics["imbalance_ratio_max_min"]) if math.isfinite(overall_for_summary.metrics["imbalance_ratio_max_min"]) else float("inf"),
                "overall_cv": float(overall_for_summary.metrics["cv_counts"]) if math.isfinite(overall_for_summary.metrics["cv_counts"]) else float("inf"),
                "overall_h": float(overall_for_summary.metrics["normalized_entropy"]),
                "overall_kl": float(overall_for_summary.metrics["kl_to_uniform"]),
                "overall_imbalanced": _imbalance_flag_overall(overall_for_summary),
                "worst_split": worst_split_name,
                "worst_ir": float(worst_split_stats.metrics["imbalance_ratio_max_min"]) if math.isfinite(worst_split_stats.metrics["imbalance_ratio_max_min"]) else float("inf"),
                "worst_h": float(worst_split_stats.metrics["normalized_entropy"]),
            })

        except FileNotFoundError:
            # 预处理产物缺失则跳过
            continue
        except Exception as e:
            print(f"[WARN] 统计失败: {ds}: {e}")
            continue

    # 全局总结（表格 + 不平衡清单）
    if summary_entries:
        lines.append("## 全局总结\n")
        lines.append("")
        # 表头
        lines.append(
            "| 数据集 | K | N_total | IR(overall) | CV(overall) | H_norm(overall) | KL | 最不平衡split | worst IR | worst H | 不平衡? |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|:---:|"
        )
        for r in summary_entries:
            ir_str = f"{r['overall_ir']:.4f}" if math.isfinite(r['overall_ir']) else "inf"
            cv_str = f"{r['overall_cv']:.4f}" if math.isfinite(r['overall_cv']) else "inf"
            worst_ir_str = f"{r['worst_ir']:.4f}" if math.isfinite(r['worst_ir']) else "inf"
            lines.append(
                f"| {r['name']} | {r['k']} | {r['n_total']} | {ir_str} | {cv_str} | {r['overall_h']:.4f} | {r['overall_kl']:.4f} | "
                f"{r['worst_split']} | {worst_ir_str} | {r['worst_h']:.4f} | {'✅' if not r['overall_imbalanced'] else '❌'} |"
            )

        # 不平衡数据集清单（按 overall 判定）
        imbalanced = [r for r in summary_entries if r["overall_imbalanced"]]
        if imbalanced:
            lines.append("")
            lines.append("### 判定为整体不平衡的数据集")
            for r in imbalanced:
                ir_str = f"{r['overall_ir']:.4f}" if math.isfinite(r['overall_ir']) else "inf"
                cv_str = f"{r['overall_cv']:.4f}" if math.isfinite(r['overall_cv']) else "inf"
                lines.append(
                    f"- {r['name']}: IR={ir_str}, CV={cv_str}, H_norm={r['overall_h']:.4f}, KL={r['overall_kl']:.4f}"
                )

    # 追加指标解释
    lines.append("---\n")
    lines.append("### 指标说明")
    lines.append("- **H_norm (归一化熵)**: 越接近 1 越均匀；0 表示极端不均衡。")
    lines.append("- **Gini_norm (归一化Gini)**: 越接近 1 越均匀；0 表示极端不均衡。")
    lines.append("- **IR (max/min)**: 最大类和最小类样本数之比，越小越好；若存在缺失类则为 inf。")
    lines.append("- **CV**: 样本数的变异系数，越小越好；若均值为0或存在缺失类导致极端值则为 inf。")
    lines.append("- **KL**: 与均匀分布的 KL 散度，越小越好；0 表示完全均匀。")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 分类数据集类别分布报告已生成: {out_path}")


if __name__ == "__main__":
    main()


