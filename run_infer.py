#!/usr/bin/env python3
"""
单个方法推理脚本（与 run_finetune 对等）
====================================

支持：
- 从已微调模型目录加载权重
- 可选训练“learned”聚合器并用于测试聚合
- 与预训练/微调一致的参数风格、日志与指标记录（Infer/*）
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.training.infer_pipeline import run_infer, resolve_finetuned_model_dir
from src.utils.config_override import (
    add_all_args,
    apply_args_to_config,
    create_experiment_name,
    print_config_summary,
)


def _infer_task_and_targets(config: ProjectConfig, udi: UnifiedDataInterface,
                            task_cli: str | None, num_classes_cli: int | None) -> tuple[str, int | None]:
    meta = udi.get_downstream_metadata()

    # 任务类型
    if task_cli:
        task = task_cli
    else:
        assert 'dataset_task_type' in meta, "数据集元数据中缺少 'dataset_task_type'"
        task = meta['dataset_task_type']

    # 分类类别数
    num_classes = num_classes_cli
    if task == 'classification' and num_classes is None:
        assert 'num_classes' in meta, "分类任务需要 num_classes，但元数据未提供"
        n = meta['num_classes']
        assert isinstance(n, int) and n > 1
        num_classes = n
    return task, num_classes


def main():
    parser = argparse.ArgumentParser(
        description="单个方法推理脚本（可选 learned 聚合）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 与预训练/微调统一的参数风格
    add_all_args(parser, include_finetune=True)
    parser.add_argument("--aggregation_mode", type=str, default="avg", choices=["avg", "best", "learned"], help="聚合模式")
    parser.add_argument("--model_dir", type=str, default=None, help="显式模型目录（含config.bin/pytorch_model.bin）")
    parser.add_argument("--save_name", type=str, default=None, help="在 model 根下的保存子目录名（如 finetune）")
    parser.add_argument("--save_name_prefix", type=str, default=None, help="日志目录前缀（可选）")
    parser.add_argument("--save_name_suffix", type=str, default=None, help="日志目录后缀（可选）")
    parser.add_argument("--plain_logs", action="store_true", help="启用无颜色、无控制符输出（与微调一致）")

    args = parser.parse_args()

    # 创建配置并应用CLI覆盖
    config = ProjectConfig()
    apply_args_to_config(config, args, common_to="finetune")
    create_experiment_name(config)

    # 推断任务/类别
    udi = UnifiedDataInterface(config, config.dataset.name)
    task, num_classes = _infer_task_and_targets(config, udi, args.task, getattr(args, 'num_classes', None))
    config.task.type = task

    # 如果未显式给出 group/name，则尝试从模型目录解析，以对齐日志目录
    try:
        model_dir = resolve_finetuned_model_dir(config, model_dir=args.model_dir, save_name=args.save_name)
        rel = model_dir.resolve().relative_to(config.model_dir.resolve())
        parts = list(rel.parts)
        if len(parts) >= 2:
            if not getattr(config, 'experiment_group', None):
                config.experiment_group = parts[0]
            if not getattr(config, 'experiment_name', None):
                config.experiment_name = parts[1]
    except Exception:
        pass

    # 计算与微调保存一致的 save_name（若未显式提供）
    eff_save_name = args.save_name
    try:
        if eff_save_name is None:
            suffix = getattr(args, 'save_name_suffix', None)
            eff_save_name = f"finetune_{suffix}" if suffix else "finetune"
    except Exception:
        eff_save_name = args.save_name or "finetune"

    # 运行推理流水线
    result = run_infer(
        config,
        task=task,
        num_classes=num_classes,
        aggregation_mode=args.aggregation_mode,
        save_name_prefix=args.save_name_prefix,
        save_name_suffix=args.save_name_suffix,
        model_dir=args.model_dir,
        save_name=eff_save_name,
    )

    # 打印与保存摘要
    print_config_summary(config)
    print("\n📈 测试集性能：")
    for metric, value in result['test_metrics'].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.6f}")
        else:
            print(f"  {metric}: {value}")

    # 将结果结构化保存至 logs/<...>/infer/
    try:
        out_dir = config.get_logs_dir() / (args.save_name_prefix + '_' if args.save_name_prefix else '') + 'infer'
    except TypeError:
        out_dir = config.get_logs_dir() / 'infer'
    (out_dir if isinstance(out_dir, Path) else Path(out_dir)).mkdir(parents=True, exist_ok=True)
    out_path = (out_dir if isinstance(out_dir, Path) else Path(out_dir)) / 'infer_metrics.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(result['test_metrics'], f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())


