#!/usr/bin/env python3
"""
端到端管线（预训练 + 微调）
==========================

输入：图数据集名称、序列化方法（可选）
输出：微调测试集性能指标

说明：
- 预训练与微调阶段仅消费已准备好的缓存工件；不再支持 in_memory 模式
"""

from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ProjectConfig  # noqa: E402
from bert_pretrain import pretrain_bert_model  # noqa: E402
from src.training.finetune_pipeline import run_finetune  # noqa: E402
from src.data.unified_data_interface import UnifiedDataInterface  # noqa: E402
from src.training.pretrain_api import pretrain as pretrain_api  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="端到端：预训练+微调")
    parser.add_argument("--dataset", required=True, help="数据集名称，如 zinc/qm9/mnist")
    parser.add_argument("--serialization_method", default=None, help="序列化方法，如 feuler/eulerian 等；可用 zinc_bpe/zmol_raw 这种后缀式标记是否压缩")
    parser.add_argument("--task", choices=["regression", "classification"], help="任务类型（可由数据集自动推断）")
    parser.add_argument("--num_classes", type=int, help="分类类别数（可自动推断）")
    # 训练层仅消费已准备好的缓存工件；不再支持 in_memory 模式
    parser.add_argument("--group", type=str, default="exp", help="实验分组(必需用于目录结构)")
    parser.add_argument("--experiment_name", type=str, default=None, help="实验名称(可选)")
    parser.add_argument("--prepare", action="store_true", help="是否在本脚本内一次性准备并注册工件（序列化、BPE、词表）")
    parser.add_argument("--data_version", type=str, default=None, help="写入 processed/<dataset>/<version> 的版本目录名(可选)")

    args = parser.parse_args()

    # 构建配置
    config = ProjectConfig()
    config.dataset.name = args.dataset
    # 支持通过数据集名称或方法后缀控制是否使用BPE压缩
    if args.serialization_method:
        method = args.serialization_method
        if method.endswith("_raw"):
            config.serialization.method = method.replace("_raw", "")
            config.serialization.bpe.enabled = False
        elif method.endswith("_bpe"):
            config.serialization.method = method.replace("_bpe", "")
            config.serialization.bpe.enabled = True
        else:
            config.serialization.method = method
    # 目录结构所需参数
    config.experiment_group = args.group
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    udi = UnifiedDataInterface(config=config, dataset=config.dataset.name)

    # 可选：在脚本内进行一次性准备与注册
    if args.prepare:
        method = config.serialization.method
        # 1) 准备并持久化序列化（BPE已集成）
        udi.prepare_serialization(method)
        # 2) 构建词表并注册（基于已准备好的缓存序列）
        sequences_with_ids, _ = udi.get_sequences(method)
        all_sequences = [seq for _, seq in sequences_with_ids]
        splits = udi.get_split_indices()
        token_splits = {
            'train': [all_sequences[i] for i in splits['train']],
            'val': [all_sequences[i] for i in splits['val']],
            'test': [all_sequences[i] for i in splits['test']],
        }
        pretrain_res = pretrain_api(config, token_splits)
        try:
            udi.register_vocab(pretrain_res['vocab_manager'], method=config.serialization.method, bpe=True)
        except Exception:
            raise Exception("注册词表失败")
    else:
        # 未选择在脚本内准备：确保工件已存在，否则明确报错
        method = config.serialization.method
        missing = []
        if not udi.has_serialized(method):
            missing.append("serialized_data")
        if missing:
            raise SystemExit(
                f"缺失工件: {missing}. 请先运行 data_prepare.py 或加上 --prepare 以在本脚本内准备。"
            )
        # 预训练：仅从缓存读取（缺失直接报错）
        pretrain_bert_model(config)

    # 微调：自动推断任务与元信息在 run_finetune 内部/finetune.py 完成
    # 自动推断任务与目标属性/类别
    # 预训练完成后，进行微调
    meta = udi.get_downstream_metadata()
    task = args.task or meta.get('dataset_task_type', 'regression')
    # 回归默认 target_property：
    # - 非QM9：无条件使用数据集默认键，避免沿用默认配置中的 'homo'
    # - QM9：若未显式指定则默认使用 'homo'
    if task == 'regression':
        if config.dataset.name.lower().startswith('qm9'):
            if not config.task.target_property:
                config.task.target_property = 'homo'
        else:
            default_prop = meta.get('default_target_property')
            if default_prop:
                config.task.target_property = default_prop
    # 分类类别数
    num_classes = args.num_classes
    if task == 'classification' and num_classes is None:
        assert 'num_classes' in meta, "分类任务需要 num_classes，但数据集元数据中未找到此字段"
        assert isinstance(meta['num_classes'], int) and meta['num_classes'] > 1, f"num_classes 应为大于1的整数，实际值: {meta['num_classes']}"
        num_classes = meta['num_classes']

    results = run_finetune(
        config,
        task=task,
        num_classes=num_classes,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


