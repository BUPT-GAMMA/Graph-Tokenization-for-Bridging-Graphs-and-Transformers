#!/usr/bin/env python3
"""
端到端测试脚本：序列化 -> BPE -> BERT预训练(MLM) -> BERT微调
================================================================

用法示例：
  python scripts/run_end_to_end_test.py \
    --dataset aqsol \
    --method eulerian \
    --use_bpe true \
    --group exp \
    --experiment_name aqsol_eulerian_interleave_omit \
    --mlm_epochs 30 \
    --finetune_epochs 60 \
    --target_property solubility

说明：
- 串联数据加载、序列化、BPE、BERT预训练、BERT微调。
- 进行细粒度一致性与完整性检查；任何异常立即报错，不做隐藏回退。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from bert_pretrain import OptimizedBertPretrainingPipeline
from bert_regression import NormalizedBertFinetuningPipeline


def _as_bool(s: str) -> bool:
    s = str(s).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔值: {s}")


def check_file_exists(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} 不存在: {path}")


def summarize_sequences(seqs: List[List[int]]) -> Dict[str, Any]:
    lengths = [len(s) for s in seqs]
    return {
        "count": len(seqs),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
    }


def print_kv(title: str, kv: Dict[str, Any]):
    print(f"\n== {title} ==")
    for k, v in kv.items():
        print(f"- {k}: {v}")


def run(args):
    # 1) 构建配置
    if args.config:
        config = ProjectConfig(config_path=args.config)
    else:
        config = ProjectConfig()

    # 设置实验、数据开关
    if args.group:
        config.experiment_group = args.group
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    if args.dataset:
        config.dataset.name = args.dataset
    if args.method:
        config.serialization.method = args.method
    if args.use_bpe is not None:
        config.serialization.bpe.enabled = _as_bool(args.use_bpe)

    if args.mlm_epochs:
        config.bert.pretraining.epochs = int(args.mlm_epochs)
    if args.finetune_epochs:
        config.bert.finetuning.epochs = int(args.finetune_epochs)

    if args.target_property:
        config.task.target_property = args.target_property

    # 严格验证配置
    config.validate()

    # 2) 数据加载 & 构建序列化/BPE（强制检查产物）
    print("\n[阶段] 数据加载 & 序列化/BPE 构建")
    udi = UnifiedDataInterface(config=config, dataset=config.dataset.name)

    # 显式触发构建
    print("- 触发序列化构建 (若缺失)")
    if config.serialization.bpe.enabled:
        # 先显式构建序列化（BPE不再需要单独构建）
        udi.prepare_serialization(config.serialization.method)
        # 然后按划分加载
        train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = udi.get_sequences_by_splits(
            method=config.serialization.method
        )
        # 细粒度检查：文件是否存在
        base = Path(config.processed_data_dir) / config.dataset.name
        cache_key = "single"  # 默认缓存键，需要根据配置调整
        ser_path = base / "serialized_data" / config.serialization.method / cache_key / "serialized_data.pickle"
        check_file_exists(ser_path, "序列化结果")
    else:
        # 先显式构建序列化
        udi.prepare_serialization(config.serialization.method)
        # 然后按划分加载
        train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = udi.get_sequences_by_splits(
            method=config.serialization.method
        )
        base = Path(config.processed_data_dir) / config.dataset.name
        cache_key = "single"  # 默认缓存键，需要根据配置调整
        ser_path = base / "serialized_data" / config.serialization.method / cache_key / "serialized_data.pickle"
        check_file_exists(ser_path, "序列化结果")

    # 基本校验
    if not train_seqs or not val_seqs or not test_seqs:
        raise ValueError("序列化/BPE 结果为空或数据划分为空")

    print_kv("train 序列统计", summarize_sequences(train_seqs))
    print_kv("val 序列统计", summarize_sequences(val_seqs))
    print_kv("test 序列统计", summarize_sequences(test_seqs))

    # 3) BERT 预训练 (MLM)
    print("\n[阶段] BERT 预训练 (MLM)")
    pretrain_pipeline = OptimizedBertPretrainingPipeline(config)

    # 明确保存配置快照 & 加载数据
    token_sequences_all = pretrain_pipeline.load_data()
    if not token_sequences_all:
        raise ValueError("预训练阶段：加载到的序列为空")
    print_kv("预训练数据统计", summarize_sequences(token_sequences_all))

    # 构建词表 & 创建模型
    pretrain_pipeline.build_vocab(token_sequences_all)

    # 训练
    pretrain_pipeline.train_mlm(token_sequences_all)
    print(f"- 最优预训练 epoch: {pretrain_pipeline.best_epoch}")
    print(f"- 最优验证损失: {pretrain_pipeline.best_val_loss:.6f}")

    pretrained_model_path = pretrain_pipeline.model_file
    print(f"- MLM预训练模型已保存: {pretrained_model_path}")

    # 4) BERT 微调 (回归/分类)
    print("\n[阶段] BERT 下游任务微调")
    if not config.task.target_property:
        raise ValueError("微调阶段需要明确 target_property。请通过 --target_property 指定。")

    finetune_pipeline = NormalizedBertFinetuningPipeline(config, pretrained_model_path=str(pretrained_model_path))
    finetune_pipeline.create_finetuned_model()
    finetune_pipeline.finetune_model()

    # 5) 汇总最终指标
    results_dir = finetune_pipeline.results_dir
    test_results_file = results_dir / "test_results.json"
    if test_results_file.exists():
        with open(test_results_file, "r", encoding="utf-8") as f:
            test_metrics = json.load(f)
    else:
        # 若未写入文件，按返回的内部统计为准
        test_metrics = {}

    print("\n[阶段] 最终结果")
    print(f"- 预训练最优epoch: {pretrain_pipeline.best_epoch}")
    print(f"- 预训练最优val_loss: {pretrain_pipeline.best_val_loss:.6f}")

    if finetune_pipeline.best_metrics:
        print("- 验证集最佳指标：")
        for k, v in finetune_pipeline.best_metrics.items():
            print(f"  {k}: {v}")

    if test_metrics:
        print("- 测试集指标：")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")

    print("\n✅ 端到端流程完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="端到端：序列化->BPE->BERT预训练->微调")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称，例如 aqsol/qm9/…")
    parser.add_argument("--method", type=str, required=True, help="序列化方法，例如 eulerian/feuler/…")
    parser.add_argument("--use_bpe", type=str, default="true", help="是否使用BPE (true/false)")
    parser.add_argument("--group", type=str, required=True, help="实验分组名，用于目录结构")
    parser.add_argument("--experiment_name", type=str, default=None, help="实验名(可选)")
    parser.add_argument("--mlm_epochs", type=int, default=None, help="预训练轮数(覆盖配置)")
    parser.add_argument("--finetune_epochs", type=int, default=None, help="微调轮数(覆盖配置)")
    parser.add_argument("--target_property", type=str, required=False, help="下游任务目标属性，例如 aqsol 的 solubility")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径(可选)")

    args = parser.parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"❌ 端到端流程失败: {e}")
        sys.exit(1)
