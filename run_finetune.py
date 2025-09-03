#!/usr/bin/env python3
"""
单个方法 BERT 微调脚本（权威用法）
================================

本脚本是项目内“标准/唯一”的微调入口，被批量/搜索脚本复用。

必须参数（命令行）:
  --dataset DATASET   例如: qm9, qm9test, zinc, ...
  --method  METHOD    例如: feuler, eulerian, cpp, fcpp, topo, smiles

加载预训练的两种方式（择一）:
  --pretrained_dir PATH       直接指定模型目录（包含 config.bin 与 pytorch_model.bin）
  --pretrain_exp_name NAME    指定预训练实验名，按标准目录结构加载

常用参数（命令行）:
  --experiment_group NAME     归档分组
  --experiment_name  NAME     微调实验名（仅影响保存路径，与加载预训练解耦）
  --bpe_encode_rank_mode MODE BPE模式: none|all|topk|random|gaussian
  --epochs / --batch_size / --learning_rate / --weight_decay / --warmup_ratio / --max_grad_norm
  --config_json JSON_OR_PATH  高级配置（JSON字符串或文件路径），用于嵌套项覆盖

目标指标: 微调阶段以测试集 MAE 为唯一优化指标（脚本内部严格检查）。
"""

from __future__ import annotations

import argparse
import sys
import os
import re
import io
from pathlib import Path
import time
from typing import Optional, Literal
from clearml import Logger, Task
# 设置项目路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir('/home/gzy/py/tokenizerGraph')

from config import ProjectConfig  # noqa: E402
from src.data.unified_data_interface import UnifiedDataInterface  # noqa: E402
from src.training.finetune_pipeline import run_finetune  # noqa: E402
from src.utils.config_override import (  # noqa: E402
    add_all_args,
    apply_args_to_config,
    create_experiment_name,
    print_config_summary,
    show_full_config
)

# ClearML 任务初始化（支持直接运行和Agent执行）
# 检测是否已经在任务上下文中（通过Agent执行的情况）
current_task = Task.current_task()
if current_task is not None:
    # 已经在任务上下文中，使用现有的任务
    task = current_task
    print(f"✅ 使用现有ClearML任务: {task.name} (ID: {task.id})")
else:
    # 直接运行的情况，创建新任务
    try:
        task = Task.init(
            project_name="TokenizerGraph",
            task_name=f"finetune_manual_{int(time.time())}",
            auto_connect_frameworks=True  # 确保自动捕获TensorBoard
        )
        print(f"✅ ClearML任务初始化成功: {task.name} (ID: {task.id})")
    except Exception as e:
        print(f"⚠️ ClearML初始化失败，将继续运行: {e}")
        task = None

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)


class _AnsiStrippingWriter:
    """包装标准输出/错误，移除ANSI颜色控制符，并确保UTF-8."""
    def __init__(self, underlying):
        self._u = underlying
        self.encoding = 'utf-8'

    def write(self, s):
        try:
            if isinstance(s, bytes):
                s = s.decode('utf-8', errors='replace')
        except Exception:
            pass
        s = _strip_ansi(str(s))
        return self._u.write(s)

    def flush(self):
        return self._u.flush()

    def isatty(self):
        return False

    def readable(self):
        return False

    def writable(self):
        return True

    def fileno(self):
        try:
            return self._u.fileno()
        except Exception:
            raise


def _ensure_utf8_streams():
    """确保标准输出/错误为UTF-8并可替换编码。"""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "C.UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        try:
            stream.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
        except Exception:
            try:
                wrapped = io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
                setattr(sys, stream_name, wrapped)
            except Exception:
                pass

    # 降噪：静默TensorFlow冗余日志（与本训练流程无关）
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 只显示WARNING及以上


def _configure_output_mode(offline: bool):
    """根据 offline 模式配置输出：禁用颜色、去除ANSI、确保UTF-8。"""
    _ensure_utf8_streams()
    if offline:
        os.environ["NO_COLOR"] = "1"
        os.environ["CLICOLOR"] = "0"
        os.environ["FORCE_COLOR"] = "0"
        os.environ["TERM"] = "dumb"
        os.environ.setdefault("TQDM_DISABLE", "1")
        if not isinstance(sys.stdout, _AnsiStrippingWriter):
            sys.stdout = _AnsiStrippingWriter(sys.stdout)
        if not isinstance(sys.stderr, _AnsiStrippingWriter):
            sys.stderr = _AnsiStrippingWriter(sys.stderr)

def run_finetuning(
    config: ProjectConfig,
    aggregation_mode: Literal["avg", "best", "learned"] = "avg",
    save_name_prefix: str | None = None,
    save_name_suffix: str | None = None,
    pretrained_dir: str | None = None,
    pretrain_exp_name: str | None = None,
    run_i: int | None = None,
) -> dict:
    """
    运行BERT微调
    
    Args:
        config: 项目配置
        task: 任务类型（可选，不指定时从数据集自动推断）
        num_classes: 分类类别数（仅分类任务需要）
        aggregation_mode: 测试时增强的聚合模式
        
    Returns:
        微调结果字典
    """
    print("🚀 开始BERT微调...")
    
    # 运行微调
    print("🎓 开始微调...")
    try:
        # 统一架构会自动从UDI推断任务类型和维度，不需要显式传递num_classes等参数
        result = run_finetune(
            config,
            aggregation_mode=aggregation_mode,
            save_name_prefix=save_name_prefix,
            save_name_suffix=save_name_suffix,
            pretrained_dir=pretrained_dir,
            pretrain_exp_name=pretrain_exp_name,
            run_i=run_i,
        )
        print("✅ 微调完成!")
        print(f"📊 最优验证损失: {result['best_val_loss']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ 微调失败: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="单个方法BERT微调脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
        """
    )
    
    # 添加所有参数（包含微调参数）
    add_all_args(parser, include_finetune=True)
    # 输出控制：plain_logs
    parser.add_argument("--plain_logs", action="store_true", help="启用无颜色、无控制符的离线输出（兼容重定向/日志文件，解决乱码）")

    # 增加TTA聚合模式参数
    parser.add_argument(
        "--aggregation_mode",
        type=str,
        default="avg",
        choices=["avg", "best", "learned"],
        help=(
            "测试时增强（TTA）的聚合模式: "
            "'avg' - 对多重采样结果取平均 (用于报告); "
            "'best' - 选择最优结果 (用于分析上限，含标签信息，勿用于正式报告); "
            "'learned' - 使用单独训练的加权聚合器（推荐）。"
        )
    )
    # 仅影响保存目录命名的前后缀（不影响加载预训练所用 experiment_name）
    parser.add_argument("--save_name_prefix", type=str, default=None,
                        help="仅用于保存目录的实验名前缀（预训练加载仍使用原 experiment_name）")
    parser.add_argument("--save_name_suffix", type=str, default=None,
                        help="仅用于保存目录的实验名后缀（预训练加载仍使用原 experiment_name）")
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="显式指定预训练权重目录（包含 config.bin 与 pytorch_model.bin）；优先于按 experiment_name 推断")
    
    # 🆕 新增：灵活的预训练实验名指定
    parser.add_argument("--pretrain_exp_name", type=str, default=None,
                        help="预训练模型实验名（如果与微调实验名不同）；用于从指定的预训练实验加载模型")
    
    # 可选：分类任务显式指定类别数（否则从数据集元信息自动推断）
    parser.add_argument(
        "--num_classes",
        type=int,
        help="分类类别数（可选；若不提供则从数据集元信息自动推断）",
    )
    
    # 解析参数
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print("\n❌ 参数解析失败！传入的参数信息:")
        print("=" * 60)
        print("脚本名称:", sys.argv[0])
        print("所有传入参数:")
        for i, arg in enumerate(sys.argv[1:], 1):
            print(f"  {i:2d}: {arg}")
        print("=" * 60)
        print("请检查参数是否正确，或使用 --help 查看帮助信息")
        print("=" * 60)
        raise
    
    print("🔧 初始化配置...")
    # 提前配置输出模式
    try:
        _configure_output_mode(bool(getattr(args, 'plain_logs', False)))
    except Exception:
        pass
    
    # 创建基础配置
    config = ProjectConfig()
    
    # 如果用户要求显示配置，先显示然后退出
    if args.show_config:
        show_full_config(config)
        return 0
    
    # 应用命令行参数到配置（统一将通用训练参数映射到微调命名空间）
    apply_args_to_config(config, args, common_to="finetune")
    
    # 自动生成实验名称（如果未指定）
    create_experiment_name(config)

    # 验证配置
    try:
        config.validate()
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return 1
    
    # 打印配置摘要
    print_config_summary(config)
    
    # 🆕 检查是否需要重复运行
    repeat_runs = getattr(config, 'repeat_runs', 1)

    if repeat_runs > 1:
        print(f"🔄 启用重复运行模式: {repeat_runs} 次")

        all_results = []
        for run_i in range(repeat_runs):
            print(f"\n{'='*60}")
            print(f"🚀 开始第 {run_i + 1}/{repeat_runs} 次运行")
            print(f"{'='*60}")

            # 设置当前运行编号
            config.current_run_i = run_i
            # 动态设置种子
            actual_seed = config.system.seed + run_i
            config.system.seed = actual_seed

            try:
                # 重新设置种子
                from config import setup_global_seeds
                setup_global_seeds(actual_seed)

                result = run_finetuning(
                    config,
                    aggregation_mode=args.aggregation_mode,
                    save_name_prefix=args.save_name_prefix,
                    save_name_suffix=args.save_name_suffix,
                    pretrained_dir=getattr(args, 'pretrained_dir', None),
                    pretrain_exp_name=getattr(args, 'pretrain_exp_name', None),
                    run_i=run_i,
                )
                result['run_i'] = run_i
                result['seed'] = actual_seed
                all_results.append(result)

                print(f"✅ 第 {run_i + 1} 次运行完成")
                print(f"📊 最优验证损失: {result['best_val_loss']:.4f}")

                # 显示测试结果
                test_metrics = result.get('test_metrics', {})
                if test_metrics:
                    print("📈 测试指标:")
                    for metric, value in test_metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric}: {value:.4f}")

            except Exception as e:
                print(f"❌ 第 {run_i + 1} 次运行失败: {e}")
                continue

        # 聚合统计结果
        if all_results:
            print(f"\n{'='*60}")
            print("📊 聚合统计结果")
            print(f"{'='*60}")

            from src.utils.stats_aggregation import aggregate_experiment_results, print_aggregated_stats

            aggregated = aggregate_experiment_results(
                config, config.experiment_name, len(all_results), "finetune"
            )
            print_aggregated_stats(aggregated, "finetune")

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)

    else:
        # 普通单次运行
        try:
            result = run_finetuning(
                config,
                aggregation_mode=args.aggregation_mode,
                save_name_prefix=args.save_name_prefix,
                save_name_suffix=args.save_name_suffix,
                pretrained_dir=getattr(args, 'pretrained_dir', None),
                pretrain_exp_name=getattr(args, 'pretrain_exp_name', None),
                run_i=None,  # 单个脚本运行时不使用run_i
            )

            print("\n" + "="*60)
            print("🎉 微调完成!")
            print("="*60)

            print(f"📁 模型路径: {result['best_dir']}")
            print(f"📊 最优验证损失: {result['best_val_loss']:.4f}")

            # 显示测试结果
            test_metrics = result['test_metrics']
            print("\n📈 test_metrics:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
            
            Logger.get_logger().current_logger().report_single_value(name="ft_metric", value=result['finetune_metrics'])

            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(0)
            print("exit后仍未结束！！！！！")

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断训练")
            task.mark_failed()
            return 130
        except Exception as e:
            print(f"\n❌ 微调失败: {e}")
            task.mark_failed()
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    main()
