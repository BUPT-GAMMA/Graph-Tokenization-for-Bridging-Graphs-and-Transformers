#!/usr/bin/env python3
"""
单个方法 BERT 预训练脚本（权威用法）
================================

本脚本是项目内“标准/唯一”的预训练入口，被批量脚本与超参搜索脚本调用。

必须参数（命令行）:
  --dataset DATASET   例如: qm9, qm9test, zinc, ...
  --method  METHOD    例如: feuler, eulerian, cpp, fcpp, topo, smilesHeyBaby
  --bpe_encode_rank_mode MODE BPE模式: none|all|topk|random|gaussian
  --epochs E                  训练轮数
  --batch_size B              批大小
  --learning_rate LR          学习率
  --config_json JSON_OR_PATH  高级配置（JSON字符串或文件路径），用于嵌套项覆盖

推荐实践：用显式 CLI 参数或 --config_json，不要使用其他“覆盖器”。

示例：用默认配置训练一个可被微调脚本识别的“默认预训练模型”
  python run_pretrain.py \
    --dataset qm9 \
    --method feuler \
    --experiment_group large_bs_hyperopt_all \
    --experiment_name  large_bs_all_pt_default \
    --bpe_encode_rank_mode all

示例：显式指定基础训练超参
  python run_pretrain.py \
    --dataset qm9 --method feuler \
    --experiment_group large_bs_hyperopt_all \
    --experiment_name  large_bs_all_pt_default \
    --bpe_encode_rank_mode all \
    --epochs 200 --batch_size 256 --learning_rate 4e-4

示例：用 JSON 覆盖嵌套配置（不建议滥用，仅用于必要场景）
  python run_pretrain.py \
    --dataset qm9 --method feuler \
    --experiment_group large_bs_hyperopt_all \
    --experiment_name  large_bs_all_pt_default \
    --config_json '{"serialization": {"bpe": {"engine": {"encode_rank_mode": "all"}}}}'
"""

from __future__ import annotations

import argparse
import sys
import os
import re
import io
from pathlib import Path
import time
from typing import Optional

from src.training.pretrain_pipeline import train_bert_mlm
from clearml import Task


# 设置项目路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir('/home/gzy/py/tokenizerGraph')
from config import ProjectConfig  # noqa: E402
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
            task_name=f"pretrain_manual_{int(time.time())}",
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
    # 环境层面
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "C.UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")

    # 流层面
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        try:
            # Python 3.7+ 支持 reconfigure
            stream.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
        except Exception:
            try:
                wrapped = io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
                setattr(sys, stream_name, wrapped)
            except Exception:
                # 最后兜底：不替换
                pass


def _configure_output_mode(offline: bool):
    """根据 offline 模式配置输出：禁用颜色、去除ANSI、尽量减少伪TTY影响。"""
    _ensure_utf8_streams()

    if offline:
        # 关闭颜色、提示TTY为dumb，关闭tqdm等进度条
        os.environ["NO_COLOR"] = "1"
        os.environ["CLICOLOR"] = "0"
        os.environ["FORCE_COLOR"] = "0"
        os.environ["TERM"] = "dumb"
        os.environ.setdefault("TQDM_DISABLE", "1")
        # 包装输出，移除ANSI控制符
        if not isinstance(sys.stdout, _AnsiStrippingWriter):
            sys.stdout = _AnsiStrippingWriter(sys.stdout)
        if not isinstance(sys.stderr, _AnsiStrippingWriter):
            sys.stderr = _AnsiStrippingWriter(sys.stderr)

def run_pretraining(config: ProjectConfig, run_i: Optional[int] = None) -> dict:
    """
    运行BERT预训练

    Args:
        config: 项目配置
        run_i: 重复运行编号

    Returns:
        训练结果字典
    """
    print("🚀 开始BERT预训练...")

    # 运行预训练
    print("🎓 开始模型训练...")
    try:
        result = train_bert_mlm(config, run_i=run_i)
        print("✅ 预训练完成!")

        print(f"📊 最优验证损失: {result['best_val_loss']:.4f}")

        return result

    except Exception as e:
        print(f"❌ 预训练失败: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="单个方法BERT预训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（默认BPE all模式）
  python run_pretrain.py --dataset qm9test --method feuler
  
  # 无BPE压缩（使用原始序列）
  python run_pretrain.py --dataset qm9test --method feuler --bpe_encode_rank_mode none
  
  # BPE Top-K压缩
  python run_pretrain.py --dataset qm9test --method feuler \\
    --bpe_encode_rank_mode topk --bpe_encode_rank_k 1000
  
  # BPE随机压缩
  python run_pretrain.py --dataset qm9test --method feuler \\
    --bpe_encode_rank_mode random --bpe_encode_rank_min 100 --bpe_encode_rank_max 2000
  
  # BPE高斯采样压缩
  python run_pretrain.py --dataset qm9test --method feuler \\
    --bpe_encode_rank_mode gaussian --bpe_encode_rank_k 1500
  
  # 自定义训练参数
  python run_pretrain.py --dataset qm9test --method feuler --epochs 10 --batch_size 32
  
  # JSON配置覆盖
  python run_pretrain.py --dataset qm9test --method feuler \\
    --config_json '{"bert": {"pretraining": {"epochs": 20, "learning_rate": 1e-4}}, 
                    "serialization": {"bpe": {"engine": {"encode_rank_mode": "topk", "encode_rank_k": 1000}}}}'
        """
    )
    
    # 添加所有参数（不包含微调参数）
    add_all_args(parser, include_finetune=False)
    # 追加输出控制参数
    parser.add_argument("--plain_logs", action="store_true", help="启用无颜色、无控制符的离线输出（兼容重定向/日志文件，解决乱码）")
    
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
    
    # 创建基础配置
    config = ProjectConfig()
    
    # 如果用户要求显示配置，先显示然后退出
    if args.show_config:
        show_full_config(config)
        return 0
    
    # 应用命令行参数到配置
    apply_args_to_config(config, args)
    
    # 自动生成实验名称（如果未指定）
    create_experiment_name(config)
    
    # 验证配置
    try:
        config.validate()
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return 1
    
    # 根据 offline/plian_logs 设置输出模式
    try:
        # 从 args 或配置中判断 offline（优先命令行）
        offline_flag = bool(getattr(args, 'plain_logs', False))
        # 若配置中定义 system.log_style=offline，也视为离线
        # 由于此处尚未 apply_args_to_config 完成合并，先用命令行开关控制输出包装
        _configure_output_mode(offline_flag)
    except Exception:
        pass

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

                result = run_pretraining(config, run_i=run_i)
                result['run_i'] = run_i
                result['seed'] = actual_seed
                all_results.append(result)

                print(f"✅ 第 {run_i + 1} 次运行完成")
                print(f"📊 最优验证损失: {result['best_val_loss']:.4f}")

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
                config, config.experiment_name, len(all_results), "pretrain"
            )
            print_aggregated_stats(aggregated, "pretrain")

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            raise
        os._exit(0)

    else:
        # 普通单次运行
        try:
            result = run_pretraining(config)

            print("\n" + "="*60)
            print("🎉 预训练完成!")
            print("="*60)

            print(f"📁 模型保存路径: {result['model_dir']}")
            print(f"🏷️ 实验名称: {config.experiment_name}")
            print(f"📊 最优验证损失: {result['best_val_loss']:.4f}")

            print("\n💡 可以使用以下命令进行微调:")
            print(f"python run_finetune.py --dataset {config.dataset.name} --method {config.serialization.method}")
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(0)
            print("exit后仍未结束！！！！！")

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断训练")
            return 130
        except Exception as e:
            print(f"\n❌ 预训练失败: {e}")
            task.mark_failed(message=str(e))
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
