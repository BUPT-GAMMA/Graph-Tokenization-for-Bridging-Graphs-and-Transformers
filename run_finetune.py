#!/usr/bin/env python3
"""
BERT fine-tuning script (single method).

This is the standard entry point for fine-tuning, called by batch/search scripts.

Required args:
  --dataset DATASET             e.g. qm9, qm9test, zinc, ...
  --method  METHOD              e.g. feuler, eulerian, cpp, fcpp, topo, smiles

Loading pretrained model (choose one):
  --pretrained_dir PATH         explicit model directory (with config.bin & pytorch_model.bin)
  --pretrain_exp_name NAME      pretrain experiment name, loads from standard directory layout

Common args:
  --experiment_group NAME       experiment group
  --experiment_name  NAME       finetune experiment name (only affects save path)
  --bpe_encode_rank_mode MODE   BPE mode: none|all|topk|random|gaussian
  --epochs / --batch_size / --learning_rate / --weight_decay / --warmup_ratio / --max_grad_norm
  --config_json JSON_OR_PATH    advanced config override (JSON string or file path)
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
# Project path setup
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

# ClearML task init (supports both direct run and Agent execution)
current_task = Task.current_task()
if current_task is not None:
    task: Task = current_task
    print(f"Using existing ClearML task: {task.name} (ID: {task.id})")
else:
    try:
        task: Task = Task.init(
            project_name="TokenizerGraph",
            task_name=f"finetune_manual_{int(time.time())}",
            auto_connect_frameworks=True
        )
        print(f"ClearML task initialized: {task.name} (ID: {task.id})")
    except Exception as e:
        print(f"ClearML init failed, continuing without it: {e}")
        task: Task = None

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)


class _AnsiStrippingWriter:
    """Wraps stdout/stderr to strip ANSI escape codes and ensure UTF-8."""
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
    """Ensure stdout/stderr use UTF-8 encoding."""
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

    # Suppress TensorFlow verbose logs (irrelevant to this training pipeline)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _configure_output_mode(offline: bool):
    """Configure output mode: disable colors, strip ANSI, ensure UTF-8."""
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
    Run BERT fine-tuning.
    
    Args:
        config: Project config
        aggregation_mode: TTA aggregation mode
        
    Returns:
        Fine-tuning result dict
    """
    print("Starting BERT fine-tuning...")
    
    print("Fine-tuning...")
    try:
        # Unified architecture auto-infers task type and dimensions from UDI
        result = run_finetune(
            config,
            aggregation_mode=aggregation_mode,
            save_name_prefix=save_name_prefix,
            save_name_suffix=save_name_suffix,
            pretrained_dir=pretrained_dir,
            pretrain_exp_name=pretrain_exp_name,
            run_i=run_i,
        )
        print("Fine-tuning complete!")
        print(f"Best val loss: {result['best_val_loss']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BERT fine-tuning script (single method)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_finetune.py --dataset qm9 --method feuler --pretrain_exp_name my_pretrain
        """
    )
    
    # Add all args (including finetune args)
    add_all_args(parser, include_finetune=True)
    parser.add_argument("--plain_logs", action="store_true", help="Enable plain text output (no color/ANSI, for log files)")

    # TTA aggregation mode
    parser.add_argument(
        "--aggregation_mode",
        type=str,
        default="learned",
        choices=["avg", "best", "learned"],
        help=(
            "TTA aggregation mode: "
            "'avg' - average over multiple sampling results (for reporting); "
            "'best' - select best result (for upper bound analysis, uses labels); "
            "'learned' - use trained weighted aggregator (recommended)."
        )
    )
    # Save directory naming prefixes/suffixes (don't affect pretrained model loading)
    parser.add_argument("--save_name_prefix", type=str, default=None,
                        help="Prefix for save directory name (pretrain loading uses original experiment_name)")
    parser.add_argument("--save_name_suffix", type=str, default=None,
                        help="Suffix for save directory name (pretrain loading uses original experiment_name)")
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="Explicit pretrained weights directory (with config.bin & pytorch_model.bin)")
    
    # Flexible pretrain experiment name
    parser.add_argument("--pretrain_exp_name", type=str, default=None,
                        help="Pretrain experiment name (if different from finetune experiment name)")
    
    # Optional: explicit num_classes for classification (otherwise auto-inferred from dataset)
    parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of classes for classification (optional; auto-inferred from dataset metadata if not provided)",
    )
    
    # Parse args
    try:
        args = parser.parse_args()
    except SystemExit as e:
        print("\nArgument parsing failed! Arguments received:")
        print("=" * 60)
        print("Script:", sys.argv[0])
        print("All args:")
        for i, arg in enumerate(sys.argv[1:], 1):
            print(f"  {i:2d}: {arg}")
        print("=" * 60)
        print("Check arguments or use --help")
        print("=" * 60)
        raise
    
    print("Initializing config...")
    # Configure output mode early
    try:
        _configure_output_mode(bool(getattr(args, 'plain_logs', False)))
    except Exception:
        pass
    
    config = ProjectConfig()
    
    # Show config and exit if requested
    if args.show_config:
        show_full_config(config)
        return 0
    
    # Apply CLI args to config (map common training params to finetune namespace)
    apply_args_to_config(config, args, common_to="finetune")
    
    # Auto-generate experiment name if not specified
    create_experiment_name(config)
    # if config.serialization.bpe.engine.encode_rank_mode == 'none' and config.encoder.type == 'gte':
    #     print(f"Warn: bpe编码模式为Raw，且encoder为GTE,降低bs为一半（由于此encoder是动态显存大小，随序列长度正比）")
    #     config.bert.finetuning.batch_size = config.bert.finetuning.batch_size // 2
        
    # Validate config
    try:
        config.validate()
    except Exception as e:
        print(f"Config validation failed: {e}")
        return 1
    
    # Print config summary
    print_config_summary(config)
    raw_seed=config.system.seed
    
    # Check if repeated runs are needed
    repeat_runs = getattr(config, 'repeat_runs', 1)

    print(f"Repeat runs: {repeat_runs}")
    all_results = []
    for run_i in range(repeat_runs):
          print(f"\n{'='*60}")
          print(f"Starting run {run_i + 1}/{repeat_runs}")
          print(f"{'='*60}")
          
          seed=raw_seed+run_i
          try:
              from config import setup_global_seeds
              setup_global_seeds(seed)
              result = run_finetuning(
                  config,
                  aggregation_mode=args.aggregation_mode,
                  save_name_prefix=args.save_name_prefix,
                  save_name_suffix=args.save_name_suffix,
                  pretrained_dir=getattr(args, 'pretrained_dir', None),
                  pretrain_exp_name=getattr(args, 'pretrain_exp_name', None),
                  run_i=run_i,
              )
              
              result['seed'] = seed
              all_results.append(result)
              print(f"Run {run_i + 1} complete")
              print(f"Best val loss: {result['best_val_loss']:.4f}")
              
              # Display test results
              test_metrics = result.get('test_metrics', {})
              if test_metrics:
                  print("Test metrics:")
                  for metric, value in test_metrics.items():
                      if isinstance(value, (int, float)):
                          print(f"  {metric}: {value:.4f}")
          except Exception as e:
              print(f"Run {run_i + 1} failed: {e}")
              task.mark_failed()
              raise
    # Aggregate statistics
    if all_results:
        print(f"\n{'='*60}")
        print("Aggregated statistics")
        print(f"{'='*60}")
        from src.utils.stats_aggregation import aggregate_experiment_results, print_aggregated_stats
        aggregated = aggregate_experiment_results(
            config, config.experiment_name, len(all_results), "finetune"
        )
        print_aggregated_stats(aggregated, "finetune")
        task.get_logger().report_single_value(name="ft_metric", value=aggregated['statistics']['test']['avg']['pk']['mean'])
        
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)

    # else:
    #     # 普通单次运行
    #     try:
    #         result = run_finetuning(
    #             config,
    #             aggregation_mode=args.aggregation_mode,
    #             save_name_prefix=args.save_name_prefix,
    #             save_name_suffix=args.save_name_suffix,
    #             pretrained_dir=getattr(args, 'pretrained_dir', None),
    #             pretrain_exp_name=getattr(args, 'pretrain_exp_name', None),
    #             run_i=None,  # 单个脚本运行时不使用run_i
    #         )

    #         print("\n" + "="*60)
    #         print("🎉 微调完成!")
    #         print("="*60)

    #         print(f"📁 模型路径: {result['best_dir']}")
    #         print(f"📊 最优验证损失: {result['best_val_loss']:.4f}")

    #         # 显示测试结果
    #         test_metrics = result['test_metrics']
    #         print("\n📈 test_metrics:")
    #         for metric, value in test_metrics.items():
    #             if isinstance(value, (int, float)):
    #                 print(f"  {metric}: {value:.4f}")
            
    #         task.get_logger().report_single_value(name="ft_metric", value=result['finetune_metrics'])

    #         try:
    #             sys.stdout.flush()
    #             sys.stderr.flush()
    #         except Exception:
    #             pass
    #         os._exit(0)
    #         print("exit后仍未结束！！！！！")

    #     except KeyboardInterrupt:
    #         print("\n⚠️ 用户中断训练")
    #         task.mark_failed()
    #         return 130
    #     except Exception as e:
    #         print(f"\n❌ 微调失败: {e}")
    #         task.mark_failed()
    #         import traceback
    #         traceback.print_exc()
    #         return 1


if __name__ == "__main__":
    main()
