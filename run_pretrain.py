#!/usr/bin/env python3
"""
BERT pre-training script (single method).

This is the standard entry point for pre-training, called by batch scripts and
hyperparameter search scripts.

Required args:
  --dataset DATASET             e.g. qm9, qm9test, zinc, ...
  --method  METHOD              e.g. feuler, eulerian, cpp, fcpp, topo, smiles
  --bpe_encode_rank_mode MODE   BPE mode: none|all|topk|random|gaussian
  --epochs E                    training epochs
  --batch_size B                batch size
  --learning_rate LR            learning rate
  --config_json JSON_OR_PATH    advanced config override (JSON string or file path)

Examples:
  python run_pretrain.py --dataset qm9 --method feuler \
    --experiment_group my_group --bpe_encode_rank_mode all

  python run_pretrain.py --dataset qm9 --method feuler \
    --epochs 200 --batch_size 256 --learning_rate 4e-4
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


# Project path setup
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




# ClearML task init (supports both direct run and Agent execution)
current_task = Task.current_task()
if current_task is not None:
    task = current_task
    print(f"Using existing ClearML task: {task.name} (ID: {task.id})")
else:
    try:
        task = Task.init(
            project_name="TokenizerGraph",
            task_name=f"pretrain_manual_{int(time.time())}",
            auto_connect_frameworks=True
        )
        print(f"ClearML task initialized: {task.name} (ID: {task.id})")
    except Exception as e:
        print(f"ClearML init failed, continuing without it: {e}")
        task = None


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
    # Environment level
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "C.UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")

    # Stream level
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        try:
            # Python 3.7+ supports reconfigure
            stream.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
        except Exception:
            try:
                wrapped = io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
                setattr(sys, stream_name, wrapped)
            except Exception:
                # Last resort: do not replace
                pass


def _configure_output_mode(offline: bool):
    """Configure output mode: disable colors, strip ANSI, ensure UTF-8."""
    _ensure_utf8_streams()

    if offline:
        # Disable color, tqdm, set TERM=dumb
        os.environ["NO_COLOR"] = "1"
        os.environ["CLICOLOR"] = "0"
        os.environ["FORCE_COLOR"] = "0"
        os.environ["TERM"] = "dumb"
        os.environ.setdefault("TQDM_DISABLE", "1")
        # Wrap output to strip ANSI codes
        if not isinstance(sys.stdout, _AnsiStrippingWriter):
            sys.stdout = _AnsiStrippingWriter(sys.stdout)
        if not isinstance(sys.stderr, _AnsiStrippingWriter):
            sys.stderr = _AnsiStrippingWriter(sys.stderr)

def run_pretraining(config: ProjectConfig, run_i: Optional[int] = None) -> dict:
    """
    Run BERT pre-training.

    Args:
        config: Project config
        run_i: Repeat run index

    Returns:
        Training result dict
    """
    print("Starting BERT pre-training...")

    print("Training model...")
    try:
        result = train_bert_mlm(config, run_i=run_i)
        print("Pre-training complete!")

        print(f"Best val loss: {result['best_val_loss']:.4f}")

        return result

    except Exception as e:
        print(f"Pre-training failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BERT pre-training script (single method)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pretrain.py --dataset qm9test --method feuler
  python run_pretrain.py --dataset qm9test --method feuler --bpe_encode_rank_mode none
  python run_pretrain.py --dataset qm9test --method feuler --epochs 10 --batch_size 32
        """
    )
    
    # Add all args (excluding finetune args)
    add_all_args(parser, include_finetune=False)
    parser.add_argument("--plain_logs", action="store_true", help="Enable plain text output (no color/ANSI, for log files)")
    
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
    
    config = ProjectConfig()
    
    # Show config and exit if requested
    if args.show_config:
        show_full_config(config)
        return 0
    
    # Apply CLI args to config
    apply_args_to_config(config, args)
    
    # Auto-generate experiment name if not specified
    create_experiment_name(config)
    # if config.serialization.bpe.engine.encode_rank_mode == 'none' and config.encoder.type == 'gte':
    #     print(f"Warn: bpe编码模式为Raw，且encoder为GTE,降低bs为一半（由于此encoder是动态显存大小，随序列长度正比）")
    #     config.bert.pretraining.batch_size = config.bert.pretraining.batch_size // 2
    
    
    # Validate config
    try:
        config.validate()
    except Exception as e:
        print(f"Config validation failed: {e}")
        return 1
    
    # Configure output mode based on --plain_logs flag
    try:
        offline_flag = bool(getattr(args, 'plain_logs', False))
        _configure_output_mode(offline_flag)
    except Exception:
        pass

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
            result = run_pretraining(config, run_i=run_i)
            result['seed'] = seed
            all_results.append(result)
            print(f"Run {run_i + 1} complete")
            print(f"Best val loss: {result['best_val_loss']:.4f}")
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
            config, config.experiment_name, len(all_results), "pretrain"
        )
        print_aggregated_stats(aggregated, "pretrain")
  
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        raise
    os._exit(0)

    # else:
    #     # 普通单次运行
    #     try:
    #         result = run_pretraining(config)

    #         print("\n" + "="*60)
    #         print("🎉 预训练完成!")
    #         print("="*60)

    #         print(f"📁 模型保存路径: {result['model_dir']}")
    #         print(f"🏷️ 实验名称: {config.experiment_name}")
    #         print(f"📊 最优验证损失: {result['best_val_loss']:.4f}")

    #         print("\n💡 可以使用以下命令进行微调:")
    #         print(f"python run_finetune.py --dataset {config.dataset.name} --method {config.serialization.method}")
    #         try:
    #             sys.stdout.flush()
    #             sys.stderr.flush()
    #         except Exception:
    #             pass
    #         os._exit(0)
    #         print("exit后仍未结束！！！！！")

    #     except KeyboardInterrupt:
    #         print("\n⚠️ 用户中断训练")
    #         return 130
    #     except Exception as e:
    #         print(f"\n❌ 预训练失败: {e}")
    #         task.mark_failed()
    #         import traceback
    #         traceback.print_exc()
    #         raise


if __name__ == "__main__":
    main()
