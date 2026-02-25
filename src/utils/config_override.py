#!/usr/bin/env python3
"""
Config override system.

Supports: basic CLI args, JSON full override, and common shortcut args.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict
from config import ProjectConfig


def add_basic_args(parser: argparse.ArgumentParser) -> None:
    """Add basic required args (not overridden by JSON)."""
    
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--method", type=str, required=True, help="Serialization method")
    parser.add_argument("--experiment_group", type=str, help="Experiment group")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--device", type=str, help="Device (cuda:0, cpu, auto)")
    parser.add_argument("--log_style", type=str, choices=["online", "offline"], help="Log style: online=tqdm; offline=summary per 10%% epoch")

    # Repeat runs
    parser.add_argument("--repeat_runs", type=int, default=1, help="Number of repeat runs (default: 1)")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common args (used when no JSON override is provided)."""
    
    # BPE compression args
    bpe_group = parser.add_argument_group('BPE compression')
    # bpe_group.add_argument("--bpe_num_merges", type=int, help="BPE合并次数，0表示不使用BPE")
    # bpe_group.add_argument("--bpe_encode_backend", type=str, choices=["python", "cpp"], 
    #                       default="cpp", help="BPE编码后端")
    bpe_group.add_argument("--bpe_encode_rank_mode", type=str, 
                          choices=["none", "all", "topk", "random", "gaussian"], default="none",
                          help="BPE encode rank mode")
    bpe_group.add_argument("--bpe_encode_rank_k", type=int, help="BPE encode top-K parameter")
    # bpe_group.add_argument("--bpe_encode_rank_min", type=int, help="BPE编码随机范围最小值")
    # bpe_group.add_argument("--bpe_encode_rank_max", type=int, help="BPE编码随机范围最大值")
    bpe_group.add_argument("--bpe_encode_rank_dist", type=str, help="BPE encode random distribution type")
    bpe_group.add_argument("--bpe_eval_mode", type=str, 
                          choices=["all", "topk"], help="BPE eval mode")
    bpe_group.add_argument("--bpe_eval_topk", type=int, help="BPE eval top-K parameter")
    
    # BERT architecture
    # arch_group = parser.add_argument_group('BERT架构')
    # arch_group.add_argument("--hidden_size", type=int, help="隐藏层大小")
    # arch_group.add_argument("--num_layers", type=int, help="层数")
    # arch_group.add_argument("--num_heads", type=int, help="注意力头数")
    
    # Training args (auto-mapped to pretrain or finetune depending on script)
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--epochs", type=int, help="Training epochs")
    train_group.add_argument("--batch_size", type=int, help="Batch size")
    train_group.add_argument("--learning_rate", "--lr", type=float, help="Learning rate")
    
    # Task args
    task_group = parser.add_argument_group('Task')

    task_group.add_argument("--target_property", type=str, help="Regression target property")
    
    # Encoder args
    encoder_group = parser.add_argument_group('Encoder')
    encoder_group.add_argument("--encoder", type=str, choices=["bert", "gte"], help="Encoder type (bert or gte)")


def add_json_override_args(parser: argparse.ArgumentParser) -> None:
    """Add JSON config override args."""
    
    json_group = parser.add_argument_group('JSON config override')
    json_group.add_argument("--config_json", type=str, 
                           help="JSON config override (JSON string or file path)")
    json_group.add_argument("--show_config", action="store_true",
                           help="Show final config and exit")


def apply_args_to_config(config: ProjectConfig, args: argparse.Namespace, *, common_to: str = "pretrain") -> None:
    """Apply CLI args to config object."""
    import json as _json_internal
    
    # Pre-read JSON (if provided) to check override precedence (JSON > common CLI)
    json_dict_for_presence = None
    if hasattr(args, 'config_json') and args.config_json:
        try:
            if args.config_json.strip().startswith('{'):
                json_dict_for_presence = _json_internal.loads(args.config_json)
            else:
                with open(args.config_json, 'r', encoding='utf-8') as _fh:
                    json_dict_for_presence = _json_internal.load(_fh)
        except Exception:
            json_dict_for_presence = None  # parse failed, skip presence check

    def _json_has_path(d: dict | None, dotted: str) -> bool:
        if not isinstance(d, dict):
            return False
        cur = d
        for key in dotted.split('.'):
            if not isinstance(cur, dict) or key not in cur:
                return False
            cur = cur[key]
        return True

    # === 1. Basic args (always apply) ===
    if args.dataset:
        config.dataset.name = args.dataset
        print(f"🎯 dataset.name = {args.dataset}")
    
    if args.method:
        config.serialization.method = args.method
        print(f"🎯 serialization.method = {args.method}")
    
    if hasattr(args, 'experiment_group') and args.experiment_group:
        config.experiment_group = args.experiment_group
        print(f"🎯 experiment_group = {args.experiment_group}")
    
    if hasattr(args, 'experiment_name') and args.experiment_name:
        config.experiment_name = args.experiment_name
        print(f"🎯 experiment_name = {args.experiment_name}")
    
    if hasattr(args, 'device') and args.device:
        config.system.device = args.device
        print(f"🎯 system.device = {args.device}")
    if hasattr(args, 'log_style') and args.log_style:
        config.system.log_style = args.log_style
        print(f"🎯 system.log_style = {args.log_style}")

    # Repeat runs
    if hasattr(args, 'repeat_runs') and args.repeat_runs is not None:
        config.repeat_runs = args.repeat_runs
        print(f"🎯 repeat_runs = {args.repeat_runs}")
    
    # BPE args
    if hasattr(args, 'bpe_num_merges') and args.bpe_num_merges is not None:
        config.serialization.bpe.num_merges = args.bpe_num_merges
        print(f"🎯 serialization.bpe.num_merges = {args.bpe_num_merges}")
    
    bpe_params = {
        'bpe_encode_backend': 'serialization.bpe.engine.encode_backend',
        'bpe_encode_rank_mode': 'serialization.bpe.engine.encode_rank_mode', 
        'bpe_encode_rank_k': 'serialization.bpe.engine.encode_rank_k',
        'bpe_encode_rank_min': 'serialization.bpe.engine.encode_rank_min',
        'bpe_encode_rank_max': 'serialization.bpe.engine.encode_rank_max',
        'bpe_encode_rank_dist': 'serialization.bpe.engine.encode_rank_dist',
    }
    for arg_name, config_path in bpe_params.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            value = getattr(args, arg_name)
            keys = config_path.split('.')
            current = config
            for key in keys[:-1]:
                current = getattr(current, key)
            setattr(current, keys[-1], value)
            print(f"🎯 {config_path} = {value}")
    
    # === 2. JSON config override (highest priority) ===
    if hasattr(args, 'config_json') and args.config_json:
        print("Applying JSON config override...")
        apply_json_config(config, args.config_json)
    
    # === 3. Common arg overrides (only for fields not declared in JSON) ===
    print("Applying common arg overrides...")
    
    # BERT architecture args
    if hasattr(args, 'hidden_size') and args.hidden_size:
        config.bert.architecture.hidden_size = args.hidden_size
        print(f"🎯 bert.architecture.hidden_size = {args.hidden_size}")
    
    if hasattr(args, 'num_layers') and args.num_layers:
        config.bert.architecture.num_hidden_layers = args.num_layers
        print(f"🎯 bert.architecture.num_hidden_layers = {args.num_layers}")
    
    if hasattr(args, 'num_heads') and args.num_heads:
        config.bert.architecture.num_attention_heads = args.num_heads
        print(f"🎯 bert.architecture.num_attention_heads = {args.num_heads}")
    
    # Training param handling
    if hasattr(args, 'epochs') and args.epochs:
        if common_to == "finetune":
            config.bert.finetuning.epochs = args.epochs
            print(f"🎯 bert.finetuning.epochs = {args.epochs}")
        else:
            config.bert.pretraining.epochs = args.epochs
            print(f"🎯 bert.pretraining.epochs = {args.epochs}")
    
    if hasattr(args, 'batch_size') and args.batch_size:
        if common_to == "finetune":
            config.bert.finetuning.batch_size = args.batch_size
            print(f"🎯 bert.finetuning.batch_size = {args.batch_size}")
        else:
            config.bert.pretraining.batch_size = args.batch_size
            print(f"🎯 bert.pretraining.batch_size = {args.batch_size}")
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        if common_to == "finetune":
            config.bert.finetuning.learning_rate = args.learning_rate
            print(f"🎯 bert.finetuning.learning_rate = {args.learning_rate}")
        else:
            config.bert.pretraining.learning_rate = args.learning_rate
            print(f"🎯 bert.pretraining.learning_rate = {args.learning_rate}")
    
    if hasattr(args, 'mult') and args.mult:
        config.serialization.multiple_sampling.num_realizations = args.mult
        config.serialization.multiple_sampling.enabled = args.mult > 1
        print(f"🎯 serialization.multiple_sampling.num_realizations = {args.mult},enable={config.serialization.multiple_sampling.enabled}")
    
    if hasattr(args, 'pool') and args.pool:
        config.bert.architecture.pooling_method = args.pool
        print(f"🎯 bert.architecture.pooling_method = {args.pool}")
    if hasattr(args, 'max_length') and args.max_length:
        config.bert.architecture.max_seq_length = args.max_length
        print(f"🎯 bert.architecture.max_seq_length = {args.max_length}")
    if hasattr(args, 'max_len_policy') and args.max_len_policy:
        config.bert.architecture.max_len_policy = args.max_len_policy
        print(f"🎯 bert.architecture.max_len_policy = {args.max_len_policy}")
    # Task args
    if hasattr(args, 'task') and args.task:
        config.task.type = args.task
        print(f"🎯 task.type = {args.task}")
    
    if hasattr(args, 'target_property') and args.target_property:
        config.task.target_property = args.target_property
        print(f"🎯 task.target_property = {args.target_property}")
    
    # Encoder arg (--encoder)
    if hasattr(args, 'encoder') and args.encoder:
        config.encoder.type = args.encoder
        print(f"🎯 encoder.type = {args.encoder}")
    
# Redundant finetune_* params removed; unified --epochs etc. used instead


def apply_json_config(config: ProjectConfig, json_input: str) -> None:
    """Apply JSON config override (string or file path)."""
    try:
        if json_input.strip().startswith('{'):
            config_dict = json.loads(json_input)
            print("Applied JSON string config")
        else:
            with open(json_input, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            print(f"Applied JSON file config: {json_input}")
        
        recursive_override(config, config_dict)
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        raise
    except FileNotFoundError:
        print(f"Config file not found: {json_input}")
        raise
    except Exception as e:
        print(f"JSON config override failed: {e}")
        raise


def recursive_override(config_obj: Any, override_dict: Dict[str, Any], path: str = "") -> None:
    """Recursively override config object fields."""
    for key, value in override_dict.items():
        current_path = f"{path}.{key}" if path else key
        
        if hasattr(config_obj, key):
            current_attr = getattr(config_obj, key)
            
            if isinstance(value, dict) and hasattr(current_attr, '__dict__'):
                recursive_override(current_attr, value, current_path)
            else:
                setattr(config_obj, key, value)
                print(f"  {current_path} = {value}")
        else:
            print(f"  [warn] skipping unknown config: {current_path}")


def create_experiment_name(config: ProjectConfig) -> None:
    """Auto-generate experiment name if not specified."""
    if not config.experiment_name:
        config.experiment_name = f"{config.dataset.name}-{config.serialization.method}"
        print(f"Auto-generated experiment name: {config.experiment_name}")


def print_config_summary(config: ProjectConfig) -> None:
    """Print config summary."""
    show_full_config(config)



def show_full_config(config: ProjectConfig) -> None:
    """Show full config in JSON format."""
    print("\n" + "="*60)
    print("Full config")
    print("="*60)
    
    config_dict = config.to_dict()
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    print("="*60)


# Convenience function
def add_all_args(parser: argparse.ArgumentParser, include_finetune: bool = True) -> None:
    """Add all args at once."""
    add_basic_args(parser)
    
    # BPE args (needed for both pretrain and finetune)
    bpe_group = parser.add_argument_group('BPE compression')
    # bpe_group.add_argument("--bpe_num_merges", type=int, help="BPE合并次数，0表示不使用BPE")
    # bpe_group.add_argument("--bpe_encode_backend", type=str, choices=["python", "cpp"], 
    #                       default="cpp", help="BPE编码后端")
    bpe_group.add_argument("--bpe_encode_rank_mode", type=str, 
                          choices=["none", "all", "topk", "random", "gaussian"], default="all",
                          help="BPE encode rank mode")
    bpe_group.add_argument("--bpe_encode_rank_k", type=int, help="BPE encode top-K parameter")
    # bpe_group.add_argument("--bpe_encode_rank_min", type=int, help="BPE编码随机范围最小值")
    # bpe_group.add_argument("--bpe_encode_rank_max", type=int, help="BPE编码随机范围最大值")
    # bpe_group.add_argument("--bpe_encode_rank_dist", type=str, help="BPE编码随机分布类型")
    # bpe_group.add_argument("--bpe_eval_mode", type=str, 
    #                       choices=["all", "topk"], help="BPE评估模式")
    # bpe_group.add_argument("--bpe_eval_topk", type=int, help="BPE评估Top-K参数")
    
    # Common training args (same name for both pretrain and finetune)
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--epochs", type=int, help="Training epochs")
    train_group.add_argument("--batch_size", type=int, help="Batch size")
    train_group.add_argument("--learning_rate", "--lr", type=float, help="Learning rate")
    train_group.add_argument("--mult", type=int, help="Multiple sampling count")
    train_group.add_argument("--pool", type=str, help="Sequence pooling method")
    train_group.add_argument("--max_length", type=int, help="Max sequence length")
    train_group.add_argument("--max_len_policy", type=str, help="Max length policy")

    # Pretrain-only architecture args (only used in pretrain script)
    # arch_group = parser.add_argument_group('BERT架构')
    # arch_group.add_argument("--hidden_size", type=int, help="隐藏层大小")
    # arch_group.add_argument("--num_layers", type=int, help="层数")
    # arch_group.add_argument("--num_heads", type=int, help="注意力头数")

    # Encoder args (both pretrain and finetune)
    encoder_group = parser.add_argument_group('Encoder')
    encoder_group.add_argument("--encoder", type=str, choices=["bert", "gte"], help="Encoder type (bert or gte)")
    
    # Task args (finetune only)
    if include_finetune:
        task_group = parser.add_argument_group('Task')
        task_group.add_argument("--target_property", type=str, help="Target property name")
    
    add_json_override_args(parser)
