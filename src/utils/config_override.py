#!/usr/bin/env python3
"""
🚀 简洁配置覆盖系统
==================

设计原则：
- 基础参数：dataset, method, experiment_name, experiment_group, device
- JSON完整覆盖：除基础参数外，所有配置都通过JSON一次性覆盖
- 常用参数：如果没有JSON，提供少量常用参数的快捷设置

特性:
- 📝 JSON完整配置覆盖
- 🎯 基础参数 + 常用参数
- 🔄 自动类型转换
- 🎨 简洁实用
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict
from config import ProjectConfig


def add_basic_args(parser: argparse.ArgumentParser) -> None:
    """添加基础必需参数（这些参数不会被JSON覆盖）"""
    
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--method", type=str, required=True, help="序列化方法")
    parser.add_argument("--experiment_group", type=str, help="实验分组")
    parser.add_argument("--experiment_name", type=str, help="实验名称")
    parser.add_argument("--device", type=str, help="设备 (cuda:0, cpu, auto)")
    parser.add_argument("--log_style", type=str, choices=["online", "offline"], help="日志样式：online=使用tqdm；offline=每个epoch按10%输出摘要")

    # 🆕 重复运行参数
    parser.add_argument("--repeat_runs", type=int, default=1, help="重复运行次数，默认1次（不重复）")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """添加常用参数（仅在没有JSON覆盖时使用）"""
    
    # BPE压缩参数
    bpe_group = parser.add_argument_group('BPE压缩配置')
    # bpe_group.add_argument("--bpe_num_merges", type=int, help="BPE合并次数，0表示不使用BPE")
    # bpe_group.add_argument("--bpe_encode_backend", type=str, choices=["python", "cpp"], 
    #                       default="cpp", help="BPE编码后端")
    bpe_group.add_argument("--bpe_encode_rank_mode", type=str, 
                          choices=["none", "all", "topk", "random", "gaussian"], default="none",
                          help="BPE编码排序模式")
    # bpe_group.add_argument("--bpe_encode_rank_k", type=int, help="BPE编码Top-K参数")
    # bpe_group.add_argument("--bpe_encode_rank_min", type=int, help="BPE编码随机范围最小值")
    # bpe_group.add_argument("--bpe_encode_rank_max", type=int, help="BPE编码随机范围最大值")
    bpe_group.add_argument("--bpe_encode_rank_dist", type=str, help="BPE编码随机分布类型")
    bpe_group.add_argument("--bpe_eval_mode", type=str, 
                          choices=["all", "topk"], help="BPE评估模式")
    bpe_group.add_argument("--bpe_eval_topk", type=int, help="BPE评估Top-K参数")
    
    # BERT架构
    # arch_group = parser.add_argument_group('BERT架构')
    # arch_group.add_argument("--hidden_size", type=int, help="隐藏层大小")
    # arch_group.add_argument("--num_layers", type=int, help="层数")
    # arch_group.add_argument("--num_heads", type=int, help="注意力头数")
    
    # 训练参数 (根据脚本自动映射到预训练或微调)
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument("--epochs", type=int, help="训练轮数")
    train_group.add_argument("--batch_size", type=int, help="批次大小")
    train_group.add_argument("--learning_rate", "--lr", type=float, help="学习率")
    
    # 任务参数
    task_group = parser.add_argument_group('任务参数')

    task_group.add_argument("--target_property", type=str, help="回归目标属性")
    
    # 编码器参数（简化：仅接收 bert/gte）
    encoder_group = parser.add_argument_group('编码器参数')
    encoder_group.add_argument("--encoder", type=str, choices=["bert", "gte"], help="编码器类型（bert 或 gte）")


def add_json_override_args(parser: argparse.ArgumentParser) -> None:
    """添加JSON配置覆盖参数"""
    
    json_group = parser.add_argument_group('JSON配置覆盖')
    json_group.add_argument("--config_json", type=str, 
                           help="JSON配置覆盖 (JSON字符串或文件路径)")
    json_group.add_argument("--show_config", action="store_true",
                           help="显示最终配置内容")


def apply_args_to_config(config: ProjectConfig, args: argparse.Namespace, *, common_to: str = "pretrain") -> None:
    """应用命令行参数到配置对象"""
    import json as _json_internal
    
    # 预读取 JSON（若提供），用于后续判断覆盖优先级（JSON > 常用CLI）
    json_dict_for_presence = None
    if hasattr(args, 'config_json') and args.config_json:
        try:
            if args.config_json.strip().startswith('{'):
                json_dict_for_presence = _json_internal.loads(args.config_json)
            else:
                with open(args.config_json, 'r', encoding='utf-8') as _fh:
                    json_dict_for_presence = _json_internal.load(_fh)
        except Exception:
            json_dict_for_presence = None  # 解析失败则放弃 presence 保护

    def _json_has_path(d: dict | None, dotted: str) -> bool:
        if not isinstance(d, dict):
            return False
        cur = d
        for key in dotted.split('.'):
            if not isinstance(cur, dict) or key not in cur:
                return False
            cur = cur[key]
        return True

    # === 1. 基础参数（总是生效） ===
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

    # 🆕 处理重复运行参数
    if hasattr(args, 'repeat_runs') and args.repeat_runs is not None:
        config.repeat_runs = args.repeat_runs
        print(f"🎯 repeat_runs = {args.repeat_runs}")
    
    # BPE参数
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
    
    # === 2. JSON配置覆盖（优先级最高，但仅覆盖其声明的字段） ===
    if hasattr(args, 'config_json') and args.config_json:
        print("📝 应用JSON配置覆盖...")
        apply_json_config(config, args.config_json)
    
    # === 3. 常用参数覆盖（当提供JSON时，仅覆盖JSON未声明的字段） ===
    print("🔧 应用常用参数覆盖...")
    
    # BERT架构参数
    if hasattr(args, 'hidden_size') and args.hidden_size:
        config.bert.architecture.hidden_size = args.hidden_size
        print(f"🎯 bert.architecture.hidden_size = {args.hidden_size}")
    
    if hasattr(args, 'num_layers') and args.num_layers:
        config.bert.architecture.num_hidden_layers = args.num_layers
        print(f"🎯 bert.architecture.num_hidden_layers = {args.num_layers}")
    
    if hasattr(args, 'num_heads') and args.num_heads:
        config.bert.architecture.num_attention_heads = args.num_heads
        print(f"🎯 bert.architecture.num_attention_heads = {args.num_heads}")
    
    # 🆕 简化的训练参数处理
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
    
    # 任务参数
    if hasattr(args, 'task') and args.task:
        config.task.type = args.task
        print(f"🎯 task.type = {args.task}")
    
    if hasattr(args, 'target_property') and args.target_property:
        config.task.target_property = args.target_property
        print(f"🎯 task.target_property = {args.target_property}")
    
    # 🆕 编码器参数（--encoder）
    if hasattr(args, 'encoder') and args.encoder:
        config.encoder.type = args.encoder
        print(f"🎯 encoder.type = {args.encoder}")
    
# 删除冗余的finetune_*参数处理，统一使用--epochs等通用参数


def apply_json_config(config: ProjectConfig, json_input: str) -> None:
    """应用JSON配置覆盖"""
    try:
        # 判断是文件路径还是JSON字符串
        if json_input.strip().startswith('{'):
            # JSON字符串
            config_dict = json.loads(json_input)
            print("📝 应用JSON字符串配置")
        else:
            # 文件路径
            with open(json_input, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            print(f"📝 应用JSON文件配置: {json_input}")
        
        # 递归覆盖配置
        recursive_override(config, config_dict)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        raise
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {json_input}")
        raise
    except Exception as e:
        print(f"❌ JSON配置覆盖失败: {e}")
        raise


def recursive_override(config_obj: Any, override_dict: Dict[str, Any], path: str = "") -> None:
    """递归覆盖配置对象"""
    for key, value in override_dict.items():
        current_path = f"{path}.{key}" if path else key
        
        if hasattr(config_obj, key):
            current_attr = getattr(config_obj, key)
            
            if isinstance(value, dict) and hasattr(current_attr, '__dict__'):
                # 嵌套对象，递归处理
                recursive_override(current_attr, value, current_path)
            else:
                # 简单值，直接设置
                setattr(config_obj, key, value)
                print(f"🎯 {current_path} = {value}")
        else:
            print(f"⚠️ 跳过不存在的配置: {current_path}")


def create_experiment_name(config: ProjectConfig) -> None:
    """如果没有指定实验名称，自动生成一个"""
    if not config.experiment_name:
        config.experiment_name = f"{config.dataset.name}-{config.serialization.method}"
        print(f"🏷️ 自动生成实验名称: {config.experiment_name}")


def print_config_summary(config: ProjectConfig) -> None:
    """打印配置摘要"""
    show_full_config(config)



def show_full_config(config: ProjectConfig) -> None:
    """显示完整配置内容（JSON格式）"""
    print("\n" + "="*60)
    print("🔍 完整配置内容")
    print("="*60)
    
    # 构建主要配置字典
    config_dict = config.to_dict()
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    print("\n💡 可以将上述JSON内容保存到文件，然后用 --config_json 参数加载")
    print("="*60)


# 便捷函数
def add_all_args(parser: argparse.ArgumentParser, include_finetune: bool = True) -> None:
    """一键添加所有参数"""
    add_basic_args(parser)
    
    # BPE参数在预训练和微调中都需要
    bpe_group = parser.add_argument_group('BPE压缩配置')
    bpe_group.add_argument("--bpe_num_merges", type=int, help="BPE合并次数，0表示不使用BPE")
    bpe_group.add_argument("--bpe_encode_backend", type=str, choices=["python", "cpp"], 
                          default="cpp", help="BPE编码后端")
    bpe_group.add_argument("--bpe_encode_rank_mode", type=str, 
                          choices=["none", "all", "topk", "random", "gaussian"], default="all",
                          help="BPE编码排序模式")
    bpe_group.add_argument("--bpe_encode_rank_k", type=int, help="BPE编码Top-K参数")
    bpe_group.add_argument("--bpe_encode_rank_min", type=int, help="BPE编码随机范围最小值")
    bpe_group.add_argument("--bpe_encode_rank_max", type=int, help="BPE编码随机范围最大值")
    bpe_group.add_argument("--bpe_encode_rank_dist", type=str, help="BPE编码随机分布类型")
    bpe_group.add_argument("--bpe_eval_mode", type=str, 
                          choices=["all", "topk"], help="BPE评估模式")
    bpe_group.add_argument("--bpe_eval_topk", type=int, help="BPE评估Top-K参数")
    
    # 通用训练参数（两阶段均可使用同名参数）
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument("--epochs", type=int, help="训练轮数")
    train_group.add_argument("--batch_size", type=int, help="批次大小")
    train_group.add_argument("--learning_rate", "--lr", type=float, help="学习率")

    # 预训练特有架构参数（仅在预训练脚本中会实际使用）
    arch_group = parser.add_argument_group('BERT架构')
    arch_group.add_argument("--hidden_size", type=int, help="隐藏层大小")
    arch_group.add_argument("--num_layers", type=int, help="层数")
    arch_group.add_argument("--num_heads", type=int, help="注意力头数")

    # 编码器参数 (预训练和微调都需要)
    encoder_group = parser.add_argument_group('编码器配置')
    encoder_group.add_argument("--encoder", type=str, choices=["bert", "gte"], help="编码器类型（bert 或 gte）")
    
    # 任务参数 (仅微调需要)
    if include_finetune:
        task_group = parser.add_argument_group('任务配置')
        task_group.add_argument("--target_property", type=str, help="目标属性名称")
    
    add_json_override_args(parser)
