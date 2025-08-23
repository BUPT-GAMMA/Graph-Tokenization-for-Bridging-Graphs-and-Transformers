#!/usr/bin/env python3
"""
单个方法BERT预训练脚本
====================

支持对指定数据集和序列化方法进行BERT预训练，具备灵活的配置参数覆盖功能。

使用示例:
  # 基本使用
  python run_pretrain.py --dataset qm9test --method feuler
  
  # 使用BPE压缩
  python run_pretrain.py --dataset qm9test --method feuler --bpe
  
  # 自定义训练参数
  python run_pretrain.py --dataset qm9test --method feuler --epochs 10 --batch_size 32 --learning_rate 1e-4
  
  # 自定义模型架构
  python run_pretrain.py --dataset qm9test --method feuler --hidden_size 768 --num_layers 6 --num_heads 12
  
  # 高级配置覆盖
  python run_pretrain.py --dataset qm9test --method feuler --config_override bert.pretraining.warmup_steps=1000 system.device=cuda:1
"""

from __future__ import annotations

import argparse
import sys
import os
import re
import io
from pathlib import Path

# 设置项目路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ProjectConfig  # noqa: E402
from src.data.unified_data_interface import UnifiedDataInterface  # noqa: E402
from src.training.pretrain_api import pretrain as pretrain_api  # noqa: E402
from src.utils.config_override import (  # noqa: E402
    add_all_args,
    apply_args_to_config,
    create_experiment_name,
    print_config_summary,
    show_full_config
)





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

def run_pretraining(config: ProjectConfig) -> dict:
    """
    运行BERT预训练
    
    Args:
        config: 项目配置
        
    Returns:
        训练结果字典
    """
    print("🚀 开始BERT预训练...")
    
    # 创建UDI实例
    udi = UnifiedDataInterface(config, config.dataset.name)
    
    # 加载训练数据
    print(f"📂 加载序列化数据: {config.serialization.method}")
    try:
        train_sequences, val_sequences, test_sequences = udi.get_training_data_flat(config.serialization.method)
        
        print("✅ 数据加载完成:")
        print(f"  训练集: {len(train_sequences)} 个序列")
        print(f"  验证集: {len(val_sequences)} 个序列")
        print(f"  测试集: {len(test_sequences)} 个序列")
        
        token_splits = {
            'train': train_sequences,
            'val': val_sequences,
            'test': test_sequences,
        }
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        raise
    
    # 加载词表
    print("📚 加载词表...")
    try:
        vocab_manager = udi.get_vocab(method=config.serialization.method)
        vocab_info = vocab_manager.get_vocab_info()
        print(f"✅ 词表加载成功: {vocab_info['vocab_size']} 个token")
    except Exception as e:
        print(f"❌ 词表加载失败: {e}")
        print("💡 请确保已运行数据准备流程构建词表")
        raise
    
    # 运行预训练
    print("🎓 开始模型训练...")
    try:
        result = pretrain_api(config, token_splits, vocab_manager, udi, config.serialization.method)
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
    args = parser.parse_args()
    
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
    
    # 运行预训练
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
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
