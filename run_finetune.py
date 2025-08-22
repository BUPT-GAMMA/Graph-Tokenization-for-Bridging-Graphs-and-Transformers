#!/usr/bin/env python3
"""
单个方法BERT微调脚本
==================

支持对指定数据集和序列化方法进行BERT微调，具备灵活的配置参数覆盖功能。

使用示例:
  # 基本回归任务
  python run_finetune.py --dataset qm9test --method feuler --task regression
  
  # 指定目标属性
  python run_finetune.py --dataset qm9test --method feuler --task regression --target_property homo
  
  # 分类任务
  python run_finetune.py --dataset mnist --method feuler --task classification --num_classes 10
  
  # 自定义微调参数
  python run_finetune.py --dataset qm9test --method feuler --task regression \\
    --finetune_epochs 20 --finetune_batch_size 16 --finetune_learning_rate 2e-5
  
  # 自定义数据处理
  python run_finetune.py --dataset qm9test --method feuler --task regression \\
    --normalization standard --pooling_method mean
  
  # 高级配置覆盖
  python run_finetune.py --dataset qm9test --method feuler --task regression \\
    --config_override bert.finetuning.warmup_ratio=0.1 system.device=cuda:1
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
from src.training.finetune_pipeline import run_finetune  # noqa: E402
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

def infer_task_and_targets(config: ProjectConfig, udi: UnifiedDataInterface, 
                          task_cli: str | None, num_classes_cli: int | None) -> tuple[str, int | None]:
    """
    推断任务类型和目标信息
    
    Args:
        config: 项目配置
        udi: 统一数据接口
        task_cli: 命令行指定的任务类型
        num_classes_cli: 命令行指定的类别数
        
    Returns:
        (task_type, num_classes) 元组
    """
    meta = udi.get_downstream_metadata()
    
    # 推断任务类型
    if task_cli:
        task = task_cli
    else:
        assert 'dataset_task_type' in meta, "数据集元数据中缺少必需字段 'dataset_task_type'"
        task = meta['dataset_task_type']
    
    # 处理回归任务的目标属性
    if task == 'regression' and not config.task.target_property:
        # QM9数据集默认使用homo属性
        if config.dataset.name.lower().startswith('qm9'):
            config.task.target_property = 'homo'
        else:
            # 其他数据集使用默认属性
            if 'default_target_property' in meta and meta['default_target_property']:
                config.task.target_property = meta['default_target_property']
        
        if config.task.target_property:
            print(f"🎯 自动设置回归目标属性: {config.task.target_property}")
    
    # 推断分类类别数和多目标维度
    num_classes = num_classes_cli
    if task in ['classification', 'multi_label_classification', 'multi_target_regression'] and num_classes is None:
        assert 'num_classes' in meta, f"{task}任务需要 num_classes，但数据集元数据中未找到此字段"
        n = meta['num_classes']
        assert isinstance(n, int) and n > 1, f"数据集元数据中 num_classes 应为大于1的整数，实际值: {n}"
        num_classes = n
        if task == 'classification':
            print(f"🏷️ 自动设置分类类别数: {num_classes}")
        elif task == 'multi_label_classification':
            print(f"🏷️ 自动设置多标签分类标签数: {num_classes}")
        elif task == 'multi_target_regression':
            print(f"🏷️ 自动设置多目标回归目标数: {num_classes}")
    
    return task, num_classes


def check_pretrained_model(config: ProjectConfig) -> bool:
    """
    检查预训练模型是否存在
    
    Args:
        config: 项目配置
        
    Returns:
        是否存在可用的预训练模型
    """
    print("🔍 检查预训练模型...")
    
    # 检查标准路径
    model_base = config.get_model_dir()
    best_dir = model_base / "best"
    final_dir = model_base / "final"
    
    def _has_model(d: Path) -> bool:
        return (d / 'config.bin').exists() and (d / 'pytorch_model.bin').exists()
    
    if _has_model(best_dir):
        print(f"✅ 找到预训练模型: {best_dir}")
        return True
    elif _has_model(final_dir):
        print(f"✅ 找到预训练模型: {final_dir}")
        return True
    else:
        # 检查兼容路径
        compat_dir = config.get_bert_model_path("pretrained").parent
        if Path(compat_dir, 'config.bin').exists() and Path(compat_dir, 'pytorch_model.bin').exists():
            print(f"✅ 找到兼容预训练模型: {compat_dir}")
            return True
        
        print("❌ 未找到预训练模型")
        print(f"   已检查路径: {best_dir}, {final_dir}, {compat_dir}")
        return False


def run_finetuning(
    config: ProjectConfig,
    task: str | None = None,
    num_classes: int | None = None,
    aggregation_mode: str = "avg",
    save_name_prefix: str | None = None,
    save_name_suffix: str | None = None,
    pretrained_dir: str | None = None,
    pretrain_exp_name: str | None = None,
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
    if task is not None:
        print(f"📋 任务类型: {task}")
    else:
        print("📋 任务类型: 将从数据集自动推断")
    
    if task == "regression" and config.task.target_property:
        print(f"📋 回归目标: {config.task.target_property}")
    elif task == "classification" and num_classes:
        print(f"📋 分类类别数: {num_classes}")
    
    # 检查预训练模型
    # if not check_pretrained_model(config):
    #     print("\n💡 请先运行预训练:")
    #     print(f"python run_pretrain.py --dataset {config.dataset.name} --method {config.serialization.method}")
    #     assert False, "预训练模型不存在"
    
    # 运行微调
    print("🎓 开始微调...")
    try:
        # 统一架构会自动从UDI推断任务类型和维度，不需要显式传递num_classes等参数
        result = run_finetune(
            config,
            task=task,
            aggregation_mode=aggregation_mode,
            save_name_prefix=save_name_prefix,
            save_name_suffix=save_name_suffix,
            pretrained_dir=pretrained_dir,
            pretrain_exp_name=pretrain_exp_name,
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
  # 基本回归任务（默认BPE all模式）
  python run_finetune.py --dataset qm9test --method feuler --task regression
  
  # 无BPE压缩微调（使用原始序列）
  python run_finetune.py --dataset qm9test --method feuler --task regression --bpe_encode_rank_mode none
  
  # BPE Top-K压缩微调
  python run_finetune.py --dataset qm9test --method feuler --task regression \\
    --bpe_encode_rank_mode topk --bpe_encode_rank_k 1000
  
  # BPE随机压缩微调（训练时随机，评估时确定性）
  python run_finetune.py --dataset qm9test --method feuler --task regression \\
    --bpe_encode_rank_mode random --bpe_eval_mode all
  
  # 指定回归目标属性
  python run_finetune.py --dataset qm9test --method feuler --task regression --target_property homo
  
  # 分类任务（自动推断类别数）
  python run_finetune.py --dataset mnist --method feuler --task classification
  
  # JSON配置覆盖
  python run_finetune.py --dataset qm9test --method feuler --task regression \\
    --config_json '{"bert": {"finetuning": {"epochs": 30, "learning_rate": 1e-5}},
                    "serialization": {"bpe": {"engine": {"encode_rank_mode": "topk", "encode_rank_k": 1000}}}}'
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
    args = parser.parse_args()
    
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
    
    # 创建UDI并推断任务信息
    udi = UnifiedDataInterface(config, config.dataset.name)
    task = args.task
    num_classes = args.num_classes
    task, num_classes = infer_task_and_targets(config, udi, task, num_classes)
    
    # 更新配置中的任务类型
    config.task.type = task
    
    # 验证配置
    try:
        config.validate()
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return 1
    
    # 打印配置摘要
    print_config_summary(config)
    
    # 运行微调
    try:
        result = run_finetuning(
            config,
            task,
            num_classes,
            aggregation_mode=args.aggregation_mode,
            save_name_prefix=args.save_name_prefix,
            save_name_suffix=args.save_name_suffix,
            pretrained_dir=getattr(args, 'pretrained_dir', None),
            pretrain_exp_name=getattr(args, 'pretrain_exp_name', None),
        )
        
        print("\n" + "="*60)
        print("🎉 微调完成!")
        print("="*60)
        
        print(f"📁 最优模型路径: {result['best_dir']}")
        print(f"📁 最终模型路径: {result['final_dir']}")
        print(f"📊 最优验证损失: {result['best_val_loss']:.4f}")
        
        # 显示测试结果
        test_metrics = result['test_metrics']
        print("\n📈 测试集性能:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断训练")
        return 130
    except Exception as e:
        print(f"\n❌ 微调失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
