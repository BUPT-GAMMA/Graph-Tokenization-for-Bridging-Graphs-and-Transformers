from __future__ import annotations
from pathlib import Path
from typing import Dict

import torch

from src.models.universal_model import UniversalModel
from src.models.unified_encoder import create_encoder
from src.training.task_handler import create_task_handler
from src.utils.check import parse_torch_dtype
from src.utils.logger import get_logger

# 创建模块级logger
logger = get_logger(__name__)


def build_task_model(
    config,
    udi,
    method,
    pretrained_dir=None,
    pretrain_exp_name=None,
    force_task_type=None,
    run_i=None,
):
    """
    构建统一任务模型 - 支持预训练和微调的统一入口
    
     流程设计：
    1. 自动判断任务类型（除非强制指定）
    2. 先完整创建模型（encoder + 任务头），此时决定是否reset权重
    3. 如果需要加载预训练，覆盖encoder权重
    4. 如果不需要加载预训练，直接返回（使用创建时的权重）
    
    Args:
        config: 项目配置
        udi: 统一数据接口
        method: 序列化方法
        pretrained_dir: 预训练模型目录（可选）
        pretrain_exp_name: 预训练实验名（可选）
        force_task_type: 强制指定任务类型（如'mlm'用于预训练）

    Returns:
        (model, task_handler) 元组
    """
    logger.info("🏗️ 开始创建模型...")
    logger.info(f"  数据集: {config.dataset.name}")
    logger.info(f"  序列化方法: {method}")
    logger.info(f"  编码器类型: {config.encoder.type}")
    
    # 🆕 内置路径解析逻辑
    task_type = force_task_type if force_task_type is not None else udi.get_dataset_task_type()
    pretrained_path = _resolve_pretrained_path_internal(config, pretrain_exp_name, pretrained_dir, run_i, task_type)
    # 规则：MLM 任务默认不加载任何预训练模型（即使存在），确保预训练严格从随机初始化开始
    if task_type == 'mlm':
        pretrained_path = None


    if pretrained_path:
        logger.info(f"📦 将使用预训练模型: {pretrained_path}")
    elif task_type == 'mlm':
        logger.info("🆕 预训练任务，将创建新模型（不加载本地预训练权重）")
    else:
        logger.warning("⚠️ 微调任务，但未指定预训练模型，将使用随机初始化权重")
    
    # === 1. 准备基础信息 ===
    method = config.serialization.method
    pooling_method = config.bert.architecture.pooling_method
    encoder_type = config.encoder.type
    vocab_manager = udi.get_vocab(method=method)
    
    # 任务类型判断
    logger.info(f"🔧 配置: {task_type}任务, {encoder_type}编码器, {method}序列化")
    
    # === 第1阶段：创建完整模型 ===
    # 创建编码器
    encoder_config = _build_encoder_config(config, encoder_type, task_type)
    from src.models.unified_encoder import create_encoder_from_config
    encoder = create_encoder_from_config(encoder_type, encoder_config, vocab_manager)
    dtype=encoder_config.get('optimization', {}).get('torch_dtype', torch.float32)
    dtype=parse_torch_dtype(dtype)
    
    # 创建任务头处理器（负责计算损失和评估指标）
    task_handler, output_dim = create_task_handler(udi, task_type, vocab_manager.vocab_size)
    assert output_dim is not None, "任务处理器创建失败"

    logger.info(f"🎯 任务: {task_type}, 输出维度: {output_dim}")
    
    # 创建统一模型
    model = UniversalModel(
        encoder=encoder,
        task_type=task_type,
        output_dim=output_dim,
        pooling_method=pooling_method,
        dtype=dtype
    )
    # 第2阶段：权重处理
    if pretrained_path is not None:
        logger.info("🔄 加载预训练权重...")
        _load_and_copy_pretrained_weights(model, pretrained_path)
        logger.info("✅ 模型创建完成 (预训练权重)")
    else:
        weight_state = "重置权重" if config.encoder.reset_weights else "默认初始化"
        logger.info(f"✅ 模型创建完成 ({weight_state})")
    return model, task_handler


def _build_encoder_config(config, encoder_type: str, task_type: str = None) -> Dict:
    """构建编码器配置 - 包含详细日志"""
    
    logger.info(f"🔧 构建编码器配置: {encoder_type}")
    logger.info(f"  任务类型: {task_type or 'unspecified'}")
    cfg={}
    
    if encoder_type == 'bert':
        # BERT编码器配置
        cfg = {
            'hidden_size': config.bert.architecture.hidden_size,
            'num_hidden_layers': config.bert.architecture.num_hidden_layers,
            'num_attention_heads': config.bert.architecture.num_attention_heads,
            'intermediate_size': config.bert.architecture.intermediate_size,
            'hidden_dropout_prob': config.bert.architecture.hidden_dropout_prob,
            'attention_probs_dropout_prob': config.bert.architecture.attention_probs_dropout_prob,
            'max_position_embeddings': config.bert.architecture.max_position_embeddings,
            'max_seq_length': config.bert.architecture.max_seq_length,
            'layer_norm_eps': config.bert.architecture.layer_norm_eps,
            'type_vocab_size': getattr(config.bert.architecture, 'type_vocab_size', 2),
            'initializer_range': getattr(config.bert.architecture, 'initializer_range', 0.02),
            # 统一传递 reset_weights
            'reset_weights': bool(config.encoder.reset_weights),
        }
        model_desc = f"{cfg['hidden_size']}d_{cfg['num_hidden_layers']}l_{cfg['num_attention_heads']}h"
        logger.info(f"🔧 BERT配置: {model_desc}, max_len={cfg['max_position_embeddings']}")
        
    elif 'gte' in encoder_type.lower():
        # GTE编码器配置
        cfg = {
            'hidden_size': 768,  # GTE固定768维
            'max_seq_length': 8096,  # GTE支持长序列
            'optimization': {
                'unpad_inputs': True,
                'use_memory_efficient_attention': True,
                'torch_dtype': 'fp32'  
            }
        }
    if cfg is None:
        logger.error(f"❌ 不支持的编码器类型: {encoder_type}")
        logger.info("📋 支持的编码器类型: bert, Alibaba-NLP/gte-multilingual-base")
        raise ValueError(f"不支持的编码器类型: {encoder_type}")
      
    reset_weights = config.encoder.reset_weights
    cfg['reset_weights'] = reset_weights
    if reset_weights:
        logger.warning(f"🔄 将重新初始化GTE整个模型权重，注意：这会丢弃GTE的预训练权重！任务类型: {task_type}")
    elif 'gte' in encoder_type.lower():
        if task_type == 'mlm':
            logger.info("将基于GTE原始权重预训练MLM任务")
        else:
            logger.info("将基于GTE原始权重微调任务")
    
    logger.info(f"🔧 编码器配置: {cfg['hidden_size']}d, max_len={cfg['max_seq_length']}")
    
    return cfg

def _load_and_copy_pretrained_weights(model, pretrained_path):
    """第2阶段：严谨加载预训练 encoder 权重，覆盖第1阶段创建的权重。

    约定：预训练保存的权重文件包含完整的 UniversalModel.state_dict()，其中 encoder.* 前缀下即为编码器参数。
    这里严格断言：
      - 必须存在 pytorch_model.bin 与 config.bin
      - checkpoint 中必须存在 encoder.* 键
      - encoder 的词嵌入形状与当前模型完全一致（词表大小）
      - 若两侧均为绝对位置嵌入，则位置嵌入长度必须一致
      - encoder_state 的参数键集合需与 model.encoder.state_dict().keys() 完全一致（严格加载）
    """

    pretrain_path = Path(pretrained_path)

    # 基础路径检查
    assert pretrain_path.exists(), f"预训练路径不存在: {pretrained_path}"
    assert (pretrain_path / 'config.bin').exists(), f"缺少预训练配置: {pretrain_path / 'config.bin'}"
    assert (pretrain_path / 'pytorch_model.bin').exists(), f"缺少预训练模型: {pretrain_path / 'pytorch_model.bin'}"

    # 加载 checkpoint（仅CPU）
    checkpoint = torch.load(pretrain_path / 'pytorch_model.bin', map_location='cpu')
    assert isinstance(checkpoint, dict) and len(checkpoint) > 0, "无效的checkpoint：内容为空"

    # 提取 encoder 权重并去掉前缀 'encoder.'
    # raw_encoder_keys_sample = [k for k in checkpoint.keys() if k.startswith('encoder.')][:5]
    # logger.info(f"🔎 预训练原始encoder键(示例): {raw_encoder_keys_sample}")
    # # 只剥离最前面的 'encoder.' 前缀，避免意外移除中间的 '...gte_model.encoder.layer...'
    encoder_state = {}
    for k, v in checkpoint.items():
        if k.startswith('encoder.'):
            new_k = k[len('encoder.'):]
            encoder_state[new_k] = v
    # enc_keys_sample = list(encoder_state.keys())[:5]
    # logger.info(f"🔎 处理后的encoder键(示例): {enc_keys_sample}")
    # assert encoder_state, "预训练模型中未找到 encoder.* 权重键"

    # 确认保存的模型类型与当前一致（bert 或 gte）
    expect_submodule = 'bert' if hasattr(model.encoder, 'bert') else ('gte_model' if hasattr(model.encoder, 'gte_model') else None)
    assert expect_submodule is not None, "当前编码器类型未知（既非BERT也非GTE）"
    has_expected = any(k.startswith(f"{expect_submodule}.") for k in encoder_state.keys())
    assert has_expected, f"encoder 权重不匹配：未检测到 '{expect_submodule}.' 前缀的键"

    # 词表大小完全一致
    # 查找词嵌入键
    we_key = None
    for k in encoder_state.keys():
        if k.endswith('embeddings.word_embeddings.weight'):
            we_key = k
            break
    assert we_key is not None, f"缺少词嵌入层权重（*embeddings.word_embeddings.weight）。示例键: {list(encoder_state.keys())[:5]}"
    pretrained_vocab_size = int(encoder_state[we_key].shape[0])

    # 当前模型词表与位置嵌入信息
    if expect_submodule == 'bert':
        current_vocab_size = int(model.encoder.bert.get_input_embeddings().num_embeddings)
        pos_module = getattr(model.encoder.bert.embeddings, 'position_embeddings', None)
    else:
        current_vocab_size = int(model.encoder.gte_model.get_input_embeddings().num_embeddings)
        pos_module = getattr(model.encoder.gte_model.embeddings, 'position_embeddings', None)
    assert pretrained_vocab_size == current_vocab_size, \
        f"词表大小不一致：预训练={pretrained_vocab_size}, 当前={current_vocab_size}"

    # 若双方均为绝对位置嵌入，则做严格长度校验
    pos_key = None
    for k in encoder_state.keys():
        if k.endswith('position_embeddings.weight'):
            pos_key = k
            break
    if pos_key is not None and pos_module is not None:
        pretrained_max_len = int(encoder_state[pos_key].shape[0])
        current_max_len = int(pos_module.weight.shape[0])
        if pretrained_max_len != current_max_len:
            logger.error("❌ 位置嵌入维度不兼容:")
            logger.error(f"  预训练模型: {pretrained_max_len}")
            logger.error(f"  当前模型: {current_max_len}")
            logger.error("💡 解决方案:")
            logger.error(f"  1. 微调时使用相同的max_seq_length: --max_seq_length {pretrained_max_len}")
            logger.error(f"  2. 或重新训练预训练模型使用max_seq_length={current_max_len}")
            raise ValueError(
                f"位置嵌入维度不匹配：预训练={pretrained_max_len}, 当前={current_max_len}。"
            )

    # 参数键集合严格一致
    current_encoder_keys = set(model.encoder.state_dict().keys())
    pretrained_encoder_keys = set(encoder_state.keys())
    missing = current_encoder_keys - pretrained_encoder_keys
    unexpected = pretrained_encoder_keys - current_encoder_keys
    assert not unexpected, f"预训练包含未预期的编码器参数: {sorted(list(unexpected))[:5]} ... 共{len(unexpected)}项，期望: {sorted(list(current_encoder_keys))[:5]} ... 共{len(current_encoder_keys)}项"
    assert not missing, f"预训练中缺少当前编码器所需的参数: {sorted(list(missing))[:5]} ... 共{len(missing)}项，期望: {sorted(list(pretrained_encoder_keys))[:5]} ... 共{len(pretrained_encoder_keys)}项"

    # 严格加载
    model.encoder.load_state_dict(encoder_state, strict=True)
    logger.info(f"✅ 预训练encoder权重严格加载成功，共 {len(encoder_state)} 个参数, 参数量: {sum(p.numel() for p in model.encoder.parameters())}")
    logger.info("📝 最终权重状态: 预训练权重（已覆盖第1阶段权重）")




def _resolve_pretrained_path_internal(config, pretrain_exp_name, pretrained_dir, run_i=None, task_type=None):
    """内部化的预训练路径解析，避免创建额外文件

    对于预训练任务，不做处理
    对于微调任务，必须要找到预训练的模型，否则报错
    """

    def _validate_model_dir(path):
        """验证模型目录是否包含必需文件"""
        required_files = ['config.bin', 'pytorch_model.bin']
        return path.exists() and all((path / f).exists() for f in required_files)

    # 对于预训练任务，直接返回None，不进行任何检查
    if task_type == 'mlm':
        return None

    # 1. 显式预训练目录优先（最高优先级）
    if pretrained_dir is not None:
        logger.debug(f"检查显式预训练目录: {pretrained_dir}")
        p = Path(pretrained_dir)
        if _validate_model_dir(p):
            logger.info(f"✅ 从指定预训练目录找到模型: {pretrained_dir}")
            return str(p)
        logger.warning(f"⚠️ 指定了预训练目录: {pretrained_dir}, 但未找到有效模型: {p}")
        return None

    # 2. 使用pretrain_exp_name（中等优先级）
    if pretrain_exp_name is not None:
        # 检查run_i指定的预训练模型（get_model_dir会将None转为run_1）
        pretrain_path = config.get_model_dir(exp_name=pretrain_exp_name, run_i=run_i) / 'best'
        logger.info(f"检查预训练模型路径: {pretrain_path}")
        if _validate_model_dir(pretrain_path):
            logger.info(f"✅ 从预训练实验找到模型: {pretrain_path}")
            return str(pretrain_path)

        # 如果run_i指定的模型不存在，且run_i不是1，则尝试run_1
        if run_i != 0:
            run_1_path = config.get_model_dir(exp_name=pretrain_exp_name, run_i=0) / 'best'
            logger.info(f"run_{run_i}不存在，尝试run_1: {run_1_path}")
            if _validate_model_dir(run_1_path):
                logger.info(f"✅ 从预训练实验run_1找到模型: {run_1_path}")
                return str(run_1_path)

        logger.error(f"⚠️ 预训练实验 {pretrain_exp_name} 未找到有效模型")

    # 3. 使用当前experiment_name（最低优先级）
    # 首先尝试指定的run_i（get_model_dir已经处理run_i=None的情况）
    current_path = config.get_model_dir(run_i=run_i) / 'best'
    logger.info(f"检查当前实验模型路径: {current_path}")
    if _validate_model_dir(current_path):
        logger.info(f"✅ 从当前实验找到模型: {current_path}")
        return str(current_path)

    # 如果指定的run不存在且不是run_1，尝试run_1
    if run_i != 0:
        current_run1_path = config.get_model_dir(run_i=0) / 'best'
        logger.info(f"尝试当前实验run_1: {current_run1_path}")
        if _validate_model_dir(current_run1_path):
            logger.info(f"✅ 从当前实验run_1找到模型: {current_run1_path}")
            return str(current_run1_path)

    # 对于微调任务，如果都没有找到预训练模型，则报错
    logger.error("❌ 微调任务必须找到预训练模型，但未找到任何有效的预训练模型")
    logger.error("请检查以下路径:")
    if pretrain_exp_name:
        logger.error(f"  - 预训练实验: {pretrain_exp_name} (检查了run_{run_i or 1})")
    logger.error(f"  - 当前实验: {config.experiment_name} (检查了run_{run_i or 1})")
    raise ValueError("微调任务必须找到预训练模型")

    return None





