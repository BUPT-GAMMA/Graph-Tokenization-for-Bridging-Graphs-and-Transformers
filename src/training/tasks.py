from __future__ import annotations

from typing import Tuple, Any
import torch

from src.models.bert.data import (
    NormalizedRegressionDataset,
    create_transforms_from_config,
    ClassificationDataset,
    LabelNormalizer,
)
from src.data.unified_data_interface import UnifiedDataInterface



def _effective_max_len(seqs, max_pos: int, config=None) -> int:
    if not seqs:
        return max_pos
    try:
        # 复用与预训练一致的策略（max 或 sigma_k）
        from src.models.bert.data import _candidate_len_from_policy
        cand = _candidate_len_from_policy([len(s) for s in seqs], config)
        cand_plus2 = int(cand) + 2
    except Exception:
        cand_plus2 = max(len(s) for s in seqs) + 2
    return min(int(cand_plus2), int(max_pos))


def build_regression_datasets(
    config,
    udi,
    method,
    train_sequences, val_sequences, test_sequences,
    train_labels, val_labels, test_labels,
    train_gids, val_gids, test_gids,
) -> Tuple[Any, Any, Any, LabelNormalizer]:
    normalizer = LabelNormalizer(method=config.task.normalization)
    normalizer.fit(train_labels)

    # 🆕 直接从配置和UDI获取所需信息
    max_pos = int(config.bert.architecture.max_position_embeddings)
    vocab_manager = udi.get_vocab(method=method)
    
    train_eff = _effective_max_len(train_sequences, max_pos, config)
    val_eff = _effective_max_len(val_sequences, max_pos, config)
    test_eff = _effective_max_len(test_sequences, max_pos, config)

    # 仅训练集启用增强；验证/测试使用NoOp
    train_transforms = create_transforms_from_config(config, vocab_manager.get_valid_tokens(), "regression", vocab_manager)
    from src.models.bert.data import NoOpTransform
    eval_transforms = NoOpTransform()
    
    train_ds = NormalizedRegressionDataset(
        train_sequences, train_labels, vocab_manager, train_transforms, train_eff,
        graph_ids=train_gids
    )
    val_ds = NormalizedRegressionDataset(
        val_sequences, val_labels, vocab_manager, eval_transforms, val_eff,
        graph_ids=val_gids
    )
    test_ds = NormalizedRegressionDataset(
        test_sequences, test_labels, vocab_manager, eval_transforms, test_eff,
        graph_ids=test_gids
    )

    train_ds.normalizer = normalizer
    train_ds.apply_normalization()
    val_ds.normalizer = normalizer
    val_ds.apply_normalization()
    test_ds.normalizer = normalizer
    test_ds.apply_normalization()
    return train_ds, val_ds, test_ds, normalizer


def build_classification_datasets(
    config,
    udi,
    method,
    train_sequences, val_sequences, test_sequences,
    train_labels, val_labels, test_labels,
    train_gids, val_gids, test_gids,
    *,
    num_classes: int,
):
    # 🆕 直接从配置和UDI获取所需信息
    max_pos = int(config.bert.architecture.max_position_embeddings)
    vocab_manager = udi.get_vocab(method=method)
    
    train_eff = _effective_max_len(train_sequences, max_pos, config)
    val_eff = _effective_max_len(val_sequences, max_pos, config)
    test_eff = _effective_max_len(test_sequences, max_pos, config)

    # 仅训练集启用增强；验证/测试使用NoOp
    train_transforms = create_transforms_from_config(config, vocab_manager.get_valid_tokens(), "classification", vocab_manager)
    from src.models.bert.data import NoOpTransform
    eval_transforms = NoOpTransform()
    
    train_ds = ClassificationDataset(train_sequences, train_labels, vocab_manager, train_transforms, train_eff, train_gids)
    val_ds = ClassificationDataset(val_sequences, val_labels, vocab_manager, eval_transforms, val_eff, val_gids)
    test_ds = ClassificationDataset(test_sequences, test_labels, vocab_manager, eval_transforms, test_eff, test_gids)
    return train_ds, val_ds, test_ds


def build_regression_loaders(
    config,
    udi,
    method,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, LabelNormalizer]:
  # 解析目标属性
    target_property = udi._resolve_target_property(config.task.target_property)
    config.task.target_property = target_property # 更新配置
    
    # 获取带graph_id的原始数据
    (
        (train_seqs_with_id, train_props),
        (val_seqs_with_id, val_props), 
        (test_seqs_with_id, test_props),
    ) = udi.get_training_data(method)
    
    # 准备扁平化数据
    train_sequences = [seq for _, seq in train_seqs_with_id]
    train_labels = udi._extract_labels_from_properties(train_props, target_property)
    train_gids = [gid for gid, _ in train_seqs_with_id]
    
    val_sequences = [seq for _, seq in val_seqs_with_id]
    val_labels = udi._extract_labels_from_properties(val_props, target_property)
    val_gids = [gid for gid, _ in val_seqs_with_id]
    
    test_sequences = [seq for _, seq in test_seqs_with_id]
    test_labels = udi._extract_labels_from_properties(test_props, target_property)
    test_gids = [gid for gid, _ in test_seqs_with_id]
    
    train_ds, val_ds, test_ds, normalizer = build_regression_datasets(
        config, udi, method,
        train_sequences, val_sequences, test_sequences,
        train_labels, val_labels, test_labels,
        train_gids, val_gids, test_gids,
    )
    
    # 创建BPE worker初始化函数（统一创建，mode控制行为）
    bpe_worker_init_fn = None
    if udi is not None and method is not None:
        try:
            from src.data.bpe_transform import create_bpe_worker_init_fn_from_udi
            bpe_worker_init_fn = create_bpe_worker_init_fn_from_udi(udi, config, method, split="train")
        except Exception as e:
            # 如果BPE创建失败，回退到无BPE模式（但不静默忽略错误）
            import logging
            logger_instance = logging.getLogger("tokenizerGraph.data")
            logger_instance.warning(f"BPE创建失败，回退到无BPE模式: {e}")
    
    _num_workers = int(config.system.num_workers)
    _persistent_workers = bool(config.system.persistent_workers and _num_workers > 0)
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=config.bert.finetuning.batch_size, 
        shuffle=True, 
        pin_memory=True,
        worker_init_fn=bpe_worker_init_fn,
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=config.bert.finetuning.batch_size, 
        shuffle=False, 
        pin_memory=True,
        worker_init_fn=create_bpe_worker_init_fn_from_udi(udi, config, method, split="val"),
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=config.bert.finetuning.batch_size, 
        shuffle=False, 
        pin_memory=True,
        worker_init_fn=create_bpe_worker_init_fn_from_udi(udi, config, method, split="test"),
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
    )
    return train_dl, val_dl, test_dl, normalizer


def build_classification_loaders(
    config,
    udi: UnifiedDataInterface,
    method,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
  # 获取带graph_id的原始数据
    (
        (train_seqs_with_id, train_props),
        (val_seqs_with_id, val_props),
        (test_seqs_with_id, test_props),
    ) = udi.get_training_data(method)
    target_property = udi._resolve_target_property(config.task.target_property)
    config.task.target_property = target_property # 更新配置
    num_classes = udi.get_num_classes()
    
    # 准备扁平化数据
    train_sequences = [seq for _, seq in train_seqs_with_id]
    train_labels = udi._extract_labels_from_properties(train_props, target_property)
    train_gids = [gid for gid, _ in train_seqs_with_id]
    
    val_sequences = [seq for _, seq in val_seqs_with_id]
    val_labels = udi._extract_labels_from_properties(val_props, target_property)
    val_gids = [gid for gid, _ in val_seqs_with_id]
    
    test_sequences = [seq for _, seq in test_seqs_with_id]
    test_labels = udi._extract_labels_from_properties(test_props, target_property)
    test_gids = [gid for gid, _ in test_seqs_with_id]
    assert num_classes > 1, "分类任务需要至少2个类别"
    train_ds, val_ds, test_ds = build_classification_datasets(
        config, udi, method,
        train_sequences, val_sequences, test_sequences,
        train_labels, val_labels, test_labels,
        train_gids, val_gids, test_gids,
        num_classes=num_classes,
    )
    
    # 创建BPE worker初始化函数（统一创建，mode控制行为）
    bpe_worker_init_fn = None
    if udi is not None and method is not None:
        try:
            from src.data.bpe_transform import create_bpe_worker_init_fn_from_udi
            bpe_worker_init_fn = create_bpe_worker_init_fn_from_udi(udi, config, method, split="train")
        except Exception as e:
            # 如果BPE创建失败，回退到无BPE模式（但不静默忽略错误）
            import logging
            logger_instance = logging.getLogger("tokenizerGraph.data")
            logger_instance.warning(f"BPE创建失败，回退到无BPE模式: {e}")
    
    _num_workers = int(config.system.num_workers)
    _persistent_workers = bool(config.system.persistent_workers and _num_workers > 0)
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=config.bert.finetuning.batch_size, 
        shuffle=True, 
        pin_memory=True,
        worker_init_fn=bpe_worker_init_fn,
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=config.bert.finetuning.batch_size, 
        shuffle=False, 
        pin_memory=True,
        worker_init_fn=create_bpe_worker_init_fn_from_udi(udi, config, method, split="val"),
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=config.bert.finetuning.batch_size, 
        shuffle=False, 
        pin_memory=True,
        worker_init_fn=create_bpe_worker_init_fn_from_udi(udi, config, method, split="test"),
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
    )
    return train_dl, val_dl, test_dl


