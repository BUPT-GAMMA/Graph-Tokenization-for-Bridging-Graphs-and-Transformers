from __future__ import annotations

from typing import Dict, Any, Optional, Literal
from pathlib import Path
import pickle

import torch

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.training.tasks import build_regression_loaders, build_classification_loaders
from src.training.model_builder import build_task_model
from src.training.evaluate import evaluate_model
from src.training.finetune_pipeline import load_pretrained_backbone
from src.utils.logger import get_logger


logger = get_logger('tokenizerGraph.training.infer_pipeline')


def _has_model_files(d: Path) -> bool:
    return (d / 'config.bin').exists() and (d / 'pytorch_model.bin').exists()


def resolve_finetuned_model_dir(config: ProjectConfig, *, model_dir: Optional[str] = None,
                                save_name: Optional[str] = None) -> Path:
    """解析微调模型目录，使其与微调保存规则对齐。

    - 若提供 model_dir，则直接校验并返回
    - 否则优先使用标准目录（若提供了 experiment_group/experiment_name）：
        get_model_dir() / [save_name] / (best|final) 或 get_model_dir() / (best|final)
    - 若仍未命中，则在 model 根目录下按 group/dataset/method/save_name 约束递归查找 best/final
    - 优先返回 best；若无则返回 final；同类取最近修改
    """
    if model_dir is not None:
        p = Path(model_dir)
        if not _has_model_files(p):
            raise FileNotFoundError(f"指定的 model_dir 缺少必要文件: {p}")
        return p

    # 标准目录优先（当 experiment_group/experiment_name 存在时）
    try:
        base_dir = config.get_model_dir()
        cand_list: list[Path] = []
        if save_name:
            cand_list += [Path(base_dir, save_name, 'best'), Path(base_dir, save_name, 'final')]
        else:
            cand_list += [Path(base_dir, 'finetune', 'best'), Path(base_dir, 'finetune', 'final')]

        cand_list += [Path(base_dir, 'best'), Path(base_dir, 'final')]
        for c in cand_list:
            if _has_model_files(c):
                return c
    except Exception:
        pass

    base_root = Path(config.model_dir)
    # 约束到实验分组（若提供）
    try:
        if getattr(config, 'experiment_group', None):
            base_root = base_root / str(config.experiment_group)
    except Exception:
        pass

    dataset = str(config.dataset.name)
    method = str(config.serialization.method)
    target_save = str(save_name) if save_name else 'finetune'

    best_cands: list[Path] = []
    final_cands: list[Path] = []

    # 构造候选保存名（兼容 finetune_avg 与 finetuneavg）
    save_aliases: list[str] = [target_save]
    if target_save.startswith('finetune_'):
        save_aliases.append('finetune' + target_save[len('finetune_'):])
    elif target_save.startswith('finetune') and not target_save.startswith('finetune_'):
        # e.g., finetuneavg -> finetune_avg
        suffix = target_save[len('finetune'):]
        if suffix:
            save_aliases.append('finetune_' + suffix)

    # 目标模式：.../<exp_name>/<dataset>/<method>/<save_alias>/(best|final)
    for alias in save_aliases:
        pattern_best = f"**/{dataset}/{method}/{alias}/best"
        pattern_final = f"**/{dataset}/{method}/{alias}/final"
        for b in base_root.rglob(pattern_best):
            if _has_model_files(b):
                best_cands.append(b)
        for f in base_root.rglob(pattern_final):
            if _has_model_files(f):
                final_cands.append(f)

    # 兜底：无 save_name 情况下，允许直接 method 层 best/final
    pattern_best_simple = f"**/{dataset}/{method}/best"
    pattern_final_simple = f"**/{dataset}/{method}/final"
    for b in base_root.rglob(pattern_best_simple):
        if _has_model_files(b):
            best_cands.append(b)
    for f in base_root.rglob(pattern_final_simple):
        if _has_model_files(f):
            final_cands.append(f)

    if best_cands:
        best_cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return best_cands[0]
    if final_cands:
        final_cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return final_cands[0]

    raise FileNotFoundError(
        f"未能在 {base_root} 下找到匹配的数据集/方法/保存名组合: dataset={dataset}, method={method}, save={target_save}"
    )


def run_infer(
    config: ProjectConfig,
    *,
    task: Literal["regression", "classification"],
    num_classes: Optional[int] = None,
    aggregation_mode: Literal["avg", "best", "learned"] = "avg",
    save_name_prefix: Optional[str] = None,
    save_name_suffix: Optional[str] = None,
    model_dir: Optional[str] = None,
    save_name: Optional[str] = None,
) -> Dict[str, Any]:
    dataset_name = config.dataset.name
    method = config.serialization.method

    udi = UnifiedDataInterface(config=config, dataset=dataset_name)

    # 构建任务模型并加载 finetuned 权重
    pretrained = load_pretrained_backbone(config, pretrained_dir=None)
    model = build_task_model(config, task, pretrained=pretrained, num_classes=num_classes)

    finetuned_dir = resolve_finetuned_model_dir(config, model_dir=model_dir, save_name=save_name)
    logger.info(f"加载微调模型权重: {finetuned_dir}")
    state = torch.load(finetuned_dir / 'pytorch_model.bin', map_location='cpu')
    model.load_state_dict(state)

    # DataLoader
    if task == "regression":
        train_dl, val_dl, test_dl, normalizer = build_regression_loaders(
            config, pretrained, udi, method
        )
        # 优先使用微调保存的 normalizer（确保与训练一致）
        try:
            saved_norm = finetuned_dir / 'label_normalizer.pkl'
            if saved_norm.exists():
                with open(saved_norm, 'rb') as f:
                    normalizer = pickle.load(f)
        except Exception:
            pass
    else:
        train_dl, val_dl, test_dl = build_classification_loaders(
            config, pretrained, udi, method, num_classes=num_classes
        )
        normalizer = None

    # 设备
    device = config.system.device if config.system.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 日志（与微调风格一致）
    from torch.utils.tensorboard import SummaryWriter
    logs_dir = config.get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    _log_name = "infer"
    if save_name_prefix:
        _log_name = f"{save_name_prefix}_{_log_name}"
    if save_name_suffix:
        _log_name = f"{_log_name}_{save_name_suffix}"
    (logs_dir / _log_name).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logs_dir / _log_name / "tensorboard")

    wandb_logger = None
    if getattr(config.logging, 'use_wandb', False):
        try:
            from src.utils.logging.wandb_logger import WandbLogger
            wandb_logger = WandbLogger(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity,
                offline=config.logging.wandb_offline,
            )
            setattr(config, "experiment_phase", "infer")
            wandb_logger.init(config)
        except Exception:
            wandb_logger = None

    # 可学习聚合器
    aggregator = None
    aggregator_dir: Optional[Path] = None
    if aggregation_mode == "learned":
        try:
            from src.training.learned_aggregation import train_variant_aggregator
            agg_cfg: Dict[str, Any] = {
                "hidden_dim": 256,
                "dropout": 0.1,
                "epochs": 100,
                "lr": 1e-3,
                "weight_decay": 1e-2,
                "early_stopping_patience": 5,
                "use_pred_as_feat": (task == "regression"),
                "batch_size": 64,
            }
            aggregator_dir = (finetuned_dir.parent / 'aggregator_eval')
            aggregator = train_variant_aggregator(
                model=model,
                train_loader=train_dl,
                val_loader=val_dl,
                device=device,
                task=task,
                label_normalizer=normalizer if task == "regression" else None,
                save_dir=str(aggregator_dir),
                cfg=agg_cfg,
            )
        except Exception as e:
            logger.warning(f"训练聚合器失败，将回退到 avg 聚合: {e}")
            aggregator = None

    # 评估
    metrics = evaluate_model(
        model,
        test_dl,
        device,
        task,
        label_normalizer=normalizer if task == 'regression' else None,
        aggregation_mode=(
            'learned' if aggregator is not None else (
                'avg' if aggregation_mode == 'learned' else aggregation_mode
            )
        ),
        epoch_num=None,
        total_epochs=None,
        log_style=getattr(config.system, 'log_style', 'online'),
        aggregator=aggregator,
    )

    # 记录
    try:
        writer.add_scalar('Infer/Loss', float(metrics.get('val_loss', 0.0)), 0)
        if task == 'regression':
            for key in ['mae', 'mse', 'rmse', 'r2', 'correlation']:
                if key in metrics:
                    writer.add_scalar(f'Infer/{key.upper()}', float(metrics[key]), 0)
        else:
            for key, name in [('accuracy','Accuracy'), ('precision','Precision'), ('recall','Recall'), ('f1','F1')]:
                if key in metrics:
                    writer.add_scalar(f'Infer/{name}', float(metrics[key]), 0)
        writer.flush()
    except Exception:
        pass

    if wandb_logger is not None:
        try:
            payload = {'infer/loss': float(metrics.get('val_loss', 0.0))}
            if task == 'regression':
                for key in ['mae', 'mse', 'rmse', 'r2', 'correlation']:
                    if key in metrics:
                        payload[f'infer/{key}'] = float(metrics[key])
            else:
                for key in ['accuracy','precision','recall','f1']:
                    if key in metrics:
                        payload[f'infer/{key}'] = float(metrics[key])
            wandb_logger.log(payload)
            wandb_logger.finish()
        except Exception:
            pass

    # 结构化保存到 logs/<...>/infer/
    try:
        out_dir = logs_dir / _log_name
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        with (out_dir / 'infer_metrics.json').open('w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        if aggregator_dir is not None:
            (out_dir / 'aggregator_dir.txt').write_text(str(aggregator_dir.resolve()), encoding='utf-8')
    except Exception as e:
        logger.warning(f"推理指标保存失败: {e}")

    return {
        'model_dir': str(finetuned_dir),
        'test_metrics': metrics,
        'aggregator_dir': str(aggregator_dir) if aggregator_dir is not None else None,
    }


