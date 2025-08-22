from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Literal, Optional
import time
import pickle
import torch
import json
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
# 直接使用UDI接口，不再需要common中的包装函数
from src.training.optim import build_from_config
from src.training.loops import train_epoch
from src.training.evaluate import evaluate_model
from src.training.model_builder import build_task_model
from src.training.tasks import build_regression_loaders, build_classification_loaders
from src.utils.logger import get_logger

logger = get_logger('tokenizerGraph.training.finetune_pipeline')
logger.propagate = False


def load_pretrained_backbone(config: ProjectConfig, pretrained_dir: Optional[str] = None, pretrain_exp_name: Optional[str] = None):
    """统一的预训练模型加载接口 - 重构版"""
    
    # 🆕 使用新的配置属性
    encoder_type = config.encoder.type
    logger.info(f"🔧 加载编码器类型: {encoder_type}")
    
    # 🆕 解析预训练模型路径
    pretrained_path = _resolve_pretrained_model_path(config, pretrained_dir, pretrain_exp_name)
    
    if pretrained_path is None:
        logger.warning("⚠️ 未找到预训练模型，将使用随机初始化")
        return None
    
    try:
        # 🆕 尝试加载UniversalModel格式
        from src.models.universal_model import UniversalModel
        from src.models.unified_encoder import create_encoder
        from src.models.model_factory import _build_encoder_config
        
        # 检查是否为UniversalModel格式
        config_path = pretrained_path / 'config.bin'
        if config_path.exists():
            # 加载配置，检查格式
            saved_config = torch.load(config_path, map_location='cpu')
            
            if 'task_type' in saved_config:
                # 新格式：UniversalModel
                logger.info(f"🔄 加载UniversalModel格式: {pretrained_path}")
                
                # 重新创建encoder  
                udi = UnifiedDataInterface(config=config, dataset=config.dataset.name)
                vocab_manager = udi.get_vocab(method=config.serialization.method)
                encoder_config = _build_encoder_config(config, encoder_type)
                encoder = create_encoder(encoder_type, encoder_config, vocab_manager)
                
                # 加载UniversalModel
                model = UniversalModel.load_model(str(pretrained_path), encoder)
                logger.info(f"✅ UniversalModel加载成功: {model.task_type}任务")
                return model
            else:
                # 旧格式，跳过加载
                logger.warning("⚠️ 检测到旧格式预训练模型，跳过加载，使用随机初始化")
                return None
        else:
            logger.warning(f"⚠️ 预训练模型配置文件不存在: {config_path}")
            return None
            
    except Exception as e:
        logger.warning(f"⚠️ 预训练模型加载失败: {e}，使用随机初始化")
        return None


def _resolve_pretrained_model_path(
    config: ProjectConfig, 
    pretrained_dir: Optional[str] = None,
    pretrain_exp_name: Optional[str] = None
) -> Optional[Path]:
    """解析预训练模型路径"""    
    # 1. 显式指定的目录优先
    if pretrained_dir is not None:
        p = Path(pretrained_dir)
        if p.exists() and (p / 'config.bin').exists():
            return p
        logger.warning(f"⚠️ 指定的预训练目录无效: {pretrained_dir}")
        return None
    
    # 2. 使用pretrain_exp_name (如果提供)
    if pretrain_exp_name is not None:
        pretrain_path = config.get_model_dir().parent / pretrain_exp_name / config.dataset.name / config.serialization.method
        for subdir in ['best', 'final']:
            candidate = pretrain_path / subdir
            if candidate.exists() and (candidate / 'config.bin').exists():
                logger.info(f"🔄 从指定预训练实验加载: {pretrain_exp_name}")
                return candidate
    
    # 3. 使用当前experiment_name
    base_dir = config.get_model_dir()
    for subdir in ['best', 'final']:
        candidate = base_dir / subdir
        if candidate.exists() and (candidate / 'config.bin').exists():
            return candidate
    
    # 4. 未找到任何预训练模型
    return None


# 旧的加载函数已重构到新的load_pretrained_backbone中




def run_finetune(
    config: ProjectConfig,
    *,
    task: Optional[Literal["mlm", "regression", "classification", "multi_label_classification", "multi_target_regression"]] = None,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    num_targets: Optional[int] = None,
    aggregation_mode: Literal["avg", "best", "learned"] = "avg",
    save_name_prefix: Optional[str] = None,
    save_name_suffix: Optional[str] = None,
    pretrained_dir: Optional[str] = None,
    pretrain_exp_name: Optional[str] = None,
) -> Dict[str, Any]:
    dataset_name = config.dataset.name
    method = config.serialization.method

    udi = UnifiedDataInterface(config=config, dataset=dataset_name)
    
    # 🆕 自动从UDI推断任务类型（如果未指定）
    if task is None:
        task = udi.get_dataset_task_type()
        logger.info(f"📋 自动推断任务类型: {task} (来源: {dataset_name})")
    else:
        logger.info(f"📋 使用指定任务类型: {task}")
        # 验证指定的任务类型与数据集是否匹配
        auto_task = udi.get_dataset_task_type()
        if task != auto_task:
            logger.warning(f"⚠️  指定任务类型 '{task}' 与数据集自动推断类型 '{auto_task}' 不匹配")

    # 🆕 简化：直接创建微调模型，支持灵活的预训练加载
    model, task_handler = build_task_model(
        config=config,
        udi=udi,
        method=method,
        pretrained_dir=pretrained_dir,
        pretrain_exp_name=pretrain_exp_name
    )
    
    # 🆕 创建数据加载器 - 直接使用统一架构
    if task == "regression":
        train_dl, val_dl, test_dl, normalizer = build_regression_loaders(
            config, udi, method
        )
    elif task == "multi_target_regression":
        # 多目标回归：复用回归数据加载器，但标签格式不同
        train_dl, val_dl, test_dl, normalizer = build_regression_loaders(
            config, udi, method
        )
    elif task in ["classification", "multi_label_classification"]:
        # 单标签和多标签分类：复用分类数据加载器
        train_dl, val_dl, test_dl = build_classification_loaders(
            config, udi, method
        )
        normalizer = None  # 分类任务不需要normalizer
    else:
        raise ValueError(f"不支持的任务类型: {task}")

    # 优化器与调度器
    total_steps = len(train_dl) * config.bert.finetuning.epochs
    optimizer, scheduler = build_from_config(model, config, total_steps=total_steps, stage="finetune")

    # 日志与可选的 W&B
    from torch.utils.tensorboard import SummaryWriter
    logs_dir = config.get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    _log_name = "finetune"
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
            setattr(config, "experiment_phase", "finetune")
            wandb_logger.init(config)
        except Exception:
            wandb_logger = None

    # 训练
    device = config.system.device if config.system.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 早停监控基线：最小化指标用+inf，最大化指标用-inf
    best_val = float('-inf') if task_handler.should_maximize_metric else float('inf')
    patience = config.bert.finetuning.early_stopping_patience
    patience_ctr = 0
    # 为避免覆盖预训练权重，微调阶段保存到独立子目录，可选加前后缀
    _base = config.get_model_dir()
    _save_name = "finetune"
    if save_name_prefix:
        _save_name = f"{save_name_prefix}_{_save_name}"
    if save_name_suffix:
        _save_name = f"{_save_name}_{save_name_suffix}"
    _save_root = _base / _save_name
    _save_root.mkdir(parents=True, exist_ok=True)
    best_dir = _save_root / "best"
    final_dir = _save_root / "final"

    # 训练时长统计与步级日志间隔
    epoch_times = []
    train_start_time = time.time()
    steps_per_epoch = len(train_dl)
    log_interval = steps_per_epoch//10
    best_val_mae = float('inf')  # 仅用于回归任务的最佳MAE记录
    # 追踪最后一轮用于汇总写盘
    last_train_loss: float | None = None
    last_val_metrics: Dict[str, float] | None = None
    last_learning_rate: float | None = None

    for epoch in range(config.bert.finetuning.epochs):
        epoch_start = time.time()

        def _on_step(step_idx: int, batch_loss: float, current_lr: float | None):
            global_step = epoch * steps_per_epoch + step_idx
            try:
                writer.add_scalar('Train/Batch_Loss', float(batch_loss), global_step)
                if current_lr is not None:
                    writer.add_scalar('Train/Learning_Rate_Step', float(current_lr), global_step)
            except Exception:
                pass
            if wandb_logger is not None:
                payload = {
                    'train/batch_loss': float(batch_loss),
                    'global_step': int(global_step),
                    'epoch': int(epoch + 1),
                }
                if current_lr is not None:
                    payload['train/learning_rate_step'] = float(current_lr)
                try:
                    wandb_logger.log(payload, step=global_step)
                except Exception:
                    pass

        train_stats = train_epoch(
            model,
            train_dl,
            optimizer,
            scheduler,
            device,
            max_grad_norm=config.bert.finetuning.max_grad_norm,
            task_handler=task_handler,
            on_step=_on_step,
            log_interval=log_interval,
            epoch_num=epoch + 1,
            total_epochs=config.bert.finetuning.epochs,
            log_style=getattr(config.system, 'log_style', 'online'),
        )
        # 训练期验证：无论传入模式如何，均计算三种聚合模式并分别记录；
        # learned 在验证阶段若无聚合器则回退为 avg（仅用于参考日志）。
        _val_agg_mode = aggregation_mode if aggregation_mode != "learned" else "avg"
        val_metrics_by_mode: Dict[str, Dict[str, float]] = {}
        for _mode in ("avg", "best"):
            try:
                    _metrics = evaluate_model(
                        model,
                        val_dl,
                        device,
                        task,
                        task_handler=task_handler,
                        label_normalizer=normalizer if task_handler.is_regression_task() else None,
                        aggregation_mode=_mode,
                        epoch_num=epoch + 1,
                        total_epochs=config.bert.finetuning.epochs,
                        log_style=getattr(config.system, 'log_style', 'online'),
                    )
            except Exception:
                # learned 回退为 avg
                _metrics = evaluate_model(
                    model,
                    val_dl,
                    device,
                    task,
                    task_handler=task_handler,
                    label_normalizer=normalizer if task_handler.is_regression_task() else None,
                    aggregation_mode="avg",
                    epoch_num=epoch + 1,
                    total_epochs=config.bert.finetuning.epochs,
                    log_style=getattr(config.system, 'log_style', 'online'),
                )
            val_metrics_by_mode[_mode] = _metrics

        # 主验证指标用于早停逻辑，仍按传入 aggregation_mode
        val_metrics = val_metrics_by_mode['avg']
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # 回归任务：计算训练集上的MAE等指标（用于日志记录）
        train_metrics_eval = None
        if task == "regression":
            # train_metrics_eval = evaluate_model(
            #     model,
            #     train_dl,
            #     device,
            #     task,
            #     label_normalizer=normalizer,
            #     aggregation_mode=aggregation_mode,
            #     epoch_num=epoch + 1,
            #     total_epochs=config.bert.finetuning.epochs,
            #     log_style=getattr(config.system, 'log_style', 'online'),
            # )
            train_metrics_eval = None

        # 记录每个 epoch 的关键日志
        try:

            assert isinstance(train_stats, dict) and "loss" in train_stats, "训练统计数据格式错误，期望包含 'loss' 字段的字典"
            train_loss = train_stats["loss"]
            logger.info(
                f"📈 Finetune Epoch {epoch + 1}: train_loss={train_loss:.4f} | "
                + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )

            # 通用指标
            writer.add_scalar('Loss/Train', float(train_loss), epoch + 1)
            writer.add_scalar('Loss/Validation', float(val_metrics['val_loss']), epoch + 1)
            writer.add_scalar('Train/Epoch_Time', float(epoch_time), epoch + 1)
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            writer.add_scalar('Train/Learning_Rate', float(current_lr), epoch + 1)
            # 记录本轮快照
            last_train_loss = float(train_loss)
            try:
                last_val_metrics = {k: float(v) for k, v in val_metrics.items()}
            except Exception:
                last_val_metrics = None
            try:
                last_learning_rate = float(current_lr)
            except Exception:
                last_learning_rate = None

            # 任务特定指标（仅两个分支：回归/分类）
            if task == "regression":
                if train_metrics_eval is not None:
                    writer.add_scalar('Regression/Train_MAE', float(train_metrics_eval['mae']), epoch + 1)
                # 追加：分别记录两种聚合模式
                for _mode, _m in val_metrics_by_mode.items():
                    base = f"Val/{_mode}"
                    # writer.add_scalar(f'{base}/Loss', float(_m['val_loss']), epoch + 1)
                    writer.add_scalar(f'{base}/MAE', float(_m['mae']), epoch + 1)
                    writer.add_scalar(f'{base}/MSE', float(_m['mse']), epoch + 1)
                    # writer.add_scalar(f'{base}/RMSE', float(_m['rmse']), epoch + 1)
                    # writer.add_scalar(f'{base}/R2', float(_m['r2']), epoch + 1)
                # 兼容原标签
                # writer.add_scalar('Regression/Val_MAE', float(val_metrics['mae']), epoch + 1)
                # writer.add_scalar('Regression/Val_MSE', float(val_metrics['mse']), epoch + 1)
                # writer.add_scalar('Regression/Val_RMSE', float(val_metrics['rmse']), epoch + 1)
                # writer.add_scalar('Regression/Val_R2', float(val_metrics['r2']), epoch + 1)
                # writer.add_scalar('Regression/Val_Correlation', float(val_metrics['correlation']), epoch + 1)
            else:  # classification
                # 追加：分别记录两种聚合模式
                for _mode, _m in val_metrics_by_mode.items():
                    base = f"Val/{_mode}"
                    # writer.add_scalar(f'{base}/Loss', float(_m['val_loss']), epoch + 1)
                    writer.add_scalar(f'{base}/Accuracy', float(_m['accuracy']), epoch + 1)
                    # writer.add_scalar(f'{base}/Precision', float(_m['precision']), epoch + 1)
                    # writer.add_scalar(f'{base}/Recall', float(_m['recall']), epoch + 1)
                    writer.add_scalar(f'{base}/F1', float(_m['f1']), epoch + 1)
                # 兼容原标签
                # writer.add_scalar('Classification/Val_Accuracy', float(val_metrics['accuracy']), epoch + 1)
                # writer.add_scalar('Classification/Val_Precision', float(val_metrics['precision']), epoch + 1)
                # writer.add_scalar('Classification/Val_Recall', float(val_metrics['recall']), epoch + 1)
                # writer.add_scalar('Classification/Val_F1', float(val_metrics['f1']), epoch + 1)

            # W&B：epoch级（train_epoch/* 与 val/*，epoch 轴）
            if wandb_logger is not None:
                payload = {
                    "train_epoch/loss": float(train_loss),
                    "train_epoch/learning_rate": float(current_lr),
                    "train_epoch/epoch_time": float(epoch_time),
                    "val/loss": float(val_metrics['val_loss']),
                    "epoch": int(epoch + 1),
                }
                if task == "regression":
                    if train_metrics_eval is not None:
                        payload['train_epoch/mae'] = float(train_metrics_eval['mae'])
                    payload.update({
                        'val/mae': float(val_metrics['mae']),
                        'val/mse': float(val_metrics['mse']),
                        'val/rmse': float(val_metrics['rmse']),
                        'val/r2': float(val_metrics['r2']),
                        # 'val/correlation': float(val_metrics['correlation']),
                    })
                else:
                    payload.update({
                        'val/accuracy': float(val_metrics['accuracy']),
                        'val/precision': float(val_metrics['precision']),
                        'val/recall': float(val_metrics['recall']),
                        'val/f1': float(val_metrics['f1']),
                    })
                _epoch_end_global_step = int((epoch + 1) * steps_per_epoch)
                wandb_logger.log(payload, step=_epoch_end_global_step)

        except Exception:
            pass
        # 使用TaskHandler确定主要指标和优化方向
        key = task_handler.primary_metric
        if task_handler.should_maximize_metric:
            flag = val_metrics[key] > best_val
        else:
            flag = val_metrics[key] < best_val
        if flag:
            best_val = val_metrics[key]
            patience_ctr = 0
            logger.info(f"🎯 新的最优模型! {key}: {val_metrics[key]:.4f}")
            model.save_model(str(best_dir))
            if task_handler.is_regression_task():
                with open(best_dir / "label_normalizer.pkl", "wb") as f:
                    pickle.dump(normalizer, f)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

        # 跟踪最佳验证MAE（仅回归任务）
        if task == "regression":
            if float(val_metrics['mae']) < best_val_mae:
                best_val_mae = float(val_metrics['mae'])

    # 最终保存与测试
    model.save_model(str(final_dir))
    if task == "regression":
        with open(final_dir / "label_normalizer.pkl", "wb") as f:
            pickle.dump(normalizer, f)

    # 可学习聚合：在测试前尝试训练聚合器（无论传入模式为何，都为 learned 评估做准备）
    aggregator = None
    try:
        from src.training.learned_aggregation import train_variant_aggregator
        agg_cfg: Dict[str, Any] = {
            "hidden_dim": 256,
            "dropout": 0.1,
            "epochs": 10,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "early_stopping_patience": 5,
            "use_pred_as_feat": task_handler.is_regression_task(),
            "batch_size": 64,
        }
        aggregator = train_variant_aggregator(
            model=model,
            train_loader=train_dl,
            val_loader=val_dl,
            device=device,
            task=task,
            label_normalizer=normalizer if task_handler.is_regression_task() else None,
            save_dir=str(final_dir / "aggregator"),
            cfg=agg_cfg,
        )
    except Exception as e:
        logger.warning(f"训练聚合器失败，将回退到 avg 聚合: {e}")
        aggregator = None

    # 测试集：三种聚合模式均评估并记录
    test_metrics_by_mode: Dict[str, Dict[str, float]] = {}
    for _mode in ("avg", "best", "learned"):
        try:
            _metrics = evaluate_model(
                model,
                test_dl,
                device,
                task,
                task_handler=task_handler,
                label_normalizer=normalizer if task_handler.is_regression_task() else None,
                aggregation_mode=_mode if not (_mode == "learned" and aggregator is None) else "avg",
                epoch_num=None,
                total_epochs=None,
                log_style=getattr(config.system, 'log_style', 'online'),
                aggregator=aggregator if _mode == "learned" else None,
            )
        except Exception:
            _metrics = evaluate_model(
                model,
                test_dl,
                device,
                task,
                task_handler=task_handler,
                label_normalizer=normalizer if task_handler.is_regression_task() else None,
                aggregation_mode="avg",
                epoch_num=None,
                total_epochs=None,
                log_style=getattr(config.system, 'log_style', 'online'),
            )
        test_metrics_by_mode[_mode] = _metrics

    # 主测试指标沿用传入 aggregation_mode 语义（learned 无聚合器则回退为 avg）
    _main_test_mode = ("learned" if aggregation_mode == "learned" and aggregator is not None else ("avg" if aggregation_mode == "learned" else aggregation_mode))
    test_metrics = test_metrics_by_mode.get(_main_test_mode, test_metrics_by_mode.get("avg"))

    # 记录测试指标（仅两种分支：回归/分类）
    try:
        # 追加：分别记录三种聚合模式到不同路径 Test/{mode}/...
        for _mode, _m in test_metrics_by_mode.items():
            base = f"Test/{_mode}"
            # writer.add_scalar(f'{base}/Loss', float(_m['val_loss']), 0)
            if task == "regression":
                writer.add_scalar(f'{base}/MAE', float(_m['mae']), 0)
                writer.add_scalar(f'{base}/MSE', float(_m['mse']), 0)
                # writer.add_scalar(f'{base}/RMSE', float(_m['rmse']), 0)
                writer.add_scalar(f'{base}/R2', float(_m['r2']), 0)
                # writer.add_scalar(f'{base}/Correlation', float(_m['correlation']), 0)
            else:
                writer.add_scalar(f'{base}/Accuracy', float(_m['accuracy']), 0)
                writer.add_scalar(f'{base}/Precision', float(_m['precision']), 0)
                writer.add_scalar(f'{base}/Recall', float(_m['recall']), 0)
                writer.add_scalar(f'{base}/F1', float(_m['f1']), 0)
                writer.add_scalar(f'{base}/ROC_AUC', float(_m.get('roc_auc', 0.0)), 0)
                writer.add_scalar(f'{base}/AP', float(_m.get('ap', 0.0)), 0)

        # 兼容原有单一路径写法
        # writer.add_scalar('Test/Loss', float(test_metrics['val_loss']), 0)
        # if task == "regression":
        #     writer.add_scalar('Test/Regression_MAE', float(test_metrics['mae']), 0)
        #     writer.add_scalar('Test/Regression_MSE', float(test_metrics['mse']), 0)
        #     writer.add_scalar('Test/Regression_RMSE', float(test_metrics['rmse']), 0)
        #     writer.add_scalar('Test/Regression_R2', float(test_metrics['r2']), 0)
        #     # writer.add_scalar('Test/Regression_Correlation', float(test_metrics['correlation']), 0)
        # else:
        #     writer.add_scalar('Test/Classification_Accuracy', float(test_metrics['accuracy']), 0)
        #     writer.add_scalar('Test/Classification_Precision', float(test_metrics['precision']), 0)
        #     writer.add_scalar('Test/Classification_Recall', float(test_metrics['recall']), 0)
        #     writer.add_scalar('Test/Classification_F1', float(test_metrics['f1']), 0)
        #     writer.add_scalar('Test/Classification_ROC_AUC', float(test_metrics.get('roc_auc', 0.0)), 0)
        #     writer.add_scalar('Test/Classification_AP', float(test_metrics.get('ap', 0.0)), 0)

        if wandb_logger is not None:
            if task == "regression":
                wb_payload = {
                    'test/loss': float(test_metrics['val_loss']),
                    'test/mae': float(test_metrics['mae']),
                    'test/mse': float(test_metrics['mse']),
                    'test/rmse': float(test_metrics['rmse']),
                    'test/r2': float(test_metrics['r2']),
                    # 'test/correlation': float(test_metrics['correlation']),
                }
            else:
                wb_payload = {
                    'test/loss': float(test_metrics['val_loss']),
                    'test/accuracy': float(test_metrics['accuracy']),
                    'test/precision': float(test_metrics['precision']),
                    'test/recall': float(test_metrics['recall']),
                    'test/f1': float(test_metrics['f1']),
                }
            wandb_logger.log(wb_payload)
    except Exception:
        pass

    # 最终汇总与关闭日志器（含写盘JSON）
    try:
        total_train_time = time.time() - train_start_time
        avg_epoch_time = (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0
        writer.add_scalar('Final/Avg_Epoch_Time', float(avg_epoch_time), 0)
        writer.add_scalar('Final/Total_Train_Time', float(total_train_time), 0)
        writer.add_scalar('Final/Best_Val_Loss', float(best_val), 0)
        if task == "regression" and best_val_mae != float('inf'):
            writer.add_scalar('Final/Best_Val_MAE', float(best_val_mae), 0)
        # 组装最终指标并写入日志目录，包含所有聚合模式的结果
        final_json = {
            'dataset': str(dataset_name),
            'method': str(method),
            'task': str(task),
            'epochs': int(config.bert.finetuning.epochs),
            'steps_per_epoch': int(steps_per_epoch),
            'best_dir': str(best_dir),
            'final_dir': str(final_dir),
            'train': {
                'last_loss': float(last_train_loss) if last_train_loss is not None else None,
                'learning_rate_last': float(last_learning_rate) if last_learning_rate is not None else None,
            },
            'val': {
                # 主验证结果（向后兼容）
                **({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (last_val_metrics or {}).items()}),
                'best_val_mae': float(best_val_mae) if task == 'regression' and best_val_mae != float('inf') else None,
                # 按聚合模式分别记录最后一轮验证结果
                'by_aggregation': {
                    mode: {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in mode_metrics.items()} 
                    for mode, mode_metrics in val_metrics_by_mode.items()
                }
            },
            'test': {
                # 主测试结果（向后兼容）
                **({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (test_metrics or {}).items()}),
                # 按聚合模式分别记录测试结果
                'by_aggregation': {
                    mode: {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in mode_metrics.items()}
                    for mode, mode_metrics in test_metrics_by_mode.items()
                }
            },
            'time': {
                'total_train_time_sec': float(total_train_time),
                'avg_epoch_time_sec': float(avg_epoch_time),
            },
            'aggregation_mode_used': str(aggregation_mode),
            'aggregator_trained': aggregator is not None,
        }
        out_json_path = logs_dir / _log_name / 'finetune_metrics.json'
        try:
            (logs_dir / _log_name).mkdir(parents=True, exist_ok=True)
            with open(out_json_path, 'w') as fjson:
                json.dump(final_json, fjson, ensure_ascii=False, indent=2)
            logger.info(f"📝 已写入最终指标: {out_json_path}")
        except Exception as e:
            logger.warning(f"最终指标写盘失败: {e}")
        if wandb_logger is not None:
            final_payload = {
                'final/avg_epoch_time': float(avg_epoch_time),
                'final/total_train_time': float(total_train_time),
                'final/best_val_loss': float(best_val),
            }
            if task == "regression" and best_val_mae != float('inf'):
                final_payload['final/best_val_mae'] = float(best_val_mae)
            _final_step = int(config.bert.finetuning.epochs * steps_per_epoch) + 1
            wandb_logger.log(final_payload, step=_final_step)
        writer.close()
        if wandb_logger is not None:
            wandb_logger.finish()
    except Exception as e:
        logger.error(f"最终指标写盘失败: {e}")
        pass

    return {
        'best_val_loss': best_val,
        'test_metrics': test_metrics,
        'best_dir': str(best_dir),
        'final_dir': str(final_dir),
        'aggregator_dir': str(final_dir / "aggregator") if aggregator is not None else None,
    }


