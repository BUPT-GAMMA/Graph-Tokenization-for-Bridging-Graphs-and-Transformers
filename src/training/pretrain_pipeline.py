"""
BERT预训练Pipeline (简化重构版)
===============================

基于src/training组件重新组织的预训练流程，不依赖old_pipeline。

核心组件：
- 使用 UDI 正确接口获取数据和词表
- 使用 src/training/loops 的训练循环
- 使用 src/training/optim 的优化器构建
- 使用 src/training/callbacks 的回调处理
"""

from __future__ import annotations

import time
from typing import Dict, List, Any, Optional
# from pathlib import Path  # unused
import json

# 🆕 Optuna剪枝支持
try:
    import optuna
except ImportError:
    optuna = None  # 可选依赖，不强制要求

import torch
from torch.utils.tensorboard import SummaryWriter

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.data.bpe_transform import create_bpe_worker_init_fn_from_udi
from src.models.bert.data import compute_effective_max_length
from src.training.model_builder import build_task_model
from src.training.loops import train_epoch, evaluate_epoch
from src.training.optim import build_from_config
from src.utils.logger import get_logger
from src.utils.info_display import (
    display_startup_config, display_data_info, display_model_info, 
    display_training_setup, display_stage_separator, display_performance_summary
)

logger = get_logger('tokenizerGraph.training.pretrain_pipeline')
logger.propagate = False



def train_bert_mlm(
    config: ProjectConfig,
    run_i: Optional[int] = None,
) -> Dict[str, Any]:
    """
    BERT MLM预训练主函数
    
    Args:
        config: 项目配置
        token_sequences: 包含"train"/"val"/"test"键的序列字典
        udi: 统一数据接口
        method: 序列化方法名，用于BPE Transform（如果需要）
        
    Returns:
        包含训练结果的字典
    """
    # 显示启动配置
    display_startup_config(logger, config, config.dataset.name, config.serialization.method, "预训练")
    udi = UnifiedDataInterface(config, config.dataset.name)
    method = config.serialization.method
    
    display_stage_separator(logger, "词表与最大长度", "验证输入数据")

    # 读取预训练图级采样配置
    pretrain_cfg = getattr(config.bert, 'pretraining')
    use_graph_level_sampling = bool(getattr(pretrain_cfg, 'use_graph_level_sampling', False))
    apply_graph_level_to_val = bool(getattr(pretrain_cfg, 'apply_graph_level_to_val', False))
    variant_selection = str(getattr(pretrain_cfg, 'graph_variant_selection', 'random'))

    # 根据是否启用图级采样选择数据接口
    if use_graph_level_sampling or apply_graph_level_to_val:
        (train, train_gids), (val, val_gids), (test, test_gids) = udi.get_training_data_flat_with_ids(method=method)
    else:
        train, val, test = udi.get_training_data_flat(method=method)
        train_gids = val_gids = test_gids = None

    # 使用提供的词表
    vocab_manager = udi.get_vocab(method=method)
    vocab_info = vocab_manager.get_vocab_info()
    # 同步配置中的词表大小
    config.bert.architecture.vocab_size = int(vocab_info['vocab_size'])

    # 计算有效最大长度
    all_sequences = train + val + test
    effective_max_length = compute_effective_max_length(all_sequences, config)
    
    # 显示数据信息
    display_data_info(
        logger,
        len(train), len(val), len(test),
        vocab_info['vocab_size'], effective_max_length
    )
    
    display_stage_separator(logger, "模型创建", "创建MLM预训练模型")
    
    # 确保配置中的位置嵌入大小与有效长度一致
    config.bert.architecture.max_position_embeddings = int(effective_max_length)
    
    # 🆕 使用统一模型创建接口（与微调完全一致）
    mlm_model, task_handler = build_task_model(
        config=config,
        udi=udi,
        method=method,
        force_task_type='mlm'
    )
    
    # 显示模型信息
    display_model_info(logger, mlm_model, 'mlm', config.encoder.type)
    
    # 注意：为避免 CUDA 初始化后再 fork 导致的 DataLoader 退出卡住问题，
    # 先构建 DataLoader（spawn/fork 子进程）再将模型迁移到 GPU。
    
    # 创建BPE Transform worker初始化函数（统一创建，mode控制行为）
    try:
        bpe_worker_init_fn = create_bpe_worker_init_fn_from_udi(udi, config, method, split="train")
        bpe_mode = config.serialization.bpe.engine.encode_rank_mode
        logger.info(f"🔧 BPE模式: {bpe_mode}")
    except Exception as e:
        logger.error(f"❌ BPE Transform创建失败: {e}")
        raise
    
    # 创建数据加载器
    display_stage_separator(logger, "数据加载器", "创建数据加载器与BPE Transform")
    
    # 创建带BPE Transform的DataLoader
    from src.models.bert.data import MLMDataset, create_transforms_from_config, NoOpTransform
    from torch.utils.data import DataLoader
    
    # 获取有效的token列表，用于数据增强
    valid_tokens = vocab_manager.get_valid_tokens()
    # 仅训练集启用数据增强；验证/测试使用NoOp
    train_transforms = create_transforms_from_config(config, valid_tokens, "mlm", vocab_manager,logger)
    eval_transforms = NoOpTransform()
    
    # 训练集DataLoader
    train_dataset = MLMDataset(
        train, vocab_manager, train_transforms, effective_max_length, config.bert.pretraining.mask_prob,
        graph_ids=train_gids, group_by_graph=use_graph_level_sampling, variant_selection=variant_selection
    )
    _num_workers = int(config.system.num_workers)
    _persistent_workers = bool(config.system.persistent_workers and _num_workers > 0)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.bert.pretraining.batch_size,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=bpe_worker_init_fn,
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
    )

    # 验证集DataLoader
    val_dataset = MLMDataset(
        val, vocab_manager, eval_transforms, effective_max_length, config.bert.pretraining.mask_prob,
        graph_ids=val_gids, group_by_graph=apply_graph_level_to_val, variant_selection=variant_selection
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.bert.pretraining.batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=create_bpe_worker_init_fn_from_udi(udi, config, method, split="val"),
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
    )
    
    # # 测试集DataLoader
    # test_dataset = MLMDataset(test_sequences, vocab_manager, transforms, effective_max_length, config.bert.pretraining.mask_prob)
    # test_dataloader = DataLoader(
    #     test_dataset, 
    #     batch_size=config.bert.pretraining.batch_size, 
    #     shuffle=False, 
    #     pin_memory=True,
    #     worker_init_fn=bpe_worker_init_fn,
    #     num_workers=4 if config.serialization.bpe.num_merges > 0 else 0
    # )
    
    # 计算训练步数
    total_steps = len(train_dataloader) * config.bert.pretraining.epochs

    # 在 DataLoader 创建完成后再初始化 CUDA 相关（迁移模型到设备）
    device = torch.device(config.device)
    mlm_model.to(device)

    # 🆕 将损失函数也移动到同一设备，避免设备不匹配错误
    if hasattr(task_handler.loss_fn, 'to'):
        task_handler.loss_fn.to(device)
    
    # 构建优化器和调度器
    optimizer, scheduler = build_from_config(
        mlm_model, config, total_steps=total_steps, stage="pretrain"
    )
    
    display_stage_separator(logger, "训练设置", "构建优化器和调度器")
    # 显示训练设置
    optimizer_info = f"{optimizer.__class__.__name__}(lr={optimizer.param_groups[0]['lr']})"
    scheduler_info = f"{scheduler.__class__.__name__}" if scheduler else "None"
    display_training_setup(
        logger,
        total_steps, len(train_dataloader), config.bert.pretraining.epochs,
        optimizer_info, scheduler_info
    )
    
    # 准备日志和模型保存
    model_dir = config.get_model_dir(run_i=run_i)
    model_dir.mkdir(parents=True, exist_ok=True)

    log_dir = config.get_logs_dir(run_i=run_i)
    log_dir = log_dir / "pretrain"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志记录器（区分 pretrain 阶段）
    writer = SummaryWriter(log_dir / "tensorboard")
    
    # W&B 记录器（可选）
    wandb_logger = None
    if config.logging.use_wandb:
        try:
            from src.utils.logging.wandb_logger import WandbLogger
            
            # 使用配置文件中的设置
            wandb_project = config.logging.wandb_project
            wandb_entity = config.logging.wandb_entity
            wandb_offline = config.logging.wandb_offline
            
            wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                offline=wandb_offline
            )
            # 标注阶段信息用于W&B分组与命名
            setattr(config, "experiment_phase", "pretrain")
            wandb_logger.init(config)
            
            mode_str = "离线模式" if wandb_offline or wandb_logger._offline else "在线模式"
            logger.info(f"✅ W&B初始化成功 ({mode_str})")
            
        except Exception as e:
            logger.warning(f"⚠️ W&B初始化失败，将只使用TensorBoard: {e}")
            wandb_logger = None
    else:
        logger.info("📊 W&B已禁用，仅使用TensorBoard记录")
    
    # 训练状态跟踪
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None  # 存储最佳模型状态，避免频繁磁盘IO
    
    display_stage_separator(logger, "训练循环", "开始训练循环")
    train_start_time = time.time()
    
    epoch_times: List[float] = []
    try:
        for epoch in range(1, config.bert.pretraining.epochs + 1):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            steps_per_epoch = len(train_dataloader)
            log_interval = steps_per_epoch//10
            # 只在第一个epoch显示训练参数
            if epoch == 1:
                logger.info(f"⚡ 训练参数: {steps_per_epoch} steps/epoch × {config.bert.pretraining.epochs} epochs = {total_steps} total steps")

            def _on_step(step_idx: int, batch_loss: float, current_lr: float | None):
                global_step = (epoch - 1) * steps_per_epoch + step_idx
                # if global_step % 20 != 0:
                #   return
                # TensorBoard: 记录更细粒度的batch级loss
                try:
                    writer.add_scalar('Batch_Loss', float(batch_loss), global_step)
                except Exception:
                    pass
                # W&B: 仅记录 train/*（step轴）
                if wandb_logger is not None:
                    payload = {
                        'train/batch_loss': float(batch_loss),
                        'global_step': int(global_step),
                        'epoch': int(epoch),
                    }
                    if current_lr is not None:
                        payload['train/learning_rate_step'] = float(current_lr)
                    try:
                        wandb_logger.log(payload, step=global_step)
                    except Exception:
                        pass

            train_metrics = train_epoch(
                model=mlm_model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                max_grad_norm=config.bert.pretraining.max_grad_norm,
                task_handler=task_handler,  # 🆕 使用TaskHandler计算MLM损失
                on_step=_on_step,
                log_interval=log_interval,
                epoch_num=epoch,
                total_epochs=config.bert.pretraining.epochs,
                log_style=getattr(config.system, 'log_style', 'online'),
                config=config  # 🆕 传入config用于一致性正则化
            )
            # torch.cuda.empty_cache()

            # 验证
            val_metrics = evaluate_epoch(
                model=mlm_model,
                dataloader=val_dataloader,
                device=device,
                task_handler=task_handler,  # 🆕 使用TaskHandler计算MLM损失
                epoch_num=epoch,
                desc="Validation",
                log_style=getattr(config.system, 'log_style', 'online')
            )
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # 记录指标
            train_loss = train_metrics['loss']
            val_loss = val_metrics['loss']
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            
            logger.info(f"📊 Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.2e}, time={epoch_time:.1f}s")
            
            # 🆕 Optuna剪枝支持：报告当前epoch的验证损失
            if getattr(config, 'optuna_trial', None) is not None:
                try:
                    config.optuna_trial.report(val_loss, epoch)
                    if config.optuna_trial.should_prune():
                        logger.info(f"⚠️ Optuna剪枝触发 (epoch {epoch})")
                        raise optuna.TrialPruned()
                except optuna.TrialPruned:
                    raise  # 🔧 剪枝异常必须向上传播到Optuna
                except Exception as e:
                    logger.warning(f"⚠️ Optuna剪枝检查失败: {e}")
                    # 其他异常不影响训练继续
            
            # TensorBoard记录
            writer.add_scalar('Loss/Training', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
            writer.add_scalar('Train/Epoch_Time', epoch_time, epoch)
            
            # W&B记录：epoch级（train_epoch/*与val/*，epoch轴）
            if wandb_logger is not None:
                _epoch_end_global_step = int(epoch * steps_per_epoch)
                wandb_logger.log({
                    "train_epoch/loss": float(train_loss),
                    "train_epoch/learning_rate": float(current_lr),
                    "train_epoch/epoch_time": float(epoch_time),
                    "val/loss": float(val_loss),
                    "epoch": int(epoch)
                }, step=_epoch_end_global_step)
            
            # 早停检查和最佳模型保存
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = float(val_loss)
                best_epoch = epoch
                patience_counter = 0
                logger.info(f"🎯 新最优 (epoch {epoch}): val_loss={best_val_loss:.4f} (↓ {improvement:.4f})")

                # 直接在内存中保存最佳模型状态，避免频繁磁盘IO
                # 对state_dict中的每个张量进行clone，避免引用问题
                del best_model_state
                state_dict_copy = {k: v.clone().detach().cpu() for k, v in mlm_model.state_dict().items()}
                best_model_state = {
                    'model_state_dict': state_dict_copy,
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                }
                logger.info("💾 最佳模型状态已缓存到内存")
            if patience_counter >= config.bert.pretraining.early_stopping_patience:
                logger.info(f"⏹️ 早停触发 (patience={config.bert.pretraining.early_stopping_patience})")
                logger.info(f"  - 最佳epoch: {best_epoch}, 最佳验证损失: {best_val_loss:.4f}")
                break
            patience_counter += 1
            # logger.info(f"清理显存")
            # torch.cuda.empty_cache()
    except KeyboardInterrupt:
        logger.info("⏹️ 训练被用户中断")
    
    finally:
        # 最终指标记录（在关闭writer前写入）
        try:
            total_time = time.time() - train_start_time
            avg_epoch_time = (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0
            writer.add_scalar('Final/Avg_Epoch_Time', float(avg_epoch_time), 0)
            writer.add_scalar('Final/Total_Train_Time', float(total_time), 0)
            writer.add_scalar('Final/Best_Val_Loss', float(best_val_loss), 0)
            if wandb_logger is not None:
                _final_step = int(config.bert.pretraining.epochs * len(train_dataloader)) + 1
                wandb_logger.log({
                    'final_avg_epoch_time': float(avg_epoch_time),
                    'final_total_train_time': float(total_time),
                    'final_best_val_loss': float(best_val_loss),
                }, step=_final_step)
        except Exception:
            pass

        # 关闭记录器
        writer.close()
        if wandb_logger is not None:
            wandb_logger.finish()
        
        #当前显存占用成分分析：
        # import torch.cuda.memory
        # logger.info(f"💾 当前显存占用成分分析: {torch.cuda.memory.summary()}")
        # logger.info(f"💾 model占用显存: {mlm_model.get_memory_footprint()/1024/1024:.2f}MB")
        
        # 保存最佳模型
        logger.info("💾 保存最佳模型...")

        best_model_dir = model_dir / "best"
        best_model_dir.mkdir(parents=True, exist_ok=True)

        if best_model_state is not None:
            # 使用深拷贝创建模型副本，然后加载最佳状态
            import copy
            temp_model = copy.deepcopy(mlm_model)
            temp_model.load_state_dict(best_model_state['model_state_dict'])
            temp_model.save_model(str(best_model_dir))

            logger.info(f"✅ 最佳模型保存成功: {best_model_dir} (epoch {best_model_state['epoch']}, val_loss={best_model_state['val_loss']:.4f})")
        else:
            # 如果没有最佳状态，使用当前模型作为最佳模型
            mlm_model.save_model(str(best_model_dir))
            logger.warning("⚠️ 未找到最佳模型状态，使用当前模型作为最佳模型")
        

        # 训练总结
        total_time = time.time() - train_start_time
        total_samples = len(train) * epoch if epoch > 0 else 0
        
        display_stage_separator(logger, "预训练完成", "训练结果总结")
        display_performance_summary(logger, total_time, total_samples, best_val_loss, best_epoch, "预训练")
        logger.info(f"💾 模型保存: {model_dir}/best/")
        
        # 保存预训练结果到JSON文件（用于实验分析）
        # 🔧 关键修复：临时清理optuna_trial以避免JSON序列化问题
        temp_optuna_trial = getattr(config, 'optuna_trial', None)
        config.optuna_trial = None  # 临时清理
        
        pretrain_metrics = {
            "dataset": config.dataset.name,
            "method": config.serialization.method, 
            "task": "pretraining",
            "epochs": config.bert.pretraining.epochs,
            "best_val_loss": float(best_val_loss),
            "best_epoch": best_epoch,
            "total_train_time_sec": total_time,
            "avg_epoch_time_sec": total_time / config.bert.pretraining.epochs if config.bert.pretraining.epochs > 0 else 0,
            "model_dir": str(model_dir),
            "effective_max_length": effective_max_length,
            # 完整配置信息（用于事后排查）
            "config": config.to_dict()
        }
        
        # 🔧 序列化完成后恢复optuna_trial（虽然通常为None）
        config.optuna_trial = temp_optuna_trial
        
        # 保存metrics文件
        metrics_file = log_dir / "pretrain_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(pretrain_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 预训练结果已保存: {metrics_file}")
    
    mlm_model.to("cpu")
    del mlm_model
    return {
        "mlm_model": mlm_model,
        "vocab_manager": vocab_manager,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "model_dir": str(model_dir),
        "effective_max_length": effective_max_length,
    }
