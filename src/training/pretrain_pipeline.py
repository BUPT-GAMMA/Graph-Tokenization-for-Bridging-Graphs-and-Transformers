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
from typing import Dict, List, Any
# from pathlib import Path  # unused
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.data.bpe_transform import create_bpe_worker_init_fn_from_udi
from src.models.model_factory import create_universal_model
from src.models.bert.data import compute_effective_max_length
from src.models.bert.vocab_manager import VocabManager
from src.training.loops import train_epoch, evaluate_epoch
from src.training.optim import build_from_config
from src.training.callbacks import update_and_check
from src.utils.logger import get_logger

logger = get_logger('tokenizerGraph.training.pretrain_pipeline')
logger.propagate = False



def train_bert_mlm(
    config: ProjectConfig,
    token_sequences: Dict[str, List[List[int]]],
    vocab_manager: VocabManager,
    udi: UnifiedDataInterface,
    method: str,
) -> Dict[str, Any]:
    """
    BERT MLM预训练主函数
    
    Args:
        config: 项目配置
        token_sequences: 包含"train"/"val"/"test"键的序列字典
        vocab_manager: 可选的词表管理器，如果None则从token_sequences构建
        udi: 可选的统一数据接口，用于BPE Transform（如果需要）
        method: 序列化方法名，用于BPE Transform（如果需要）
        
    Returns:
        包含训练结果的字典
    """
    logger.info("🎓 开始BERT MLM预训练...")
    
    # 验证输入数据
    required_splits = ["train", "val", "test"]
    for split in required_splits:
        assert split in token_sequences, f"缺少数据划分: {split}"
    
    train_sequences = token_sequences["train"]
    val_sequences = token_sequences["val"]
    test_sequences = token_sequences["test"]
    
    logger.info(f"📊 数据集分割: 训练集 {len(train_sequences)}, 验证集 {len(val_sequences)}, 测试集 {len(test_sequences)}")
    
    # 使用提供的词表
    vocab_info = vocab_manager.get_vocab_info()
    logger.info(f"✅ 使用提供的词表: {vocab_info['vocab_size']} 个token")
    
    # 同步配置中的词表大小
    config.bert.architecture.vocab_size = int(vocab_info['vocab_size'])
    
    # 计算有效最大长度
    all_sequences = train_sequences + val_sequences + test_sequences
    effective_max_length = compute_effective_max_length(all_sequences, config)
    logger.info(f"📏 计算得到有效最大长度: {effective_max_length}")
    
    # 🆕 创建统一模型（MLM任务）
    logger.info("🏗️ 创建统一MLM模型...")
    
    # 确保配置中的位置嵌入大小与有效长度一致
    config.bert.architecture.max_position_embeddings = int(effective_max_length)
    
    # 使用统一模型创建接口
    mlm_model, task_handler = create_universal_model(
        config=config,
        vocab_manager=vocab_manager,
        task_type='mlm'  # 🎯 MLM作为任务类型
    )
    
    logger.info(f"✅ MLM模型创建完成: 编码器({mlm_model.encoder.get_hidden_size()}维) + MLM头({mlm_model.output_dim})")
    
    device = torch.device(config.device)
    mlm_model.to(device)
    
    # 打印模型信息（适配UniversalModel）
    try:
        hidden_size = mlm_model.encoder.get_hidden_size()
        vocab_size = vocab_manager.vocab_size
        logger.info("📊 模型信息:")
        logger.info(f"  - 编码器类型: {config.encoder.type}")
        logger.info(f"  - 隐藏维度: {hidden_size}")
        logger.info(f"  - 词表大小: {vocab_size}")
        logger.info("  - 任务类型: MLM")
        logger.info(f"  - 输出维度: {mlm_model.output_dim}")
    except Exception as e:
        logger.warning(f"打印模型信息失败: {e}")
    
    logger.info(f"✅ 统一MLM模型创建完成: {config.bert.architecture.hidden_size}d_{config.bert.architecture.num_hidden_layers}l_{config.bert.architecture.num_attention_heads}h")
    
    # 创建BPE Transform worker初始化函数（统一创建，mode控制行为）
    try:
        bpe_worker_init_fn = create_bpe_worker_init_fn_from_udi(udi, config, method)
        bpe_mode = config.serialization.bpe.engine.encode_rank_mode
        if bpe_mode == "none":
            logger.info("📦 BPE模式: none (无压缩，使用原始序列)")
        else:
            logger.info(f"🔧 BPE模式: {bpe_mode} (启用BPE压缩)")
    except Exception as e:
        logger.error(f"❌ BPE Transform创建失败: {e}")
        raise
    
    # 创建数据加载器
    logger.info("📦 创建数据加载器...")
    
    # 创建带BPE Transform的DataLoader
    from src.models.bert.data import MLMDataset, create_transforms_from_config
    from torch.utils.data import DataLoader
    
    # 获取有效的token列表，用于数据增强
    valid_tokens = vocab_manager.get_valid_tokens()
    transforms = create_transforms_from_config(config, valid_tokens, "mlm")
    
    # 训练集DataLoader
    train_dataset = MLMDataset(train_sequences, vocab_manager, transforms, effective_max_length, config.bert.pretraining.mask_prob)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.bert.pretraining.batch_size, 
        shuffle=True, 
        pin_memory=True,
        worker_init_fn=bpe_worker_init_fn,
        num_workers=4  # 统一使用多进程，BPE mode控制具体行为
    )
    
    # 验证集DataLoader
    val_dataset = MLMDataset(val_sequences, vocab_manager, transforms, effective_max_length, config.bert.pretraining.mask_prob)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.bert.pretraining.batch_size, 
        shuffle=False, 
        pin_memory=True,
        worker_init_fn=bpe_worker_init_fn,
        num_workers=4  # 统一使用多进程，BPE mode控制具体行为
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
    logger.info(f"📈 总训练步数: {total_steps} ({len(train_dataloader)} steps/epoch × {config.bert.pretraining.epochs} epochs)")
    
    # 构建优化器和调度器
    optimizer, scheduler = build_from_config(
        mlm_model, config, total_steps=total_steps, stage="pretrain"
    )
    
    # 准备日志和模型保存
    model_dir = config.get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = config.get_logs_dir()
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
    
    logger.info("🚀 开始训练循环...")
    train_start_time = time.time()
    
    epoch_times: List[float] = []
    try:
        for epoch in range(1, config.bert.pretraining.epochs + 1):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            steps_per_epoch = len(train_dataloader)
            log_interval = steps_per_epoch//10
            print(f"log_interval: {log_interval}, steps_per_epoch: {steps_per_epoch}")

            def _on_step(step_idx: int, batch_loss: float, current_lr: float | None):
                global_step = (epoch - 1) * steps_per_epoch + step_idx
                # if global_step % 20 != 0:
                #   return
                # TensorBoard: 记录更细粒度的batch级loss
                try:
                    writer.add_scalar('Train/Batch_Loss', float(batch_loss), global_step)
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
                log_style=getattr(config.system, 'log_style', 'online')
            )
            
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
            
            # TensorBoard记录
            writer.add_scalar('Loss/Train', train_loss, epoch)
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
            new_best_val_loss, patience_counter, should_stop = update_and_check(
                best_metric=best_val_loss,
                new_metric=val_loss,
                patience_counter=patience_counter,
                patience=config.bert.pretraining.early_stopping_patience
            )
            
            # 检查是否有改进（val_loss降低了）
            if new_best_val_loss < best_val_loss:
                best_val_loss = new_best_val_loss
                best_epoch = epoch
                
                # 保存最佳模型
                best_model_dir = model_dir / "best"
                mlm_model.save_model(str(best_model_dir))
                logger.info(f"💾 保存最佳模型 (epoch {epoch}, val_loss={val_loss:.4f}): {best_model_dir}")
            
            if should_stop:
                logger.info(f"⏹️ 早停触发 (patience={config.bert.pretraining.early_stopping_patience})")
                break
    
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
        
        # 保存最终模型
        final_model_dir = model_dir / "final"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        mlm_model.save_model(str(final_model_dir))
        
        # 保存配置
        config_path = model_dir / "config.json"
        config_data = {
            "model_config": {
                "hidden_size": config.bert.architecture.hidden_size,
                "num_hidden_layers": config.bert.architecture.num_hidden_layers,
                "num_attention_heads": config.bert.architecture.num_attention_heads,
                "intermediate_size": config.bert.architecture.intermediate_size,
                "max_position_embeddings": effective_max_length,
                "vocab_size": vocab_info['vocab_size'],
            },
            "training_config": {
                "epochs": config.bert.pretraining.epochs,
                "batch_size": config.bert.pretraining.batch_size,
                "learning_rate": config.bert.pretraining.learning_rate,
                "mask_prob": config.bert.pretraining.mask_prob,
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - train_start_time
        logger.info(f"✅ 预训练完成! 总用时: {total_time:.1f}s")
        logger.info(f"📊 最佳验证损失: {best_val_loss:.4f} (epoch {best_epoch})")
        logger.info(f"💾 模型保存路径: {model_dir}")
    
    return {
        "mlm_model": mlm_model,
        "vocab_manager": vocab_manager,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "model_dir": str(model_dir),
        "effective_max_length": effective_max_length,
    }
