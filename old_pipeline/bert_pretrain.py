#!/usr/bin/env python3
"""
BERT预训练Pipeline (优化版)
==========================

优化重点：
1. 精简指标记录，只保留核心必要指标
2. 减少冗余信息
3. 提高训练效率
4. 保持关键监控能力

核心指标：
- 训练损失 (实时)
- 验证损失 (每epoch)
- 学习率 (实时)
- 训练时间 (每epoch)
- 最佳模型信息 (最终)
"""

import os  # noqa: F401
import sys  # noqa: F401
import json
import time
import argparse
from pathlib import Path  # noqa: F401
from typing import Dict, List, Any
import logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn as nn  # noqa: F401 (kept for potential future use)
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 导入项目模块
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.training.common import load_sequences_splits, flatten_all_sequences

# 导入BERT相关模块
# 词表在 UDI 中维护与加载；此处不直接依赖 VocabManager 接口
from src.models.bert.model import BertMLM, create_bert_mlm, print_model_info
from src.models.bert.data import create_mlm_dataloader, compute_effective_max_length
from src.training.loops import train_epoch, evaluate_epoch
from src.training.optim import build_optimizer_and_scheduler  # noqa: F401 (kept for compatibility)
from src.training.callbacks import update_and_check

class OptimizedBertPretrainingPipeline:
    """优化的BERT预训练Pipeline"""
    
    def __init__(self, config: ProjectConfig):
        """
        初始化BERT预训练Pipeline
        
        Args:
            config: 项目配置
        """
        self.config = config
        
        # 验证配置
        self.config.validate()
        
        # 设置实验目录（统一目录结构）
        self.experiment_name = self.config.get_experiment_name(pipeline='bert')
        self.logs_dir, self.model_dir = self.config.ensure_experiment_dirs()
        self.model_file = self.model_dir / "model.pkl"
        
        # 设置目录与日志
        self._setup_directories()
        self._setup_logging()
        
        # 初始化TensorBoard写入器（直接写入标准 logs_dir）
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))
        
        # 初始化组件
        self.vocab_manager = None
        self.mlm_model = None
        self.effective_max_length: int | None = None
        # 缓存拆分序列，避免重复数据加载
        self._train_sequences = None
        self._val_sequences = None
        self._test_sequences = None
        self._all_sequences = None
        
        # 训练状态 (精简)
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.epoch_times = []  # 记录每个epoch的耗时
        
        logger.info(f"🚀 初始化BERT预训练Pipeline: {self.experiment_name}")
        logger.info(f"📁 模型输出目录: {self.model_dir}")
        logger.info(f"📁 训练日志目录: {self.logs_dir}")
    
    def _setup_directories(self):
        """与统一目录结构对齐：直接使用 logs_dir 和 model_dir 根层级。"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # 配置快照路径（文件名将把'/'替换为'_'）
        exp_id = self.config.build_experiment_id()
        self.config_path = self.config.get_config_snapshot_path(exp_id)
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.logs_dir / "pretraining.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # 获取logger
        global logger
        logger = logging.getLogger(__name__)
        
        logger.info(f"📝 日志文件: {log_file}")
    
    def _save_config(self):
        """保存配置"""
        config_data = {
            "experiment_name": self.experiment_name,
            "project_config": self.config.to_dict(),
            "bert_config": {
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
                "max_seq_length_upper_bound": self.config.bert.architecture.max_seq_length,
                "effective_max_length": self.effective_max_length,
                "hidden_dropout_prob": self.config.bert.architecture.hidden_dropout_prob,
                "attention_probs_dropout_prob": self.config.bert.architecture.attention_probs_dropout_prob,
                "mlm_epochs": self.config.bert.pretraining.epochs,
                "mlm_batch_size": self.config.bert.pretraining.batch_size,
                "mlm_learning_rate": self.config.bert.pretraining.learning_rate,
                "vocab_min_freq": self.config.bert.pretraining.vocab_min_freq,
                "max_vocab_size": self.config.bert.pretraining.max_vocab_size
            }
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 配置已保存: {self.config_path}")
    
    def load_data(self) -> List[List[int]]:
        """加载序列化数据；返回合并后的序列用于构建词表。"""
        logger.info("📂 加载序列化数据...")
        logger.info(f"📂 数据集: {self.config.dataset.name}")
        logger.info(f"📂 序列化方法: {self.config.serialization.method}")
        logger.info(f"📂 使用预处理BPE数据集: {self.config.serialization.bpe.enabled}")
        
        try:
            dataset_name = self.config.dataset.name
            method = self.config.serialization.method
            use_bpe = self.config.serialization.bpe.enabled
            
            # 使用 UnifiedDataInterface + 通用加载工具（仅缓存模式，缺失直接报错）
            udi = UnifiedDataInterface(config=self.config, dataset=dataset_name)
            
            train_sequences, val_sequences, test_sequences, _ = load_sequences_splits(
                udi, method, use_bpe
            )
            # 保存分割供后续复用，避免重复数据加载
            self._train_sequences = train_sequences
            self._val_sequences = val_sequences
            self._test_sequences = test_sequences

            # 合并所有序列用于词表构建
            all_sequences = flatten_all_sequences(train_sequences, val_sequences, test_sequences)
            logger.info(f"✅ 数据加载完成: {len(all_sequences)} 个序列")
            logger.info(f"   - 训练集: {len(train_sequences)} 个序列")
            logger.info(f"   - 验证集: {len(val_sequences)} 个序列")
            logger.info(f"   - 测试集: {len(test_sequences)} 个序列")
            
            self._all_sequences = all_sequences
            # 加载词表（与 数据集+method 绑定），训练阶段不再构建/注册词表
            try:
                self.vocab_manager = udi.get_vocab(method=method)
                vocab_info = self.vocab_manager.get_vocab_info()
                logger.info(f"✅ 词表已从 UDI 加载: {vocab_info['vocab_size']} 个token (method={method})")
                # 同步配置中的词表大小
                self.config.bert.architecture.vocab_size = int(vocab_info['vocab_size'])
            except Exception as e:
                logger.error(f"❌ 词表加载失败（请先通过准备器构建并注册）: {e}")
                raise
            return all_sequences
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise
    
    # 词表与数据集+method 绑定，训练阶段不再构建/注册；保留接口（不再使用）
    
    def create_bert_model(self) -> BertMLM:
        """创建BERT模型"""
        logger.info("🏗️ 创建BERT MLM模型...")
        
        # 使用配置创建模型
        self.mlm_model = create_bert_mlm(
            vocab_manager=self.vocab_manager,
            hidden_size=self.config.d_model,
            num_hidden_layers=self.config.n_layers,
            num_attention_heads=self.config.n_heads,
            intermediate_size=self.config.d_ff,
            hidden_dropout_prob=self.config.bert.architecture.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.bert.architecture.attention_probs_dropout_prob,
            max_position_embeddings=self.effective_max_length,
            layer_norm_eps=self.config.bert.architecture.layer_norm_eps,
            type_vocab_size=self.config.bert.architecture.type_vocab_size,
            initializer_range=self.config.bert.architecture.initializer_range
        )
        # 确保保存到checkpoint的配置与模型权重一致（位置嵌入大小）
        if self.effective_max_length is not None:
            self.config.bert.architecture.max_position_embeddings = int(self.effective_max_length)

        # 打印模型详细信息
        try:
            print_model_info(self.mlm_model, self.mlm_model.config)
        except Exception as e:
            logger.warning(f"打印模型信息失败: {e}")
        
        logger.info(f"✅ BERT模型创建完成: {self.config.d_model}d_{self.config.n_layers}l_{self.config.n_heads}h")
        return self.mlm_model
    
    def train_mlm(self, _token_sequences: List[List[int]] | None = None) -> BertMLM:
        """训练BERT MLM模型 (优化版)"""
        logger.info("🎓 开始MLM预训练...")
        
        # 复用 load_data 阶段得到的分割，避免重复数据加载
        assert self._train_sequences is not None and self._val_sequences is not None and self._test_sequences is not None, "load_data 未先调用"
        train_sequences = self._train_sequences
        val_sequences = self._val_sequences
        test_sequences = self._test_sequences
        
        # 获取配置中的BPE状态
        use_bpe = self.config.serialization.bpe.enabled
        dataset_name = self.config.dataset.name
        method = self.config.serialization.method
        # 使用默认版本
        data_version = "latest"
        udi = UnifiedDataInterface(config=self.config, dataset=dataset_name)
        
        train_size = len(train_sequences)
        val_size = len(val_sequences)
        test_size = len(test_sequences)
        logger.info(f"📊 数据集分割: 训练集 {train_size} 个序列, 验证集 {val_size} 个序列, 测试集 {test_size} 个序列")
        
        # TODO: [BPE重构] 使用原始序列长度计算位置嵌入
        # 当前使用预处理BPE数据集，但仍需基于原始序列确定模型的max_position_embeddings
        # 未来将实现真正的在线BPE Transform流程。
        
        # 临时方案：始终基于原始序列计算 max_position_embeddings
        if use_bpe:
            logger.info("🔧 基于原始序列计算BERT位置嵌入大小（实际训练使用预处理BPE压缩序列）...")
            # 加载原始序列用于长度计算
            raw_train, raw_val, raw_test, _ = load_sequences_splits(
                udi, method, use_bpe=False, data_version=data_version
            )
            raw_all_sequences = flatten_all_sequences(raw_train, raw_val, raw_test)
            self.effective_max_length = compute_effective_max_length(raw_all_sequences, self.config, split_name="raw_for_position")
        else:
            # 使用当前数据计算（已经是原始序列）
            all_sequences = train_sequences + val_sequences + test_sequences
            self.effective_max_length = compute_effective_max_length(all_sequences, self.config, split_name="all")
        # 使用同一长度创建所有 DataLoader
        train_dataloader = create_mlm_dataloader(
            token_sequences=train_sequences,
            vocab_manager=self.vocab_manager,
            batch_size=self.config.bert.pretraining.batch_size,
            max_length=self.effective_max_length,
            mlm_probability=self.config.bert.pretraining.mask_prob,
            config=self.config
        )
        
        val_dataloader = create_mlm_dataloader(
            token_sequences=val_sequences,
            vocab_manager=self.vocab_manager,
            batch_size=self.config.bert.pretraining.batch_size,
            max_length=self.effective_max_length,
            mlm_probability=self.config.bert.pretraining.mask_prob,
            config=self.config
        )
        
        # 创建BERT模型
        self.create_bert_model()
        
        # 设置设备
        device = torch.device(self.config.device)
        self.mlm_model.to(device)

        # 优化器与学习率调度器（等价替换）
        total_batches = len(train_dataloader)
        total_steps = total_batches * self.config.bert.pretraining.epochs
        from src.training.optim import build_from_config
        optimizer, scheduler = build_from_config(self.mlm_model, self.config, total_steps=total_steps, stage="pretrain")

        # 获取早停耐心值
        early_stopping_patience = self.config.bert.pretraining.early_stopping_patience

        # 训练循环
        logger.info(f"🔄 开始训练: {self.config.bert.pretraining.epochs} epochs, {total_batches} batches/epoch")
        for epoch in range(self.config.bert.pretraining.epochs):
            epoch_start_time = time.time()

            train_stats = train_epoch(
                self.mlm_model,
                train_dataloader,
                optimizer,
                scheduler,
                device,
                max_grad_norm=self.config.bert.pretraining.max_grad_norm,
            )
            self.global_step += train_stats['steps']
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            val_stats = evaluate_epoch(self.mlm_model, val_dataloader, device)
            avg_epoch_loss = train_stats['loss']
            val_loss = val_stats['loss']

            # 记录epoch统计 (核心指标)
            try:
                self.writer.add_scalar('Epoch/Train_Loss', avg_epoch_loss, epoch + 1)
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch + 1)
                self.writer.add_scalar('Epoch/Time', epoch_time, epoch + 1)
            except Exception:
                pass

            logger.info(
                f"📈 {self.config.dataset.name}/{self.config.serialization.method}/"
                f"{'BPE-Preprocessed' if self.config.serialization.bpe.enabled else 'Raw'} "
                f"Epoch {epoch + 1} 完成, 耗时: {epoch_time:.2f}s, train Loss: {avg_epoch_loss:.4f}, val Loss: {val_loss:.4f}"
            )

            # 最优模型与早停
            old_best = self.best_val_loss
            self.best_val_loss, self.patience_counter, should_stop = update_and_check(
                best_metric=self.best_val_loss,
                new_metric=val_loss,
                patience_counter=self.patience_counter,
                patience=early_stopping_patience,
            )
            if self.best_val_loss < old_best:
                self.best_epoch = epoch + 1
                logger.info(f"🎯 新的最优模型! 验证损失: {val_loss:.4f}, 保存模型")
                # 使用模型自带的保存接口（目录形式）
                best_dir = self.model_dir / "best"
                self.mlm_model.save_model(str(best_dir))
            else:
                logger.info(f"⏳ 早停计数器: {self.patience_counter}/{early_stopping_patience}")

            if should_stop:
                logger.info(f"🔥 早停触发! 最优epoch: {self.best_epoch}, 最优损失: {self.best_val_loss:.4f}")
                break

        # 最终模型保存（使用最后一个 epoch 的训练损失作为代表）
        _final_avg_loss = avg_epoch_loss  # 仅用于兼容旧日志语义
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # 使用模型自带保存接口输出最终模型（目录形式）
        final_dir = self.model_dir / "final"
        self.mlm_model.save_model(str(final_dir))
        # 兼容：另存一份到标准预训练目录，供下游统一加载
        try:
            pretrain_target_dir = self.config.get_bert_model_path("pretrained").parent
            pretrain_target_dir.mkdir(parents=True, exist_ok=True)
            self.mlm_model.save_model(str(pretrain_target_dir))
            logger.info(f"💾 预训练模型兼容保存: {pretrain_target_dir}")
        except Exception as e:
            logger.warning(f"⚠️ 兼容保存预训练模型失败: {e}")
        
        logger.info(f"💾 MLM预训练模型已保存: {self.model_dir}")
        
        # 关闭TensorBoard写入器（后续将替换为 WandB）
        try:
            self.writer.add_scalar('Final/Best_Val_Loss', self.best_val_loss, 1)
            self.writer.add_scalar('Final/Best_Epoch', self.best_epoch, 1)
            self.writer.add_scalar('Final/Total_Time', time.time() - epoch_start_time, 1)
            self.writer.add_scalar('Final/Avg_Epoch_Time', np.mean(self.epoch_times), 1)
            self.writer.close()
        except Exception:
            pass
        
        return self.mlm_model
    
    def _evaluate_on_validation(self, val_dataloader, device) -> float:
        """在验证集上评估模型"""
        self.mlm_model.eval()
        total_val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.mlm_model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                total_val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_val_loss / num_batches
        self.mlm_model.train()
        
        return avg_val_loss
    
    def run_pretraining_pipeline(self) -> Dict[str, Any]:
        """运行完整的预训练Pipeline"""
        logger.info("🚀 开始BERT预训练Pipeline...")

        # 在任何可能修改配置之前，打印当前配置快照
        try:
            logger.info("🧾 配置快照(进入Pipeline时):\n" + json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False))
        except Exception:
            logger.info("🧾 配置快照打印失败，使用简要输出")
            logger.info(str(self.config.to_dict()))
        
        # 保存配置
        self._save_config()
        
        # 1. 加载数据
        token_sequences = self.load_data()
        
        # 3. MLM预训练
        self.train_mlm(token_sequences)
        
        logger.info("🎉 BERT预训练Pipeline完成！")
        
        return {
            'vocab_manager': self.vocab_manager,
            'mlm_model': self.mlm_model,
            'model_path': str(self.model_dir / "best"),
            'experiment_name': self.experiment_name,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }

# ===========================================
# 便捷函数接口
# ===========================================

def pretrain_bert_model(config: ProjectConfig) -> str:
    """预训练BERT模型的便捷函数"""
    pipeline = OptimizedBertPretrainingPipeline(config)
    results = pipeline.run_pretraining_pipeline()
    return results['model_path']

# ===========================================
# 主函数
# ===========================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BERT预训练Pipeline (优化版)")
    parser.add_argument("--config", type=str, help="配置文件路径 (可选)")
    parser.add_argument("--experiment_name", type=str, help="实验名称 (可选)")
    parser.add_argument("--group", type=str, help="实验分组(必须用于目录结构)")
    parser.add_argument("--serialization_method", type=str, help="序列化方法(如 eulerian/feuler 等)")
    parser.add_argument("--mlm_epochs", type=int, help="MLM训练轮数")
    parser.add_argument("--mlm_batch_size", type=int, help="MLM批次大小")
    parser.add_argument("--mlm_learning_rate", type=float, help="MLM学习率")
    parser.add_argument("--dataset", type=str, help="数据集名称")
    
    args = parser.parse_args()
    
    # 创建配置（支持从文件路径初始化）
    if args.config:
        config = ProjectConfig(config_path=args.config)
    else:
        config = ProjectConfig()
    
    # 如果提供了实验名称/分组/方法/数据集，则设置
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.group:
        config.experiment_group = args.group
    if args.serialization_method:
        config.serialization.method = args.serialization_method
    
    # 从命令行参数更新常用训练参数
    if args.mlm_epochs:
        config.bert.pretraining.epochs = args.mlm_epochs
    if args.mlm_batch_size:
        config.bert.pretraining.batch_size = args.mlm_batch_size
    if args.mlm_learning_rate:
        config.bert.pretraining.learning_rate = args.mlm_learning_rate
    if args.dataset:
        config.dataset.name = args.dataset
    try:
        model_path = pretrain_bert_model(config)
        print(f"✅ 预训练完成！模型保存在: {model_path}")
    except Exception as e:
        print(f"❌ 预训练失败: {e}")
        raise

if __name__ == "__main__":
    main() 