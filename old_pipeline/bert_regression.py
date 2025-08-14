#!/usr/bin/env python3
"""
BERT下游任务微调Pipeline (标准化版)
==================================

专门用于BERT下游任务微调，支持：
1. 加载预训练的BERT模型
2. 加载带标签的数据
3. 标签标准化处理（训练时标准化，测试时反标准化）
4. 微调下游任务模型
5. 保存微调模型
6. 提供推理和评估接口
7. TensorBoard指标记录

标准化特性：
- 训练时：标签标准化到标准正态分布
- 测试时：预测值反标准化到原始空间
- 评估指标在原始空间计算
- 支持多种标准化方案（StandardScaler, MinMaxScaler等）
"""

import os  # noqa: F401 (kept for potential future use)
import sys  # noqa: F401 (kept for potential future use)
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn as nn  # noqa: F401 (kept for potential future use)
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler  # noqa: F401
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # noqa: F401
from torch.utils.data import DataLoader  # noqa: F401

# 导入项目模块
from src.models.bert.data import LabelNormalizer
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.training.common import load_training_data
from src.training.loops import train_epoch
from src.training.evaluate import evaluate_model
from src.training.optim import build_optimizer_and_scheduler  # noqa: F401 (kept for compatibility)
# from src.training.early_stopping import update_and_check  # noqa: F401 (might be used in later refactor)
# from src.training.checkpoint import save_best, save_final  # noqa: F401 (compat placeholders)
# 任务头通过统一构建器在 finetune_pipeline 中使用

# 导入BERT相关模块  bert_demo 是bert_demo.py的文件夹  

from src.models.bert.vocab_manager import VocabManager  # noqa: F401
from src.models.bert.model import BertMLM, BertRegression, print_model_info
from src.models.bert.model import BertConfig


logger = logging.getLogger(__name__)


class NormalizedBertFinetuningPipeline:
    """BERT下游任务微调Pipeline (标准化版)"""
    
    def __init__(self, config: ProjectConfig, pretrained_model_path: str = None):
        """
        初始化BERT微调Pipeline
        
        Args:
            config: 项目配置
            pretrained_model_path: 预训练模型路径（可选）
        """
        self.config = config
        self.pretrained_model_path = Path(pretrained_model_path) if pretrained_model_path else self.config.get_bert_model_path("pretrained")
        assert isinstance(self.pretrained_model_path, Path), "pretrained_model_path must be initialized as a Path object"
        logger.info(f"🔍 预训练模型路径: {self.pretrained_model_path}")
        
        # 验证配置
        self.config.validate()
        
        # 设置实验目录（统一目录结构）
        self.experiment_name = self.config.get_experiment_name(pipeline='bert')
        self.logs_dir, self.model_dir = self.config.ensure_experiment_dirs()
        self.model_path = self.model_dir / "model.pkl"
        # 确保目录与结果子目录存在，并设置配置快照路径
        self._setup_directories()
        
        # 初始化TensorBoard写入器（延迟初始化）
        self.writer = None
        
        # 初始化组件
        self.pretrained_model = None
        self.vocab_manager = None
        self.finetuned_model = None
        
        # 标签标准化器
        self.label_normalizer = LabelNormalizer(
            method=config.task.normalization
        )
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.best_epoch = 0
        self.best_metrics = {}  # 保存最优epoch的详细指标
        self.patience_counter = 0
        self.epoch_times = []
        
        # 加载预训练模型
        if pretrained_model_path:
            self.pretrained_model = self.load_pretrained_model(self.pretrained_model_path)
        else:
            # 如果没有指定路径，尝试加载默认路径的预训练模型
            try:
                self.pretrained_model = self.load_pretrained_model(self.pretrained_model_path)
            except Exception as e:
                logger.warning(f"⚠️ 无法加载默认预训练模型: {e}")
                logger.info("💡 请确保已运行预训练阶段，或手动指定预训练模型路径")
                raise Exception("❌ 无法加载默认预训练模型")
    
    def _setup_directories(self):
        """与统一目录结构对齐：直接使用 logs_dir 和 model_dir 根层级。"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.logs_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # 配置快照路径（文件名将把'/'替换为'_'）
        exp_id = self.config.build_experiment_id()
        self.config_path = self.config.get_config_snapshot_path(exp_id)
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.logs_dir / "finetuning.log"
        
        # 获取序列化方法名和BPE使用情况
        serialization_method = self.config.serialization.method
        use_bpe = self.config.serialization.bpe.enabled
        bpe_status = "BPE" if use_bpe else "raw"
        
        # 在格式中包含序列化方法名和BPE状态
        log_format = f'%(asctime)s - %(name)s/{serialization_method}-{bpe_status} - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _init_tensorboard(self):
        """延迟初始化TensorBoard写入器"""
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=str(self.logs_dir))
            logger.info(f"📊 TensorBoard日志目录: {self.logs_dir}")
    
    def _save_config(self):
        """保存配置"""
        config_data = {
            "project_config": self.config.to_dict(),
            "bert_config": {
                "d_model": self.config.bert.architecture.hidden_size,
                "n_layers": self.config.bert.architecture.num_hidden_layers,
                "n_heads": self.config.bert.architecture.num_attention_heads,
                "max_seq_length_upper_bound": self.config.bert.architecture.max_seq_length,
                "hidden_dropout_prob": self.config.bert.architecture.hidden_dropout_prob,
                "attention_probs_dropout_prob": self.config.bert.architecture.attention_probs_dropout_prob,
                "finetune_epochs": self.config.bert.finetuning.epochs,
                "finetune_batch_size": self.config.bert.finetuning.batch_size,
                "finetune_learning_rate": self.config.bert.finetuning.learning_rate,
                "pooling_method": self.config.bert.architecture.pooling_method,
                "task_type": self.config.task.type,
                "target_property": self.config.task.target_property,
                "early_stopping_patience": self.config.bert.finetuning.early_stopping_patience,
                "normalization_method": self.config.task.normalization
            },
            "pretrained_model_path": str(self.pretrained_model_path)
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 配置已保存: {self.config_path}")
    
    def load_pretrained_model(self, pretrained_model_path: Path) -> BertMLM:
        """
        加载预训练的BERT MLM模型
        
        Args:
            pretrained_model_path: 预训练模型路径
            
        Returns:
            BertMLM: 预训练的MLM模型
        """
        logger.info(f"📂 加载预训练模型: {pretrained_model_path}")
        
        try:
            # 检查是否是pickle文件格式
            if pretrained_model_path.suffix == '.pkl':
                # 加载pickle格式的预训练模型
                checkpoint = torch.load(str(pretrained_model_path), map_location='cpu')
                
                # 提取组件
                model_state_dict = checkpoint['model_state_dict']
                config_dict = checkpoint['config']
                vocab_manager = checkpoint['vocab_manager']
                
                # 创建BERT配置
                config = BertConfig(
                    vocab_size=config_dict['vocab_size'],
                    hidden_size=config_dict['d_model'],
                    num_hidden_layers=config_dict['n_layers'],
                    num_attention_heads=config_dict['n_heads'],
                    intermediate_size=config_dict['d_ff'],
                    hidden_dropout_prob=config_dict['bert']['architecture']['hidden_dropout_prob'],
                    attention_probs_dropout_prob=config_dict['bert']['architecture']['attention_probs_dropout_prob'],
                    max_position_embeddings=config_dict['bert']['architecture']['max_position_embeddings'],
                    layer_norm_eps=config_dict['bert']['architecture']['layer_norm_eps'],
                    type_vocab_size=config_dict['bert']['architecture']['type_vocab_size'],
                    initializer_range=config_dict['bert']['architecture']['initializer_range']
                )
                
                self.pretrained_model = BertMLM(config, vocab_manager)
                self.pretrained_model.load_state_dict(model_state_dict)
                self.vocab_manager = vocab_manager
                
                logger.info("✅ 预训练模型加载成功 (pickle格式)")
                logger.info(f"📊 词表大小: {self.vocab_manager.get_vocab_info()['vocab_size']}")
                logger.info(f"📊 模型配置: {config_dict['d_model']}d_{config_dict['n_layers']}l_{config_dict['n_heads']}h")
                
                return self.pretrained_model
                
            else:
                # 尝试加载目录格式的预训练模型
                self.pretrained_model = BertMLM.load_model(str(pretrained_model_path))
                self.vocab_manager = self.pretrained_model.vocab_manager
                
                logger.info("✅ 预训练模型加载成功 (目录格式)")
                logger.info(f"📊 词表大小: {self.vocab_manager.get_vocab_info()['vocab_size']}")
                
                return self.pretrained_model
            
        except Exception as e:
            logger.error(f"❌ 预训练模型加载失败: {e}")
            raise
    
    def _check_vocab_compatibility(self, token_sequences: List[List[int]]):
        from src.training.utils import check_vocab_compatibility
        check_vocab_compatibility(token_sequences, self.vocab_manager)
    
    def create_finetuned_model(self) -> BertRegression:
        """
        创建下游任务模型
        
        Returns:
            BertRegression: 下游任务模型
        """
        logger.info("🔧 创建下游任务模型...")
        
        # 通过统一接口构建任务头并加载 backbone 权重
        from src.training.model_builder import build_task_model
        self.finetuned_model = build_task_model(
            self.config,
            task="regression",
            pretrained=self.pretrained_model,
        )
        logger.info("✅ 预训练权重加载完成")
        print_model_info(self.finetuned_model, self.pretrained_model.config)
        return self.finetuned_model
    
    def finetune_model(self) -> BertRegression:
        """
        微调下游任务模型 (标准化版)
        
        Args:
            token_sequences: Token序列列表（未使用，直接获取已划分数据）
            labels: 原始标签列表（未使用，直接获取已划分数据）
            
        Returns:
            BertRegression: 微调后的模型
        """
        logger.info(f"🎯 开始{self.config.task.type}任务微调 (标准化版)...")
        
        # 1. 直接获取已划分的数据
        dataset_name = self.config.dataset.name
        method = self.config.serialization.method
        use_bpe = self.config.serialization.bpe.enabled
        target_property = self.config.task.target_property
        
        # 使用 UnifiedDataInterface + 通用加载工具
        udi = UnifiedDataInterface(config=self.config, dataset=dataset_name)
        (
            train_sequences,
            val_sequences,
            test_sequences,
            train_labels,
            val_labels,
            test_labels,
            _,
        ) = load_training_data(
            udi=udi,
            method=method,
            target_property=target_property,
            use_bpe=use_bpe,
        )
        # 基础校验
        if target_property and (not train_labels or not val_labels or not test_labels):
            raise ValueError(f"未能获取到目标属性标签: {target_property}")
        
        # 记录数据集规模
        train_size = len(train_sequences)
        val_size = len(val_sequences)
        test_size = len(test_sequences)
        logger.info(f"📊 数据集分割: 训练:{train_size} 验证:{val_size} 测试:{test_size}")
        
        # 2-5. 使用统一构建器生成 DataLoader 与标准化器
        from src.training.tasks import build_regression_loaders
        train_dataloader, val_dataloader, test_dataloader, self.label_normalizer = build_regression_loaders(
            self.config,
            self.pretrained_model,
            train_sequences, val_sequences, test_sequences,
            train_labels, val_labels, test_labels,
        )
        
        # 4. 设置设备
        device = self.config.system.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finetuned_model.to(device)
        
        # 5. 优化器/调度器（从 config 构建，减少入参）
        total_batches = len(train_dataloader)
        total_steps = total_batches * self.config.bert.finetuning.epochs
        from src.training.optim import build_from_config
        optimizer, scheduler = build_from_config(self.finetuned_model, self.config, total_steps=total_steps, stage="finetune")
        
        # 7. 训练循环 (使用通用循环，保持指标记录等价)
        logger.info(f"🔄 开始训练: {self.config.bert.finetuning.epochs} epochs, {total_batches} batches/epoch")
        for epoch in range(self.config.bert.finetuning.epochs):
            epoch_start_time = time.time()

            def _on_step(step_idx: int, train_loss: float, lr: float | None):
                try:
                    self._init_tensorboard()
                    self.writer.add_scalar('Train/Regression_Loss', train_loss, self.global_step + step_idx)
                    if lr is not None:
                        self.writer.add_scalar('Train/Learning_Rate', lr, self.global_step + step_idx)
                except Exception:
                    pass

            train_stats = train_epoch(
                self.finetuned_model,
                train_dataloader,
                optimizer,
                scheduler,
                device,
                max_grad_norm=self.config.bert.finetuning.max_grad_norm,
                on_step=_on_step,
                log_interval=100,
            )
            self.global_step += train_stats['steps']
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            logger.info(f"🔍 epoch {epoch + 1} validation:")
            val_metrics = evaluate_model(self.finetuned_model, val_dataloader, device, task="regression", label_normalizer=self.label_normalizer)

            # 记录epoch统计 (详细性能指标)
            avg_epoch_loss = train_stats['loss']
            try:
                self._init_tensorboard()
                self.writer.add_scalar('Epoch/Train_Loss', avg_epoch_loss, epoch + 1)
                self.writer.add_scalar('Epoch/Val_Loss', val_metrics['val_loss'], epoch + 1)
                self.writer.add_scalar('Epoch/Time', epoch_time, epoch + 1)
            except Exception:
                pass

            logger.info(f"📈 Epoch {epoch + 1} 完成，耗时: {epoch_time:.2f}s, trainLoss: {avg_epoch_loss:.4f}")
            logger.info(f"   val Loss: {val_metrics['val_loss']:.4f}, val_MAE: {val_metrics['mae']:.4f},val_R²: {val_metrics['r2']:.4f}")

            # 检查是否是最优模型 (基于验证损失)
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_val_mae = val_metrics['mae']
                self.best_epoch = epoch + 1
                self.best_metrics = val_metrics.copy()
                self.patience_counter = 0

                # 使用模型自带保存接口（保存到 <model_dir>/best）
                best_dir = self.model_dir / "best"
                self.finetuned_model.save_model(str(best_dir))
                # 额外保存标准化器
                import pickle
                with open(best_dir / "label_normalizer.pkl", "wb") as f:
                    pickle.dump(self.label_normalizer, f)
                logger.info(f"💾 最优模型与标准化器已保存: {best_dir}")
            else:
                self.patience_counter += 1
                logger.info(f"⏳ 早停计数器: {self.patience_counter}/{self.config.bert.finetuning.early_stopping_patience}")

            if self.patience_counter >= self.config.bert.finetuning.early_stopping_patience:
                logger.info(f"🔥 早停触发! 最优epoch: {self.best_epoch}, 最优损失: {self.best_val_loss:.4f}")
                break
        
        # 保存最终模型和标准化器到新的model目录结构
        # 确保模型目录存在
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存最终模型（目录 <model_dir>/final），并保存标准化器
        final_dir = self.model_dir / "final"
        self.finetuned_model.save_model(str(final_dir))
        import pickle
        with open(final_dir / "label_normalizer.pkl", "wb") as f:
            pickle.dump(self.label_normalizer, f)
        logger.info(f"💾 最终模型与标准化器已保存: {final_dir}")
        
        # 8. 训练完成后的测试集评估
        logger.info("🎯 训练完成，开始测试集最终评估...")
        
        # 在测试集上评估
        test_metrics = evaluate_model(self.finetuned_model, test_dataloader, device, task="regression", label_normalizer=self.label_normalizer)
        
        # 记录最终核心指标 (使用固定横轴值1，与预训练保持一致)
        self._init_tensorboard()
        self.writer.add_scalar('Final/Best_Val_Loss', self.best_val_loss, 1)
        self.writer.add_scalar('Final/Best_Val_MAE', self.best_val_mae, 1)
        self.writer.add_scalar('Final/Best_Epoch', self.best_epoch, 1)
        self.writer.add_scalar('Final/Total_Time', time.time() - epoch_start_time, 1)
        self.writer.add_scalar('Final/Avg_Epoch_Time', np.mean(self.epoch_times), 1)
        
        # 记录测试集指标
        self.writer.add_scalar('Final/Test_Loss', test_metrics['test_loss'], 1)
        self.writer.add_scalar('Final/Test_MSE', test_metrics['test_mse'], 1)
        self.writer.add_scalar('Final/Test_MAE', test_metrics['test_mae'], 1)
        self.writer.add_scalar('Final/Test_RMSE', test_metrics['test_rmse'], 1)
        self.writer.add_scalar('Final/Test_R2', test_metrics['test_r2'], 1)
        self.writer.add_scalar('Final/Test_Correlation', test_metrics['test_correlation'], 1)
        
        # 关闭TensorBoard写入器
        if self.writer:
            self.writer.close()
        
        # 记录最终结果
        logger.info("🎉 微调完成！最终结果:")
        logger.info(f"   - 最优epoch: {self.best_epoch}")
        logger.info(f"   - 最优验证损失: {self.best_val_loss:.4f}")
        logger.info(f"   - 测试损失: {test_metrics['test_loss']:.4f}")
        logger.info(f"   - 测试MSE: {test_metrics['test_mse']:.4f}")
        logger.info(f"   - 测试R²: {test_metrics['test_r2']:.4f}")
        
        return self.finetuned_model
    
    def run_finetuning_pipeline(self) -> Dict[str, Any]:
        """
        运行完整的微调Pipeline
        
        Returns:
            微调结果字典
        """
        logger.info("🚀 开始BERT下游任务微调Pipeline (标准化版)...")

        # 在任何可能修改配置之前，打印当前配置快照
        try:
            logger.info("🧾 配置快照(进入Finetune Pipeline时):\n" + json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False))
        except Exception:
            logger.info("🧾 配置快照打印失败，使用简要输出")
            logger.info(str(self.config.to_dict()))
        
        # 设置日志
        self._setup_logging()
        
        # 保存配置
        self._save_config()
        
        # 1. 加载预训练模型
        if not self.pretrained_model:
            # 如果还没有加载预训练模型，尝试加载
            try:
                self.pretrained_model = self.load_pretrained_model(self.pretrained_model_path)
            except Exception as e:
                logger.error(f"❌ 无法加载预训练模型: {e}")
                logger.error("💡 请确保已运行预训练阶段，或手动指定预训练模型路径")
                raise
        
        # 3. 创建下游任务模型
        self.create_finetuned_model()
        
        # 4. 微调模型
        self.finetune_model()
        
        # 5. 保存训练统计
        self._save_training_stats()
        
        logger.info("🎉 BERT下游任务微调Pipeline (标准化版)完成！")
        
        return {
            'vocab_manager': self.vocab_manager,
            'pretrained_model': self.pretrained_model,
            'finetuned_model': self.finetuned_model,
            'label_normalizer': self.label_normalizer,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'model_path': str(self.model_dir / "best")  # 返回最优模型路径
        }
    
    def _save_training_stats(self):
        """保存训练统计"""
        stats_path = self.results_dir / "finetuning_stats.json"
        
        stats = {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,  # 最优epoch的详细指标
            'total_epochs': len(self.epoch_times),
            'avg_epoch_time': np.mean(self.epoch_times),
            'total_training_time': sum(self.epoch_times),
            'early_stopping_triggered': self.patience_counter >= self.config.bert.finetuning.early_stopping_patience,
            'loaded_model_path': str(self.pretrained_model_path),
            'normalization_method': self.label_normalizer.method,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': self.config.to_dict()
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 微调统计已保存: {stats_path}")
        
        # 记录最终性能总结
        logger.info("🎯 训练完成总结:")
        logger.info(f"   最优epoch: {self.best_epoch}")
        logger.info(f"   最优验证损失: {self.best_val_loss:.4f}")
        if self.best_metrics:
            logger.info(f"   最优MSE: {self.best_metrics['mse']:.4f}")
            logger.info(f"   最优MAE: {self.best_metrics['mae']:.4f}")
            logger.info(f"   最优RMSE: {self.best_metrics['rmse']:.4f}")
            logger.info(f"   最优R²: {self.best_metrics['r2']:.4f}")
            logger.info(f"   最优相关系数: {self.best_metrics['correlation']:.4f}")
        logger.info(f"   总训练时间: {sum(self.epoch_times):.2f}s")
        logger.info(f"   平均epoch时间: {np.mean(self.epoch_times):.2f}s")


# ===========================================
# 便捷函数接口
# ===========================================

def finetune_bert_model_normalized(config: ProjectConfig, pretrained_model_path: str = None) -> str:
    """
    微调BERT模型的便捷函数 (标准化版)
    
    Args:
        config: 项目配置
        pretrained_model_path: 预训练模型路径（可选）
        
    Returns:
        最优模型目录路径
    """
    # 创建Pipeline
    pipeline = NormalizedBertFinetuningPipeline(config, pretrained_model_path)
    
    # 运行微调
    results = pipeline.run_finetuning_pipeline()
    
    return results['model_path']


# ===========================================
# 主函数
# ===========================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BERT下游任务微调Pipeline (标准化版)")
    parser.add_argument("--pretrained_model", required=False, help="预训练模型路径")
    parser.add_argument("--config", type=str, help="配置文件路径 (可选)")
    parser.add_argument("--experiment_name", type=str, help="实验名称 (可选)")
    parser.add_argument("--group", type=str, help="实验分组(必须用于目录结构)")
    parser.add_argument("--serialization_method", type=str, help="序列化方法(如 eulerian/feuler 等)")
    parser.add_argument("--dataset", type=str, help="数据集名称")
    parser.add_argument("--target_property", type=str, help="下游目标属性(如 aqsol 的 solubility)")
    parser.add_argument("--finetune_epochs", type=int, help="微调epoch数")
    parser.add_argument("--finetune_batch_size", type=int, help="微调批次大小")
    parser.add_argument("--finetune_learning_rate", type=float, help="微调学习率")
    parser.add_argument("--early_stopping_patience", type=int, help="早停耐心值")
    
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
    
    # 如果提供了数据集名称/目标属性，则设置
    if args.dataset:
        config.dataset.name = args.dataset
    if args.target_property:
        config.task.target_property = args.target_property
    
    # 从命令行参数更新常用训练参数
    if args.finetune_epochs:
        config.bert.finetuning.epochs = args.finetune_epochs
    if args.finetune_batch_size:
        config.bert.finetuning.batch_size = args.finetune_batch_size
    if args.finetune_learning_rate:
        config.bert.finetuning.learning_rate = args.finetune_learning_rate
    if args.early_stopping_patience:
        config.bert.finetuning.early_stopping_patience = args.early_stopping_patience
    
    try:
        model_path = finetune_bert_model_normalized(config, args.pretrained_model)
        print(f"✅ 微调完成！最优模型保存在: {model_path}")
    except Exception as e:
        print(f"❌ 微调失败: {e}")
        raise

if __name__ == "__main__":
    main() 