#!/usr/bin/env python3
"""
BERT分类任务微调Pipeline
==================================

专门用于BERT分类任务微调，支持：
1. 加载预训练的BERT模型
2. 加载带标签的分类数据
3. 微调分类任务模型
4. 保存微调模型
5. 提供推理和评估接口
6. TensorBoard指标记录

分类特性：
- 支持多分类任务
- 自动处理类别不平衡（可选）
- 支持多种评估指标（准确率、精确率、召回率、F1等）
- 支持混淆矩阵生成
"""

import os  # noqa: F401
import sys  # noqa: F401
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn as nn  # noqa: F401
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# 导入项目模块
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface

# 导入BERT相关模块
from src.models.bert.model import BertMLM, BertClassification, print_model_info
from src.training.tasks import build_classification_loaders
from src.training.evaluate import evaluate_model
from src.training.loops import train_epoch
from src.training.optim import build_from_config
from src.training.model_builder import build_task_model
from src.models.bert.model import BertConfig

logger = logging.getLogger(__name__)


class BertClassificationFinetuningPipeline:
    """BERT分类任务微调Pipeline"""
    
    def __init__(self, config: ProjectConfig, pretrained_model_path: str = None):
        """
        初始化BERT分类微调Pipeline
        
        Args:
            config: 项目配置
            pretrained_model_path: 预训练模型路径（可选）
        """
        self.config = config
        # 预训练模型路径需由调用方提供；不再通过旧接口构造
        if pretrained_model_path is None:
            raise ValueError("必须提供预训练模型路径 pretrained_model_path")
        self.pretrained_model_path = Path(pretrained_model_path)
        logger.info(f"🔍 预训练模型路径: {self.pretrained_model_path}")
        
        # 验证配置
        self.config.validate()
        
        # 设置输出目录
        self._setup_directories()
        
        # 初始化TensorBoard写入器（延迟初始化）
        self.writer = None
        
        # 初始化组件
        self.pretrained_model = None
        self.vocab_manager = None
        self.finetuned_model = None
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
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
        """设置输出目录"""
        # 使用新的目录结构
        self.experiment_name = self.config.get_experiment_name(pipeline='bert')
        
        # 模型保存在统一的 model_dir 中
        _, self.model_dir = self.config.ensure_experiment_dirs()
        self.model_path = self.model_dir / "model.pkl"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志保存在统一的 logs_dir 中
        self.logs_dir, _ = self.config.ensure_experiment_dirs()
        self.results_dir = self.logs_dir / "results"
        
        for dir_path in [self.logs_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 配置文件
        self.config_path = self.logs_dir / "finetuning_config.json"
    
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
                "task_type": "classification",
                "target_property": self.config.task.target_property,
                "early_stopping_patience": self.config.bert.finetuning.early_stopping_patience
            },
            "pretrained_model_path": str(self.pretrained_model_path)
        }
        
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
        """
        检查微调数据的词表与预训练模型词表的兼容性
        
        Args:
            token_sequences: 微调数据的token序列
        """
        if not self.vocab_manager:
            logger.warning("⚠️ 词表管理器未加载，跳过词表兼容性检查")
            return
        
        logger.info("🔍 检查词表兼容性...")
        
        # 统计所有token
        all_tokens = set()
        for seq in token_sequences:
            all_tokens.update(seq)
        
        # 检查哪些token不在预训练词表中
        unknown_tokens = set()
        for token in all_tokens:
            if token not in self.vocab_manager.token_to_id:
                unknown_tokens.add(token)
        
        # 计算统计信息
        total_tokens = len(all_tokens)
        unknown_count = len(unknown_tokens)
        unknown_ratio = unknown_count / total_tokens * 100 if total_tokens > 0 else 0
        
        logger.info("📊 词表兼容性统计:")
        logger.info(f"   总token类型数: {total_tokens}")
        logger.info(f"   未知token类型数: {unknown_count}")
        logger.info(f"   未知token比例: {unknown_ratio:.2f}%")
        
        if unknown_count > 0:
            logger.warning("⚠️ 发现 %d 个未知token类型", unknown_count)
            logger.warning("   这些token将被替换为UNK token，可能导致信息丢失")
            
            # 如果未知token比例过高，给出严重警告
            if unknown_ratio > 10:
                logger.error(f"❌ 未知token比例过高 ({unknown_ratio:.2f}%)！")
                logger.error("   建议使用与预训练相同的数据处理流程")
                logger.error("   或者重新预训练模型以包含这些token")
                
                # 显示一些未知token示例
                unknown_examples = list(unknown_tokens)[:10]
                logger.error(f"   未知token示例: {unknown_examples}")
                
                # 询问是否继续
                logger.error("   继续微调可能导致性能严重下降")
            elif unknown_ratio > 5:
                logger.warning(f"⚠️ 未知token比例较高 ({unknown_ratio:.2f}%)")
                logger.warning("   建议检查数据处理流程是否与预训练一致")
            else:
                logger.info(f"✅ 词表兼容性良好，未知token比例较低 ({unknown_ratio:.2f}%)")
        else:
            logger.info("✅ 词表完全兼容，所有token都在预训练词表中")
    
    def create_finetuned_model(self, num_classes: int) -> BertClassification:
        """
        创建分类任务模型
        
        Args:
            num_classes: 类别数量
            
        Returns:
            BertClassification: 分类任务模型
        """
        logger.info(f"🔧 创建分类任务模型 (类别数: {num_classes})...")
        
        # 通过统一构建器创建任务模型并加载预训练 backbone 权重
        self.finetuned_model = build_task_model(
            self.config,
            task="classification",
            pretrained=self.pretrained_model,
            num_classes=num_classes,
        )
        logger.info("✅ 预训练权重加载完成")
        print_model_info(self.finetuned_model, self.pretrained_model.config)
        return self.finetuned_model

    def finetune_model(self, num_classes: int) -> BertClassification:
        """
        微调分类任务模型
        
        Args:
            num_classes: 类别数量
            
        Returns:
            BertClassification: 微调后的模型
        """
        logger.info(f"🎯 开始分类任务微调 (类别数: {num_classes})...")
        
        # 1. 直接获取已划分的数据
        dataset_name = self.config.dataset.name
        method = self.config.serialization.method
        use_bpe = self.config.serialization.bpe.enabled
        target_property = self.config.task.target_property
        
        # 使用 UnifiedDataInterface 获取数据
        udi = UnifiedDataInterface(config=self.config, dataset=dataset_name)
        
        # 使用重构后的训练数据加载接口
        from src.training.common import load_training_data
        (
            train_sequences,
            val_sequences,
            test_sequences,
            train_cont_labels,
            val_cont_labels,
            test_cont_labels,
            _,
        ) = load_training_data(
            udi=udi,
            method=method,
            target_property=target_property,
            use_bpe=use_bpe,
        )
        if target_property is None:
            raise ValueError("分类任务必须提供 target_property 以生成类别标签")
        # 将连续标签转换为分类标签
        train_labels = self._convert_to_classification_labels(train_cont_labels, num_classes)
        val_labels = self._convert_to_classification_labels(val_cont_labels, num_classes)
        test_labels = self._convert_to_classification_labels(test_cont_labels, num_classes)
        
        # 将连续值转换为分类标签（这里需要根据具体任务调整）
        # 示例：将连续值按分位数分为多个类别
        train_size = len(train_sequences)
        val_size = len(val_sequences)
        test_size = len(test_sequences)
        
        logger.info(f"📊 数据集分割: 训练:{train_size}:测试:{test_size}:验证:{val_size}")
        
        # 2. 创建分类标签（这里需要根据具体任务实现）
        # 示例：将连续值转换为分类标签
        train_class_labels = self._convert_to_classification_labels(train_labels, num_classes)
        val_class_labels = self._convert_to_classification_labels(val_labels, num_classes)
        test_class_labels = self._convert_to_classification_labels(test_labels, num_classes)
        
        # 3-4. 使用统一构建器生成 DataLoader
        train_dataloader, val_dataloader, test_dataloader = build_classification_loaders(
            self.config,
            self.pretrained_model,
            train_sequences, val_sequences, test_sequences,
            train_class_labels, val_class_labels, test_class_labels,
            num_classes=num_classes,
        )
        
        # 5. 设置设备
        device = self.config.system.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finetuned_model.to(device)
        
        # 6-7. 优化器与调度器（统一构建器）
        total_batches = len(train_dataloader)
        total_steps = total_batches * self.config.bert.finetuning.epochs
        optimizer, scheduler = build_from_config(self.finetuned_model, self.config, total_steps=total_steps, stage="finetune")
        
        # 8. 训练循环
        logger.info(f"🔄 开始训练: {self.config.bert.finetuning.epochs} epochs, {total_batches} batches/epoch")
        for epoch in range(self.config.bert.finetuning.epochs):
            epoch_start_time = time.time()
            train_stats = train_epoch(
                self.finetuned_model,
                train_dataloader,
                optimizer,
                scheduler,
                device,
                max_grad_norm=self.config.bert.finetuning.max_grad_norm,
            )
            
            # 计算epoch耗时
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # 验证集评估
            logger.info(f"🔍 epoch {epoch + 1} validation:")
            val_metrics = evaluate_model(self.finetuned_model, val_dataloader, device, task="classification")
            
            # 记录epoch统计
            avg_epoch_loss = train_stats['loss']
            self._init_tensorboard()
            
            # 训练指标
            self.writer.add_scalar('Epoch/Train_Loss', avg_epoch_loss, epoch + 1)
            self.writer.add_scalar('Epoch/Time', epoch_time, epoch + 1)
            
            # 验证指标
            self.writer.add_scalar('Epoch/Val_Loss', val_metrics['val_loss'], epoch + 1)
            self.writer.add_scalar('Epoch/Val_Accuracy', val_metrics['accuracy'], epoch + 1)
            self.writer.add_scalar('Epoch/Val_F1', val_metrics['f1'], epoch + 1)
            self.writer.add_scalar('Epoch/Val_Precision', val_metrics['precision'], epoch + 1)
            self.writer.add_scalar('Epoch/Val_Recall', val_metrics['recall'], epoch + 1)
            
            logger.info(f"📈 Epoch {epoch + 1} 完成，耗时: {epoch_time:.2f}s, trainLoss: {avg_epoch_loss:.4f}")
            logger.info(f"   val Loss: {val_metrics['val_loss']:.4f}, val_Accuracy: {val_metrics['accuracy']:.4f}, val_F1: {val_metrics['f1']:.4f}")
            
            # 检查是否是最优模型 (基于验证准确率)
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch + 1
                self.best_metrics = val_metrics.copy()
                self.patience_counter = 0
                logger.info(f"🎯 新的最优模型! 验证准确率: {val_metrics['accuracy']:.4f}")
                
                # 保存最优模型
                with open(self.model_path, 'wb') as f:
                    torch.save({
                        'model_state_dict': self.finetuned_model.state_dict(),
                        'config': self.finetuned_model.config,
                        'num_classes': self.finetuned_model.num_classes,
                        'pooling_method': self.config.bert.architecture.pooling_method,
                        'vocab_manager': self.finetuned_model.vocab_manager,
                        'is_best': True,
                        'epoch': epoch + 1,
                        'best_metrics': val_metrics
                    }, f)
                logger.info(f"💾 最优模型已保存: {self.model_path}")
            else:
                self.patience_counter += 1
                logger.info(f"⏳ 早停计数器: {self.patience_counter}/{self.config.bert.finetuning.early_stopping_patience}")
            
            # 早停检查
            if self.patience_counter >= self.config.bert.finetuning.early_stopping_patience:
                logger.info(f"🔥 早停触发! 最优epoch: {self.best_epoch}, 最优准确率: {self.best_val_accuracy:.4f}")
                break
        
        # 保存最终模型
        if not hasattr(self, 'best_metrics') or not self.best_metrics:
            with open(self.model_path, 'wb') as f:
                torch.save({
                    'model_state_dict': self.finetuned_model.state_dict(),
                    'config': self.finetuned_model.config,
                    'num_classes': self.finetuned_model.num_classes,
                    'pooling_method': self.config.bert.architecture.pooling_method,
                    'vocab_manager': self.finetuned_model.vocab_manager,
                    'is_best': False,
                    'epoch': len(self.epoch_times),
                }, f)
            logger.info(f"💾 最终模型已保存: {self.model_path}")
        else:
            logger.info(f"💾 已保存最优模型，无需重复保存最终模型: {self.model_path}")
        
        # 9. 训练完成后的测试集评估
        logger.info("🎯 训练完成，开始测试集最终评估...")
        
        # 在测试集上评估
        test_metrics = self._evaluate_on_test(test_dataloader, device)
        
        # 记录最终核心指标
        self._init_tensorboard()
        self.writer.add_scalar('Final/Best_Val_Loss', self.best_val_loss, 1)
        self.writer.add_scalar('Final/Best_Val_Accuracy', self.best_val_accuracy, 1)
        self.writer.add_scalar('Final/Best_Epoch', self.best_epoch, 1)
        self.writer.add_scalar('Final/Total_Time', time.time() - epoch_start_time, 1)
        self.writer.add_scalar('Final/Avg_Epoch_Time', np.mean(self.epoch_times), 1)
        
        # 记录测试集指标
        self.writer.add_scalar('Final/Test_Loss', test_metrics['test_loss'], 1)
        self.writer.add_scalar('Final/Test_Accuracy', test_metrics['test_accuracy'], 1)
        self.writer.add_scalar('Final/Test_F1', test_metrics['test_f1'], 1)
        self.writer.add_scalar('Final/Test_Precision', test_metrics['test_precision'], 1)
        self.writer.add_scalar('Final/Test_Recall', test_metrics['test_recall'], 1)
        
        # 关闭TensorBoard写入器
        if self.writer:
            self.writer.close()
        
        # 记录最终结果
        logger.info("🎉 微调完成！最终结果:")
        logger.info(f"   - 最优epoch: {self.best_epoch}")
        logger.info(f"   - 最优验证准确率: {self.best_val_accuracy:.4f}")
        logger.info(f"   - 测试准确率: {test_metrics['test_accuracy']:.4f}")
        logger.info(f"   - 测试F1: {test_metrics['test_f1']:.4f}")
        
        return self.finetuned_model
    
    def _convert_to_classification_labels(self, continuous_labels: List[float], num_classes: int) -> List[int]:
        """
        将连续值转换为分类标签
        
        Args:
            continuous_labels: 连续值标签列表
            num_classes: 类别数量
            
        Returns:
            分类标签列表
        """
        # 这里需要根据具体任务实现
        # 示例：使用分位数将连续值分为多个类别
        labels_array = np.array(continuous_labels)
        
        if num_classes == 2:
            # 二分类：使用中位数作为阈值
            threshold = np.median(labels_array)
            return [1 if label > threshold else 0 for label in continuous_labels]
        else:
            # 多分类：使用分位数
            percentiles = np.linspace(0, 100, num_classes + 1)
            thresholds = np.percentile(labels_array, percentiles[1:-1])
            
            class_labels = []
            for label in continuous_labels:
                class_label = 0
                for i, threshold in enumerate(thresholds):
                    if label > threshold:
                        class_label = i + 1
                class_labels.append(class_label)
            
            return class_labels
    def run_finetuning_pipeline(self, num_classes: int = 2) -> Dict[str, Any]:
        """
        运行完整的分类微调Pipeline
        
        Args:
            num_classes: 类别数量
            
        Returns:
            微调结果字典
        """
        logger.info(f"🚀 开始BERT分类任务微调Pipeline (类别数: {num_classes})...")

        # 在任何可能修改配置之前，打印当前配置快照
        try:
            logger.info("🧾 配置快照(进入Classification Pipeline时):\n" + json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False))
        except Exception:
            logger.info("🧾 配置快照打印失败，使用简要输出")
            logger.info(str(self.config.to_dict()))
        
        # 设置日志
        self._setup_logging()
        
        # 保存配置
        self._save_config()
        
        # 1. 加载预训练模型
        if not self.pretrained_model:
            try:
                self.pretrained_model = self.load_pretrained_model(self.pretrained_model_path)
            except Exception as e:
                logger.error(f"❌ 无法加载预训练模型: {e}")
                logger.error("💡 请确保已运行预训练阶段，或手动指定预训练模型路径")
                raise
        
        # 2. 创建分类任务模型
        self.create_finetuned_model(num_classes)
        
        # 3. 微调模型
        self.finetune_model(num_classes)
        
        # 4. 保存训练统计
        self._save_training_stats()
        
        logger.info("🎉 BERT分类任务微调Pipeline完成！")
        
        return {
            'vocab_manager': self.vocab_manager,
            'pretrained_model': self.pretrained_model,
            'finetuned_model': self.finetuned_model,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'model_path': str(self.model_dir / "best")
        }
    
    def _save_training_stats(self):
        """保存训练统计"""
        stats_path = self.results_dir / "finetuning_stats.json"
        
        stats = {
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.epoch_times),
            'avg_epoch_time': np.mean(self.epoch_times),
            'total_training_time': sum(self.epoch_times),
            'early_stopping_triggered': self.patience_counter >= self.config.bert.finetuning.early_stopping_patience,
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
        logger.info(f"   最优验证准确率: {self.best_val_accuracy:.4f}")
        if self.best_metrics:
            logger.info(f"   最优F1: {self.best_metrics['f1']:.4f}")
            logger.info(f"   最优精确率: {self.best_metrics['precision']:.4f}")
            logger.info(f"   最优召回率: {self.best_metrics['recall']:.4f}")
        logger.info(f"   总训练时间: {sum(self.epoch_times):.2f}s")
        logger.info(f"   平均epoch时间: {np.mean(self.epoch_times):.2f}s")


# ===========================================
# 便捷函数接口
# ===========================================

def finetune_bert_classification(config: ProjectConfig, num_classes: int = 2, pretrained_model_path: str = None) -> str:
    """
    微调BERT分类模型的便捷函数
    
    Args:
        config: 项目配置
        num_classes: 类别数量
        pretrained_model_path: 预训练模型路径（可选）
        
    Returns:
        最优模型目录路径
    """
    # 创建Pipeline
    pipeline = BertClassificationFinetuningPipeline(config, pretrained_model_path)
    
    # 运行微调
    results = pipeline.run_finetuning_pipeline(num_classes)
    
    return results['model_path']


# ===========================================
# 主函数
# ===========================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BERT分类任务微调Pipeline")
    parser.add_argument("--pretrained_model", required=False, help="预训练模型路径")
    parser.add_argument("--config", type=str, help="配置文件路径 (可选)")
    parser.add_argument("--experiment_name", type=str, help="实验名称 (可选)")
    parser.add_argument("--dataset", type=str, help="数据集名称")
    parser.add_argument("--num_classes", type=int, default=2, help="类别数量")
    parser.add_argument("--finetune_epochs", type=int, help="微调epoch数")
    parser.add_argument("--finetune_batch_size", type=int, help="微调批次大小")
    parser.add_argument("--finetune_learning_rate", type=float, help="微调学习率")
    parser.add_argument("--early_stopping_patience", type=int, help="早停耐心值")
    
    args = parser.parse_args()
    
    # 创建配置
    config = ProjectConfig()
    
    # 如果提供了配置文件，则加载
    if args.config:
        config.load_from_file(args.config)
    
    # 如果提供了实验名称，则设置
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # 如果提供了数据集名称，则设置
    if args.dataset:
        config.dataset.name = args.dataset
    
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
        model_path = finetune_bert_classification(config, args.num_classes, args.pretrained_model)
        print(f"✅ 分类微调完成！最优模型保存在: {model_path}")
    except Exception as e:
        print(f"❌ 分类微调失败: {e}")
        raise

if __name__ == "__main__":
    main()
