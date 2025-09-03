#!/usr/bin/env python3
"""
实验数据加载器 - 简单版本
========================

从log目录加载实验结果，提取配置和性能指标，生成统一格式的DataFrame。

主要功能：
- 扫描log目录结构，找到所有metrics JSON文件
- 从JSON中提取配置参数和性能指标
- 支持不同聚合模式的指标提取
- 生成统一格式的DataFrame供分析使用
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings


class ExperimentDataLoader:
    """
    实验数据加载器 - 简单版本

    功能：
    1. 扫描log目录，找到所有实验的metrics文件
    2. 从JSON中提取配置和性能指标
    3. 生成统一的DataFrame格式
    """

    def __init__(self, log_base_dir: str = "log"):
        """
        初始化数据加载器

        Args:
            log_base_dir: log目录路径，默认为"log"
        """
        self.log_base_dir = Path(log_base_dir)
        if not self.log_base_dir.exists():
            raise FileNotFoundError(f"log目录不存在: {log_base_dir}")

    def load_experiments(self,
                        experiment_groups: List[str] = None,
                        datasets: List[str] = None,
                        methods: List[str] = None,
                        prefixes: List[str] = None) -> pd.DataFrame:
        """
        加载实验数据

        Args:
            experiment_groups: 实验组列表，None表示加载所有
            datasets: 数据集列表，None表示加载所有
            methods: 方法列表，None表示加载所有
            prefixes: 前缀列表，None表示加载所有

        Returns:
            包含所有实验数据的DataFrame
        """
        all_results = []

        # 扫描实验目录
        for exp_group_dir in self.log_base_dir.iterdir():
            if not exp_group_dir.is_dir():
                continue

            exp_group = exp_group_dir.name
            if experiment_groups and exp_group not in experiment_groups:
                continue

            print(f"📂 处理实验组: {exp_group}")

            # 扫描实验目录
            for exp_dir in exp_group_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                exp_name = exp_dir.name

                # 扫描数据集目录
                for dataset_dir in exp_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue

                    dataset = dataset_dir.name

                    # 过滤数据集
                    if datasets and dataset not in datasets:
                        continue

                    # 扫描方法目录
                    for method_dir in dataset_dir.iterdir():
                        if not method_dir.is_dir():
                            continue

                        method = method_dir.name

                        # 过滤方法
                        if methods and method not in methods:
                            continue

                        # 扫描prefix目录
                        for prefix_dir in method_dir.iterdir():
                            if not prefix_dir.is_dir():
                                continue

                            prefix = prefix_dir.name
                            if not prefix.endswith('finetune'):
                                continue

                            if prefixes:
                                prefix_base = prefix.replace('finetune', '')
                                if prefix_base and prefix_base not in prefixes:
                                    continue

                            # 查找metrics文件
                            metrics_file = prefix_dir / "finetune_metrics.json"
                            if not metrics_file.exists():
                                continue

                            # 解析metrics文件
                            try:
                                result = self._parse_metrics_file(metrics_file, exp_group, exp_name, dataset, method, prefix)
                                if result:
                                    all_results.append(result)
                            except Exception as e:
                                print(f"⚠️ 解析metrics文件失败: {metrics_file} - {e}")
                                continue

        if not all_results:
            print("❌ 未找到任何有效的实验结果")
            return pd.DataFrame()

        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        print(f"✅ 成功加载 {len(df)} 个实验结果")

        return df

    def _parse_metrics_file(self, metrics_file: Path, exp_group: str, exp_name: str,
                           dataset: str, method: str, prefix: str) -> Optional[Dict]:
        """
        解析单个metrics JSON文件

        Args:
            metrics_file: metrics文件路径
            exp_group: 实验组名
            exp_name: 实验名
            dataset: 数据集名
            method: 方法名
            prefix: 前缀

        Returns:
            解析后的实验数据字典，失败时返回None
        """
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        # 基本信息
        result = {
            'experiment_group': exp_group,
            'experiment_name': exp_name,
            'dataset': dataset,
            'method': method,
            'bpe_method': 'unknown',  # 将在config提取后更新
            'prefix': prefix.replace('finetune', ''),
            'task_type': metrics.get('task', 'unknown'),
            'epochs': metrics.get('epochs', 0),
            'steps_per_epoch': metrics.get('steps_per_epoch', 0),
            'aggregation_mode_used': metrics.get('aggregation_mode_used', 'unknown'),
            'aggregator_trained': metrics.get('aggregator_trained', False),
        }

        # 提取时间信息
        time_info = metrics.get('time', {})
        result.update({
            'total_train_time_sec': time_info.get('total_train_time_sec', 0),
            'avg_epoch_time_sec': time_info.get('avg_epoch_time_sec', 0),
        })

        # 提取训练信息
        train_info = metrics.get('train', {})
        result.update({
            'train_last_loss': train_info.get('last_loss'),
            'train_learning_rate_last': train_info.get('learning_rate_last'),
        })

        # 提取验证信息
        val_info = metrics.get('val', {})
        result.update({
            'val_loss': val_info.get('val_loss'),
            'val_best_mae': val_info.get('best_val_mae'),
        })

        # 提取测试信息 - 不同聚合模式
        test_info = metrics.get('test', {})
        result.update({
            'test_val_loss': test_info.get('val_loss'),
        })

        # 提取不同聚合模式的测试指标
        by_aggregation = test_info.get('by_aggregation', {})
        for agg_mode in ['avg', 'best', 'learned']:
            if agg_mode in by_aggregation:
                agg_data = by_aggregation[agg_mode]
                for key, value in agg_data.items():
                    result[f'test_{key}_{agg_mode}'] = value

        # 提取配置信息
        config = metrics.get('config', {})
        if config:
            config_info = self._extract_config_info(config)
            result.update(config_info)
            # 更新bpe_method
            if 'bpe_method' in config_info:
                result['bpe_method'] = config_info['bpe_method']

        # 文件路径
        result['metrics_file'] = str(metrics_file)
        result['model_dir'] = metrics.get('best_dir', '')

        return result

    def _extract_config_info(self, config: Dict) -> Dict:
        """
        从config中提取关键配置信息

        Args:
            config: 配置字典

        Returns:
            提取的配置信息字典
        """
        config_info = {}

        # 基本配置
        config_info.update({
            'seed': config.get('system', {}).get('seed'),
            'device': config.get('device', config.get('system', {}).get('device')),
        })

        # 数据集配置
        dataset_config = config.get('dataset', {})
        config_info.update({
            'dataset_limit': dataset_config.get('limit'),
            'use_cache': dataset_config.get('use_cache'),
        })

        # 序列化配置
        serialization_config = config.get('serialization', {})
        config_info.update({
            'serialization_method': serialization_config.get('method'),
        })

        # BPE配置
        bpe_config = serialization_config.get('bpe', {})
        config_info.update({
            'bpe_enabled': bpe_config.get('enabled'),
            'bpe_num_merges': bpe_config.get('num_merges'),
            'bpe_min_frequency': bpe_config.get('min_frequency'),
        })

        # BPE引擎配置
        bpe_engine = bpe_config.get('engine', {})
        config_info.update({
            'bpe_method': bpe_engine.get('encode_rank_mode', 'unknown'),
            'bpe_encode_rank_mode': bpe_engine.get('encode_rank_mode'),
            'bpe_encode_rank_k': bpe_engine.get('encode_rank_k'),
        })

        # 编码器配置
        encoder_config = config.get('encoder', {})
        config_info.update({
            'encoder_type': encoder_config.get('type'),
        })

        # BERT架构配置
        bert_config = config.get('bert', {})
        arch_config = bert_config.get('architecture', {})
        config_info.update({
            'd_model': arch_config.get('hidden_size'),
            'n_heads': arch_config.get('num_attention_heads'),
            'n_layers': arch_config.get('num_hidden_layers'),
            'vocab_size': arch_config.get('vocab_size'),
            'max_seq_length': arch_config.get('max_seq_length'),
        })

        # 微调配置
        finetune_config = bert_config.get('finetuning', {})
        config_info.update({
            'finetune_epochs': finetune_config.get('epochs'),
            'finetune_batch_size': finetune_config.get('batch_size'),
            'finetune_learning_rate': finetune_config.get('learning_rate'),
            'finetune_weight_decay': finetune_config.get('weight_decay'),
        })

        # 任务配置
        task_config = config.get('task', {})
        config_info.update({
            'task_type_config': task_config.get('type'),
            'target_property': task_config.get('target_property'),
        })

        # 数据分割
        splits = dataset_config.get('splits', {})
        config_info.update({
            'train_split': splits.get('train'),
            'val_split': splits.get('val'),
            'test_split': splits.get('test'),
        })

        return config_info


def load_experiment_data(experiment_groups: List[str] = None,
                        datasets: List[str] = None,
                        methods: List[str] = None,
                        prefixes: List[str] = None,
                        log_base_dir: str = "log") -> pd.DataFrame:
    """
    便捷函数：加载实验数据

    Args:
        experiment_groups: 实验组列表
        datasets: 数据集列表
        methods: 方法列表
        prefixes: 前缀列表
        log_base_dir: log目录路径

    Returns:
        实验数据DataFrame
    """
    loader = ExperimentDataLoader(log_base_dir)
    return loader.load_experiments(experiment_groups, datasets, methods, prefixes)


# 使用示例
if __name__ == "__main__":
    # 示例：加载所有实验
    df = load_experiment_data()

    if not df.empty:
        print("加载的数据列：")
        print(df.columns.tolist())
        print(f"\n数据形状: {df.shape}")
        print(f"\n前5行数据:")
        print(df.head())

        # 示例：过滤特定实验
        gt_experiments = df[
            (df['method'].isin(['feuler', 'cpp'])) &
            (df['bpe_method'] == 'all')
        ]
        print(f"\nGT实验数量: {len(gt_experiments)}")
