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
import os


class ExperimentDataLoader:
    """
    实验数据加载器 - 适配聚合统计结构

    功能：
    1. 扫描log目录，找到所有实验的finetune_aggregated_stats.json文件
    2. 从JSON中提取配置和性能指标（适配新的聚合统计格式）
    3. 默认提供pk指标的learned模式结果
    4. 生成统一的DataFrame格式
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
                        prefixes: List[str] = None,
                        include_time_info: bool = True,
                        include_train_info: bool = False,
                        pk_aggregation_mode: str = "learned") -> pd.DataFrame:
        """
        加载实验数据

        Args:
            experiment_groups: 实验组列表，None表示加载所有
            datasets: 数据集列表，None表示加载所有
            methods: 方法列表，None表示加载所有
            prefixes: 前缀列表，None表示加载所有
            include_time_info: 是否包含时间信息，默认True
            include_train_info: 是否包含训练信息，默认False
            pk_aggregation_mode: pk指标的聚合模式，默认为"learned"

        Returns:
            包含所有实验数据的DataFrame
        """
        all_results = []

        # 扫描实验目录
        # 目录结构: log/[exp_group]/[mult]/[1]/[experiment_name]/finetune_aggregated_stats.json
        for exp_group_dir in self.log_base_dir.iterdir():
            if not exp_group_dir.is_dir():
                continue

            exp_group = exp_group_dir.name
            if experiment_groups and exp_group not in experiment_groups:
                continue

            print(f"📂 处理实验组: {exp_group}")

            # 递归扫描所有子目录，寻找包含 finetune_aggregated_stats.json 的目录
            for root_dir, dirs, files in os.walk(exp_group_dir):
                if "finetune_aggregated_stats.json" not in files:
                    continue

                exp_dir = Path(root_dir)
                exp_name = exp_dir.name

                # 查找聚合统计文件
                stats_file = exp_dir / "finetune_aggregated_stats.json"
                if not stats_file.exists():
                    continue

                # 先读取并验证文件结构
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        stats = json.load(f)

                    # 检查必需的结构
                    config = stats.get('config')
                    if not config:
                        print(f"⚠️ 跳过: {stats_file} - 缺少config")
                        continue

                    summary = stats.get('summary')
                    if not summary:
                        print(f"⚠️ 跳过: {stats_file} - 缺少summary")
                        continue

                    # 检查必需的config属性
                    dataset_config = config.get('dataset')
                    if not dataset_config or not dataset_config.get('name'):
                        print(f"⚠️ 跳过: {stats_file} - 缺少dataset.name")
                        continue

                    serialization_config = config.get('serialization')
                    if not serialization_config or not serialization_config.get('method'):
                        print(f"⚠️ 跳过: {stats_file} - 缺少serialization.method")
                        continue

                    bpe_config = serialization_config.get('bpe', {})
                    bpe_engine = bpe_config.get('engine', {})
                    if not bpe_engine or not bpe_engine.get('encode_rank_mode'):
                        print(f"⚠️ 跳过: {stats_file} - 缺少bpe.engine.encode_rank_mode")
                        continue

                    encoder_config = config.get('encoder')
                    if not encoder_config or not encoder_config.get('type'):
                        print(f"⚠️ 跳过: {stats_file} - 缺少encoder.type")
                        continue

                    # 检查summary中有pk
                    pk_stats = summary.get('pk_stats')
                    if not pk_stats:
                        print(f"⚠️ 跳过: {stats_file} - 缺少pk_stats")
                        continue

                    # 提取信息
                    dataset = dataset_config['name']
                    method = serialization_config['method']
                    bpe_mode = bpe_engine['encode_rank_mode']
                    encoder_type = encoder_config['type']
                    print(f"🔍 提取信息成功，从 {stats_file} 中提取到: {dataset}, {method}, {bpe_mode}, {encoder_type}")

                except Exception as e:
                    print(f"⚠️ 读取config失败: {stats_file} - {e}")
                    continue

                # 过滤条件
                if datasets and dataset not in datasets:
                    continue
                if methods and method not in methods:
                    continue

                # 解析聚合统计文件
                try:
                    result = self._parse_aggregated_stats_file(
                        stats_file, exp_group, exp_name, dataset, method, exp_name,
                        include_time_info, include_train_info, pk_aggregation_mode
                    )
                    if result:
                        # 添加从目录名解析的额外信息
                        result.update({
                            'bpe_encode_rank_mode': bpe_mode,
                            'encoder_type': encoder_type,
                        })
                        all_results.append(result)
                except Exception as e:
                    print(f"⚠️ 解析聚合统计文件失败: {stats_file} - {e}")
                    continue

        if not all_results:
            print("❌ 未找到任何有效的实验结果")
            return pd.DataFrame()

        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        print(f"✅ 成功加载 {len(df)} 个实验结果")

        return df

    def _parse_aggregated_stats_file(self, stats_file: Path, exp_group: str, exp_name: str,
                                    dataset: str, method: str, prefix: str,
                                    include_time_info: bool, include_train_info: bool,
                                    pk_aggregation_mode: str) -> Optional[Dict]:
        """
        解析单个聚合统计JSON文件

        Args:
            stats_file: 聚合统计文件路径
            exp_group: 实验组名
            exp_name: 实验名
            dataset: 数据集名
            method: 方法名
            prefix: 前缀
            include_time_info: 是否包含时间信息
            include_train_info: 是否包含训练信息
            pk_aggregation_mode: pk指标的聚合模式

        Returns:
            解析后的实验数据字典，失败时返回None
        """
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        # 基本信息
        summary = stats.get('summary', {})
        result = {
            'experiment_group': exp_group,
            'experiment_name': exp_name,
            'dataset': dataset,
            'method': method,
            'prefix': prefix.replace('finetune', ''),
            'task_type': summary.get('task_type', 'unknown'),
            'total_runs': summary.get('total_runs', 0),
            'aggregation_timestamp': summary.get('aggregation_timestamp', ''),
        }

        # 提取PK统计信息（默认learned模式）
        pk_stats = summary.get('pk_stats', {})
        if pk_aggregation_mode in pk_stats:
            pk_data = pk_stats[pk_aggregation_mode]
            result.update({
                'pk_mean': pk_data.get('mean'),
                'pk_std': pk_data.get('std'),
            })
        else:
            # 如果没有pk指标，跳过这个实验
            return None

        # 提取统计信息
        statistics = stats.get('statistics', {})

        # 验证集指标
        val_stats = statistics.get('val', {})
        for metric_name, metric_data in val_stats.items():
            if isinstance(metric_data, dict):
                result[f'val_{metric_name}_mean'] = metric_data.get('mean')
                result[f'val_{metric_name}_std'] = metric_data.get('std')

        # 测试集指标
        test_stats = statistics.get('test', {})
        test_avg = test_stats.get('avg', {})

        # PK指标
        pk_data = test_avg.get('pk', {})
        if pk_data:
            result.update({
                'test_pk_mean': pk_data.get('mean'),
                'test_pk_std': pk_data.get('std'),
            })

        # 其他测试指标
        for metric_name, metric_data in test_avg.items():
            if metric_name != 'pk' and isinstance(metric_data, dict):
                result[f'test_{metric_name}_mean'] = metric_data.get('mean')
                result[f'test_{metric_name}_std'] = metric_data.get('std')

        # 时间信息
        if include_time_info:
            time_stats = statistics.get('time', {})
            for time_metric, time_data in time_stats.items():
                if isinstance(time_data, dict):
                    result[f'{time_metric}_mean'] = time_data.get('mean')
                    result[f'{time_metric}_std'] = time_data.get('std')

        # 训练信息
        if include_train_info:
            train_stats = statistics.get('train', {})
            for train_metric, train_data in train_stats.items():
                if isinstance(train_data, dict):
                    result[f'train_{train_metric}_mean'] = train_data.get('mean')
                    result[f'train_{train_metric}_std'] = train_data.get('std')

        # 提取配置信息
        config = stats.get('config', {})
        if config:
            config_info = self._extract_config_info_from_aggregated(config)
            result.update(config_info)

        # 文件路径
        result['stats_file'] = str(stats_file)

        # 处理浮点数，保留四位小数
        result = self._round_floats(result)

        return result

    def _extract_config_info_from_aggregated(self, config: Dict) -> Dict:
        """
        从聚合统计文件的config中提取关键配置信息

        Args:
            config: 配置字典

        Returns:
            提取的配置信息字典
        """
        config_info = {}

        # 基本配置
        config_info.update({
            'seed': config.get('seed'),
            'device': config.get('device'),
        })

        # 数据集配置
        dataset_config = config.get('dataset', {})
        config_info.update({
            'dataset_name': dataset_config.get('name'),
            'dataset_limit': dataset_config.get('limit'),
            'use_cache': dataset_config.get('use_cache'),
        })

        # 序列化配置
        serialization_config = config.get('serialization', {})
        config_info.update({
            'serialization_method': serialization_config.get('method'),
        })

        # 多重采样配置
        multiple_sampling = serialization_config.get('multiple_sampling', {})
        config_info.update({
            'multiple_sampling_enabled': multiple_sampling.get('enabled'),
            'num_realizations': multiple_sampling.get('num_realizations'),
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
            'bpe_train_backend': bpe_engine.get('train_backend'),
            'bpe_encode_backend': bpe_engine.get('encode_backend'),
            'bpe_encode_rank_mode': bpe_engine.get('encode_rank_mode'),
            'bpe_encode_rank_k': bpe_engine.get('encode_rank_k'),
        })

        # 编码器配置
        encoder_config = config.get('encoder', {})
        config_info.update({
            'encoder_type': encoder_config.get('type'),
            'encoder_reset_weights': encoder_config.get('reset_weights'),
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
            'max_len_policy': arch_config.get('max_len_policy'),
            'pooling_method': arch_config.get('pooling_method'),
        })

        # 微调配置
        finetune_config = bert_config.get('finetuning', {})
        config_info.update({
            'finetune_epochs': finetune_config.get('epochs'),
            'finetune_batch_size': finetune_config.get('batch_size'),
            'finetune_learning_rate': finetune_config.get('learning_rate'),
            'finetune_weight_decay': finetune_config.get('weight_decay'),
            'finetune_warmup_ratio': finetune_config.get('warmup_ratio'),
            'finetune_head_lr_multiplier': finetune_config.get('head_lr_multiplier'),
        })

        # 任务配置
        task_config = config.get('task', {})
        config_info.update({
            'task_type': task_config.get('type'),
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

    def _round_floats(self, data: Dict) -> Dict:
        """
        递归处理字典中的浮点数，保留四位小数

        Args:
            data: 要处理的字典

        Returns:
            处理后的字典
        """
        if isinstance(data, dict):
            return {key: self._round_floats(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._round_floats(item) for item in data]
        elif isinstance(data, float):
            return round(data, 4)
        else:
            return data

def load_experiment_data(experiment_groups: List[str] = None,
                        datasets: List[str] = None,
                        methods: List[str] = None,
                        prefixes: List[str] = None,
                        log_base_dir: str = "log",
                        include_time_info: bool = True,
                        include_train_info: bool = False,
                        pk_aggregation_mode: str = "learned") -> pd.DataFrame:
    """
    便捷函数：加载实验数据

    Args:
        experiment_groups: 实验组列表
        datasets: 数据集列表
        methods: 方法列表
        prefixes: 前缀列表
        log_base_dir: log目录路径
        include_time_info: 是否包含时间信息，默认True
        include_train_info: 是否包含训练信息，默认False
        pk_aggregation_mode: pk指标的聚合模式，默认为"learned"

    Returns:
        实验数据DataFrame
    """
    loader = ExperimentDataLoader(log_base_dir)
    return loader.load_experiments(
        experiment_groups, datasets, methods, prefixes,
        include_time_info, include_train_info, pk_aggregation_mode
    )


# 使用示例
if __name__ == "__main__":
    # 示例：加载所有实验（默认包含时间信息，不包含训练信息，pk使用learned模式）
    df = load_experiment_data()

    if not df.empty:
        print("加载的数据列：")
        print(df.columns.tolist())
        print(f"\n数据形状: {df.shape}")
        print(f"\n前5行数据:")
        print(df.head())

        # 示例：过滤特定实验
        specific_experiments = df[
            (df['serialization_method'].isin(['dfs', 'eulerian'])) &
            (df['bpe_encode_rank_mode'] == 'all')
        ]
        print(f"\n特定实验数量: {len(specific_experiments)}")

        # 示例：加载特定实验组，包含训练信息
        df_with_train = load_experiment_data(
            experiment_groups=['pre_comp1'],
            include_train_info=True
        )
        print(f"\n包含训练信息的实验数量: {len(df_with_train)}")
