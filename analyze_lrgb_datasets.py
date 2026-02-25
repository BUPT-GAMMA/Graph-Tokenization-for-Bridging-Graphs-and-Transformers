#!/usr/bin/env python3
"""
LRGB和MOLHIV数据集统计分析脚本
==============================

分析Peptides-func、Peptides-struct和ogbg-molhiv三个数据集的特征统计信息。
"""

import numpy as np
import torch
from typing import Dict, List, Any, Tuple
import json
from collections import Counter
from pathlib import Path

from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader
from utils.logger import get_logger

logger = get_logger(__name__)


def calculate_class_balance_metrics(labels: List[Any], num_classes: int = None) -> Dict[str, Any]:
    """计算类别平衡性指标"""
    if not labels:
        return {}
    
    # 处理多标签情况
    if isinstance(labels[0], (list, tuple, np.ndarray)):
        # 处理嵌套列表格式（如peptides_func的[[...]]格式）
        processed_labels = []
        for label in labels:
            if isinstance(label, (list, tuple)) and len(label) == 1 and isinstance(label[0], (list, tuple, np.ndarray)):
                # 嵌套格式：[[...]] -> [...]
                processed_labels.append(label[0])
            elif isinstance(label, np.ndarray) and label.ndim > 1:
                # 多维数组：flatten
                processed_labels.append(label.flatten())
            else:
                # 直接格式
                processed_labels.append(label)
        
        # 转换为多标签二进制矩阵
        if num_classes is None:
            if processed_labels:
                first_label = processed_labels[0]
                if hasattr(first_label, '__len__'):
                    num_classes = len(first_label)
                else:
                    num_classes = 1
            else:
                num_classes = 1
        
        class_counts = [0] * num_classes
        for label_vec in processed_labels:
            # 确保label_vec是可迭代的
            if isinstance(label_vec, np.ndarray):
                label_vec = label_vec.flatten()
            elif not isinstance(label_vec, (list, tuple)):
                label_vec = [label_vec]
            
            for i, val in enumerate(label_vec):
                if i < num_classes:
                    # 安全地转换为数值进行比较
                    try:
                        numeric_val = float(val)
                        if numeric_val > 0:  # 对于分类任务，假设>0表示正类
                            class_counts[i] += 1
                    except (ValueError, TypeError):
                        continue
                    
        counts = class_counts
    else:
        # 单标签分类
        if isinstance(labels[0], torch.Tensor):
            labels = [int(l.item()) for l in labels]
        elif isinstance(labels[0], np.ndarray):
            labels = [int(l.item()) if hasattr(l, 'item') else int(l) for l in labels]
        else:
            labels = [int(l) for l in labels]
            
        counter = Counter(labels)
        if num_classes is None:
            num_classes = max(counter.keys()) + 1
        counts = [counter.get(i, 0) for i in range(num_classes)]
    
    total_samples = sum(counts)
    if total_samples == 0:
        return {}
    
    proportions = [c / total_samples for c in counts]
    
    # 计算各种平衡性指标
    non_zero_counts = [c for c in counts if c > 0]
    non_zero_props = [p for p in proportions if p > 0]
    
    if len(non_zero_counts) == 0:
        return {}
    
    # Imbalance Ratio (IR): max/min ratio
    ir = max(non_zero_counts) / min(non_zero_counts) if len(non_zero_counts) > 1 else 1.0
    
    # Coefficient of Variation (CV)
    if len(non_zero_counts) > 1:
        mean_count = np.mean(non_zero_counts)
        cv = np.std(non_zero_counts) / mean_count if mean_count > 0 else float('inf')
    else:
        cv = 0.0
    
    # Normalized Entropy
    if len(non_zero_props) > 1:
        entropy = -sum(p * np.log(p) for p in non_zero_props if p > 0)
        max_entropy = np.log(len(non_zero_props))
        h_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        h_norm = 0.0
    
    # Normalized Gini coefficient
    if len(non_zero_counts) > 1:
        sorted_counts = sorted(non_zero_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * sum((i + 1) * count for i, count in enumerate(sorted_counts))) / (n * sum(sorted_counts)) - (n + 1) / n
        gini_norm = 1 - gini  # Normalized: 1 = perfect balance, 0 = maximum imbalance
    else:
        gini_norm = 1.0
    
    # KL divergence from uniform distribution
    uniform_prop = 1.0 / len(non_zero_props)
    kl_div = sum(p * np.log(p / uniform_prop) for p in non_zero_props if p > 0)
    
    return {
        'counts': counts,
        'proportions': proportions,
        'total_samples': total_samples,
        'num_classes': len(non_zero_counts),
        'ir': ir,
        'cv': cv,
        'h_norm': h_norm,
        'gini_norm': gini_norm,
        'kl_div': kl_div
    }


def analyze_graph_statistics(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析图结构统计信息"""
    if not data_list:
        return {}
    
    num_nodes_list = []
    num_edges_list = []
    node_types = Counter()
    edge_types = Counter()
    
    for sample in data_list:
        if 'dgl_graph' in sample:
            g = sample['dgl_graph']
            num_nodes_list.append(g.num_nodes())
            num_edges_list.append(g.num_edges())
            
            # 统计节点类型
            if 'node_type_id' in g.ndata:
                node_type_ids = g.ndata['node_type_id'].tolist()
                for ntype in node_type_ids:
                    node_types[ntype] += 1
            elif 'x' in g.ndata and g.ndata['x'].size(1) > 0:
                # 使用第一个特征维度作为类型
                node_type_ids = g.ndata['x'][:, 0].long().tolist()
                for ntype in node_type_ids:
                    node_types[ntype] += 1
            
            # 统计边类型
            if 'edge_type_id' in g.edata:
                edge_type_ids = g.edata['edge_type_id'].tolist()
                for etype in edge_type_ids:
                    edge_types[etype] += 1
            elif 'edge_attr' in g.edata and g.edata['edge_attr'].size(1) > 0:
                # 使用第一个特征维度作为类型
                edge_type_ids = g.edata['edge_attr'][:, 0].long().tolist()
                for etype in edge_type_ids:
                    edge_types[etype] += 1
        elif 'num_nodes' in sample and 'num_edges' in sample:
            num_nodes_list.append(sample['num_nodes'])
            num_edges_list.append(sample['num_edges'])
    
    stats = {
        'total_graphs': len(data_list),
    }
    
    if num_nodes_list:
        stats.update({
            'avg_nodes': float(np.mean(num_nodes_list)),
            'std_nodes': float(np.std(num_nodes_list)),
            'min_nodes': int(np.min(num_nodes_list)),
            'max_nodes': int(np.max(num_nodes_list)),
        })
    
    if num_edges_list:
        stats.update({
            'avg_edges': float(np.mean(num_edges_list)),
            'std_edges': float(np.std(num_edges_list)),
            'min_edges': int(np.min(num_edges_list)),
            'max_edges': int(np.max(num_edges_list)),
        })
    
    # Token统计（基于预计算的token_ids）
    node_tokens = Counter()
    edge_tokens = Counter()
    
    for sample in data_list:
        if 'dgl_graph' in sample:
            g = sample['dgl_graph']
            
            # 统计节点tokens
            if 'node_token_ids' in g.ndata:
                node_token_ids = g.ndata['node_token_ids'].view(-1).tolist()
                for token in node_token_ids:
                    node_tokens[token] += 1
            
            # 统计边tokens
            if 'edge_token_ids' in g.edata:
                edge_token_ids = g.edata['edge_token_ids'].view(-1).tolist()
                for token in edge_token_ids:
                    edge_tokens[token] += 1
    
    # Token维度和唯一性统计
    stats['node_token_dim'] = 1  # 单个token维度
    stats['edge_token_dim'] = 1  # 单个token维度
    stats['unique_node_tokens'] = len(node_tokens)
    stats['unique_edge_tokens'] = len(edge_tokens)
    
    # 为了向后兼容，保留node_types/edge_types统计
    if node_types:
        stats['unique_node_types'] = len(node_types)
    else:
        stats['unique_node_types'] = stats['unique_node_tokens']
    
    if edge_types:
        stats['unique_edge_types'] = len(edge_types)
    else:
        stats['unique_edge_types'] = stats['unique_edge_tokens']
    
    return stats


def analyze_dataset(dataset_name: str, config: ProjectConfig) -> Dict[str, Any]:
    """分析单个数据集"""
    logger.info(f"🔍 分析数据集: {dataset_name}")
    
    try:
        # 创建数据加载器
        loader = get_dataloader(dataset_name, config)
        
        # 加载数据
        train_data, val_data, test_data, train_labels, val_labels, test_labels = loader.load_data()
        
        # 获取数据集元信息
        task_type = loader.get_dataset_task_type()
        num_classes = loader.get_num_classes()
        
        results = {
            'dataset_name': dataset_name,
            'task_type': task_type,
            'num_classes': num_classes,
        }
        
        # 分析图结构统计
        all_data = train_data + val_data + test_data
        graph_stats = analyze_graph_statistics(all_data)
        results.update(graph_stats)
        
        # 分析类别分布（仅对分类任务）
        if 'classification' in task_type.lower():
            results['class_balance'] = {}
            
            for split_name, labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:
                balance_stats = calculate_class_balance_metrics(labels, num_classes)
                results['class_balance'][split_name] = balance_stats
            
            # 总体类别分布
            all_labels = train_labels + val_labels + test_labels
            results['class_balance']['overall'] = calculate_class_balance_metrics(all_labels, num_classes)
        
        # 分析回归目标统计（仅对回归任务）
        elif 'regression' in task_type.lower():
            results['target_statistics'] = {}
            
            for split_name, labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:
                if labels:
                    # 处理多目标回归
                    if isinstance(labels[0], (list, tuple, np.ndarray)):
                        # 转换为numpy数组
                        label_array = np.array(labels)
                        stats = {
                            'num_targets': label_array.shape[1] if len(label_array.shape) > 1 else 1,
                            'mean': np.mean(label_array, axis=0).tolist(),
                            'std': np.std(label_array, axis=0).tolist(),
                            'min': np.min(label_array, axis=0).tolist(),
                            'max': np.max(label_array, axis=0).tolist(),
                        }
                    else:
                        # 单目标回归
                        label_array = np.array(labels)
                        stats = {
                            'num_targets': 1,
                            'mean': float(np.mean(label_array)),
                            'std': float(np.std(label_array)),
                            'min': float(np.min(label_array)),
                            'max': float(np.max(label_array)),
                        }
                    
                    results['target_statistics'][split_name] = stats
        
        logger.info(f"✅ {dataset_name} 分析完成")
        return results
        
    except Exception as e:
        logger.error(f"❌ 分析 {dataset_name} 时出错: {e}")
        return {'dataset_name': dataset_name, 'error': str(e)}


def generate_stats_report(results_list: List[Dict[str, Any]]) -> str:
    """生成类似DATASETS_STATS.md的统计报告"""
    report = "# 数据集统计总览\n\n"
    report += "本表统计当前已分析的数据集规模与特征维度：\n\n"
    
    # 表头
    report += "| 数据集 | 图数量 | 节点数均值 | 节点数Std | 节点min | 节点max | 边数均值 | 边数Std | 边min | 边max | 节点Token维度Dn | 边Token维度De | 节点Token唯一 | 边Token唯一 |\n"
    report += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    
    for result in results_list:
        if 'error' in result:
            continue
            
        name = result.get('dataset_name', 'Unknown')
        total_graphs = result.get('total_graphs', 0)
        avg_nodes = result.get('avg_nodes', 0)
        std_nodes = result.get('std_nodes', 0)
        min_nodes = result.get('min_nodes', 0)
        max_nodes = result.get('max_nodes', 0)
        avg_edges = result.get('avg_edges', 0)
        std_edges = result.get('std_edges', 0)
        min_edges = result.get('min_edges', 0)
        max_edges = result.get('max_edges', 0)
        node_token_dim = result.get('node_token_dim', 0)
        edge_token_dim = result.get('edge_token_dim', 0)
        # 优先使用新的token统计，回退到旧的类型统计
        unique_node_tokens = result.get('unique_node_tokens', result.get('unique_node_types', 0))
        unique_edge_tokens = result.get('unique_edge_tokens', result.get('unique_edge_types', 0))
        
        report += f"| {name} | {total_graphs} | {avg_nodes:.2f} | {std_nodes:.2f} | {min_nodes} | {max_nodes} | {avg_edges:.2f} | {std_edges:.2f} | {min_edges} | {max_edges} | {node_token_dim} | {edge_token_dim} | {unique_node_tokens} | {unique_edge_tokens} |\n"
    
    return report


def generate_balance_report(results_list: List[Dict[str, Any]]) -> str:
    """生成类似CLASS_BALANCE_REPORT.md的类别平衡报告"""
    report = "# 分类数据集类别分布与均衡性检查\n\n"
    report += "本文档统计当前已分析的分类数据集的类别分布与均衡性指标。\n\n"
    
    classification_datasets = [r for r in results_list if 'classification' in r.get('task_type', '').lower()]
    
    for result in classification_datasets:
        if 'error' in result or 'class_balance' not in result:
            continue
            
        name = result['dataset_name']
        num_classes = result.get('num_classes', 2)
        balance_data = result['class_balance']
        
        report += f"## {name.upper()} (K={num_classes})\n\n\n"
        
        # 各个split的统计
        for split in ['train', 'val', 'test', 'overall']:
            if split not in balance_data:
                continue
                
            stats = balance_data[split]
            if not stats:
                continue
                
            counts = stats.get('counts', [])
            proportions = stats.get('proportions', [])
            total = stats.get('total_samples', 0)
            ir = stats.get('ir', 1.0)
            cv = stats.get('cv', 0.0)
            h_norm = stats.get('h_norm', 1.0)
            gini_norm = stats.get('gini_norm', 1.0)
            kl_div = stats.get('kl_div', 0.0)
            
            split_name = {'train': 'Train', 'val': 'Val', 'test': 'Test', 'overall': 'Overall'}[split]
            
            report += f"- **{split_name}**: N={total}\n"
            
            if counts:
                counts_str = ", ".join([f"{i}: {c}" for i, c in enumerate(counts)])
                report += f"  - **Counts**: {counts_str}\n"
            
            if proportions:
                props_str = ", ".join([f"{i}: {p:.4f}" for i, p in enumerate(proportions)])
                report += f"  - **Proportions**: {props_str}\n"
            
            report += f"  - **Metrics**: H_norm={h_norm:.4f}, Gini_norm={gini_norm:.4f}, IR={ir}, CV={cv}, KL={kl_div:.4f}\n"
        
        report += "\n"
    
    # 全局总结表
    if classification_datasets:
        report += "## 全局总结\n\n\n"
        report += "| 数据集 | K | N_total | IR(overall) | CV(overall) | H_norm(overall) | KL | 最不平衡split | worst IR | worst H | 不平衡? |\n"
        report += "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|:---:|\n"
        
        for result in classification_datasets:
            if 'error' in result or 'class_balance' not in result:
                continue
                
            name = result['dataset_name']
            num_classes = result.get('num_classes', 2)
            balance_data = result['class_balance']
            
            overall_stats = balance_data.get('overall', {})
            total = overall_stats.get('total_samples', 0)
            ir = overall_stats.get('ir', 1.0)
            cv = overall_stats.get('cv', 0.0)
            h_norm = overall_stats.get('h_norm', 1.0)
            kl_div = overall_stats.get('kl_div', 0.0)
            
            # 找到最不平衡的split
            worst_ir = ir
            worst_h = h_norm
            worst_split = 'overall'
            
            for split in ['train', 'val', 'test']:
                if split in balance_data:
                    split_ir = balance_data[split].get('ir', 1.0)
                    split_h = balance_data[split].get('h_norm', 1.0)
                    if split_ir > worst_ir:
                        worst_ir = split_ir
                        worst_split = split
                        worst_h = split_h
            
            # 判断是否平衡（简单判断：IR < 1.5且H_norm > 0.95）
            balanced = "✅" if ir < 1.5 and h_norm > 0.95 else "❌"
            
            report += f"| {name} | {num_classes} | {total} | {ir:.4f} | {cv:.4f} | {h_norm:.4f} | {kl_div:.4f} | {worst_split} | {worst_ir:.4f} | {worst_h:.4f} | {balanced} |\n"
    
    return report


def main():
    """主函数"""
    logger.info("🚀 开始分析LRGB和MOLHIV数据集...")
    
    # 加载配置
    config = ProjectConfig()
    
    # 要分析的数据集
    datasets = ["molhiv", "peptides_func", "peptides_struct"]
    
    results = []
    for dataset_name in datasets:
        result = analyze_dataset(dataset_name, config)
        results.append(result)
    
    # 生成报告
    logger.info("📊 生成统计报告...")
    
    # 基础统计报告
    stats_report = generate_stats_report(results)
    stats_file = Path("LRGB_DATASETS_STATS.md")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(stats_report)
    logger.info(f"✅ 数据集统计报告已保存: {stats_file}")
    
    # 类别平衡报告
    balance_report = generate_balance_report(results)
    balance_file = Path("LRGB_CLASS_BALANCE_REPORT.md")
    with open(balance_file, "w", encoding="utf-8") as f:
        f.write(balance_report)
    logger.info(f"✅ 类别平衡报告已保存: {balance_file}")
    
    # 详细分析结果JSON
    json_file = Path("lrgb_analysis_results.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ 详细分析结果已保存: {json_file}")
    
    logger.info("🎉 分析完成!")
    
    # 打印简要统计
    logger.info("📋 数据集简要统计:")
    for result in results:
        if 'error' not in result:
            name = result['dataset_name']
            total = result.get('total_graphs', 0)
            task = result.get('task_type', 'unknown')
            logger.info(f"  {name}: {total} 图, 任务类型: {task}")


if __name__ == "__main__":
    main()
