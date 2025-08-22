#!/usr/bin/env python3
"""
聚合训练结果脚本
从log目录收集模型在不同数据集上的性能指标，生成8个CSV表格
"""

import json
import os
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_directory_name(dir_name):
    """
    解析目录名，提取数据集、方法、BPE设置等信息
    例如：aqsolaqsol_cpp_all_noaug -> dataset=aqsol, method=cpp, bpe=all
    例如：peptides_func_cpp_all_aug -> dataset=peptides_func, method=cpp, bpe=all
    """
    parts = dir_name.split('_')
    
    # 特殊处理peptides数据集
    if parts[0] == 'peptides' and len(parts) > 1 and parts[1] in ['func', 'struct']:
        dataset = f"{parts[0]}_{parts[1]}"
        method = parts[2]
        bpe = parts[3]
        return dataset, method, bpe
    
    # 提取数据集名（去除重复）
    dataset_double = parts[0]
    if len(dataset_double) % 2 == 0:
        mid = len(dataset_double) // 2
        if dataset_double[:mid] == dataset_double[mid:]:
            dataset = dataset_double[:mid]
        else:
            dataset = dataset_double
    else:
        dataset = dataset_double
    
    # 提取方法
    method = parts[1]
    
    # 提取BPE设置
    bpe = parts[2]
    
    return dataset, method, bpe


def load_metrics(metrics_file):
    """加载指标文件"""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_metric_value(metrics, task_type, aggregation_mode):
    """
    根据任务类型和聚合模式提取指标值
    """
    if not metrics or 'test' not in metrics:
        return None
    
    test_data = metrics['test']
    
    # 获取聚合数据
    if aggregation_mode in ['avg', 'learned']:
        if 'by_aggregation' not in test_data or aggregation_mode not in test_data['by_aggregation']:
            return None
        agg_data = test_data['by_aggregation'][aggregation_mode]
    elif aggregation_mode == 'best':
        if 'by_aggregation' not in test_data or 'best' not in test_data['by_aggregation']:
            return None
        agg_data = test_data['by_aggregation']['best']
    else:  # best_of_fair
        # 对于best_of_fair，我们需要比较avg和learned，取较好的值
        if 'by_aggregation' not in test_data:
            return None
        avg_data = test_data['by_aggregation'].get('avg')
        learned_data = test_data['by_aggregation'].get('learned')
        if not avg_data or not learned_data:
            return None
        
        # 根据任务类型选择指标并比较
        if task_type == 'regression':
            avg_val = avg_data.get('mae')
            learned_val = learned_data.get('mae')
            if avg_val is None or learned_val is None:
                return None
            # MAE越小越好
            agg_data = avg_data if avg_val <= learned_val else learned_data
        else:  # classification
            dataset = metrics.get('dataset', '')
            if dataset == 'molhiv':
                avg_val = avg_data.get('roc_auc')
                learned_val = learned_data.get('roc_auc')
            elif dataset == 'peptides_func':
                avg_val = avg_data.get('ap')
                learned_val = learned_data.get('ap')
            else:
                avg_val = avg_data.get('accuracy')
                learned_val = learned_data.get('accuracy')
            
            if avg_val is None or learned_val is None:
                return None
            # 这些指标都是越大越好
            agg_data = avg_data if avg_val >= learned_val else learned_data
    
    # 提取具体指标值
    if task_type == 'regression':
        return agg_data.get('mae')
    else:  # classification
        dataset = metrics.get('dataset', '')
        if dataset == 'molhiv':
            return agg_data.get('roc_auc')
        elif dataset == 'peptides_func':
            return agg_data.get('ap')
        else:
            return agg_data.get('accuracy')


def collect_results(log_dir, group_names, prefix_names):
    """
    收集指定group(s)和prefix(es)的所有结果
    支持多个group和多个prefix，会检查重叠并合并结果
    """
    if isinstance(group_names, str):
        group_names = [group_names]
    if isinstance(prefix_names, str):
        prefix_names = [prefix_names]
    
    all_results = defaultdict(lambda: defaultdict(dict))
    experiment_tracking = {}  # 跟踪每个实验的来源(group, prefix)
    found_experiments = 0
    
    for group_name in group_names:
        group_path = Path(log_dir) / group_name
        if not group_path.exists():
            print(f"Group directory not found: {group_path}")
            continue
        
        print(f"Processing group: {group_name}")
        
        # 遍历所有实验目录
        for exp_dir in group_path.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # 解析目录名
            dataset, method, bpe = parse_directory_name(exp_dir.name)
            
            # 为每个前缀尝试加载结果
            for prefix_name in prefix_names:
                # 创建实验唯一标识（包含前缀）
                exp_id = f"{dataset}_{method}_{bpe}_{prefix_name}"
                
                # 检查是否有重叠实验
                if exp_id in experiment_tracking:
                    print(f"Warning: Overlapping experiment found: {exp_id}")
                    print(f"  Previous: {experiment_tracking[exp_id]}")
                    print(f"  Current: {group_name}")
                    print(f"  Skipping current to avoid duplication")
                    continue
                
                # 构建metrics文件路径
                metrics_file = exp_dir / dataset / method / f"{prefix_name}_finetune" / "finetune_metrics.json"
                
                if not metrics_file.exists():
                    # 对于多前缀，不打印缺失警告，因为很常见
                    continue
                
                # 加载metrics
                metrics = load_metrics(metrics_file)
                if not metrics:
                    print(f"Failed to load metrics: {metrics_file}")
                    continue
                
                # 记录实验来源
                experiment_tracking[exp_id] = f"{group_name}:{prefix_name}"
                found_experiments += 1
                
                # 确定任务类型
                task = metrics.get('task', '')
                if task in ['regression','multi_target_regression']:
                    task_type = 'regression'
                elif task in ['classification', 'multi_label_classification']:
                    task_type = 'classification'
                else:
                    print(f"Unknown task type: {task} for {dataset}")
                    continue
                
                # 为每种聚合模式提取指标
                for agg_mode in ['best', 'avg', 'learned', 'best_of_fair']:
                    metric_value = get_metric_value(metrics, task_type, agg_mode)
                    if metric_value is not None:
                        model_key = f"{method}_{bpe}"
                        all_results[(task_type, agg_mode)][dataset][model_key] = metric_value
    
    combined_group_name = "_".join(group_names) if len(group_names) > 1 else group_names[0]
    combined_prefix_name = "_".join(prefix_names) if len(prefix_names) > 1 else prefix_names[0]
    
    print(f"Found {found_experiments} experiments across {len(group_names)} groups and {len(prefix_names)} prefixes")
    
    return all_results, combined_group_name, combined_prefix_name


def create_tables(results, group_name, prefix_name):
    """
    创建CSV表格
    """
    # 定义数据集顺序
    regression_datasets = ['qm9', 'zinc', 'aqsol', 'peptides_struct']
    classification_datasets = ['colors3', 'proteins', 'synthetic', 'mutagenicity', 
                             'coildel', 'dblp', 'dd', 'twitter', 'molhiv', 'peptides_func']
    
    # 定义模型排序 (method, bpe)
    method_order = ['cpp', 'eulerian', 'fcpp', 'feuler', 'topo', 'smiles']
    bpe_order = ['all', 'gaussian', 'random', 'raw']
    
    def get_model_order_key(model_key):
        method, bpe = model_key.split('_')
        method_idx = method_order.index(method) if method in method_order else len(method_order)
        bpe_idx = bpe_order.index(bpe) if bpe in bpe_order else len(bpe_order)
        return (method_idx, bpe_idx)
    
    tables = {}
    
    # 为每种任务类型和聚合模式创建表格
    for task_type in ['regression', 'classification']:
        datasets = regression_datasets if task_type == 'regression' else classification_datasets
        metric_name = 'MAE' if task_type == 'regression' else 'Acc'
        
        for agg_mode in ['best', 'avg', 'learned', 'best_of_fair']:
            key = (task_type, agg_mode)
            if key not in results:
                continue
            
            # 收集所有模型
            all_models = set()
            for dataset in datasets:
                if dataset in results[key]:
                    all_models.update(results[key][dataset].keys())
            
            # 按规定顺序排序模型
            sorted_models = sorted(all_models, key=get_model_order_key)
            
            # 创建DataFrame
            data = []
            for model in sorted_models:
                row = {'Model': model}
                for dataset in datasets:
                    value = results[key][dataset].get(model)
                    if value is not None:
                        row[dataset] = f"{value:.4f}"
                    else:
                        row[dataset] = "N/A"
                data.append(row)
            
            if data:  # 只有当有数据时才创建表格
                df = pd.DataFrame(data)
                table_name = f"{metric_name}_{agg_mode}"
                tables[table_name] = df
    
    return tables


def save_tables(tables, output_dir, group_name, prefix_name):
    """
    保存表格到CSV文件
    """
    output_path = Path(output_dir) / group_name / prefix_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    for table_name, df in tables.items():
        csv_file = output_path / f"{table_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='聚合训练结果')
    parser.add_argument('--log_dir', default='log', help='日志目录路径')
    parser.add_argument('--group', nargs='+', required=True, help='实验组名，支持多个，如818_noaug 820_new_lrgb_ogbg')
    parser.add_argument('--prefix', nargs='+', required=True, help='实验前缀，支持多个，如PnFn PaFa')
    parser.add_argument('--output_dir', default='agg', help='输出目录')
    
    args = parser.parse_args()
    
    print(f"Processing groups: {args.group}, prefixes: {args.prefix}")
    
    # 收集结果
    results, group_name, prefix_name = collect_results(args.log_dir, args.group, args.prefix)
    
    if not results:
        print("No results collected!")
        return
    
    print(f"Collected results for {len(results)} task/aggregation combinations")
    
    # 创建表格
    tables = create_tables(results, group_name, prefix_name)
    
    if not tables:
        print("No tables created!")
        return
    
    print(f"Created {len(tables)} tables")
    
    # 保存表格
    save_tables(tables, args.output_dir, group_name, prefix_name)
    
    print("Done!")


if __name__ == "__main__":
    main()
