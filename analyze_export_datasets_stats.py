#!/usr/bin/env python3
"""
导出数据集图统计分析脚本
========================

分析所有导出数据集中图的节点数、边数统计信息，包括：
- 最大值、最小值
- 均值、方差、标准差
- 3σ覆盖率（99.7%置信区间）
- 导出为CSV文件
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_exported_dataset(filepath: Path) -> Dict[str, Any]:
    """加载导出的数据集"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_graph_stats(graphs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算图统计信息
    
    Args:
        graphs: 图数据列表
        
    Returns:
        统计信息字典
    """
    num_nodes_list = [graph['num_nodes'] for graph in graphs]
    num_edges_list = [len(graph['src']) for graph in graphs]
    
    def compute_stats(values: List[int], prefix: str) -> Dict[str, float]:
        """计算统计值"""
        arr = np.array(values)
        mean_val = np.mean(arr)
        std_val = np.std(arr, ddof=1)  # 样本标准差
        var_val = np.var(arr, ddof=1)  # 样本方差
        
        # 3σ区间边界
        lower_3sigma = mean_val - 3 * std_val
        upper_3sigma = mean_val + 3 * std_val
        
        # 3σ覆盖率（在区间内的样本比例）
        in_3sigma = np.sum((arr >= lower_3sigma) & (arr <= upper_3sigma))
        coverage_3sigma = in_3sigma / len(arr)
        
        return {
            f'{prefix}_count': len(arr),
            f'{prefix}_min': int(np.min(arr)),
            f'{prefix}_max': int(np.max(arr)),
            f'{prefix}_mean': float(mean_val),
            f'{prefix}_std': float(std_val),
            f'{prefix}_var': float(var_val),
            f'{prefix}_3sigma_lower': float(lower_3sigma),
            f'{prefix}_3sigma_upper': float(upper_3sigma),
            f'{prefix}_3sigma_coverage': float(coverage_3sigma),
        }
    
    stats = {}
    stats.update(compute_stats(num_nodes_list, 'nodes'))
    stats.update(compute_stats(num_edges_list, 'edges'))
    
    # 额外计算一些有用的统计量
    stats['avg_degree'] = float(np.mean([2 * edges / nodes for edges, nodes in zip(num_edges_list, num_nodes_list)]))
    stats['density_mean'] = float(np.mean([edges / (nodes * (nodes - 1) / 2) if nodes > 1 else 0 
                                         for edges, nodes in zip(num_edges_list, num_nodes_list)]))
    
    return stats

def analyze_all_datasets(export_dir: Path) -> List[Dict[str, Any]]:
    """
    分析所有导出数据集
    
    Args:
        export_dir: 导出目录路径
        
    Returns:
        所有数据集的统计信息列表
    """
    results = []
    
    # 获取所有导出文件
    export_files = list(export_dir.glob("*_export.pkl"))
    export_files.sort()  # 按文件名排序
    
    print(f"找到 {len(export_files)} 个导出数据集文件")
    
    for filepath in export_files:
        dataset_name = filepath.stem.replace('_export', '')
        print(f"正在分析: {dataset_name}")
        
        try:
            # 加载数据
            data = load_exported_dataset(filepath)
            graphs = data['graphs']
            labels = data['labels']
            splits = data['splits']
            
            # 基本信息
            total_graphs = len(graphs)
            train_size = len(splits['train'])
            val_size = len(splits['val']) 
            test_size = len(splits['test'])
            
            # 计算图统计
            graph_stats = compute_graph_stats(graphs)
            
            # 标签类型分析
            if isinstance(labels[0], dict):
                label_type = "multi_property_regression"
                num_targets = len(labels[0])
            elif isinstance(labels[0], list):
                if isinstance(labels[0][0], (int, np.integer)):
                    label_type = "multi_label_classification" 
                    num_targets = len(labels[0])
                else:
                    label_type = "multi_target_regression"
                    num_targets = len(labels[0])
            elif isinstance(labels[0], (int, np.integer)):
                label_type = "classification"
                unique_labels = len(set(labels))
                num_targets = unique_labels
            elif isinstance(labels[0], (float, np.floating)):
                label_type = "regression"
                num_targets = 1
            else:
                label_type = "unknown"
                num_targets = None
            
            # 文件大小（MB）
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # 组合结果
            result = {
                'dataset': dataset_name,
                'total_graphs': total_graphs,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size,
                'label_type': label_type,
                'num_targets': num_targets,
                'file_size_mb': round(file_size_mb, 2),
                **graph_stats
            }
            
            results.append(result)
            print(f"  ✓ {dataset_name}: {total_graphs}图, "
                  f"节点{graph_stats['nodes_min']}-{graph_stats['nodes_max']}(avg={graph_stats['nodes_mean']:.1f}), "
                  f"边{graph_stats['edges_min']}-{graph_stats['edges_max']}(avg={graph_stats['edges_mean']:.1f})")
            
        except Exception as e:
            print(f"  ✗ {dataset_name}: 分析失败 - {e}")
            # 添加错误记录
            results.append({
                'dataset': dataset_name,
                'error': str(e),
                **{k: None for k in ['total_graphs', 'train_size', 'val_size', 'test_size', 
                                   'label_type', 'num_targets', 'file_size_mb']}
            })
    
    return results

def export_to_csv(results: List[Dict[str, Any]], output_path: Path):
    """导出结果为CSV文件"""
    df = pd.DataFrame(results)
    
    # 重新排列列的顺序，将重要信息放在前面
    important_cols = [
        'dataset', 'total_graphs', 'label_type', 'num_targets',
        'train_size', 'val_size', 'test_size', 'file_size_mb'
    ]
    
    node_cols = [col for col in df.columns if col.startswith('nodes_')]
    edge_cols = [col for col in df.columns if col.startswith('edges_')]
    other_cols = [col for col in df.columns if col not in important_cols + node_cols + edge_cols]
    
    column_order = important_cols + node_cols + edge_cols + other_cols
    column_order = [col for col in column_order if col in df.columns]  # 过滤不存在的列
    
    df_ordered = df[column_order]
    
    # 保存CSV
    df_ordered.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n📊 结果已保存到: {output_path}")
    
    # 显示摘要统计
    print(f"\n📈 数据集摘要:")
    print(f"  总数据集数: {len(results)}")
    if 'total_graphs' in df.columns:
        successful = df[df['total_graphs'].notna()]
        print(f"  成功分析: {len(successful)}")
        print(f"  总图数: {successful['total_graphs'].sum():,}")
        print(f"  平均图数: {successful['total_graphs'].mean():.0f}")
    
    print(f"\n📂 导出文件列:")
    for result in results:
        status = "✓" if 'error' not in result else "✗"
        graphs = result.get('total_graphs', 'N/A')
        print(f"  {status} {result['dataset']}: {graphs} 图")

def main():
    """主函数"""
    print("🔍 导出数据集图统计分析器")
    print("=" * 50)
    
    # 设置路径
    script_dir = Path(__file__).parent
    export_dir = script_dir / "export_system" / "exported"
    output_file = script_dir / "export_datasets_graph_stats.csv"
    
    # 检查导出目录
    if not export_dir.exists():
        print(f"❌ 导出目录不存在: {export_dir}")
        return
    
    # 分析所有数据集
    print(f"📂 扫描导出目录: {export_dir}")
    results = analyze_all_datasets(export_dir)
    
    if not results:
        print("❌ 没有找到可分析的数据集")
        return
    
    # 导出结果
    export_to_csv(results, output_file)
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
