"""
训练效率对比画图脚本
===================

绘制不同方法的训练效率对比图，包括：
- 序列化方法（DFS、Eulerian、CPP等）- 分别有无BPE压缩两种情况，柱子贴在一起
- 传统图神经网络方法（GCN、GraphGPS、Graph Mamba）
- 横轴：方法名称
- 纵轴：每个epoch的平均训练时间（秒）

使用统一的matplotlib样式设置
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 添加plot_utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from plot_utils import *


def plot_training_efficiency(csv_file: str, dataset_name: str = "QM9Hook", 
                            output_dir: str = None, show_plot: bool = False):
    """
    绘制训练效率对比图
    
    Args:
        csv_file: CSV数据文件路径
        dataset_name: 数据集名称
        output_dir: 输出目录（默认为当前目录）
        show_plot: 是否显示图表
    """
    # 设置matplotlib样式
    setup_matplotlib_style()
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"读取数据: {csv_file}")
    print(f"数据形状: {df.shape}")
    print("数据预览:")
    print(df.head())
    
    # 仅支持最新格式
    required_columns = {'method_name', 'method_group', 'epoch_time_seconds_wo_bpe', 'epoch_time_seconds_w_bpe', 'epoch_time_seconds', 'method_type'}
    if not required_columns.issubset(df.columns):
        missing = required_columns.difference(df.columns)
        raise ValueError(f"CSV文件缺少必要列: {sorted(list(missing))}")
    
    # 创建图表
    fig, ax = create_figure_with_style(figsize=(10,6))
    
    # 获取样式设置
    color_dict = get_color_dict()
    fontsize_dict = get_fontsize_dict()
    
    # 分组处理数据
    serialization_groups = {}
    gnn_methods = {}

    # 最新格式：w/o 与 w/ BPE 为列
    for _, row in df.iterrows():
        if row['method_type'] == 'graph_neural_network' or row['method_group'] == 'gnn':
            gnn_methods[row['method_name']] = {
                'time': row.get('epoch_time_seconds', np.nan),
                'type': row['method_type']
            }
        else:
            group = row['method_group']
            if group not in serialization_groups:
                serialization_groups[group] = {}
            wo = row.get('epoch_time_seconds_wo_bpe', np.nan)
            wb = row.get('epoch_time_seconds_w_bpe', np.nan)
            # 支持空字符串
            try:
                wo = float(wo) if str(wo).strip() != '' else np.nan
            except Exception:
                wo = np.nan
            try:
                wb = float(wb) if str(wb).strip() != '' else np.nan
            except Exception:
                wb = np.nan
            serialization_groups[group]['no_bpe'] = wo
            serialization_groups[group]['with_bpe'] = wb
    
    # 准备绘图数据（过滤无效值，避免留空与下标）
    def _is_valid(v):
        try:
            if v is None:
                return False
            if isinstance(v, float) and np.isnan(v):
                return False
            return float(v) > 0
        except Exception:
            return False

    group_names = []
    no_bpe_times = []
    with_bpe_times = []
    gnn_names = []
    gnn_times = []

    # 序列化方法分组（至少有一个有效值才纳入）
    for group, data in serialization_groups.items():
        no_bpe = data.get('no_bpe', np.nan)
        w_bpe = data.get('with_bpe', np.nan)
        if not (_is_valid(no_bpe) or _is_valid(w_bpe)):
            continue
        group_names.append(group)
        no_bpe_times.append(no_bpe)
        with_bpe_times.append(w_bpe)

    # GNN方法（仅保留有效值）
    for name, data in gnn_methods.items():
        t = data['time']
        if not _is_valid(t):
            continue
        gnn_names.append(name)
        gnn_times.append(t)
    
    # 计算柱子位置
    n_serialization = len(group_names)
    n_gnn = len(gnn_names)
    
    # 设置柱子宽度和间距
    bar_width = 0.35
    group_gap = 0.8
    method_gap = 1.0  # 序列化方法和GNN方法之间的间距
    
    # 序列化方法的x坐标
    serial_x = np.arange(n_serialization) * group_gap
    
    # GNN方法的x坐标
    gnn_start = (serial_x[-1] + method_gap) if n_serialization > 0 else 0
    gnn_x = np.arange(n_gnn)*group_gap + gnn_start
    
    # 绘制序列化方法的柱子（分组）
    # 将 NaN 替换为 0 以便绘图，但不显示数值标签
    plot_no_bpe = [0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in no_bpe_times]
    plot_w_bpe = [0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in with_bpe_times]
    plot_gnn = [0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in gnn_times]

    bars_no_bpe = ax.bar(serial_x - bar_width/2, plot_no_bpe, bar_width, 
                        color=color_dict['navy'], alpha=0.8, 
                        label='w/o BPE',  linewidth=1)
    bars_with_bpe = ax.bar(serial_x + bar_width/2, plot_w_bpe, bar_width, 
                          color=color_dict['blue'], alpha=0.8, 
                          label='w/ BPE',  linewidth=1)
    
    # 绘制GNN方法的柱子
    bars_gnn = ax.bar(gnn_x, plot_gnn, bar_width * 1, 
                     color=color_dict['red'], alpha=0.8, 
                     label='GNN', linewidth=1)
    
    # 设置图表样式
    setup_plot_style(ax, 
                    xlabel='Method', 
                    ylabel='Epoch Time (s)',)
                    # title=f'{dataset_name} Dataset - Training Efficiency Comparison')
    
    # 设置x轴标签
    all_x_pos = list(serial_x) + list(gnn_x)
    all_labels = group_names + gnn_names
    
    ax.set_xticks(all_x_pos)
    ax.set_xticklabels(all_labels,fontsize=fontsize_dict['ticks'])  # 不旋转标签
    
    # 添加数值标签
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            if value is not None and not (isinstance(value, float) and np.isnan(value)) and value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.1f}s',
                       ha='center', va='bottom', fontsize=fontsize_dict['ticks']-2)
    
    add_value_labels(bars_no_bpe, no_bpe_times)
    add_value_labels(bars_with_bpe, with_bpe_times)
    add_value_labels(bars_gnn, gnn_times)
    
    # 创建图例
    ax.legend(loc='upper left', fontsize=fontsize_dict['legend'])
    
    # 设置y轴从0开始
    all_times = [v for v in (no_bpe_times + with_bpe_times + gnn_times) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    ymax = max(all_times) * 1.15 if all_times else 1.0
    ax.set_ylim(0, ymax)
    
    # 添加加速比信息
    # if gnn_times and (no_bpe_times or with_bpe_times):
    #     max_gnn_time = max(gnn_times)
    #     serialization_times = [t for t in with_bpe_times if t > 0]
    #     if serialization_times:
    #         avg_serialization_time = sum(serialization_times) / len(serialization_times)
    #         speedup = max_gnn_time / avg_serialization_time
    #         ax.text(0.02, 0.98, f'Average Speedup: {speedup:.1f}x', 
    #                transform=ax.transAxes, fontsize=fontsize_dict['ticks'],
    #                verticalalignment='top', 
    #                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 保存图表
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    save_plot(fig, f'{dataset_name}_training_efficiency', output_dir)
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def batch_plot_training_efficiency(data_dir: str = None, show_plots: bool = False):
    """
    批量绘制多个数据集的训练效率对比图
    
    Args:
        data_dir: 数据目录（默认为当前目录）
        show_plots: 是否显示图表
    """
    if data_dir is None:
        data_dir = os.path.dirname(__file__)
    
    # 查找所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # 为每个CSV文件绘制图表
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        
        # 从文件名提取数据集名称
        dataset_name = csv_file.replace('_train_efficiency.csv', '').replace('.csv', '')
        dataset_name = dataset_name.upper()
        
        print(f"\n绘制 {dataset_name} 数据集的训练效率对比图...")
        
        try:
            plot_training_efficiency(csv_path, dataset_name, data_dir, show_plots)
            print(f"✓ 成功生成 {dataset_name} 的图表")
        except Exception as e:
            print(f"✗ 绘制 {dataset_name} 图表时出错: {e}")


if __name__ == "__main__":
    # 设置matplotlib样式
    setup_matplotlib_style()
    
    # 批量处理当前目录下的所有CSV文件
    print("开始批量绘制训练效率对比图...")
    batch_plot_training_efficiency(show_plots=True)
    
    print("\n所有图表绘制完成！")