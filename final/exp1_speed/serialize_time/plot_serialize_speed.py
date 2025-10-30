"""
序列化速度对比画图脚本
===================

绘制不同序列化方法的速度对比图：
- 横轴：序列化方法
- 纵轴：每秒序列化的图数量 (graphs/second)
- 每个方法一个柱子

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


def plot_serialization_speed(csv_file: str, dataset_name: str = "QM9Hook", 
                            output_dir: str = None, show_plot: bool = False):
    """
    绘制序列化速度对比图
    
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
    
    # 检查必要的列
    required_columns = ['serialization_method', 'graphs_per_second']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要的列: {col}")
    
    # 创建图表
    fig, ax = create_figure_with_style(figsize=(12, 7))
    
    # 获取样式设置
    color_dict = get_color_dict()
    fontsize_dict = get_fontsize_dict()
    
    # 准备数据
    methods = df['serialization_method'].tolist()
    speeds = df['graphs_per_second'].tolist()
    
    # 为每个方法分配颜色
    colors = [get_color_by_index(i) for i in range(len(methods))]
    
    # 绘制柱状图
    bars = ax.bar(methods, speeds, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # 设置图表样式
    setup_plot_style(ax, 
                    xlabel='Serialization Method', 
                    ylabel='Serialization Speed (graphs/second)',
                    title=f'{dataset_name} Dataset - Serialization Speed Comparison')
    
    # 在柱子上添加数值标签
    for bar, speed in zip(bars, speeds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{speed:.1f}',
                ha='center', va='bottom', fontsize=fontsize_dict['ticks'])
    
    # 不旋转x轴标签
    # plt.xticks(rotation=45, ha='right')
    
    # 设置y轴从0开始
    ax.set_ylim(0, max(speeds) * 1.15)
    
    # 保存图表
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    save_plot(fig, f'{dataset_name}_serialization_speed', output_dir)
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def batch_plot_serialization_speed(data_dir: str = None, show_plots: bool = False):
    """
    批量绘制多个数据集的序列化速度对比图
    
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
        dataset_name = csv_file.replace('_serialize_speed.csv', '').replace('.csv', '')
        dataset_name = dataset_name.upper()
        
        print(f"\n绘制 {dataset_name} 数据集的序列化速度对比图...")
        
        try:
            plot_serialization_speed(csv_path, dataset_name, data_dir, show_plots)
            print(f"✓ 成功生成 {dataset_name} 的图表")
        except Exception as e:
            print(f"✗ 绘制 {dataset_name} 图表时出错: {e}")


if __name__ == "__main__":
    # 设置matplotlib样式
    setup_matplotlib_style()
    
    # 批量处理当前目录下的所有CSV文件
    print("开始批量绘制序列化速度对比图...")
    batch_plot_serialization_speed(show_plots=True)
    
    print("\n所有图表绘制完成！")
