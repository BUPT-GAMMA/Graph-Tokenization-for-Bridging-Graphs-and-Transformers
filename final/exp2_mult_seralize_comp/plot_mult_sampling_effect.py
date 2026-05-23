"""
多重采样效果对比画图脚本
========================

用于绘制不同多重采样次数对序列化效率影响的曲线图，包含：
- 左图：BPE压缩关闭 (bpe=false) 的各方法曲线
- 右图：BPE压缩开启 (bpe=true) 的各方法曲线
- 每个图包含欧拉、feuler、cpp、fcpp、topo五个方法的曲线

数据格式：CSV文件包含method, mult, bpe, pk四列
- method: 序列化方法名
- mult: 多重采样次数
- bpe: BPE状态 (true/false)
- pk: 性能指标值

数据来源：同目录下的CSV文件
输出文件名：{dataset}_mult_sampling_effect_comparison.jpg

使用方法：
python plot_mult_sampling_effect.py                          # 处理所有CSV文件
python plot_mult_sampling_effect.py --datasets zinc         # 只处理zinc数据集
python plot_mult_sampling_effect.py --input_dir /path/to/csvs  # 指定输入目录
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
from typing import Dict, List, Tuple
from scipy import interpolate

# 添加plot_utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from plot_utils import (
    setup_matplotlib_style, get_color_dict, get_fontsize_dict,
    create_figure_with_style, setup_plot_style, save_plot
)


def load_mult_sampling_data(directory: str) -> Dict[str, pd.DataFrame]:
    """
    加载目录中的所有CSV文件，按数据集分组

    Args:
        directory: 目录路径

    Returns:
        Dict[str, pd.DataFrame]: {数据集名称: DataFrame}
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and f != 'plot_mult_sampling_effect.py']

    if not csv_files:
        raise FileNotFoundError(f"在目录 {directory} 中未找到CSV文件")

    # 按数据集分组
    dataset_data = {}

    for csv_file in csv_files:
        try:
            # 从文件名提取数据集名称（去掉.csv扩展名）
            dataset_name = csv_file[:-4]  # 移除.csv

            csv_path = os.path.join(directory, csv_file)
            df = pd.read_csv(csv_path)

            # 验证必要的列是否存在
            required_cols = ['method', 'mult', 'bpe', 'pk']
            if not all(col in df.columns for col in required_cols):
                print(f"警告: {csv_file} 缺少必要的列，跳过")
                continue

            dataset_data[dataset_name] = df

        except Exception as e:
            print(f"跳过无法处理的文件: {csv_file} - {e}")
            continue

    return dataset_data


def plot_mult_sampling_effect(df: pd.DataFrame,
                             dataset_name: str,
                             save_dir: str = None) -> None:
    """
    绘制多重采样效果对比图

    Args:
        df: 包含所有数据的DataFrame
        dataset_name: 数据集名称
        save_dir: 保存目录，为None时使用当前目录
    """
    # 设置matplotlib样式
    setup_matplotlib_style()
    colors = get_color_dict()
    fontsize = get_fontsize_dict()

    # 使用更大的字体
    fontsize = {
        'legend': 20,
        'label': 18,
        'ticks': 16,
        'title': 20
    }

    # 创建图形 - 两个子图并排
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 获取数据中实际存在的序列化方法
    available_methods = df['method'].unique().tolist()

    # 定义方法标签映射（暂时不画dfs，因为与其他方法差距太大）
    method_label_map = {
        'eulerian': 'Euler',
        'feuler': 'FEuler',
        'cpp': 'CPP',
        'fcpp': 'FCPP'
        # 'dfs': 'DFS'  # 暂时不画dfs
    }

    # 过滤出有标签映射的方法
    target_methods = [m for m in available_methods if m in method_label_map]
    method_labels = [method_label_map[m] for m in target_methods]

    # 颜色映射（循环使用）
    method_colors = []
    color_list = [colors['blue'], colors['green'], colors['red'], colors['orange'], colors['purple']]
    for i in range(len(target_methods)):
        method_colors.append(color_list[i % len(color_list)])

    # 绘制左图：BPE = False
    plot_single_bpe_effect(ax1, df, target_methods, method_labels, method_colors,
                          'Without BPE', False, fontsize)

    # 绘制右图：BPE = True
    plot_single_bpe_effect(ax2, df, target_methods, method_labels, method_colors,
                          'With BPE', True, fontsize)

    # 设置整体标题
    fig.suptitle(f'{dataset_name.upper()} Dataset - Multi-Sampling Effect Comparison',
                fontsize=fontsize['title'], y=0.98)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f'{dataset_name}_mult_sampling_effect_comparison'
    save_plot(fig, filename, save_dir)

    plt.show()


def plot_single_bpe_effect(ax, df: pd.DataFrame, target_methods: List[str],
                          method_labels: List[str], method_colors: List[str],
                          title: str, bpe_state: bool, fontsize: Dict[str, int]) -> None:
    """
    绘制单个BPE状态下的多重采样效果图

    Args:
        ax: matplotlib轴对象
        df: 包含所有数据的DataFrame
        target_methods: 要绘制的序列化方法列表
        method_labels: 方法标签列表
        method_colors: 方法颜色列表
        title: 图标题
        bpe_state: BPE状态 (True/False)
        fontsize: 字体大小字典
    """
    # 筛选出对应BPE状态的数据
    bpe_data = df[df['bpe'] == bpe_state]

    if bpe_data.empty:
        ax.text(0.5, 0.5, f'No data for BPE={bpe_state}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=fontsize['title'])
        return

    # 获取所有可用的多重采样次数并排序
    mult_nums = sorted(bpe_data['mult'].unique())

    # 为每个方法绘制曲线
    for i, (method, label) in enumerate(zip(target_methods, method_labels)):
        method_data = bpe_data[bpe_data['method'] == method]

        if not method_data.empty:
            # 按多重采样次数排序
            method_data = method_data.sort_values('mult')

            # 提取数据
            mult_values = method_data['mult'].tolist()
            pk_values = method_data['pk'].tolist()

            # 使用线性插值创建平滑曲线，保持原始趋势
            if len(mult_values) > 1:
                # 创建插值函数，使用线性插值严格保持数据趋势
                f = interpolate.interp1d(mult_values, pk_values, kind='linear')

                # 创建更密集的x值用于平滑曲线
                mult_dense = np.linspace(min(mult_values), max(mult_values), 50)
                pk_smooth = f(mult_dense)

                ax.plot(mult_dense, pk_smooth,
                       color=method_colors[i], linewidth=3,
                       label=label, alpha=0.9)
            else:
                # 如果只有一个点，直接绘制
                ax.plot(mult_values, pk_values,
                       color=method_colors[i], linewidth=3,
                       label=label, alpha=0.9)

    # 设置轴标签和标题
    ax.set_xlabel('Multi-Sampling Count', fontsize=fontsize['label'])
    ax.set_ylabel('PK Value', fontsize=fontsize['label'])
    ax.set_title(title, fontsize=fontsize['title'])

    # 设置x轴刻度
    ax.set_xticks(mult_nums)

    # 移除y轴数值标签
    ax.set_yticklabels([])

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 添加图例
    ax.legend(loc='upper right', fontsize=fontsize['legend']-2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多重采样效果对比画图脚本")
    parser.add_argument("--datasets", type=str, nargs='+',
                       help="要处理的数据集名称，如 --datasets zinc")
    parser.add_argument("--input_dir", type=str,
                       help="输入CSV文件的目录，默认使用脚本所在目录")

    args = parser.parse_args()

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir if args.input_dir else script_dir

    try:
        # 加载所有CSV文件
        dataset_data = load_mult_sampling_data(input_dir)

        if not dataset_data:
            print(f"在目录 {input_dir} 中未找到有效的CSV文件")
            return

        print(f"找到 {len(dataset_data)} 个数据集:")
        for dataset_name, df in dataset_data.items():
            methods = df['method'].unique().tolist()
            mult_values = sorted(df['mult'].unique().tolist())
            bpe_states = df['bpe'].unique().tolist()
            print(f"  {dataset_name}: 方法 {methods}, 多重采样次数 {mult_values}, BPE状态 {bpe_states}")

        # 处理每个数据集
        for dataset_name, df in dataset_data.items():
            print(f"\n处理数据集: {dataset_name}")

            # 检查数据集是否有足够的数据（至少需要2个多重采样点）
            mult_counts = df['mult'].nunique()
            if mult_counts < 2:
                print(f"  跳过: 数据集 {dataset_name} 没有足够的采样点数据")
                continue

            # 检查是否同时有BPE true和false的数据
            bpe_counts = df['bpe'].nunique()
            if bpe_counts < 2:
                print(f"  跳过: 数据集 {dataset_name} 没有同时包含BPE true和false的数据")
                continue

            # 绘制对比图
            plot_mult_sampling_effect(df, dataset_name, script_dir)
            print(f"  完成绘制: {dataset_name}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
