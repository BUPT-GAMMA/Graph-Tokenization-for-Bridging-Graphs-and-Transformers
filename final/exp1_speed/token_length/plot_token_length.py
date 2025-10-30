"""
序列化长度对比画图脚本
========================

用于绘制不同序列化方法的长度对比图，包含：
- 原始总序列长度 (K tokens)
- BPE压缩后总序列长度 (K tokens)
- 压缩比 (越高越好)

支持多重采样对比：
- 可以同时比较同一个数据集下不同多重采样次数的结果
- 多重采样信息从CSV文件名中自动解析
- 支持选择特定的多重采样次数进行比较

数据来源：同目录下的CSV文件，支持新旧两种文件名格式
输出文件名不包含多重采样信息（展示数据集层面的方法对比）

使用方法：
python plot_token_length.py                          # 处理所有CSV文件
python plot_token_length.py --mult 1 5             # 只比较多重采样1和5的结果
python plot_token_length.py --datasets qm9test     # 只处理qm9test数据集
python plot_token_length.py --input_dir /path/to/csvs  # 指定输入目录
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import re
from typing import Dict, List, Tuple

# 添加plot_utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from plot_utils import (
    setup_matplotlib_style, get_color_dict, get_fontsize_dict,
    create_figure_with_style, setup_plot_style, save_plot
)


def load_data(csv_file: str) -> pd.DataFrame:
    """
    加载CSV数据文件

    Args:
        csv_file: CSV文件路径

    Returns:
        加载的DataFrame
    """
    return pd.read_csv(csv_file)


def parse_filename(filename: str) -> Tuple[str, int]:
    """
    从CSV文件名中解析数据集名称和多重采样次数

    Args:
        filename: CSV文件名（如 "qm9test_mult5_token_length.csv"）

    Returns:
        Tuple[str, int]: (数据集名称, 多重采样次数)
    """
    # 使用正则表达式匹配文件名模式
    pattern = r'^([^_]+)_mult(\d+)_token_length\.csv$'
    match = re.match(pattern, filename)

    if match:
        dataset_name = match.group(1)
        mult_num = int(match.group(2))
        return dataset_name, mult_num
    else:
        # 如果不匹配新格式，尝试旧格式
        if filename.endswith('_token_length.csv'):
            dataset_name = filename[:-len('_token_length.csv')]
            return dataset_name, 1  # 默认1
        else:
            raise ValueError(f"无法解析文件名: {filename}")


def load_multiple_csv_files(directory: str, mult_values: List[int] = None) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    加载目录中的所有CSV文件，按数据集和多重采样次数分组

    Args:
        directory: 目录路径
        mult_values: 要包含的多重采样次数列表，为None时包含所有

    Returns:
        Dict[str, Dict[int, pd.DataFrame]]: {数据集名称: {多重采样次数: DataFrame}}
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('_token_length.csv')]

    if not csv_files:
        raise FileNotFoundError(f"在目录 {directory} 中未找到CSV文件")

    # 按数据集分组
    dataset_data = {}

    for csv_file in csv_files:
        try:
            dataset_name, mult_num = parse_filename(csv_file)

            # 如果指定了多重采样次数列表，只处理指定的
            if mult_values is not None and mult_num not in mult_values:
                continue

            if dataset_name not in dataset_data:
                dataset_data[dataset_name] = {}

            csv_path = os.path.join(directory, csv_file)
            df = load_data(csv_path)
            dataset_data[dataset_name][mult_num] = df

        except ValueError as e:
            print(f"跳过无法解析的文件: {csv_file} - {e}")
            continue

    return dataset_data


def plot_token_length_comparison(df: pd.DataFrame,
                                dataset_name: str = "qm9hook",
                                save_dir: str = None) -> None:
    """
    绘制序列化长度对比图

    Args:
        df: 包含序列化数据的DataFrame
        dataset_name: 数据集名称
        save_dir: 保存目录，为None时使用当前目录
    """
    # 设置matplotlib样式
    setup_matplotlib_style()
    colors = get_color_dict()
    fontsize = get_fontsize_dict()
    
    # 创建图形
    fig, ax = create_figure_with_style(figsize=(10,6))
    
    # 准备数据（兼容旧字段名）
    methods = df['serialization_method'].tolist()
    if 'original_total_tokens' in df.columns and 'compressed_total_tokens' in df.columns:
        original_tokens = df['original_total_tokens'].astype(float).tolist()
        compressed_tokens = df['compressed_total_tokens'].astype(float).tolist()
    else:
        # 兼容旧版：以K为单位写入
        original_k = df['original_length'].astype(float).tolist()
        compressed_k = df['bpe_compressed_length'].astype(float).tolist()
        original_tokens = [v * 1000.0 for v in original_k]
        compressed_tokens = [v * 1000.0 for v in compressed_k]

    # 压缩比
    if 'compression_ratio' in df.columns:
        csv_ratios = df['compression_ratio'].astype(float).tolist()
    else:
        csv_ratios = []

    # 验证压缩比：compression_ratio = original_total_tokens / compressed_total_tokens
    calculated_ratios = []
    for orig, comp in zip(original_tokens, compressed_tokens):
        if comp > 0:
            calculated_ratios.append(orig / comp)
        else:
            calculated_ratios.append(1.0)
    
    # 若CSV提供了列，则进行一致性校验；否则使用计算值
    if csv_ratios:
        for i, (calc, csvv) in enumerate(zip(calculated_ratios, csv_ratios)):
            if abs(calc - csvv) > 1e-6:
                print(f"警告: {methods[i]}的压缩比不一致: 计算值={calc:.6f}, CSV值={csvv:.6f}")
        compression_ratios = csv_ratios
    else:
        compression_ratios = calculated_ratios
    
    # 转换为K单位仅用于显示
    original_lengths_k = [v / 1000.0 for v in original_tokens]
    compressed_lengths_k = [v / 1000.0 for v in compressed_tokens]

    # 设置柱子位置
    x = np.arange(len(methods))
    width = 0.35  # 两个柱子的宽度，调宽一些
    
    # 绘制两组柱子（总序列长度，以K为单位）
    bars1 = ax.bar(x - width/2, original_lengths_k, width, 
                   color=colors['navy'], label='w/o BPE', alpha=0.8)
    bars2 = ax.bar(x + width/2, compressed_lengths_k, width, 
                   color=colors['blue'], label='w/ BPE', alpha=0.8)
    
    # 为压缩比创建第二个y轴，绘制折线图
    ax2 = ax.twinx()
    line = ax2.plot(x, compression_ratios, 
                    color=colors['red'], marker='o', linewidth=2, markersize=8,
                    label='Compression Ratio', alpha=0.9)
    
    # 设置第一个y轴（序列长度）
    ax.set_xlabel('Method', fontsize=fontsize['label'])
    ax.set_ylabel('Sequence Length (K)', fontsize=fontsize['label'])
    ax.set_xticks(x)
    ax.set_xticklabels(methods,fontsize=fontsize['ticks'])  # 不旋转标签
    
    # 设置第二个y轴（压缩比）
    ax2.set_ylabel('Compress Ratio', fontsize=fontsize['label'])
    ax2.set_ylim(1, max(compression_ratios) * 1.3)  # 动态设置范围
    
    # 设置标题
    # title = f'{dataset_name.upper()} Dataset - Serialization Length Comparison'
    # ax.set_title(title, fontsize=fontsize['title'], pad=20)
    
    # 添加数值标签
    def add_value_labels(bars, ax_obj, format_str='{:.1f}'):
        for bar in bars:
            height = bar.get_height()
            ax_obj.text(bar.get_x() + bar.get_width()/2., height,
                       format_str.format(height),
                       ha='center', va='bottom', fontsize=16)

    # 为折线图添加数值标签
    def add_line_labels(x_pos, y_values, ax_obj, format_str='{:.1f}'):
        for x, y in zip(x_pos, y_values):
            ax_obj.text(x, y + max(y_values) * 0.05,
                       format_str.format(y),
                       ha='center', va='bottom', fontsize=16)
    
    # 不为原始序列长度柱子添加数值标签
    # add_value_labels(bars1, ax, '{:.1f}')
    # add_value_labels(bars2, ax, '{:.1f}')
    add_line_labels(x, compression_ratios, ax2, '{:.2f}x')
    
    # 添加图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left', fontsize=fontsize['legend'])
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f'{dataset_name}_token_length_comparison'
    save_plot(fig, filename, save_dir)
    
    plt.show()


def plot_multiple_mult_comparison(dataset_data: Dict[int, pd.DataFrame],
                                  dataset_name: str,
                                  mult_values: List[int],
                                  save_dir: str = None) -> None:
    """
    绘制同一个数据集不同多重采样次数的对比图

    Args:
        dataset_data: {多重采样次数: DataFrame}
        dataset_name: 数据集名称
        mult_values: 多重采样次数列表
        save_dir: 保存目录
    """
    if len(dataset_data) <= 1:
        # 只有一个多重采样次数，直接使用原函数
        mult_num = list(dataset_data.keys())[0]
        df = dataset_data[mult_num]
        plot_token_length_comparison(df, dataset_name, save_dir)
        return

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

    # 创建图形
    fig, ax = create_figure_with_style(figsize=(16, 10))

    # 获取所有方法的并集
    all_methods = set()
    for df in dataset_data.values():
        all_methods.update(df['serialization_method'].tolist())
    methods = sorted(list(all_methods))

    # 设置柱子位置
    x = np.arange(len(methods))
    width = 0.8 / len(mult_values)  # 根据多重采样次数调整柱子宽度

    # 为每个多重采样次数绘制柱子
    mult_colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    bars_original = []
    bars_compressed = []

    for i, mult_num in enumerate(mult_values):
        if mult_num not in dataset_data:
            continue

        df = dataset_data[mult_num]

        # 准备数据
        method_to_data = {row['serialization_method']: row for _, row in df.iterrows()}

        original_lengths = []
        compressed_lengths = []
        compression_ratios = []

        for method in methods:
            if method in method_to_data:
                data = method_to_data[method]
                original_lengths.append(data['original_total_tokens'] / 1000.0)  # 转换为K
                compressed_lengths.append(data['compressed_total_tokens'] / 1000.0)  # 转换为K
                compression_ratios.append(data['compression_ratio'])
            else:
                original_lengths.append(0)
                compressed_lengths.append(0)
                compression_ratios.append(1.0)

        # 绘制柱子
        color = mult_colors[i % len(mult_colors)]
        bars_orig = ax.bar(x + (i - len(mult_values)/2) * width,
                          original_lengths, width,
                          color=color, alpha=0.7,
                          label=f'Mult-{mult_num} Original')
        bars_comp = ax.bar(x + (i - len(mult_values)/2) * width,
                          compressed_lengths, width,
                          color=color, alpha=0.9, hatch='//',
                          label=f'Mult-{mult_num} Compressed')

        bars_original.append(bars_orig)
        bars_compressed.append(bars_comp)

    # 设置轴标签
    ax.set_xlabel('Serialization Method', fontsize=fontsize['label'])
    ax.set_ylabel('Total Sequence Length (K tokens)', fontsize=fontsize['label'])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')

    # 添加图例
    ax.legend(loc='upper right', fontsize=fontsize['legend']-2)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    # 文件名不包含多重采样信息，因为我们要展示数据集上的方法对比
    filename = f'{dataset_name}_mult_comparison_token_length'
    save_plot(fig, filename, save_dir)

    plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="序列化长度对比画图脚本")
    parser.add_argument("--mult", type=int, nargs='+',
                       help="要比较的多重采样次数，如 --mult 1 5 10")
    parser.add_argument("--datasets", type=str, nargs='+',
                       help="要处理的数据集名称，如 --datasets qm9test zinc")
    parser.add_argument("--input_dir", type=str,
                       help="输入CSV文件的目录，默认使用脚本所在目录")

    args = parser.parse_args()

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir if args.input_dir else script_dir

    try:
        # 加载所有CSV文件
        dataset_data = load_multiple_csv_files(input_dir, args.mult)

        if not dataset_data:
            print(f"在目录 {input_dir} 中未找到有效的CSV文件")
            return

        print(f"找到 {len(dataset_data)} 个数据集:")
        for dataset_name, mult_data in dataset_data.items():
            print(f"  {dataset_name}: 多重采样次数 {list(mult_data.keys())}")

        # 处理每个数据集
        for dataset_name, mult_data in dataset_data.items():
            print(f"\n处理数据集: {dataset_name}")

            # 获取可用的多重采样次数
            available_mults = sorted(mult_data.keys())

            # 如果用户指定了特定的多重采样次数，只使用指定的
            if args.mult:
                available_mults = [m for m in available_mults if m in args.mult]
                if not available_mults:
                    print(f"  跳过: 数据集 {dataset_name} 没有指定的多重采样次数")
                    continue

            # 绘制对比图
            plot_multiple_mult_comparison(mult_data, dataset_name, available_mults, script_dir)
            print(f"  完成绘制: {dataset_name} (多重采样: {available_mults})")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()