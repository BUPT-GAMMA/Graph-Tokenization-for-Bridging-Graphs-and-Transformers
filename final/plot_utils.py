"""
画图工具函数 - 统一matplotlib画图格式设置
========================================

提供统一的matplotlib画图格式设置函数，用于保持一致的视觉风格。
基于process.py中的画图设置提取和优化。

主要功能：
- 设置matplotlib全局参数
- 提供统一的颜色、线型、字体大小等配置
- 统一的图例和标签格式
"""

import matplotlib.pyplot as plt
import matplotlib
import os
from typing import Dict, Optional


def setup_matplotlib_style(dpi: int = 300, save_format: str = 'jpg'):
    """
    设置matplotlib的全局样式参数

    Args:
        dpi: 图片分辨率，默认300
        save_format: 保存格式，默认'jpg'
    """
    # 字体设置
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.titleweight'] = 'bold'

    # 线条和标签设置
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['xtick.labelsize'] = 'large'
    matplotlib.rcParams['ytick.labelsize'] = 'large'

    # 设置全局变量（如果需要的话）
    global _global_dpi, _global_save_format
    _global_dpi = dpi
    _global_save_format = save_format


def get_color_dict() -> Dict[str, str]:
    """
    获取统一的颜色字典

    Returns:
        颜色字典 - 按照固定顺序分配常用颜色
    """
    return {
        'red': '#FF6B6B',
        'blue': '#4ECDC4',
        'green': '#45B7D1',
        'orange': '#FFA07A',
        'purple': '#98D8C8',
        'gold': '#F7DC6F',
        'pink': '#F8C8DC',
        'teal': '#A8E6CF',
        'navy': '#4A90E2',
        'coral': '#FF9A8B'
    }


def get_linestyle_dict() -> Dict[str, str]:
    """
    获取统一的线型字典

    Returns:
        线型字典 - 按照固定顺序分配常用线型
    """
    return {
        'solid': '-',
        'dashed': '--',
        'dotted': ':',
        'dashdot': '-.',
        'solid_thick': '-',
        'dashed_thick': '--',
        'dotted_thick': ':',
        'dashdot_thick': '-.'
    }


def get_fontsize_dict() -> Dict[str, int]:
    """
    获取统一的字体大小字典

    Returns:
        字体大小字典
    """
    return {
        'legend': 12,
        'label': 16,
        'ticks': 16,
        'title': 16
    }


def get_legend_loc_dict() -> Dict[str, str]:
    """
    获取统一的图例位置字典

    Returns:
        图例位置字典
    """
    return {
        'Accuracy': 'lower right',
        'Loss': 'upper right',
        'default': 'best'
    }


def get_default_figsize() -> tuple:
    """
    获取默认的图片大小

    Returns:
        默认图片大小 (width, height)
    """
    return (10, 7)


def setup_plot_style(ax: plt.Axes,
                    xlabel: str,
                    ylabel: str,
                    title: Optional[str] = None,
                    fontsize_dict: Optional[Dict] = None):
    """
    设置单个子图的样式

    Args:
        ax: matplotlib axes对象
        xlabel: x轴标签
        ylabel: y轴标签
        title: 图标题（可选）
        fontsize_dict: 字体大小字典（可选，使用默认值）
    """
    if fontsize_dict is None:
        fontsize_dict = get_fontsize_dict()

    ax.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict['label'])

    if title:
        ax.set_title(title, fontsize=fontsize_dict['title'])

    ax.xaxis.set_tick_params(labelsize=fontsize_dict['ticks'])
    ax.yaxis.set_tick_params(labelsize=fontsize_dict['ticks'])

    # 添加网格
    ax.grid(linestyle='--', linewidth='0.5')


def save_plot(fig: plt.Figure,
             filename: str,
             output_dir: str = 'output/vis',
             dpi: Optional[int] = None,
             save_format: Optional[str] = None):
    """
    保存图片到指定目录

    Args:
        fig: matplotlib figure对象
        filename: 文件名（不含扩展名）
        output_dir: 输出目录
        dpi: 分辨率（可选，使用全局设置）
        save_format: 保存格式（可选，使用全局设置）
    """
    if dpi is None:
        dpi = _global_dpi
    if save_format is None:
        save_format = _global_save_format

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图片
    filepath = os.path.join(output_dir, f'{filename}.{save_format}')
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
    print(f"图片已保存到: {filepath}")


def get_color_by_index(index: int) -> str:
    """
    根据索引获取颜色（循环使用）

    Args:
        index: 颜色索引

    Returns:
        颜色字符串
    """
    colors = list(get_color_dict().values())
    return colors[index % len(colors)]


def get_linestyle_by_index(index: int) -> str:
    """
    根据索引获取线型（循环使用）

    Args:
        index: 线型索引

    Returns:
        线型字符串
    """
    linestyles = list(get_linestyle_dict().values())
    return linestyles[index % len(linestyles)]


def get_plot_style_by_index(index: int) -> tuple:
    """
    根据索引获取完整的绘图样式（颜色和线型）

    Args:
        index: 样式索引

    Returns:
        (color, linestyle) 元组
    """
    return get_color_by_index(index), get_linestyle_by_index(index)


def create_figure_with_style(figsize: Optional[tuple] = None) -> tuple:
    """
    创建带有统一样式的figure和axes

    Args:
        figsize: 图片大小（可选，使用默认值）

    Returns:
        (fig, ax) 元组
    """
    if figsize is None:
        figsize = get_default_figsize()

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# 全局变量初始化
_global_dpi = 300
_global_save_format = 'jpg'


# 使用示例
if __name__ == "__main__":
    # 设置样式
    setup_matplotlib_style()

    # 创建图表
    fig, ax = create_figure_with_style()

    # 设置样式
    setup_plot_style(ax, 'X轴', 'Y轴', '示例图表')

    # 添加一些示例数据
    x = [1, 2, 3, 4, 5]
    y1 = [1, 4, 9, 16, 25]
    y2 = [1, 2, 4, 8, 16]

    # 使用索引自动分配颜色和线型
    color1, linestyle1 = get_plot_style_by_index(0)
    color2, linestyle2 = get_plot_style_by_index(1)

    ax.plot(x, y1, color=color1, linestyle=linestyle1, label='y = x²')
    ax.plot(x, y2, color=color2, linestyle=linestyle2, label='y = 2^x')

    # 设置图例
    ax.legend(loc=get_legend_loc_dict()['default'],
             fontsize=get_fontsize_dict()['legend'])

    # 保存图片
    save_plot(fig, 'example_plot')

    plt.show()
