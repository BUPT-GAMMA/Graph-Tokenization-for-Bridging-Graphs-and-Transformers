"""
可视化辅助工具
===============

提供中文字体支持、图表美化等可视化相关的辅助函数
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

def set_chinese_font():
    """设置中文字体支持"""
    
    # 尝试使用常见的中文字体
    cjk_fonts = ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei']
    
    for font in plt.rcParams['font.sans-serif']:
        if any(cjk in font for cjk in ['Noto', 'CJK', 'WenQuanYi']):
            print(f"✅ 使用现有中文字体: {font}")
            break
    else:
        # 如果没有找到合适的字体，使用回退选项
        plt.rcParams['font.sans-serif'] = cjk_fonts + plt.rcParams['font.sans-serif']
        print("📝 使用系统默认中文字体列表")
    
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 尝试特殊解决方案：使用Agg后端（更稳定）
    try:
        plt.switch_backend('Agg')
        print("🖼️  使用Agg后端进行图片渲染")
    except Exception as e:
        print(f"⚠️  后端切换失败，使用默认后端: {e}")
    
    # 忽略字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
set_chinese_font()

def setup_plot_style():
    """设置图表样式"""
    # 设置中文字体
    set_chinese_font()
    
    # 设置图表样式
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['grid.alpha'] = 0.3
    
    print("🎨 图表样式配置完成")

def create_performance_comparison_plot(results, output_file='performance_comparison.png'):
    """
    创建性能对比图表
    
    Args:
        results: 算法性能结果字典
        output_file: 输出文件路径
    """
    setup_plot_style()
    
    # 提取数据
    methods = []
    compression_ratios = []
    tokens_saved = []
    original_lengths = []
    compressed_lengths = []
    diversities = []
    
    for method_name, method_results in results.items():
        if method_results['serialization']['success_count'] > 0:
            methods.append(method_name)
            diversities.append(method_results['serialization']['sequence_diversity'])
            
            if method_results['compression'] and 'compression_stats' in method_results['compression']:
                comp_stats = method_results['compression']['compression_stats']
                compression_ratios.append(comp_stats['compression_ratio'])
                tokens_saved.append(comp_stats['tokens_saved'])
                original_lengths.append(comp_stats['original_token_count'])
                compressed_lengths.append(comp_stats['compressed_token_count'])
            else:
                compression_ratios.append(1.0)
                tokens_saved.append(0)
                original_lengths.append(0)
                compressed_lengths.append(0)
    
    # 创建四子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('分子图序列化算法性能对比', fontsize=16, y=0.95)
    
    # 子图1: 压缩率对比
    bars1 = ax1.bar(methods, compression_ratios, color='skyblue', alpha=0.7)
    ax1.set_title('BPE压缩率对比 (越低越好)')
    ax1.set_ylabel('压缩率')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, ratio in zip(bars1, compression_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{ratio:.3f}', ha='center', va='bottom')
    
    # 子图2: 节省tokens对比
    bars2 = ax2.bar(methods, tokens_saved, color='lightgreen', alpha=0.7)
    ax2.set_title('节省Tokens数量对比 (越高越好)')
    ax2.set_ylabel('节省Tokens数')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, saved in zip(bars2, tokens_saved):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
                 f'{saved:,}', ha='center', va='bottom')
    
    # 子图3: 压缩前后长度对比
    x = range(len(methods))
    width = 0.35
    
    bars3a = ax3.bar([i - width/2 for i in x], original_lengths, width, 
                     label='压缩前', color='lightcoral', alpha=0.7)
    bars3b = ax3.bar([i + width/2 for i in x], compressed_lengths, width,
                     label='压缩后', color='darkred', alpha=0.7)
    
    ax3.set_title('压缩前后长度对比')
    ax3.set_ylabel('Token数量')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (orig, comp) in enumerate(zip(original_lengths, compressed_lengths)):
        if orig > 0:
            ax3.text(i - width/2, orig + orig*0.02, f'{orig:,}', 
                    ha='center', va='bottom', fontsize=8)
        if comp > 0:
            ax3.text(i + width/2, comp + comp*0.02, f'{comp:,}', 
                    ha='center', va='bottom', fontsize=8)
    
    # 子图4: 序列多样性对比
    bars4 = ax4.bar(methods, diversities, color='gold', alpha=0.7)
    ax4.set_title('序列多样性对比 (越高越好)')
    ax4.set_ylabel('多样性系数')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, diversity in zip(bars4, diversities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{diversity:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 性能对比图表已保存: {output_file}")
    return output_file

def create_comprehensive_analysis_plot(results, output_file='comprehensive_analysis.png'):
    """创建综合分析图表"""
    setup_plot_style()
    
    # 提取所有算法数据
    all_algorithms = []
    
    for method_name, method_results in results.items():
        if method_results['serialization']['success_count'] > 0:
            data_point = {
                'name': method_name,
                'compression_ratio': 1.0,
                'tokens_saved': 0,
                'sequence_length': method_results['serialization']['avg_length'],
                'diversity': method_results['serialization']['sequence_diversity']
            }
            
            if method_results['compression'] and 'compression_stats' in method_results['compression']:
                comp_stats = method_results['compression']['compression_stats']
                data_point['compression_ratio'] = comp_stats['compression_ratio']
                data_point['tokens_saved'] = comp_stats['tokens_saved']
            
            all_algorithms.append(data_point)
    
    # 创建散点图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('分子图序列化算法综合分析', fontsize=16)
    
    # 左图: 压缩率 vs 序列多样性
    x1 = [d['diversity'] for d in all_algorithms]
    y1 = [d['compression_ratio'] for d in all_algorithms]
    names1 = [d['name'] for d in all_algorithms]
    
    scatter = ax1.scatter(x1, y1, s=100, alpha=0.7, color='blue')
    
    # 添加标签
    for i, name in enumerate(names1):
        ax1.annotate(name, (x1[i], y1[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('序列多样性')
    ax1.set_ylabel('压缩率 (越低越好)')
    ax1.set_title('压缩率 vs 多样性分析')
    ax1.grid(True, alpha=0.3)
    
    # 右图: 序列长度 vs 节省tokens
    x2 = [d['sequence_length'] for d in all_algorithms]
    y2 = [d['tokens_saved'] for d in all_algorithms]
    names2 = [d['name'] for d in all_algorithms]
    
    scatter = ax2.scatter(x2, y2, s=100, alpha=0.7, color='green')
    
    # 添加标签
    for i, name in enumerate(names2):
        ax2.annotate(name, (x2[i], y2[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('平均序列长度')
    ax2.set_ylabel('节省Tokens数')
    ax2.set_title('序列长度 vs 压缩效果分析')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"🔍 综合分析图表已保存: {output_file}")
    return output_file 