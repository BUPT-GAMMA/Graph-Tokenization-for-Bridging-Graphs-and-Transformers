"""
Visualization helpers.
可视化辅助工具。

CJK font setup, plot styling, and chart generation utilities.
CJK字体设置、绘图样式和图表生成工具。
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

def set_chinese_font():
    """Set up CJK font support for matplotlib."""
    
    # Try common CJK fonts
    cjk_fonts = ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei']
    
    for font in plt.rcParams['font.sans-serif']:
        if any(cjk in font for cjk in ['Noto', 'CJK', 'WenQuanYi']):
            print(f"Using CJK font: {font}")
            break
    else:
        # Fallback: prepend CJK font list
        plt.rcParams['font.sans-serif'] = cjk_fonts + plt.rcParams['font.sans-serif']
        print("Using system default CJK font list")
    
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering
    
    # Use Agg backend for stability
    try:
        plt.switch_backend('Agg')
        print("Using Agg backend for rendering")
    except Exception as e:
        print(f"Backend switch failed, using default: {e}")
    
    # Suppress font warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
set_chinese_font()

def setup_plot_style():
    """Configure plot style."""
    set_chinese_font()
    
    # Plot style defaults
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['grid.alpha'] = 0.3
    
    print("Plot style configured")

def create_performance_comparison_plot(results, output_file='performance_comparison.png'):
    """
    Create performance comparison charts.
    
    Args:
        results: Algorithm performance results dict
        output_file: Output file path
    """
    setup_plot_style()
    
    # Extract data
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
    
    # Create 4-panel figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Serialization Algorithm Performance Comparison', fontsize=16, y=0.95)
    
    # Panel 1: Compression ratio
    bars1 = ax1.bar(methods, compression_ratios, color='skyblue', alpha=0.7)
    ax1.set_title('BPE Compression Ratio (lower is better)')
    ax1.set_ylabel('Compression Ratio')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, ratio in zip(bars1, compression_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{ratio:.3f}', ha='center', va='bottom')
    
    # Panel 2: Tokens saved
    bars2 = ax2.bar(methods, tokens_saved, color='lightgreen', alpha=0.7)
    ax2.set_title('Tokens Saved (higher is better)')
    ax2.set_ylabel('Tokens Saved')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, saved in zip(bars2, tokens_saved):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
                 f'{saved:,}', ha='center', va='bottom')
    
    # Panel 3: Pre/post compression length
    x = range(len(methods))
    width = 0.35
    
    bars3a = ax3.bar([i - width/2 for i in x], original_lengths, width, 
                     label='Before BPE', color='lightcoral', alpha=0.7)
    bars3b = ax3.bar([i + width/2 for i in x], compressed_lengths, width,
                     label='After BPE', color='darkred', alpha=0.7)
    
    ax3.set_title('Length Before/After Compression')
    ax3.set_ylabel('Token Count')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (orig, comp) in enumerate(zip(original_lengths, compressed_lengths)):
        if orig > 0:
            ax3.text(i - width/2, orig + orig*0.02, f'{orig:,}', 
                    ha='center', va='bottom', fontsize=8)
        if comp > 0:
            ax3.text(i + width/2, comp + comp*0.02, f'{comp:,}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Panel 4: Sequence diversity
    bars4 = ax4.bar(methods, diversities, color='gold', alpha=0.7)
    ax4.set_title('Sequence Diversity (higher is better)')
    ax4.set_ylabel('Diversity Score')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, diversity in zip(bars4, diversities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{diversity:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Performance comparison chart saved: {output_file}")
    return output_file

def create_comprehensive_analysis_plot(results, output_file='comprehensive_analysis.png'):
    """Create comprehensive analysis charts."""
    setup_plot_style()
    
    # Extract data for all algorithms
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
    
    # Create scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Serialization Algorithm Analysis', fontsize=16)
    
    # Left: Compression ratio vs diversity
    x1 = [d['diversity'] for d in all_algorithms]
    y1 = [d['compression_ratio'] for d in all_algorithms]
    names1 = [d['name'] for d in all_algorithms]
    
    scatter = ax1.scatter(x1, y1, s=100, alpha=0.7, color='blue')
    
    # Add labels
    for i, name in enumerate(names1):
        ax1.annotate(name, (x1[i], y1[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Sequence Diversity')
    ax1.set_ylabel('Compression Ratio (lower is better)')
    ax1.set_title('Compression Ratio vs Diversity')
    ax1.grid(True, alpha=0.3)
    
    # Right: Sequence length vs tokens saved
    x2 = [d['sequence_length'] for d in all_algorithms]
    y2 = [d['tokens_saved'] for d in all_algorithms]
    names2 = [d['name'] for d in all_algorithms]
    
    scatter = ax2.scatter(x2, y2, s=100, alpha=0.7, color='green')
    
    # Add labels
    for i, name in enumerate(names2):
        ax2.annotate(name, (x2[i], y2[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Average Sequence Length')
    ax2.set_ylabel('Tokens Saved')
    ax2.set_title('Sequence Length vs Compression Effect')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Analysis chart saved: {output_file}")
    return output_file 