"""
结果可视化脚本
=============

绘制训练曲线和对比图表
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import sys
from pathlib import Path
import argparse
from typing import Dict, Any, List
import numpy as np

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from config import RESULTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_all_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """加载所有结果文件"""
    all_results = {}
    
    for result_file in results_dir.glob("*_results.json"):
        model_name = result_file.stem.replace("_results", "")
        
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            all_results[model_name] = results
        except Exception as e:
            logger.warning(f"无法加载 {result_file}: {e}")
    
    return all_results


def plot_training_curves(all_results: Dict[str, Dict], output_dir: Path):
    """
    绘制训练曲线
    
    为每个模型绘制训练/验证准确率和损失曲线
    """
    logger.info("绘制训练曲线...")
    
    # 创建多个子图
    n_models = len(all_results)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
    
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (model_name, results) in enumerate(sorted(all_results.items())):
        history = results.get('training_history', {})
        
        if not history:
            continue
        
        epochs = history.get('epochs', [])
        train_accs = history.get('train_accs', [])
        val_accs = history.get('val_accs', [])
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        
        # 准确率曲线
        ax_acc = axes[0, idx]
        if train_accs:
            ax_acc.plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
        if val_accs:
            ax_acc.plot(epochs, val_accs, 'r-', label='Val', linewidth=2)
        ax_acc.set_title(f'{model_name}\nAccuracy', fontsize=10)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        
        # 损失曲线
        ax_loss = axes[1, idx]
        if train_losses:
            ax_loss.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        if val_losses:
            ax_loss.plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
        ax_loss.set_title('Loss', fontsize=10)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'training_curves_all.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  保存到: {output_path}")


def plot_accuracy_comparison(all_results: Dict[str, Dict], output_dir: Path):
    """
    绘制准确率对比柱状图
    """
    logger.info("绘制准确率对比图...")
    
    # 提取数据
    model_names = []
    test_accs = []
    val_accs = []
    
    for model_name, results in sorted(all_results.items()):
        model_names.append(model_name.replace('_', '\n'))  # 换行以便显示
        test_accs.append(results.get('final_test_acc', 0.0))
        val_accs.append(results.get('best_val_acc', 0.0))
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, test_accs, width, label='Test Acc', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, val_accs, width, label='Val Acc',
                   color='coral', alpha=0.8)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_path = output_dir / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  保存到: {output_path}")


def plot_efficiency_comparison(all_results: Dict[str, Dict], output_dir: Path):
    """
    绘制效率对比图（参数量 vs 准确率，训练时间 vs 准确率）
    """
    logger.info("绘制效率对比图...")
    
    # 提取数据
    model_names = []
    test_accs = []
    params = []
    times = []
    
    for model_name, results in sorted(all_results.items()):
        model_names.append(model_name)
        test_accs.append(results.get('final_test_acc', 0.0))
        params.append(results.get('total_params', 0) / 1e6)  # 转换为百万
        times.append(results.get('training_time_total', 0.0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 参数量 vs 准确率
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    ax1.scatter(params, test_accs, s=200, c=colors, alpha=0.7, edgecolors='black')
    
    for i, name in enumerate(model_names):
        ax1.annotate(name, (params[i], test_accs[i]), 
                    fontsize=8, ha='right', va='bottom')
    
    ax1.set_xlabel('Parameters (M)', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Parameters vs Accuracy', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 训练时间 vs 准确率
    ax2.scatter(times, test_accs, s=200, c=colors, alpha=0.7, edgecolors='black')
    
    for i, name in enumerate(model_names):
        ax2.annotate(name, (times[i], test_accs[i]),
                    fontsize=8, ha='right', va='bottom')
    
    ax2.set_xlabel('Training Time (s)', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Training Time vs Accuracy', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'efficiency_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  保存到: {output_path}")


def plot_individual_curves(all_results: Dict[str, Dict], output_dir: Path):
    """
    为每个模型单独绘制详细的训练曲线
    """
    logger.info("绘制单独训练曲线...")
    
    curves_dir = output_dir / 'individual_curves'
    curves_dir.mkdir(exist_ok=True)
    
    for model_name, results in all_results.items():
        history = results.get('training_history', {})
        
        if not history:
            continue
        
        epochs = history.get('epochs', [])
        train_accs = history.get('train_accs', [])
        val_accs = history.get('val_accs', [])
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        lrs = history.get('learning_rates', [])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 准确率
        axes[0, 0].plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, val_accs, 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 损失
        axes[0, 1].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率
        if lrs:
            axes[1, 0].plot(epochs, lrs, 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # 摘要信息
        axes[1, 1].axis('off')
        info_text = f"Model: {model_name}\n\n"
        info_text += f"Test Acc: {results.get('final_test_acc', 0.0):.4f}\n"
        info_text += f"Best Val Acc: {results.get('best_val_acc', 0.0):.4f}\n"
        info_text += f"Best Epoch: {results.get('best_epoch', 0)}\n"
        info_text += f"Params: {results.get('total_params', 0):,}\n"
        info_text += f"Training Time: {results.get('training_time_total', 0.0):.2f}s\n"
        
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=11,
                       verticalalignment='center',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{model_name} - Training History', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = curves_dir / f'{model_name}_curves.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"  保存到: {curves_dir}")


def main(args):
    """主流程"""
    
    logger.info("="*60)
    logger.info("结果可视化")
    logger.info("="*60)
    
    # 1. 加载所有结果
    logger.info("\n1. 加载结果文件...")
    all_results = load_all_results(RESULTS_DIR)
    
    if not all_results:
        logger.error("没有找到任何结果文件")
        return
    
    logger.info(f"  找到 {len(all_results)} 个结果文件")
    
    # 2. 创建可视化
    logger.info("\n2. 生成可视化图表...")
    
    if args.plot_type in ['all', 'curves']:
        plot_training_curves(all_results, RESULTS_DIR)
        plot_individual_curves(all_results, RESULTS_DIR)
    
    if args.plot_type in ['all', 'comparison']:
        plot_accuracy_comparison(all_results, RESULTS_DIR)
        plot_efficiency_comparison(all_results, RESULTS_DIR)
    
    logger.info("\n完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化训练结果")
    parser.add_argument("--plot_type", type=str, default="all",
                       choices=["all", "curves", "comparison"],
                       help="绘图类型")
    
    args = parser.parse_args()
    main(args)

