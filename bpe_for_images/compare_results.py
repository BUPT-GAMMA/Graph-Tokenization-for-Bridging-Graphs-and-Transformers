"""
结果对比脚本
===========

汇总所有模型的训练结果，生成对比表格
"""

import json
import csv
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from config import RESULTS_DIR, COMPARISON_CSV
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_all_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    加载所有结果文件
    
    Returns:
        {model_name: results_dict}
    """
    all_results = {}
    
    for result_file in results_dir.glob("*_results.json"):
        model_name = result_file.stem.replace("_results", "")
        
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            all_results[model_name] = results
            logger.info(f"加载结果: {model_name}")
        except Exception as e:
            logger.warning(f"无法加载 {result_file}: {e}")
    
    return all_results


def create_comparison_table(all_results: Dict[str, Dict[str, Any]]) -> List[Dict]:
    """
    创建对比表格
    
    Returns:
        对比数据列表
    """
    table = []
    
    for model_name, results in all_results.items():
        # 提取关键指标
        row = {
            'model_name': model_name,
            'test_acc': results.get('final_test_acc', 0.0),
            'best_val_acc': results.get('best_val_acc', 0.0),
            'best_epoch': results.get('best_epoch', 0),
            'total_params': results.get('total_params', 0),
            'training_time': results.get('training_time_total', 0.0),
        }
        
        # 计算训练历史统计
        history = results.get('training_history', {})
        if history:
            epochs = history.get('epochs', [])
            train_accs = history.get('train_accs', [])
            val_accs = history.get('val_accs', [])
            epoch_times = history.get('epoch_times', [])
            
            if epochs:
                row['num_epochs'] = len(epochs)
                row['final_train_acc'] = train_accs[-1] if train_accs else 0.0
                row['final_val_acc'] = val_accs[-1] if val_accs else 0.0
                row['avg_epoch_time'] = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        
        # 提取配置信息
        config = results.get('config', {})
        if 'bpe_num_merges' in config:
            row['bpe_merges'] = config['bpe_num_merges']
            row['avg_seq_len'] = config.get('avg_sequence_length', 0.0)
        else:
            row['bpe_merges'] = 0
            row['avg_seq_len'] = 784 if 'transformer' in model_name else 0
        
        if 'd_model' in config:
            row['d_model'] = config['d_model']
            row['n_layers'] = config.get('n_layers', 0)
        
        table.append(row)
    
    # 按测试准确率排序
    table.sort(key=lambda x: x['test_acc'], reverse=True)
    
    return table


def save_to_csv(table: List[Dict], output_path: Path):
    """保存为CSV文件"""
    if not table:
        logger.warning("没有数据可保存")
        return
    
    # 定义列顺序
    fieldnames = [
        'model_name', 'test_acc', 'best_val_acc', 'final_train_acc',
        'best_epoch', 'num_epochs', 'total_params', 'training_time',
        'avg_epoch_time', 'bpe_merges', 'avg_seq_len', 'd_model', 'n_layers'
    ]
    
    # 过滤只保留存在的字段
    fieldnames = [f for f in fieldnames if any(f in row for row in table)]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            # 只写入存在的字段
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)
    
    logger.info(f"CSV文件已保存到: {output_path}")


def print_comparison_table(table: List[Dict]):
    """打印对比表格"""
    if not table:
        logger.warning("没有数据可显示")
        return
    
    logger.info("\n" + "="*100)
    logger.info("模型对比结果")
    logger.info("="*100)
    
    # 表头
    header = f"{'模型':<30} {'测试准确率':>10} {'验证准确率':>10} {'参数量':>12} {'训练时间':>10} {'每轮时间':>10}"
    logger.info(header)
    logger.info("-" * 100)
    
    # 数据行
    for row in table:
        line = (f"{row['model_name']:<30} "
               f"{row['test_acc']:>10.4f} "
               f"{row['best_val_acc']:>10.4f} "
               f"{row['total_params']:>12,} "
               f"{row['training_time']:>10.2f}s "
               f"{row.get('avg_epoch_time', 0.0):>10.2f}s")
        logger.info(line)
    
    logger.info("="*100)
    
    # 最佳模型
    best_model = table[0]
    logger.info(f"\n🏆 最佳模型: {best_model['model_name']}")
    logger.info(f"   测试准确率: {best_model['test_acc']:.4f}")
    logger.info(f"   参数量: {best_model['total_params']:,}")
    
    # 统计分析
    logger.info("\n📊 统计分析:")
    
    # 按类型分组
    baselines = [r for r in table if r['model_name'] in ['mlp', 'lenet']]
    transformers = [r for r in table if 'transformer' in r['model_name'] and 'bpe' not in r['model_name']]
    bpe_transformers = [r for r in table if 'bpe_transformer' in r['model_name']]
    
    if baselines:
        avg_acc = sum(r['test_acc'] for r in baselines) / len(baselines)
        logger.info(f"  Baseline模型平均准确率: {avg_acc:.4f}")
    
    if transformers:
        avg_acc = sum(r['test_acc'] for r in transformers) / len(transformers)
        logger.info(f"  Transformer平均准确率: {avg_acc:.4f}")
    
    if bpe_transformers:
        avg_acc = sum(r['test_acc'] for r in bpe_transformers) / len(bpe_transformers)
        logger.info(f"  BPE+Transformer平均准确率: {avg_acc:.4f}")
        
        # 序列压缩统计
        if 'avg_seq_len' in bpe_transformers[0]:
            avg_len = sum(r.get('avg_seq_len', 0) for r in bpe_transformers) / len(bpe_transformers)
            compression = avg_len / 784
            logger.info(f"  平均序列长度: {avg_len:.1f} (压缩率: {compression:.2%})")


def main(args):
    """主流程"""
    
    logger.info("="*60)
    logger.info("结果对比和汇总")
    logger.info("="*60)
    
    # 1. 加载所有结果
    logger.info("\n1. 加载结果文件...")
    all_results = load_all_results(RESULTS_DIR)
    
    if not all_results:
        logger.error("没有找到任何结果文件")
        return
    
    logger.info(f"  找到 {len(all_results)} 个结果文件")
    
    # 2. 创建对比表格
    logger.info("\n2. 生成对比表格...")
    table = create_comparison_table(all_results)
    
    # 3. 保存CSV
    logger.info("\n3. 保存CSV文件...")
    csv_path = RESULTS_DIR / COMPARISON_CSV
    save_to_csv(table, csv_path)
    
    # 4. 打印表格
    print_comparison_table(table)
    
    logger.info("\n完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比所有模型结果")
    args = parser.parse_args()
    main(args)

