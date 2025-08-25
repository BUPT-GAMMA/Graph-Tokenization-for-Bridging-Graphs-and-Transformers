#!/usr/bin/env python3
"""
预训练结果收集和分析脚本
=======================

收集所有预训练实验的validation loss结果，排序分析
"""

import json
import sys
from pathlib import Path
import pandas as pd
import argparse


def collect_pretrain_results(experiment_group: str, log_base: str = "log") -> pd.DataFrame:
    """收集预训练结果"""
    
    log_base_path = Path(log_base)
    results = []
    
    # 遍历实验组目录
    exp_group_path = log_base_path / experiment_group
    if not exp_group_path.exists():
        print(f"❌ 实验组目录不存在: {exp_group_path}")
        return pd.DataFrame()
    
    print(f"🔍 搜索实验结果: {exp_group_path}")
    
    # 查找所有pretrain_metrics.json文件
    metrics_files = list(exp_group_path.rglob("pretrain_metrics.json"))
    
    if not metrics_files:
        print(f"❌ 未找到预训练结果文件，搜索路径: {exp_group_path}/**/pretrain_metrics.json")
        return pd.DataFrame()
    
    print(f"📊 找到 {len(metrics_files)} 个结果文件")
    
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # 从路径解析实验名
            # 路径格式: log/{group}/{exp_name}/{dataset}/{method}/pretrain_metrics.json
            path_parts = metrics_file.parts
            exp_name = path_parts[-4]  # experiment_name
            dataset = path_parts[-3]   # dataset
            method = path_parts[-2]    # method
            
            result = {
                'experiment_name': exp_name,
                'dataset': dataset,
                'method': method,
                'best_val_loss': metrics.get('best_val_loss', float('inf')),
                'best_epoch': metrics.get('best_epoch', -1),
                'total_train_time_sec': metrics.get('total_train_time_sec', 0),
                'avg_epoch_time_sec': metrics.get('avg_epoch_time_sec', 0),
                'epochs': metrics.get('epochs', 0),
                'metrics_file': str(metrics_file)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"⚠️ 读取失败: {metrics_file} - {e}")
    
    if not results:
        print("❌ 未成功解析任何结果文件")
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    print(f"✅ 成功收集 {len(df)} 个实验结果")
    return df


def analyze_and_save_results(df: pd.DataFrame, output_dir: Path):
    """分析结果并保存"""
    
    if df.empty:
        print("❌ 无数据可分析")
        return
    
    # 按验证损失排序
    df_sorted = df.sort_values('best_val_loss', ascending=True).reset_index(drop=True)
    
    # 保存完整结果
    results_file = output_dir / "pretrain_results.csv"
    df_sorted.to_csv(results_file, index=False)
    print(f"💾 完整结果已保存: {results_file}")
    
    # 保存top结果
    top_k = min(8, len(df_sorted))  # 取前8名或全部
    df_top = df_sorted.head(top_k)
    top_file = output_dir / "pretrain_top_results.csv"
    df_top.to_csv(top_file, index=False)
    print(f"🏆 Top-{top_k} 结果已保存: {top_file}")
    
    # 打印分析结果
    print(f"\n📈 预训练实验结果分析:")
    print(f"{'排名':<4} {'实验名':<8} {'验证损失':<12} {'最佳Epoch':<8} {'训练时间(s)':<12} {'平均Epoch时间':<12}")
    print("-" * 70)
    
    for idx, row in df_sorted.iterrows():
        rank = idx + 1
        print(f"{rank:<4} {row['experiment_name']:<8} {row['best_val_loss']:<12.4f} "
              f"{row['best_epoch']:<8} {row['total_train_time_sec']:<12.1f} {row['avg_epoch_time_sec']:<12.2f}")
    
    # 保存分析报告
    analysis_file = output_dir / "pretrain_analysis.txt"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("预训练实验结果分析报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"实验数量: {len(df_sorted)}\n")
        f.write(f"最佳验证损失: {df_sorted.iloc[0]['best_val_loss']:.4f} (实验: {df_sorted.iloc[0]['experiment_name']})\n")
        f.write(f"最差验证损失: {df_sorted.iloc[-1]['best_val_loss']:.4f} (实验: {df_sorted.iloc[-1]['experiment_name']})\n")
        f.write(f"平均验证损失: {df_sorted['best_val_loss'].mean():.4f}\n")
        f.write(f"验证损失标准差: {df_sorted['best_val_loss'].std():.4f}\n\n")
        
        f.write("实验编码说明:\n")
        f.write("P{seq3位}{train1位}\n")
        f.write("序列级增强: deletion(0位), swap(1位), truncation(2位)\n")
        f.write("训练级增强: consistency(0位)\n\n")
        
        f.write("Top结果:\n")
        for idx, row in df_sorted.head(8).iterrows():
            f.write(f"{idx+1}. {row['experiment_name']}: val_loss={row['best_val_loss']:.4f}, "
                   f"epoch={row['best_epoch']}, time={row['total_train_time_sec']:.1f}s\n")
    
    print(f"📄 分析报告已保存: {analysis_file}")
    
    return df_sorted


def main():
    parser = argparse.ArgumentParser(description="收集预训练实验结果")
    parser.add_argument("--experiment_group", type=str, default="aug_pretrain", 
                       help="实验组名称")
    parser.add_argument("--log_base", type=str, default="log", 
                       help="日志根目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"🚀 开始收集预训练结果...")
    print(f"   实验组: {args.experiment_group}")
    print(f"   日志目录: {args.log_base}")
    print(f"   输出目录: {output_dir}")
    
    # 收集结果
    df = collect_pretrain_results(args.experiment_group, args.log_base)
    
    if df.empty:
        print("❌ 未找到任何结果，请检查实验是否完成")
        sys.exit(1)
    
    # 分析结果
    df_sorted = analyze_and_save_results(df, output_dir)
    
    # 输出建议
    print(f"\n💡 建议:")
    top_3 = df_sorted.head(3)['experiment_name'].tolist()
    print(f"   基于验证损失，建议对以下实验进行微调测试: {', '.join(top_3)}")
    
    print(f"\n🎯 下一步:")
    print(f"   1. 基于Top结果生成微调实验: python generate_finetune_experiments.py --pretrain_experiments {' '.join(top_3[:3])}")
    print(f"   2. 查看详细结果: cat {output_dir}/pretrain_analysis.txt")


if __name__ == "__main__":
    main()
