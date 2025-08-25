#!/usr/bin/env python3
"""
微调结果收集和分析脚本
=======================

收集所有微调实验的结果，按测试集性能排序分析
"""

import json
import sys
from pathlib import Path
import pandas as pd
import argparse
import numpy as np


def collect_finetune_results(experiment_group: str, log_base: str = "log") -> pd.DataFrame:
    """收集微调结果"""
    
    log_base_path = Path(log_base)
    results = []
    
    # 遍历实验组目录
    exp_group_path = log_base_path / experiment_group
    if not exp_group_path.exists():
        print(f"❌ 实验组目录不存在: {exp_group_path}")
        return pd.DataFrame()
    
    print(f"🔍 搜索微调实验结果: {exp_group_path}")
    
    # 查找所有finetune_metrics.json文件
    metrics_files = list(exp_group_path.rglob("finetune_metrics.json"))
    
    if not metrics_files:
        print(f"❌ 未找到微调结果文件，搜索路径: {exp_group_path}/**/finetune_metrics.json")
        return pd.DataFrame()
    
    print(f"📊 找到 {len(metrics_files)} 个结果文件")
    
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # 从路径解析实验名
            # 路径格式: log/{group}/{exp_name}/{dataset}/{method}/finetune/finetune_metrics.json
            path_parts = metrics_file.parts
            exp_name = path_parts[-5]     # experiment_name  
            dataset = path_parts[-4]      # dataset
            method = path_parts[-3]       # method
            
            # 解析实验编码
            pretrain_base = "Unknown"
            if exp_name.startswith("F") and "_" in exp_name:
                parts = exp_name.split("_", 1)
                pretrain_base = parts[0][1:]  # 去掉F前缀
            
            # 提取关键指标 (根据zinc回归任务)
            test_metrics = metrics.get('test', {})
            val_metrics = metrics.get('val', {})
            time_metrics = metrics.get('time', {})
            
            # 对于回归任务，主要关注MSE/RMSE，但这里用的是val_loss
            test_loss = test_metrics.get('val_loss', float('inf'))  # 实际是test loss
            val_loss = val_metrics.get('val_loss', float('inf'))
            train_time = time_metrics.get('total_train_time_sec', 0)
            avg_epoch_time = time_metrics.get('avg_epoch_time_sec', 0)
            
            result = {
                'experiment_name': exp_name,
                'pretrain_base': pretrain_base,
                'dataset': dataset,
                'method': method,
                'task': metrics.get('task', 'regression'),
                'test_loss': test_loss,
                'val_loss': val_loss,
                'best_val_loss': val_metrics.get('best_val_loss', val_loss),  # 如果有的话
                'train_time_sec': train_time,
                'avg_epoch_time_sec': avg_epoch_time,
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
    print(f"✅ 成功收集 {len(df)} 个微调实验结果")
    return df


def analyze_results_by_pretrain(df: pd.DataFrame, output_dir: Path):
    """按预训练基础分析结果"""
    
    if df.empty:
        return
    
    # 按预训练基础分组分析
    analysis_results = []
    
    for pretrain_base in df['pretrain_base'].unique():
        subset = df[df['pretrain_base'] == pretrain_base].copy()
        subset_sorted = subset.sort_values('test_loss', ascending=True)
        
        best_exp = subset_sorted.iloc[0]
        worst_exp = subset_sorted.iloc[-1]
        
        analysis = {
            'pretrain_base': pretrain_base,
            'num_experiments': len(subset),
            'best_experiment': best_exp['experiment_name'],
            'best_test_loss': best_exp['test_loss'],
            'best_val_loss': best_exp['val_loss'], 
            'worst_test_loss': worst_exp['test_loss'],
            'mean_test_loss': subset['test_loss'].mean(),
            'std_test_loss': subset['test_loss'].std(),
            'mean_train_time': subset['train_time_sec'].mean(),
            'improvement_over_worst': worst_exp['test_loss'] - best_exp['test_loss']
        }
        analysis_results.append(analysis)
    
    # 保存分析结果
    analysis_df = pd.DataFrame(analysis_results)
    analysis_df = analysis_df.sort_values('best_test_loss', ascending=True)
    
    analysis_file = output_dir / "finetune_analysis_by_pretrain.csv"
    analysis_df.to_csv(analysis_file, index=False)
    
    print(f"\n📈 按预训练基础的分析结果:")
    print(f"{'预训练基础':<12} {'实验数':<8} {'最佳测试损失':<12} {'最佳实验':<15} {'改进幅度':<10}")
    print("-" * 75)
    
    for _, row in analysis_df.iterrows():
        print(f"{row['pretrain_base']:<12} {row['num_experiments']:<8} "
              f"{row['best_test_loss']:<12.4f} {row['best_experiment']:<15} "
              f"{row['improvement_over_worst']:<10.4f}")
    
    return analysis_df


def analyze_augmentation_effects(df: pd.DataFrame, output_dir: Path):
    """分析增强方法的效果"""
    
    if df.empty:
        return
    
    print(f"\n🔍 增强方法效果分析:")
    
    # 解析增强配置（从实验名）
    augmentation_analysis = []
    
    for _, row in df.iterrows():
        exp_name = row['experiment_name']
        if not exp_name.startswith("F") or "_" not in exp_name:
            continue
            
        try:
            # 解析编码 F{pretrain}_{seq4位}{train3位}
            parts = exp_name.split("_")
            if len(parts) >= 2 and len(parts[1]) >= 7:
                seq_code = parts[1][:4]  # 前4位是序列级
                train_code = parts[1][4:7]  # 后3位是训练级
                
                # 序列级增强
                seq_augs = []
                if seq_code[0] == '1': seq_augs.append('deletion')
                if seq_code[1] == '1': seq_augs.append('swap') 
                if seq_code[2] == '1': seq_augs.append('truncation')
                if seq_code[3] == '1': seq_augs.append('masking')
                
                # 训练级增强
                train_augs = []
                if train_code[0] == '1': train_augs.append('consistency')
                if train_code[1] == '1': train_augs.append('noise')
                if train_code[2] == '1': train_augs.append('mixup')
                
                aug_info = {
                    'experiment_name': exp_name,
                    'pretrain_base': row['pretrain_base'],
                    'test_loss': row['test_loss'],
                    'seq_aug_count': len(seq_augs),
                    'train_aug_count': len(train_augs),
                    'total_aug_count': len(seq_augs) + len(train_augs),
                    'seq_augs': '+'.join(seq_augs) if seq_augs else 'None',
                    'train_augs': '+'.join(train_augs) if train_augs else 'None',
                    'has_consistency': 'consistency' in train_augs,
                    'has_mixup': 'mixup' in train_augs,
                    'has_noise': 'noise' in train_augs
                }
                augmentation_analysis.append(aug_info)
                
        except Exception as e:
            print(f"⚠️ 解析实验名失败: {exp_name} - {e}")
    
    if not augmentation_analysis:
        print("❌ 无法解析增强配置")
        return
    
    aug_df = pd.DataFrame(augmentation_analysis)
    
    # 保存详细分析
    aug_file = output_dir / "augmentation_effects.csv"
    aug_df.to_csv(aug_file, index=False)
    
    # 统计分析
    print(f"   总增强数量 vs 性能:")
    for count in range(8):  # 最多7种增强
        subset = aug_df[aug_df['total_aug_count'] == count]
        if len(subset) > 0:
            mean_loss = subset['test_loss'].mean()
            print(f"     {count} 种增强: {len(subset)} 个实验, 平均测试损失: {mean_loss:.4f}")
    
    # 特定增强方法分析
    print(f"   特定增强方法效果:")
    for aug_type in ['has_consistency', 'has_mixup', 'has_noise']:
        with_aug = aug_df[aug_df[aug_type] == True]['test_loss']
        without_aug = aug_df[aug_df[aug_type] == False]['test_loss']
        
        if len(with_aug) > 0 and len(without_aug) > 0:
            improvement = without_aug.mean() - with_aug.mean()
            method_name = aug_type.replace('has_', '')
            print(f"     {method_name}: 改进 {improvement:+.4f} "
                  f"(有: {with_aug.mean():.4f}, 无: {without_aug.mean():.4f})")
    
    return aug_df


def main():
    parser = argparse.ArgumentParser(description="收集微调实验结果")
    parser.add_argument("--experiment_group", type=str, default="aug_pretrain", 
                       help="实验组名称")
    parser.add_argument("--log_base", type=str, default="log", 
                       help="日志根目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"🚀 开始收集微调结果...")
    print(f"   实验组: {args.experiment_group}")
    print(f"   日志目录: {args.log_base}")
    print(f"   输出目录: {output_dir}")
    
    # 收集结果
    df = collect_finetune_results(args.experiment_group, args.log_base)
    
    if df.empty:
        print("❌ 未找到任何结果，请检查实验是否完成")
        sys.exit(1)
    
    # 保存原始结果
    df_sorted = df.sort_values('test_loss', ascending=True)
    results_file = output_dir / "finetune_results.csv"
    df_sorted.to_csv(results_file, index=False)
    print(f"💾 完整结果已保存: {results_file}")
    
    # 分析结果
    print(f"\n📊 微调实验总结:")
    print(f"   总实验数: {len(df)}")
    print(f"   最佳测试损失: {df['test_loss'].min():.4f} (实验: {df_sorted.iloc[0]['experiment_name']})")
    print(f"   最差测试损失: {df['test_loss'].max():.4f}")
    print(f"   平均测试损失: {df['test_loss'].mean():.4f} ± {df['test_loss'].std():.4f}")
    
    # 按预训练基础分析
    analyze_results_by_pretrain(df, output_dir)
    
    # 增强方法效果分析
    analyze_augmentation_effects(df, output_dir)
    
    # 输出最佳结果
    print(f"\n🏆 Top-10 微调实验:")
    print(f"{'排名':<4} {'实验名':<20} {'预训练基础':<10} {'测试损失':<12} {'训练时间':<10}")
    print("-" * 70)
    
    for idx, row in df_sorted.head(10).iterrows():
        rank = df_sorted.index.get_loc(idx) + 1
        print(f"{rank:<4} {row['experiment_name']:<20} {row['pretrain_base']:<10} "
              f"{row['test_loss']:<12.4f} {row['train_time_sec']:<10.0f}")


if __name__ == "__main__":
    main()
