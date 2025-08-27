#!/usr/bin/env python3
"""
从Optuna数据库中提取最佳参数，用于微调优化搜索
特别提取top3总体结果和各序列化方法的top1结果
"""
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

def load_study_results(journal_file, target_study=None):
    """从Journal存储中加载研究结果"""
    from optuna.storages.journal import JournalFileBackend
    storage = JournalStorage(JournalFileBackend(journal_file))
    
    # 获取所有study信息
    all_studies = storage.get_all_studies()
    study_names = [study.study_name for study in all_studies]
    
    print(f"🔍 发现 {len(study_names)} 个研究:")
    for name in study_names:
        print(f"  - {name}")
    
    # 如果指定了目标study，只分析该study
    if target_study:
        if target_study in study_names:
            study_names = [target_study]
            print(f"🎯 专注分析: {target_study}")
        else:
            print(f"❌ 未找到目标研究: {target_study}")
    
    return study_names, storage

def extract_best_params(storage, study_names):
    """提取最佳参数组合"""
    all_completed_trials = []
    
    for study_name in study_names:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            for trial in completed_trials:
                trial_info = {
                    'study_name': study_name,
                    'trial_number': trial.number,
                    'value': trial.value,
                    'lr': trial.params.get('lr'),
                    'bs': trial.params.get('bs'),
                    'wd': trial.params.get('wd'),
                    'grad_norm': trial.params.get('grad_norm'),
                    'mask_prob': trial.params.get('mask_prob'),
                    'warmup_ratio': trial.params.get('warmup_ratio'),
                    'training_time_minutes': trial.user_attrs.get('training_time_minutes', np.nan),
                    'bpe_mode': trial.user_attrs.get('bpe_mode'),
                    'serialization_method': trial.params.get('method', 'unknown'),
                }
                all_completed_trials.append(trial_info)
                
        except Exception as e:
            print(f"❌ 处理研究 {study_name} 失败: {e}")
            
    return pd.DataFrame(all_completed_trials)

def analyze_and_extract_params(df):
    """分析并提取关键参数组合"""
    if df.empty:
        print("⚠️ 没有完成的试验数据")
        return {}
    
    # 总体Top 3
    print(f"\n🏆 总体Top 3最佳结果:")
    top3_overall = df.nsmallest(3, 'value')
    
    best_params = {}
    
    for i, (_, row) in enumerate(top3_overall.iterrows(), 1):
        print(f"\n  🥇 Top {i}: loss={row['value']:.4f}")
        print(f"    Study: {row['study_name']}")
        print(f"    Method: {row['serialization_method']}")
        print(f"    lr={row['lr']:.4e}, bs={row['bs']}, wd={row['wd']:.3f}")
        print(f"    grad_norm={row['grad_norm']:.2f}, mask_prob={row['mask_prob']:.3f}, warmup={row['warmup_ratio']:.3f}")
        
        best_params[f'top{i}_overall'] = {
            'lr': row['lr'],
            'bs': row['bs'],
            'wd': row['wd'],
            'grad_norm': row['grad_norm'],
            'mask_prob': row['mask_prob'],
            'warmup_ratio': row['warmup_ratio'],
            'method': row['serialization_method'],
            'loss': row['value']
        }
    
    # 各序列化方法的Top 1
    print(f"\n🎯 各序列化方法Top 1结果:")
    methods = ['eulerian', 'feuler', 'cpp', 'fcpp']
    
    for method in methods:
        method_df = df[df['serialization_method'] == method]
        if method_df.empty:
            print(f"  ⚠️ {method}: 没有数据")
            continue
            
        best_method = method_df.loc[method_df['value'].idxmin()]
        print(f"\n  🎯 {method}: loss={best_method['value']:.4f}")
        print(f"    lr={best_method['lr']:.4e}, bs={best_method['bs']}, wd={best_method['wd']:.3f}")
        print(f"    grad_norm={best_method['grad_norm']:.2f}, mask_prob={best_method['mask_prob']:.3f}, warmup={best_method['warmup_ratio']:.3f}")
        
        best_params[f'{method}_best'] = {
            'lr': best_method['lr'],
            'bs': best_method['bs'],
            'wd': best_method['wd'],
            'grad_norm': best_method['grad_norm'],
            'mask_prob': best_method['mask_prob'],
            'warmup_ratio': best_method['warmup_ratio'],
            'method': method,
            'loss': best_method['value']
        }
    
    return best_params

def create_default_config_params():
    """创建默认配置参数"""
    # 这里应该基于项目默认配置设置
    return {
        'default_config': {
            'lr': 5e-4,
            'bs': 32,
            'wd': 0.01,
            'grad_norm': 1.0,
            'mask_prob': 0.15,
            'warmup_ratio': 0.1,
            'method': 'default',
            'loss': None  # 需要训练才知道
        }
    }

def save_best_params_for_finetuning(best_params, output_file):
    """保存最佳参数用于微调"""
    print(f"\n💾 保存最佳参数到: {output_file}")
    
    # 转换为JSON可序列化的格式
    serializable_params = {}
    for key, params in best_params.items():
        serializable_params[key] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in params.items()
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_params, f, indent=2, ensure_ascii=False)

def create_finetuning_search_ranges(best_params):
    """基于最佳参数创建微调搜索范围"""
    print(f"\n🔍 建议的微调搜索范围:")
    
    # 收集所有最佳参数的统计信息
    lr_values = [p['lr'] for p in best_params.values() if p.get('lr')]
    wd_values = [p['wd'] for p in best_params.values() if p.get('wd')]
    warmup_values = [p['warmup_ratio'] for p in best_params.values() if p.get('warmup_ratio')]
    
    lr_min, lr_max = min(lr_values) * 0.5, max(lr_values) * 2.0
    wd_min, wd_max = min(wd_values) * 0.5, max(wd_values) * 2.0
    warmup_min, warmup_max = min(warmup_values) * 0.5, max(warmup_values) * 2.0
    
    search_ranges = {
        'lr': [lr_min, lr_max],
        'wd': [wd_min, wd_max],
        'warmup_ratio': [warmup_min, warmup_max],
        'batch_sizes': [32, 64, 128],  # 微调通常用较小的batch size
        'epochs': [3, 5, 10],
        'grad_norm': [0.5, 2.0]
    }
    
    print(f"  Learning Rate: [{lr_min:.2e}, {lr_max:.2e}]")
    print(f"  Weight Decay: [{wd_min:.3f}, {wd_max:.3f}]")
    print(f"  Warmup Ratio: [{warmup_min:.3f}, {warmup_max:.3f}]")
    print(f"  Batch Sizes: {search_ranges['batch_sizes']}")
    print(f"  Epochs: {search_ranges['epochs']}")
    print(f"  Grad Norm: {search_ranges['grad_norm']}")
    
    return search_ranges

def main():
    parser = argparse.ArgumentParser(description='提取最佳参数用于微调优化搜索')
    parser.add_argument('--journal', default='../journal/large_batch.db', help='Journal数据库文件')
    parser.add_argument('--output_dir', default='../results', help='输出目录')
    parser.add_argument('--target_study', default='methods_large_batch_pretrain_all', help='目标研究名称')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载结果 - 专门分析methods study
    study_names, storage = load_study_results(args.journal, args.target_study)
    
    # 提取所有试验数据
    df = extract_best_params(storage, study_names)
    
    if df.empty:
        print("❌ 没有找到完成的试验数据")
        return
    
    # 分析并提取最佳参数
    best_params = analyze_and_extract_params(df)
    
    # 添加默认配置参数
    default_params = create_default_config_params()
    best_params.update(default_params)
    
    # 保存最佳参数
    output_file = Path(args.output_dir) / "best_pretrain_params_for_finetuning.json"
    save_best_params_for_finetuning(best_params, output_file)
    
    # 创建搜索范围建议
    search_ranges = create_finetuning_search_ranges(best_params)
    
    # 保存搜索范围
    search_ranges_file = Path(args.output_dir) / "finetuning_search_ranges.json"
    with open(search_ranges_file, 'w', encoding='utf-8') as f:
        json.dump(search_ranges, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完成! 参数保存到:")
    print(f"  最佳参数: {output_file}")
    print(f"  搜索范围: {search_ranges_file}")

if __name__ == '__main__':
    main()

