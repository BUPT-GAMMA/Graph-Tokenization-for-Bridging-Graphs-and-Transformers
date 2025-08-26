#!/usr/bin/env python3
"""
从Optuna数据库中分析超参数搜索结果
特别关注不同batch size的性能vs时间权衡
"""
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_study_results(journal_file, study_name_pattern=None):
    """从Journal存储中加载研究结果"""
    from optuna.storages.journal import JournalFileBackend
    storage = JournalStorage(JournalFileBackend(journal_file))
    
    # 获取所有study信息
    all_studies = storage.get_all_studies()
    study_names = [study.study_name for study in all_studies]
    
    if study_name_pattern:
        study_names = [name for name in study_names if study_name_pattern in name]
    
    print(f"🔍 发现 {len(study_names)} 个研究:")
    for name in study_names:
        print(f"  - {name}")
    
    return study_names, storage

def analyze_pretrain_results(storage, study_name):
    """分析预训练结果"""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        print(f"\n📊 {study_name} 分析:")
        print(f"  试验总数: {len(study.trials)}")
        print(f"  完成试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"  失败试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        print(f"  剪枝试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        
        # 分析完成的试验
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print("  ⚠️ 没有完成的试验")
            return None
            
        # 创建DataFrame进行分析
        data = []
        for trial in completed_trials:
            row = {
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
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 按batch size分组分析
        print(f"\n🎯 按Batch Size分析 (Top 10 overall):")
        top_10 = df.nsmallest(10, 'value')
        for _, row in top_10.iterrows():
            time_str = f"{row['training_time_minutes']:.1f}min" if not pd.isna(row['training_time_minutes']) else "时间未知"
            print(f"  Trial {row['trial_number']:3d}: loss={row['value']:.4f}, bs={row['bs']:3d}, time={time_str}")
            print(f"    lr={row['lr']:.2e}, wd={row['wd']:.3f}, warmup={row['warmup_ratio']:.3f}")
        
        # 按batch size分组的最佳结果
        print(f"\n🏆 各Batch Size最佳结果:")
        for bs in sorted(df['bs'].unique()):
            bs_df = df[df['bs'] == bs]
            best = bs_df.loc[bs_df['value'].idxmin()]
            time_str = f"{best['training_time_minutes']:.1f}min" if not pd.isna(best['training_time_minutes']) else "时间未知"
            print(f"  BS={bs:3d}: loss={best['value']:.4f}, time={time_str} (Trial {best['trial_number']})")
            print(f"    lr={best['lr']:.2e}, wd={best['wd']:.3f}, warmup={best['warmup_ratio']:.3f}")
        
        # 效率分析
        if not df['training_time_minutes'].isna().all():
            print(f"\n⚡ 效率分析:")
            df_time = df.dropna(subset=['training_time_minutes'])
            for bs in sorted(df_time['bs'].unique()):
                bs_df = df_time[df_time['bs'] == bs]
                avg_time = bs_df['training_time_minutes'].mean()
                best_loss = bs_df['value'].min()
                print(f"  BS={bs:3d}: 平均时间={avg_time:.1f}min, 最佳loss={best_loss:.4f}")
        
        return df
        
    except Exception as e:
        print(f"❌ 加载研究失败: {e}")
        return None

def suggest_good_starts(df, target_batch_sizes):
    """基于已有结果建议好的起始点"""
    if df is None or df.empty:
        print("⚠️ 没有数据可供分析")
        return
        
    print(f"\n💡 建议的起始参数 (基于已有最佳结果):")
    
    for bs in target_batch_sizes:
        # 找到该batch size或相近的最佳结果
        available_bs = sorted(df['bs'].unique())
        closest_bs = min(available_bs, key=lambda x: abs(x - bs))
        
        bs_df = df[df['bs'] == closest_bs]
        if bs_df.empty:
            continue
            
        best = bs_df.loc[bs_df['value'].idxmin()]
        
        print(f"\n  推荐用于 BS={bs} 的起始点 (基于BS={closest_bs}最佳结果):")
        print(f"    lr: {best['lr']:.2e}")
        print(f"    wd: {best['wd']:.3f}")
        print(f"    grad_norm: {best['grad_norm']:.2f}")
        print(f"    mask_prob: {best['mask_prob']:.3f}")
        print(f"    warmup_ratio: {best['warmup_ratio']:.3f}")

def add_good_trials_to_study(storage, source_study_name, target_study_name, good_trials_params, bpe_mode):
    """将好的试验参数添加到新的研究中作为起始点"""
    try:
        # 创建或加载目标研究
        target_study = optuna.create_study(
            study_name=target_study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )
        
        print(f"\n🎯 向研究 {target_study_name} 添加好的起始点:")
        
        for i, params in enumerate(good_trials_params):
            # 创建一个新的trial并设置参数
            trial = target_study.ask()
            
            # 设置建议的参数值
            for param_name, param_value in params.items():
                if param_name in ['lr', 'wd', 'grad_norm', 'mask_prob', 'warmup_ratio']:
                    trial.suggest_float(param_name, param_value, param_value)  # 固定值
                elif param_name == 'bs':
                    trial.suggest_categorical(param_name, [param_value])  # 固定值
            
            print(f"  添加起始点 {i+1}: {params}")
            
            # 注意：这些trial需要实际运行才会有值，这里只是添加了参数建议
            
    except Exception as e:
        print(f"❌ 添加起始点失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='分析Optuna超参数搜索结果')
    parser.add_argument('--journal', default='journal/zinc_hyperopt.db', help='Journal数据库文件')
    parser.add_argument('--bpe_mode', default='all', help='BPE模式')
    parser.add_argument('--target_bs', nargs='+', type=int, default=[256, 512], help='目标batch sizes')
    
    args = parser.parse_args()
    
    # 加载结果
    study_names, storage = load_study_results(args.journal, f"pretrain_{args.bpe_mode}")
    
    # 分析每个预训练研究
    all_results = []
    for study_name in study_names:
        if 'pretrain' in study_name:
            df = analyze_pretrain_results(storage, study_name)
            if df is not None:
                all_results.append(df)
    
    # 合并所有结果进行综合分析
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"\n🌟 综合分析 (总计 {len(combined_df)} 个试验):")
        
        # 建议好的起始点
        suggest_good_starts(combined_df, args.target_bs)
        
        # 保存结果到CSV
        output_file = f"latest_pretrain_results_{args.bpe_mode}.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\n💾 结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
