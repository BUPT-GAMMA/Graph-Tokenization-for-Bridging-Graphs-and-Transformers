#!/usr/bin/env python3
"""
查看超参数搜索结果
"""

import argparse
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from pathlib import Path


def show_study_results(journal_file, study_name):
    """显示研究结果"""
    if not Path(journal_file).exists():
        print(f"❌ Journal文件不存在: {journal_file}")
        return
    
    storage = JournalStorage(JournalFileStorage(journal_file))
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f"❌ 找不到研究: {study_name}")
        print("可用的研究:")
        try:
            studies = storage.get_all_study_summaries()
            for s in studies:
                print(f"  - {s.study_name}")
        except:
            pass
        return
    
    print(f"📊 研究: {study_name}")
    print(f"🎯 方向: {'最小化' if study.direction == optuna.study.StudyDirection.MINIMIZE else '最大化'}")
    print(f"📈 试验数: {len(study.trials)}")
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        print(f"✅ 完成: {len(completed)}")
        print(f"🏆 最优值: {study.best_value:.4f}")
        print(f"🎯 最优参数:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        
        # Top-5结果
        sorted_trials = sorted(completed, key=lambda t: t.value)
        print(f"\n🏆 Top-5 结果:")
        for i, trial in enumerate(sorted_trials[:5]):
            print(f"  #{i+1}: Trial {trial.number}, 值={trial.value:.4f}")
            print(f"      参数: {trial.params}")
    
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if pruned:
        print(f"✂️ 剪枝: {len(pruned)}")
    
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    if failed:
        print(f"❌ 失败: {len(failed)}")


def main():
    parser = argparse.ArgumentParser(description="查看超参数搜索结果")
    parser.add_argument("--journal_dir", default="./journals", help="Journal目录")
    parser.add_argument("--bpe_mode", help="指定BPE模式")
    
    args = parser.parse_args()
    
    journal_dir = Path(args.journal_dir)
    if not journal_dir.exists():
        print(f"❌ Journal目录不存在: {journal_dir}")
        return
    
    if args.bpe_mode:
        # 显示指定BPE模式的结果
        modes = [args.bpe_mode]
    else:
        # 显示所有模式的结果
        modes = ['none', 'all', 'random', 'gaussian']
    
    for mode in modes:
        journal_file = journal_dir / f"zinc_{mode}.journal"
        
        print("="*60)
        print(f"🎛️ BPE模式: {mode}")
        print("="*60)
        
        # 预训练结果
        print("\n🔍 预训练阶段:")
        show_study_results(str(journal_file), f"zinc_hyperopt_pretrain_{mode}")
        
        # 微调结果  
        print("\n🎯 微调阶段:")
        show_study_results(str(journal_file), f"zinc_hyperopt_finetune_{mode}")


if __name__ == "__main__":
    main()
