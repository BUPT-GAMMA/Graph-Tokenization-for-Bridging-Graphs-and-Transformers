#!/usr/bin/env python
"""
停止超参数搜索 - 通过检查试验数量来限制执行
"""
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

def check_search_status():
    journal_file = "./journal/zinc_hyperopt.db"
    
    try:
        storage = JournalStorage(JournalFileStorage(journal_file))
        study = optuna.load_study(
            study_name="zinc_hyperopt_pretrain_all",
            storage=storage
        )
        
        total_trials = len(study.trials)
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        print(f"📊 当前状态:")
        print(f"   总试验数: {total_trials}")
        print(f"   完成试验: {completed}")
        print(f"   目标数量: 30")
        print(f"   超出数量: {total_trials - 30}")
        
        if completed >= 30:
            print(f"✅ 预训练搜索已超过目标，建议停止后续试验")
            print(f"🏆 最佳结果: {study.best_value:.6f}")
            
            # 建议top-K模型用于微调
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            sorted_trials = sorted(completed_trials, key=lambda x: x.value)
            top_k = sorted_trials[:5]
            
            print(f"\n🎯 推荐用于微调的Top-5试验:")
            for i, trial in enumerate(top_k, 1):
                print(f"   {i}. Trial {trial.number}: loss={trial.value:.6f}")
                print(f"      {trial.params}")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")

if __name__ == "__main__":
    check_search_status()
