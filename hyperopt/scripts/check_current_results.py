#!/usr/bin/env python
"""
直接查看当前超参数搜索的结果
"""
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import pandas as pd

def main():
    journal_file = "./journal/zinc_hyperopt.db"
    
    try:
        # 连接到存储
        storage = JournalStorage(JournalFileStorage(journal_file))
        
        # 尝试直接加载已知的study
        study_names = ["zinc_hyperopt_pretrain_all", "zinc_hyperopt_finetune_all"]
        
        print("📋 检查已知的研究:")
        print("=" * 60)
        
        for study_name in study_names:
            print(f"\n🔬 研究: {study_name}")
            
            try:
                # 加载具体study
                study = optuna.load_study(
                    study_name=study_name, 
                    storage=storage
                )
                
                print(f"   试验总数: {len(study.trials)}")
                print(f"   完成试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
                print(f"   剪枝试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
                print(f"   失败试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
                
                # 如果有完成的试验，显示最佳结果
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if completed_trials:
                    best_trial = study.best_trial
                    print(f"   🏆 最佳值: {study.best_value:.6f}")
                    print(f"   📊 最佳参数:")
                    for key, value in best_trial.params.items():
                        print(f"      {key}: {value}")
                    
                    # 显示最近几个完成的试验
                    print(f"   📈 最近完成的试验:")
                    recent_trials = sorted(completed_trials, key=lambda x: x.number, reverse=True)[:5]
                    for trial in recent_trials:
                        print(f"      Trial {trial.number:3d}: {trial.value:.6f} | {trial.params}")
                
            except Exception as e:
                print(f"   ❌ 加载study失败: {e}")
        
        # 也尝试检查其他可能的study名称
        print(f"\n🔍 扫描journal文件中的内容...")
        try:
            with open(journal_file, 'r') as f:
                content = f.read()
                if 'zinc_hyperopt_pretrain' in content:
                    print("   ✅ 发现预训练相关记录")
                if 'zinc_hyperopt_finetune' in content:
                    print("   ✅ 发现微调相关记录")
                    
        except Exception as e:
            print(f"   ❌ 读取文件失败: {e}")
                
        print("\n" + "=" * 60)
        print("💡 提示:")
        print("   - Web Dashboard: http://localhost:8080")
        print(f"   - Journal文件: {journal_file}")
        print("   - 可以通过dashboard查看详细的参数图表和历史")
        print("   - 如果没有数据，请确保超参数搜索正在运行")
        
    except Exception as e:
        print(f"❌ 连接存储失败: {e}")

if __name__ == "__main__":
    main()
