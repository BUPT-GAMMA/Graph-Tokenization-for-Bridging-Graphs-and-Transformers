#!/usr/bin/env python
"""
简单查看超参数搜索结果（无需matplotlib）
"""
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import pandas as pd
from pathlib import Path

def main():
    journal_file = "./journal/zinc_hyperopt.db"
    
    print("📋 分析超参数搜索结果...")
    
    try:
        print("🔗 连接存储...")
        storage = JournalStorage(JournalFileStorage(journal_file))
        study_name = "zinc_hyperopt_pretrain_all"
        
        print(f"📊 加载研究: {study_name}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        
        print(f"✅ 成功加载研究!")
        print(f"   试验总数: {len(study.trials)}")
        
        # 分类统计
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        
        print(f"   完成试验: {len(completed)}")
        print(f"   剪枝试验: {len(pruned)}")  
        print(f"   失败试验: {len(failed)}")
        print(f"   运行中: {len(running)}")
        
        if completed:
            print(f"\n🏆 最佳结果:")
            print(f"   最佳值: {study.best_value:.6f}")
            print(f"   最佳参数:")
            for key, value in study.best_params.items():
                print(f"     {key}: {value}")
            
            # 创建详细表格
            print(f"\n📈 所有完成的试验:")
            data = []
            for trial in completed:
                row = {
                    'trial': trial.number,
                    'loss': trial.value,
                    **trial.params
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df_sorted = df.sort_values('loss')
            
            print("=" * 100)
            print(df_sorted.to_string(index=False, float_format='%.6f'))
            print("=" * 100)
            
            # 保存CSV
            csv_file = "hyperopt_results.csv"
            df_sorted.to_csv(csv_file, index=False)
            print(f"💾 结果已保存到: {csv_file}")
            
            # 统计分析
            print(f"\n📊 参数统计:")
            for param in study.best_params.keys():
                values = [trial.params[param] for trial in completed if param in trial.params]
                if values:
                    if isinstance(values[0], (int, float)):
                        print(f"   {param}: min={min(values):.6f}, max={max(values):.6f}, mean={sum(values)/len(values):.6f}")
                    else:
                        from collections import Counter
                        counter = Counter(values)
                        print(f"   {param}: {dict(counter)}")
            
        else:
            print("❌ 没有完成的试验")
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
