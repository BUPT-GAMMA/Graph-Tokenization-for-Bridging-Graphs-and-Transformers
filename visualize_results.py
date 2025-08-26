#!/usr/bin/env python
"""
可视化超参数搜索结果
"""
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_optimization_history(study, save_dir="./plots"):
    """绘制优化历史"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("❌ 没有完成的试验，无法绘制优化历史")
        return
    
    trial_numbers = [t.number for t in completed_trials]
    values = [t.value for t in completed_trials]
    
    ax.plot(trial_numbers, values, 'o-', alpha=0.7, linewidth=2, markersize=6)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Objective Value (Loss)')
    ax.set_title(f'Optimization History - {study.study_name}')
    ax.grid(True, alpha=0.3)
    
    # 标注最佳点
    best_idx = np.argmin(values)
    ax.annotate(f'Best: {values[best_idx]:.4f}', 
                xy=(trial_numbers[best_idx], values[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/optimization_history_{study.study_name}.png", dpi=300)
    plt.close()  # 释放内存
    print(f"✅ 优化历史图已保存: {save_dir}/optimization_history_{study.study_name}.png")

def plot_param_importance(study, save_dir="./plots"):
    """绘制参数重要性"""
    Path(save_dir).mkdir(exist_ok=True)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) < 3:
        print("❌ 完成的试验太少（<3），无法计算参数重要性")
        return
    
    try:
        importance = optuna.importance.get_param_importances(study)
        
        params = list(importance.keys())
        values = list(importance.values())
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.barh(params, values)
        ax.set_xlabel('Importance')
        ax.set_title(f'Parameter Importance - {study.study_name}')
        
        # 颜色渐变
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(values[i] / max(values)))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/param_importance_{study.study_name}.png", dpi=300)
        plt.close()  # 释放内存
        print(f"✅ 参数重要性图已保存: {save_dir}/param_importance_{study.study_name}.png")
        
    except Exception as e:
        print(f"❌ 计算参数重要性失败: {e}")

def create_results_table(study):
    """创建结果表格"""
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return None
    
    data = []
    for trial in completed_trials:
        row = {'trial': trial.number, 'value': trial.value}
        row.update(trial.params)
        data.append(row)
    
    df = pd.DataFrame(data)
    return df.sort_values('value').head(10)  # Top 10

def main():
    journal_file = "./journal/zinc_hyperopt.db"
    
    print("🎨 创建超参数搜索结果可视化...")
    
    try:
        storage = JournalStorage(JournalFileStorage(journal_file))
        study_name = "zinc_hyperopt_pretrain_all"
        
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        
        print(f"📊 研究: {study_name}")
        print(f"   试验总数: {len(study.trials)}")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"   完成试验: {len(completed_trials)}")
        
        if completed_trials:
            print(f"   🏆 最佳值: {study.best_value:.6f}")
            print(f"   📊 最佳参数: {study.best_params}")
            
            # 创建Top 10表格
            top_10 = create_results_table(study)
            if top_10 is not None:
                print(f"\n📈 Top 10 试验:")
                print(top_10.round(6))
                
                # 保存到CSV
                top_10.to_csv("top_10_trials.csv", index=False)
                print(f"💾 Top 10 结果已保存到: top_10_trials.csv")
            
            # 绘图
            print(f"\n🎨 生成可视化图表...")
            plot_optimization_history(study)
            plot_param_importance(study)
            
            print(f"✅ 可视化完成！图表保存在 ./plots/ 目录")
        else:
            print("❌ 没有完成的试验")
            
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")

if __name__ == "__main__":
    # 设置中文字体和样式
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    main()
