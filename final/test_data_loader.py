#!/usr/bin/env python3
"""
实验数据加载器测试脚本
演示如何使用experiment_data_loader.py加载和分析实验数据
"""

import sys
from pathlib import Path
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from final.experiment_data_loader import load_experiment_data


def test_basic_loading():
    """测试基本数据加载功能"""
    print("🧪 测试基本数据加载...")

    # 加载所有实验数据
    df = load_experiment_data()

    if df.empty:
        print("❌ 未加载到任何数据")
        return

    print(f"✅ 成功加载 {len(df)} 个实验结果")
    print(f"📊 数据包含 {len(df.columns)} 个字段")
    print(f"📁 涉及 {df['experiment_group'].nunique()} 个实验组")
    print(f"🏷️ 涉及 {df['dataset'].nunique()} 个数据集")
    print(f"🔧 涉及 {df['method'].nunique()} 个序列化方法")

    return df


def test_filtering(df):
    """测试数据过滤功能"""
    print("\n🔍 测试数据过滤...")

    # 过滤特定数据集
    qm9_data = df[df['dataset'] == 'qm9']
    print(f"QM9数据集实验数量: {len(qm9_data)}")

    # 过滤特定方法
    gt_methods = ['feuler', 'cpp']
    gt_data = df[df['method'].isin(gt_methods)]
    print(f"GT方法(feuler/cpp)实验数量: {len(gt_data)}")

    # 过滤回归任务
    regression_data = df[df['task_type'] == 'regression']
    print(f"回归任务实验数量: {len(regression_data)}")

    return gt_data


def test_performance_analysis(df):
    """测试性能分析功能"""
    print("\n📈 测试性能分析...")

    # 按方法分析MAE性能 (回归任务)
    regression_df = df[df['task_type'] == 'regression']

    if not regression_df.empty:
        method_mae = regression_df.groupby('method')['test_mae_best'].mean()
        print("各方法平均MAE性能 (越小越好):")
        print(method_mae.sort_values().head(10))

    # 按数据集分析性能
    dataset_mae = regression_df.groupby('dataset')['test_mae_best'].mean()
    print("\n各数据集平均MAE性能:")
    print(dataset_mae.sort_values().head(10))

    # 分析训练时间
    time_analysis = df.groupby('method')['total_train_time_sec'].mean()
    print("\n各方法平均训练时间 (秒):")
    print(time_analysis.sort_values().head(10))


def test_config_analysis(df):
    """测试配置分析功能"""
    print("\n⚙️ 测试配置分析...")

    # 分析学习率分布
    if 'finetune_learning_rate' in df.columns:
        lr_counts = df['finetune_learning_rate'].value_counts()
        print("学习率使用统计:")
        print(lr_counts.head())

    # 分析模型尺寸分布
    if 'd_model' in df.columns:
        model_sizes = df.groupby(['d_model', 'n_layers']).size()
        print("\n模型尺寸分布:")
        print(model_sizes.sort_values(ascending=False).head())

    # 分析BPE方法分布
    if 'bpe_method' in df.columns:
        bpe_dist = df['bpe_method'].value_counts()
        print("\nBPE方法分布:")
        print(bpe_dist)


def test_experiment_structure(df):
    """测试实验结构分析"""
    print("\n🏗️ 测试实验结构分析...")

    # 分析实验组分布
    group_dist = df['experiment_group'].value_counts()
    print("实验组分布:")
    print(group_dist.head())

    # 分析数据集-方法组合
    dataset_method = df.groupby(['dataset', 'method']).size()
    print("\n数据集-方法组合统计 (前10):")
    print(dataset_method.sort_values(ascending=False).head(10))


def main():
    """主测试函数"""
    print("🚀 开始测试实验数据加载器\n")

    # 测试基本加载
    df = test_basic_loading()
    if df is None or df.empty:
        print("❌ 测试失败：无法加载数据")
        return

    # 测试过滤功能
    filtered_df = test_filtering(df)

    # 测试性能分析
    test_performance_analysis(df)

    # 测试配置分析
    test_config_analysis(df)

    # 测试实验结构分析
    test_experiment_structure(df)

    print("\n✅ 所有测试完成！")
    print("💡 提示：可以根据需要修改过滤条件来分析特定实验子集")


if __name__ == "__main__":
    main()
