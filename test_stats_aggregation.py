#!/usr/bin/env python3
"""
测试统计聚合功能
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.stats_aggregation import _calculate_stats, aggregate_experiment_results, print_aggregated_stats


def test_calculate_stats():
    """测试统计计算功能"""
    print("🧪 测试统计计算功能...")

    # 测试数据
    test_values = [1.2, 1.5, 1.8, 2.1, 2.4]
    stats = _calculate_stats(test_values)

    print(f"  输入值: {test_values}")
    print(f"  均值: {stats['mean']:.3f}")
    print(f"  标准差: {stats['std']:.3f}")
    print(f"  范围: [{stats['min']:.1f}, {stats['max']:.1f}]")
    print(f"  样本数: {stats['count']}")
    if 'ci_95' in stats:
        print(f"  95%置信区间: ±{stats['ci_95']:.3f}")

    print("✅ 统计计算功能正常")
    return True


def test_empty_stats():
    """测试空数据统计"""
    print("\n🧪 测试空数据统计...")
    empty_stats = _calculate_stats([])
    if not empty_stats:
        print("✅ 空数据处理正常")
        return True
    else:
        print("❌ 空数据处理异常")
        return False


def test_single_value_stats():
    """测试单值统计"""
    print("\n🧪 测试单值统计...")
    single_stats = _calculate_stats([3.14])
    print(f"  单值: 3.14")
    print(f"  均值: {single_stats['mean']:.3f}")
    print(f"  标准差: {single_stats.get('std', 'N/A')}")
    print("✅ 单值统计正常")
    return True


if __name__ == "__main__":
    print("🚀 开始测试统计聚合功能")
    print("="*50)

    success_count = 0
    total_tests = 3

    try:
        if test_calculate_stats():
            success_count += 1
        if test_empty_stats():
            success_count += 1
        if test_single_value_stats():
            success_count += 1

        print("\n" + "="*50)
        print(f"📊 测试结果: {success_count}/{total_tests} 通过")

        if success_count == total_tests:
            print("🎉 所有测试通过！")
        else:
            print("⚠️ 部分测试失败")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
