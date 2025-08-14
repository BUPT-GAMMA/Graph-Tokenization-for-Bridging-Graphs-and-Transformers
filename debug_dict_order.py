#!/usr/bin/env python3
"""
调试字典顺序问题
"""

def test_dict_max_behavior():
    """测试Python字典的max行为"""
    
    # 模拟第3轮的情况：多个pairs频次相同
    stats1 = {(3, 9): 1, (8, 3): 1, (8, 4): 1}
    stats2 = {(8, 3): 1, (3, 9): 1, (8, 4): 1}  # 不同的插入顺序
    
    print("字典1:", stats1)
    print("max选择:", max(stats1, key=stats1.get))
    
    print("字典2:", stats2) 
    print("max选择:", max(stats2, key=stats2.get))
    
    # 模拟构建顺序
    print("\n模拟序列中的pair出现顺序:")
    sequence = [8, 3, 7, 3, 9, 7, 8, 4, 7, 9, 6]  # 第2轮merge后
    separator = 7
    
    print(f"序列: {sequence}")
    
    # 按出现顺序收集pairs
    pairs_in_order = []
    for i in range(len(sequence) - 1):
        if sequence[i] != separator and sequence[i+1] != separator:
            pair = (sequence[i], sequence[i+1])
            if pair not in pairs_in_order:
                pairs_in_order.append(pair)
    
    print(f"按出现顺序的pairs: {pairs_in_order}")
    
    # 构建stats字典
    stats = {}
    for pair in pairs_in_order:
        count = 0
        for i in range(len(sequence) - 1):
            if (sequence[i], sequence[i+1]) == pair:
                count += 1
        if count > 0:
            stats[pair] = count
    
    print(f"构建的stats: {stats}")
    print(f"max选择: {max(stats, key=stats.get)}")

if __name__ == "__main__":
    test_dict_max_behavior()
