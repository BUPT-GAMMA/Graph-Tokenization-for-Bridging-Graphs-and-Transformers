#!/usr/bin/env python3
"""
简单的平局测试
"""

def test_tie_breaking():
    """测试平局时的选择策略"""
    
    # 模拟一个简单的平局情况
    pairs_with_same_freq = [
        ((3, 9), 1),
        ((8, 3), 1),
        ((8, 4), 1),
    ]
    
    print("模拟的平局情况 (频次都是1):")
    for pair, freq in pairs_with_same_freq:
        print(f"  {pair}: {freq}")
    
    # 不同的平局打破策略
    print("\n不同的平局打破策略:")
    
    # 策略1: 按字典序
    sorted_by_lexical = sorted(pairs_with_same_freq, key=lambda x: x[0])
    print(f"字典序策略: {sorted_by_lexical[0][0]}")
    
    # 策略2: 按第一个token ID
    sorted_by_first_token = sorted(pairs_with_same_freq, key=lambda x: x[0][0])
    print(f"按第一个token策略: {sorted_by_first_token[0][0]}")
    
    # 策略3: 按第二个token ID
    sorted_by_second_token = sorted(pairs_with_same_freq, key=lambda x: x[0][1])
    print(f"按第二个token策略: {sorted_by_second_token[0][0]}")
    
    # 策略4: 按token ID之和
    sorted_by_sum = sorted(pairs_with_same_freq, key=lambda x: x[0][0] + x[0][1])
    print(f"按token和策略: {sorted_by_sum[0][0]}")

if __name__ == "__main__":
    test_tie_breaking()
