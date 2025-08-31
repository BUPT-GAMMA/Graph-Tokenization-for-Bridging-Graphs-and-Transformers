#!/usr/bin/env python3
"""
测试peptides_struct数据集使用peptides_func预训练模型的逻辑
"""

def test_peptides_logic():
    """测试peptides数据集的预训练模型选择逻辑"""

    # 测试数据
    test_cases = [
        ("peptides_func", "peptides_func"),
        ("peptides_struct", "peptides_func"),  # 应该使用func的预训练模型
        ("qm9", "qm9"),
        ("zinc", "zinc"),
    ]

    print("🧪 测试peptides_struct预训练模型选择逻辑")
    print("=" * 50)

    for dataset, expected_pretrain_dataset in test_cases:
        # 模拟batch_finetune_simple.py中的逻辑
        pretrain_dataset = "peptides_func" if dataset == "peptides_struct" else dataset

        # 检查结果
        if pretrain_dataset == expected_pretrain_dataset:
            status = "✅ 通过"
        else:
            status = "❌ 失败"

        print(f"{dataset:<15} → {pretrain_dataset:<15} {status}")

    print("=" * 50)

    # 详细说明预期行为
    print("\n📋 预期行为说明：")
    print("• peptides_func → 使用 peptides_func 的预训练模型")
    print("• peptides_struct → 使用 peptides_func 的预训练模型（数据相同）")
    print("• 其他数据集 → 使用对应数据集的预训练模型")

if __name__ == "__main__":
    test_peptides_logic()
