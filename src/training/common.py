"""
通用训练工具（已废弃）
=====================

注意：这个文件中原有的包装函数已被移除。
请直接使用 UnifiedDataInterface 的方法：

- load_training_data() → udi.get_training_data(method, target_property)
- load_sequences_splits() → udi.get_sequences_by_splits(method) + 手动提取序列
- flatten_all_sequences() → 简单的 list(seq1) + list(seq2) + list(seq3)

根目录下的 bert_*.py 文件是备份脚本，如需使用请手动更新其导入。
"""

# 此文件保留为空，所有逻辑已移至 UnifiedDataInterface


