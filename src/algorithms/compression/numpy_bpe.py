"""
基于 NumPy 的 BPE 训练实现（频率统计使用 NumPy 向量化），接口与 StandardBPECompressor 对齐。

注意：本实现的主要加速点在每轮 pair 频率统计阶段；merge 操作仍采用逐序列扫描（可进一步用 numba/并行强化）。
"""

from typing import Any, Dict, List, Tuple
from collections import defaultdict  # noqa: F401 (reserved for future use)

import numpy as np

from .main_bpe import StandardBPECompressor  # 保留以备参考；主线已不使用 numpy 版本


class NumpyBPECompressor(StandardBPECompressor):
    """NumPy 版 BPE：每轮使用 NumPy 统计全局相邻 pair 频率。

    - 词表构建、ID 转换、码本更新、编码/解码均复用基类逻辑
    - 训练时在每轮基于当前 id_sequences 重新统计全局 pair 频率（向量化）
    - 合并应用采用与基类一致的逐序列线性扫描策略
    """

    def _count_pairs_numpy(self, id_sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
        # 将所有序列拼接 pair，并使用 np.unique 统计
        # 编码 pair: code = (left << 32) | right （均使用 int64）
        if not id_sequences:
            return {}
        codes: List[np.ndarray] = []
        for seq in id_sequences:
            if len(seq) < 2:
                continue
            arr = np.asarray(seq, dtype=np.int64)
            left = arr[:-1]
            right = arr[1:]
            code = (left << 32) | right
            codes.append(code)
        if not codes:
            return {}
        all_codes = np.concatenate(codes)
        uniq, cnts = np.unique(all_codes, return_counts=True)
        # 解码回 (l, r) -> count
        pair_freqs: Dict[Tuple[int, int], int] = {}
        left_ids = (uniq >> 32).astype(np.int64)
        right_ids = (uniq & np.int64(0xFFFFFFFF)).astype(np.int64)
        for left_id, right_id, c in zip(left_ids.tolist(), right_ids.tolist(), cnts.tolist()):
            pair_freqs[(int(left_id), int(right_id))] = int(c)
        return pair_freqs

    def train(self, token_sequences: List[List[int]]) -> Dict[str, Any]:
        self._check_token_sequences(token_sequences)
        if not token_sequences:
            raise ValueError("训练序列为空")

        # 1) 词表构建
        self._build_base_vocab(token_sequences)
        if len(self.token_to_id) < 2:
            raise ValueError("基础词汇表太小，无法进行合并")

        # 2) 转为 ID 序列
        id_sequences = self._convert_to_id_sequences(token_sequences)

        # 3) 迭代合并
        merge_count = 0
        for _ in range(self.num_merges):
            # NumPy 统计全局相邻 pair 频率
            pair_freqs = self._count_pairs_numpy(id_sequences)
            if not pair_freqs:
                break
            # 过滤阈值
            valid_pairs = {pair: freq for pair, freq in pair_freqs.items() if freq >= self.min_frequency}
            if not valid_pairs:
                break
            # 选择最佳 pair
            best_pair = max(valid_pairs, key=valid_pairs.get)

            # 更新码本
            new_id = self.next_id
            self.next_id += 1
            self.merge_rules.append((best_pair[0], best_pair[1], new_id))
            merged_token = (self.id_to_token[best_pair[0]], self.id_to_token[best_pair[1]])
            self.token_to_id[merged_token] = new_id
            self.id_to_token[new_id] = merged_token

            # 应用合并：minBPE 风格的非重叠掩码合并
            left_id, right_id = best_pair
            for i, seq in enumerate(id_sequences):
                if len(seq) < 2:
                    continue
                arr = np.asarray(seq, dtype=np.int64)
                # 标记第一个位置匹配 (left,right)
                is_first = (arr[:-1] == left_id) & (arr[1:] == right_id)
                if not is_first.any():
                    continue
                # 扩展到与 arr 同长，并抑制重叠：只有不被上一个匹配占用的首位才保留
                is_first_full = np.concatenate([is_first, np.array([False], dtype=bool)])
                is_second = np.roll(is_first_full, 1)
                is_first_full = is_first_full & (~is_second)
                is_second = np.roll(is_first_full, 1)

                out = arr.copy()
                out[is_first_full] = new_id
                # 删除第二个位置
                delete_mask = ~is_second
                out = out[delete_mask]
                id_sequences[i] = out.astype(int).tolist()

            merge_count += 1

        if merge_count == 0:
            raise ValueError("未执行任何合并操作")

        return {
            'num_merges_performed': merge_count,
            'final_vocab_size': len(self.token_to_id),
            'merge_rules_count': len(self.merge_rules),
        }


