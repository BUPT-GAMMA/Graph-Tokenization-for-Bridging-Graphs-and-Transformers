from __future__ import annotations

from typing import Tuple
import numpy as np

try:
    from numba import njit  # noqa: F401
except Exception:
    # 为保持兼容性而保留文件；主线不再调用 numba 版本
    def njit(*a, **kw):  # type: ignore
        def deco(fn):
            return fn
        return deco

"""
Numba 加速的 BPE 训练核心（遵循标准 minbpe 逻辑）

设计目标：
- 完全遵循 minbpe 的训练逻辑：每轮重新统计，选择频次最高的pair
- 使用 numba 加速关键循环：pair统计 与 merge操作
- 数据形态：int32 连续数组；pair 键以 int64 打包：(l<<32)|r

提供的函数：
- count_pairs_ragged(flat:int32[:], offsets:int32[:]) -> (pair_keys:int64[:], counts:int32[:])
- apply_merge_ragged(flat:int32[:], offsets:int32[:], left:int32, right:int32, new_id:int32) -> (new_flat:int32[:], new_offsets:int32[:])

注意：
- 这两个函数均为纯数值核，不依赖 Python 字典；适合在训练循环中被重复调用。
"""


@njit(cache=True, inline="always")
def _pack_pair_int64(left: np.int32, right: np.int32) -> np.int64:
    """将 (left,right) 打包为 int64 键: (left<<32)|right。
    使用显式的 int64 转换避免溢出。
    """
    return (np.int64(left) << np.int64(32)) | (np.int64(right) & np.int64(0xFFFFFFFF))


@njit(cache=True)
def count_pairs_ragged(flat: np.ndarray, offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """统计 ragged 序列的全局相邻 pair 频次。

    参数：
    - flat: int32 一维数组，拼接后的所有 token
    - offsets: int32 一维数组，长度为 num_seqs+1，offsets[i]:offsets[i+1] 表示第 i 个序列

    返回：
    - pair_keys: int64 一维数组，去重后的 pair 键
    - counts: int32 一维数组，对应 pair 的出现次数
    """
    num_seqs = offsets.size - 1

    # 计算总的 pair 数量
    total_pairs = 0
    for s in range(num_seqs):
        start = int(offsets[s])
        end = int(offsets[s + 1])
        L = end - start
        if L >= 2:
            total_pairs += (L - 1)

    if total_pairs == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)

    # 收集所有 pair 键
    pair_keys = np.empty(total_pairs, dtype=np.int64)
    k = 0
    for s in range(num_seqs):
        start = int(offsets[s])
        end = int(offsets[s + 1])
        for i in range(start, end - 1):
            pair_keys[k] = _pack_pair_int64(flat[i], flat[i + 1])
            k += 1

    # 排序并进行游程压缩得到 (unique_key, count)
    pair_keys.sort()

    uniq_keys = np.empty(total_pairs, dtype=np.int64)
    counts = np.empty(total_pairs, dtype=np.int32)

    out_idx = 0
    run_key = pair_keys[0]
    run_count = 1
    for i in range(1, total_pairs):
        key = pair_keys[i]
        if key == run_key:
            run_count += 1
        else:
            uniq_keys[out_idx] = run_key
            counts[out_idx] = run_count
            out_idx += 1
            run_key = key
            run_count = 1
    # 最后一段
    uniq_keys[out_idx] = run_key
    counts[out_idx] = run_count
    out_idx += 1

    return uniq_keys[:out_idx], counts[:out_idx]


@njit(cache=True)
def select_best_pair(pair_keys: np.ndarray, pair_counts: np.ndarray, min_frequency: np.int32) -> Tuple[np.int32, np.int32, np.int32]:
    """从 (pair_keys, pair_counts) 中选择频次最高、且满足 min_frequency 的 pair。

    在频次相同的情况下，按 (left_id, right_id) 的字典序最小进行 tie-break。

    返回：(best_left, best_right, best_freq)。若无满足者，best_freq 返回 -1。
    """
    best_freq = np.int32(-1)
    best_left = np.int32(0)
    best_right = np.int32(0)
    for i in range(pair_keys.size):
        freq = pair_counts[i]
        if freq < min_frequency:
            continue
        key = pair_keys[i]
        left_id = np.int32((key >> np.int64(32)) & np.int64(0xFFFFFFFF))
        right_id = np.int32(key & np.int64(0xFFFFFFFF))
        if (
            freq > best_freq
            or (
                freq == best_freq
                and (left_id < best_left or (left_id == best_left and right_id < best_right))
            )
        ):
            best_freq = freq
            best_left = left_id
            best_right = right_id
    return best_left, best_right, best_freq

@njit(cache=True)
def _compute_new_lengths(flat: np.ndarray, offsets: np.ndarray, left: np.int32, right: np.int32) -> Tuple[np.ndarray, int]:
    """第一遍：计算每个序列在应用 merge 后的长度，以及总长度。"""
    num_seqs = offsets.size - 1
    new_lengths = np.empty(num_seqs, dtype=np.int32)
    total_new = 0
    for s in range(num_seqs):
        start = int(offsets[s])
        end = int(offsets[s + 1])
        j = start
        new_len = 0
        while j < end:
            if j + 1 < end and flat[j] == left and flat[j + 1] == right:
                # 命中 (left,right) -> new_id，非重叠替换
                new_len += 1
                j += 2
            else:
                new_len += 1
                j += 1
        new_lengths[s] = new_len
        total_new += new_len
    return new_lengths, total_new


@njit(cache=True)
def _fill_merged(flat: np.ndarray, offsets: np.ndarray, left: np.int32, right: np.int32, new_id: np.int32,
                 out_flat: np.ndarray, out_offsets: np.ndarray) -> None:
    """第二遍：按 out_offsets 提供的起点，填充合并后的序列到 out_flat。串行以降低并发复杂度。"""
    num_seqs = offsets.size - 1
    for s in range(num_seqs):
        start = int(offsets[s])
        end = int(offsets[s + 1])
        out_pos = int(out_offsets[s])
        j = start
        while j < end:
            if j + 1 < end and flat[j] == left and flat[j + 1] == right:
                out_flat[out_pos] = new_id
                out_pos += 1
                j += 2
            else:
                out_flat[out_pos] = flat[j]
                out_pos += 1
                j += 1


@njit(cache=True)
def apply_merge_ragged(flat: np.ndarray, offsets: np.ndarray,
                       left: np.int32, right: np.int32, new_id: np.int32) -> Tuple[np.ndarray, np.ndarray]:
    """对 ragged 序列应用一次 (left,right)->new_id 的非重叠 merge。

    返回合并后的新 flat 与新 offsets。
    """
    # 第一遍：长度统计
    new_lengths, total_new = _compute_new_lengths(flat, offsets, left, right)

    num_seqs = offsets.size - 1
    out_offsets = np.empty(num_seqs + 1, dtype=np.int32)
    out_offsets[0] = 0
    for s in range(num_seqs):
        out_offsets[s + 1] = out_offsets[s] + new_lengths[s]

    out_flat = np.empty(total_new, dtype=np.int32)

    # 第二遍：并行填充
    _fill_merged(flat, offsets, left, right, new_id, out_flat, out_offsets)

    return out_flat, out_offsets


