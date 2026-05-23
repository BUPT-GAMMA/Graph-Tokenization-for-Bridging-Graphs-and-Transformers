from __future__ import annotations

from typing import Tuple
import numpy as np

try:
    from numba import njit  # noqa: F401
except Exception:
    # File kept for compatibility; main path no longer calls numba version
    def njit(*a, **kw):  # type: ignore
        def deco(fn):
            return fn
        return deco

"""
Numba-accelerated BPE training core (follows standard minbpe logic).

Key functions:
- count_pairs_ragged: count adjacent pair frequencies in ragged sequences
- apply_merge_ragged: apply one non-overlapping merge on ragged sequences
- select_best_pair: pick highest-frequency pair with lexicographic tie-break

Data layout: int32 flat arrays; pair keys packed as int64: (left<<32)|right.
"""


@njit(cache=True, inline="always")
def _pack_pair_int64(left: np.int32, right: np.int32) -> np.int64:
    """Pack (left, right) into int64 key: (left<<32)|right."""
    return (np.int64(left) << np.int64(32)) | (np.int64(right) & np.int64(0xFFFFFFFF))


@njit(cache=True)
def count_pairs_ragged(flat: np.ndarray, offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Count global adjacent pair frequencies in ragged sequences."""
    num_seqs = offsets.size - 1

    # Count total pairs
    total_pairs = 0
    for s in range(num_seqs):
        start = int(offsets[s])
        end = int(offsets[s + 1])
        L = end - start
        if L >= 2:
            total_pairs += (L - 1)

    if total_pairs == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)

    # Collect all pair keys
    pair_keys = np.empty(total_pairs, dtype=np.int64)
    k = 0
    for s in range(num_seqs):
        start = int(offsets[s])
        end = int(offsets[s + 1])
        for i in range(start, end - 1):
            pair_keys[k] = _pack_pair_int64(flat[i], flat[i + 1])
            k += 1

    # Sort and run-length encode to get (unique_key, count)
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
    # Last run
    uniq_keys[out_idx] = run_key
    counts[out_idx] = run_count
    out_idx += 1

    return uniq_keys[:out_idx], counts[:out_idx]


@njit(cache=True)
def select_best_pair(pair_keys: np.ndarray, pair_counts: np.ndarray, min_frequency: np.int32) -> Tuple[np.int32, np.int32, np.int32]:
    """Select highest-frequency pair meeting min_frequency, with lexicographic tie-break."""
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
    """Pass 1: compute post-merge length per sequence and total length."""
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
                # Match (left, right) -> new_id, non-overlapping
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
    """Pass 2: fill merged sequences into out_flat using out_offsets."""
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
    """Apply one non-overlapping (left,right)->new_id merge on ragged sequences."""
    # Pass 1: length stats
    new_lengths, total_new = _compute_new_lengths(flat, offsets, left, right)

    num_seqs = offsets.size - 1
    out_offsets = np.empty(num_seqs + 1, dtype=np.int32)
    out_offsets[0] = 0
    for s in range(num_seqs):
        out_offsets[s + 1] = out_offsets[s] + new_lengths[s]

    out_flat = np.empty(total_new, dtype=np.int32)

    # Pass 2: fill merged data
    _fill_merged(flat, offsets, left, right, new_id, out_flat, out_offsets)

    return out_flat, out_offsets


