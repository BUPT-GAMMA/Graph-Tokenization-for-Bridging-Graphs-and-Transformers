"""
NumPy-based BPE training (vectorized pair frequency counting).
Interface aligned with StandardBPECompressor.
"""

from typing import Any, Dict, List, Tuple
from collections import defaultdict  # noqa: F401 (reserved for future use)

import numpy as np

from .main_bpe import StandardBPECompressor  # kept for reference; main path no longer uses numpy version


class NumpyBPECompressor(StandardBPECompressor):
    """NumPy BPE: vectorized global pair frequency counting each round.

    Vocab building, ID conversion, codebook updates, and encode/decode
    are inherited from the base class.
    """

    def _count_pairs_numpy(self, id_sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
        # Concatenate all pairs and count with np.unique
        # Encode pair: code = (left << 32) | right (int64)
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
        # Decode back to (l, r) -> count
        pair_freqs: Dict[Tuple[int, int], int] = {}
        left_ids = (uniq >> 32).astype(np.int64)
        right_ids = (uniq & np.int64(0xFFFFFFFF)).astype(np.int64)
        for left_id, right_id, c in zip(left_ids.tolist(), right_ids.tolist(), cnts.tolist()):
            pair_freqs[(int(left_id), int(right_id))] = int(c)
        return pair_freqs

    def train(self, token_sequences: List[List[int]]) -> Dict[str, Any]:
        self._check_token_sequences(token_sequences)
        if not token_sequences:
            raise ValueError("Training sequences are empty")

        # 1) Build base vocab
        self._build_base_vocab(token_sequences)
        if len(self.token_to_id) < 2:
            raise ValueError("Base vocabulary too small for merging")

        # 2) Convert to ID sequences
        id_sequences = self._convert_to_id_sequences(token_sequences)

        # 3) Iterative merging
        merge_count = 0
        for _ in range(self.num_merges):
            # NumPy vectorized pair frequency counting
            pair_freqs = self._count_pairs_numpy(id_sequences)
            if not pair_freqs:
                break
            # Filter by threshold
            valid_pairs = {pair: freq for pair, freq in pair_freqs.items() if freq >= self.min_frequency}
            if not valid_pairs:
                break
            # Select best pair
            best_pair = max(valid_pairs, key=valid_pairs.get)

            # Update codebook
            new_id = self.next_id
            self.next_id += 1
            self.merge_rules.append((best_pair[0], best_pair[1], new_id))
            merged_token = (self.id_to_token[best_pair[0]], self.id_to_token[best_pair[1]])
            self.token_to_id[merged_token] = new_id
            self.id_to_token[new_id] = merged_token

            # Apply merge: minBPE-style non-overlapping mask merge
            left_id, right_id = best_pair
            for i, seq in enumerate(id_sequences):
                if len(seq) < 2:
                    continue
                arr = np.asarray(seq, dtype=np.int64)
                # Mark first positions matching (left, right)
                is_first = (arr[:-1] == left_id) & (arr[1:] == right_id)
                if not is_first.any():
                    continue
                # Extend to arr length, suppress overlaps
                is_first_full = np.concatenate([is_first, np.array([False], dtype=bool)])
                is_second = np.roll(is_first_full, 1)
                is_first_full = is_first_full & (~is_second)
                is_second = np.roll(is_first_full, 1)

                out = arr.copy()
                out[is_first_full] = new_id
                # Delete second positions
                delete_mask = ~is_second
                out = out[delete_mask]
                id_sequences[i] = out.astype(int).tolist()

            merge_count += 1

        if merge_count == 0:
            raise ValueError("No merges performed")

        return {
            'num_merges_performed': merge_count,
            'final_vocab_size': len(self.token_to_id),
            'merge_rules_count': len(self.merge_rules),
        }


