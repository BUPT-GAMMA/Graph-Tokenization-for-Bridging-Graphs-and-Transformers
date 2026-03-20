from __future__ import annotations

from typing import List, Tuple, Optional, Dict


class _PythonBPEEncoder:
    """Minimal pure-Python encoder fallback for environments without `_cpp_bpe`."""

    def __init__(self, merge_rules: List[Tuple[int, int, int]]) -> None:
        self._merge_rules = list(merge_rules)
        self._pair_to_rank = {(left, right): rank for rank, (left, right, _) in enumerate(self._merge_rules)}
        self._pair_to_newid = {(left, right): new_id for left, right, new_id in self._merge_rules}

    @staticmethod
    def _apply_single_merge(seq: List[int], left: int, right: int, new_id: int) -> List[int]:
        merged: List[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
                merged.append(new_id)
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        return merged

    def encode(self, seq: List[int]) -> List[int]:
        ids = list(seq)
        while len(ids) >= 2:
            best_pair: Optional[Tuple[int, int]] = None
            best_rank: Optional[int] = None
            for pair in zip(ids, ids[1:]):
                rank = self._pair_to_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            ids = self._apply_single_merge(ids, best_pair[0], best_pair[1], self._pair_to_newid[best_pair])
        return ids

    def batch_encode(self, seqs: List[List[int]]) -> List[List[int]]:
        return [self.encode(seq) for seq in seqs]


class BPEEngine:
    """
    Unified high-performance BPE train + encode engine (pluggable backends).

    Training backends: "cpp" | "numba" | "python"
    Encode backend: "cpp" (C++/pybind11, recommended) | "python" (fallback, deterministic modes only)

    Output is semantically equivalent to minBPE reference:
    - merge_rules: List[Tuple[left_id, right_id, new_id]]
    - vocab_size = base_vocab_size + num_merges_performed
    """

    def __init__(self, *, train_backend: str = "cpp", encode_backend: str = "cpp",
                 encode_rank_mode: str = "all", encode_rank_k: Optional[int] = None,
                 encode_rank_min: Optional[int] = None, encode_rank_max: Optional[int] = None,
                 encode_rank_dist: Optional[str] = None, verbose: bool = False) -> None:
        self.train_backend = train_backend
        self.encode_backend = encode_backend
        self.encode_rank_mode = encode_rank_mode  # 'all' | 'topk' | 'random' | 'gaussian'
        self.encode_rank_k = encode_rank_k
        self.encode_rank_min = encode_rank_min
        self.encode_rank_max = encode_rank_max
        self.encode_rank_dist = encode_rank_dist  # 'uniform'|'triangular'|'gaussian'
        self.verbose = bool(verbose)
        self.merge_rules: List[Tuple[int, int, int]] = []
        self.vocab_size: Optional[int] = None
        self._encoder = None

    # ---------------- Training ----------------
    def train(self, token_sequences: List[List[int]], *, num_merges: int, min_frequency: int) -> Dict:
        if self.train_backend == "numba":
            # numba minBPE implementation (re-stats each round + ragged merge)
            return self._train_minbpe_numba(token_sequences, num_merges, min_frequency)
        elif self.train_backend == "python":
            from .main_bpe import StandardBPECompressor
            comp = StandardBPECompressor(num_merges=num_merges, min_frequency=min_frequency, debug=False)
            stats = comp.train(token_sequences)
            # Extract merge_rules and vocab_size for unified downstream interface
            self.merge_rules = list(comp.merge_rules)
            self.vocab_size = len(comp.token_to_id)
            return {"num_merges_performed": stats.get("num_merges_performed", len(self.merge_rules)),
                    "final_vocab_size": self.vocab_size}
        elif self.train_backend == "cpp":
            # C++ native minBPE training
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp(token_sequences, int(num_merges), int(min_frequency))
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            return {"num_merges_performed": int(results["num_merges_performed"]),
                    "final_vocab_size": self.vocab_size}
        else:
            raise ValueError(f"Unsupported train_backend: {self.train_backend}")

    

    def _train_minbpe_numba(self, token_sequences: List[List[int]], num_merges: int, min_frequency: int) -> Dict:
        """Numba-accelerated BPE training (strict minBPE logic)."""
        from .numba_bpe_train import count_pairs_ragged, apply_merge_ragged, select_best_pair

        # Count base vocabulary
        base_vocab: Dict[int, None] = {}
        for s in token_sequences:
            for t in s:
                base_vocab[int(t)] = None

        base_vocab_size = len(base_vocab)
        max_base_id = max(base_vocab.keys()) if base_vocab else -1

        # Align with minBPE ID assignment
        separator_token = max_base_id + 1  # virtual separator for next_id offset
        next_id = separator_token + 1

        # Convert to ragged data structure
        import numpy as np
        offsets = [0]
        flat = []
        for s in token_sequences:
            flat.extend([int(x) for x in s])
            offsets.append(len(flat))
        flat = np.asarray(flat, dtype=np.int32)
        offsets = np.asarray(offsets, dtype=np.int32)

        # Warm up numba JIT (small sample)
        if len(offsets) > 2:
            warm_n = min(2, len(offsets) - 1)
            w_flat = flat[:offsets[warm_n]]
            w_offsets = offsets[:warm_n + 1].copy()
            _ = count_pairs_ragged(w_flat, w_offsets)
            if w_offsets[-1] >= 3:
                _ = apply_merge_ragged(w_flat, w_offsets, np.int32(w_flat[0]), np.int32(w_flat[1]), np.int32(10))

        # Initial pair stats
        pair_keys, pair_counts = count_pairs_ragged(flat, offsets)

        merges_done = 0
        merge_rules: List[Tuple[int, int, int]] = []

        while merges_done < num_merges:
            # Select best pair in numba kernel
            best_left_i32, best_right_i32, best_freq_i32 = select_best_pair(pair_keys, pair_counts, np.int32(min_frequency))
            best_freq = int(best_freq_i32)
            if best_freq < int(min_frequency):
                break

            left_id = int(best_left_i32)
            right_id = int(best_right_i32)
            new_id = next_id
            next_id += 1

            # Apply one merge
            flat, offsets = apply_merge_ragged(
                flat, offsets, np.int32(left_id), np.int32(right_id), np.int32(new_id)
            )

            # Re-count pair frequencies for next round
            pair_keys, pair_counts = count_pairs_ragged(flat, offsets)

            merge_rules.append((int(left_id), int(right_id), int(new_id)))
            merges_done += 1

            if self.verbose and (merges_done % 200 == 0 or merges_done <= 5):
                print(f"Merge {merges_done}: ({left_id}, {right_id}) -> {new_id} (freq: {best_freq})")

        self.merge_rules = merge_rules
        self.vocab_size = base_vocab_size + merges_done
        return {"num_merges_performed": merges_done, "final_vocab_size": self.vocab_size}

    # ---------------- Encode backend ----------------
    def build_encoder(self) -> None:
        if self.merge_rules is None or self.vocab_size is None:
            raise ValueError("Train first or load a codebook")
        if self.encode_backend == "cpp":
            from .cpp_bpe_backend import CppBPEBackend
            self._encoder = CppBPEBackend.from_codebook(
                merge_rules=self.merge_rules,
                pair_to_rank=None,
                pair_to_newid=None,
                vocab_size=self.vocab_size,
            )
        elif self.encode_backend == "python":
            if self.encode_rank_mode not in ("all", "none"):
                raise ValueError(
                    f"encode_backend='python' only supports encode_rank_mode='all' or 'none', got {self.encode_rank_mode}"
                )
            self._encoder = _PythonBPEEncoder(self.merge_rules)
        else:
            raise ValueError(f"Unsupported encode_backend: {self.encode_backend}")

    def encode(self, seq: List[int]) -> List[int]:
        # Special mode: none - skip BPE encoding entirely
        if self.encode_rank_mode == "none":
            return seq
            
        # Sample topk if needed
        if self.encode_backend == "cpp" and self.encode_rank_mode in ("topk","random","gaussian"):
            topk = self._sample_topk()
            from .cpp_bpe_backend import CppBPEBackend
            if isinstance(self._encoder, CppBPEBackend) and topk is not None:
                return self._encoder.encode_topk(seq, int(topk))
        if self._encoder is None:
            self.build_encoder()
        assert self._encoder is not None, "Encoder not built"
        return self._encoder.encode(seq)

    def batch_encode(self, seqs: List[List[int]]) -> List[List[int]]:
        # Special mode: none - skip BPE encoding entirely
        if self.encode_rank_mode == "none":
            return seqs
            
        # Sample topk (shared across batch for stability)
        if self.encode_backend == "cpp" and self.encode_rank_mode in ("topk","random","gaussian"):
            topk = self._sample_topk()
            from .cpp_bpe_backend import CppBPEBackend
            if isinstance(self._encoder, CppBPEBackend) and topk is not None:
                return self._encoder.batch_encode_topk(seqs, int(topk))
        if self._encoder is None:
            self.build_encoder()
        assert self._encoder is not None, "Encoder not built"
        if hasattr(self._encoder, "batch_encode"):
            return self._encoder.batch_encode(seqs)
        # Fallback to single encode
        return [self.encode(s) for s in seqs]

    # Internal topk sampling strategy
    def _sample_topk(self) -> Optional[int]:
        import random
        if self.encode_rank_mode == 'all':
            return None
        if self.encode_rank_mode == 'topk':
            return int(self.encode_rank_k) if self.encode_rank_k is not None else None
        # random/gaussian sample in [min, max]
        if self.encode_rank_min is None or self.encode_rank_max is None:
            a = 0
            b = len(self.merge_rules) if self.merge_rules is not None else 0
        else:
            a, b = int(self.encode_rank_min), int(self.encode_rank_max)
        if a < 0:
            a = 0
        if b <= a:
            b = a
        if self.encode_rank_mode == 'random':
            if (self.encode_rank_dist or 'uniform') == 'uniform':
                return random.randint(a, b)
            if self.encode_rank_dist == 'triangular':
                mode = b  # bias toward using more rules
                return int(random.triangular(a, b, mode))
            # default: uniform
            return random.randint(a, b)
        if self.encode_rank_mode == 'gaussian':
            # Truncated Gaussian centered at max (bias toward more rules)
            mu = float(b)
            sigma = max(1.0, (b - a) / 3.0)
            # Simple truncation
            for _ in range(8):
                x = int(random.gauss(mu, sigma))
                if a <= x <= b:
                    return x
            return b
        return None

    # ---------------- Codebook IO (lightweight, path-agnostic policy) ----------------
    def to_codebook(self) -> Dict:
        """Export minimal codebook dict."""
        if self.merge_rules is None or self.vocab_size is None:
            raise ValueError("Codebook not initialized")
        return {
            "merge_rules": list(self.merge_rules),
            "vocab_size": int(self.vocab_size),
        }

    def load_codebook_dict(self, data: Dict) -> None:
        """Load codebook from dict."""
        assert isinstance(data, dict) and 'merge_rules' in data and 'vocab_size' in data, (
            "Codebook dict missing required fields: 'merge_rules' or 'vocab_size'"
        )
        self.merge_rules = [tuple(int(x) for x in t) for t in data['merge_rules']]
        self.vocab_size = int(data['vocab_size'])
        self._encoder = None

    def save_codebook(self, path: str) -> None:
        """Save codebook to file (.pkl or .json)."""
        import os
        import pickle
        import json
        if self.merge_rules is None or self.vocab_size is None:
            raise ValueError("Codebook not initialized")
        suffix = os.path.splitext(path)[1].lower()
        data = self.to_codebook()
        if suffix == '.pkl' or suffix == '.pickle':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported codebook file extension: {suffix}. Use .pkl or .json.")

    def load_codebook(self, path: str) -> None:
        """Load codebook from file (.pkl or .json)."""
        import os
        import pickle
        import json
        suffix = os.path.splitext(path)[1].lower()
        if suffix == '.pkl' or suffix == '.pickle':
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported codebook file extension: {suffix}. Use .pkl or .json.")
        self.load_codebook_dict(data)

    @classmethod
    def from_codebook_dict(cls, data: Dict, *, encode_backend: str = 'cpp', **engine_kwargs) -> "BPEEngine":
        """Build encode-only engine from codebook dict."""
        eng = cls(train_backend='python', encode_backend=encode_backend, **engine_kwargs)
        eng.load_codebook_dict(data)
        return eng

    @classmethod
    def load_codebook_to_engine(cls, path: str, *, encode_backend: str = 'cpp', **engine_kwargs) -> "BPEEngine":
        """Load codebook from file and return an engine instance."""
        eng = cls(train_backend='python', encode_backend=encode_backend, **engine_kwargs)
        eng.load_codebook(path)
        return eng

