from __future__ import annotations

from typing import List, Tuple, Optional, Dict


class BPEEngine:
    """
    统一的高性能 BPE 训练+编码引擎（可选后端）。

    - 训练（train_backend="cpp"|"numba"|"python"）
      - cpp  : C++ 原生 minBPE 训练（每轮重统计、词典序 tie-break、非重叠合并）
      # - numba: Numba 加速 minBPE 训练（ragged 扁平数组 + JIT 内核）
      - python: StandardBPECompressor（增量频次表，作兼容/对照）

    - 编码（encode_backend="cpp"）
      - cpp  : C++/pybind11 后端（推荐且唯一支持）

    训练产物与 minBPE 参考语义等价：
    - 产出 merge_rules: List[Tuple[left_id, right_id, new_id]]；
    - vocab_size = base_vocab_size + num_merges_performed。
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
            # 使用遵循 minBPE 逻辑的 numba 实现（每轮重统计 + ragged 合并）
            return self._train_minbpe_numba(token_sequences, num_merges, min_frequency)
        elif self.train_backend == "python":
            from .main_bpe import StandardBPECompressor
            comp = StandardBPECompressor(num_merges=num_merges, min_frequency=min_frequency, debug=False)
            stats = comp.train(token_sequences)
            # 提取 merge_rules 与 vocab_size 以统一下游接口
            self.merge_rules = list(comp.merge_rules)
            self.vocab_size = len(comp.token_to_id)
            return {"num_merges_performed": stats.get("num_merges_performed", len(self.merge_rules)),
                    "final_vocab_size": self.vocab_size}
        elif self.train_backend == "cpp":
            # 使用 C++ 原生 minBPE 训练
            from .cpp_bpe_backend import CppBPEBackend
            results = CppBPEBackend.train_minbpe_cpp(token_sequences, int(num_merges), int(min_frequency))
            self.merge_rules = results["merge_rules"]
            self.vocab_size = int(results["final_vocab_size"])
            return {"num_merges_performed": int(results["num_merges_performed"]),
                    "final_vocab_size": self.vocab_size}
        else:
            raise ValueError(f"不支持的 train_backend: {self.train_backend}")

    

    def _train_minbpe_numba(self, token_sequences: List[List[int]], num_merges: int, min_frequency: int) -> Dict:
        """使用 numba 加速的 BPE 训练（严格遵循 minBPE 逻辑）。"""
        from .numba_bpe_train import count_pairs_ragged, apply_merge_ragged, select_best_pair

        # 统计基础词表
        base_vocab: Dict[int, None] = {}
        for s in token_sequences:
            for t in s:
                base_vocab[int(t)] = None

        base_vocab_size = len(base_vocab)
        max_base_id = max(base_vocab.keys()) if base_vocab else -1

        # 与 minBPE 的 ID 分配策略对齐
        separator_token = max_base_id + 1  # 虚拟分隔符，仅用于计算 next_id 起点
        next_id = separator_token + 1

        # 转换为 ragged 数据结构
        import numpy as np
        offsets = [0]
        flat = []
        for s in token_sequences:
            flat.extend([int(x) for x in s])
            offsets.append(len(flat))
        flat = np.asarray(flat, dtype=np.int32)
        offsets = np.asarray(offsets, dtype=np.int32)

        # 预热 numba JIT（小样本，避免长时间 warmup）
        if len(offsets) > 2:
            warm_n = min(2, len(offsets) - 1)
            w_flat = flat[:offsets[warm_n]]
            w_offsets = offsets[:warm_n + 1].copy()
            _ = count_pairs_ragged(w_flat, w_offsets)
            if w_offsets[-1] >= 3:
                _ = apply_merge_ragged(w_flat, w_offsets, np.int32(w_flat[0]), np.int32(w_flat[1]), np.int32(10))

        # 初始统计
        pair_keys, pair_counts = count_pairs_ragged(flat, offsets)

        merges_done = 0
        merge_rules: List[Tuple[int, int, int]] = []

        while merges_done < num_merges:
            # 在 numba 内核中完成“选择最佳 pair”（减少 Python 循环）
            best_left_i32, best_right_i32, best_freq_i32 = select_best_pair(pair_keys, pair_counts, np.int32(min_frequency))
            best_freq = int(best_freq_i32)
            if best_freq < int(min_frequency):
                break

            left_id = int(best_left_i32)
            right_id = int(best_right_i32)
            new_id = next_id
            next_id += 1

            # 应用一次合并
            flat, offsets = apply_merge_ragged(
                flat, offsets, np.int32(left_id), np.int32(right_id), np.int32(new_id)
            )

            # 重新统计下一轮的 pair 频次
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
            raise ValueError("请先完成训练或从文件加载 codebook")
        if self.encode_backend == "cpp":
            from .cpp_bpe_backend import CppBPEBackend
            self._encoder = CppBPEBackend.from_codebook(
                merge_rules=self.merge_rules,
                pair_to_rank=None,
                pair_to_newid=None,
                vocab_size=self.vocab_size,
            )
        else:
            raise ValueError(f"不支持的 encode_backend: {self.encode_backend}")

    def encode(self, seq: List[int]) -> List[int]:
        # 特殊模式：none - 不进行任何BPE编码，直接返回原序列（零拷贝优化）
        if self.encode_rank_mode == "none":
            return seq  # 直接返回，避免不必要的切片开销
            
        # 按需采样 topk
        if self.encode_backend == "cpp" and self.encode_rank_mode in ("topk","random","gaussian"):
            topk = self._sample_topk()
            from .cpp_bpe_backend import CppBPEBackend
            if isinstance(self._encoder, CppBPEBackend) and topk is not None:
                return self._encoder.encode_topk(seq, int(topk))
        if self._encoder is None:
            self.build_encoder()
        # 所有编码器后端都应该实现统一的 encode 接口
        assert self._encoder is not None, "编码器未正确构建"
        return self._encoder.encode(seq)

    def batch_encode(self, seqs: List[List[int]]) -> List[List[int]]:
        # 特殊模式：none - 不进行任何BPE编码，直接返回原序列（零拷贝优化）
        if self.encode_rank_mode == "none":
            return seqs  # 直接返回，避免不必要的切片开销
            
        # 按需采样 topk（同一批共用一次采样，稳定性能）
        if self.encode_backend == "cpp" and self.encode_rank_mode in ("topk","random","gaussian"):
            topk = self._sample_topk()
            from .cpp_bpe_backend import CppBPEBackend
            if isinstance(self._encoder, CppBPEBackend) and topk is not None:
                return self._encoder.batch_encode_topk(seqs, int(topk))
        if self._encoder is None:
            self.build_encoder()
        # 所有编码器后端都应该实现 batch_encode，或提供 encode 的回退实现
        assert self._encoder is not None, "编码器未正确构建"
        if hasattr(self._encoder, "batch_encode"):
            return self._encoder.batch_encode(seqs)
        # 回退到单个编码（某些简单后端可能只实现 encode）
        return [self.encode(s) for s in seqs]

    # 采样 topk 的内部策略
    def _sample_topk(self) -> Optional[int]:
        import random
        if self.encode_rank_mode == 'all':
            return None
        if self.encode_rank_mode == 'topk':
            return int(self.encode_rank_k) if self.encode_rank_k is not None else None
        # random/gaussian 需要在 [min,max] 之间采样
        # 默认区间：若未提供 min/max，则使用 [0, len(merge_rules)]
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
                mode = b  # 偏向使用更多 rules
                return int(random.triangular(a, b, mode))
            # 默认 uniform
            return random.randint(a, b)
        if self.encode_rank_mode == 'gaussian':
            # 以 max 为均值的截断高斯，偏向使用更多 rules
            mu = float(b)
            sigma = max(1.0, (b - a) / 3.0)
            # 简易截断
            for _ in range(8):
                x = int(random.gauss(mu, sigma))
                if a <= x <= b:
                    return x
            return b
        return None

    # ---------------- Codebook IO (lightweight, path-agnostic policy) ----------------
    def to_codebook(self) -> Dict:
        """导出最小 codebook 字典（与 UDI 的存取格式一致）。"""
        if self.merge_rules is None or self.vocab_size is None:
            raise ValueError("codebook 未初始化")
        return {
            "merge_rules": list(self.merge_rules),
            "vocab_size": int(self.vocab_size),
        }

    def load_codebook_dict(self, data: Dict) -> None:
        """从字典加载 codebook（不涉及路径组织）。"""
        assert isinstance(data, dict) and 'merge_rules' in data and 'vocab_size' in data, (
            "codebook 字典缺少必要字段: 'merge_rules' 或 'vocab_size'"
        )
        self.merge_rules = [tuple(int(x) for x in t) for t in data['merge_rules']]
        self.vocab_size = int(data['vocab_size'])
        self._encoder = None

    def save_codebook(self, path: str) -> None:
        """保存 codebook 到给定路径。

        说明：仅负责文件写入，不负责路径层级的组织。调用方应提供完整路径。
        支持 .pkl 或 .json 后缀；其他后缀直接报错。
        """
        import os
        import pickle
        import json
        if self.merge_rules is None or self.vocab_size is None:
            raise ValueError("codebook 未初始化")
        suffix = os.path.splitext(path)[1].lower()
        data = self.to_codebook()
        if suffix == '.pkl' or suffix == '.pickle':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的 codebook 文件后缀: {suffix}，请使用 .pkl 或 .json")

    def load_codebook(self, path: str) -> None:
        """从给定路径加载 codebook（支持 .pkl/.json）。"""
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
            raise ValueError(f"不支持的 codebook 文件后缀: {suffix}，请使用 .pkl 或 .json")
        self.load_codebook_dict(data)

    @classmethod
    def from_codebook_dict(cls, data: Dict, *, encode_backend: str = 'cpp', **engine_kwargs) -> "BPEEngine":
        """从 codebook 字典构建只用于编码的引擎。"""
        eng = cls(train_backend='python', encode_backend=encode_backend, **engine_kwargs)
        eng.load_codebook_dict(data)
        return eng

    @classmethod
    def load_codebook_to_engine(cls, path: str, *, encode_backend: str = 'cpp', **engine_kwargs) -> "BPEEngine":
        """从文件加载 codebook 并返回引擎实例（编码后端可选）。"""
        eng = cls(train_backend='python', encode_backend=encode_backend, **engine_kwargs)
        eng.load_codebook(path)
        return eng


