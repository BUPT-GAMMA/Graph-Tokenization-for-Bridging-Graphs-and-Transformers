from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from . import _cpp_bpe  # 已编译的 pybind11 模块
except Exception as e:  # noqa: E722
    raise ImportError(
        "未找到 _cpp_bpe 扩展，请先构建 C++ 后端（见 README/文档构建说明）。"
    ) from e


class CppBPEBackend:
    def __init__(self, encoder: "_cpp_bpe.MinBPEEncoder", vocab_size: Optional[int] = None) -> None:
        self._encoder = encoder
        self._vocab_size = int(vocab_size) if vocab_size is not None else None

    @classmethod
    def from_bpe_model(cls, model) -> "CppBPEBackend":
        # 优先使用模型派生结构（若有）
        if hasattr(model, "_minbpe_pair_to_rank") and hasattr(model, "_minbpe_pair_to_newid"):
            pair_to_rank = model._minbpe_pair_to_rank
            pair_to_newid = model._minbpe_pair_to_newid
        else:
            pair_to_rank = {}
            pair_to_newid = {}
            for idx, (l, r, new_id) in enumerate(getattr(model, "merge_rules", [])):
                pair_to_rank[(int(l), int(r))] = int(idx)
                pair_to_newid[(int(l), int(r))] = int(new_id)
        lefts, rights, ranks, new_ids = [], [], [], []
        for (l, r), rk in pair_to_rank.items():
            lefts.append(int(l))
            rights.append(int(r))
            ranks.append(int(rk))
            new_ids.append(int(pair_to_newid[(l, r)]))
        # 转为 int32 以匹配 C++ 实现的数据宽度
        enc = _cpp_bpe.MinBPEEncoder(
            np.asarray(lefts, dtype=np.int32).tolist(),
            np.asarray(rights, dtype=np.int32).tolist(),
            np.asarray(ranks, dtype=np.int32).tolist(),
            np.asarray(new_ids, dtype=np.int32).tolist(),
        )
        vocab_size = len(getattr(model, "token_to_id", {})) or None
        return cls(enc, vocab_size=vocab_size)

    @classmethod
    def from_codebook(
        cls,
        *,
        merge_rules: Optional[List[Tuple[int, int, int]]],
        pair_to_rank: Optional[Dict[Tuple[int, int], int]],
        pair_to_newid: Optional[Dict[Tuple[int, int], int]],
        vocab_size: Optional[int] = None,
    ) -> "CppBPEBackend":
        if pair_to_rank is None or pair_to_newid is None:
            if merge_rules is None:
                raise ValueError("需要提供 pair_to_rank/pair_to_newid 或 merge_rules 以构建 codebook")
            pair_to_rank = {}
            pair_to_newid = {}
            for idx, (l, r, new_id) in enumerate(merge_rules):
                pair_to_rank[(int(l), int(r))] = int(idx)
                pair_to_newid[(int(l), int(r))] = int(new_id)
        lefts, rights, ranks, new_ids = [], [], [], []
        for (l, r), rk in pair_to_rank.items():
            lefts.append(int(l))
            rights.append(int(r))
            ranks.append(int(rk))
            new_ids.append(int(pair_to_newid[(l, r)]))
        enc = _cpp_bpe.MinBPEEncoder(
            np.asarray(lefts, dtype=np.int32).tolist(),
            np.asarray(rights, dtype=np.int32).tolist(),
            np.asarray(ranks, dtype=np.int32).tolist(),
            np.asarray(new_ids, dtype=np.int32).tolist(),
        )
        return cls(enc, vocab_size=vocab_size)

    # -------- Training (minBPE) --------
    @staticmethod
    def train_minbpe_cpp(token_sequences: List[List[int]], num_merges: int, min_frequency: int) -> Dict:
        """使用 C++ 原生实现进行 minBPE 训练，返回与 Python 引擎一致的统计字典。"""
        results = _cpp_bpe.train_minbpe(token_sequences, int(num_merges), int(min_frequency))
        # 归一化返回结构为 Python 侧惯用 dict
        merge_rules_py: List[Tuple[int, int, int]] = []
        for t in results["merge_rules"]:
            l, r, nid = int(t[0]), int(t[1]), int(t[2])
            merge_rules_py.append((l, r, nid))
        return {
            "merge_rules": merge_rules_py,
            "num_merges_performed": int(results["num_merges_performed"]),
            "final_vocab_size": int(results["final_vocab_size"]),
            "base_vocab_size": int(results["base_vocab_size"]),
            "separator_token": int(results["separator_token"]),
        }

    def encode(self, seq: List[int]) -> List[int]:
        if not isinstance(seq, (list, tuple)):
            raise TypeError("encode 需要 List[int]")
        arr = np.asarray(seq, dtype=np.int32)
        return [int(x) for x in self._encoder.encode(arr.tolist())]

    def encode_topk(self, seq: List[int], topk: int) -> List[int]:
        if not isinstance(seq, (list, tuple)):
            raise TypeError("encode_topk 需要 List[int]")
        arr = np.asarray(seq, dtype=np.int32)
        return [int(x) for x in self._encoder.encode_with_limit(arr.tolist(), int(topk))]

    def batch_encode(self, seqs: List[List[int]]) -> List[List[int]]:
        if not isinstance(seqs, (list, tuple)):
            raise TypeError("batch_encode 需要 List[List[int]]")
        # 优先走 ragged 接口以减少 Python 循环与对象分配
        offsets = [0]
        flat = []
        for s in seqs:
            arr = np.asarray(s, dtype=np.int32)
            flat.extend(arr.tolist())
            offsets.append(len(flat))
        flat_out, out_offsets = self._encoder.encode_ragged(flat, offsets)
        results: List[List[int]] = []
        for i in range(len(out_offsets) - 1):
            b = out_offsets[i]
            e = out_offsets[i + 1]
            results.append([int(x) for x in flat_out[b:e]])
        return results

    def batch_encode_topk(self, seqs: List[List[int]], topk: int) -> List[List[int]]:
        if not isinstance(seqs, (list, tuple)):
            raise TypeError("batch_encode_topk 需要 List[List[int]]")
        # 使用 C++ 的批量限幅接口，避免 Python 循环
        arrs: List[List[int]] = []
        for s in seqs:
            arrs.append(np.asarray(s, dtype=np.int32).tolist())
        outs = self._encoder.batch_encode_with_limit(arrs, int(topk))
        return [[int(x) for x in out] for out in outs]

    # 调试/验证用：返回 (encoded, ranks_applied)
    def encode_with_limit_trace(self, seq: List[int], rank_limit: int) -> Tuple[List[int], List[int]]:
        arr = np.asarray(seq, dtype=np.int32)
        out, ranks = self._encoder.encode_with_limit_trace(arr.tolist(), int(rank_limit))
        return [int(x) for x in out], [int(r) for r in ranks]


