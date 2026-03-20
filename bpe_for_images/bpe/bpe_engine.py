from __future__ import annotations

from typing import List, Tuple, Optional, Dict

from .cpp_bpe_backend import CppBPEBackend


class BPEEngine:
    """
    极简独立版 BPE 引擎（仅支持 C++ 后端）。
    - 训练：minBPE 语义（每轮重统计、非重叠、稳定 tie-break）
    - 编码：使用 C++/pybind11 后端
    """

    def __init__(
        self,
        *,
        train_backend: str = "cpp",
        encode_backend: str = "cpp",
        verbose: bool = False,
    ) -> None:
        if train_backend != "cpp" or encode_backend != "cpp":
            raise ValueError("独立版BPEEngine仅支持 C++ 后端：train_backend=encode_backend='cpp'")
        self.train_backend = train_backend
        self.encode_backend = encode_backend
        self.verbose = bool(verbose)

        self.merge_rules: List[Tuple[int, int, int]] = []
        self.vocab_size: Optional[int] = None
        self._encoder: Optional[CppBPEBackend] = None

    # ---------------- Training ----------------
    def train(self, token_sequences: List[List[int]], *, num_merges: int, min_frequency: int) -> Dict:
        results = CppBPEBackend.train_minbpe_cpp(
            token_sequences, int(num_merges), int(min_frequency)
        )
        self.merge_rules = list(results["merge_rules"])  # [(l,r,new_id), ...]
        self.vocab_size = int(results["final_vocab_size"])
        return {
            "num_merges_performed": int(results["num_merges_performed"]),
            "final_vocab_size": self.vocab_size,
        }

    # ---------------- Encoding ----------------
    def build_encoder(self) -> None:
        self._encoder = CppBPEBackend.from_codebook(
            merge_rules=self.merge_rules,
            pair_to_rank=None,
            pair_to_newid=None,
            vocab_size=self.vocab_size,
        )

    def encode(self, seq: List[int]) -> List[int]:
        if self._encoder is None:
            raise RuntimeError("请先调用 build_encoder() 再进行编码")
        return self._encoder.encode(seq)

    def batch_encode(self, seqs: List[List[int]]) -> List[List[int]]:
        if self._encoder is None:
            raise RuntimeError("请先调用 build_encoder() 再进行编码")
        return self._encoder.batch_encode(seqs)










