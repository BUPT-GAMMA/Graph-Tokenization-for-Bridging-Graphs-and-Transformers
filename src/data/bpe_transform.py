from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from src.algorithms.compression.bpe_engine import BPEEngine

if TYPE_CHECKING:
    from src.data.unified_data_interface import UnifiedDataInterface
    from config import ProjectConfig


class BPETokenTransform:
    """
    在线 BPE 编码 Transform。

    用法：
      - 在 DataLoader worker 内构造（从 codebook 重建编码端）
      - 支持 encode_rank_mode: all|topk|random|gaussian
      - random/gaussian 在未提供 min/max 时使用 [0, len(merge_rules)] 的默认区间
      - 批内仅采样一次 k（通过 BPEEngine 的 batch_encode 调用内部逻辑）
    """

    def __init__(
        self,
        *,
        merge_rules: List[tuple[int, int, int]],
        vocab_size: int,
        encode_backend: str = "cpp",
        encode_rank_mode: str = "all",
        encode_rank_k: Optional[int] = None,
        encode_rank_min: Optional[int] = None,
        encode_rank_max: Optional[int] = None,
        encode_rank_dist: Optional[str] = None,
        # 评估模式配置
        eval_mode: Optional[str] = None,
        eval_topk: Optional[int] = None,
    ) -> None:
        self._engine = BPEEngine(
            train_backend="python",  # 不训练，仅编码
            encode_backend=encode_backend,
            encode_rank_mode=encode_rank_mode,
            encode_rank_k=encode_rank_k,
            encode_rank_min=encode_rank_min,
            encode_rank_max=encode_rank_max,
            encode_rank_dist=encode_rank_dist,
        )
        self._engine.merge_rules = [tuple(map(int, t)) for t in merge_rules]
        self._engine.vocab_size = int(vocab_size)
        self._engine.build_encoder()
        
        # 评估模式配置
        self._eval_mode = eval_mode
        self._eval_topk = eval_topk
        self._is_training = True  # 默认处于训练模式

    def train(self, mode: bool = True) -> None:
        """设置训练/评估模式"""
        self._is_training = mode
        
    def eval(self) -> None:
        """设置为评估模式（确定性编码）"""
        self.train(False)
    
    def _get_current_engine(self) -> BPEEngine:
        """根据当前模式获取相应的编码引擎"""
        if not self._is_training and self._eval_mode is not None:
            # 评估模式：使用确定性编码
            eval_engine = BPEEngine(
                train_backend="python",
                encode_backend=self._engine.encode_backend,
                encode_rank_mode=self._eval_mode,
                encode_rank_k=self._eval_topk,
                encode_rank_min=None,
                encode_rank_max=None,
                encode_rank_dist=None,
            )
            eval_engine.merge_rules = self._engine.merge_rules
            eval_engine.vocab_size = self._engine.vocab_size
            eval_engine.build_encoder()
            return eval_engine
        else:
            # 训练模式：使用原始配置的编码
            return self._engine

    def encode(self, seq: List[int]) -> List[int]:
        engine = self._get_current_engine()
        return engine.encode(seq)

    def batch_encode(self, seqs: List[List[int]]) -> List[List[int]]:
        engine = self._get_current_engine()
        return engine.batch_encode(seqs)


# ---------------- Worker helper (optional) ----------------
_g_bpe_transform: BPETokenTransform | None = None


def make_worker_init_fn(codebook: dict, engine_kwargs: dict):
    """构建 DataLoader worker_init_fn：在 worker 内重建 BPE Transform。

    参数：
      - codebook: {'merge_rules': List[Tuple[int,int,int]], 'vocab_size': int}
      - engine_kwargs: 与 BPETokenTransform 构造的编码相关 kwargs（encode_backend/mode 等）
    """

    def _init_worker(_):
        global _g_bpe_transform
        _g_bpe_transform = BPETokenTransform(
            merge_rules=codebook['merge_rules'],
            vocab_size=int(codebook['vocab_size']),
            **engine_kwargs,
        )

    return _init_worker


def bpe_batch_encode_in_worker(seqs: list[list[int]]) -> list[list[int]]:
    """在 worker 中执行批量编码（依赖 make_worker_init_fn 预先初始化）。"""
    if _g_bpe_transform is None:
        raise RuntimeError("BPE Transform 未初始化：请在 DataLoader 中设置 worker_init_fn")
    return _g_bpe_transform.batch_encode(seqs)


# ---------------- UDI Integration ----------------

def create_bpe_transform_from_udi(
    udi: "UnifiedDataInterface", 
    config: "ProjectConfig",
    method: str
) -> BPETokenTransform:
    """
    从UDI和配置创建BPE Transform
    
    Args:
        udi: 统一数据接口实例
        config: 项目配置
        method: 序列化方法名
        
    Returns:
        BPE Transform实例（总是创建，根据mode控制行为）
    """
    
    print("🔧 创建BPE Transform...")
    
    try:
        # 从UDI获取BPE codebook（即使是none模式也需要基础词表信息）
        codebook = udi.get_bpe_codebook(method=method)
        print(f"✅ 获取BPE codebook成功: {codebook['vocab_size']} tokens, {len(codebook['merge_rules'])} merge rules")
        
        # 从配置获取BPE参数
        bpe_config = config.serialization.bpe
        
        # 创建BPE Transform（统一创建，mode控制行为）
        transform = BPETokenTransform(
            merge_rules=codebook['merge_rules'],
            vocab_size=codebook['vocab_size'],
            encode_backend=bpe_config.encode_backend,
            encode_rank_mode=bpe_config.encode_rank_mode,
            encode_rank_k=bpe_config.encode_rank_k,
            encode_rank_min=bpe_config.encode_rank_min,
            encode_rank_max=bpe_config.encode_rank_max,
            encode_rank_dist=bpe_config.encode_rank_dist,
            eval_mode=bpe_config.eval_mode,
            eval_topk=bpe_config.eval_topk,
        )
        
        print("✅ BPE Transform创建成功")
        print(f"   编码后端: {bpe_config.encode_backend}")
        print(f"   排序模式: {bpe_config.encode_rank_mode}")
        if bpe_config.encode_rank_mode == "none":
            print("   模式: 无BPE压缩（0次合并）")
        elif bpe_config.encode_rank_k:
            print(f"   Top-K: {bpe_config.encode_rank_k}")
        
        return transform
        
    except Exception as e:
        print(f"❌ BPE Transform创建失败: {e}")
        print("💡 请确保已构建BPE codebook")
        raise


def create_bpe_worker_init_fn_from_udi(
    udi: "UnifiedDataInterface",
    config: "ProjectConfig", 
    method: str
):
    """
    从UDI和配置创建DataLoader的worker_init_fn
    
    Args:
        udi: 统一数据接口实例
        config: 项目配置
        method: 序列化方法名
        
    Returns:
        worker_init_fn函数，总是返回有效的函数（统一创建，mode控制行为）
    """
    
    try:
        # 获取codebook（统一获取，不管什么mode）
        codebook = udi.get_bpe_codebook(method=method)
        
        # 构建engine参数（统一构建，mode参数控制实际行为）
        bpe_config = config.serialization.bpe
        engine_config = bpe_config.engine
        engine_kwargs = {
            'encode_backend': engine_config.encode_backend,
            'encode_rank_mode': engine_config.encode_rank_mode,  # 包括"none"模式
            'encode_rank_k': engine_config.encode_rank_k,
            'encode_rank_min': engine_config.encode_rank_min,
            'encode_rank_max': engine_config.encode_rank_max,
            'encode_rank_dist': engine_config.encode_rank_dist,
        }
        
        # 统一创建worker_init_fn（不管mode，都创建transform对象）
        return make_worker_init_fn(codebook, engine_kwargs)
        
    except Exception as e:
        print(f"❌ BPE worker_init_fn创建失败: {e}")
        raise


def get_bpe_transform_info(config: "ProjectConfig") -> dict:
    """
    获取BPE Transform配置信息
    
    Args:
        config: 项目配置
        
    Returns:
        BPE配置信息字典
    """
    
    bpe_config = config.serialization.bpe
    engine_config = bpe_config.engine
    
    # 统一返回配置信息，mode控制具体行为
    return {
        "mode": engine_config.encode_rank_mode,  # 核心控制参数
        "num_merges": bpe_config.num_merges,
        "encode_backend": engine_config.encode_backend,
        "encode_rank_mode": engine_config.encode_rank_mode,
        "encode_rank_k": engine_config.encode_rank_k,
        "encode_rank_min": engine_config.encode_rank_min,
        "encode_rank_max": engine_config.encode_rank_max,
        "encode_rank_dist": engine_config.encode_rank_dist,
    }


