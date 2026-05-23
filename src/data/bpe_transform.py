from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from src.algorithms.compression.bpe_engine import BPEEngine

if TYPE_CHECKING:
    from src.data.unified_data_interface import UnifiedDataInterface
    from config import ProjectConfig


class BPETokenTransform:
    """Online BPE encoding transform.

    - Constructed inside DataLoader workers (rebuilds encoder from codebook).
    - Supports encode_rank_mode: all|topk|random|gaussian.
    - random/gaussian default to [0, len(merge_rules)] when min/max not provided.
    - Samples k once per batch (via BPEEngine.batch_encode internal logic).
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
        # Eval mode config
        eval_mode: Optional[str] = None,
        eval_topk: Optional[int] = None,
    ) -> None:
        self._engine = BPEEngine(
            train_backend="python",  # encode only, no training
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
        
        # Eval mode config
        self._eval_mode = eval_mode
        self._eval_topk = eval_topk
        self._is_training = True  # default to training mode

    def train(self, mode: bool = True) -> None:
        """Set training/eval mode."""
        self._is_training = mode
        
    def eval(self) -> None:
        """Set eval mode (deterministic encoding)."""
        self.train(False)
    
    def _get_current_engine(self) -> BPEEngine:
        """Return the appropriate encoding engine for the current mode."""
        if not self._is_training and self._eval_mode is not None:
            # Eval mode: deterministic encoding
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
            # Training mode: use original config
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
    """Build DataLoader worker_init_fn that reconstructs BPE Transform in each worker.

    Args:
        codebook: {'merge_rules': [...], 'vocab_size': int}
        engine_kwargs: encoding kwargs for BPETokenTransform (encode_backend, mode, etc.)
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
    """Batch-encode in worker (requires make_worker_init_fn to be set)."""
    if _g_bpe_transform is None:
        raise RuntimeError("BPE Transform not initialized; set worker_init_fn in DataLoader")
    return _g_bpe_transform.batch_encode(seqs)


# ---------------- UDI Integration ----------------

def create_bpe_transform_from_udi(
    udi: "UnifiedDataInterface", 
    config: "ProjectConfig",
    method: str
) -> BPETokenTransform:
    """Create a BPE Transform from UDI and config.

    Args:
        udi: unified data interface instance
        config: project config
        method: serialization method name

    Returns:
        BPETokenTransform instance (always created; mode controls behavior).
    """
    
    print("Creating BPE Transform...")
    
    try:
        # Get BPE codebook from UDI
        codebook = udi.get_bpe_codebook(method=method)
        print(f"BPE codebook loaded: {codebook['vocab_size']} tokens, {len(codebook['merge_rules'])} merge rules")
        
        # Get BPE params from config
        bpe_config = config.serialization.bpe
        
        # Create BPE Transform
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
        
        print("BPE Transform created successfully")
        print(f"   Backend: {bpe_config.encode_backend}")
        print(f"   Rank mode: {bpe_config.encode_rank_mode}")
        if bpe_config.encode_rank_mode == "none":
            print("   Mode: no BPE compression (0 merges)")
        elif bpe_config.encode_rank_k:
            print(f"   Top-K: {bpe_config.encode_rank_k}")
        
        return transform
        
    except Exception as e:
        print(f"Failed to create BPE Transform: {e}")
        print("Ensure BPE codebook has been built")
        raise


def create_bpe_worker_init_fn_from_udi(
    udi: "UnifiedDataInterface",
    config: "ProjectConfig", 
    method: str,
    *,
    split: str = "train",
):
    """Create DataLoader worker_init_fn from UDI and config.

    Args:
        udi: unified data interface instance
        config: project config
        method: serialization method name

    Returns:
        worker_init_fn (always valid; mode controls behavior).
    """
    
    try:
        # Get codebook
        codebook = udi.get_bpe_codebook(method=method)
        
        # Build engine kwargs
        bpe_config = config.serialization.bpe
        engine_config = bpe_config.engine
        # Train/eval split:
        # - Train: use training config (allows random/sampling)
        # - Val/test: prefer eval_mode/eval_topk; if unset and train is random/gaussian,
        #   map to topk(k=expected_value); for none/all/topk, reuse training settings.
        if str(split).lower() in {"val", "eval", "test"}:
            eval_mode = getattr(bpe_config, 'eval_mode', None)
            eval_topk = getattr(bpe_config, 'eval_topk', None)
            if eval_mode is not None:
                encode_rank_mode = eval_mode
                encode_rank_k = eval_topk
            else:
                train_mode = str(engine_config.encode_rank_mode).lower()
                # Default mapping rules
                if train_mode in {"random", "gaussian"}:
                    encode_rank_mode = "topk"
                    # Compute expected value
                    k_val = engine_config.encode_rank_k
                    k_min = engine_config.encode_rank_min
                    k_max = engine_config.encode_rank_max
                    if train_mode == "gaussian":
                        # Gaussian: use encode_rank_k as mean; fallback to max
                        if k_val is None:
                            k_val = k_max
                    else:  # random
                        # Random: prefer (min+max)/2; fallback to k or max
                        if (k_min is not None) and (k_max is not None):
                            try:
                                k_val = int(round((int(k_min) + int(k_max)) / 2))
                            except Exception:
                                k_val = k_val if k_val is not None else k_max
                        else:
                            k_val = k_val if k_val is not None else k_max
                    # Clip to [min, max] if provided
                    if (k_min is not None) and (k_max is not None) and (k_val is not None):
                        k_val = max(int(k_min), min(int(k_max), int(k_val)))
                    encode_rank_k = int(k_val) if k_val is not None else None
                else:
                    # none/all/topk: reuse training settings
                    encode_rank_mode = engine_config.encode_rank_mode
                    encode_rank_k = engine_config.encode_rank_k
        else:
            encode_rank_mode = engine_config.encode_rank_mode
            encode_rank_k = engine_config.encode_rank_k
        engine_kwargs = {
            'encode_backend': engine_config.encode_backend,
            'encode_rank_mode': encode_rank_mode,
            'encode_rank_k': encode_rank_k,
            'encode_rank_min': engine_config.encode_rank_min,
            'encode_rank_max': engine_config.encode_rank_max,
            'encode_rank_dist': engine_config.encode_rank_dist,
        }
        
        # Create worker_init_fn
        return make_worker_init_fn(codebook, engine_kwargs)
        
    except Exception as e:
        print(f"Failed to create BPE worker_init_fn: {e}")
        raise


def get_bpe_transform_info(config: "ProjectConfig") -> dict:
    """Return BPE transform configuration info."""
    
    bpe_config = config.serialization.bpe
    engine_config = bpe_config.engine
    
    # Return config info; mode controls actual behavior
    return {
        "mode": engine_config.encode_rank_mode,
        "num_merges": bpe_config.num_merges,
        "encode_backend": engine_config.encode_backend,
        "encode_rank_mode": engine_config.encode_rank_mode,
        "encode_rank_k": engine_config.encode_rank_k,
        "encode_rank_min": engine_config.encode_rank_min,
        "encode_rank_max": engine_config.encode_rank_max,
        "encode_rank_dist": engine_config.encode_rank_dist,
    }


