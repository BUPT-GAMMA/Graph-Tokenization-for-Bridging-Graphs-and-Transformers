
from __future__ import annotations

from typing import List

import torch

from config import ProjectConfig

def check_vocab_compatibility(logger, token_sequences: List[List[int]], vocab_manager) -> None:
  
    if not vocab_manager:
        logger.warning("Vocab manager not loaded, skipping compatibility check")
        return

    logger.info("Checking vocab compatibility...")
    all_tokens = set()
    for seq in token_sequences:
        all_tokens.update(seq)

    unknown_tokens = [t for t in all_tokens if t not in vocab_manager.token_to_id]
    total_tokens = len(all_tokens)
    unknown_count = len(unknown_tokens)
    unknown_ratio = (unknown_count / total_tokens * 100) if total_tokens > 0 else 0.0

    logger.info("Vocab compatibility stats:")
    logger.info(f"   Total token types: {total_tokens}")
    logger.info(f"   Unknown token types: {unknown_count}")
    logger.info(f"   Unknown token ratio: {unknown_ratio:.2f}%")

    if unknown_count > 0:
        if unknown_ratio > 10:
            logger.error(f"Unknown token ratio too high ({unknown_ratio:.2f}%)!")
            logger.error("   Consider using the same data pipeline as pretraining or retraining the model")
            logger.error(f"   Unknown token examples: {unknown_tokens[:10]}")
        elif unknown_ratio > 5:
            logger.warning(f"Unknown token ratio is high ({unknown_ratio:.2f}%)")
            logger.warning("   Check that data pipeline matches pretraining")
        else:
            logger.info(f"Vocab compatibility good, unknown token ratio low ({unknown_ratio:.2f}%)")
    else:
        logger.info("Vocab fully compatible, all tokens found in pretrained vocab")
        

def infer_task_property(config: ProjectConfig, udi) -> tuple[str, int | None]:
    """
    Infer task type and target info from dataset metadata.
    
    Args:
        config: Project config
        udi: Unified data interface
    """
    meta = udi.get_downstream_metadata()
    

    assert 'dataset_task_type' in meta, "Dataset metadata missing required field 'dataset_task_type'"
    task = meta['dataset_task_type']
    
    # Set default target property for regression tasks
    if task == 'regression' and not config.task.target_property:
        # QM9 datasets default to 'homo'
        if config.dataset.name.lower().startswith('qm9'):
            config.task.target_property = 'homo'
        else:
            # Other datasets use their default property
            if 'default_target_property' in meta and meta['default_target_property']:
                config.task.target_property = meta['default_target_property']
        
        if config.task.target_property:
            print(f"Auto-set regression target property: {config.task.target_property}")
    
    
        
def parse_torch_dtype(dtype_val):
    if dtype_val is None:
        return None
    if isinstance(dtype_val, torch.dtype):
        return dtype_val
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    return mapping.get(str(dtype_val).lower(), torch.float32)