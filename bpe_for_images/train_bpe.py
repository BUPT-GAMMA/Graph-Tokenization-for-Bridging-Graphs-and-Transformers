"""
训练BPE模型
===========

在MNIST训练集上训练BPE压缩模型
"""

import sys
from pathlib import Path
import argparse
import time

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATA_DIR, BPE_NUM_MERGES, BPE_MIN_FREQUENCY,
    get_bpe_model_path
)
from data import get_mnist_raw_data, prepare_flattened_sequences
from data.bpe_processor import ImageBPEProcessor, analyze_sequence_statistics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(args):
    """主流程"""
    
    logger.info("="*60)
    logger.info("训练BPE模型（MNIST灰度值序列）")
    logger.info("="*60)
    
    # 1. 加载MNIST训练数据
    logger.info("\n1. 加载MNIST训练数据...")
    images, labels = get_mnist_raw_data(str(DATA_DIR), train=True)
    logger.info(f"  训练样本数: {len(images)}")
    
    # 2. 准备展平的灰度值序列
    logger.info("\n2. 准备灰度值序列...")
    sequences = prepare_flattened_sequences(images)
    logger.info(f"  序列数量: {len(sequences)}")
    logger.info(f"  序列长度: {len(sequences[0])}")
    logger.info(f"  灰度值范围: [0, 255]")
    
    # 3. 训练BPE
    logger.info("\n3. 训练BPE模型...")
    logger.info(f"  合并次数: {args.num_merges}")
    logger.info(f"  最小频率: {args.min_frequency}")
    
    processor = ImageBPEProcessor(
        num_merges=args.num_merges,
        min_frequency=args.min_frequency
    )
    
    train_start = time.time()
    train_stats = processor.train(sequences)
    train_time = time.time() - train_start
    
    logger.info(f"  训练完成！耗时: {train_time:.2f}s")
    logger.info(f"  实际合并次数: {train_stats['actual_merges']}")
    logger.info(f"  最终词汇表大小: {train_stats['vocab_size']}")
    
    # 4. 编码所有序列（测试压缩效果）
    logger.info("\n4. 测试压缩效果...")
    encode_start = time.time()
    encoded_sequences, encode_stats = processor.encode(sequences)
    encode_time = time.time() - encode_start
    
    logger.info(f"  编码完成！耗时: {encode_time:.2f}s")
    logger.info(f"  平均压缩率: {encode_stats['avg_compression_ratio']:.2%}")
    
    # 5. 详细统计
    logger.info("\n5. 压缩统计...")
    stats = analyze_sequence_statistics(sequences, encoded_sequences)
    logger.info(f"  原始序列长度: {stats['original_length']['mean']:.1f}")
    logger.info(f"  压缩后序列长度: {stats['encoded_length']['mean']:.1f} "
               f"(min={stats['encoded_length']['min']}, "
               f"max={stats['encoded_length']['max']})")
    logger.info(f"  压缩率: {stats['compression_ratio']:.2%}")
    
    # 6. 保存BPE模型
    logger.info("\n6. 保存BPE模型...")
    save_path = get_bpe_model_path() if args.output is None else args.output
    processor.save(str(save_path))
    logger.info(f"  模型已保存到: {save_path}")
    
    # 7. 验证保存和加载
    logger.info("\n7. 验证保存/加载...")
    loaded_processor = ImageBPEProcessor.load(str(save_path))
    logger.info(f"  加载成功！词汇表大小: {loaded_processor.get_vocab_size()}")
    
    # 随机抽取几个样本验证编码/解码一致性
    test_indices = [0, 100, 1000, 10000]
    test_sequences = [sequences[i] for i in test_indices]
    encoded_test, _ = loaded_processor.encode(test_sequences)
    decoded_test = loaded_processor.decode(encoded_test)
    
    match_count = sum(
        1 for orig, dec in zip(test_sequences, decoded_test)
        if orig == dec
    )
    logger.info(f"  编码/解码验证: {match_count}/{len(test_sequences)} 匹配")
    
    logger.info("\n" + "="*60)
    logger.info("BPE训练完成！")
    logger.info(f"  词汇表大小: {processor.get_vocab_size()}")
    logger.info(f"  压缩率: {stats['compression_ratio']:.2%}")
    logger.info(f"  总耗时: {train_time + encode_time:.2f}s")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练BPE模型")
    parser.add_argument("--num_merges", type=int, default=BPE_NUM_MERGES,
                       help="BPE合并次数")
    parser.add_argument("--min_frequency", type=int, default=BPE_MIN_FREQUENCY,
                       help="最小频率阈值")
    parser.add_argument("--output", type=str, default=None,
                       help="输出路径（默认使用config中的路径）")
    
    args = parser.parse_args()
    main(args)

