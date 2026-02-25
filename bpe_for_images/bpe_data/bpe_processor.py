"""
BPE数据处理器
=============

使用本地C++ BPE引擎实现，应用于图像灰度值序列的压缩
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from bpe_for_images.bpe import BPEEngine
from bpe_for_images.utils.logger import get_logger

logger = get_logger(__name__)


class ImageBPEProcessor:
    """
    图像BPE处理器
    
    将展平的灰度值序列（0-255）通过BPE压缩，减少序列长度
    """
    
    def __init__(
        self,
        num_merges: int = 200,
        min_frequency: int = 100,
        train_backend: str = "cpp",  # 使用C++后端
        encode_backend: str = "cpp"  # 使用C++后端
    ):
        """
        Args:
            num_merges: BPE合并次数
            min_frequency: 最小频率阈值
            train_backend: 训练后端 ("cpp" 或 "python")
            encode_backend: 编码后端 ("cpp" 或 "python")
        """
        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.train_backend = train_backend
        self.encode_backend = encode_backend
        self.bpe_engine = None
        self.is_trained = False
        
        logger.info(f"初始化ImageBPEProcessor: "
                   f"num_merges={num_merges}, min_frequency={min_frequency}, "
                   f"train_backend={train_backend}, encode_backend={encode_backend}")
    
    def train(
        self,
        sequences: List[List[int]]
    ) -> Dict[str, Any]:
        """
        在灰度值序列上训练BPE
        
        Args:
            sequences: 灰度值序列列表，每个序列为长度784的int列表
        
        Returns:
            训练统计信息
        """
        logger.info(f"开始训练BPE: {len(sequences)}个序列")
        
        # 创建BPE引擎（使用C++后端）
        self.bpe_engine = BPEEngine(
            train_backend=self.train_backend,
            encode_backend=self.encode_backend,
            verbose=False
        )
        
        # 训练BPE
        stats = self.bpe_engine.train(
            token_sequences=sequences,
            num_merges=self.num_merges,
            min_frequency=self.min_frequency
        )
        
        # 构建编码器（用于后续编码）
        self.bpe_engine.build_encoder()
        self.is_trained = True
        
        logger.info(f"BPE训练完成: "
                   f"实际合并次数={stats['num_merges_performed']}, "
                   f"词汇表大小={stats['final_vocab_size']}")
        
        return {
            'actual_merges': stats['num_merges_performed'],
            'vocab_size': stats['final_vocab_size'],
            'num_merges_performed': stats['num_merges_performed'],
            'final_vocab_size': stats['final_vocab_size']
        }
    
    def encode(
        self,
        sequences: List[List[int]]
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        使用训练好的BPE编码序列
        
        Args:
            sequences: 待编码的灰度值序列
        
        Returns:
            (encoded_sequences, stats):
                encoded_sequences: 编码后的序列
                stats: 压缩统计信息
        """
        if not self.is_trained:
            raise RuntimeError("BPE模型未训练，请先调用train()方法")
        
        # 批量编码序列
        encoded_sequences = self.bpe_engine.batch_encode(sequences)
        
        # 计算统计信息
        orig_lengths = [len(seq) for seq in sequences]
        enc_lengths = [len(seq) for seq in encoded_sequences]
        
        avg_compression_ratio = np.mean(enc_lengths) / np.mean(orig_lengths) if orig_lengths else 0.0
        
        stats = {
            'avg_length': np.mean(enc_lengths),
            'avg_compression_ratio': avg_compression_ratio,
            'num_sequences': len(sequences)
        }
        
        logger.info(f"BPE编码完成: {len(sequences)}个序列, "
                   f"平均压缩率={stats['avg_compression_ratio']:.2%}")
        
        return encoded_sequences, stats
    
    def decode(
        self,
        encoded_sequences: List[List[int]]
    ) -> List[List[int]]:
        """
        解码BPE序列（注意：BPEEngine不支持解码，此方法仅用于兼容性）
        
        Args:
            encoded_sequences: BPE编码的序列
        
        Returns:
            原始灰度值序列（当前实现不支持，返回空列表）
        """
        if not self.is_trained:
            raise RuntimeError("BPE模型未训练，请先调用train()方法")
        
        # BPEEngine的C++后端不支持解码，这是一个单向操作
        logger.warning("BPEEngine的C++后端不支持解码操作，返回空列表")
        return []
    
    def get_vocab_size(self) -> int:
        """获取BPE词汇表大小"""
        if not self.is_trained:
            return 256  # 初始灰度值词汇表大小
        return self.bpe_engine.vocab_size if self.bpe_engine.vocab_size is not None else 256
    
    def get_merge_rules(self) -> List[Tuple[int, int, int]]:
        """获取BPE合并规则"""
        if not self.is_trained:
            return []
        return self.bpe_engine.merge_rules if self.bpe_engine.merge_rules else []
    
    def save(self, save_path: str) -> None:
        """
        保存BPE模型
        
        Args:
            save_path: 保存路径（.pkl文件）
        """
        if not self.is_trained:
            raise RuntimeError("BPE模型未训练，无法保存")
        
        # 保存BPEEngine的状态（merge_rules和vocab_size）
        model_data = {
            "num_merges": self.num_merges,
            "min_frequency": self.min_frequency,
            "train_backend": self.train_backend,
            "encode_backend": self.encode_backend,
            "merge_rules": self.bpe_engine.merge_rules,
            "vocab_size": self.bpe_engine.vocab_size,
            "is_trained": self.is_trained
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"BPE模型已保存到: {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'ImageBPEProcessor':
        """
        加载BPE模型
        
        Args:
            load_path: 模型路径（.pkl文件）
        
        Returns:
            加载的ImageBPEProcessor实例
        """
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 兼容旧格式（StandardBPECompressor）
        if "bpe_compressor" in model_data:
            logger.warning("检测到旧格式BPE模型（StandardBPECompressor），将转换为BPEEngine格式")
            # 从旧格式提取信息
            old_compressor = model_data["bpe_compressor"]
            merge_rules = getattr(old_compressor, 'merge_rules', [])
            vocab_size = getattr(old_compressor, 'next_id', 256)
            train_backend = "cpp"
            encode_backend = "cpp"
        else:
            # 新格式（BPEEngine）
            merge_rules = model_data.get("merge_rules", [])
            vocab_size = model_data.get("vocab_size", 256)
            train_backend = model_data.get("train_backend", "cpp")
            encode_backend = model_data.get("encode_backend", "cpp")
        
        processor = cls(
            num_merges=model_data["num_merges"],
            min_frequency=model_data["min_frequency"],
            train_backend=train_backend,
            encode_backend=encode_backend
        )
        
        # 重建BPEEngine
        processor.bpe_engine = BPEEngine(
            train_backend=train_backend,
            encode_backend=encode_backend,
            verbose=False
        )
        processor.bpe_engine.merge_rules = merge_rules
        processor.bpe_engine.vocab_size = vocab_size
        processor.bpe_engine.build_encoder()
        processor.is_trained = model_data.get("is_trained", True)
        
        logger.info(f"BPE模型已从 {load_path} 加载")
        
        return processor


def analyze_sequence_statistics(
    original_sequences: List[List[int]],
    encoded_sequences: List[List[int]]
) -> Dict[str, Any]:
    """
    分析序列压缩统计信息
    
    Args:
        original_sequences: 原始序列
        encoded_sequences: 编码后的序列
    
    Returns:
        统计信息字典
    """
    orig_lengths = [len(seq) for seq in original_sequences]
    enc_lengths = [len(seq) for seq in encoded_sequences]
    
    stats = {
        "num_sequences": len(original_sequences),
        "original_length": {
            "mean": np.mean(orig_lengths),
            "min": np.min(orig_lengths),
            "max": np.max(orig_lengths)
        },
        "encoded_length": {
            "mean": np.mean(enc_lengths),
            "min": np.min(enc_lengths),
            "max": np.max(enc_lengths)
        },
        "compression_ratio": np.mean(enc_lengths) / np.mean(orig_lengths)
    }
    
    logger.info(f"序列统计: 原始长度={stats['original_length']['mean']:.1f}, "
               f"压缩后长度={stats['encoded_length']['mean']:.1f}, "
               f"压缩率={stats['compression_ratio']:.2%}")
    
    return stats


# ============== 测试代码 ==============
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from bpe_config import DATA_DIR, BPE_NUM_MERGES, BPE_MIN_FREQUENCY, get_bpe_model_path
    from bpe_data.mnist_loader import get_mnist_raw_data, prepare_flattened_sequences
    
    print("测试BPE处理器（C++后端）...")
    
    # 加载MNIST数据（使用子集测试）
    print("\n1. 加载数据...")
    images, labels = get_mnist_raw_data(str(DATA_DIR), train=True)
    sequences = prepare_flattened_sequences(images[:1000])  # 使用1000个样本测试
    print(f"   加载了 {len(sequences)} 个序列")
    
    # 训练BPE
    print("\n2. 训练BPE（C++后端）...")
    processor = ImageBPEProcessor(
        num_merges=50,  # 测试用较小值
        min_frequency=10,
        train_backend="cpp",
        encode_backend="cpp"
    )
    train_stats = processor.train(sequences[:500])  # 用一半训练
    print(f"   训练统计: {train_stats}")
    
    # 编码序列
    print("\n3. 编码序列（C++后端）...")
    encoded_seqs, encode_stats = processor.encode(sequences[500:])  # 用另一半测试
    print(f"   编码统计: {encode_stats}")
    
    # 检查编码结果
    print(f"\n4. 编码验证:")
    print(f"   原始序列长度: {len(sequences[500])}")
    print(f"   编码后长度: {len(encoded_seqs[0])}")
    print(f"   压缩率: {encode_stats['avg_compression_ratio']:.2%}")
    
    # 保存和加载测试
    print("\n5. 保存和加载测试...")
    test_save_path = "/tmp/test_bpe_cpp.pkl"
    processor.save(test_save_path)
    loaded_processor = ImageBPEProcessor.load(test_save_path)
    print(f"   保存/加载成功，词汇表大小={loaded_processor.get_vocab_size()}")
    
    print("\n测试完成！")

