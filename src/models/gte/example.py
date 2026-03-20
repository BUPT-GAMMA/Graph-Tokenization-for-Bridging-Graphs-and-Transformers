#!/usr/bin/env python3
"""
GTE模型基础使用示例
演示如何加载和使用GTE-multilingual-base模型
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def example_gte_usage():
    """GTE模型基础使用示例"""
    
    print("=== GTE模型基础使用示例 ===")
    # 强制仅使用GPU，禁止CPU回退
    if not torch.cuda.is_available():
        raise RuntimeError("必须使用GPU执行模型相关计算，但当前未检测到可用的CUDA设备")
    device = torch.device("cuda")
    
    # 1. 加载模型（推荐的高性能配置）
    print("1. 加载GTE模型...")
    model = AutoModel.from_pretrained(
        'Alibaba-NLP/gte-multilingual-base',
        trust_remote_code=True,
        unpad_inputs=True,                    # 关键优化：消除padding浪费
        use_memory_efficient_attention=True,  # 内存高效attention
        torch_dtype=torch.float16           # 半精度加速
    )
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        'Alibaba-NLP/gte-multilingual-base',
        trust_remote_code=True
    )
    
    print(f"模型加载成功，设备: {model.device}")
    print(f"隐藏维度: {model.config.hidden_size}")
    
    # 2. 测试文本数据
    test_texts = [
        "This is a short text",
        "This is a much longer text with many more words to test the model's ability to handle different length sequences",
        "Short again"
    ]
    
    print("\n2. 测试数据:")
    for i, text in enumerate(test_texts):
        print(f"  文本{i+1}: {len(text.split())}词 - \"{text[:40]}...\"")
    
    # 3. 编码处理
    print("\n3. 编码处理...")
    
    # 分词
    inputs = tokenizer(test_texts, 
                      padding=True, 
                      truncation=True, 
                      max_length=8192,
                      return_tensors='pt').to(device)
    
    print(f"输入shape: {inputs.input_ids.shape}")
    print(f"padding比例: {(inputs.input_ids == tokenizer.pad_token_id).float().mean():.2%}")
    
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 参照官方示例：使用CLS位置的向量作为句向量，并可选截断维度
    dimension = 768  # 输出embedding维度，应在[128, 768]
    embeddings = outputs.last_hidden_state[:, 0, :dimension]
    embeddings = F.normalize(embeddings)

    # 简单相似度演示：第一条与其余的相似度
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    print(f"相似度分数: {scores.tolist()}")
    
    # 4. 结果展示（每条向量范数，应接近1）
    print("\n4. 编码结果:")
    for i, embedding in enumerate(embeddings):
        norm = torch.norm(embedding).item()
        print(f"  文本{i+1} embedding范数: {norm:.4f}")
    
    print("\n✅ GTE模型示例运行完成")
    
    return embeddings

def example_token_id_input():
    """测试token ID直接输入（化学分子序列场景）"""
    
    print("\n=== Token ID直接输入测试 ===")
    # 强制仅使用GPU，禁止CPU回退
    if not torch.cuda.is_available():
        raise RuntimeError("必须使用GPU执行模型相关计算，但当前未检测到可用的CUDA设备")
    device = torch.device("cuda")
    
    # 加载模型
    model = AutoModel.from_pretrained(
        'Alibaba-NLP/gte-multilingual-base',
        trust_remote_code=True,
        unpad_inputs=True,
        use_memory_efficient_attention=True,
        torch_dtype=torch.float16
    )
    model = model.to(device)
    model.eval()
    
    # 模拟化学分子token ID序列
    token_sequences = [
        [1, 15, 23, 45, 67, 2],        # 短序列
        [1, 12, 34, 56, 78, 90, 11, 22, 33, 44, 55, 2],  # 中等序列
        [1, 8, 16, 24, 32, 2]          # 另一个短序列
    ]
    
    print("模拟分子token序列:")
    for i, seq in enumerate(token_sequences):
        print(f"  分子{i+1}: {seq} (长度: {len(seq)})")
    
    # 转换为tensor并padding
    max_len = max(len(seq) for seq in token_sequences)
    padded_sequences = []
    attention_masks = []
    
    for seq in token_sequences:
        padded = seq + [0] * (max_len - len(seq))  # 0是pad token
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        padded_sequences.append(padded)
        attention_masks.append(mask)
    
    input_ids = torch.tensor(padded_sequences, device=device)
    attention_mask = torch.tensor(attention_masks, device=device)
    
    print(f"\n输入tensor shape: {input_ids.shape}")
    print(f"attention mask shape: {attention_mask.shape}")
    
    # 模型推理
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # 与文本示例保持一致的句向量提取
    dimension = 768
    embeddings = outputs.last_hidden_state[:, 0, :dimension]
    embeddings = F.normalize(embeddings, p=2, dim=1)

    print(f"输出embedding shape: {embeddings.shape}")
    print("✅ Token ID输入测试成功")
    
    return embeddings

if __name__ == "__main__":
    # 运行基础示例
    example_gte_usage()
    
    # 运行token ID测试
    example_token_id_input()