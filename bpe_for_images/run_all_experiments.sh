#!/bin/bash
# 运行所有实验的便捷脚本

echo "=========================================="
echo "BPE for Images - 完整实验流程"
echo "=========================================="

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(dirname $(pwd))"

# 1. 训练Baseline模型
echo ""
echo "==> 1. 训练MLP Baseline..."
python train_mlp.py

echo ""
echo "==> 2. 训练LeNet-5 Baseline..."
python train_lenet.py

# 2. 训练Transformer（灰度值直接作为token）
echo ""
echo "==> 3. 训练Transformer-BERT..."
python train_transformer.py --transformer_type bert

echo ""
echo "==> 4. 训练Transformer-GTE..."
python train_transformer.py --transformer_type gte

# 3. 训练BPE模型
echo ""
echo "==> 5. 训练BPE模型..."
python train_bpe.py

# 4. 训练BPE+Transformer
echo ""
echo "==> 6. 训练BPE+Transformer-BERT..."
python train_bpe_transformer.py --transformer_type bert

echo ""
echo "==> 7. 训练BPE+Transformer-GTE..."
python train_bpe_transformer.py --transformer_type gte

# 5. 评估和对比
echo ""
echo "==> 8. 评估所有模型..."
python evaluate.py

echo ""
echo "==> 9. 生成对比报告..."
python compare_results.py

echo ""
echo "==> 10. 生成可视化图表..."
python visualize_results.py

echo ""
echo "=========================================="
echo "实验完成！"
echo "结果保存在 ./results/ 目录"
echo "=========================================="

