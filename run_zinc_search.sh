#!/bin/bash
"""
简单的ZINC搜索启动脚本
"""

# 默认参数
JOURNAL_DIR="./journals"
PRETRAIN_TRIALS=50
FINETUNE_TRIALS=100

# 创建目录
mkdir -p "$JOURNAL_DIR"

# 4个BPE模式
BPE_MODES=("none" "all" "random" "gaussian")

for mode in "${BPE_MODES[@]}"; do
  echo "🎛️ 搜索BPE模式: $mode"
  
  python zinc_hyperopt.py \
    --bpe_mode "$mode" \
    --journal_file "$JOURNAL_DIR/zinc_${mode}.journal" \
    --stage both \
    --pretrain_trials $PRETRAIN_TRIALS \
    --finetune_trials $FINETUNE_TRIALS \
    --top_k 5
    
  if [ $? -eq 0 ]; then
    echo "✅ $mode 完成"
  else
    echo "❌ $mode 失败"
    break
  fi
done

echo "🎉 全部完成!"
