#!/bin/bash
# 序列化方法速度测试运行脚本

echo "🚀 开始序列化方法速度基准测试"
echo "=================================="

# 创建输出目录
mkdir -p benchmark_results
cd /home/gzy/py/tokenizerGraph

# 快速测试（仅 qm9test 数据集）
echo "📊 运行快速测试 (qm9test数据集)..."
python benchmark_serialization_speed.py \
    --datasets qm9test \
    --batch-sizes 32 128 \
    --output benchmark_results/quick_benchmark_$(date +%Y%m%d_%H%M%S).csv

# 完整测试（包含 qm9 数据集，如果存在的话）
echo ""
echo "📊 运行完整测试 (qm9test + qm9数据集，如果可用)..."
python benchmark_serialization_speed.py \
    --datasets qm9test qm9 \
    --batch-sizes 32 128 \
    --output benchmark_results/full_benchmark_$(date +%Y%m%d_%H%M%S).csv

# 串行测试（不使用并行处理）
echo ""
echo "📊 运行串行测试 (仅串行处理)..."
python benchmark_serialization_speed.py \
    --datasets qm9test \
    --batch-sizes 32 128 \
    --no-parallel \
    --output benchmark_results/serial_benchmark_$(date +%Y%m%d_%H%M%S).csv

echo ""
echo "✅ 所有基准测试已完成！"
echo "📁 结果文件保存在 benchmark_results/ 目录中"
