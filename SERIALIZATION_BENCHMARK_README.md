# 序列化方法速度基准测试

本工具集用于测试和分析不同序列化方法在不同数据集上的性能表现，特别关注批处理和多线程并行处理的效果。

## 📁 文件说明

- `benchmark_serialization_speed.py` - 主要的基准测试脚本
- `analyze_benchmark_results.py` - 结果分析和可视化脚本  
- `run_serialization_benchmark.sh` - 便捷的批量测试脚本

## 🚀 快速开始

### 基本测试

```bash
# 测试所有方法在qm9test数据集上的性能
python benchmark_serialization_speed.py --datasets qm9test

# 测试特定方法
python benchmark_serialization_speed.py --methods feuler eulerian bfs --datasets qm9test

# 测试不同批处理大小
python benchmark_serialization_speed.py --batch-sizes 16 32 64 128 --datasets qm9test
```

### 使用便捷脚本

```bash
# 运行预设的多种测试
./run_serialization_benchmark.sh
```

### 分析结果

```bash
# 分析基准测试结果并生成可视化报告
python analyze_benchmark_results.py serialization_benchmark_results.csv
```

## 📊 测试内容

### 测试的序列化方法
- `feuler` - 频率引导欧拉路径 (推荐)
- `eulerian` - 标准欧拉路径
- `bfs` - 广度优先搜索
- `dfs` - 深度优先搜索  
- `topo` - 拓扑排序
- `cpp` - 中国邮路问题
- `fcpp` - 频率引导中国邮路

### 测试配置
- **数据集**: qm9test, qm9 (根据可用性)
- **批处理大小**: 32, 128 (可自定义)
- **处理模式**: 串行 vs 并行
- **线程数**: 自动检测CPU核心数，最大16

### 性能指标
- 总处理时间 
- 每样本平均处理时间
- 每秒处理样本数 (samples/sec)
- 成功率
- 并行加速比

## 📈 结果输出

### CSV结果文件包含字段：
- `method` - 序列化方法名称
- `dataset` - 数据集名称
- `batch_size` - 批处理大小
- `num_samples` - 测试样本数
- `total_time` - 总处理时间
- `avg_time_per_sample` - 每样本平均时间
- `samples_per_second` - 处理速度
- `parallel_enabled` - 是否使用并行
- `success_count` - 成功处理的样本数
- `error_count` - 失败的样本数
- `success_rate` - 成功率

### 分析报告包含：
- 📊 性能对比图表
- 🔥 性能热图  
- 📝 详细的Markdown分析报告
- 💡 优化建议

## 🛠️ 高级用法

### 自定义测试参数

```bash
# 指定最大工作线程数
python benchmark_serialization_speed.py --max-workers 8

# 只进行串行测试
python benchmark_serialization_speed.py --no-parallel

# 自定义输出文件名
python benchmark_serialization_speed.py --output my_benchmark_results.csv
```

### 大规模测试示例

```bash
# 测试所有方法，多种批处理大小
python benchmark_serialization_speed.py \
    --datasets qm9test qm9 \
    --batch-sizes 8 16 32 64 128 256 \
    --max-workers 16 \
    --output comprehensive_benchmark.csv
```

## 🔍 结果分析

分析脚本会自动生成：

1. **性能对比图表**
   - 各方法平均性能对比
   - 并行vs串行性能对比
   - 批处理大小影响分析
   - 数据集性能对比

2. **性能热图**
   - 方法×配置的性能矩阵
   - 成功率热图

3. **详细报告** (`benchmark_analysis_report.md`)
   - 性能排名
   - 并行加速效果分析
   - 失败案例分析
   - 优化建议

## ⚡ 性能优化建议

基于测试结果，通常表现较好的配置：

1. **快速序列化**: `feuler` 或 `eulerian` 方法
2. **并行处理**: 启用并行可显著提升大批量处理速度
3. **批处理大小**: 32-128 通常是较好的平衡点
4. **线程数**: 建议设置为 CPU 核心数的 0.5-1 倍

## 🐛 故障排除

### 常见问题
1. **内存不足**: 减少批处理大小或最大工作线程数
2. **某些方法失败**: 检查数据预处理是否正确
3. **速度过慢**: 先在qm9test小数据集上测试

### 调试模式
```bash
# 添加详细输出
python -v benchmark_serialization_speed.py --datasets qm9test
```

## 📋 依赖要求

确保已安装以下包：
- pandas  
- matplotlib
- seaborn
- numpy
- tqdm

## 📞 支持

如有问题，请检查：
1. 数据集是否正确加载
2. 序列化器是否正确初始化
3. 系统资源是否充足

---

**注意**: 首次运行建议先在 qm9test 数据集上进行快速测试，确认系统配置正确后再进行大规模基准测试。
