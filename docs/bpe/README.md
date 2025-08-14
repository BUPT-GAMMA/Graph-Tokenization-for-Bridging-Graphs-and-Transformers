# BPE (Byte Pair Encoding) 系统文档

## 📋 文档目录

本目录包含了TokenizerGraph项目中BPE系统的完整文档：

### 📚 **核心文档**

1. **[bpe_transform_rework.md](./bpe_transform_rework.md)** - BPE重构设计文档
   - 重构计划和技术架构设计
   - 实施阶段和里程碑规划
   - 配置管理和数据流设计

2. **[BPE_PROJECT_COMPLETION_REPORT.md](./BPE_PROJECT_COMPLETION_REPORT.md)** - 项目完成报告
   - 项目目标达成情况
   - 性能验证和质量保证
   - 生产部署建议和交接清单

3. **[BPE_Comprehensive_Performance_Report.md](./BPE_Comprehensive_Performance_Report.md)** - 性能测试报告
   - 6种编码模式的详细性能对比
   - QM9完整数据集基准测试结果
   - 生产配置推荐

### 📋 **迁移文档**

4. **[script_migration_plan.md](./script_migration_plan.md)** - 脚本迁移策略
   - 旧脚本识别和分类
   - 迁移优先级和执行计划

5. **[script_migration_report.md](./script_migration_report.md)** - 迁移执行报告
   - 迁移对照表和完成情况
   - 功能验证和性能确认

## 🚀 快速开始

### 基本使用

```python
from src.algorithms.compression.bpe_engine import BPEEngine

# 1. 创建BPE引擎
engine = BPEEngine(
    train_backend="numba",      # 训练后端: numba/python
    encode_backend="cpp",       # 编码后端: cpp/numba/python
    encode_rank_mode="topk",    # 编码模式: all/topk/random/gaussian
    encode_rank_k=1000         # 模式参数
)

# 2. 训练BPE
sequences = [[1, 2, 3, 4], [2, 3, 4, 5], ...]  # 输入序列
result = engine.train(
    token_sequences=sequences,
    num_merges=2000,           # 合并次数
    min_frequency=10           # 最小频率
)

# 3. 编码序列
encoded = engine.encode([1, 2, 3, 4])           # 单序列编码
encoded_batch = engine.batch_encode(sequences)  # 批量编码
```

### DataLoader集成

```python
from src.data.unified_data_interface import UnifiedDataInterface

# 1. 创建UDI实例
config = ProjectConfig()
config.dataset.transforms.use_bpe = True
config.dataset.transforms.bpe_mode = "topk"
udi = UnifiedDataInterface(config, "qm9")

# 2. 获取DataLoader组件
worker_init_fn, collate_fn = udi.get_bpe_worker_init_and_collate("feuler")

# 3. 创建DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=4,
    worker_init_fn=worker_init_fn,
    collate_fn=collate_fn
)
```

## ⚙️ 配置说明

### config.py 配置

```python
class SerializationConfig:
    bpe: BPEConfig = field(default_factory=BPEConfig)

class BPEConfig:
    bpe_num_merges: int = 2000
    bpe_min_frequency: int = 10
    
    class EngineConfig:
        train_backend: str = "numba"
        encode_backend: str = "cpp"
        encode_rank_mode: str = "all"
        encode_rank_k: Optional[int] = None
        encode_rank_min: Optional[int] = None
        encode_rank_max: Optional[int] = None
        encode_rank_dist: Optional[str] = None
```

### default_config.yml 配置

```yaml
serialization:
  bpe:
    bpe_num_merges: 2000
    bpe_min_frequency: 10
    engine:
      train_backend: "numba"
      encode_backend: "cpp"
      encode_rank_mode: "all"
      
dataset:
  transforms:
    use_bpe: false
    bpe_mode: "all"
    use_multiple_serialization: false
    num_realizations: 5
    flatten_variants: true
```

## 🎯 编码模式说明

| 模式 | 描述 | 性能 | 使用场景 |
|------|------|------|----------|
| **all** | 使用全部BPE规则 | 184,908 seq/s | 完整压缩，最大语义保持 |
| **topk** | 使用前k个规则 | 195,457 seq/s | 生产环境，稳定高性能 |
| **random** | 随机选择规则范围 | 195,407 seq/s | 实验研究，增加随机性 |
| **gaussian** | 高斯分布选择 | 194,971 seq/s | 偏向常用规则的随机化 |

## 📊 性能基准

基于QM9完整数据集（130,831序列）的测试结果：

### 训练性能
- **训练时间**: 26.96秒
- **训练速度**: 4,851.9 sequences/s
- **内存开销**: 3.6 MB

### 编码性能
- **最佳吞吐量**: 195,457 seq/s (Top-K模式)
- **最佳批次大小**: 256
- **内存开销**: 接近零

## 🛠️ 开发指南

### 添加新后端

1. 在`src/algorithms/compression/`下创建新后端文件
2. 实现`train`和`encode`接口
3. 在`BPEEngine`中注册新后端
4. 添加相应的测试用例

### 添加新编码模式

1. 在`BPEEngine._sample_topk()`中添加新模式逻辑
2. 更新配置验证规则
3. 添加配置参数说明
4. 编写单元测试

## 🧪 测试

运行BPE相关测试：

```bash
# 运行所有BPE测试
pytest tests/test_bpe_*

# 运行特定测试
pytest tests/test_bpe_token_transform.py
pytest tests/test_bpe_rework_comprehensive.py

# 运行性能基准测试
python scripts/benchmark_bpe_encode_unified.py
```

## 🚨 故障排除

### 常见问题

1. **导入错误**: 确保项目根目录在Python路径中
2. **配置错误**: 检查config.py和default_config.yml的参数
3. **内存不足**: 减少batch_size或使用流式处理
4. **性能问题**: 确保使用推荐的后端组合（numba+cpp）

### 错误处理

BPE系统使用确定性错误处理，所有错误都会提供清晰的错误信息：

```python
# 示例错误信息
ValueError: encode_rank_mode 必须是 ['all', 'topk', 'random', 'gaussian'] 之一
ValueError: encode_rank_k 必须是非负整数
ValueError: 缺少 BPE codebook（merge_rules/vocab_size）
```

## 📞 技术支持

- **Bug报告**: 请在项目issue中提交
- **功能请求**: 请参考项目贡献指南
- **性能问题**: 请提供详细的环境和数据信息

---

*文档更新时间: 2025-08-10*  
*版本: v2.0 (BPE重构完成版)*

