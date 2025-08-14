# BPE脚本迁移计划

## 迁移目标分类

### 🎯 需要迁移的脚本（优先级高）

#### 1. 基准测试脚本
- `scripts/benchmark_bpe_encode.py` - BPE编码性能测试
- `scripts/benchmark_bpe_train.py` - BPE训练性能测试  
- `scripts/profile_bpe_breakdown.py` - BPE详细性能分析
- `scripts/profile_bpe_from_bin.py` - 二进制BPE性能测试

#### 2. 比较和验证脚本
- `run_serialization_bpe_comparison_simple.py` - 序列化BPE比较

### 📁 备份脚本（低优先级）
- `backup/data_refactor/quick_interface.py` - 数据接口（已弃用）
- `backup/legacy_scripts/enhanced_bert_training.py` - 增强BERT训练（已弃用）

### 🔧 支持文件（需要更新导入）
- `src/algorithms/compression/__init__.py` - 模块导入声明

### ✅ 已迁移/无需迁移
- `src/data/unified_data_interface.py` - 已迁移到BPEEngine
- `data_prepare.py` - 已迁移到BPEEngine
- `bert_pretrain.py` - 已迁移使用UDI
- `src/algorithms/compression/main_bpe.py` - 保留作为python后端
- `src/algorithms/compression/bpe_backup.py` - 备份文件，保持原样

## 迁移策略

### 阶段1: 基准测试脚本迁移
1. **benchmark_bpe_encode.py** 
   - 替换 `StandardBPECompressor` → `BPEEngine`
   - 更新性能测试逻辑以支持多后端
   - 保持接口兼容性

2. **benchmark_bpe_train.py**
   - 迁移训练基准到 `BPEEngine`
   - 支持numba/python后端比较

3. **profile_bpe_breakdown.py**
   - 增加BPEEngine选项
   - 保持现有实现作为对照

### 阶段2: 应用脚本迁移
1. **run_serialization_bpe_comparison_simple.py**
   - 迁移到新的BPE流程
   - 使用UDI接口

### 阶段3: 清理工作
1. 更新模块导入
2. 文档更新
3. 创建迁移对照表

## 迁移原则

### 1. 保持向后兼容
- 保留原有脚本作为`_legacy`版本
- 新脚本使用统一的BPEEngine接口

### 2. 统一配置
- 所有脚本使用ProjectConfig
- 支持命令行参数覆盖配置

### 3. 性能基准
- 迁移前后性能对比
- 确保新实现不劣于旧版本

### 4. 测试验证
- 每个迁移脚本都要有功能验证
- 输出结果一致性检查

## 实施计划

### Week 1: 基准脚本迁移
- [ ] 迁移 benchmark_bpe_encode.py
- [ ] 迁移 benchmark_bpe_train.py  
- [ ] 性能验证

### Week 2: 应用脚本迁移
- [ ] 迁移 run_serialization_bpe_comparison_simple.py
- [ ] 迁移 profile脚本
- [ ] 功能验证

### Week 3: 清理和文档
- [ ] 更新模块导入
- [ ] 创建迁移对照表
- [ ] 文档更新

## 风险与应对

### 风险1: 性能回退
- **应对**: 详细的性能基准对比
- **回退方案**: 保留legacy脚本

### 风险2: 接口不兼容  
- **应对**: 渐进式迁移，保持双接口
- **验证**: 输出一致性测试

### 风险3: 配置复杂化
- **应对**: 统一配置管理
- **文档**: 清晰的配置指南

