# 下一步行动指南

## 🔥 立即处理的关键问题

### 1. 修复序列化器dtype不匹配问题

**问题**：LRGB数据集序列化时出现dtype错误
```
RuntimeError: Index put requires the source and destination dtypes match, 
got Long for the destination and Int for the source.
```

**排查步骤**：

1. **检查LRGB数据的实际格式**：
```bash
python -c "
import gzip, pickle, dgl
with gzip.open('data/peptides_func/data.pkl.gz', 'rb') as f:
    data = pickle.load(f)
sample = data[0]
if isinstance(sample, tuple):
    g = sample[0]  # 假设图在tuple的第一个位置
elif isinstance(sample, dict):
    g = sample['dgl_graph']
else:
    g = sample
print('节点token IDs类型:', g.ndata['node_token_ids'].dtype)
print('边token IDs类型:', g.edata['edge_token_ids'].dtype)
"
```

2. **检查序列化器中所有可能的dtype不匹配位置**：
   - `src/algorithms/serializer/base_serializer.py:957` ✅ 已修复
   - `src/algorithms/serializer/base_serializer.py:960` ✅ 已修复
   - 检查是否还有其他位置需要修复

3. **验证修复生效**：
```bash
python prepare_data_new.py -d peptides_func -m bfs -v
```

### 2. 验证数据格式一致性

**检查预处理和加载的数据格式契约**：
- 确认数据预处理保存的格式
- 验证数据加载器的访问方式
- 修复任何不一致的地方

## 📋 后续工作优先级

### 优先级1：核心功能完善
- [ ] 完成dtype问题修复
- [ ] 验证LRGB数据正确加载
- [ ] 成功运行序列化预处理

### 优先级2：Pipeline集成
- [ ] 更新`src/training/finetune_pipeline.py`使用统一模型
- [ ] 测试完整的训练流程
- [ ] 验证所有任务类型正常工作

### 优先级3：完善测试
- [ ] 运行`test_unified_model_finetune.py`
- [ ] 测试所有数据集：zinc, synthetic, peptides_func, peptides_struct
- [ ] 验证统一模型性能

## 🛠️ 具体命令清单

### 排查问题
```bash
# 检查数据格式
python -c "检查脚本..."

# 测试序列化
python prepare_data_new.py -d peptides_func -m bfs -v

# 测试统一模型
python test_unified_model_finetune.py
```

### 验证修复
```bash
# 完整预处理测试
python prepare_data_new.py -d peptides_func -m feuler -v

# 运行分析验证
python analyze_lrgb_datasets.py
```

## 📁 关键文件位置

**需要关注的文件**：
- `src/algorithms/serializer/base_serializer.py` - dtype修复位置
- `src/data/loader/peptides_func_loader.py` - 数据加载器
- `test_unified_model_finetune.py` - 测试脚本
- `UNIFIED_MODEL_PROGRESS.md` - 详细进度报告

**当前TODO状态**：
- ✅ 统一模型架构设计完成
- ✅ LRGB数据集集成完成  
- ✅ 数据处理严格化完成
- 🚧 **dtype问题修复中（当前阻塞）**
- ⏳ Pipeline更新等待中
- ⏳ 完整测试等待中

---

*下次继续工作时，直接从dtype问题排查开始*
