# TokenizerGraph 项目 TODO 列表

## 🔍 词汇表OOV检查增强（未来泛化时需要）

### 背景
当前项目只使用单一数据集（QM9），所以OOV率为0%。但在泛化到多数据集时需要全面的OOV检查。

### 需要修改的代码位置

#### 1. 扩展现有OOV检查功能
**文件**：`bert_finetuning_pipeline_normalized.py:484-542`
**当前功能**：只检查同数据集内的token兼容性
**需要增强**：
- 跨数据集OOV检查（qm9 vs mutag, zinc等）
- 跨序列化方法OOV检查（smiles vs graph_seq vs topological）
- 不同BPE设置的OOV检查（raw vs bpe_500 vs bpe_2000）

#### 2. 创建专门的OOV测试脚本
**新文件**：`test_vocabulary_oov.py`
**功能**：
- `test_cross_dataset_oov()`: 测试跨数据集OOV率
- `test_cross_method_oov()`: 测试跨序列化方法OOV率  
- `test_bpe_robustness()`: 测试不同BPE设置的鲁棒性
- `generate_oov_report()`: 生成OOV分析报告

#### 3. 增强词汇表管理器
**文件**：`bert_demo/vocab_manager.py:132-170`
**当前功能**：基本的token转换和未知token警告
**需要增强**：
- 详细的OOV统计和分析
- 按token频率分析OOV影响
- 可视化OOV分布

#### 4. 配置文件增强
**文件**：`config.py`
**需要添加**：
```python
# OOV检查配置
oov_check_enabled: bool = True
oov_warning_threshold: float = 0.05  # 5%以上OOV发出警告
oov_error_threshold: float = 0.15    # 15%以上OOV报错
cross_dataset_test: List[str] = ["qm9", "mutag", "zinc"]
cross_method_test: List[str] = ["smiles", "graph_seq", "topological"]
```

### 实施优先级
- **P3 - 低优先级**：当前单数据集场景不需要
- **触发条件**：开始多数据集实验时
- **预计工作量**：1-2天

---

## 🎯 数据分割标准化（高优先级 - 当前需要解决）

### 问题描述
各个管道使用不同的数据分割方式，导致验证集标签分布异常。

### 解决方案
创建统一的数据分割公共函数，采用深度学习标准的train/val/test三分割。

### 需要修改的文件
1. **新增**：`src/utils/data_split.py` - 公共数据分割函数
2. **修改**：`bert_finetuning_pipeline_normalized.py:593-600`
3. **修改**：`gnn_training_pipeline.py:386-420` 
4. **修改**：`bert_pretraining_pipeline_optimized.py:231-261`

### 实施计划
- [x] 创建公共数据分割函数 (`src/data/processed_data_loader.py`)
- [x] 修改所有管道使用统一分割
- [x] 添加数据分布验证
- [x] 确保所有管道使用相同随机种子

### 公共函数接口
```python
from src.data.processed_data_loader import split_data

# 深度学习标准三分割 (两步分割策略)
train_data, val_data, test_data = split_data(
    sequences=sequences, 
    labels=labels, 
    train_ratio=0.8, 
    random_seed=42
)
# 分割策略：先80%训练，剩余20%对半分给验证/测试
train_sequences, train_labels = train_data
val_sequences, val_labels = val_data
test_sequences, test_labels = test_data
```

### 已修改的文件
- ✅ `bert_pretraining_pipeline_optimized.py` - 使用标准接口
- ✅ `bert_finetuning_pipeline_normalized.py` - 使用标准接口  
- ✅ `gnn_training_pipeline.py` - 使用标准接口
- ✅ `src/models/simplified_processor.py` - 使用标准接口
- ✅ `src/models/data_processor.py` - 使用标准接口 


























欧拉、cpp、gseq的边token加入，开销都极大，检查逻辑原因。