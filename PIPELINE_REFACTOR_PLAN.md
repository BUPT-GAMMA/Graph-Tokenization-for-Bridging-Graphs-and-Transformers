# Pipeline完整适配重构规划
# Pipeline Complete Adaptation Refactor Plan

**目的**: 完全适配UniversalModel架构，消除所有备份代码依赖

---

## 🚨 **发现的问题**

### **1. 上层Pipeline未完全适配 (严重)**
```python
# ❌ src/training/finetune_pipeline.py 中的问题
from backup.models.bert.model_legacy import BertMLM, create_bert_mlm  # 还在调用备份代码！
return BertMLM.load_model(str(p), config)  # 返回过时的模型类型
backbone = create_bert_mlm(...)  # 创建过时的模型
```

### **2. 预训练模型加载逻辑不灵活**
```python
# ❌ 当前逻辑：只能从同实验名加载
pretrained_path = f"model/{experiment_group}/{experiment_name}/..."

# ✅ 期望逻辑：支持灵活的预训练实验名指定
--experiment_name finetune_exp --pretrain_exp_name pretrain_exp
```

---

## 🔧 **完整重构规划**

### **Phase 1: 清理backup代码依赖**

#### **1.1 重构 src/training/finetune_pipeline.py**

**目标**: 完全移除backup代码调用

**变更内容**:
```python
# Before: 错误的backup依赖
def load_pretrained_backbone(config):
    encoder_type = getattr(config, 'encoder_type', 'bert')  # ❌ 使用旧的属性名
    from backup.models.bert.model_legacy import BertMLM  # ❌ 调用备份代码

# After: 统一架构
def load_pretrained_backbone(config, pretrained_dir=None, pretrain_exp_name=None):
    encoder_type = config.encoder.type  # ✅ 使用新的配置
    # 🆕 返回UniversalModel或None，不再返回BertMLM
```

**具体修改**:
- [ ] 删除所有`from backup.models`导入
- [ ] 修改`load_pretrained_backbone()`返回`UniversalModel`或`None`
- [ ] 实现灵活的预训练模型路径解析
- [ ] 更新`_load_bert_backbone()`和`_load_gte_backbone()`

#### **1.2 重构 src/training/model_builder.py**

**目标**: 统一使用UniversalModel架构

**变更内容**:
```python
# Before: 分离的加载逻辑
def _load_bert_backbone(config):
    return BertMLM.load_model(...)  # ❌ 返回过时类型

# After: 统一的加载逻辑
def _load_pretrained_universal_model(config, model_path):
    return UniversalModel.load_model(model_path)  # ✅ 返回统一类型
```

#### **1.3 检查其他pipeline文件**

**需要检查的文件**:
- [ ] `src/training/pretrain_pipeline.py` - 已重构 ✅
- [ ] `src/training/finetune_pipeline.py` - 需要重构 ❌
- [ ] `src/training/model_builder.py` - 需要重构 ❌
- [ ] `src/training/pretrain_api.py` - 检查是否需要适配

### **Phase 2: 灵活的预训练模型加载**

#### **2.1 新的参数设计**

**命令行参数扩展**:
```bash
# 情况1: 默认行为
python run_finetune.py --dataset zinc --task regression
# → experiment_name="zinc_feuler_regression_xxxxx"
# → 从同名预训练模型加载

# 情况2: 指定实验名
python run_finetune.py --dataset zinc --task regression --experiment_name my_finetune
# → 保存到experiment_name="my_finetune"
# → 从同名预训练模型"my_finetune"加载

# 情况3: 分离预训练和微调实验名 (新功能)
python run_finetune.py --dataset zinc --task regression \
    --experiment_name my_finetune --pretrain_exp_name my_pretrain
# → 保存到experiment_name="my_finetune"  
# → 从pretrain_exp_name="my_pretrain"加载预训练模型
```

#### **2.2 实现预训练模型路径解析器**

**新增模块**: `src/utils/model_path_resolver.py`
```python
class ModelPathResolver:
    """预训练模型路径解析器"""
    
    def resolve_pretrained_path(
        config: ProjectConfig,
        experiment_name: str = None,
        pretrain_exp_name: str = None
    ) -> Optional[Path]:
        """
        解析预训练模型路径
        
        优先级:
        1. pretrain_exp_name (如果提供)
        2. experiment_name  
        3. 默认路径查找
        """
```

#### **2.3 更新命令行参数处理**

**修改文件**: `src/utils/config_override.py`
```python
# 新增参数
parser.add_argument("--pretrain_exp_name", type=str, 
                   help="预训练模型实验名（如果与微调实验名不同）")
```

### **Phase 3: UniversalModel保存/加载实现**

#### **3.1 实现UniversalModel.load_model()**

**目标**: 支持加载UniversalModel格式的预训练模型

**实现位置**: `src/models/universal_model.py`
```python
@classmethod
def load_model(cls, model_path: str, config: ProjectConfig = None) -> 'UniversalModel':
    """
    加载UniversalModel
    
    支持格式:
    1. UniversalModel原生格式
    2. 旧的BertMLM格式 (自动转换)
    """
```

#### **3.2 实现格式转换器**

**新增模块**: `src/utils/model_converter.py`
```python
def convert_bertmlm_to_universal(bertmlm_path: str, output_path: str):
    """将旧的BertMLM格式转换为UniversalModel格式"""
```

### **Phase 4: Pipeline接口统一**

#### **4.1 统一所有pipeline的模型创建**

**目标**: 所有pipeline都使用`create_universal_model`

**修改范围**:
- `src/training/pretrain_pipeline.py` ✅ 已完成
- `src/training/finetune_pipeline.py` ❌ 需要修改
- `src/training/pretrain_api.py` ❌ 需要检查

#### **4.2 统一错误处理和日志**

**目标**: 一致的错误消息和日志格式

**示例**:
```python
# 统一的错误消息
logger.warning("⚠️ 未找到预训练UniversalModel，将使用随机初始化")
logger.info(f"✅ UniversalModel加载成功: {model.task_type}任务")
```

---

## 📋 **具体实施计划**

### **🔥 高优先级 (立即修复)**

#### **Step 1: 完全清理backup依赖**
- [ ] 修复 `src/training/finetune_pipeline.py` 的backup导入
- [ ] 修复 `src/training/model_builder.py` 的backup依赖
- [ ] 实现 `UniversalModel.load_model()` 方法

#### **Step 2: 实现灵活的预训练加载**
- [ ] 添加 `--pretrain_exp_name` 参数
- [ ] 实现 `ModelPathResolver` 类
- [ ] 更新路径解析逻辑

#### **Step 3: 验证主要脚本**
- [ ] 测试 `run_pretrain.py` 30轮预训练
- [ ] 测试 `run_finetune.py` 30轮微调
- [ ] 验证预训练→微调流程

### **🟡 中优先级 (后续改进)**

#### **Step 4: 完善格式转换**
- [ ] 实现BertMLM→UniversalModel格式转换
- [ ] 提供迁移工具脚本
- [ ] 更新文档说明

#### **Step 5: 统一接口规范**
- [ ] 统一所有pipeline的返回格式
- [ ] 统一错误处理和日志格式
- [ ] 完善类型注解

---

## 🎯 **预期效果**

### **修复后的使用方式**

#### **预训练 (不变)**
```bash
python run_pretrain.py --dataset zinc --method feuler --epochs 30 \
    --experiment_group my_group --experiment_name my_pretrain
```

#### **微调 (新增灵活性)**
```bash
# 情况1: 默认行为
python run_finetune.py --dataset zinc --task regression --finetune_epochs 30

# 情况2: 指定微调实验名，从同名预训练加载
python run_finetune.py --dataset zinc --task regression --finetune_epochs 30 \
    --experiment_name my_finetune

# 情况3: 分离预训练和微调实验名 (新功能)
python run_finetune.py --dataset zinc --task regression --finetune_epochs 30 \
    --experiment_name my_finetune --pretrain_exp_name my_pretrain
```

### **架构效果**
- ✅ 所有pipeline使用UniversalModel
- ✅ 完全消除backup代码依赖
- ✅ 灵活的预训练模型指定
- ✅ 主要脚本正常运行

---

## ⚠️ **注意事项**

1. **完全适配**: 不考虑向前兼容，彻底使用新架构
2. **路径管理**: 预训练和微调可以使用不同的实验名
3. **错误处理**: 当找不到指定的预训练模型时，给出清晰的错误信息
4. **格式转换**: 自动检测和转换旧格式的预训练模型

准备好开始实施这个完整的pipeline适配吗？
