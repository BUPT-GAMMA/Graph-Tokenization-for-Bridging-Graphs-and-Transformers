# 代码规范指南 (Coding Standards Guide)

## 1. 总体原则

### 1.1 代码哲学
- **清晰性优于简洁性**：代码首先应该易于理解，而不是追求最少的代码行数
- **显式优于隐式**：避免隐含的假设和魔法行为
- **一致性优于个人偏好**：团队规范高于个人编码习惯
- **可测试性作为设计目标**：代码应该易于测试和验证

### 1.2 科研代码特殊原则
- **无静默失败**：任何异常都应该明确报错，不使用默认fallback
- **数据真实性**：除初期管道测试外，禁止使用mock数据
- **算法透明性**：核心算法实现应该清晰可追踪
- **实验可重现**：所有随机性必须可控

## 2. 项目结构规范

### 2.1 目录组织
```
project_root/
├── src/                      # 源代码
│   ├── data/                 # 数据层接口
│   ├── algorithms/           # 核心算法实现
│   │   ├── serializer/       # 序列化算法
│   │   └── compression/      # 压缩算法
│   ├── models/               # 模型实现
│   └── utils/                # 工具函数
├── config/                   # 配置文件
├── data/                     # 数据存储
├── model/                    # 模型存储
├── logs/                     # 日志文件
├── outputs/                  # 实验输出
├── tests/                    # 测试代码
└── docs/                     # 文档
```

### 2.2 文件组织原则
- **单一职责**：每个文件应该有明确的单一功能
- **模块化分离**：大型功能模块使用文件夹组织，每个方法一个文件
- **接口统一**：通过基类定义接口，通过工厂模式创建实例
- **避免循环依赖**：模块间依赖应该是有向无环的

## 3. 命名约定

### 3.1 Python命名规范
```python
# 类名：使用PascalCase
class GraphSerializer:
    pass

# 函数和方法：使用snake_case
def serialize_graph(graph_data):
    pass

# 常量：使用UPPER_SNAKE_CASE
MAX_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 32

# 私有方法：使用单下划线前缀
def _internal_helper():
    pass

# 变量：使用snake_case
node_features = []
edge_indices = []
```

### 3.2 文件命名
- 模块文件：`module_name.py`
- 测试文件：`test_module_name.py`
- 配置文件：`config.py` 或 `config.yaml`
- 脚本文件：`run_experiment_name.py`

### 3.3 特殊命名约定示例
```python
# 数据加载相关
loader = DataFactory.create_loader("dataset_name")

# 序列化方法命名
def serialize(self, graph):         # 单图单次
def batch_serialize(self, graphs):  # 多图单次
def multiple_serialize(self, graph): # 单图多次
def batch_multiple_serialize(self, graphs): # 多图多次

# 配置参数命名
config.data_dir           # 数据目录
config.cache_dir          # 缓存目录
config.model_checkpoint   # 模型检查点
```

## 4. 代码风格

### 4.1 导入规范
```python
# 标准库导入
import os
import sys
from typing import List, Dict, Optional

# 第三方库导入
import numpy as np
import torch
from torch import nn

# 本地模块导入
from src.data.base_loader import BaseLoader
from src.algorithms.serializer import SerializerFactory
```

### 4.2 类型注解
```python
def process_graph(
    graph_data: Dict[str, torch.Tensor],
    max_length: int = 512,
    padding: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    处理图数据并返回序列化结果。
    
    Args:
        graph_data: 包含节点和边信息的字典
        max_length: 最大序列长度
        padding: 是否进行填充
        
    Returns:
        (token_ids, attention_mask) 元组
    """
    pass
```

### 4.3 异常处理
```python
# 明确的错误处理
def load_dataset(dataset_name: str) -> Dataset:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets are: {SUPPORTED_DATASETS}"
        )
    
    # 不使用静默的fallback
    # 错误示例：
    # try:
    #     return load_primary_dataset()
    # except:
    #     return load_backup_dataset()  # 避免这种模式
```

如果确有需要的fallback处理，则应该与用户/主管确认。通过后再相应位置添加注释说明

## 5. 数据处理严格编程原则

### 5.1 核心原则：预处理与加载的严格一致性

**❌ 错误做法：假设性编程**
```python
def process_labels(label_obj):
    if isinstance(label_obj, np.ndarray):
        if label_obj.ndim == 0:
            # 处理标量
            return label_obj.item()
        elif label_obj.size == 1:
            # 处理单元素
            return label_obj.item()
        else:
            # 处理多元素，可能是one-hot
            return np.argmax(label_obj)
    elif isinstance(label_obj, (list, tuple)):
        # 处理列表，可能是...
        if len(label_obj) == 1:
            return label_obj[0]
        else:
            return np.argmax(label_obj)
    else:
        # 其他情况，猜测处理
        return int(label_obj)
```

**✅ 正确做法：严格契约**
```python
def process_labels(label_obj):
    # 预处理阶段明确定义：输出shape=(10,)的numpy数组
    assert isinstance(label_obj, np.ndarray) and label_obj.shape == (10,), \
        f"标签格式错误，期望shape=(10,)的numpy数组，得到: {type(label_obj)}, shape={getattr(label_obj, 'shape', 'no-shape')}"
    return label_obj.tolist()
```

### 5.2 数据处理黄金规则

1. **单一数据源原则**
   - 预处理阶段定义唯一的数据格式
   - 加载器严格按照预处理定义的格式处理
   - 禁止在加载器中处理多种可能的格式

2. **无回退逻辑原则**
   - 不写"如果A不行就试B"的代码
   - 数据格式不符合预期直接报错
   - 避免使用`.get(key, default_value)`处理关键数据字段

3. **fail-fast原则**
   - 使用`assert`而不是`if-else`处理预期格式
   - 数据格式错误立即抛异常，不要静默处理
   - 及时暴露数据问题，而不是掩盖

4. **严格类型验证**
   ```python
   # ❌ 错误：多种兼容性处理
   if isinstance(data, list):
       # 处理列表
   elif isinstance(data, np.ndarray):
       # 处理数组
   elif hasattr(data, 'tolist'):
       # 处理其他可转换类型
   
   # ✅ 正确：严格验证预期格式
   assert isinstance(data, np.ndarray) and data.shape == (expected_shape), \
       f"数据格式错误，期望{expected_shape}的numpy数组"
   ```

5. **数据存储契约原则**
   - 预处理脚本明确定义存储格式
   - 加载器严格按照存储格式读取
   - 存储路径、文件名、数据结构都应该确定

### 5.3 实践指导

**预处理阶段 (prepare_*.py)**：
- 明确定义输出数据的类型、形状、命名
- 保存数据格式说明（如token_mappings.json）
- 使用确定性的算法和固定种子

**数据加载器 (src/data/loader/*_loader.py)**：
- 严格按照预处理定义的格式读取
- 对预期外的数据格式直接报错
- 不要写兼容性代码处理多种可能格式

**示例对比**：
```python
# ❌ 错误：充满假设的兼容性代码
def load_graph_data(file_path):
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        # 尝试其他格式...
        
    # 处理多种可能的数据结构...
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], tuple):
            # 处理tuple格式
        elif isinstance(data[0], dict):
            # 处理dict格式
        # ... 更多假设

# ✅ 正确：严格按照预处理契约
def load_graph_data(file_path):
    # 预处理明确定义：保存为data.pkl.gz的gzip压缩pickle文件
    # 格式：List[Tuple[lightweight_graph_dict, label_array]]
    with gzip.open(file_path, 'rb') as f:
        data: List[Tuple[Dict[str, Any], np.ndarray]] = pickle.load(f)
    return data
```

## 6. 配置管理

### 6.1 单一配置源
```python
# config.py - 所有配置的单一来源
@dataclass
class Config:
    # 数据配置
    data_dir: str = "data/raw"
    cache_dir: str = "data/cache"
    
    # 模型配置
    model_type: str = "bert"
    hidden_size: int = 512
    num_layers: int = 4
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    
    # 实验配置
    seed: int = 42
    device: str = "cuda"
```

### 5.2 配置使用原则
- **禁止硬编码**：除非是函数内部的局部常量
- **统一访问**：通过config对象访问所有配置
- **命令行覆盖**：支持但最小化命令行参数覆盖，应主要用于自动化脚本
- **配置验证**：启动时验证配置的合法性

## 6. 文档规范

### 6.1 函数文档示例
```python
def serialize_graph_eulerian(
    graph: nx.Graph,
    frequency_map: Dict[str, int],
    start_node: Optional[int] = None
) -> List[str]:
    """
    使用基于频率的欧拉路径算法序列化图。
    
    该算法通过全局频率统计指导遍历顺序，确保相同的图
    总是产生相同的序列化结果。
    
    Args:
        graph: NetworkX图对象
        frequency_map: 全局边类型频率映射
        start_node: 起始节点（None则自动选择）
        
    Returns:
        表示序列化路径的token列表
        
    Raises:
        ValueError: 如果图不连通
        RuntimeError: 如果无法找到欧拉路径
        
    Examples:
        >>> g = nx.Graph([(0,1), (1,2)])
        >>> freq = {"single": 100, "double": 50}
        >>> tokens = serialize_graph_eulerian(g, freq)
        >>> print(tokens)
        ['[CLS]', 'C', 'single', 'C', 'single', 'C', '[SEP]']
    """
    pass
```

### 6.2 类文档示例
```python
class GraphSerializer(BaseSerializer):
    """
    图序列化器的XX实现。
    
    该类提供了将分子图转换为token序列的核心功能，
    支持多种序列化策略和批处理操作。
    
    Attributes:
        vocab: 词汇表映射
        max_length: 最大序列长度
        padding_token: 填充token
        
    Methods:
        serialize: 序列化单个图
        batch_serialize: 批量序列化
        
    Note:
        该类不应直接实例化，应通过SerializerFactory创建。
    """
```

## 7. 测试规范

### 7.1 测试组织
```python
# tests/test_serializer.py
import pytest
from src.algorithms.serializer import GraphSerializer

class TestGraphSerializer:
    """测试GraphSerializer类的功能。"""
    
    @pytest.fixture
    def serializer(self):
        """创建测试用的序列化器实例。"""
        return GraphSerializer(vocab_size=1000)
    
    def test_single_graph_serialization(self, serializer):
        """测试单个图的序列化。"""
        graph = create_test_graph()
        result = serializer.serialize(graph)
        assert isinstance(result, list)
        assert len(result) > 0
```

### 7.2 测试原则
- **单元测试覆盖核心算法、数据**：所有关键算法、数据格式与接口必须有测试
- **使用真实数据测试**：在各个真实数据集上验证
- **确定性测试**：测试结果应该是确定的
- **边界条件测试**：测试极端和异常输入

### 7.3 测试策略  
由于项目经常进行大规模重构，必须确保：
- 重构不影响实验结果的可重现性
- 每次重构前后运行完整测试套件
- 保存关键测试的预期输出作为基准

## 8. 性能优化原则

### 8.1 并行处理
```python
# 并行处理需谨慎使用
from concurrent.futures import ProcessPoolExecutor
import time

def batch_process_with_check(items: List, process_fn, max_workers: int = 4):
    """带性能检查的批量并行处理。"""
    # 先测试串行性能
    start = time.time()
    serial_result = [process_fn(item) for item in items[:10]]
    serial_time = time.time() - start
    
    # 再测试并行性能
    start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        parallel_result = list(executor.map(process_fn, items[:10]))
    parallel_time = time.time() - start
    
    # 如果并行没有优势，使用串行
    if parallel_time > serial_time * 0.8:
        logger.warning("并行性能无优势，使用串行处理")
        return [process_fn(item) for item in items]
    
    # 使用并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(process_fn, items))
```

**注意事项**：
- 并行可能因Python GIL导致性能下降
- 必须预先编写性能测试脚本
- 检查是否出现死锁或串行化

### 8.2 缓存策略
```python
# 仅在确实需要时使用缓存
from functools import lru_cache

# 示例：仅对频繁调用且计算昂贵的函数使用
@lru_cache(maxsize=128)
def expensive_computation(param: str) -> Result:
    """缓存昂贵的计算结果。"""
    # 如：大规模矩阵运算、复杂图算法等
    pass
```

**使用原则**：
- 通过性能分析确认瓶颈
- 确保缓存的收益大于开销
- 避免过度优化

## 9. 版本控制

### 9.1 提交规范
```bash
# 提交信息格式
<type>: <subject>

# 类型
feat: 新功能
fix: 修复bug
refactor: 重构代码
docs: 文档更新
test: 测试相关
perf: 性能优化
style: 代码格式
```

### 9.2 分支策略
- `main`: 稳定版本
- `dev`: 开发分支
- `feature/*`: 功能分支（重构完成后启用）

**当前策略**：仅使用main和dev分支，简化开发流程

## 10. 代码审查清单

### 10.1 提交前检查
- [ ] 代码通过所有测试
- [ ] 添加了必要的文档
- [ ] 遵循命名约定
- [ ] 无硬编码值
- [ ] 无注释掉的代码
- [ ] 性能考虑（是否需要并行化）
- [ ] 错误处理完善

### 10.2 科研代码特殊检查
- [ ] 算法实现是确定性的
- [ ] 无隐式fallback逻辑
- [ ] 使用真实数据（非mock）
- [ ] 配置可追踪
- [ ] 结果可重现

---

*本文档是活文档，应根据项目发展持续更新。*
*最后更新：2025.8.8*
