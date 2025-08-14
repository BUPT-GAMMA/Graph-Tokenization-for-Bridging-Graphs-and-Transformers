# 项目架构文档 (Project Architecture)

## 1. 系统概述

### 1.1 项目目标
TokenizerGraph项目旨在将分子图结构转换为语言模型可处理的序列格式，通过预训练语言模型学习分子的"语法规则"，实现分子属性的准确预测。

### 1.2 核心创新
- **频率引导的欧拉路径序列化**：确保相同分子产生唯一确定的序列
- **BPE压缩**：自动发现并编码常见的分子子结构
- **统一接口设计**：标准化的数据加载和算法调用接口

### 1.3 技术栈
- **深度学习框架**：PyTorch
- **图处理**：NetworkX, DGL
- **数据处理**：NumPy, Pandas
- **分子处理**：RDKit
- **语言模型**：Transformers (BERT)

## 2. 系统架构

### 2.1 整体架构图
```
┌─────────────────────────────────────────────────────────────┐
│                        用户接口层                            │
│  (训练脚本、预处理脚本、评估脚本)                           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                        配置管理层                            │
│  (Config, ConfigBuilder, ConfigValidator)                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                        业务逻辑层                            │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   数据层      │  算法层       │   模型层      │    工具层       │
│  (Data)      │ (Algorithms) │  (Models)    │  (Utils)       │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                        存储层                               │
│  (文件系统、缓存、模型检查点)                               │
└─────────────────────────────────────────────────────────────┘
```

**调用方向**：数据层 → 序列化 → BPE（固定的依赖方向）

### 2.2 数据流架构
```
原始数据 (分子图/图像/图论数据)
    ↓ [数据加载器]
图结构 (Graph)
    ↓ [序列化算法]
原始序列 (Token Sequence)
    ↓ [BPE压缩]
压缩序列 (Compressed Tokens)
    ↓ [BERT编码器]
特征表示 (Feature Embedding)
    ↓ [任务头]
预测结果 (Prediction)
```

**注意**：
- 原始数据不限于分子图，还包括MNIST、CIFAR-10等图像数据以及图论数据集
- BPE输出是压缩后的token，不是embedding
## 3. 模块详细设计

### 3.1 数据层 (src/data/)

#### 设计理念
数据层不是传统的DataLoader，而是提供统一的数据访问接口，负责：
- 数据集加载和版本管理
- 数据划分和采样
- 数据集特定的token映射
- 缓存管理

#### 核心组件
```python
# 基础接口定义
class BaseLoader:
    """所有数据加载器的基类。"""
    def load_data(self) -> DataDict
    def split_data(self, ratios: Tuple) -> Tuple[DataDict, ...]
    def get_token_mapping(self) -> Dict[str, str]
    def get_statistics(self) -> DataStatistics

# 工厂模式创建
class DataFactory:
    """数据加载器工厂。"""
    @staticmethod
    def create_loader(dataset_name: str, **kwargs) -> BaseLoader

# 统一数据接口
class UnifiedDataInterface: !这个设计是很好的，不过应该仔细考虑设计其接口，并且注意区分版本，灵活利用预处理的缓存数据等!
    """提供统一的数据访问接口。"""
    def __init__(self, loader: BaseLoader, use_cache: bool = True)
    def get_graphs(self) -> List[Graph]
    def get_sequences(self, version: str = "latest") -> List[Sequence]  
    def get_compressed_sequences(self, bpe_model: str) -> List[CompressedSequence]
    # 设计要点：支持版本管理，灵活使用预处理缓存
```

#### 数据集支持
- **QM9**: 量子化学属性预测
- **ZINC**: 药物分子生成
- **AqSol**: 水溶性预测
- **MNIST**: 图像转图结构
- **CIFAR-10** (计划): 复杂图像数据
- **图论数据集** (计划): Cluster, Community Detection等

#### 未来扩展
- 支持多数据集联合训练
- 分布式数据处理

### 3.2 算法层 (src/algorithms/)

#### 3.2.1 序列化算法 (src/algorithms/serializer/)

##### 设计原则
- **确定性**：相同输入产生相同输出
- **可逆性**：序列可还原为原始图（部分算法）
- **效率性**：支持批量并行处理

##### 核心接口
```python
class BaseSerializer:
    """序列化器基类。"""
    
    @abstractmethod
    def serialize(self, graph: Graph) -> List[str]:
        """单图单次序列化。"""
        
    @abstractmethod
    def batch_serialize(self, graphs: List[Graph]) -> List[List[str]]:
        """批量序列化。"""
        
    @abstractmethod
    def multiple_serialize(self, graph: Graph, n: int) -> List[List[str]]:
        """单图多次序列化（用于数据增强）。"""

class SerializerFactory:
    """序列化器工厂。"""
    @staticmethod
    def create_serializer(method: str, **kwargs) -> BaseSerializer
```

##### 实现算法
- **GraphSeq**: 频率引导的欧拉路径
- **DFS/BFS**: 深度/广度优先遍历
- **TopologicalSort**: 拓扑排序
- **EulerianPath**: 欧拉回路
- **CPP/FreqCPP**: 中国邮递员问题及其变体
- **ImageScan** (计划): 图像数据的按行/列扫描
- **Canonical** (TODO): 规范化SMILES（保留现有load方式）

##### 优化方向
- 并行化处理优化
- 支持流式处理
- 增加图像视角的序列化

#### 3.2.2 压缩算法 (src/algorithms/compression/)

##### BPE算法实现
```python
class BPECompressor:
    """BPE压缩器。"""
    
    def train(self, sequences: List[List[str]], num_merges: int):
        """训练BPE模型。"""
        
    def encode(self, sequence: List[str]) -> List[str]:
        """编码序列。"""
        
    def decode(self, compressed: List[str]) -> List[str]:
        """解码序列。"""
        
    def save_vocab(self, path: Path):
        """保存词汇表。"""
```

##### 优化需求
- **当前任务**：使用NumPy向量化操作，参考minbpe实现，避免for循环
- **并行支持**：编码/解码和train过程中的merge并行化
- **注意**：所有效率优化必须先创建完善的测试脚本确保正确性

### 3.3 模型层 (src/models/)

#### 3.3.1 BERT模型 (src/models/bert/)
!不应该这么命名，并且bert模型整体上应该按照现在的实现，不要做太多改动（下面那个任务头的设计重构是可以的，但是bert本体的相关部分不要改了）!
```python
# 保持现有BERT实现，不做大的修改
class BERTModel(nn.Module): 
    """分子BERT模型。"""
    
    def __init__(self, config: BERTConfig):
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)
        
    def forward(self, input_ids, attention_mask=None):
        """前向传播。"""
        embeddings = self.embeddings(input_ids)
        encoded = self.encoder(embeddings, attention_mask)
        pooled = self.pooler(encoded)
        return encoded, pooled
```

#### 3.3.2 任务头
```python
class PropertyPredictionHead(nn.Module):
    """属性预测头。"""
    
    def __init__(self, hidden_size: int, num_targets: int):
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(hidden_size, num_targets)
```

#### 3.3.3 其他模型
- GNN模型：在其他项目中实现，不在本项目范围

### 3.4 工具层 (src/utils/)

#### 当前工具
- **Logger**: 统一日志管理
- **Metrics**: 评估指标计算  !这个需要完善!
- **Visualization**: 结果可视化 !删除，后面用到了再写!

#### 需要添加
- **Profiler**: 性能分析
- **Monitor**: 训练监控
- **Checkpoint**: 检查点管理

## 4. 接口设计规范

### 4.1 工厂模式
所有主要组件通过工厂创建，确保接口统一：
```python
# 数据加载
loader = DataFactory.create_loader("qm9")

# 序列化器
serializer = SerializerFactory.create_serializer("graph_seq")

# 模型
model = ModelFactory.create_model("bert", config)
```

### 4.2 统一返回格式
```python
@dataclass
class Result:
    """统一的结果格式。"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
```

### 4.3 错误处理
!注意，error直接打印抛出即可——此外，整个项目鼓励使用assert，随时检查状态是否符合预期，这个应该大量添加!
```python
# 简单直接的错误处理
def process_data(data):
    assert data is not None, "Data cannot be None"
    assert len(data) > 0, "Data cannot be empty"
    # 直接抛出异常，不做复杂封装
    if not validate_data(data):
        raise ValueError(f"Invalid data format: {data}")
```

**原则**：
- 大量使用assert检查状态
- 直接抛出标准异常
- 不做过度封装

## 5. 并发设计

**重要**：所有并发优化必须先创建完善的性能测试对比脚本！

### 5.1 数据处理并发
```python
from concurrent.futures import ProcessPoolExecutor

class ParallelProcessor:
    """并行处理器。"""
    
    def process_batch(self, items: List, process_fn, max_workers: int = 4):
        """批量并行处理。"""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_fn, item) for item in items]
            results = [f.result() for f in futures]
        return results
```

### 5.2 模型训练并发
- 数据并行 (DataParallel)
- 分布式训练 (DistributedDataParallel)
- 混合精度训练 (AMP)

### 5.3 实验并发
- 多GPU多实验并行
- 超参数搜索并行
- 交叉验证并行

## 6. 缓存策略

### 6.1 简单缓存策略
- **文件缓存**：主要依赖文件系统缓存
- **内存缓存**：仅对确认瓶颈的函数使用LRU
- **原则**：保持简单，避免过度设计

### 6.2 缓存使用示例
```python
class CacheManager:
    """缓存管理器。"""
    
    def __init__(self, cache_dir: Path, max_size: int = 1000):
        self.cache_dir = cache_dir
        self.memory_cache = LRUCache(max_size)
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存。"""
        # 先查内存
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # 再查磁盘
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            data = pickle.load(open(cache_file, 'rb'))
            self.memory_cache[key] = data
            return data
            
        return None
```

## 7. 模块化设计

### 7.1 配置驱动
通过配置文件控制系统行为：
- 算法选择
- 模型架构
- 训练策略

### 7.2 模块原则
- 清晰的接口定义
- 最小化依赖
- 单一职责

## 8. 性能考虑

### 8.1 瓶颈分析
当前瓶颈：
1. **序列化速度**：Python原生实现较慢
2. **BPE编码**：大词表时性能下降
3. **数据加载**：大数据集I/O密集

### 8.2 优化策略
- **算法优化**：使用更高效的数据结构
- **并行化**：充分利用多核CPU
- **缓存**：避免重复计算
- **向量化**：使用NumPy/PyTorch操作
- **JIT编译**：使用TorchScript
- **混合语言**：关键部分用C++/Rust

### 8.3 性能监控
```python
class PerformanceMonitor:
    """性能监控器。"""
    
    def profile_function(self, func, *args, **kwargs):
        """分析函数性能。"""
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.print_stats()
        return result
```

## 9. 测试架构

### 9.1 测试层级
```
单元测试 (Unit Tests)
    ↓
集成测试 (Integration Tests)
    ↓
端到端测试 (E2E Tests)
    ↓
性能测试 (Performance Tests)
```
!性能测试的粒度应该更细：比如我要优化bpe，那么应该对bpe有一个端到端的正确性和性能测试。加载真是的序列化数据，然后测试不同版本的写法性能之类。同时保证输出结果不变!
**性能测试粒度**：
- **模块级测试**：BPE、序列化等单独模块的性能测试
- **端到端测试**：加载真实数据，测试不同版本性能
- **正确性验证**：确保优化后输出结果不变
### 9.2 测试覆盖
- **数据层**：
  - 数据加载、划分、转换
  - 格式验证：图和label范围检查
  - 回归测试：与基准结果对比
- **算法层**：
  - 序列化正确性：重构图一致性、点边数验证
  - BPE编解码：可逆性测试
  - 路径正确性：检查是否有无效边或路径
- **模型层**：前向传播、梯度流
- **端到端**：完整训练流程

### 9.3 测试数据策略
- **BPE测试**：使用真实序列化数据缓存
- **模型测试**：加载真实数据集序列
- **单元测试**：可使用小型合成数据
- **端到端测试**：必须使用真实数据

## 10. 维护和演进

### 10.1 版本策略
**注意**：进行接口重构时，应让整个项目同步使用新接口，避免添加过多兼容处理逻辑。

### 10.2 迁移指南
为重大变更提供：
- 迁移文档
- 详细的变更日志

---
*本文档描述了TokenizerGraph项目的整体架构设计，是理解和开发项目的重要参考。*
*最后更新：2025.8.8*
