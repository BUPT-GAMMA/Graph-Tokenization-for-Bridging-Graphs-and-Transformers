# 配置管理指南 (Configuration Management Guide)

## 1. 配置管理核心原则

### 1.1 单一真相源 (Single Source of Truth)
- **中央配置文件**：所有配置集中在 `config.py`
- **禁止分散定义**：不在各个模块中独立定义配置
- **显式依赖**：所有模块显式导入配置
- **无隐式默认值**：避免在函数内部定义默认参数

### 1.2 配置层次结构
```
config.py (全局配置)
    ↓
命令行参数 (特定任务覆盖)
    ↓
实验配置 (实验特定参数)
```

### 1.3 配置不变性原则
- **运行时不可变**：配置一旦加载，运行期间不应修改（并行运行时为每个任务创建独立配置）
- **版本追踪**：每次实验保存完整配置快照
- **向后兼容**：新增配置项提供合理默认值

## 2. 配置文件结构

### 2.1 标准配置组织
```python
# config.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

@dataclass
class DataConfig:
    """数据相关配置。"""
    # 数据路径
    data_dir: Path = Path("data/raw")
    cache_dir: Path = Path("cache")
    processed_data_dir: Path = Path("data/processed")
    
    # 数据集设置
    dataset_name: str = "qm9"
    dataset_version: str = "1.0.0"
    # subgraph_limit: Optional[int] = None  # 已弃用，使用数据集自身的划分
    
    # 数据处理
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    
    # 缓存策略
    use_cache: bool = True
    cache_format: str = "pickle"  # "pickle", "parquet", "feather"
    
    def __post_init__(self):
        """验证和转换路径。"""
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        
        # 验证比例
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6
        

@dataclass
class ModelConfig:
    """模型相关配置。"""
    # 模型类型
    model_type: str = "bert"  # "bert", "gnn", "hybrid"
    
    # BERT配置
    hidden_size: int = 512
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # 序列设置
    max_position_embeddings: int = 512
    max_seq_length: int = 64
    vocab_size: int = 30000
    
    # 特殊token
    pad_token: str = "[PAD]"
    mask_token: str = "[MASK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    unk_token: str = "[UNK]"
    
    def validate(self):
        """验证模型配置的合理性。"""
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        assert self.max_seq_length <= self.max_position_embeddings, \
            f"max_seq_length ({self.max_seq_length}) cannot exceed max_position_embeddings ({self.max_position_embeddings})"


@dataclass
class TrainingConfig:
    """训练相关配置。"""
    # 基本训练参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    gradient_accumulation_steps: int = 1
    
    # 优化器设置
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # "linear", "cosine", "polynomial"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    
    # 训练策略
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    mixed_precision: bool = False
    
    # 早停策略
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # 检查点
    save_checkpoint_steps: int = 1000
    save_total_limit: int = 3
    checkpoint_dir: Path = Path("model/checkpoints")
    
    # 日志
    logging_steps: int = 100
    eval_steps: int = 500
    eval_strategy: str = "epoch"  # "steps", "epoch"


@dataclass
class SerializationConfig:
    """序列化算法配置。"""
    # 序列化方法
    serialization_method: str = "graph_seq"  # "graph_seq", "dfs", "bfs", "canonical"
    
    # 频率统计
    use_global_frequency: bool = True
    frequency_cache_file: Optional[Path] = None  # 动态生成: cache/{dataset}_{version}_freq.pkl
    
    def get_frequency_cache_path(self, dataset_name: str, dataset_version: str) -> Path:
        """根据数据集生成唯一的频率缓存路径。"""
        import hashlib
        hash_str = hashlib.md5(f"{dataset_name}_{dataset_version}".encode()).hexdigest()[:8]
        return Path(f"cache/{dataset_name}_{hash_str}_freq.pkl")
    
    # BPE配置
    bpe_num_merges: int = 2000
    bpe_min_frequency: int = 100  # QM9使用100，小数据集使用10  
    bpe_cache_dir: Path = Path("cache/bpe")  # 仅保存BPE模型，序列压缩结果由预处理脚本生成
    
    # 并行处理
    num_workers: int = 4
    batch_size: int = 1000


@dataclass
class ExperimentConfig:
    """实验管理配置。"""
    # 实验标识
    experiment_name: str = "default_experiment"  # 通过命令行 --name 轻松覆盖
    experiment_group: str = "default_group"  # 实验组，用于组织相关实验
    # 在没有特殊说明的情况下，一段时间内运行的多组实验都属于一个group
    # 一个group用于验证某个猜想或对比不同方法
    experiment_id: Optional[str] = None  # 自动生成
    tags: List[str] = field(default_factory=list)
    
    # 随机种子
    seed: int = 42
    deterministic: bool = True
    
    # 设备设置
    device: str = "cuda"  # "cuda", "cpu", "mps"
    cuda_visible_devices: Optional[str] = None
    num_gpus: int = 1
    
    # 输出路径
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    tensorboard_dir: Path = Path("logs/tensorboard")
    
    # 日志设置
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    use_tensorboard: bool = True  # 当前使用tensorboard，计划迁移到wandb
    save_local_metrics: bool = True  # 同时保存本地文件便于后续分析
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    
    # 调试模式
    debug_mode: bool = False
    profile: bool = False
    
    def setup_experiment(self, training: Optional[TrainingConfig] = None):
        """设置实验环境。"""
        import random
        import numpy as np
        import torch
        
        # 设置随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # 设置确定性行为
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成实验ID（包含组名和日期）。
        # 如未指定实验名，建议使用训练配置的关键参数（batch_size / learning_rate）作为标识。
        # 注意：为避免在此处引用外层配置对象，推荐在调用处传入 training 配置以生成名称。
        if self.experiment_id is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # 默认直接使用 experiment_name，调用处可在默认名时覆盖为基于训练配置的名字
            exp_name = self.experiment_name
            self.experiment_id = f"{self.experiment_group}/{exp_name}_{timestamp}"


# 实用函数：基于训练配置生成默认实验名（示例）
def default_experiment_name_from_training(training: TrainingConfig) -> str:
    """根据训练配置生成可读的默认实验名。"""
    # 使用常见关键超参数，保持简洁且可区分
    return f"bs{training.batch_size}_lr{training.learning_rate:g}"


@dataclass
class Config:
    """主配置类，整合所有子配置。"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置。"""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            serialization=SerializationConfig(**config_dict.get('serialization', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def to_dict(self) -> Dict:
        """转换为字典。"""
        from dataclasses import asdict
        return asdict(self)
    
    def save(self, path: Path):
        """保存配置到文件。"""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        # 转换Path对象为字符串
        config_dict = self._paths_to_str(config_dict)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """从文件加载配置。"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def _paths_to_str(self, obj):
        """递归转换Path对象为字符串。"""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._paths_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._paths_to_str(item) for item in obj]
        return obj
    
    def validate(self):
        """验证所有配置。"""
        self.model.validate()
        # 可以添加更多验证逻辑
```

## 3. 配置使用模式（与测试对齐）

### 3.1 基本使用
```python
# 导入配置
from config import Config

# 创建默认配置
config = Config()

# 访问配置项
batch_size = config.training.batch_size
data_dir = config.data.data_dir

# 设置实验环境
config.experiment.setup_experiment()
```

### 3.2 命令行覆盖
```python
import argparse
from config import Config

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    # 建议：每个实验加载同一个项目默认配置，通过命令行做最小覆盖；
    # 实际使用的配置应保存在实验记录文件夹中。
    
    # 覆盖特定参数（保持参数名一致性）
    parser.add_argument('--bs', type=int, default=None,
                       help='batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='number of epochs')
    parser.add_argument('--seed', type=int, default=None,
                       help='random seed')
    parser.add_argument('--name', type=str, default=None,
                       help='experiment name')
    parser.add_argument('--group', type=str, default=None,
                       help='experiment group')
    
    return parser.parse_args()

def create_config(args):
    """根据命令行参数创建配置。"""
    # 总是从项目默认配置创建（示例）
    config = Config()
    
    # 应用命令行覆盖（最小化覆盖）
    if args.bs is not None:
        config.training.batch_size = args.bs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.name is not None:
        config.experiment.experiment_name = args.name
    if args.group is not None:
        config.experiment.experiment_group = args.group
    
    # 验证配置
    config.validate()
    
    return config
```

### 3.3 实验配置管理（路径/命名与测试一致）
```python
def run_experiment(config: Config):
    """运行实验并保存配置。"""
    # 若未指定实验名，用训练配置生成一个默认可读实验名
    if config.experiment.experiment_name == "default_experiment":
        config.experiment.experiment_name = default_experiment_name_from_training(config.training)
    # 设置实验
    config.experiment.setup_experiment(training=config.training)
    
    # 保存配置快照（注意：作为文件名使用时将'/'替换为'_'）
    config_path = config.experiment.output_dir / f"{config.experiment.experiment_id.replace('/', '_')}_config.json"
    config.save(config_path)
    
    # 记录配置
    logger.info(f"Experiment ID: {config.experiment.experiment_id}")
    logger.info(f"Configuration saved to: {config_path}")
    logger.info(f"Configuration:\n{json.dumps(config.to_dict(), indent=2)}")
    
    # 执行实验...
```

## 4. 路径管理（与 ProjectConfig 路径 API 对齐）

### 4.1 路径配置原则
```python
# 使用Path对象而非字符串
from pathlib import Path

# 正确的路径配置
data_dir: Path = Path("data")
cache_dir: Path = Path("cache")  # 相对路径或软链接

# 动态构建路径
def get_dataset_path(config: Config, dataset_name: str) -> Path:
    """构建数据集路径。"""
    return config.data.data_dir / dataset_name

def get_model_path(config: Config, model_name: str) -> Path:
    """构建模型路径。"""
    return config.training.checkpoint_dir / model_name
```

### 4.2 路径管理简化
```python
# 使用软链接统一管理路径
# 在项目根目录创建软链接：
# ln -s /local/data ./local_data
# ln -s /local/cache ./local_cache

# 然后在配置中直接使用：
cache_dir: Path = Path("local_cache")  # 通过软链接指向本地存储
data_dir: Path = Path("data")  # 一般数据
```

### 4.3 路径最佳实践
- 使用相对路径便于项目迁移
- 大文件使用软链接指向本地存储
- 缓存路径应该可配置且易于清理
- 根据项目实际情况调整路径配置
## 5. 环境变量管理

### 5.1 环境变量使用原则
```python
import os
from pathlib import Path

# 环境变量仅用于系统级配置
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
TOKENIZER_GRAPH_HOME = os.getenv("TOKENIZER_GRAPH_HOME", str(Path.home() / "tokenizerGraph"))

# 不使用环境变量存储实验参数
# 错误示例：
# batch_size = int(os.getenv("BATCH_SIZE", "32"))  # 避免这种做法
```

### 5.2 环境配置文件
```bash
# .env.example
# 系统级配置
CUDA_VISIBLE_DEVICES=0,1
TOKENIZER_GRAPH_HOME=/home/user/tokenizerGraph

# 路径配置（可选）
LOCAL_DATA_PATH=/local/data
LOCAL_CACHE_PATH=/local/cache
```

## 6. 配置验证

### 6.1 启动时验证
```python
def validate_config(config: Config):
    """全面验证配置。"""
    errors = []
    
    # 检查路径存在性
    if not config.data.data_dir.exists():
        errors.append(f"Data directory not found: {config.data.data_dir}")
    
    # 检查参数范围
    if config.training.learning_rate <= 0:
        errors.append(f"Invalid learning rate: {config.training.learning_rate}")
    
    if config.training.batch_size <= 0:
        errors.append(f"Invalid batch size: {config.training.batch_size}")
    
    # 检查设备可用性
    if config.experiment.device == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA requested but not available")
    
    # 检查依赖关系
    if config.training.warmup_ratio > 0 and config.training.warmup_steps > 0:
        errors.append("Cannot specify both warmup_ratio and warmup_steps")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
```

### 6.2 运行时检查
```python
class ConfigChecker:
    """运行时配置检查器。"""
    
    def __init__(self, config: Config):
        self.config = config
        self.initial_config = config.to_dict()
    
    def check_unchanged(self):
        """检查配置是否被意外修改。"""
        current_config = self.config.to_dict()
        if current_config != self.initial_config:
            raise RuntimeError("Configuration was modified during runtime!")
    
    def check_compatibility(self, checkpoint_config: Dict):
        """检查与检查点的兼容性。"""
        incompatible = []
        
        # 检查关键参数
        if checkpoint_config['model']['hidden_size'] != self.config.model.hidden_size:
            incompatible.append('model.hidden_size')
        
        if incompatible:
            raise ValueError(f"Incompatible configuration items: {incompatible}")
```

## 7. 多配置管理

### 7.1 配置组合
```python
# configs/presets.py
"""预定义配置组合。"""

def get_debug_config() -> Config:
    """调试配置（小批量、少轮次）。"""
    config = Config()
    config.training.batch_size = 4
    config.training.max_epochs = 2
    config.experiment.debug_mode = True
    # config.data.subgraph_limit = 100  # 已弃用
    return config

def get_test_config() -> Config:
    """测试配置（中等规模）。"""
    config = Config()
    config.training.max_epochs = 10
    config.training.batch_size = 16
    return config

def get_production_config() -> Config:
    """生产配置（完整训练）。"""
    config = Config()
    # config.data.subgraph_limit = None  # 已弃用，数据集不使用此参数
    config.training.max_epochs = 100
    config.training.mixed_precision = True
    return config
```

### 7.2 配置继承
```python
class ConfigBuilder:
    """配置构建器，支持继承和组合。"""
    
    @staticmethod
    def build(base: str = "default", overrides: Dict = None) -> Config:
        """构建配置。"""
        # 加载基础配置
        if base == "debug":
            config = get_debug_config()
        elif base == "test":
            config = get_test_config()
        elif base == "production":
            config = get_production_config()
        else:
            config = Config()
        
        # 应用覆盖
        if overrides:
            ConfigBuilder._apply_overrides(config, overrides)
        
        return config
    
    @staticmethod
    def _apply_overrides(config: Config, overrides: Dict):
        """递归应用配置覆盖。"""
        for key, value in overrides.items():
            if '.' in key:
                # 处理嵌套键如 "training.batch_size"
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(config, key, value)
```

## 8. 配置最佳实践

### 8.1 DO - 推荐做法
✅ **集中管理**：所有配置在config.py中定义
✅ **类型注解**：使用dataclass和类型提示
✅ **验证检查**：启动时验证配置合理性
✅ **版本追踪**：保存每次实验的配置快照
✅ **清晰命名**：配置项名称自解释
✅ **分组组织**：相关配置分组到子类
✅ **默认值合理**：提供经过验证的默认值

### 8.2 DON'T - 避免做法
❌ **分散定义**：在各模块中定义配置
❌ **硬编码**：在代码中硬编码参数
❌ **动态修改**：运行时修改配置
❌ **隐式默认**：在函数参数中定义默认值
❌ **环境依赖**：过度依赖环境变量
❌ **过度嵌套**：配置层次过深
❌ **魔法数字**：使用无解释的数值

## 9. 配置迁移指南

### 9.1 从旧配置迁移
```python
def migrate_old_config(old_config_path: str) -> Config:
    """从旧配置格式迁移。"""
    import json
    
    with open(old_config_path, 'r') as f:
        old_config = json.load(f)
    
    # 创建新配置
    new_config = Config()
    
    # 映射旧字段到新字段
    if 'batch_size' in old_config:
        new_config.training.batch_size = old_config['batch_size']
    if 'lr' in old_config:
        new_config.training.learning_rate = old_config['lr']
    # ... 更多映射
    
    return new_config
```

### 9.2 配置版本管理
```python
@dataclass
class VersionedConfig(Config):
    """带版本的配置。"""
    config_version: str = "2.0.0"
    
    def is_compatible(self, required_version: str) -> bool:
        """检查版本兼容性。"""
        from packaging import version
        return version.parse(self.config_version) >= version.parse(required_version)
```

## 10. 故障排查

### 10.1 常见问题
1. **配置未生效**：检查是否正确导入和使用配置对象
2. **路径错误**：使用绝对路径或确保相对路径基准正确
3. **类型错误**：确保配置项类型与预期一致
4. **覆盖失败**：检查命令行参数名称是否正确

### 10.2 调试工具
```python
def debug_config(config: Config):
    """调试配置问题。"""
    print("=== Configuration Debug Info ===")
    print(f"Config object ID: {id(config)}")
    print(f"Data dir exists: {config.data.data_dir.exists()}")
    print(f"Device available: {torch.cuda.is_available() if config.experiment.device == 'cuda' else True}")
    print(f"Config validation: ", end="")
    try:
        config.validate()
        print("PASSED")
    except Exception as e:
        print(f"FAILED - {e}")
    print("\n=== Full Configuration ===")
    print(json.dumps(config.to_dict(), indent=2))
```

---

*本文档定义了项目的配置管理标准，所有配置相关的代码都应遵循这些规范。*
*最后更新：2025.8.8*
