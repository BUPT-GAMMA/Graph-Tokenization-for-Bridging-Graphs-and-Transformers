# 实验指南 (Experiment Guidelines)

## 1. 实验设计原则

### 1.1 科学性原则
- **假设明确**：每个实验必须有明确的假设和预期结果，或为明确的对比实验
- **变量控制**：一次只改变一个变量，其他保持不变
- **对照实验**：设置合理的baseline和对照组
- **统计显著性**：多次运行取平均，报告标准差

### 1.2 可重现性原则
- **完全确定性**：相同输入必须产生相同输出
- **环境固定**：记录并固定所有依赖版本
- **种子控制**：所有随机数生成器使用固定种子
- **数据版本化**：数据集必须有明确的版本标识

### 1.3 透明性原则
- **无隐藏处理**：所有数据处理步骤必须记录
- **失败即停止**：不使用fallback机制掩盖问题
- **原始结果优先**：先记录原始数据，后进行分析
- **完整记录**：保存所有中间结果和日志

## 2. 实验流程标准

### 2.1 实验准备阶段示例
```python
# 1. 配置验证
def validate_experiment_config(config):
    """验证实验配置的完整性和合理性。"""
    assert config.seed is not None, "必须设置随机种子"
    assert config.data_version is not None, "必须指定数据版本"
    assert config.model_checkpoint_dir.exists(), "检查点目录必须存在"
    
# 2. 环境设置
def setup_experiment_environment(config):
    """设置实验环境，确保可重现性。"""
    # 固定所有随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    #设定日志记录——注意，都要能够快速确认到记录信息是属于哪一组实验
    logger.save……
    #或者
    tensorboard writter init……
    #或者
    wandb init……

    # 设置确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# 3. 数据准备
def prepare_data(config):
    """准备数据，记录数据统计信息。"""
    # 使用统一的数据加载接口
    loader = DataFactory.create_loader(config.dataset_name)
    
    # 记录数据集信息
    log_dataset_statistics(loader)
    
    # 验证数据完整性
    validate_data_integrity(loader)
```

### 2.2 实验执行阶段示例
```python
# 实验主循环示例
def run_experiment(config):
    """执行实验的标准流程。"""
    # 1. 初始化
    setup_experiment_environment(config)
    experiment_id = generate_experiment_id()
    logger = setup_logger(experiment_id)
    
    # 2. 记录配置
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # 3. 数据加载
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # 4. 模型初始化
    model = create_model(config)
    logger.info(f"Model parameters: {count_parameters(model)}")
    
    # 5. 训练循环
    results = train_model(model, train_loader, val_loader, config)
    
    # 6. 评估
    test_results = evaluate_model(model, test_loader)
    
    # 7. 保存结果
    save_experiment_results(experiment_id, config, results, test_results)
    
    return experiment_id
```

### 2.3 结果记录标准
```python
@dataclass
class ExperimentResult:
    """标准实验结果格式。"""
    # 实验元数据
    experiment_id: str
    timestamp: datetime
    config: Dict
    
    # 训练结果
    train_loss: List[float]
    val_loss: List[float]
    train_metrics: Dict[str, List[float]]
    val_metrics: Dict[str, List[float]]
    
    # 测试结果
    test_loss: float
    test_metrics: Dict[str, float]
    
    # 统计信息
    best_epoch: int
    total_time: float
    peak_memory: float
    
    # 可重现性信息
    random_seed: int
    git_commit: str
    environment_info: Dict
```

## 3. 数据管理规范

### 3.1 数据版本控制
```yaml
# data_version.yaml
dataset_name: qm9
version: 1.0.0
source_url: "https://..."
download_date: 2024-01-01
md5_checksum: "abc123..."
preprocessing_version: 2.0.0
split_seed: 42
statistics:
  total_samples: 133885
  train_samples: 110000
  val_samples: 10000
  test_samples: 13885
```

### 3.2 数据处理流水线
```python
# 原则：
# 1) 数据处理必须是确定性的（同样输入得到同样输出）
# 2) 建议采用“预处理脚本”离线固定数据集划分（保存train/val/test索引），
#    下游 loader 仅按固定索引加载与拼接，不在内部进行随机划分。
# 示例（伪代码）：
#   splits = load_json('splits/qm9_v1_seed42.json')  # 包含train/val/test的样本ID
#   train_ids, val_ids, test_ids = splits['train'], splits['val'], splits['test']
#
#   # loader 侧：
#   train_set = dataset.select(train_ids)
#   val_set = dataset.select(val_ids)
#   test_set = dataset.select(test_ids)
#   # 不再在此处做随机划分
class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.preprocessors = []
        
    def add_preprocessor(self, preprocessor):
        """添加预处理步骤。"""
        self.preprocessors.append(preprocessor)
        
    def process(self, raw_data):
        """执行数据处理流水线。"""
        data = raw_data
        for i, preprocessor in enumerate(self.preprocessors):
            logger.info(f"Step {i}: {preprocessor.__class__.__name__}")
            data = preprocessor.transform(data)
            
            # 验证每步输出
            self._validate_step_output(data, i)
            
        return data
        
    def _validate_step_output(self, data, step_idx):
        """验证处理步骤的输出。"""
        assert data is not None, f"Step {step_idx} produced None"
        assert len(data) > 0, f"Step {step_idx} produced empty data"
```

### 3.3 数据集划分原则
- **固定划分**：使用固定的种子和比例
- **分层采样**：保持类别分布一致（如适用）
- **时间划分**：时序数据按时间顺序划分
- **无数据泄露**：严格隔离训练/验证/测试集

## 4. 模型训练规范

### 4.1 训练监控
```python
class TrainingMonitor:
    """训练过程监控器。"""
    
    def __init__(self, config):
        self.config = config
        self.metrics = defaultdict(list)
        
    def on_epoch_end(self, epoch, train_loss, val_loss, metrics):
        """每个epoch结束时的记录。"""
        # 记录损失
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        
        # 记录其他指标
        for key, value in metrics.items():
            self.metrics[key].append(value)
            
        # 检查异常情况
        self._check_anomalies(epoch)
        
        # 保存检查点
        if self._should_save_checkpoint(epoch):
            self.save_checkpoint(epoch)
            
    def _check_anomalies(self, epoch):
        """检查训练异常。"""
        # 检查NaN
        if np.isnan(self.metrics['train_loss'][-1]):
            raise ValueError(f"NaN loss at epoch {epoch}")
            
        # 检查梯度爆炸
        if self.metrics['train_loss'][-1] > 1e10:
            raise ValueError(f"Loss explosion at epoch {epoch}")
            
        # 检查过拟合
        if epoch > 10:
            recent_val = self.metrics['val_loss'][-10:]
            if all(recent_val[i] > recent_val[i-1] for i in range(1, 10)):
                logger.warning(f"Validation loss increasing for 10 epochs")
```

### 4.2 超参数管理
```python
# 超参数配置示例
@dataclass
class HyperParameters:
    """实验超参数定义。"""
    # 模型架构
    model_type: str = "bert"
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    
    # 训练参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    
    # 正则化
    dropout: float = 0.1
    gradient_clip: float = 1.0
    
    # 早停策略
    patience: int = 10
    min_delta: float = 1e-4
    
    def validate(self):
        """验证超参数的合理性。"""
        assert self.hidden_size % self.num_heads == 0
        assert self.learning_rate > 0
        assert 0 <= self.dropout < 1
```

## 5. 评估标准

### 5.1 评估指标定义
```python
class MetricsCalculator:
    """标准评估指标计算器。"""
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """回归任务指标。"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'pearson': pearsonr(y_true, y_pred)[0],
            'spearman': spearmanr(y_true, y_pred)[0]
        }
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_prob=None):
        """分类任务指标。"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            # 如果是二分类
            'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None else None,
            'ap': average_precision_score(y_true, y_prob) if y_prob is not None else None
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            
        return metrics
```

### 5.2 统计显著性测试
```python
def statistical_significance_test(results_a, results_b, test_type='t-test'):
    """统计显著性测试。"""
    if test_type == 't-test':
        statistic, p_value = ttest_ind(results_a, results_b)
    elif test_type == 'wilcoxon':
        statistic, p_value = wilcoxon(results_a, results_b)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
        
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

## 6. 实验比较规范

### 6.1 Baseline设置
```python
# 参考baseline（根据实验需要选择）
SUGGESTED_BASELINES = {
    'random': "随机预测基准",
    'majority': "多数类基准（分类）",
    'mean': "均值预测基准（回归）",
    'previous_best': "之前的最佳方法",
}

def setup_baselines(task_type, dataset):
    """设置实验baseline。"""
    baselines = {}
    
    if task_type == 'regression':
        baselines['mean'] = MeanPredictor()
        baselines['random'] = RandomRegressor()
    elif task_type == 'classification':
        baselines['majority'] = MajorityClassifier()
        baselines['random'] = RandomClassifier()
        
    return baselines
```

**灵活使用Baseline**：
- 内部对比实验可不设置baseline
- 根据实验目的选择合适的对照组
- 可以使用前后实验结果作为对比

### 6.2 公平比较原则
- **相同数据划分**：所有方法使用完全相同的训练/验证/测试集
- **相同预处理**：数据预处理步骤必须一致
- **相同评估协议**：使用相同的评估指标和方法
- **多次运行**：每个方法至少运行5次，报告均值和标准差

## 7. 结果报告格式

### 7.1 实验报告模板 !同上，应该更灵活一些，不必每次实验都这样。不过，一组实验结果能够导出一个表格文件的话还是有必要的!
```markdown
# 实验报告

## 实验信息
- **实验ID**: exp_20240101_001
- **日期**: 2024-01-01
- **目的**: 验证序列化算法对预测性能的影响
- **假设**: 频率引导的序列化能提升模型性能

## 配置
- **数据集**: QM9 v1.0
- **模型**: BERT-Small
- **超参数**: [详细列表]

## 结果

### 主要指标
| Method | MAE | RMSE | R² | Time(s) |
|--------|-----|------|----|---------|
| Baseline | 0.50±0.02 | 0.65±0.03 | 0.75±0.01 | 100 |
| Our Method | 0.35±0.01 | 0.45±0.02 | 0.85±0.01 | 120 |

### 统计检验
- t-test p-value: 0.001
- 结论: 统计显著提升

## 分析
[详细分析]

## 结论
[实验结论]
```

**灵活记录原则**：
- 不必每次实验都使用完整模板
- 重点是能导出结果表格便于对比
- 根据实验类型选择记录内容

### 7.2 可视化标准
```python
def plot_training_curves(metrics, save_path):
    """绘制标准训练曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 损失曲线
    axes[0, 0].plot(metrics['train_loss'], label='Train')
    axes[0, 0].plot(metrics['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 其他指标...
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

**注意**：优先使用TensorBoard（当前）或WandB（计划迁移）进行可视化，避免重复实现

## 8. 实验记录工具

### 8.1 使用TensorBoard/WandB
```python
# 使用TensorBoard记录
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs/{experiment_name}')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)

# 未来迁移到WandB
import wandb
wandb.init(project="tokenizerGraph", name=experiment_name)
wandb.log({"loss": loss, "val_loss": val_loss})
```

### 8.2 本地结果保存
```python
def save_metrics_to_file(metrics, save_path):
    """保存指标到本地文件便于后续分析。"""
    import json
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
```
!这个也没有必要!
## 9. 常见陷阱与最佳实践

### 9.1 需要避免的错误
- ❌ **数据泄露**：在预处理时使用测试集信息
- ❌ **选择性报告**：只报告最好的结果
- ❌ **过度调参**：在测试集上调参
- ❌ **忽视失败**：使用fallback隐藏问题
- ❌ **不固定种子**：导致结果不可重现

### 9.2 最佳实践
- ✅ **预注册实验**：实验前明确假设和评估方法
- ✅ **完整记录**：保存所有日志和中间结果
- ✅ **版本锁定**：固定所有依赖版本
- ✅ **代码审查**：实验代码需要审查
- ✅ **结果验证**：独立验证关键结果

## 10. 检查清单

### 10.1 实验前检查
- [ ] 明确定义实验假设
- [ ] 设置合适的baseline
- [ ] 准备数据版本
- [ ] 配置随机种子
- [ ] 创建实验分支
- [ ] 验证评估协议

### 10.2 实验中检查
- [ ] 监控训练过程
- [ ] 检查异常情况
- [ ] 定期保存检查点
- [ ] 记录所有配置
- [ ] 保存中间结果

### 10.3 实验后检查
- [ ] 运行统计检验
- [ ] 生成可视化
- [ ] 撰写实验报告
- [ ] 代码和数据归档
- [ ] 标记git版本
- [ ] 更新实验记录

---

*本指南旨在确保科研实验的科学性、可重现性和透明性。*
*最后更新：2025.8.8*
