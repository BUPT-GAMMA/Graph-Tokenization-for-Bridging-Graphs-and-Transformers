# BPE 引擎（训练 + 编码）使用说明与性能摘要

本目录提供统一的 BPE 引擎 `BPEEngine`，涵盖：
- 训练后端：C++ 原生（minBPE 逻辑）、Python（标准优化版）
- 编码后端：C++/pybind11（推荐）、Python（兼容）

## 一、快速开始

```python
from src.algorithms.compression.bpe_engine import BPEEngine

# 训练（推荐：已构建 C++ 扩展时使用 cpp；否则用 numba）
engine = BPEEngine(train_backend='cpp', encode_backend='cpp')
stats = engine.train(token_sequences, num_merges=2000, min_frequency=10)

# 编码（推荐：cpp；也可 'python'）
engine.build_encoder()
ids = engine.encode(seq)
ids_list = engine.batch_encode(seqs)
```

训练产出 `merge_rules`，编码语义为 minBPE 风格非重叠合并，与参考 minBPE 逻辑等价（已在对拍脚本中验证）。

## 二、后端选择
- 训练：
  - `cpp`：C++ 原生 minBPE 训练（每轮重统计 + 确定性 tie-break + 非重叠合并），速度最快。
  - `python`：标准 BPE 优化实现（增量频率表），语义与 minBPE 不完全相同，仅作兼容/对照。
- 编码：
  - `cpp`：C++/pybind11，推荐。
  - `python`：兼容路径。

如使用 C++ 后端，需要先构建扩展：
```
pip install pybind11
python setup.py build_ext --inplace
```

## 三、脚本与对拍

- 全面对拍（训练规则一致 + 全量编码一致）：
```bash
python comprehensive_bpe_verification.py \
  --dataset qm9test \
  --num-merges 2000 --min-frequency 100 \
  --encoding-sample 1000000000 \
  --our-train-backend cpp --our-encode-backend cpp
```

- 训练性能对比：
```bash
python benchmark_bpe_new_vs_old.py \
  --dataset qm9test \
  --num-merges 2000 --min-frequency 100 \
  --baseline minbpe \
  --engine-backend cpp
```

示例（qm9test，2000 merges，min_freq=100，单线程）：
- 训练：minBPE 参考 ~13.1s；C++ 训练 ~0.33s（环境相关，供参考）
- 编码（cpp）：批量高吞吐，单线程可达 1e5+ seq/s（视硬件而定）

## 四、语义与兼容性
- C++ 训练后端与 minBPE 参考实现语义等价：训练规则顺序与编码结果对拍一致。
- Python `StandardBPECompressor` 为“标准 BPE 优化实现”，采用维护频率表的增量策略，语义与 minBPE 可能不同，仅用于对照/兼容。
- 上层接口统一：无论选择何种后端，`train` → `build_encoder` → `encode/batch_encode` 的调用方式完全一致。

## 五、注意事项
- 线程：默认 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，避免嵌套并行影响尾延迟。
- 大数据集：如需更高上限，建议优先使用 C++ 训练；Numba 训练也可在核函数中进一步并行化。



## 六、rank-limit（top-k）与采样策略

- 目标：控制 BPE 编码压缩深度，支持“仅使用前 k 条规则”，以及基于分布的 top-k 采样。
- 语义：运行时 rank 过滤（不截断 codebook）；只允许使用 `rank < k` 的规则。
- 模式：
  - `encode_rank_mode='all'`：使用全部规则（默认）。
  - `encode_rank_mode='topk', encode_rank_k=K`：固定 topk。
  - `encode_rank_mode='random', encode_rank_min=A, encode_rank_max=B, encode_rank_dist='uniform|triangular'`：每次 encode 采样一个 k∈[A,B]。
  - `encode_rank_mode='gaussian', encode_rank_min=A, encode_rank_max=B`：截断高斯采样，均值偏向上界（更多规则）。
- 批次行为：同一批次仅采样一次 k，并调用 C++ 的 `batch_encode_with_limit` 以减少开销。
- 正确性保障：C++ 提供 `encode_with_limit_trace`，pytest 断言 `max(rank) < k`。

示例：
```python
from src.algorithms.compression.bpe_engine import BPEEngine

# 固定 topk=128（示例沿用 encode 侧特性，与训练后端无关）
eng = BPEEngine(train_backend='cpp', encode_backend='cpp', encode_rank_mode='topk', encode_rank_k=128)
eng.train(seqs, num_merges=2000, min_frequency=10)
eng.build_encoder()
ids = eng.encode(seq)
ids_batch = eng.batch_encode(batch)

# 每次 encode 随机采样 k∈[64,256]
eng2 = BPEEngine(train_backend='cpp', encode_backend='cpp', encode_rank_mode='random',
                 encode_rank_min=64, encode_rank_max=256, encode_rank_dist='uniform')
eng2.merge_rules = eng.merge_rules; eng2.vocab_size = eng.vocab_size; eng2.build_encoder()
ids2 = eng2.encode(seq)  # 本次会随机采样一个 k，并仅使用 rank<k 的规则
```

## 七、DataLoader 集成（worker 内构造后端）

```python
def worker_init_fn(_):
    from src.algorithms.compression.bpe_engine import BPEEngine
    global g_encoder
    # 从 UDI 加载 codebook（略）
    g_engine = BPEEngine(train_backend='python', encode_backend='cpp',
                         encode_rank_mode='gaussian', encode_rank_min=64, encode_rank_max=256)
    g_engine.merge_rules = merge_rules
    g_engine.vocab_size = vocab_size
    g_engine.build_encoder()
    g_encoder = g_engine

def collate(batch):
    seqs = [item['seq'] for item in batch]
    ids  = g_encoder.batch_encode(seqs)
    return ids
```

## 八、pytest 覆盖
- `tests/test_bpe_rank_modes_pytest.py`
  - 固定 topk：C++ trace 验证 `max(rank) < k`；
  - 批次：一次采样并与 `batch_encode_topk` 一致；
  - random uniform：均值近似 [min,max] 中点；
  - gaussian：均值偏向上界，区间内截断。


