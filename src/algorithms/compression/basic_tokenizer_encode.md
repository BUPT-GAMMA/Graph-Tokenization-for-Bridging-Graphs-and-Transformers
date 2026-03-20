## BasicTokenizer.encode 设计与实现说明（不涉及字符串处理）

本文档从整数 id 序列的角度，说明 BasicTokenizer 的 `encode` 算法流程与依赖，不讨论文本到字节的转换细节。

### 范围与目标

- **输入**: 一个由 0..255 之间整数组成的序列 `ids`（可视为 UTF-8 字节流的数值形式）。
- **输出**: 经过 BPE 合并后的整数序列。序列中可能包含 ≥256 的新 token id，这些是训练阶段按顺序生成的合并 token。
- **目标**: 按已训练好的合并规则，反复将当前序列中“可合并且在训练时最早定义”的相邻对替换为其合并 id，直到无法继续合并。

### 依赖的内部状态

- `merges: Dict[Tuple[int, int], int]`
  - 键为相邻对 `(p0, p1)`，值为该对在训练中生成的新 token id（从 256 起递增）。
  - “id 越小，表示该合并在训练中出现得越早”。
- `vocab: Dict[int, bytes]`
  - 由 `merges` 推导得到，用于解码；`encode` 的核心流程不直接依赖它。

### 依赖的辅助函数

- `get_stats(ids, counts=None) -> Dict[Tuple[int,int], int]`
  - 统计序列中所有相邻对的出现次数，返回形如 `{(a,b): count}` 的字典。
- `merge(ids, pair, idx) -> List[int]`
  - 在线性扫描中，将序列中所有不重叠的 `pair=(p0,p1)` 替换为 `idx`，返回新的序列。

### 算法流程

1. 准备初始序列 `ids`（由 0..255 的整数构成）。
2. 进入循环，只要 `len(ids) >= 2`：
   - 调用 `get_stats(ids)` 统计所有相邻对的出现次数。
   - 在所有出现过的相邻对中，选出训练时“最早定义”的可合并对：
     - 具体实现为：`pair = min(stats, key=lambda p: merges.get(p, +inf))`。
     - 若该 `pair` 不在 `merges` 中（表示当前序列不存在任何可用合并），则终止循环。
   - 令 `idx = merges[pair]`，调用 `merge(ids, pair, idx)`，将序列中所有不重叠的该对替换为 `idx`，并将结果作为新的 `ids`。
3. 当不存在可继续的合并时，返回当前的 `ids` 作为编码结果。

### 关键性质

- **确定性**: `merges` 在训练时按固定顺序生成；`encode` 始终优先应用“合并 id 最小”的可合并对，因此对同一输入序列输出一致。
- **不重叠替换**: `merge` 采用线性扫描，匹配到一处 `pair` 后跳过两个位置，确保同一轮内替换不重叠。
- **收敛**: 每轮合并都会减少序列长度或保持不变；一旦没有相邻对在 `merges` 中出现，循环终止。

### 与训练的关系

- 训练阶段每一轮：
  - 在当前序列上统计相邻对，选择出现次数最多的相邻对进行合并；
  - 为该合并分配新的 token id（从 256 起递增），记入 `merges`；
  - 由此形成“合并先后顺序”。
- 编码阶段通过最小化 `merges[pair]` 来复用训练顺序：即在所有可合并对里，优先应用“训练时更早得到”的合并。

### 复杂度与边界情况

- **时间复杂度（粗略）**: 每轮一次 `get_stats` 和一次 `merge`，都为 O(n)；总轮数最多为训练产生的合并数或直至不可合并，整体近似 O(n × 轮数)。
- **空间复杂度**: 序列按轮更新，空间为 O(n)。
- **边界**:
  - 若初始 `ids` 长度 < 2，则直接返回（无任何合并）。
  - 若 `merges` 为空（未训练），返回初始 `ids`。

### 伪代码（忽略字符串处理）

```
function encode(ids: List[int], merges: Dict[(int,int) -> int]) -> List[int]:
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = argmin_over_pairs(stats.keys(), key = merges.get(pair, +inf))
        if pair not in merges:
            break
        idx = merges[pair]
        ids = merge(ids, pair, idx)  # non-overlapping replacement in one pass
    return ids
```

---

## 源码实现（摘录）

以下为相关函数的实际实现（截至当前仓库版本），便于对照阅读。

### base.get_stats

```python
def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

### base.merge

```python
def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

### BasicTokenizer.train（与 merges 构建相关）

```python
def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    # input text preprocessing
    text_bytes = text.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers in range 0..255

    # iteratively merge the most common pairs to create new tokens
    merges = {} # (int, int) -> int
    vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
    for i in range(num_merges):
        # count up the number of times every consecutive pair appears
        stats = get_stats(ids)
        # find the pair with the highest count
        pair = max(stats, key=stats.get)
        # mint a new token: assign it the next available id
        idx = 256 + i
        # replace all occurrences of pair in ids with idx
        ids = merge(ids, pair, idx)
        # save the merge
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        # prints
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

    # save class variables
    self.merges = merges # used in encode()
    self.vocab = vocab   # used in decode()
```

### BasicTokenizer.encode

```python
def encode(self, text):
    # given a string text, return the token ids
    text_bytes = text.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers in range 0..255
    while len(ids) >= 2:
        # find the pair with the lowest merge index
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
        # subtle: if there are no more merges available, the key will
        # result in an inf for every single pair, and the min will be
        # just the first pair in the list, arbitrarily
        # we can detect this terminating case by a membership check
        if pair not in self.merges:
            break # nothing else can be merged anymore
        # otherwise let's merge the best pair (lowest merge index)
        idx = self.merges[pair]
        ids = merge(ids, pair, idx)
    return ids
```


