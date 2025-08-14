MNIST-RAW 数据层接口与内容规范
================================

目的
----
为支持“图像式扫描序列化”（行优先、蛇形、斜对角 zigzag），本规范约束 MNIST-RAW 的数据预处理与 DataLoader 输出的图与样本结构，确保序列化器无需任何回退逻辑即可工作，且全流程确定可复现。

范围
----
- 预处理脚本：`scripts/prepare_mnist_raw.py`
- 数据加载器：`src/data/mnist_raw_loader.py`（类名建议 `MNISTRawDataLoader`，数据集名固定为 `mnist_raw`）

存储格式（磁盘）
--------------
- 目录：`{config.data_dir}/mnist_raw`
- 文件：
  - `data.pkl`：列表，每个元素为 `(np.ndarray[H,W], int_label)`；MNIST 固定 `H=W=28`，`dtype=np.uint8`，像素范围 `0..255`
  - `train_index.json`, `val_index.json`, `test_index.json`：索引列表（整数）
- 说明：构图在加载器中进行（按需从像素数组转换为 DGL 栅格图），以减小磁盘占用。

样本结构（内存）
--------------
DataLoader 产出样本为 Python dict，必须包含如下键：

- 必需字段
  - `id: str`：唯一 ID（如 `"image_{idx}"`）
  - `dgl_graph: dgl.DGLGraph`：同构图；节点表示像素，边为 4-邻接双向边
  - `image_shape: Tuple[int,int,int]`：`(H, W, C)`；MNIST 固定为 `(28, 28, 1)`
  - `num_nodes: int == H*W`
  - `num_edges: int`：无向边数（注意 DGL 双向存为两条有向边，统计时可给无向计数）
  - `properties: Dict[str, Any]`：至少包含 `{'label': int}`
  - `dataset_name: str == 'mnist_raw'`
  - `data_type: str == 'image_grid_graph'`

- 约束与不变量
  - 节点 ID 必须严格按 row-major（行优先展平）编号：`node_id = row * W + col`
  - `dgl_graph.num_nodes() == H * W`，否则直接报错
  - 不存任何隐式回退（字段缺失时应抛出异常）

图内特征（g.ndata / g.edata）
-----------------------------
- 节点特征（必需）
  - `g.ndata['feature']`: `LongTensor` 或 `UInt8Tensor` 可被安全转换为 Long，形状 `[N, 1]` 或 `[N, C]`
    - MNIST：`[N, 1]`，值域 `0..255`（像素值整数化）
  - 建议但非必需：`g.ndata['valid']`（BoolTensor `[N]`）全部为 True（未来可用于 ROI/裁剪）

- 边特征（建议）
  - 4-邻接双向边：对每个无向相邻像素 `(u,v)`，在 DGL 中存 `(u->v)` 与 `(v->u)` 两条边
  - `g.edata['feature']`: `LongTensor [E]`，固定常数 1（占位）
  - 若后续需要方向区分，可扩展 `g.edata['etype_id']`（horizontal/vertical 等），本批序列化默认不读取边特征

DataLoader 接口实现要求
-------------------------
在 `MNISTRawDataLoader(BaseDataLoader)` 中，以下方法的行为应满足：

- 构图与样本生成
  - `_load_processed_data()`：从 `data.pkl` 读取像素与标签，按索引构建样本列表；样本字典必须包含上述“样本结构（内存）”的所有必需键
  - `_image_to_dgl(img_uint8: np.ndarray) -> dgl.DGLGraph`：
    - 创建 `H*W` 节点，严格以 row-major 规则编号
    - 添加 4-邻接双向边
    - 赋值 `g.ndata['feature'] = torch.from_numpy(img_uint8.reshape(-1, 1))`（后续由接口转换为 Long）
    - 赋值 `g.edata['feature'] = torch.ones(E, dtype=torch.long)`

- 标签与元信息
  - `_extract_labels(...)`：返回 `List[int]` 的标签
  - `_get_data_metadata()`：补充统计信息与像素范围 `0..255`

- Token/属性接口（满足序列化器张量化路径）
  - `get_graph_node_token_ids(g) -> LongTensor [N, C]`：从 `g.ndata['feature']` 转为 `long` 并返回（MNIST 为 `[N,1]`）
  - `get_graph_edge_token_ids(g) -> LongTensor [E, De]`：从 `g.edata['feature']` 转为 `long` 并返回（占位 `[E,1]`）
  - `get_graph_node_type_ids(g) -> LongTensor [N]`：允许全部返回常数 1
  - `get_graph_edge_type_ids(g) -> LongTensor [E]`：允许全部返回常数 1
  - `get_most_frequent_edge_type() -> str`：返回固定字符串（如 `'edge'`）
  - `get_edge_type_id_by_name(name: str) -> int`：至少支持 `name=='edge' -> 1`

对现有实现的差异与修改点
--------------------------
基于当前 `src/data/mnist_raw_loader.py`（已实现行优先节点编号与 4-邻接构图），需补充：

1) 在 `_load_processed_data()` 生成 `sample` 时，新增：

```python
sample.update({
    'image_shape': (28, 28, 1),  # H, W, C
})
```

2) 确认/保持：
- `g.ndata['feature']` 为 `[N,1]` 且可转换为 Long（当前已满足）
- `dataset_name == 'mnist_raw'`，`data_type == 'image_grid_graph'`（当前已满足）
- 不添加 `coord` 等冗余字段（行优先 ID 已足够）

统一性与错误处理
----------------
- 所有必需字段缺失时应立刻抛出异常，不做任何回退（不要用默认值掩盖问题）
- 所有配置来自全局 `config.py`；路径等通过 `ProjectConfig.data_dir` 统一管理

对序列化器的依赖假设
--------------------
图像扫描序列化器仅依赖以下事实：

1) 节点 ID 为 row-major：`node_id = row * W + col`
2) 样本中提供 `image_shape=(H,W,C)`，满足 `g.num_nodes() == H*W`
3) `get_graph_node_token_ids(g)` 返回 `[N,C]` 的 LongTensor（MNIST 为 `[N,1]`）
4) 可忽略边 token（序列化器内部将固定 `include_edge_tokens=False`）

示例样本字典（最小）
------------------
```python
{
  'id': 'image_123',
  'dgl_graph': g,                 # 28x28 栅格同构图
  'image_shape': (28, 28, 1),
  'num_nodes': 784,
  'num_edges': 1512,              # 无向边数；DGL 实际存为 3024 有向边
  'properties': {'label': 7},
  'dataset_name': 'mnist_raw',
  'data_type': 'image_grid_graph',
}
```

备注
----
- 如需通过工厂创建 DataLoader，建议在 `src/data/unified_data_factory.py` 注册 `'mnist_raw' -> MNISTRawDataLoader`（延迟导入）
- 为后续可扩展到多通道/可变大小图像，本规范不引入 `coord`；若将来有 ROI 或非规则掩码需求，可新增 `g.ndata['valid']`



