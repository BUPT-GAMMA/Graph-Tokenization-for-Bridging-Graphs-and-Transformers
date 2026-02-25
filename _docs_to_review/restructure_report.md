# 数据集重构报告

## 重构概览

- **重构时间**: 2025-07-02T10:02:55.678475
- **成功重构**: 7/7 个数据集
- **源目录**: data/raw/small
- **目标目录**: data

## 重构后的数据集结构

每个数据集现在都包含以下标准文件：

- `graph.bin`: DGL格式的图数据文件
- `metadata.json`: 完整的数据集元数据
- `node_mapping.pkl`: 节点ID映射（如果可用）
- `token_mapping.json`: Token映射表
- `README.md`: 数据集说明文档

## 重构的数据集

### cora
- 状态: ✅ 重构成功
- 文件数: 5
- 文件列表: README.md, metadata.json, token_mapping.json, node_mapping.pkl, graph.bin

### citeseer
- 状态: ✅ 重构成功
- 文件数: 5
- 文件列表: node_mapping.pkl, token_mapping.json, metadata.json, README.md, graph.bin

### dblp
- 状态: ✅ 重构成功
- 文件数: 5
- 文件列表: node_mapping.pkl, README.md, token_mapping.json, metadata.json, graph.bin

### imdb
- 状态: ✅ 重构成功
- 文件数: 5
- 文件列表: node_mapping.pkl, README.md, graph.bin, token_mapping.json, metadata.json

### lastfm
- 状态: ✅ 重构成功
- 文件数: 5
- 文件列表: graph.bin, metadata.json, README.md, token_mapping.json, node_mapping.pkl

### pubmed
- 状态: ✅ 重构成功
- 文件数: 5
- 文件列表: node_mapping.pkl, token_mapping.json, README.md, graph.bin, metadata.json

### yelp
- 状态: ✅ 重构成功
- 文件数: 5
- 文件列表: metadata.json, README.md, graph.bin, node_mapping.pkl, token_mapping.json


## 接下来的步骤

1. 更新数据集加载器以使用新的存储格式
2. 测试重构后的数据集是否能正常加载
3. 更新文档和示例代码
4. 删除旧的`data/raw/small`目录（已备份）

## 备份信息

原始数据已备份到: `data/backup_small`
