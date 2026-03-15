"""
MNIST-RAW 数据准备模块（数据层标准位置）
=====================================

功能：
- 使用 torchvision 下载/加载 MNIST 原始数据
- 生成轻量 data.pkl（仅像素 uint8[28,28] 与 int 标签）
- 生成 train/val/test 三份索引 JSON
- 提供验证函数，检查是否符合 docs/data_layer_spec_mnist_raw.md 规范

说明：不包含任何回退逻辑；外网不可用时会直接报错。
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import json
import pickle

import numpy as np

from config import ProjectConfig


def prepare_mnist_raw(config: ProjectConfig, val_ratio: float = 0.1, test_ratio: float = 0.1, download: bool = True) -> None:
	"""准备 MNIST-RAW 数据到 {config.data_dir}/mnist_raw。

	生成：data.pkl、train_index.json、val_index.json、test_index.json。
	"""
	from torchvision import datasets, transforms

	out_dir = Path(config.data_dir) / "mnist_raw"
	out_dir.mkdir(parents=True, exist_ok=True)

	# 下载/加载 MNIST（使用 torchvision）
	transform = transforms.Compose([transforms.ToTensor()])
	train_set = datasets.MNIST(str(out_dir), train=True, download=download, transform=transform)
	test_set = datasets.MNIST(str(out_dir), train=False, download=download, transform=transform)

	# 合并数据
	images: List[np.ndarray] = []
	labels: List[int] = []
	for img, lbl in train_set:
		arr = (img.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
		images.append(arr)
		labels.append(int(lbl))
	for img, lbl in test_set:
		arr = (img.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
		images.append(arr)
		labels.append(int(lbl))

	total = len(images)
	assert total > 0, "MNIST 加载为空"

	val_n = int(total * val_ratio)
	test_n = int(total * test_ratio)
	train_n = total - val_n - test_n
	assert train_n > 0 and val_n > 0 and test_n > 0, "无效的划分比例导致样本数为0"

	train_indices = list(range(0, train_n))
	val_indices = list(range(train_n, train_n + val_n))
	test_indices = list(range(train_n + val_n, total))

	# 写索引
	(out_dir / "train_index.json").write_text(json.dumps(train_indices))
	(out_dir / "val_index.json").write_text(json.dumps(val_indices))
	(out_dir / "test_index.json").write_text(json.dumps(test_indices))

	# 轻量 data.pkl
	data_path = out_dir / "data.pkl"
	with data_path.open('wb') as f:
		pickle.dump([(img, lbl) for img, lbl in zip(images, labels)], f, protocol=pickle.HIGHEST_PROTOCOL)

	print(f"Saved MNIST_RAW to {data_path} with {total} samples.")


def validate_mnist_raw_format(config: ProjectConfig) -> None:
	"""校验 mnist_raw 是否符合数据层规范。失败将抛出异常。"""
	from src.data.unified_data_factory import get_dataloader
	import torch

	base = Path(config.data_dir) / 'mnist_raw'
	for name in ["data.pkl", "train_index.json", "val_index.json", "test_index.json"]:
		p = base / name
		assert p.exists(), f"缺少文件: {p}"

	# 进一步检查 Loader 样本与图结构
	loader = get_dataloader('mnist_raw', config)
	train, val, test, ytr, yv, yte = loader.load_data()
	assert len(train) > 0 and len(val) > 0 and len(test) > 0, "样本划分为空"
	s = train[0]
	required = {'id','dgl_graph','image_shape','num_nodes','num_edges','properties','dataset_name','data_type'}
	assert required.issubset(set(s.keys())), f"样本字段缺失: {required - set(s.keys())}"
	H,W,C = s['image_shape']
	assert (H,W,C) == (28,28,1), f"image_shape 非 28x28x1: {s['image_shape']}"
	g = s['dgl_graph']
	assert g.num_nodes() == H * W, "节点数不匹配"
	expected_undirected_edges = (W - 1) * H + (H - 1) * W
	assert g.num_edges() == 2 * expected_undirected_edges, "边数不匹配（应为双向）"
	nt = loader.get_graph_node_token_ids(g)
	et = loader.get_graph_edge_token_ids(g)
	assert isinstance(nt, torch.Tensor) and nt.dtype == torch.long and nt.shape == (g.num_nodes(), 1), "节点 token 张量形状/类型不符"
	assert isinstance(et, torch.Tensor) and et.dtype == torch.long and et.shape == (g.num_edges(), 1), "边 token 张量形状/类型不符"
	print("mnist_raw format OK")


if __name__ == "__main__":
	cfg = ProjectConfig()
	prepare_mnist_raw(cfg)
	validate_mnist_raw_format(cfg)




