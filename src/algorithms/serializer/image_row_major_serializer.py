from typing import Dict, Any, List, Tuple
import torch
import dgl

from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping


class ImageRowMajorSerializer(BaseGraphSerializer):
    """
    图像行优先扫描序列化器：
    - 要求图为 HxW 的栅格，节点 ID 严格为 row-major：node_id = r * W + c
    - 仅输出节点像素 token（include_edge_tokens 固定为 False）
    - 返回标准 SerializationResult
    """

    def __init__(self):
        super().__init__()
        self.name = "image_row_major"
        self.include_edge_tokens = False

    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        # 无需统计
        return

    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        dgl_graph = self._validate_graph_data(graph_data)
        shape = graph_data.get('image_shape', None)
        if shape is None:
            raise ValueError("缺少 image_shape (H,W,C)")
        H, W, C = shape
        if dgl_graph.num_nodes() != H * W:
            raise ValueError(f"节点数与 H*W 不一致: N={dgl_graph.num_nodes()}, H*W={H*W}")

        # 行优先顺序: 0..N-1
        N = H * W
        node_path = list(range(N))

        # 一次性张量化取数并转为序列
        token_ids, element_ids = self._convert_path_to_tokens(node_path, graph_data)

        id_map = GlobalIDMapping(dgl_graph)
        return SerializationResult([token_ids], [element_ids], id_map)



