from typing import Dict, Any, List
import dgl

from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping


class ImageDiagZigzagSerializer(BaseGraphSerializer):
    """
    图像斜对角 Zigzag 扫描序列化器（JPEG 风格）：
    - 对角层 s = r + c，从 0 到 H+W-2；偶层自上而下，奇层自下而上（或反之）
    - 仅节点像素 token，禁用边 token
    - 需 image_shape=(H,W,C) 且 N==H*W
    """

    def __init__(self):
        super().__init__()
        self.name = "image_diag_zigzag"
        self.include_edge_tokens = False

    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        return

    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        dgl_graph = self._validate_graph_data(graph_data)
        shape = graph_data.get('image_shape', None)
        if shape is None:
            raise ValueError("缺少 image_shape (H,W,C)")
        H, W, C = shape
        if dgl_graph.num_nodes() != H * W:
            raise ValueError(f"节点数与 H*W 不一致: N={dgl_graph.num_nodes()}, H*W={H*W}")

        node_path: List[int] = []
        max_s = H + W - 2
        for s in range(max_s + 1):
            r_min = max(0, s - (W - 1))
            r_max = min(H - 1, s)
            if (s % 2) == 0:
                # 偶数层：r 从 r_max 到 r_min 递减
                r_iter = range(r_max, r_min - 1, -1)
            else:
                # 奇数层：r 从 r_min 到 r_max 递增
                r_iter = range(r_min, r_max + 1)
            for r in r_iter:
                c = s - r
                node_path.append(r * W + c)

        token_ids, element_ids = self._convert_path_to_tokens(node_path, graph_data)
        id_map = GlobalIDMapping(dgl_graph)
        return SerializationResult([token_ids], [element_ids], id_map)



