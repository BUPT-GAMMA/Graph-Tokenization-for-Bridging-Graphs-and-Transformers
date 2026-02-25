from typing import Dict, Any, List
import dgl

from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping


class ImageSerpentineSerializer(BaseGraphSerializer):
    """Image serpentine (boustrophedon) scan serializer."""

    def __init__(self):
        super().__init__()
        self.name = "image_serpentine"
        self.include_edge_tokens = False

    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        return

    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        dgl_graph = self._validate_graph_data(graph_data)
        shape = graph_data.get('image_shape', None)
        if shape is None:
            raise ValueError("Missing image_shape (H,W,C)")
        H, W, C = shape
        if dgl_graph.num_nodes() != H * W:
            raise ValueError(f"Node count mismatch: N={dgl_graph.num_nodes()}, H*W={H*W}")

        node_path: List[int] = []
        for r in range(H):
            if (r % 2) == 0:
                for c in range(W):
                    node_path.append(r * W + c)
            else:
                for c in range(W - 1, -1, -1):
                    node_path.append(r * W + c)

        token_ids, element_ids = self._convert_path_to_tokens(node_path, graph_data)
        id_map = GlobalIDMapping(dgl_graph)
        return SerializationResult([token_ids], [element_ids], id_map)



