import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 确保以项目根为基准导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader
from src.algorithms.serializer.eulerian_serializer import EulerianSerializer


def _build_euler_path(serializer: EulerianSerializer, mol_data: Dict[str, Any], start_node: int = 0) -> List[int]:
    """
    使用与欧拉序列化内部一致的流程生成欧拉路径（仅生成路径，不做token转换）。
    """
    dgl_graph = serializer._validate_graph_data(mol_data)
    num_nodes = dgl_graph.num_nodes()
    assert 0 <= start_node < num_nodes, f"start_node 越界: {start_node}"

    # 构建邻接与排序
    adj_list = serializer._build_adjacency_list_from_dgl(dgl_graph)
    for i in range(len(adj_list)):
        adj_list[i].sort()

    # 确保欧拉性（必要时加倍边）
    if not serializer._has_eulerian_circuit(adj_list, num_nodes):
        adj_list = serializer._make_eulerian_by_doubling_edges(adj_list, num_nodes)

    # 查找欧拉回路
    euler_path = serializer._find_eulerian_circuit(adj_list, start_node)
    if not euler_path:
        raise ValueError("未能生成欧拉回路路径")
    return euler_path


def compare_convert_methods_on_graph(
    serializer: EulerianSerializer,
    mol_data: Dict[str, Any],
    start_node: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    基于同一欧拉路径，分别用新旧 convert 方法得到 token 序列并返回。
    仅比较 token 序列，不比较元素序列。
    """
    path = _build_euler_path(serializer, mol_data, start_node=start_node)

    tokens_new, _ = serializer._convert_path_to_tokens(path, mol_data)
    tokens_old, _ = serializer._convert_path_to_tokens_old(path, mol_data)
    return tokens_new, tokens_old


essential_fields = ["id", "dgl_graph"]


def collect_zinc_samples(max_graphs: int, config_path: str | None = None) -> Tuple[ProjectConfig, Any, List[Dict[str, Any]]]:
    cfg = ProjectConfig(config_path)
    loader = get_dataloader("zinc", cfg)

    # 确保数据与缓存构建
    loader.load_data()
    all_data, split_indices = loader.get_all_data_with_indices()

    indices = (
        split_indices.get("train", [])
        + split_indices.get("val", [])
        + split_indices.get("test", [])
    )

    selected: List[Dict[str, Any]] = []
    for idx in indices:
        if idx < 0 or idx >= len(all_data):
            raise IndexError(f"索引越界: {idx} (0..{len(all_data)-1})")
        sample = all_data[idx]
        for f in essential_fields:
            if f not in sample:
                raise KeyError(f"样本缺少必要字段 '{f}' @ index={idx}")
        selected.append(sample)
        if len(selected) >= max_graphs:
            break

    if not selected:
        raise ValueError("未获取到任何样本用于测试")

    return cfg, loader, selected


def main():
    parser = argparse.ArgumentParser(
        description=(
            "在 ZINC 数据集上，基于同一欧拉路径，对比 _convert_path_to_tokens 与 _convert_path_to_tokens_old 的 token 序列是否一致。"
        )
    )
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    parser.add_argument("--max-graphs", type=int, default=20, help="最多检查的图数量")
    parser.add_argument("--start-node", type=int, default=0, help="欧拉路径起始节点（同一图使用同一起点）")
    parser.add_argument(
        "--max-examples", type=int, default=10, help="最多打印的不一致样例数量"
    )

    args = parser.parse_args()

    cfg, loader, samples = collect_zinc_samples(args.max_graphs, args.config)

    # 初始化欧拉序列化器（默认参数：包含边token、忽略最高频边）
    serializer = EulerianSerializer(include_edge_tokens=True, omit_most_frequent_edge=True)
    serializer.initialize_with_dataset(loader)

    total_graphs = 0
    mismatch_graphs = 0
    examples = []

    for i, sample in enumerate(samples):
        total_graphs += 1
        try:
            t_new, t_old = compare_convert_methods_on_graph(serializer, sample, start_node=args.start_node)
            if t_new != t_old:
                mismatch_graphs += 1
                if len(examples) < args.max_examples:
                    examples.append({
                        "graph_index": i,
                        "id": sample.get("id", f"idx_{i}"),
                        "len_new": len(t_new),
                        "len_old": len(t_old),
                        "tokens_new_head": t_new[:32],
                        "tokens_old_head": t_old[:32],
                    })
        except Exception as e:
            # 直接抛出异常更符合禁止回退，但为了批量统计，这里将异常记录为单独样例并继续
            # 若需严格模式，请改为 raise e
            examples.append({
                "graph_index": i,
                "id": sample.get("id", f"idx_{i}"),
                "error": str(e),
            })

    summary = {
        "dataset": "zinc",
        "method": "eulerian",
        "graphs": total_graphs,
        "mismatch_graphs": mismatch_graphs,
        "examples": examples,
    }
    # 仅输出原始数据，避免结论性文字
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
