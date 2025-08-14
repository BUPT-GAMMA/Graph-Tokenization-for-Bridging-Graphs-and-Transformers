from __future__ import annotations

from typing import List, Tuple


def flatten_variants(
    variants_per_graph: List[List[List[int]]],
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    将每图的多变体序列结构展平为样本列表，并返回 graph_id 与 variant_id 映射。

    参数：
      - variants_per_graph: 长度为 G 的列表；每项是该图的变体序列列表（每个变体为 List[int]）。

    返回：
      - flat_sequences: List[List[int]]
      - graph_ids: 与 flat_sequences 对齐的 graph_id 列表（图索引）
      - variant_ids: 与 flat_sequences 对齐的 variant_id 列表（在该图下的变体序号）
    """
    flat_sequences: List[List[int]] = []
    graph_ids: List[int] = []
    variant_ids: List[int] = []
    for gid, variants in enumerate(variants_per_graph):
        for vid, seq in enumerate(variants):
            flat_sequences.append([int(x) for x in seq])
            graph_ids.append(gid)
            variant_ids.append(vid)
    return flat_sequences, graph_ids, variant_ids



