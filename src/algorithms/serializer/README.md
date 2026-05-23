# Graph Serialization

This module converts graph structures into token sequences that can be fed into sequence models. All serializers share a common interface and are created through a factory.

## Available Methods

### Graph traversal methods

| Method | File | Description |
|--------|------|-------------|
| `feuler` | `freq_eulerian_serializer.py` | **Recommended.** Frequency-guided Eulerian circuit |
| `eulerian` | `eulerian_serializer.py` | Standard Eulerian circuit |
| `cpp` | `chinese_postman_serializer.py` | Chinese Postman path |
| `fcpp` | `freq_chinese_postman_serializer.py` | Frequency-guided Chinese Postman |
| `dfs` | `dfs_serializer.py` | Depth-first traversal |
| `bfs` | `bfs_serializer.py` | Breadth-first traversal |
| `topo` | `topo_serializer.py` | Topological sort (directed graphs) |

### SMILES methods (molecular graphs only)

| Method | File | Description |
|--------|------|-------------|
| `smiles` | `smiles_serializer.py` | Default SMILES |
| `smiles_1` .. `smiles_4` | `smiles_serializer.py` | SMILES variants |

### Image methods (grid graphs)

| Method | File | Description |
|--------|------|-------------|
| `image_row_major` | `image_row_major_serializer.py` | Row-major scan |
| `image_serpentine` | `image_serpentine_serializer.py` | Serpentine (boustrophedon) |
| `image_diag_zigzag` | `image_diag_zigzag_serializer.py` | Diagonal zigzag |

## Interface

All serializers inherit from `BaseGraphSerializer` and implement:

```python
serialize(graph_data)                     # Single graph → SerializationResult
multiple_serialize(graph_data, n)         # Single graph, n random variants
batch_serialize(graph_list, parallel)     # Batch of graphs
batch_multiple_serialize(graph_list, n)   # Batch × n variants
```

Each returns a `SerializationResult` with `get_sequence(i) -> (token_ids, element_labels)`.

## Usage

Most users should go through `UnifiedDataInterface` rather than calling serializers directly (see `src/data/README.md`). But for direct use:

```python
from src.algorithms.serializer.serializer_factory import SerializerFactory

serializer = SerializerFactory.create_serializer('feuler')
serializer.initialize_with_dataset(loader, train_data)  # feuler needs global freq stats

result = serializer.serialize(graph_data)
token_seq, element_seq = result.get_sequence(0)
```

## Choosing a Method

- **Molecular graphs, deterministic**: `feuler` (recommended)
- **Molecular graphs, chemical semantics**: `smiles`
- **General graphs, simple**: `dfs` or `bfs`
- **Directed graphs**: `topo`
- **Image/grid graphs**: `image_row_major`

## Method Notes

- **feuler** requires `initialize_with_dataset()` to collect frequency statistics before use. It is deterministic given the same dataset.
- **eulerian** is a simpler alternative that doesn't use frequency guidance.
- **DFS/BFS** are the simplest and fastest but may produce longer sequences.
- **SMILES** methods require RDKit and valid molecular structures.
- To reduce sequence length, set `omit_most_frequent_edge=True` (default) or apply BPE compression.

## Legacy Name Mapping

| Old name | Current name |
|----------|-------------|
| `graph_seq` | `feuler` |
| `topological` | `topo` |
