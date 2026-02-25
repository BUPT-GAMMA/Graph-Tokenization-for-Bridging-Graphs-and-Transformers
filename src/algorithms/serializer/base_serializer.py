"""
Base class for graph serializers.
图序列化器基类。

Defines the unified token management and serialization interface.
定义统一的token管理和序列化接口。
"""

from collections import defaultdict
from math import log10
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import dgl
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.base_loader import BaseDataLoader
from utils.logger import get_logger
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import threading

logger = get_logger(__name__)


# ============== Multiprocess batch serialization (fork-only) ==============
# Uses fork on Linux/posix to share memory without pickling serializer
# instances or DGLGraph objects. No implicit fallback on non-fork platforms.
_MP_GLOBAL_SERIALIZER = None  # type: Optional["BaseGraphSerializer"]
_MP_GRAPH_DATA_LIST = None    # type: Optional[List[Dict[str, Any]]]
_MP_STATS_GRAPH_LIST = None   # type: Optional[List[Dict[str, Any]]]


def _mp_init_worker(serializer_ref, graph_data_ref):
    global _MP_GLOBAL_SERIALIZER, _MP_GRAPH_DATA_LIST
    _MP_GLOBAL_SERIALIZER = serializer_ref
    _MP_GRAPH_DATA_LIST = graph_data_ref


def _mp_process_range(lo_hi_kwargs: Tuple[int, int, Dict[str, Any]]):
    lo, hi, kwargs = lo_hi_kwargs
    ser = _MP_GLOBAL_SERIALIZER
    lst = _MP_GRAPH_DATA_LIST
    assert ser is not None and lst is not None, "Multiprocess serialization not properly initialized"
    out: List[Tuple[int, SerializationResult]] = []
    for i in range(lo, hi):
        res = ser.serialize(lst[i], **kwargs)
        out.append((i, res))
    return out

# ============== batch_multiple_serialize multiprocess worker ==============

def _mp_multiple_process_range(args: Tuple[int, int, int, Dict[str, Any]]):
    """Multiprocess range handler: calls multiple_serialize."""
    lo, hi, num_samples, kwargs = args
    ser = _MP_GLOBAL_SERIALIZER
    lst = _MP_GRAPH_DATA_LIST
    assert ser is not None and lst is not None, "Multiprocess serialization not properly initialized"
    out: List[Tuple[int, SerializationResult]] = []
    for i in range(lo, hi):
        res = ser.multiple_serialize(lst[i], num_samples=num_samples, parallel=False, **kwargs)
        out.append((i, res))
    return out

# ============== Multiprocess stats collection (fork-only) ==============
_MP_STATS_SERIALIZER = None  # type: Optional["BaseGraphSerializer"]
_MP_STATS_GRAPH_LIST = None  # type: Optional[List[Dict[str, Any]]]


def _mp_stats_init(serializer_ref, graph_data_ref):
    global _MP_STATS_SERIALIZER, _MP_STATS_GRAPH_LIST
    _MP_STATS_SERIALIZER = serializer_ref
    _MP_STATS_GRAPH_LIST = graph_data_ref


def _mp_stats_max_range(lo_hi: Tuple[int, int]) -> Tuple[int, int]:
    lo, hi = lo_hi
    ser = _MP_STATS_SERIALIZER
    lst = _MP_STATS_GRAPH_LIST
    assert ser is not None and lst is not None
    import torch as _torch
    max_node = _torch.tensor(0, dtype=_torch.long)
    max_edge = _torch.tensor(0, dtype=_torch.long)
    for i in range(lo, hi):
        assert 'dgl_graph' in lst[i], f"Graph data missing required field 'dgl_graph': index {i}"
        g = lst[i]['dgl_graph']
        node_type_ids = ser._dataset_loader.get_graph_node_type_ids(g)
        edge_type_ids = ser._dataset_loader.get_graph_edge_type_ids(g)
        if node_type_ids.numel() > 0:
            max_node = _torch.maximum(max_node, node_type_ids.max())
        if edge_type_ids.numel() > 0:
            max_edge = _torch.maximum(max_edge, edge_type_ids.max())
    # Return max_id + 1 as dimension hint
    return int(max_node.item()) + 1 if max_node.item() >= 0 else 0, int(max_edge.item()) + 1 if max_edge.item() >= 0 else 0


def _mp_stats_count_range(args: Tuple[int, int, int, int]) -> "torch.Tensor":
    lo, hi, Vdim0, Edim = args
    ser = _MP_STATS_SERIALIZER
    lst = _MP_STATS_GRAPH_LIST
    assert ser is not None and lst is not None
    import torch as _torch
    Vdim1 = Vdim0
    flat_len = Vdim0 * Edim * Vdim1
    # Use int32 to reduce memory; bincount outputs int64, convert to int32 before accumulation
    flat = _torch.zeros(flat_len, dtype=_torch.int32)
    for i in range(lo, hi):
        assert 'dgl_graph' in lst[i], f"Graph data missing required field 'dgl_graph': index {i}"
        g = lst[i]['dgl_graph']
        node_type_ids = ser._dataset_loader.get_graph_node_type_ids(g)
        edge_type_ids = ser._dataset_loader.get_graph_edge_type_ids(g)
        if edge_type_ids.numel() == 0:
            continue
        src, dst = ser._get_all_edges_from_heterograph(g)
        src_t = node_type_ids.index_select(0, src).long()
        dst_t = node_type_ids.index_select(0, dst).long()
        et_t = edge_type_ids.long()
        # Bounds check
        if src_t.numel() == 0:
            continue
        if src_t.max() >= Vdim0 or dst_t.max() >= Vdim1 or et_t.max() >= Edim:
            raise ValueError("Type ID out of bounds during stats counting. Check dimension inference.")
        lin = src_t * (Edim * Vdim1) + et_t * Vdim1 + dst_t
        cnt = _torch.bincount(lin, minlength=flat_len)
        flat.add_(cnt.to(_torch.int32))
    return flat


def _mp_stats_count_range_sparse(args: Tuple[int, int, int, int]) -> Dict[int, int]:
    """Multiprocess sparse stats: returns linear_key -> count dict.
    Avoids building a dense tensor; only accumulates observed keys.
    """
    lo, hi, Vdim0, Edim = args
    ser = _MP_STATS_SERIALIZER
    lst = _MP_STATS_GRAPH_LIST
    assert ser is not None and lst is not None
    import torch as _torch
    Vdim1 = Vdim0
    counts: Dict[int, int] = {}
    for i in range(lo, hi):
        assert 'dgl_graph' in lst[i], f"Graph data missing required field 'dgl_graph': index {i}"
        g = lst[i]['dgl_graph']
        edge_type_ids = ser._dataset_loader.get_graph_edge_type_ids(g)
        if edge_type_ids.numel() == 0:
            continue
        node_type_ids = ser._dataset_loader.get_graph_node_type_ids(g)
        src, dst = ser._get_all_edges_from_heterograph(g)
        src_t = node_type_ids.index_select(0, src).long()
        dst_t = node_type_ids.index_select(0, dst).long()
        et_t = edge_type_ids.long()
        if src_t.numel() == 0:
            continue
        if src_t.max() >= Vdim0 or dst_t.max() >= Vdim1 or et_t.max() >= Edim:
            raise ValueError("Type ID out of bounds during stats counting. Check dimension inference.")
        lin = (src_t * (Edim * Vdim1) + et_t * Vdim1 + dst_t).to(_torch.long)
        if lin.numel() == 0:
            continue
        lin_sorted, _ = _torch.sort(lin)
        uniq, cts = _torch.unique_consecutive(lin_sorted, return_counts=True)
        for k, v in zip(uniq.tolist(), cts.tolist()):
            counts[k] = counts.get(k, 0) + int(v)
    return counts

class GlobalIDMapping:
    """Global-to-local ID mapping manager."""
    def __init__(self, dgl_graph: dgl.DGLGraph):
        self.global_to_local = {}  # global_id -> (ntype, local_id)
        self.local_to_global = {}  # (ntype, local_id) -> global_id
        self.current_global_id = 0
        
        # Build mapping
        for ntype in dgl_graph.ntypes:
            num_nodes = dgl_graph.num_nodes(ntype)
            for local_id in range(num_nodes):
                self.global_to_local[self.current_global_id] = (ntype, local_id)
                self.local_to_global[(ntype, local_id)] = self.current_global_id
                self.current_global_id += 1
    
    def to_global_id(self, ntype: str, local_id: int) -> int:
        """Convert local ID to global ID."""
        return self.local_to_global.get((ntype, local_id))
    
    def to_local_id(self, global_id: int) -> Tuple[str, int]:
        """Convert global ID to (ntype, local_id)."""
        return self.global_to_local.get(global_id)
    
    def get_total_nodes(self) -> int:
        """Get total node count."""
        return self.current_global_id


class SerializationResult:
    """Serialization result in unified format."""
    def __init__(self, token_sequences: List[List[int]], element_sequences: List[List[str]], id_mapping: GlobalIDMapping):
        assert token_sequences and element_sequences, "token_sequences and element_sequences must not be empty"
        
        assert len(token_sequences) == len(element_sequences), "token_sequences and element_sequences must have equal length"
        
        # Validate per-sequence length consistency
        for i, (token_seq, element_seq) in enumerate(zip(token_sequences, element_sequences)):
            assert len(token_seq) == len(element_seq), f"Sequence {i}: token/element length mismatch: {len(token_seq)} vs {len(element_seq)}"
        
        self.token_sequences = token_sequences
        self.element_sequences = element_sequences
        self.id_mapping = id_mapping
    
    def get_sequence_count(self) -> int:
        """Get number of sequences."""
        return len(self.token_sequences)
    
    def get_sequence(self, idx: int) -> Tuple[List[int], List[str]]:
        """Get sequence at given index.
        
        Args:
            idx: Sequence index
            
        Returns:
            Tuple[List[int], List[str]]: (token_ids, element_ids)
        """
        assert 0 <= idx < len(self.token_sequences), f"Sequence index {idx} out of range [0, {len(self.token_sequences)})"
        
        return (self.token_sequences[idx], self.element_sequences[idx])
    
    def get_element_info(self, idx: int) -> Tuple[str, str, int]:
        """Get full info for an element at a position in the first sequence.
        
        Args:
            idx: Position in the first sequence
        
        Returns:
            Tuple[element_type, element_id, token]
        """
        assert 0 <= idx < len(self.element_sequences[0]), f"Element index {idx} out of range [0, {len(self.element_sequences[0])})"
        
        element_id = self.element_sequences[0][idx]
        token = self.token_sequences[0][idx]
        
        # Parse element type
        if element_id.startswith('node_'):
            element_type = 'node'
        elif element_id.startswith('edge_'):
            element_type = 'edge'
        else:
            element_type = 'unknown'
        
        return (element_type, element_id, token)
    
    def get_graph_feature(self, idx: int, dgl_graph, feature_name: str):
        """Get feature from the DGL graph for element at given position.
        
        Args:
            idx: Position in the first sequence
            dgl_graph: DGL graph object
            feature_name: Feature name (e.g. 'feat', 'atomic_num', 'bond_type')
            
        Returns:
            Feature value (scalar or list)
        """
        element_type, element_id, token = self.get_element_info(idx)
        
        try:
            if element_type == 'node':
                # Parse node ID
                node_id = int(element_id.split('_')[1])
                if feature_name in dgl_graph.ndata:
                    feature_tensor = dgl_graph.ndata[feature_name][node_id]
                    if feature_tensor.dim() == 0:
                        return feature_tensor.item()
                    else:
                        return feature_tensor.detach().cpu().numpy().tolist()
                else:
                    raise ValueError(f"Node feature '{feature_name}' not found in graph")
                    
            elif element_type == 'edge':
                # Parse edge ID
                edge_id = int(element_id.split('_')[1])
                if feature_name in dgl_graph.edata:
                    feature_tensor = dgl_graph.edata[feature_name][edge_id]
                    if feature_tensor.dim() == 0:
                        return feature_tensor.item()
                    else:
                        return feature_tensor.detach().cpu().numpy().tolist()
                else:
                    raise ValueError(f"Edge feature '{feature_name}' not found in graph")
            else:
                raise ValueError(f"Unknown element type: {element_type}")
                
        except (IndexError, ValueError) as e:
            raise ValueError(f"Cannot get feature '{feature_name}' for element {element_id}: {str(e)}")


class BaseGraphSerializer(ABC):
    """Base class for graph serializers - unified interface."""
    
    def __init__(self):
        self.id_mapper = None
        self.name = "base_serializer"
        self._initialized = False
        self._dataset_loader: BaseDataLoader = None
        self._dataset_stats = {}
        # Edge ID mapping cache
        self._current_edge_id_mapping = None
        # Thread-local storage for concurrent safety
        self._thread_local = threading.local()
        # Numeric triplet frequency tensor and dims (V, E, V)
        self._triplet_frequency_tensor = None
        self._triplet_tensor_dims = None
        # Sparse triplet stats (linear_key -> count) and cached key/value tensors
        self._triplet_sparse_counts: Dict[int, int] = {}
        self._triplet_sparse_keys = None  # torch.Tensor[int64], sorted
        self._triplet_sparse_vals = None  # torch.Tensor[int32]
        # Parallelism and stats config
        self.stats_parallel_enabled: bool = True
        self.parallel_num: int = max(os.cpu_count()//2, 1)
        self.enable_string_stats: bool = False  # Disabled by default
    
    # ==================== Unified interface ====================
    
    def initialize_with_dataset(self, dataset_loader: BaseDataLoader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """
        Initialize serializer with dataset.
        
        Args:
            dataset_loader: Dataset loader providing node/edge token mappings
            graph_data_list: Graph data list for collecting statistics (optional)
        """
        self._dataset_loader = dataset_loader
        self._dataset_stats = {}
        # self._dataset_loader.load_data()
        
        # Call subclass-specific initialization
        self._initialize_serializer(dataset_loader, graph_data_list)
        
        self.most_frequent_edge_type = self._dataset_loader.get_most_frequent_edge_type()
        
        self._initialized = True
        logger.info(f"{self.name} serializer initialized")
    
    def serialize(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """
        Serialize a single graph.
        
        Args:
            graph_data: Graph data dict containing dgl_graph etc.
            **kwargs: Additional serialization parameters
            
        Returns:
            SerializationResult
        """
        assert self._initialized, f"{self.name} not initialized; call initialize_with_dataset first"
        
        assert 'dgl_graph' in graph_data and graph_data['dgl_graph'] is not None, "Graph data missing valid dgl_graph field"
        
        dgl_graph = graph_data['dgl_graph']
        assert dgl_graph.num_nodes() > 0, "Graph has no nodes, cannot serialize"
        
        # Set thread-local edge ID mapping to avoid concurrency conflicts
        setattr(self._thread_local, 'current_edge_id_mapping', self._build_edge_id_mapping(dgl_graph))
        
        # Delegate to subclass
        return self._serialize_single_graph(graph_data, **kwargs)
    
    def multiple_serialize(self, graph_data: Dict[str, Any], num_samples: int = 1, *, parallel: bool = False,**kwargs) -> SerializationResult:
        """
        Serialize a single graph multiple times with different start nodes.
        
        Args:
            graph_data: Graph data dict containing dgl_graph etc.
            num_samples: Number of samples (different starting nodes)
            **kwargs: Additional serialization parameters
            
        Returns:
            SerializationResult with multiple sequences
        """
        if 'dgl_graph' not in graph_data or graph_data['dgl_graph'] is None:
            raise ValueError("Graph data missing valid dgl_graph field")
        
        dgl_graph = graph_data['dgl_graph']
        
        if dgl_graph.num_nodes() == 0:
            raise ValueError("Graph has no nodes, cannot serialize")
        
        # Create global ID mapping
        id_mapping = GlobalIDMapping(dgl_graph)
        total_nodes = id_mapping.get_total_nodes()
        
        
        # Cap samples at node count
        actual_samples = min(num_samples, total_nodes)
        
        token_sequences: List[List[int]] = []
        element_sequences: List[List[str]] = []
        
        # Select different start nodes
        if actual_samples == 1:
            start_nodes = [0]
        else:
            # Uniform distribution of start nodes
            if total_nodes == 1:
                start_nodes = [0] * actual_samples
            else:
                step = max(1, total_nodes // actual_samples)
                start_nodes = [(i * step) % total_nodes for i in range(actual_samples)]
        
        # Execute multiple serializations
        if not parallel or actual_samples <= 1:
            for start_node in start_nodes:
                result = self.serialize(graph_data, start_node=start_node, **kwargs)
                if result and result.get_sequence_count() > 0:
                    first_token_seq, first_element_seq = result.get_sequence(0)
                    if first_token_seq:
                        token_sequences.append(first_token_seq)
                        element_sequences.append(first_element_seq)
        else:
            # Parallel: same instance + thread-local state to avoid shared data conflicts
            futures = []  # type: list
            with ThreadPoolExecutor(max_workers=self.parallel_num) as executor:
                for start_node in start_nodes:
                    def _make_task(sn: int):
                        def _task() -> Tuple[List[int], List[str]]:
                            res = self.serialize(graph_data, start_node=sn, **kwargs)
                            ts, es = res.get_sequence(0)
                            return ts, es
                        return _task
                    futures.append(executor.submit(_make_task(start_node)))

                for fut in futures:
                    ts, es = fut.result()
                    token_sequences.append(ts)
                    element_sequences.append(es)
        
        if not token_sequences:
            return SerializationResult([[]], [[]], id_mapping)
        
        return SerializationResult(token_sequences, element_sequences, id_mapping)
    
    def batch_serialize(self, graph_data_list: List[Dict[str, Any]], desc: str = None, *, parallel: bool = True, **kwargs) -> List[SerializationResult]:
        """
        Batch serialize multiple graphs.
        
        When parallel=True, uses multiprocess (fork) only; no thread path.
        
        Args:
            graph_data_list: List of graph data dicts
            desc: Progress bar description
            **kwargs: Additional serialization parameters
            
        Returns:
            List[SerializationResult]
        """
        if desc is None:
            desc = f"{self.name} batch serialize"
        
        if not parallel:
            results: List[SerializationResult] = []
            success_count = 0
            with tqdm(total=len(graph_data_list), desc=desc,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for i, graph_data in enumerate(graph_data_list):
                    result = self.serialize(graph_data, **kwargs)
                    results.append(result)
                    success_count += 1
                    pbar.set_postfix({'ok': success_count, 'fail': 0})
                    pbar.update(1)
            return results
        else:
            # Parallel: multiprocess (fork), strictly ordered
            n = len(graph_data_list)
            if n == 0:
                return []
            workers = min(self.parallel_num, n)
            # Split into worker chunks
            chunk_sizes = [(n // workers) + (1 if i < (n % workers) else 0) for i in range(workers)]
            indices = []
            start = 0
            for sz in chunk_sizes:
                indices.append((start, start + sz))
                start += sz

            results: List[Optional[SerializationResult]] = [None] * n

            # Always use fork context (even if outer context is spawn)
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                raise RuntimeError("Multiprocess serialization requires fork (Linux/posix only)")
            global _MP_GLOBAL_SERIALIZER, _MP_GRAPH_DATA_LIST
            _MP_GLOBAL_SERIALIZER = self
            _MP_GRAPH_DATA_LIST = graph_data_list
            with ctx.Pool(processes=workers, initializer=_mp_init_worker, initargs=(self, graph_data_list)) as pool:
                for pairs in pool.imap_unordered(_mp_process_range, [(lo, hi, kwargs) for (lo, hi) in indices if lo < hi], chunksize=1):
                    for i, res in pairs:
                        results[i] = res
            return [r for r in results if r is not None]
    
    def batch_multiple_serialize(self, graph_data_list: List[Dict[str, Any]], num_samples: int = 1, desc: str = None, *, parallel: bool = True, **kwargs) -> List[SerializationResult]: 
        """
        Batch multiple-serialize: serialize each graph multiple times.
        
        Args:
            graph_data_list: List of graph data dicts
            num_samples: Number of samples per graph
            desc: Progress bar description
            **kwargs: Additional serialization parameters
            
        Returns:
            List[SerializationResult]
        """
        if desc is None:
            desc = f"{self.name} batch multiple-serialize"
        
        if not parallel:
            results: List[SerializationResult] = []
            success_count = 0
            with tqdm(total=len(graph_data_list), desc=desc,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for i, graph_data in enumerate(graph_data_list):
                    result = self.multiple_serialize(graph_data, num_samples=num_samples, **kwargs)
                    results.append(result)
                    success_count += 1
                    pbar.set_postfix({'ok': success_count, 'fail': 0})
                    pbar.update(1)
            return results
        else:
            # Parallel: multiprocess (fork), strictly ordered
            n = len(graph_data_list)
            if n == 0:
                return []
            workers = min(self.parallel_num, n)

            # Split into worker chunks
            chunk_sizes = [(n // workers) + (1 if i < (n % workers) else 0) for i in range(workers)]
            indices = []
            start = 0
            for sz in chunk_sizes:
                indices.append((start, start + sz))
                start += sz

            results: List[Optional[SerializationResult]] = [None] * n

            # Always use fork context
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                raise RuntimeError("Multiprocess serialization requires fork (Linux/posix only)")

            global _MP_GLOBAL_SERIALIZER, _MP_GRAPH_DATA_LIST
            _MP_GLOBAL_SERIALIZER = self
            _MP_GRAPH_DATA_LIST = graph_data_list

            with ctx.Pool(processes=workers, initializer=_mp_init_worker, initargs=(self, graph_data_list)) as pool:
                it_args = [(lo, hi, num_samples, kwargs) for (lo, hi) in indices if lo < hi]
                for pairs in pool.imap_unordered(_mp_multiple_process_range, it_args, chunksize=1):
                    for i, res in pairs:
                        results[i] = res

            return [r for r in results if r is not None]
    
    # ==================== Abstract methods (subclass must implement) ====================
    
    @abstractmethod
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """
        Subclass initialization logic.
        
        Args:
            dataset_loader: Dataset loader
            graph_data_list: Graph data list for statistics collection
        """
        pass
    
    @abstractmethod
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """
        Subclass single-graph serialization logic.
        
        Args:
            graph_data: Graph data
            **kwargs: Additional serialization parameters
            
        Returns:
            SerializationResult
        """
        pass
    
    # ==================== Utility methods ====================
    
    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        """
        Get token IDs for a single node.
        
        Args:
            graph: DGL graph
            node_id: Node ID
            
        Returns:
            List[int]: Node token IDs
        """
        assert self._initialized, "Serializer not initialized"
          
        return self._dataset_loader.get_node_token(graph, node_id, ntype)
    
    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        """
        Get token IDs for a single edge.
        
        Args:
            graph: DGL graph
            edge_id: Edge ID
            
        Returns:
            List[int]: Edge token IDs
        """
        assert self._initialized, "Serializer not initialized"
        
        return self._dataset_loader.get_edge_token(graph, edge_id, etype)
    
    
    def tokens_to_string(self, token_ids: List[int]) -> str:
        """
        Convert token ID sequence to string representation.
        
        Args:
            token_ids: Token ID sequence
            
        Returns:
            String representation
        """
        assert self._initialized, "Serializer not initialized"
        assert hasattr(self._dataset_loader, 'get_token_readable'), "Dataset loader missing get_token_readable method"
        
        return ''.join([self._dataset_loader.get_token_readable(token_id) for token_id in token_ids])
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dict[str, Any]: Statistics dict
        """
        return self._dataset_stats.copy()
    
    def is_initialized(self) -> bool:
        """Check if serializer is initialized."""
        return self._initialized
    
    # ==================== Compatibility methods ====================
    
    def deserialize(self, result: SerializationResult, **kwargs) -> dgl.DGLGraph:
        """Reconstruct graph from serialization result (compatibility).
        
        Args:
            result: Serialization result
            **kwargs: Additional deserialization parameters
            
        Returns:
            dgl.DGLGraph: Reconstructed graph
        """
        raise NotImplementedError("This serializer does not support deserialization")
    
    # ====================== Shared serialization methods ======================
    
    
    def _validate_graph_data(self, mol_data: Dict[str, Any],) -> dgl.DGLGraph:
        """
        Validate graph data and return DGL graph.
        
        Args:
            mol_data: Graph data dict
            
        Returns:
            dgl.DGLGraph: Validated DGL graph
        """
        assert 'dgl_graph' in mol_data, "Input data missing 'dgl_graph' field"
        
        assert mol_data['dgl_graph'].num_nodes() > 0, "Graph has no nodes, cannot serialize"
        
        return mol_data['dgl_graph']
    
    
    def _get_all_edges_from_heterograph(self, dgl_graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all edges from graph, compatible with both homogeneous and heterogeneous.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (src_nodes, dst_nodes)
        """
        if len(dgl_graph.etypes) == 1:
            # Homogeneous or single edge type
            return dgl_graph.edges()
        else:
            # Heterogeneous: iterate all edge types
            all_src = []
            all_dst = []
            
            for etype in dgl_graph.etypes:
                src, dst = dgl_graph.edges(etype=etype)
                all_src.append(src)
                all_dst.append(dst)
            
            # Merge all edges
            if all_src:
                import torch
                combined_src = torch.cat(all_src, dim=0)
                combined_dst = torch.cat(all_dst, dim=0) 
                return combined_src, combined_dst
            else:
                raise ValueError("Graph has no edges")
    
    def _build_edge_id_mapping(self, dgl_graph: dgl.DGLGraph) -> Dict[Tuple[int, int], int]:
        """
        Build edge ID mapping for fast lookup.
        
        Args:
            dgl_graph: DGL graph
            
        Returns:
            Dict[Tuple[int, int], int]: (src, dst) -> edge_id mapping
        """
        edge_mapping = {}
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        dgl_graph.edge_ids
        # Use index position as edge ID directly
        for i, (src, dst) in enumerate(zip(src_nodes.numpy(), dst_nodes.numpy())):
            src, dst = int(src), int(dst)
            edge_mapping[(src, dst)] = i
        
        return edge_mapping
    
    def _get_edge_id(self, dgl_graph: dgl.DGLGraph, src: int, dst: int) -> int:
        """
        Get edge ID in a DGL graph.
        
        Args:
            dgl_graph: DGL graph
            src: Source node
            dst: Destination node
            
        Returns:
            int: Edge ID, or -1 if not found
        """
        # Prefer thread-local cached edge ID mapping
        thread_mapping = getattr(self._thread_local, 'current_edge_id_mapping', None)
        if thread_mapping is not None:
            edge_id = thread_mapping.get((src, dst))
            assert edge_id is not None, f"Edge ID not found: ({src}, {dst})"
            return edge_id
        # Fallback: instance-level mapping (not recommended under parallelism)
        if self._current_edge_id_mapping is not None:
            edge_id = self._current_edge_id_mapping.get((src, dst))
            assert edge_id is not None, f"Edge ID not found: ({src}, {dst})"
            return edge_id
        
        # Fallback to DGL API
        try:
            edge_ids = dgl_graph.edge_ids(src, dst)
            return int(edge_ids)
        except Exception as e:
            raise ValueError(f"Failed to get edge ID ({src}->{dst}): {e}")
      
    def _get_edge_type(self, dgl_graph: dgl.DGLGraph, src: int, dst: int) -> str:
        """
        Get edge type in a DGL graph.
        
        Args:
            dgl_graph: DGL graph
            src: Source node
            dst: Destination node
            
        Returns:
            str: Edge type
        """
        edge_id = self._get_edge_id(dgl_graph, src, dst)
        edge_type = self._dataset_loader.get_edge_type(dgl_graph, edge_id=edge_id)
        return edge_type
      
    def _get_node_type(self, dgl_graph: dgl.DGLGraph, node_id: int) -> str:
        """Get node type in a DGL graph."""
        node_type = self._dataset_loader.get_node_type(dgl_graph, node_id)
        return node_type
    
    
    def _build_adjacency_list_from_dgl(self, dgl_graph: dgl.DGLGraph) -> List[List[int]]:
        """
        Build adjacency list from DGL graph.
        
        Each DGL edge represents one traversable path. For bidirectional edges
        (u,v) and (v,u), both directions appear in the adjacency list.
        
        Args:
            dgl_graph: DGL graph
            
        Returns:
            List[List[int]]: Adjacency list, one entry per DGL edge
        """
        num_nodes = dgl_graph.num_nodes()
        adj_list = [[] for _ in range(num_nodes)]
        
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        
        # Build adjacency list directly from DGL edges
        for src, dst in zip(src_nodes.numpy(), dst_nodes.numpy()):
            src, dst = int(src), int(dst)
            adj_list[src].append(dst)
        
        return adj_list
      
    def _convert_path_to_tokens(self, node_path: List[int], mol_data: Dict[str, Any]) -> Tuple[List[int], List[str]]:
        """Convert node path to token and element sequences (tensorized)."""
        if not node_path:
            return [], []

        import torch as _torch

        dgl_graph = mol_data['dgl_graph']

        # -------- Node tokens (full graph tensor + index_select) --------
        node_ids_tensor: torch.Tensor = _torch.as_tensor([int(n) for n in node_path], dtype=_torch.long)
        node_tok_2d: torch.Tensor = self._dataset_loader.get_graph_node_token_ids(dgl_graph)  # [N, Dn]
        selected_node_tok: torch.Tensor = node_tok_2d.index_select(0, node_ids_tensor)  # [P, Dn]

        # -------- Edge ids (full graph + batch compute + optional mask) --------
        P = int(node_ids_tensor.shape[0])
        include_edges = bool(getattr(self, 'include_edge_tokens', True) and P > 1)
        if include_edges:
            src_tensor = node_ids_tensor[:-1]
            dst_tensor = node_ids_tensor[1:]
            try:
                edge_ids_tensor_full = dgl_graph.edge_ids(src_tensor, dst_tensor)  # [P-1]
            except Exception as e:
                # Fallback for multi-edges or heterogeneous graphs
                # Query edge IDs one by one, take the first match
                import torch as _torch
                eid_list: List[int] = []
                src_list = src_tensor.tolist()
                dst_list = dst_tensor.tolist()
                for u, v in zip(src_list, dst_list):
                    try:
                        # Single-pair query: DGL returns 1D tensor for multi-edges; take first
                        eids = dgl_graph.edge_ids(int(u), int(v), return_uv=False)
                        if hasattr(eids, 'numel') and eids.numel() > 1:
                            eid_list.append(int(eids[0].item()))
                        else:
                            eid_list.append(int(eids.item()))
                    except Exception:
                        # Last resort: build (u,v)->eid mapping
                        mapping = self._build_edge_id_mapping(dgl_graph)
                        eid = mapping.get((int(u), int(v)))
                        if eid is None:
                            raise ValueError(f"Edge ID not found: (src={u}, dst={v})")
                        eid_list.append(int(eid))
                edge_ids_tensor_full = _torch.as_tensor(eid_list, dtype=_torch.long)

            # Mask: omit most frequent edge type
            keep_mask = None
            if getattr(self, 'omit_most_frequent_edge', False):
                edge_type_ids_all = self._dataset_loader.get_graph_edge_type_ids(dgl_graph)  # [E]
                edge_type_ids_seq = edge_type_ids_all.index_select(0, edge_ids_tensor_full)  # [P-1]
                mf_name = getattr(self, 'most_frequent_edge_type', None)
                if mf_name is not None:
                    mf_id = int(self._dataset_loader.get_edge_type_id_by_name(mf_name))
                    keep_mask = edge_type_ids_seq.ne(mf_id)

            if keep_mask is None:
                edge_ids_tensor = edge_ids_tensor_full
                keep_mask_bool = _torch.ones(edge_ids_tensor_full.shape[0], dtype=_torch.bool)
            else:
                edge_ids_tensor = edge_ids_tensor_full[keep_mask]
                keep_mask_bool = keep_mask
            Ke = int(edge_ids_tensor.shape[0])
        else:
            edge_ids_tensor = _torch.empty((0,), dtype=_torch.long)
            keep_mask_bool = _torch.empty((0,), dtype=_torch.bool)
            Ke = 0

        # -------- Node/edge tokens tensorized retrieval --------
        Dn = int(selected_node_tok.shape[1])
        if include_edges:
            edge_tok_2d_all = self._dataset_loader.get_graph_edge_token_ids(dgl_graph)  # [E, De]
            # Reuse precomputed edge_ids_tensor_full
            selected_edge_tok_all = edge_tok_2d_all.index_select(0, edge_ids_tensor_full)  # [P-1, De]
            if keep_mask_bool.numel() > 0 and keep_mask_bool.sum().item() != keep_mask_bool.numel():
                selected_edge_tok = selected_edge_tok_all[keep_mask_bool]
            else:
                selected_edge_tok = selected_edge_tok_all
            De = int(selected_edge_tok.shape[1]) if selected_edge_tok.numel() > 0 else int(edge_tok_2d_all.shape[1])
        else:
            # No edges
            De = 1
            selected_edge_tok = _torch.empty((0, De), dtype=_torch.long)

        # -------- Output order: interleaved (node_i then edge_i,i+1) --------
        # Fast path: Dn==1 and (no edges or De==1)
        if Dn == 1 and (not include_edges or De == 1):
            P = int(selected_node_tok.shape[0])
            L = P + Ke
            seq = _torch.empty(L, dtype=_torch.long)
            # Node positions: i + prefix sum of kept edges
            if include_edges and keep_mask_bool.numel() > 0:
                shift = _torch.cat([_torch.zeros(1, dtype=_torch.long), keep_mask_bool.cumsum(0)])
            else:
                shift = _torch.zeros(P, dtype=_torch.long)
            pos_nodes = _torch.arange(P, dtype=_torch.long) + shift
            node_flat = selected_node_tok.view(-1).to(_torch.long)
            seq[pos_nodes] = node_flat
            if include_edges and Ke > 0:
                pos_edges = pos_nodes[:-1][keep_mask_bool] + 1
                seq[pos_edges] = selected_edge_tok.view(-1).to(_torch.long)
            token_list = seq.tolist()
            element_list = [""] * len(token_list)
            return token_list, element_list

        # General path: arbitrary Dn/De
        out_chunks = []
        node_rows = selected_node_tok.tolist()
        edge_rows = selected_edge_tok.tolist() if (include_edges and Ke>0) else []
        e_ptr = 0
        for i in range(len(node_rows)):
            out_chunks.extend(node_rows[i])
            if i < len(node_rows)-1 and (include_edges and Ke>0):
                if bool(keep_mask_bool[i].item()):
                    out_chunks.extend(edge_rows[e_ptr])
                    e_ptr += 1
        token_list = out_chunks
        element_list = [""] * len(token_list)
        return token_list, element_list
      
    def _convert_path_to_tokens_old(self, node_path: List[int], mol_data: Dict[str, Any]) -> Tuple[List[int], List[str]]:
        """Convert node path to token and element sequences (legacy)
        
        Returns:
            Tuple[List[int], List[str]]: (token_sequence, element_sequence)
        """
        
        if not node_path:
            return [], []
        dgl_graph = mol_data['dgl_graph']
        
        result_token_ids = []
        result_element_ids = []
        
        # Build interleaved node/edge sequence
        for i, node in enumerate(node_path):
            # Add node tokens and element info
            node_tokens = self.get_node_token(dgl_graph, node)
            result_token_ids.extend(node_tokens)
            
            # Add element info for each token
            if len(node_tokens) > 1:
                result_element_ids.append(f"START_NODE_{node}")
                for j, token in enumerate(node_tokens[1:-1]):
                      result_element_ids.append(f"node_{node}_dim_{j}")
                result_element_ids.append(f"END_NODE_{node}")
            else:
                result_element_ids.append(f"node_{node}")
                
            if not self.include_edge_tokens:
                continue
            
            # If not the last node, add edge tokens
            if i >= len(node_path) - 1:
              continue
            
            next_node = node_path[i + 1]
            edge_id = self._get_edge_id(dgl_graph, node, next_node)
            edge_tokens = self.get_edge_token(dgl_graph, edge_id)
            
            if not(self.omit_most_frequent_edge and 
                   self._dataset_loader.get_edge_type(dgl_graph, edge_id) == self.most_frequent_edge_type):
                result_token_ids.extend(edge_tokens)
                if len(edge_tokens) > 1:
                    result_element_ids.append(f"START_EDGE_{edge_id}")
                    for j, token in enumerate(edge_tokens[1:-1]):
                        result_element_ids.append(f"edge_{edge_id}_dim_{j}")
                    result_element_ids.append(f"END_EDGE_{edge_id}")
                else:
                    result_element_ids.append(f"edge_{edge_id}")
       
        
        return result_token_ids, result_element_ids
      
    #===================== Statistics collection ======================
    
    
    def _collect_statistics_from_graphs(self, graph_data_list: List[Dict[str, Any]]) -> None:
        """Collect global statistics from graphs.
        
        - Numeric triplet frequency hot path only; string stats optional (off by default)
        - Two phases: global dimension inference, then sharded counting
        - Supports multiprocess sharding (fork), normalized after merge
        - Dimensions inferred from actual max type IDs
        """
        import torch as _torch

        N = len(graph_data_list)
        if N == 0:
            self._triplet_frequency_tensor = _torch.zeros((0, 0, 0), dtype=_torch.long)
            self._triplet_tensor_dims = (0, 0, 0)
            self.triplet_frequencies = {}
            self.two_hop_frequencies = {}
            self.statistics_collected = True
            return

        # Phase 1: global type-space dimension inference (max id + 1)
        if self.stats_parallel_enabled:
            if mp.get_start_method(allow_none=True) not in ("fork", None):
                raise RuntimeError("Parallel stats requires fork platform (Linux/posix)")
            W = max(1, int(self.parallel_num))
            W = min(W, N)
            chunk_sizes = [(N // W) + (1 if i < (N % W) else 0) for i in range(W)]
            ranges = []
            s = 0
            for sz in chunk_sizes:
                if sz > 0:
                    ranges.append((s, s + sz))
                s += sz
            global _MP_STATS_SERIALIZER, _MP_STATS_GRAPH_LIST
            _MP_STATS_SERIALIZER = self
            _MP_STATS_GRAPH_LIST = graph_data_list
            max_v = 0
            max_e = 0
            with mp.Pool(processes=W, initializer=_mp_stats_init, initargs=(self, graph_data_list)) as pool:
                for mv, me in pool.imap_unordered(_mp_stats_max_range, ranges, chunksize=1):
                    if mv > max_v:
                        max_v = mv
                    if me > max_e:
                        max_e = me
            Vdim0 = int(max_v)
            Edim = int(max_e)
        else:
            Vdim0 = 0
            Edim = 0
            for i in range(N):
                g = graph_data_list[i].get('dgl_graph')
                if g is None:
                    continue
                node_type_ids = self._dataset_loader.get_graph_node_type_ids(g)
                edge_type_ids = self._dataset_loader.get_graph_edge_type_ids(g)
                if node_type_ids.numel() > 0:
                    Vdim0 = max(Vdim0, int(node_type_ids.max().item()) + 1)
                if edge_type_ids.numel() > 0:
                    Edim = max(Edim, int(edge_type_ids.max().item()) + 1)

        if Vdim0 <= 0 or Edim <= 0:
            raise ValueError("Cannot infer type-space dimensions. Check dataset type ID features.")

        Vdim1 = Vdim0

        # Phase 2: sparse counting (no dense 3D tensor)
        sparse_counts: Dict[int, int] = {}
        if self.stats_parallel_enabled:
            if mp.get_start_method(allow_none=True) not in ("fork", None):
                raise RuntimeError("Parallel stats requires fork platform (Linux/posix)")
            W = max(1, int(self.parallel_num))
            W = min(W, N)
            chunk_sizes = [(N // W) + (1 if i < (N % W) else 0) for i in range(W)]
            ranges = []
            s = 0
            for sz in chunk_sizes:
                if sz > 0:
                    ranges.append((s, s + sz))
                s += sz
            # Reuse global refs declared above
            _MP_STATS_SERIALIZER = self
            _MP_STATS_GRAPH_LIST = graph_data_list
            with mp.Pool(processes=W, initializer=_mp_stats_init, initargs=(self, graph_data_list)) as pool:
                args_list = [(lo, hi, Vdim0, Edim) for (lo, hi) in ranges]
                for part in pool.imap_unordered(_mp_stats_count_range_sparse, args_list, chunksize=1):
                    for k, v in part.items():
                        sparse_counts[k] = sparse_counts.get(k, 0) + int(v)
        else:
            # Sequential sparse counting
            for i in range(N):
                g = graph_data_list[i].get('dgl_graph')
                if g is None:
                    continue
                if self.enable_string_stats:
                    _ = self._extract_all_statistics(g)  # optional
                node_type_ids = self._dataset_loader.get_graph_node_type_ids(g)
                edge_type_ids = self._dataset_loader.get_graph_edge_type_ids(g)
                if edge_type_ids.numel() == 0:
                    continue
                src, dst = self._get_all_edges_from_heterograph(g)
                src_t = node_type_ids.index_select(0, src).long()
                dst_t = node_type_ids.index_select(0, dst).long()
                et_t = edge_type_ids.long()
                if src_t.numel() == 0:
                    continue
                if src_t.max() >= Vdim0 or dst_t.max() >= Vdim1 or et_t.max() >= Edim:
                    raise ValueError("Type ID out of bounds during stats counting. Check dimension inference.")
                lin = (src_t * (Edim * Vdim1) + et_t * Vdim1 + dst_t).to(_torch.long)
                if lin.numel() == 0:
                    continue
                lin_sorted, _ = _torch.sort(lin)
                uniq, cts = _torch.unique_consecutive(lin_sorted, return_counts=True)
                for k, v in zip(uniq.tolist(), cts.tolist()):
                    sparse_counts[k] = sparse_counts.get(k, 0) + int(v)

        # Save dims and sparse key/value cache; no dense tensor
        self._triplet_tensor_dims = (Vdim0, Edim, Vdim1)
        self._triplet_frequency_tensor = None
        if len(sparse_counts) == 0:
            self._triplet_sparse_counts = {}
            self._triplet_sparse_keys = None
            self._triplet_sparse_vals = None
        else:
            keys = _torch.tensor(sorted(sparse_counts.keys()), dtype=_torch.long)
            vals = _torch.tensor([sparse_counts[int(k)] for k in keys.tolist()], dtype=_torch.int32)
            self._triplet_sparse_counts = sparse_counts
            self._triplet_sparse_keys = keys
            self._triplet_sparse_vals = vals

        # Align with legacy interface (string tables empty; fields preserved)
        self.triplet_frequencies = {}
        self.two_hop_frequencies = {}
        self.statistics_collected = True
        logger.info("Sparse stats collected (parallel=%s): V=%d, E=%d, nonzero=%d, total=%d",
                    str(self.stats_parallel_enabled), Vdim0, Edim, len(self._triplet_sparse_counts), int(sum(self._triplet_sparse_counts.values())))
        
    def _extract_all_statistics(self, dgl_graph: dgl.DGLGraph) -> Tuple[Dict[Tuple[str, str, str], int], Dict[Tuple[str, str, str, str, str], int]]:
        """Extract all statistics at once: triplets and two-hop paths."""
        
        # Build adjacency table (build once, use many times)
        adjacency = defaultdict(list)
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        # Use thread-local mapping to avoid cross-thread sharing
        setattr(self._thread_local, 'current_edge_id_mapping', self._build_edge_id_mapping(dgl_graph))
        
        for i, (src, dst) in enumerate(zip(src_nodes.numpy(), dst_nodes.numpy())):
            src, dst = int(src), int(dst)
            edge_type = self._get_edge_type(dgl_graph, src, dst)
            adjacency[src].append((dst, edge_type))
            adjacency[dst].append((src, edge_type))  # undirected: add both directions
        
        # Collect triplet statistics
        triples = defaultdict(int)
        for src in adjacency:
            for dst, edge_type in adjacency[src]:
                src_type, dst_type = self._get_node_type(dgl_graph, src), self._get_node_type(dgl_graph, dst)
                # Each edge contributes one triplet
                triples[(src_type, edge_type, dst_type)] += 1
                # triples[(dst_type, edge_type, src_type)] += 1
        
        # Two-hop path stats (disabled by design; placeholder comments kept)
        # two_hop_paths = defaultdict(int)
        #
        # for node1 in range(dgl_graph.num_nodes()):
        #     node1_type = self._get_node_type(dgl_graph, node1)
        #     # Iterate node1 neighbors
        #     for node2, edge1_type in adjacency[node1]:
        #         node2_type = self._get_node_type(dgl_graph, node2)
        #         # Iterate node2 neighbors for two-hop path
        #         for node3, edge2_type in adjacency[node2]:
        #             node3_type = self._get_node_type(dgl_graph, node3)
        #             # Record two-hop path (ensure deterministic handling if enabled)
        #             path = (node1_type, edge1_type, node2_type, edge2_type, node3_type)
        #             two_hop_paths[path] += 1
        #
        # return dict(triples), dict(two_hop_paths)
        # Return triplet stats; two_hop is empty dict by design
        return dict(triples), {}
    
    def _calculate_edge_weights(self, dgl_graph: dgl.DGLGraph) -> Dict[Tuple[int, int], float]:
        """
        Compute edge weights from DGL graph. Priority: sparse stats > dense tensor > string freq table.
        """
        import torch as _torch
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        node_type_ids = self._dataset_loader.get_graph_node_type_ids(dgl_graph)
        edge_type_ids = self._dataset_loader.get_graph_edge_type_ids(dgl_graph)

        src_t = node_type_ids.index_select(0, src_nodes)
        dst_t = node_type_ids.index_select(0, dst_nodes)
        et_t = edge_type_ids

        # 1) Prefer sparse stats
        if self._triplet_sparse_keys is not None and self._triplet_sparse_vals is not None:
            V0, E0, V1 = self._triplet_tensor_dims
            if src_t.numel() == 0:
                return {}
            if src_t.max().item() >= V0 or dst_t.max().item() >= V1 or et_t.max().item() >= E0:
                raise ValueError("Type ID exceeds stats dimensions. Rebuild stats with full dataset.")
            lin = (src_t.long() * (E0 * V1) + et_t.long() * V1 + dst_t.long())
            # searchsorted lookup
            pos = _torch.searchsorted(self._triplet_sparse_keys, lin)
            # Equality mask (avoid false hits for missing keys)
            valid = (pos < self._triplet_sparse_keys.shape[0]) & (self._triplet_sparse_keys.index_select(0, pos) == lin)
            counts = _torch.zeros_like(lin, dtype=_torch.int32)
            if valid.any():
                counts[valid] = self._triplet_sparse_vals.index_select(0, pos[valid])
            weights = _torch.log10(counts.clamp(min=1).to(_torch.float32))
            return {(int(s), int(d)): float(w) for s, d, w in zip(src_nodes.tolist(), dst_nodes.tolist(), weights.tolist())}

        # 2) Fallback: dense tensor
        if self._triplet_frequency_tensor is not None:
            V0, E0, V1 = self._triplet_tensor_dims
            if src_t.numel() == 0:
                return {}
            if src_t.max().item() >= V0 or dst_t.max().item() >= V1 or et_t.max().item() >= E0:
                raise ValueError("Type ID exceeds freq tensor dimensions. Rebuild stats with full dataset.")
            freq_vals = self._triplet_frequency_tensor[src_t.long(), et_t.long(), dst_t.long()].clamp_min_(1).to(_torch.float32)
            weights = _torch.log10(freq_vals)
            return {(int(s), int(d)): float(w) for s, d, w in zip(src_nodes.tolist(), dst_nodes.tolist(), weights.tolist())}

        # 3) Last resort: string frequency table
        edge_weights: Dict[Tuple[int, int], float] = {}
        for s, d, etid in zip(src_nodes.tolist(), dst_nodes.tolist(), et_t.tolist()):
            s_name = self._get_node_type(dgl_graph, int(s))
            d_name = self._get_node_type(dgl_graph, int(d))
            e_name = self._get_edge_type(dgl_graph, int(s), int(d))
            triplet = (s_name, e_name, d_name)
            if triplet not in self.triplet_frequencies:
                raise ValueError(f"Missing stats: triplet {triplet} not in frequency table. Provide more graphs during initialization.")
            edge_weights[(int(s), int(d))] = log10(self.triplet_frequencies[triplet])
        return edge_weights
        V0, E0, V1 = self._triplet_tensor_dims
        if src_t.numel() == 0:
            return {}
        if src_t.max().item() >= V0 or dst_t.max().item() >= V1 or et_t.max().item() >= E0:
            raise ValueError("Type ID exceeds freq tensor dimensions. Rebuild stats with full dataset.")

        freq_vals = self._triplet_frequency_tensor[src_t.long(), et_t.long(), dst_t.long()].clamp_min_(1).to(_torch.float32)
    def _split_connected_components(self, dgl_graph: dgl.DGLGraph) -> List[dgl.DGLGraph]:
        """
        Split DGL graph into connected component subgraphs.
        
        Useful for serialization algorithms that require connected graphs
        (e.g. Chinese Postman, Eulerian circuit).
        
        Args:
            dgl_graph: Input DGL graph
            
        Returns:
            List[dgl.DGLGraph]: Subgraphs sorted by component size (descending)
        """
        import networkx as nx
        
        # Convert to NetworkX for connected components
        nx_graph = nx.Graph()
        
        # Add all nodes
        for i in range(dgl_graph.num_nodes()):
            nx_graph.add_node(i)
        
        # Add all edges
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        for src, dst in zip(src_nodes.numpy(), dst_nodes.numpy()):
            src, dst = int(src), int(dst)
            nx_graph.add_edge(src, dst)
        
        # Get connected components
        components = list(nx.connected_components(nx_graph))
        
        # Sort by component size descending
        components.sort(key=len, reverse=True)
        
        # Create subgraph for each component
        subgraphs = []
        for component in components:
            if len(component) == 0:
                continue
                
            # Create subgraph; DGL auto-remaps node IDs
            subgraph = dgl.node_subgraph(dgl_graph, list(component))
            subgraphs.append(subgraph)
        
        return subgraphs
    
    def _convert_dgl_to_networkx(self, dgl_graph: dgl.DGLGraph) -> nx.MultiGraph:
        """Convert DGL graph to NetworkX MultiGraph.
        
        Deduplicates bidirectional edges (u,v) and (v,u) into a single
        undirected edge.
        
        Args:
            dgl_graph: DGL graph
            
        Returns:
            nx.MultiGraph
        """
        
        G = nx.MultiGraph()
        
        # Add nodes
        G.add_nodes_from(range(dgl_graph.num_nodes()))
        
        # Get edge info
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        
        # Deduplicate: add each undirected edge once
        added_edges = set()
        edge_list = []
        
        for src, dst in zip(src_nodes.numpy(), dst_nodes.numpy()):
            src, dst = int(src), int(dst)
            # Canonical edge key (smaller node first)
            edge_key = (min(src, dst), max(src, dst))
            
            if edge_key not in added_edges:
                added_edges.add(edge_key)
                edge_list.append((src, dst))
                G.add_edge(src, dst, weight=1)
        
        return G