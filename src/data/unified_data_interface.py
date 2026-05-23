"""Unified Data Interface.
统一数据接口。

Design principles / 设计原则:
- Single data file + index-based splits. / 单数据文件 + 基于索引的分割。
- Provides a unified read interface for upper layers (serialization / BPE / training). / 为上层提供统一读取接口。
- Serialization and BPE cache management are internal details; data building must be triggered explicitly. / 序列化和BPE缓存管理为内部细节，数据构建需显式触发。
- Simplified API: two core methods get_sequences() and get_sequences_by_splits(). / 简化API：两个核心方法。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader
from src.data.base_loader import BaseDataLoader
from src.algorithms.serializer.serializer_factory import SerializerFactory
from src.utils.logger import get_logger

# from src.models.bert.vocab_manager import build_vocab_from_sequences  # lazy import to avoid circular deps
logger = get_logger(__name__)

@dataclass
class UnifiedDataInterface:
    config: ProjectConfig
    dataset: str
    # Optional preloaded cache (reused within the same process to avoid repeated IO)
    _preloaded_graphs: List[Dict[str, Any]] | None = None
    _preloaded_splits: Dict[str, List[int]] | None = None
    _loader: BaseDataLoader | None = None

    def _resolve_processed_dir(self) -> Path:
        return Path(self.config.processed_data_dir) / self.dataset


    def _load_split_indices(self) -> Dict[str, List[int]]:
        """Load split index files."""
        # Use project config data_dir to ensure we load real project data
        data_dir = Path(self.config.data_dir) / self.dataset
        
        train_path = data_dir / "train_index.json"
        val_path = data_dir / "val_index.json"
        test_path = data_dir / "test_index.json"
        
        splits = {}
        
        # Load train indices
        if train_path.exists():
            with open(train_path, 'r') as f:
                splits['train'] = json.load(f)
        else:
            raise FileNotFoundError(f"Train index file not found: {train_path}")
        
        # Load val indices
        if val_path.exists():
            with open(val_path, 'r') as f:
                splits['val'] = json.load(f)
        else:
            raise FileNotFoundError(f"Val index file not found: {val_path}")
        
        # Load test indices
        if test_path.exists():
            with open(test_path, 'r') as f:
                splits['test'] = json.load(f)
        else:
            raise FileNotFoundError(f"Test index file not found: {test_path}")
        
        return splits

    # ----------------------- Preload graphs/indices -----------------------
    def preload_graphs(self) -> None:
        """Preload and cache all graphs and split indices for reuse."""
        loader = self.get_dataset_loader()
        graphs, _ = loader.get_all_data_with_indices()
        self._loader=loader
        self._preloaded_graphs = graphs
        self._preloaded_splits = self._load_split_indices()

    def _get_serialization_cache_key(self) -> str:
        """Generate a config-based serialization cache key."""
        # Read multiple sampling settings from config
        ms = self.config.serialization.multiple_sampling
        use_multi = getattr(ms, 'enabled', False)
        num_realizations = getattr(ms, 'num_realizations', 1)
        
        if use_multi and num_realizations > 1:
            return f"multi_{num_realizations}"
        else:
            return "single"
    
    def _load_serialization_result(self, method: str) -> Dict[str, Any]:
        """Load serialization result from disk."""
        base = self._resolve_processed_dir()
        cache_key = self._get_serialization_cache_key()
        result_path = base / "serialized_data" / method / cache_key / "serialized_data.pickle"
        
        if not result_path.exists():
            raise FileNotFoundError(f"Serialization result not found: {result_path}")
        
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
            
        assert 'serialization_method' in data, "Serialization result missing 'serialization_method' field"
        assert data['serialization_method'] == method, f"Requested method '{method}' != saved method '{data['serialization_method']}'"
        
        return data

    # ----------------------- Build & persist (explicit trigger) -----------------------
    def _extract_properties_from_graphs(self, graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract property info (numeric / short strings) from raw graphs for caching."""
        try:
            import numpy as np  # local import
        except Exception:
            np = None  # not strictly required in test environments

        graph_structure_keys = {
            'dgl_graph', 'graph', 'edge_index', 'edge_attr', 'node_features', 'edge_features',
            'num_nodes', 'num_edges', 'adjacency', 'adjacency_matrix', 'node_attr',
            'smiles', 'mol', 'molecule', 'rdkit_mol',
            'id', 'index', 'idx', 'dataset_name', 'data_type',
            'node_ids', 'edge_ids', 'global_node_ids',
            'smiles_1', 'smiles_2', 'smiles_3', 'smiles_4'
        }

        properties: List[Dict[str, Any]] = []
        for item in graphs:
            prop: Dict[str, Any] = {}
            if isinstance(item, dict):
                if 'properties' in item and isinstance(item['properties'], dict):
                    prop.update(item['properties'])
                for k, v in item.items():
                    if k in graph_structure_keys:
                        continue
                    if isinstance(v, (int, float)):
                        prop[k] = float(v)
                    elif isinstance(v, str) and len(v) < 50:
                        prop[k] = v
                    elif np is not None and isinstance(v, (np.integer, np.floating)):
                        prop[k] = float(v)
            properties.append(prop)
        return properties

    def _build_and_persist_serialization(self, method: str) -> Path:
        """Build serialization result deterministically and persist when cache is missing."""
        # Get data
        if self._preloaded_graphs is not None:
            graphs = self._preloaded_graphs
            loader = self.get_dataset_loader()
        else:
            loader = self.get_dataset_loader()
            graphs, _ = loader.get_all_data_with_indices()

        # Initialize serializer
        serializer = SerializerFactory.create_serializer(method)
        serializer.initialize_with_dataset(loader, graphs)

        # Multiple sampling: decide whether to produce multiple variants per graph
        ms = self.config.serialization.multiple_sampling
        use_multi = bool(getattr(ms, 'enabled', False))
        num_realizations = int(getattr(ms, 'num_realizations', 1))

        sequences: List[List[int]] = []
        graph_ids: List[int] = []
        flattened_properties: List[Dict[str, Any]] = []

        # Extract properties (aligned with sequences)
        properties = self._extract_properties_from_graphs(graphs)

        if use_multi and num_realizations > 1:
            # Enable internal multi-process parallel (fork-only) with CPU core count as workers
            batch_results = serializer.batch_multiple_serialize(
                graphs,
                num_samples=num_realizations,
                desc=f"serialize-multi-{method}",
                parallel=True,
            )
            # Expand variants per graph
            for gid, res in enumerate(batch_results):
                if res is None:
                    raise ValueError(f"Serialization failed: graph {gid} returned None")
                if not hasattr(res, 'token_sequences'):
                    raise ValueError(f"Bad serialization result: graph {gid} missing token_sequences")
                if not res.token_sequences:
                    raise ValueError(f"Empty serialization result: graph {gid} token_sequences is empty")
                    
                for vid, seq in enumerate(res.token_sequences):
                    sequences.append(seq)
                    graph_ids.append(gid)
                    assert gid < len(properties), f"Property index out of range: graph {gid} >= {len(properties)}"
                    flattened_properties.append(properties[gid])
        else:
            # Single serialization
            batch_results = serializer.batch_serialize(graphs, desc=f"serialize-{method}")
            for gid, res in enumerate(batch_results):
                if res is None:
                    raise ValueError(f"Serialization failed: graph {gid} returned None")
                if not hasattr(res, 'token_sequences'):
                    raise ValueError(f"Bad serialization result: graph {gid} missing token_sequences")
                if not res.token_sequences:
                    raise ValueError(f"Empty serialization result: graph {gid} token_sequences is empty")
                    
                sequences.append(res.token_sequences[0])
                graph_ids.append(gid)
                if gid >= len(properties):
                    raise IndexError(f"Property index out of range: graph {gid} >= {len(properties)}")
                flattened_properties.append(properties[gid])

        # Write to disk
        cache_key = self._get_serialization_cache_key()
        out_dir = self._resolve_processed_dir() / "serialized_data" / method / cache_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "serialized_data.pickle"
        with out_path.open('wb') as f:
            pickle.dump({
                'sequences': sequences,
                'properties': flattened_properties,
                'serialization_method': method,
                'graph_ids': graph_ids,
            }, f)
        return out_path



    # ----------------------- BPE codebook & Transform -----------------------
    def get_bpe_codebook(self, method: str) -> Dict[str, Any]:
        """Read BPE codebook (managed by single/multi_<k> subdirectories)."""
        cache_key = self._get_serialization_cache_key()
        model_path = self.config.model_dir / "bpe" / self.dataset / method / cache_key / "bpe_codebook.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"BPE codebook not found: {model_path}")
            
        with model_path.open('rb') as f:
            data = pickle.load(f)
            
        assert isinstance(data, dict) and 'merge_rules' in data and 'vocab_size' in data, "Bad BPE codebook format: missing merge_rules or vocab_size"
            
        return {'merge_rules': data['merge_rules'], 'vocab_size': int(data['vocab_size'])}

    def get_bpe_encoder(self, method: str, *, encode_backend: str = 'cpp', **engine_kwargs):
        """Read codebook and return a ready-to-use BPEEngine."""
        from src.algorithms.compression.bpe_engine import BPEEngine  # local import to avoid circular deps
        codebook = self.get_bpe_codebook(method)
        engine = BPEEngine.from_codebook_dict(codebook, encode_backend=encode_backend, **engine_kwargs)
        engine.build_encoder()
        return engine
    
    def save_bpe_codebook(self, method: str, merge_rules: List, vocab_size: int) -> Path:
        """Save BPE codebook (by single/multi_<k> subdirectory)."""
        cache_key = self._get_serialization_cache_key()
        model_path = self.config.model_dir / "bpe" / self.dataset / method / cache_key / "bpe_codebook.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'merge_rules': merge_rules,
            'vocab_size': int(vocab_size)
        }
        
        with model_path.open('wb') as f:
            pickle.dump(data, f)
            
        return model_path



    # ----------------------- Core sequence read interface -----------------------
    def get_sequences(self, method: str) -> Tuple[List[Tuple[int, List[int]]], List[Dict[str, Any]]]:
        serialized = self._load_serialization_result(method)
        
        assert 'sequences' in serialized, "Serialization result missing required field 'sequences'"
        sequences = serialized['sequences']
        assert sequences, "Serialized sequences are empty"
        
        assert 'graph_ids' in serialized, "Serialization result missing required field 'graph_ids'"
        graph_ids = serialized['graph_ids']
            
        # Always fetch properties from DataLoader by graph_id for label consistency
        # Ignore the properties field in serialization result
        loader = self.get_dataset_loader()
        all_graphs = self.get_graphs()
        properties: List[Dict[str, Any]] = []
        for gid in graph_ids:
            try:
                ig = int(gid)
                if 0 <= ig < len(all_graphs):
                    properties.append(all_graphs[ig].get('properties', {}))
                else:
                    properties.append({})
            except Exception:
                print(f"Data loading error: cannot find label for graph ID {gid} in original dataset.")
                raise ValueError(f"Invalid graph ID: {gid}")
            
        # Assemble return format: graph ID first
        sequences_with_ids = [(gid, seq) for seq, gid in zip(sequences, graph_ids)]
        
        return sequences_with_ids, properties

    def get_sequences_by_splits(self, method: str) -> Tuple[
        List[Tuple[int, List[int]]], List[Dict[str, Any]],  # train
        List[Tuple[int, List[int]]], List[Dict[str, Any]],  # val  
        List[Tuple[int, List[int]]], List[Dict[str, Any]]   # test
    ]:
        # Get all data
        all_sequences, all_labels = self.get_sequences(method)
        
        # Get split indices
        split_indices = self._load_split_indices()
        
        # Build graph_id -> index mapping
        graph_id_to_indices = {}
        for idx, (graph_id, seq) in enumerate(all_sequences):
            if graph_id not in graph_id_to_indices:
                graph_id_to_indices[graph_id] = []
            graph_id_to_indices[graph_id].append(idx)
        
        def extract_split_data(split_name: str):
            if split_name not in split_indices:
                raise ValueError(f"Invalid split name: {split_name}")
            
            split_graph_ids = set(split_indices[split_name])
            split_sequences = []
            split_labels = []
            
            for graph_id in split_graph_ids:
                if graph_id in graph_id_to_indices:
                    for idx in graph_id_to_indices[graph_id]:
                        split_sequences.append(all_sequences[idx])
                        split_labels.append(all_labels[idx])
            
            return split_sequences, split_labels
        
        train_seqs, train_labels = extract_split_data('train')
        val_seqs, val_labels = extract_split_data('val') 
        test_seqs, test_labels = extract_split_data('test')
        
        return train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels

    def get_training_data(
        self,
        method: str,
    ) -> Tuple[
        Tuple[List[Tuple[int, List[int]]], List[Dict[str, Any]]],  # train (seqs_with_id, props)
        Tuple[List[Tuple[int, List[int]]], List[Dict[str, Any]]],  # val (seqs_with_id, props)
        Tuple[List[Tuple[int, List[int]]], List[Dict[str, Any]]],  # test (seqs_with_id, props)
    ]:
        """Get data for downstream tasks: return sequences with graph_ids and full property dicts for all splits."""
        # get_sequences_by_splits returns 6 values; reorganize
        train_seqs, train_props, val_seqs, val_props, test_seqs, test_props = self.get_sequences_by_splits(method)
        train_data = (train_seqs, train_props)
        val_data = (val_seqs, val_props)
        test_data = (test_seqs, test_props)
        return train_data, val_data, test_data

    def get_training_data_flat(
        self,
        method: str,
    ) -> Tuple[
        List[List[int]], List[List[int]], List[List[int]]  # train, val, test sequences (flattened)
    ]:
        """Get flattened training sequences (without graph_ids), for pretraining."""
        (
            (train_seqs_with_id, _),
            (val_seqs_with_id, _),
            (test_seqs_with_id, _),
        ) = self.get_training_data(method)

        # Extract sequences only, discard graph_id
        train_sequences = [seq for _, seq in train_seqs_with_id]
        val_sequences = [seq for _, seq in val_seqs_with_id]
        test_sequences = [seq for _, seq in test_seqs_with_id]

        return train_sequences, val_sequences, test_sequences

    def get_training_data_flat_with_ids(
        self,
        method: str,
    ) -> Tuple[
        Tuple[List[List[int]], List[int]],  # train (sequences, graph_ids)
        Tuple[List[List[int]], List[int]],  # val (sequences, graph_ids)
        Tuple[List[List[int]], List[int]],  # test (sequences, graph_ids)
    ]:
        """Get flattened training sequences with graph_ids, for graph-level sampling."""
        (
            (train_seqs_with_id, _),
            (val_seqs_with_id, _),
            (test_seqs_with_id, _),
        ) = self.get_training_data(method)

        # Extract sequences and graph_ids
        train_sequences = [seq for _, seq in train_seqs_with_id]
        train_gids = [gid for gid, _ in train_seqs_with_id]

        val_sequences = [seq for _, seq in val_seqs_with_id]
        val_gids = [gid for gid, _ in val_seqs_with_id]

        test_sequences = [seq for _, seq in test_seqs_with_id]
        test_gids = [gid for gid, _ in test_seqs_with_id]

        return (train_sequences, train_gids), (val_sequences, val_gids), (test_sequences, test_gids)

    def _resolve_target_property(self, requested_target_property: str | None) -> str | None:
        """Resolve and determine the final target_property to use."""
        metadata = self.get_downstream_metadata()
        if 'downstream_label_keys' not in metadata:
            raise ValueError("Dataset metadata missing 'downstream_label_keys' field")
        available_labels = metadata['downstream_label_keys']
        
        if requested_target_property:
            if requested_target_property not in available_labels:
                raise ValueError(
                    f"Requested target_property '{requested_target_property}' not in available labels: {available_labels}"
                )
            return requested_target_property
        
        # User did not specify; auto-resolve
        if not available_labels:
            return None
            
        if len(available_labels) == 1:
            # Single available label: auto-select
            return available_labels[0]
        
        # Multiple labels: try default
        default_property = metadata.get('default_target_property')
        if default_property and default_property in available_labels:
            return default_property
            
        # Cannot resolve ambiguity
        raise ValueError(
            f"Dataset '{self.dataset}' has multiple available properties {available_labels} with no default. "
            f"Please specify one via target_property."
        )

    def _extract_labels_from_properties(self, properties: List[Dict[str, Any]], target_property: str) -> List[Any]:
        """Extract specified target labels from a list of property dicts."""
        labels = []
        for prop in properties:
            if target_property not in prop:
                raise ValueError(f"target_property '{target_property}' not found in property dict {prop}")
            labels.append(prop[target_property])
        return labels


    # ----------------------- Status query & registration -----------------------
    def has_serialized(self, method: str) -> bool:
        base = self._resolve_processed_dir()
        cache_key = self._get_serialization_cache_key()
        result_path = base / "serialized_data" / method / cache_key / "serialized_data.pickle"
        return result_path.exists()

    def has_vocab(self, method: str, bpe: bool) -> bool:
        base = self._resolve_processed_dir()
        sub = "bpe" if bpe else "raw"
        vocab_dir = base / "vocab" / method / sub
        return (vocab_dir / "vocab.json").exists()

    def get_vocab(self, method: str):
        """Read the vocab bound to this dataset. Raises if missing."""
        base = self._resolve_processed_dir()
        cache_key = self._get_serialization_cache_key()
        # Read full vocab (raw + BPE merge tokens), by single/multi_<k> subdirectory
        vocab_path = base / "vocab" / method / "bpe" / cache_key / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab not found: {vocab_path}")
        from src.models.bert.vocab_manager import VocabManager  # local import to avoid circular deps
        return VocabManager.load_vocab(str(vocab_path), self.config)

    def register_vocab(self, vocab_manager, method: str) -> Path:
        """Register (persist) vocab to the dataset's processed directory."""
        base = self._resolve_processed_dir()
        # Register to full vocab location, by single/multi_<k> subdirectory
        cache_key = self._get_serialization_cache_key()
        out_dir = base / "vocab" / method / "bpe" / cache_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vocab.json"
        vocab_manager.save_vocab(str(out_path))
        return out_path

    def register_serialized_sequences(
        self,
        method: str,
        sequences: List[List[int]],
        properties: List[Dict[str, Any]] | None,
        split_indices: Dict[str, List[int]] | None,
    ) -> Path:
        """Register (persist) externally provided serialization results."""
        cache_key = self._get_serialization_cache_key()
        out_dir = self._resolve_processed_dir() / "serialized_data" / method / cache_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "serialized_data.pickle"
        with out_path.open('wb') as f:
            pickle.dump({
                'sequences': sequences,
                'properties': properties or [],
                'serialization_method': method,
            }, f)
        return out_path

    def prepare_serialization(self, method: str) -> Path:
        """Explicitly build and persist serialization results."""
        return self._build_and_persist_serialization(method)

    def get_graphs(self) -> List[Dict[str, Any]]:
        # Read graphs and properties (via loader); no splitting here
        if self._preloaded_graphs is not None:
            return self._preloaded_graphs
        loader = get_dataloader(self.dataset, self.config)
        all_data, _ = loader.get_all_data_with_indices()
        return all_data







    # 2) In-memory end-to-end processing: deprecated (no implicit fallback / no building inside UDI)





    def get_split_indices(
        self,
    ) -> Dict[str, List[int]]:
        """Return train/val/test split index dict."""
        return self._load_split_indices()

    def get_dataset_loader(self) -> BaseDataLoader:
        """Get dataset loader instance (for internal use by serializers etc.)."""
        if self._loader is not None:
            return self._loader
        else:
            self._loader = get_dataloader(self.dataset, self.config)
            return self._loader
    
    def get_num_classes(self) -> int:
        loader = self.get_dataset_loader()
        return loader.get_num_classes()
    
    def get_dataset_task_type(self) -> str:
        """Get dataset task type."""
        loader = self.get_dataset_loader()
        return loader.get_dataset_task_type()

    def get_loss_config(self) -> Optional[Dict[str, Any]]:
        """Get loss config, with hyperparameter override support."""
        # 1. Check for config override
        if hasattr(self.config, 'task') and hasattr(self.config.task, 'loss_config') and self.config.task.loss_config:
            # Hyperparameter search override
            return self.config.task.loss_config

        # 2. Return dataset default config
        loader = self.get_dataset_loader()
        return loader.get_loss_config()

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Get class weights (proxy to DataLoader)."""
        loader = self.get_dataset_loader()
        return loader.get_class_weights()

    def create_loss_function(self, task_type: str, num_classes: int) -> nn.Module:
        loss_config = self.get_loss_config()

        if task_type == "mlm":
            return nn.CrossEntropyLoss(ignore_index=-100)
        elif task_type in ["binary_classification", "classification"]:
            if loss_config and loss_config.get('method') == 'focal':
                return self._create_focal_loss(loss_config)
            elif loss_config and loss_config.get('method') == 'weighted':
                return self._create_weighted_loss(loss_config, num_classes)
            else:
                return nn.CrossEntropyLoss()
        elif task_type == "regression":
            return nn.MSELoss()
        elif task_type == "multi_target_regression":
            return nn.L1Loss()
        elif task_type == "multi_label_classification":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _create_focal_loss(self, config: Dict[str, Any]) -> nn.Module:
        gamma = config.get('gamma', 2.0)
        alpha = config.get('alpha', 1.0)

        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, alpha=1.0):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()

        return FocalLoss(gamma=gamma, alpha=alpha)

    def _create_weighted_loss(self, config: Dict[str, Any], num_classes: int) -> nn.Module:
        auto_weights = config.get('auto_weights', True)

        if auto_weights:
            # Get auto-computed weights from DataLoader
            weights = self.get_class_weights()
            if weights is None:
                # Fall back to uniform weights
                logger.warning("Cannot compute class weights; using uniform weights")
                weights = torch.ones(num_classes)
        else:
            # Use custom weights
            custom_weights = config.get('weights')
            if custom_weights is not None:
                weights = torch.tensor(custom_weights, dtype=torch.float)
            else:
                weights = torch.ones(num_classes)

        return nn.CrossEntropyLoss(weight=weights)
    
    def create_empty_dataset_loader(self) -> BaseDataLoader:
        """Create an 'empty' dataset loader instance (for metadata/convention methods only)."""
        return get_dataloader(self.dataset, self.config)









    # ----------------------- Downstream task metadata -----------------------
    def get_downstream_metadata(self) -> Dict[str, Any]:
        """Return read-only metadata for downstream tasks (label keys, num_classes, etc.)."""
        loader = self.get_dataset_loader()
        meta: Dict[str, Any] = {
            'dataset': self.dataset,
        }
        # Assume all loaders implement these methods
        for attr in [
            'get_downstream_label_keys',
            'get_num_classes',
            'get_default_target_property',
            'get_dataset_task_type',
        ]:
            # Base class provides defaults; subclasses may override
            meta[attr.replace('get_', '')] = getattr(loader, attr)()
        return meta
