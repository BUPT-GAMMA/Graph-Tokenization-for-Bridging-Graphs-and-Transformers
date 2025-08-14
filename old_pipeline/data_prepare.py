"""
数据源流水线 - 为项目提供标准化的数据源
=========================================

这是一个完全独立的数据准备脚本，可以将各种图数据集转换为标准化的数据源格式，
支持序列化和BPE压缩，为后续的机器学习任务提供数据。

功能：
1. 数据加载：从多种数据集加载原始数据
2. 数据预处理：清洗、标准化图数据
3. 序列化：将图转换为token序列
4. BPE压缩：对序列进行BPE压缩
5. 数据保存：保存为项目可用的数据源格式

设计原则：
- 每个阶段的数据都独立保存，便于调试和复用
- 输出格式标准化，作为项目的数据源
- 支持增量处理，避免重复计算
- 提供简单的数据加载接口
- 支持多种数据集和序列化方法
- 按数据集名称组织存储结构，便于管理

使用方法：
==========

1. 查看可用选项：
   python data_prepare.py --list_datasets      # 查看可用数据集
   python data_prepare.py --list_serializers   # 查看可用序列化方法
   python data_prepare.py --list_data_sources  # 查看已生成的数据源

2. 基本使用：
   python data_prepare.py --dataset qm9 --subgraph_limt 1000
   
3. 指定序列化方法：
   python data_prepare.py --dataset qm9 --serialization_method feuler
   
4. 自定义BPE参数：
   python data_prepare.py --dataset qm9 --bpe_num_merges 1000 --bpe_min_frequency 10
   
5. 从指定阶段开始（跳过已完成阶段）：
   python data_prepare.py --dataset qm9 --start_from serialization
   
6. 自定义输出目录：
   python data_prepare.py --dataset qm9 --output_dir my_outputs

支持的数据集：
==============
- qm9: QM9分子数据集（默认）
- cora: Cora引文网络
- citeseer: CiteSeer引文网络
- pubmed: PubMed引文网络
- dblp: DBLP异构网络
- imdb: IMDB网络
- lastfm: Last.FM网络
- yelp: Yelp网络

支持的序列化方法：
==================
- feuler: 图序列化（默认，推荐）
- dfs: 深度优先搜索
- bfs: 广度优先搜索
- eulerian: 欧拉回路
- topo: 拓扑排序
- smiles: SMILES字符串（仅适用于分子数据）

输出结构：
==========
processed_data_dir/
├── qm9/                    # QM9数据集
│   ├── raw_data/           # 原始数据
│   ├── preprocessed_data/     # 预处理数据
│   ├── serialized_data/    # 序列化数据
│   ├── bpe_compressed/     # BPE压缩数据
│   ├── final_data/         # 最终数据源
│   ├── pipeline_config.json # 流水线配置
│   └── stages_status.json  # 阶段状态
├── qm9/                # QM9测试数据集
└── ...                     # 其他数据集

数据源加载：
===========
# 加载指定数据集的数据源
from data_prepare import load_data_source
data_source = load_data_source('qm9')

# 查看数据源信息
print(f"序列数量: {len(data_source.sequences)}")
print(f"压缩率: {data_source.metadata['compression_stats']['compression_ratio']}")

示例：
=====

# 使用QM9数据集，限制1000个分子，使用feuler序列化
python data_prepare.py --dataset qm9 --subgraph_limt 1000 --serialization_method feuler

# 使用Cora数据集，自定义BPE参数
python data_prepare.py --dataset cora --bpe_num_merges 2000 --bpe_min_frequency 20

# 从序列化阶段开始（跳过数据加载和预处理）
python data_prepare.py --dataset qm9 --start_from serialization

# 查看所有可用选项
python data_prepare.py --help

注意事项：
==========
1. 首次运行可能需要下载数据集，请确保网络连接正常
2. 大规模数据处理时注意内存使用情况
3. BPE训练需要足够的序列多样性，某些序列化方法可能不适合
4. 流水线支持断点续传，可以从任意阶段开始
5. 所有中间结果都会保存，便于调试和分析
6. 数据源按数据集名称存储，便于管理和查找

故障排除：
==========
1. 如果BPE训练失败，尝试降低bpe_min_frequency或增加数据量
2. 如果内存不足，减少subgraph_limt或使用更小的数据集
3. 如果序列化失败，检查数据集是否包含有效的图结构
4. 查看日志输出了解详细的执行情况
5. 数据源存储在outputs/data_sources/{dataset_name}/目录下
"""

# import os  # 未使用
import json
import pickle
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
# import logging  # 未使用

import numpy as np
# import pandas as pd  # 未使用

# 导入项目模块
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import ProjectConfig
from src.algorithms.compression.bpe_engine import BPEEngine
from src.algorithms.serializer import SerializerFactory, create_serializer
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DataSource:
    """数据源格式 - 标准化的数据源结构"""
    sequences: List[List[int]]  # token序列列表
    properties: List[Dict[str, float]]  # 图属性列表
    metadata: Dict[str, Any]  # 元数据
    bpe_model: Optional[Any] = None  # BPE模型（如果有）：统一为 BPEEngine 或其 codebook 表示
    compressed_sequences: Optional[List[List[int]]] = None  # 压缩后的序列（如果有）

@dataclass
class ProcessingStage:
    """处理阶段配置"""
    name: str
    completed: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    output_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> Optional[float]:
        """计算处理时长"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class DataSourcePipeline:
    """数据源流水线 - 为项目提供标准化的数据源"""
    
    def __init__(self, config: ProjectConfig, dataset_name: str = "qm9", method_suffix: str = None):
        """
        初始化数据源流水线
        
        Args:
            config: 项目统一配置
            dataset_name: 数据集名称
            method_suffix: 序列化方法后缀，用于区分不同方法的输出
        """
        self.config = config
        self.dataset_name: str = dataset_name
        self.method_suffix: str = method_suffix
        self.stages: Dict[str, ProcessingStage] = {}
        self.results: Dict[str, Any] = {}
        
        # 设置数据源目录
        self._setup_directories()
        
        # 初始化阶段
        self._init_stages()
        
        # 加载或创建配置
        self._load_or_create_config()
    
    def _setup_directories(self):
        """设置数据源目录结构"""
        # 数据源主目录 - 直接使用config.processed_data_dir，不添加data_sources层
        self.data_source_dir = Path(self.config.paths.processed_dir) / self.dataset_name
        self.data_source_dir.mkdir(parents=True, exist_ok=True)
        
        # 各阶段数据目录
        self.stage_dirs = {
            "preprocessed_data": self.data_source_dir / "preprocessed_data", 
            "serialized_data": self.data_source_dir / "serialized_data",
            "bpe_compressed": self.data_source_dir / "bpe_compressed"
        }
        
        # 创建基础目录
        for dir_path in self.stage_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 为序列化阶段创建方法子目录
        if self.method_suffix:
            method_dir = self.stage_dirs["serialized_data"] / self.method_suffix
            method_dir.mkdir(parents=True, exist_ok=True)
            self.stage_dirs["serialized_data"] = method_dir
            
            # 为BPE压缩阶段创建方法子目录
            bpe_method_dir = self.stage_dirs["bpe_compressed"] / self.method_suffix
            bpe_method_dir.mkdir(parents=True, exist_ok=True)
            self.stage_dirs["bpe_compressed"] = bpe_method_dir
        
        # 配置文件路径
        self.config_path = self.data_source_dir / "pipeline_config.json"
        self.stages_path = self.data_source_dir / "stages_status.json"
    
    def _init_stages(self):
        """初始化处理阶段"""
        self.stages = {
            "data_loading": ProcessingStage("数据加载"),
            "serialization": ProcessingStage("序列化"),
            "bpe_compression": ProcessingStage("BPE压缩")
        }
    
    def _load_or_create_config(self):
        """加载或创建配置文件"""
        # 保存主配置
        config_data = {
            "project_config": self.config.to_dict(),
            "dataset_name": self.dataset_name,
            "stage_directories": {k: str(v) for k, v in self.stage_dirs.items()}
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # 加载或创建阶段状态
        if self.stages_path.exists():
            self._load_stages_status()
        else:
            self._save_stages_status()
    
    def _load_stages_status(self):
        """加载阶段状态"""
        with open(self.stages_path, 'r', encoding='utf-8') as f:
            stages_data = json.load(f)
            for stage_name, stage_data in stages_data.items():
                if stage_name in self.stages:
                    self.stages[stage_name] = ProcessingStage(**stage_data)
    
    def _save_stages_status(self):
        """保存阶段状态"""
        stages_data = {name: asdict(stage) for name, stage in self.stages.items()}
        with open(self.stages_path, 'w', encoding='utf-8') as f:
            json.dump(stages_data, f, indent=2, ensure_ascii=False)
    
    def _mark_stage_complete(self, stage_name: str, output_path: str = None, metadata: Dict[str, Any] = None):
        """标记阶段完成"""
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            stage.completed = True
            stage.end_time = time.time()
            if output_path:
                stage.output_path = output_path
            if metadata:
                stage.metadata = metadata
            self._save_stages_status()
            logger.info(f"✅ 阶段 '{stage_name}' 完成，耗时: {stage.duration:.2f}秒")
    
    def _start_stage(self, stage_name: str):
        """开始阶段"""
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            stage.start_time = time.time()
            logger.info(f"🚀 开始阶段: {stage_name}")
    
    def _load_serialized_data(self):
        """从序列化阶段加载数据"""
        logger.info("📂 从序列化阶段加载数据")
        
        # 检查序列化数据文件是否存在
        serialized_data_path = self.stage_dirs["serialized_data"] / "serialized_data.pickle"
        if not serialized_data_path.exists():
            raise FileNotFoundError(f"序列化数据文件不存在: {serialized_data_path}")
        
        # 加载序列化数据
        with open(serialized_data_path, 'rb') as f:
            serialized_data = pickle.load(f)
        
        # 验证数据完整性
        required_keys = ['sequences', 'properties', 'serialization_method']
        for key in required_keys:
            if key not in serialized_data:
                raise ValueError(f"序列化数据缺少必要字段: {key}")
        
        # 将数据加载到results中
        self.results["serialized_data"] = serialized_data
        
        logger.info(f"✅ 成功加载序列化数据: {len(serialized_data['sequences'])} 个序列")
        logger.info(f"📊 序列化方法: {serialized_data['serialization_method']}")
        
        # 验证序列化阶段状态
        if not self.stages["serialization"].completed:
            logger.warning("⚠️ 序列化阶段状态显示未完成，但数据文件存在，将标记为完成")
            self._mark_stage_complete("serialization", str(serialized_data_path), {
                'num_sequences': len(serialized_data['sequences']),
                'serialization_method': serialized_data['serialization_method']
            })

    def run_pipeline(self, start_from: str = "data_loading") -> DataSource:
        """运行数据源流水线"""
        logger.info(f"🎯 开始数据源流水线，从阶段 '{start_from}' 开始")
        logger.info(f"📁 数据源目录: {self.data_source_dir}")
        logger.info(f"📊 数据集: {self.dataset_name}")
        
        # 确定起始阶段 - 移除data_preprocessing阶段
        stage_order = ["data_loading", "serialization", "bpe_compression"]
        
        # 如果不指定start_from或指定为data_loading，从头开始运行
        if start_from == "data_loading" or start_from not in stage_order:
            start_idx = 0
            # 重置所有阶段状态
            for stage_name in self.stages:
                self.stages[stage_name].completed = False
                self.stages[stage_name].start_time = None
                self.stages[stage_name].end_time = None
                self.stages[stage_name].output_path = None
                self.stages[stage_name].metadata = None
        else:
            start_idx = stage_order.index(start_from)
            
            # 根据起始阶段加载相应的数据
            if start_from == "serialization":
                # 从序列化阶段开始，需要加载预处理数据
                self._load_preprocessed_data()
            elif start_from == "bpe_compression":
                # 从BPE压缩阶段开始，需要加载序列化数据
                self._load_serialized_data()
                
        
        # 运行各阶段
        for stage_name in stage_order[start_idx:]:
            self._start_stage(stage_name)
            
            if stage_name == "data_loading":
                self._load_data()
            elif stage_name == "serialization":
                self._serialize_data()
            elif stage_name == "bpe_compression":
                self._compress_data()
        
        # 返回BPE压缩结果作为最终结果
        compressed_data = self.results.get("compressed_data")
        if compressed_data is None:
            raise ValueError("BPE压缩数据不存在")
        
        # 创建简化的数据源对象
        final_data_source = DataSource(
            sequences=compressed_data['sequences'],
            properties=compressed_data['properties'],
            metadata={
                'dataset_name': self.dataset_name,
                'serialization_method': compressed_data['serialization_method'],
                'compression_stats': compressed_data['compression_stats'],
                'train_stats': compressed_data['train_stats'],
                'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            bpe_model=compressed_data['bpe_model'],
            compressed_sequences=compressed_data['compressed_sequences']
        )
        
        logger.info("🎉 数据源流水线完成！")
        return final_data_source

    def _load_preprocessed_data(self):
        """从预处理阶段加载数据"""
        logger.info("📂 从预处理阶段加载数据")
        
        # 检查预处理数据文件是否存在
        preprocessed_data_path = self.stage_dirs["preprocessed_data"] / "preprocessed_data.pickle"
        if not preprocessed_data_path.exists():
            raise FileNotFoundError(f"预处理数据文件不存在: {preprocessed_data_path}")
        
        # 加载预处理数据
        with open(preprocessed_data_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
        
        # 验证数据完整性
        required_keys = ['graphs', 'properties']
        for key in required_keys:
            if key not in preprocessed_data:
                raise ValueError(f"预处理数据缺少必要字段: {key}")
        
        # 将数据加载到results中
        self.results["preprocessed_data"] = preprocessed_data
        
        logger.info(f"✅ 成功加载预处理数据: {len(preprocessed_data['graphs'])} 个图")
        
        # 验证数据加载阶段状态
        if not self.stages["data_loading"].completed:
            logger.warning("⚠️ 数据加载阶段状态显示未完成，但数据文件存在，将标记为完成")
            self._mark_stage_complete("data_loading", str(preprocessed_data_path), {
                'num_graphs': len(preprocessed_data['graphs']),
                'dataset_name': self.dataset_name
            })
    
    def _load_data(self):
        """加载原始数据"""
        logger.info(f"📂 加载{self.dataset_name}数据")
        
        try:
            # 使用新的统一数据加载器API
            from src.data.unified_data_interface import UnifiedDataInterface
            
            # 通过 UDI 获取数据加载器
            udi = UnifiedDataInterface(self.config, self.dataset_name)
            loader = udi.get_dataset_loader()
            
            # 获取全部数据和划分索引
            logger.info("🔄 加载全部数据和划分索引...")
            all_data, split_indices = loader.get_all_data_with_indices()
            
            # 保存划分索引到结果中，供后续使用
            self.results["split_indices"] = split_indices
            logger.info(f"📊 数据集划分: 训练集{len(split_indices['train'])}个, 测试集{len(split_indices['test'])}个, 验证集{len(split_indices['val'])}个")
            
            # 提取属性信息 - 通用方法，自动提取所有非图结构属性
            properties = []
            
            # 定义图结构相关的键，这些不应该作为属性
            graph_structure_keys = {
                'dgl_graph', 'graph', 'edge_index', 'edge_attr', 'node_features', 'edge_features',
                'num_nodes', 'num_edges', 'adjacency', 'adjacency_matrix', 'node_attr',
                'smiles', 'mol', 'molecule', 'rdkit_mol',  # 分子表示
                'id', 'index', 'idx', 'dataset_name', 'data_type',  # 元数据
                'node_ids', 'edge_ids', 'global_node_ids',  # ID映射
                'smiles_1', 'smiles_2', 'smiles_3', 'smiles_4'  # SMILES格式（已添加到数据中）
            }
            
            for i, graph_data in enumerate(all_data):
                prop_dict = {}
                
                # 从图数据中提取属性信息
                if isinstance(graph_data, dict):
                    # 如果有专门的properties字段，优先使用
                    if 'properties' in graph_data and isinstance(graph_data['properties'], dict):
                        prop_dict.update(graph_data['properties'])
                    
                    # 提取所有数值类型的属性（排除图结构相关的键）
                    for key, value in graph_data.items():
                        if key not in graph_structure_keys:
                            # 提取数值类型的属性
                            if isinstance(value, (int, float, np.integer, np.floating)):
                                prop_dict[key] = float(value)
                            # 提取字符串类型的分类属性（如果需要的话）
                            elif isinstance(value, str) and len(value) < 50:  # 限制长度避免提取长文本
                                prop_dict[key] = value
                
                elif hasattr(graph_data, '__dict__'):
                    # 如果是对象，提取其属性
                    for attr_name in dir(graph_data):
                        if not attr_name.startswith('_') and attr_name not in graph_structure_keys:
                            try:
                                attr_value = getattr(graph_data, attr_name)
                                if isinstance(attr_value, (int, float, np.integer, np.floating)):
                                    prop_dict[attr_name] = float(attr_value)
                                elif isinstance(attr_value, str) and len(attr_value) < 50:
                                    prop_dict[attr_name] = attr_value
                            except Exception:
                                continue
                
                # 如果没有提取到任何属性，记录警告但继续
                if not prop_dict:
                    logger.warning(f"第{i}个图数据中未找到任何属性信息")
                    if i < 3:  # 只打印前3个的详细信息，避免日志过多
                        if isinstance(graph_data, dict):
                            logger.warning(f"  可用键: {list(graph_data.keys())}")
                        else:
                            logger.warning(f"  数据类型: {type(graph_data)}")
                
                properties.append(prop_dict)
            
            logger.info(f"📊 提取属性信息: {len(properties)} 个")
            if len(properties) > 0:
                # 统计所有属性键
                all_property_keys = set()
                numeric_properties = set()
                string_properties = set()
                
                for prop in properties:
                    for key, value in prop.items():
                        all_property_keys.add(key)
                        if isinstance(value, (int, float)):
                            numeric_properties.add(key)
                        elif isinstance(value, str):
                            string_properties.add(key)
                
                logger.info("📊 发现的属性类型统计:")
                logger.info(f"   数值属性 ({len(numeric_properties)}个): {sorted(list(numeric_properties))}")
                if string_properties:
                    logger.info(f"   字符串属性 ({len(string_properties)}个): {sorted(list(string_properties))}")
                
                # 显示第一个图的属性示例
                logger.info(f"📊 第一个图的属性示例: {properties[0]}")
                
                # 统计每个属性的覆盖率
                if len(numeric_properties) > 0:
                    logger.info("📊 数值属性覆盖率:")
                    for prop_key in sorted(numeric_properties):
                        count = sum(1 for prop in properties if prop_key in prop and prop[prop_key] is not None)
                        coverage = count / len(properties) * 100
                        logger.info(f"   {prop_key}: {count}/{len(properties)} ({coverage:.1f}%)")
            else:
                logger.warning("⚠️ 未提取到任何属性信息！")
            
            # 数据加载器已经处理好了数据，直接使用
            self.results["preprocessed_data"] = {
                'graphs': all_data,  # 数据加载器返回的已经是处理好的图数据
                'properties': properties  # 正确提取的属性信息
            }
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise
        
        # 保存处理好的数据到preprocessed_data目录
        processed_output_path = self.stage_dirs["preprocessed_data"] / "preprocessed_data.pickle"
        with open(processed_output_path, 'wb') as f:
            pickle.dump(self.results["preprocessed_data"], f)
        
        # 保存处理数据统计信息
        processed_stats_path = self.stage_dirs["preprocessed_data"] / "processed_stats.json"
        processed_stats = {
            "num_graphs": len(self.results["preprocessed_data"]['graphs']),
            "dataset_name": self.dataset_name,
            "data_type": "preprocessed_data",
            "subgraph_limt": self.config.dataset.limit if self.dataset_name == "qm9" else None
        }
        
        with open(processed_stats_path, 'w', encoding='utf-8') as f:
            json.dump(processed_stats, f, indent=2, ensure_ascii=False)
        
        self._mark_stage_complete("data_loading", str(processed_output_path), processed_stats)
    
    def _get_dataset_loader(self):
        """获取数据集加载器"""
        try:
            # 通过 UDI 获取数据加载器
            from src.data.unified_data_interface import UnifiedDataInterface
            udi = UnifiedDataInterface(self.config, self.dataset_name)
            loader = udi.get_dataset_loader()
            return loader
        except Exception as e:
            logger.error(f"❌ 获取数据集加载器失败: {e.with_traceback()}")
            raise
    
    def _serialize_data(self):
        """序列化数据"""
        logger.info("🔄 序列化数据")
        
        preprocessed_data = self.results["preprocessed_data"]
        
        # 获取序列化方法
        serializer_name = self.config.serialization.method
        logger.info(f"🔄 使用序列化方法: {serializer_name}")
        
        # 创建序列化器
        serializer = create_serializer(serializer_name)
        
        # 获取数据集加载器（用于统一接口）
        dataset_loader = self._get_dataset_loader()
        
        # 使用统一接口初始化序列化器
        logger.info(f"🔧 初始化{serializer_name}序列化器...")
        serializer.initialize_with_dataset(dataset_loader, preprocessed_data['graphs'])
        logger.info(f"✅ {serializer_name}序列化器初始化完成")
        
        # 执行批量序列化
        logger.info(f"🔄 开始批量序列化 {len(preprocessed_data['graphs'])} 个图...")
        batch_results = serializer.batch_serialize(
            preprocessed_data['graphs'], 
            desc=f"🔄 {serializer_name}批量序列化",
            max_workers=None  # 使用默认线程数
        )
        
        # 处理批量序列化结果
        sequences = []
        success_count = 0
        failure_count = 0
        
        for i, result in enumerate(batch_results):
            if result and result.token_sequences:
                # 新API返回多个序列，我们取第一个作为主要序列
                sequences.append(result.token_sequences[0])
                success_count += 1
            else:
                logger.warning(f"图 {i} 序列化返回空结果")
                failure_count += 1
        
        logger.info(f"✅ 批量序列化完成: 成功 {success_count} 个, 失败 {failure_count} 个")
        
        if not sequences:
            raise ValueError(f"序列化方法 {serializer_name} 没有产生任何有效序列")
          
        assert len(preprocessed_data['properties']) == len(sequences), "属性数量与序列数量不匹配"
        
        # 保存序列化结果
        serialized_data = {
            'sequences': sequences,
            'properties': preprocessed_data['properties'],
            'serialization_method': serializer_name,
            'num_sequences': len(sequences),
            'avg_sequence_length': np.mean([len(seq) for seq in sequences]),
            'max_sequence_length': max([len(seq) for seq in sequences]),
            'min_sequence_length': min([len(seq) for seq in sequences])
        }
        
        self.results["serialized_data"] = serialized_data
        
        # 保存序列化数据
        output_path = self.stage_dirs["serialized_data"] / "serialized_data.pickle"
        with open(output_path, 'wb') as f:
            pickle.dump(serialized_data, f)
        
        serialized_stat={
          'num_sequences': len(sequences),
          'avg_sequence_length': np.mean([len(seq) for seq in sequences]),
          'max_sequence_length': max([len(seq) for seq in sequences]),
          'min_sequence_length': min([len(seq) for seq in sequences])
          }
        
        # 保存序列化统计信息
        stats_path = self.stage_dirs["serialized_data"] / "serialization_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_stat, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 序列化完成: {len(sequences)} 个序列")
        self._mark_stage_complete("serialization", str(output_path), serialized_data)
    

    
    def _compress_data(self):
        """BPE压缩数据"""
        logger.info("🔄 BPE压缩数据")
        
        serialized_data = self.results["serialized_data"]
        sequences = serialized_data['sequences']
        
        # 创建BPE压缩器
        # 构建 BPEEngine（训练 + 编码）
        eng_cfg = self.config.serialization.bpe.engine
        engine = BPEEngine(
            train_backend=str(eng_cfg.train_backend),
            encode_backend=str(eng_cfg.encode_backend),
            encode_rank_mode=str(eng_cfg.encode_rank_mode),
            encode_rank_k=getattr(eng_cfg, 'encode_rank_k', None),
            encode_rank_min=getattr(eng_cfg, 'encode_rank_min', None),
            encode_rank_max=getattr(eng_cfg, 'encode_rank_max', None),
            encode_rank_dist=getattr(eng_cfg, 'encode_rank_dist', None),
        )
        
        # 训练BPE模型
        logger.info("🎓 训练BPE模型...")
        train_stats = engine.train(sequences, num_merges=int(self.config.serialization.bpe.num_merges),
                                   min_frequency=int(self.config.serialization.bpe.min_frequency))
        engine.build_encoder()
        
        # 压缩序列
        logger.info("🔄 压缩序列...")
        compressed_sequences = engine.batch_encode(sequences)
        
        if not compressed_sequences:
            raise ValueError("BPE压缩没有产生任何有效序列")
        
        # 计算压缩统计
        # 统计压缩比（引擎不直接带 decode，这里仅记录长度比）
        total_original = sum(len(s) for s in sequences)
        total_compressed = sum(len(s) for s in compressed_sequences)
        compression_stats = {
            'original_token_count': total_original,
            'compressed_token_count': total_compressed,
            'compression_ratio': (total_compressed / total_original) if total_original > 0 else 1.0,
        }
        
        # 保存压缩数据
        compressed_data = {
            'sequences': sequences,  # 原始序列
            'compressed_sequences': compressed_sequences,  # 压缩序列
            'properties': serialized_data['properties'],  # 属性
            # 保存 codebook（轻量）：由 engine 的 merge_rules 与 vocab_size 即可重建编码器
            'bpe_model': {
                'merge_rules': engine.merge_rules,
                'vocab_size': engine.vocab_size,
                'encode_backend': str(eng_cfg.encode_backend),
                'encode_rank_mode': str(eng_cfg.encode_rank_mode),
                'encode_rank_k': getattr(eng_cfg, 'encode_rank_k', None),
                'encode_rank_min': getattr(eng_cfg, 'encode_rank_min', None),
                'encode_rank_max': getattr(eng_cfg, 'encode_rank_max', None),
                'encode_rank_dist': getattr(eng_cfg, 'encode_rank_dist', None),
            },
            'serialization_method': serialized_data['serialization_method'],
            'compression_stats': compression_stats,
            'train_stats': train_stats
        }
        
        self.results["compressed_data"] = compressed_data
        
        # 保存压缩数据
        output_path = self.stage_dirs["bpe_compressed"] / "compressed_data.pickle"
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_data, f)
        
        # 单独保存BPE模型到新的model/bpe目录
        # 保存 codebook 为轻量文件（与原 save 路径保持一致）
        bpe_model_path = self.config.get_bpe_model_path(self.dataset_name, self.config.serialization.method)
        bpe_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bpe_model_path, 'wb') as f:
            pickle.dump({'merge_rules': engine.merge_rules, 'vocab_size': engine.vocab_size}, f)
        
        # 也保存一份到原位置以保持兼容性
        old_bpe_model_path = self.stage_dirs["bpe_compressed"] / "bpe_model.pickle"
        with open(old_bpe_model_path, 'wb') as f:
            pickle.dump({'merge_rules': engine.merge_rules, 'vocab_size': engine.vocab_size}, f)
            
        
        # 保存压缩统计信息
        stats_path = self.stage_dirs["bpe_compressed"] / "compression_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(compression_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ BPE压缩完成: 方法{self.config.serialization.method}，压缩率 {compression_stats['compression_ratio']:.3f}")
        self._mark_stage_complete("bpe_compression", str(output_path), compression_stats)
    
    def get_stage_status(self) -> Dict[str, bool]:
        """获取各阶段状态"""
        return {name: stage.completed for name, stage in self.stages.items()}
    
    def get_data_source_info(self) -> Dict[str, Any]:
        """获取数据源信息"""
        # 从BPE压缩结果中获取信息
        compressed_data = self.results.get("compressed_data")
        if compressed_data is None:
            return {}
        
        return {
            'dataset_name': self.dataset_name,
            'num_sequences': len(compressed_data['sequences']),
            'num_properties': len(compressed_data['properties']),
            'serialization_method': compressed_data['serialization_method'],
            'compression_ratio': compressed_data['compression_stats']['compression_ratio'],
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }

# ===========================================
# 便捷的数据源加载函数
# ===========================================

def create_data_source_pipeline(
    config: ProjectConfig = None,
    dataset_name: str = "qm9",
    method_suffix: str = None
) -> DataSourcePipeline:
    """创建数据源流水线"""
    if config is None:
        config = ProjectConfig()
    
    return DataSourcePipeline(config, dataset_name, method_suffix)

def list_available_datasets() -> List[str]:
    """列出可用的数据集"""
    from src.data.unified_data_factory import list_available_datasets as _list_available_datasets
    return _list_available_datasets()

def list_available_serializers(method: str = None) -> List[str]:
    """列出可用的序列化方法"""
    if method is None:
        return SerializerFactory.get_available_serializers()
    else:
        return method.split(',')

def run_single_method_pipeline(args_tuple: Tuple[str, str, ProjectConfig, str]) -> Dict[str, Any]:
    """
    运行单个序列化方法的流水线
    
    Args:
        args_tuple: (method_name, dataset_name, config, start_from)
        
    Returns:
        结果字典
    """
    method_name, dataset_name, config, start_from = args_tuple
    
    # 创建新的配置副本，避免多进程间的配置冲突
    method_config = ProjectConfig()
    method_config.__dict__.update(config.__dict__)
    method_config.serialization.method = method_name
    
    try:
        # 创建流水线，使用方法名作为后缀
        pipeline = DataSourcePipeline(method_config, dataset_name, method_name)
        
        # 运行流水线
        data_source = pipeline.run_pipeline(start_from=start_from)
        
        return {
            'method': method_name,
            'success': True,
            'num_sequences': len(data_source.sequences),
            'compression_ratio': data_source.metadata['compression_stats']['compression_ratio'],
            'output_dir': str(pipeline.data_source_dir)
        }
        
    except Exception as e:
        return {
            'method': method_name,
            'success': False,
            'error': str(e)
        }

def run_parallel_pipelines(dataset_name: str, config: ProjectConfig, start_from: str = "data_loading") -> List[Dict[str, Any]]:
    """
    并行运行多个序列化方法的流水线
    
    Args:
        dataset_name: 数据集名称
        config: 项目配置
        start_from: 起始阶段
        
    Returns:
        结果列表
    """
    # 获取所有可用的序列化方法
    available_methods = list_available_serializers(config.serialization.method)
    
    if not available_methods:
        raise ValueError("没有可用的序列化方法")
    
    print(f"🚀 开始并行处理 {len(available_methods)} 个序列化方法:")
    for method in available_methods:
        print(f"  - {method}")
    
    # 准备参数
    args_list = [(method, dataset_name, config, start_from) for method in available_methods]
    
    # 使用进程池并行处理
    with mp.Pool(processes=min(len(available_methods), mp.cpu_count())) as pool:
        results = pool.map(run_single_method_pipeline, args_list)
    
    return results

# ===========================================
# 命令行接口
# ===========================================

def main():
    """主函数 - 命令行入口"""
    parser = argparse.ArgumentParser(
        description="数据源流水线 - 为项目提供标准化的数据源",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用 - QM9数据集
  python data_prepare.py --dataset qm9 --subgraph_limt 1000
  
  # 指定序列化方法
  python data_prepare.py --dataset qm9 --serialization_method feuler
  
  # 对所有可用序列化方法并行处理
  python data_prepare.py --dataset qm9 
  
  # 自定义BPE参数
  python data_prepare.py --dataset qm9 --bpe_num_merges 1000 --bpe_min_frequency 10
  
  # 从指定阶段开始（跳过已完成阶段）
  python data_prepare.py --dataset qm9 --start_from serialization
  
  # 使用其他数据集
  python data_prepare.py --dataset cora
  
  # 查看可用选项
  python data_prepare.py --list_datasets
  python data_prepare.py --list_serializers
        """
    )
    
    # 基本参数
    parser.add_argument("--dataset", type=str, default="qm9", 
                       help="数据集名称 (默认: qm9)")
    parser.add_argument("--method", type=str, default=None,
                       help="序列化方法 (默认: 无，即处理全部)")
    parser.add_argument("--start_from", type=str, default="data_loading",
                       choices=["data_loading", "serialization", "bpe_compression"],
                       help="从指定阶段开始 (默认: data_loading)")
    
    # 数据配置
    parser.add_argument("--subgraph_limt", type=int, default=None,
                       help="QM9数据限制 (默认: 无限制)")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="数据目录 (默认: data/raw)")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="缓存目录 (默认: /local/gzy/tokg/data/cache)")
    parser.add_argument("--use_cache", action="store_true", default=True,
                       help="使用缓存 (默认: True)")
    
    # BPE配置
    parser.add_argument("--bpe_num_merges", type=int, default=2000,
                       help="BPE合并次数 (默认: 5000)")
    parser.add_argument("--bpe_min_frequency", type=int, default=10,
                       help="BPE最小频率 (默认: 10)")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="输出目录 (默认: outputs)")
    
    # 信息查询
    parser.add_argument("--list_datasets", action="store_true",
                       help="列出可用的数据集")
    parser.add_argument("--list_serializers", action="store_true",
                       help="列出可用的序列化方法")
    
    args = parser.parse_args()
    
    # 信息查询模式
    if args.list_datasets:
        datasets = list_available_datasets()
        print("📂 可用的数据集:")
        for dataset in datasets:
            print(f"  - {dataset}")
        return
    
    if args.list_serializers:
        serializers = list_available_serializers()
        print("🔄 可用的序列化方法:")
        for serializer in serializers:
            print(f"  - {serializer}")
        return
    

    
    # 创建配置
    config = ProjectConfig()
    
    # 覆盖配置参数
    if args.subgraph_limt is not None:
        config.dataset.limit = args.subgraph_limt
    if args.data_dir:
        config.paths.data_dir = args.data_dir
    if args.cache_dir:
        config.paths.cache_dir = args.cache_dir
    if not args.use_cache:
        config.dataset.use_cache = False
    if args.bpe_num_merges:
        config.serialization.bpe.num_merges = args.bpe_num_merges
    if args.bpe_min_frequency:
        config.serialization.bpe.min_frequency = args.bpe_min_frequency
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    
    config.serialization.method = args.method
    
    # 验证配置
    config.validate()
    
    try:
        if not args.method or len(args.method.split(','))>1:
            # 并行处理所有序列化方法
            print("🎯 并行处理所有序列化方法")
            print(f"📊 数据集: {args.dataset}")
            print(f"🚀 起始阶段: {args.start_from}")
            
            results = run_parallel_pipelines(args.dataset, config, args.start_from)

            # 输出结果汇总
            print("\n📊 并行处理结果汇总:")
            print(f"{'方法':<15} {'状态':<8} {'序列数':<8} {'压缩率':<10} {'输出目录':<30}")
            print("-" * 80)
            
            success_count = 0
            for result in results:
                if result['success']:
                    print(f"{result['method']:<15} {'✅成功':<8} {result['num_sequences']:<8} {result['compression_ratio']:<10.3f} {result['output_dir']:<30}")
                    success_count += 1
                else:
                    print(f"{result['method']:<15} {'❌失败':<8} {'N/A':<8} {'N/A':<10} {'N/A':<30}")
                    print(f"   错误: {result['error']}")
            
            print(f"\n🎉 并行处理完成！成功: {success_count}/{len(results)}")
            
        else:
            # 处理单个序列化方法
            pipeline = create_data_source_pipeline(
                config=config,
                dataset_name=args.dataset,
                method_suffix=None  # 单个方法不使用后缀
            )
            
            data_source = pipeline.run_pipeline(start_from=args.start_from)
            print("\n🎉 数据源流水线完成！")
            print(f"📊 数据集: {args.dataset}")
            print(f"🔄 序列化方法: {args.method}")
            print(f"📈 序列数量: {len(data_source.sequences)}")
            print(f"🗜️ 压缩率: {data_source.metadata['compression_stats']['compression_ratio']:.3f}")
            print(f"📁 输出目录: {pipeline.data_source_dir}")
            print("\n💡 提示: 可以使用以下命令加载数据源:")
            print("   from src.data import load_final_data_source")
            print(f"   data_source = load_final_data_source('{args.dataset}', '{args.method}')")
        
    except Exception as e:
        logger.error(f"❌ 流水线执行失败: {e}")
        raise

if __name__ == "__main__":
    main()