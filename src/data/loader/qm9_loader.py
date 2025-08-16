"""
QM9数据加载器
================

直接从预处理数据目录读取QM9数据集。
支持多任务标签选择和多种SMILES格式。
"""

 
import time
 
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle

# 必需依赖
import dgl
import torch

from ..base_loader import BaseDataLoader
from config import ProjectConfig
from utils.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)


class QM9Loader(BaseDataLoader):
    """QM9数据加载器 - 支持多任务和数据集划分"""
    
    # QM9数据集信息
    QM9_PROPERTIES = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", 
        "u0", "u298", "h298", "g298", "cv", "u0_atom", "u298_atom", 
        "h298_atom", "g298_atom"
    ]
    BOND_TYPES = {0:'NONE', 1:'SINGLE', 2:'DOUBLE', 3:'TRIPLE', 4:'AROMATIC'}
    ATOM_TYPES = {1:'H', 2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",41:"Nb",42:"Mo",43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",51:"Sb",52:"Te",53:"I",54:"Xe",55:"Cs",56:"Ba",57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",67:"Ho",68:"Er",69:"Tm",70:"Yb",71:"Lu",72:"Hf",73:"Ta",74:"W",75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",80:"Hg",81:"Tl",82:"Pb",83:"Bi",84:"Po",85:"At",86:"Rn",87:"Fr",88:"Ra",89:"Ac",90:"Th",91:"Pa",92:"U",93:"Np",94:"Pu",95:"Am",96:"Cm",97:"Bk",98:"Cf",99:"Es",100:"Fm",101:"Md",102:"No",103:"Lr",104:"Rf",105:"Db",106:"Sg",107:"Bh",108:"Hs",109:"Mt",110:"Ds",111:"Rg",112:"Cn",113:"Nh",114:"Fl",115:"Mc",116:"Lv",117:"Ts",118:"Og"} 
    
    
    
    def __init__(self, config: ProjectConfig, dataset_name: str='qm9',
                 target_property: Optional[str] = None):
        """
        初始化QM9加载器
        
        Args:
            config: 项目配置
            target_property: 目标属性（对于多标签数据集，None表示返回所有属性）
        """
        super().__init__(dataset_name, config, target_property)
        
        # 属性缓存（仅内存缓存，不持久化）
        self._node_attr_cache = {}  # id(graph) -> {node_id -> attr_value}
        self._edge_attr_cache = {}  # id(graph) -> {edge_id -> attr_value}
        self._cache_built = False
        
        # 全部数据缓存
        self._all_data = None
        self.load_data()
    
    def _load_processed_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        从预处理数据目录加载数据
        
        新的数据结构：
        - 统一的数据文件：data.pkl
        - 三个index文件：train_index.json, test_index.json, val_index.json
        - 使用索引来标识不同划分的数据条目
        
        Returns:
            Tuple[List, List, List]: (训练集, 验证集, 测试集)
        """
        logger.info(f"📂 从预处理数据目录加载QM9数据: {self.data_dir}")
        
        # 检查index文件是否存在
        train_index_file = self.data_dir / "train_index.json"
        test_index_file = self.data_dir / "test_index.json"
        val_index_file = self.data_dir / "val_index.json"
        
        if not all(f.exists() for f in [train_index_file, test_index_file, val_index_file]):
            raise FileNotFoundError("索引文件不存在，请先运行数据重构脚本")
        
        # 加载全部数据
        data_file = self.data_dir / "data.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"统一数据文件不存在: {data_file}")
        
        # 加载统一的数据文件
        logger.info("🔄 加载统一数据文件...")
        if self._all_data is None:
            all_data = self._load_data_file(data_file)
            self._all_data = all_data
        all_data = self._all_data
        
        # 加载4种格式的SMILES
        logger.info("🔄 加载SMILES数据...")
        all_data = self._add_smiles_to_all_data(all_data)
        
        logger.info(f"✅ 全部数据加载完成: {len(all_data)} 个样本")
        
        # 加载划分索引
        logger.info("🔄 加载划分索引...")
        split_indices = self.get_split_indices()
        
        # 根据索引划分数据
        train_indices = split_indices['train']
        test_indices = split_indices['test']
        val_indices = split_indices['val']
        
        # 根据索引提取数据
        train_data = [all_data[i] for i in train_indices if i < len(all_data)]
        test_data = [all_data[i] for i in test_indices if i < len(all_data)]
        val_data = [all_data[i] for i in val_indices if i < len(all_data)]
        
        logger.info(f"✅ 数据加载完成: 训练集{len(train_data)}, 验证集{len(val_data)}, 测试集{len(test_data)}")
        
        return train_data, val_data, test_data
    
    
    def _add_smiles_to_all_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为全部数据添加4种格式的SMILES
        
        Args:
            data: 数据列表
            
        Returns:
            添加了SMILES的数据列表
        """
        # 4种SMILES格式的文件名
        smiles_files = {
            "smiles_1": "smiles_1_direct.txt",
            "smiles_2": "smiles_2_explicit_h.txt", 
            "smiles_3": "smiles_3_addhs.txt",
            "smiles_4": "smiles_4_addhs_explicit_h.txt"
        }
        
        # 加载每种格式的SMILES
        smiles_data = {}
        for key, filename in smiles_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        smiles_list = [line.strip() for line in f.readlines()]
                    smiles_data[key] = smiles_list
                    logger.debug(f"✅ 加载 {filename}: {len(smiles_list)} 个SMILES")
                except Exception as e:
                    logger.warning(f"⚠️ 加载 {filename} 失败: {e}")
                    smiles_data[key] = []
            else:
                logger.warning(f"⚠️ SMILES文件不存在: {file_path}")
                smiles_data[key] = []
        
        # 为每个数据样本添加SMILES（按顺序）
        for i, sample in enumerate(data):
            for key, smiles_list in smiles_data.items():
                if i < len(smiles_list):
                    sample[key] = smiles_list[i]
                else:
                    sample[key] = ""
        
        return data
    
    def _load_data_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载单个数据文件"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = pickle.load(f)

            # 转换tuple格式为dict格式
            data = []
            for i, (graph, label) in enumerate(raw_data):
                sample = {
                    'id': f"molecule_{i}",
                    'dgl_graph': graph,
                    'num_nodes': graph.num_nodes(),
                    'num_edges': graph.num_edges() // 2,  # DGL图是双向的
                    'properties': label,
                    'dataset_name': self.dataset_name,
                    'data_type': 'molecular_graph'
                }
                data.append(sample)
            
            return data
        except Exception as e:
            logger.error(f"❌ 加载数据文件失败 {file_path}: {e}")
            raise
    
    def _extract_labels(self, data: List[Dict[str, Any]]) -> List[Any]:
        """
        从数据中提取标签
        
        Args:
            data: 数据列表
            
        Returns:
            标签列表（单标签）或标签字典列表（多标签）
        """
        labels = []
        
        for sample in data:
            properties = sample['properties']
            
            if self.target_property:
                # 指定了目标属性 - 返回单标签
                if self.target_property in properties:
                    label = properties[self.target_property]
                    labels.append(label)
                else:
                    logger.warning(f"目标属性 {self.target_property} 不存在于样本中")
                    labels.append(0.0)
            else:
                # 没有指定目标属性 - 返回所有属性（多标签）
                labels.append(properties)
        
        return labels
    
    def _get_data_metadata(self) -> Dict[str, Any]:
        """
        获取数据元信息
        
        Returns:
            元信息字典
        """
        # 确保数据已加载
        if self._train_data is None:
            self.load_data()
        
        all_data = self._train_data + self._val_data + self._test_data
        
        if not all_data:
            return {}
        
        # 统计信息
        num_samples = len(all_data)
        num_nodes_list = [sample.get('num_nodes', 0) for sample in all_data]
        num_edges_list = [sample.get('num_edges', 0) for sample in all_data]
        
        # 属性统计
        property_stats = {}
        if all_data and 'properties' in all_data[0]:
            properties = all_data[0]['properties']
            for prop_name in self.QM9_PROPERTIES:
                if prop_name in properties:
                    prop_values = [sample.get('properties', {}).get(prop_name) for sample in all_data 
                                 if prop_name in sample.get('properties', {})]
                    if prop_values:
                        property_stats[prop_name] = {
                            'min': float(np.min(prop_values)),
                            'max': float(np.max(prop_values)),
                            'mean': float(np.mean(prop_values)),
                            'std': float(np.std(prop_values))
                        }
        
        metadata = {
            'dataset_name': self.dataset_name,
            'dataset_type': 'molecular_graph',
            'data_source': 'preprocessed_qm9',
            'total_molecules': num_samples,
            'avg_num_nodes': int(np.mean(num_nodes_list)),
            'avg_num_edges': int(np.mean(num_edges_list)),
            'target_property': self.target_property,
            'qm9_properties': self.QM9_PROPERTIES,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_dir': str(self.data_dir),
            'property_availability': property_stats,
            'split_ratios': {
                'train': self.TRAIN_RATIO,
                'val': self.VAL_RATIO,
                'test': self.TEST_RATIO
            }
        }
        
        return metadata

    # ---------------- 下游任务元信息（供上层自动推断） ----------------
    def get_dataset_task_type(self) -> str:
        """QM9 为回归任务（可多属性回归）。"""
        return "regression"

    def get_default_target_property(self) -> str:
        """返回一个常用默认属性，若未指定 target_property 时可采用。"""
        return self.QM9_PROPERTIES[0]
    
    def get_downstream_label_keys(self) -> List[str]:
        """返回数据集支持的所有标签属性名列表。"""
        return self.QM9_PROPERTIES.copy()
    
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], 
                           List[Any], List[Any], List[Any]]:
        """重写load_data方法，添加属性缓存构建"""
        res = super().load_data()
        # 构建属性缓存
        if self._all_data is not None:
            self._build_attribute_cache(self._all_data)
        return res
    
    def _build_attribute_cache(self, processed_data: List[Dict[str, Any]]) -> None:
        """
        构建节点和边的属性缓存
        
        Args:
            processed_data: 处理后的数据列表
        """
        import time
        start_time = time.time()
        logger.info("🔄 构建属性缓存...")
        
        for sample in tqdm(processed_data, desc="构建缓存"):
            dgl_graph = sample['dgl_graph']
            graph_id = id(dgl_graph)

            # ========= 张量化提取并写入类型ID =========
            # 节点原子序数: ndata['attr'] 第6列（索引5）
            assert 'attr' in dgl_graph.ndata, "图缺少 ndata['attr']，无法构建类型特征"
            attr_features = dgl_graph.ndata['attr']  # [N, 12]
            atomic_num_tensor = attr_features[:, 5].long().clamp(min=0)
            dgl_graph.ndata['node_type_id'] = atomic_num_tensor

            # 边键类型: edata['edge_attr'] 前4维 one-hot 的 argmax，再 +1 映射到统一空间
            assert 'edge_attr' in dgl_graph.edata, "图缺少 edata['edge_attr']，无法构建类型特征"
            edge_attrs = dgl_graph.edata['edge_attr']  # [E, 4]
            bond_type_base = torch.argmax(edge_attrs[:, :4], dim=1).long()  # 0..3
            edge_type_ids = bond_type_base + 1  # 1..4 (NONE 预留为0)
            dgl_graph.edata['edge_type_id'] = edge_type_ids

            # ========= 构建内存缓存（供现有单项接口复用） =========
            self._node_attr_cache[graph_id] = {int(i): int(v) for i, v in enumerate(atomic_num_tensor.tolist())}
            self._edge_attr_cache[graph_id] = {int(i): int(v) for i, v in enumerate(edge_type_ids.tolist())}

            # ========= 张量化写入定长 token =========
            # QM9 当前 token 维度均为1：节点 token = 2*atomic_num+1；边 token = 2*edge_type_id
            node_token_ids = (atomic_num_tensor * 2 + 1).view(-1, 1)
            edge_token_ids = (edge_type_ids * 2).view(-1, 1)
            dgl_graph.ndata['node_token_ids'] = node_token_ids
            dgl_graph.edata['edge_token_ids'] = edge_token_ids
        
        self._cache_built = True
        cache_time = time.time() - start_time
        
        # 统计缓存信息
        total_nodes = sum(len(cache) for cache in self._node_attr_cache.values())
        total_edges = sum(len(cache) for cache in self._edge_attr_cache.values())
        
        logger.info(f"✅ 属性缓存构建完成: {len(self._node_attr_cache)} 个图的节点缓存, {len(self._edge_attr_cache)} 个图的边缓存")
        logger.info(f"📊 缓存统计: {total_nodes} 个节点属性, {total_edges} 个边属性")
        logger.info(f"⏱️ 缓存构建耗时: {cache_time:.2f}秒")
    
    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> int:
        """
        获取指定节点的关键属性（用于token映射）
        
        Args:
            graph: DGL图
            node_id: 节点ID
            ntype: 节点类型（可选）
            
        Returns:
            int: 节点的关键属性值
        """
        assert node_id >= 0 and node_id < graph.num_nodes(), f"❌ 节点ID {node_id} 超出范围"
        
        # 优先使用缓存
        if self._cache_built:
            graph_id = id(graph)
            if graph_id in self._node_attr_cache:
                node_cache = self._node_attr_cache[graph_id]
                if node_id in node_cache:
                    return node_cache[node_id]
        
        # 回退到原始方法
        if 'attr' in graph.ndata:
            attr_features = graph.ndata['attr']
            if node_id < len(attr_features):
                atomic_num = int(attr_features[node_id][5].item())  # 维度5是原子序数
                return atomic_num
        
        raise ValueError(f"❌ 无法获取节点 {node_id} 的属性")
    
    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> int:
        """
        获取指定边的关键属性（用于token映射）
        
        Args:
            graph: DGL图
            edge_id: 边ID
            etype: 边类型（可选）
            
        Returns:
            int: 边的关键属性值
        """
        assert edge_id >= 0 and edge_id < graph.num_edges(), f"❌ 边ID {edge_id} 超出范围"
        
        # 优先使用缓存
        if self._cache_built:
            graph_id = id(graph)
            if graph_id in self._edge_attr_cache:
                edge_cache = self._edge_attr_cache[graph_id]
                if edge_id in edge_cache:
                    return edge_cache[edge_id]
        
        # 回退到原始方法
        if 'edge_attr' in graph.edata:
            edge_attrs = graph.edata['edge_attr']
            if edge_id < len(edge_attrs):
                # 从one-hot编码中提取键类型
                bond_type = torch.argmax(edge_attrs[edge_id][:4]).item()  # 取前4维作为键类型
                return bond_type
        
        raise ValueError(f"❌ 无法获取边 {edge_id} 的属性")
    
    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        """
        获取节点的类型
        """
        return self.ATOM_TYPES[self.get_node_attribute(graph, node_id)]
    
    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        """
        获取边的类型
        """
        return self.BOND_TYPES[self.get_edge_attribute(graph, edge_id)]

    # ==================== 批量张量化接口 ====================

    def get_node_types_bulk(self, graph: dgl.DGLGraph, node_ids: List[int]) -> List[str]:
        assert 'node_type_id' in graph.ndata, "缺少 node_type_id 特征"
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        type_ids = graph.ndata['node_type_id'][ids].tolist()
        return [self.ATOM_TYPES[int(tid)] for tid in type_ids]

    def get_edge_types_bulk(self, graph: dgl.DGLGraph, edge_ids: List[int]) -> List[str]:
        assert 'edge_type_id' in graph.edata, "缺少 edge_type_id 特征"
        ids = torch.as_tensor(edge_ids, dtype=torch.long)
        type_ids = graph.edata['edge_type_id'][ids].tolist()
        return [self.BOND_TYPES[int(tid)] for tid in type_ids]

    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: List[int]) -> List[List[int]]:
        assert 'node_token_ids' in graph.ndata, "缺少 node_token_ids 特征"
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        tok = graph.ndata['node_token_ids'][ids]  # [N, D]
        return tok.tolist()

    def get_edge_tokens_bulk(self, graph: dgl.DGLGraph, edge_ids: List[int]) -> List[List[int]]:
        assert 'edge_token_ids' in graph.edata, "缺少 edge_token_ids 特征"
        ids = torch.as_tensor(edge_ids, dtype=torch.long)
        tok = graph.edata['edge_token_ids'][ids]
        return tok.tolist()
      
    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        """
        获取节点的token列表
        """
        node_key = ('atom', self.get_node_attribute(graph, node_id, ntype))
        return [self.token_map[node_key]]
      
    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        """
        获取边的token列表
        """
        edge_key = ('bond', self.get_edge_attribute(graph, edge_id, etype))
        return [self.token_map[edge_key]]
    
    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        """
        获取整个数据集的token映射
        """
        
        node_token_map = {}
        # 为所有可能的原子序数创建token映射
        # 使用奇数ID避免与边token冲突
        for atomic_num in range(1, 119):  # 元素周期表1-118号元素
            odd_token = 2 * atomic_num + 1
            node_token_map[('atom', atomic_num)] = odd_token
            
        edge_token_map = {}
        # 为所有可能的化学键类型创建token映射
        # 使用偶数ID避免与节点token冲突
        for bond_type_id, bond_type_name in self.BOND_TYPES.items():
            even_token = 2 * bond_type_id
            edge_token_map[('bond', bond_type_id)] = even_token
            
        return {**node_token_map, **edge_token_map}
    
    def get_token_readable(self, token_id: int) -> str:
        """获取token到可读字符串的映射"""
        
        if token_id % 2 == 0:
            return {0:'',1:"-",2:"=",3:"*",4:"@"}[token_id // 2]
        else:
            return self.ATOM_TYPES[token_id // 2]
      
    def get_most_frequent_edge_type(self) -> str:
        """获取最频繁的边类型"""
        return 'SINGLE'

    # ==================== 整图张量接口实现 ====================
    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'node_type_id' in graph.ndata, "缺少 node_type_id"
        return graph.ndata['node_type_id']

    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'edge_type_id' in graph.edata, "缺少 edge_type_id"
        return graph.edata['edge_type_id']

    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'node_token_ids' in graph.ndata, "缺少 node_token_ids"
        return graph.ndata['node_token_ids']

    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'edge_token_ids' in graph.edata, "缺少 edge_token_ids"
        return graph.edata['edge_token_ids']

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()

    def get_edge_type_id_by_name(self, name: str) -> int:
        inv = {v: k for k, v in self.BOND_TYPES.items()}
        assert name in inv, f"未知边类型: {name}"
        return int(inv[name])
    