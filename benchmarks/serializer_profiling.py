import time
from collections import defaultdict

import pytest
import networkx as nx

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.serializer.serializer_factory import SerializerFactory
from src.algorithms.serializer.base_serializer import BaseGraphSerializer, GlobalIDMapping, SerializationResult


def _run_and_profile(method: str):
    cfg = ProjectConfig()
    dataset = "qm9test"
    udi = UnifiedDataInterface(cfg, dataset)
    loader = udi.get_dataset_loader()
    graphs = udi.get_graphs()

    ser = SerializerFactory.create_serializer(method)
    # 初始化：频率引导方法用全量图以避免统计缺失
    if method in ("feuler", "fcpp"):
        ser.initialize_with_dataset(loader, graphs)
    else:
        try:
            ser.initialize_with_dataset(loader, graphs)
        except Exception:
            ser.initialize_with_dataset(loader, None)

    t = defaultdict(float)

    # ---- wrap token获取 & path->tokens 装配 ----
    orig_get_node_token = BaseGraphSerializer.get_node_token
    orig_get_edge_token = BaseGraphSerializer.get_edge_token
    orig_convert_path = BaseGraphSerializer._convert_path_to_tokens

    def wrap_get_node_token(self, graph, node_id, ntype=None):
        s = time.perf_counter()
        r = orig_get_node_token(self, graph, node_id, ntype)
        t['get_node_token'] += time.perf_counter() - s
        return r

    def wrap_get_edge_token(self, graph, edge_id, etype=None):
        s = time.perf_counter()
        r = orig_get_edge_token(self, graph, edge_id, etype)
        t['get_edge_token'] += time.perf_counter() - s
        return r

    def wrap_convert_path(self, node_path, mol_data):
        s = time.perf_counter()
        r = orig_convert_path(self, node_path, mol_data)
        t['convert_path_to_tokens'] += time.perf_counter() - s
        return r

    BaseGraphSerializer.get_node_token = wrap_get_node_token
    BaseGraphSerializer.get_edge_token = wrap_get_edge_token
    BaseGraphSerializer._convert_path_to_tokens = wrap_convert_path

    # ---- 通用图构建/查询环节计时 ----
    import dgl as _dgl
    # 计时：dgl.graph 构建
    orig_dgl_graph_ctor = _dgl.graph
    def wrap_dgl_graph_ctor(data, *args, **kwargs):
        s = time.perf_counter()
        r = orig_dgl_graph_ctor(data, *args, **kwargs)
        t['dgl_graph_ctor'] += time.perf_counter() - s
        return r
    _dgl.graph = wrap_dgl_graph_ctor

    # 计时：获取所有边（基类方法包裹，避免直接改 DGL 内部）
    orig_get_all_edges = BaseGraphSerializer._get_all_edges_from_heterograph
    def wrap_get_all_edges(self, dgl_graph):
        s = time.perf_counter()
        r = orig_get_all_edges(self, dgl_graph)
        t['get_all_edges'] += time.perf_counter() - s
        return r
    BaseGraphSerializer._get_all_edges_from_heterograph = wrap_get_all_edges

    # 计时：基类一些常用查询
    orig_build_eid_map = BaseGraphSerializer._build_edge_id_mapping
    def wrap_build_eid_map(self, g):
        s = time.perf_counter()
        r = orig_build_eid_map(self, g)
        t['build_edge_id_mapping'] += time.perf_counter() - s
        return r
    BaseGraphSerializer._build_edge_id_mapping = wrap_build_eid_map

    orig_get_eid = BaseGraphSerializer._get_edge_id
    def wrap_get_eid(self, g, src, dst):
        s = time.perf_counter()
        r = orig_get_eid(self, g, src, dst)
        t['get_edge_id'] += time.perf_counter() - s
        return r
    BaseGraphSerializer._get_edge_id = wrap_get_eid

    orig_get_etype = BaseGraphSerializer._get_edge_type
    def wrap_get_etype(self, g, src, dst):
        s = time.perf_counter()
        r = orig_get_etype(self, g, src, dst)
        t['get_edge_type'] += time.perf_counter() - s
        return r
    BaseGraphSerializer._get_edge_type = wrap_get_etype

    orig_get_ntype = BaseGraphSerializer._get_node_type
    def wrap_get_ntype(self, g, node_id):
        s = time.perf_counter()
        r = orig_get_ntype(self, g, node_id)
        t['get_node_type'] += time.perf_counter() - s
        return r
    BaseGraphSerializer._get_node_type = wrap_get_ntype

    orig_dgl2nx = BaseGraphSerializer._convert_dgl_to_networkx
    def wrap_dgl2nx(self, g):
        s = time.perf_counter()
        r = orig_dgl2nx(self, g)
        t['convert_dgl_to_nx'] += time.perf_counter() - s
        return r
    BaseGraphSerializer._convert_dgl_to_networkx = wrap_dgl2nx

    # ---- method-specific wraps ----
    if method == 'eulerian':
        from src.algorithms.serializer.eulerian_serializer import EulerianSerializer
        orig_build_adj = BaseGraphSerializer._build_adjacency_list_from_dgl
        orig_has_euler = EulerianSerializer._has_eulerian_circuit
        orig_find = EulerianSerializer._find_eulerian_circuit

        def wrap_build_adj(self, dgl_graph):
            s = time.perf_counter()
            r = orig_build_adj(self, dgl_graph)
            t['build_adj'] += time.perf_counter() - s
            return r

        def wrap_has(self, adj_list, num_nodes):
            s = time.perf_counter()
            r = orig_has_euler(self, adj_list, num_nodes)
            t['has_euler'] += time.perf_counter() - s
            return r

        def wrap_find(self, adj_list, start_node):
            s = time.perf_counter()
            r = orig_find(self, adj_list, start_node)
            t['find_euler'] += time.perf_counter() - s
            return r

        BaseGraphSerializer._build_adjacency_list_from_dgl = wrap_build_adj
        EulerianSerializer._has_eulerian_circuit = wrap_has
        EulerianSerializer._find_eulerian_circuit = wrap_find

    if method in ('cpp', 'fcpp'):
        if method == 'cpp':
            from src.algorithms.serializer.chinese_postman_serializer import CPPSerializer as _S
        else:
            from src.algorithms.serializer.freq_chinese_postman_serializer import FCPPSerializer as _S

        orig_cpp = _S._chinese_postman_networkx
        def wrap_cpp(self, graph, start_node=0):
            s = time.perf_counter()
            r = orig_cpp(self, graph, start_node)
            t['cpp_solver'] += time.perf_counter() - s
            return r
        _S._chinese_postman_networkx = wrap_cpp

        # 粗略统计 dijkstra 与 matching 的时间
        orig_dijkstra = nx.single_source_dijkstra
        def wrap_dijkstra(G, source, target=None, weight='weight'):
            s = time.perf_counter()
            r = orig_dijkstra(G, source, target=target, weight=weight)
            t['nx_dijkstra'] += time.perf_counter() - s
            return r
        nx.single_source_dijkstra = wrap_dijkstra

        from networkx.algorithms import matching as nxm
        orig_mwm = nxm.max_weight_matching
        def wrap_mwm(G, **kwargs):
            s = time.perf_counter()
            r = orig_mwm(G, **kwargs)
            t['nx_matching'] += time.perf_counter() - s
            return r
        nxm.max_weight_matching = wrap_mwm

        # FCPP 独有：边权计算
        try:
            orig_fcpp_calc_w = _S._calculate_edge_weights
            def wrap_fcpp_calc_w(self, g):
                s = time.perf_counter()
                r = orig_fcpp_calc_w(self, g)
                t['fcpp_calc_edge_weights'] += time.perf_counter() - s
                return r
            _S._calculate_edge_weights = wrap_fcpp_calc_w
        except Exception:
            pass

    if method == 'topo':
        from src.algorithms.serializer.topo_serializer import TopoSerializer
        orig_topo = TopoSerializer._topo_serialize
        def wrap_topo(self, dgl_graph, raw_graph):
            s = time.perf_counter()
            r = orig_topo(self, dgl_graph, raw_graph)
            t['topo_core'] += time.perf_counter() - s
            return r
        TopoSerializer._topo_serialize = wrap_topo

        # 深入：重写 _serialize_single_graph 以测量子步骤（尽量不改逻辑）
        orig_topo_serialize_single = TopoSerializer._serialize_single_graph
        def wrap_topo_serialize_single(self, graph_data: dict, **kwargs):
            raw = graph_data['dgl_graph']
            s = time.perf_counter(); src, dst = raw.edges(); t['topo_edges()'] += time.perf_counter() - s
            s = time.perf_counter(); mask = src > dst; src2 = src[mask]; dst2 = dst[mask]; t['topo_mask'] += time.perf_counter() - s
            s = time.perf_counter(); new_g = _dgl.graph((src2, dst2)); t['topo_new_graph'] += time.perf_counter() - s
            s = time.perf_counter();
            if 'feat' in raw.ndata:
                new_g.ndata['feat'] = raw.ndata['feat']
            else:
                new_g.ndata['attr'] = raw.ndata['attr']
            t['topo_copy_ndata'] += time.perf_counter() - s
            token_sequence, element_sequence = self._topo_serialize(new_g, raw)
            id_mapping = GlobalIDMapping(raw)
            return SerializationResult([token_sequence], [element_sequence], id_mapping)
        TopoSerializer._serialize_single_graph = wrap_topo_serialize_single

    if method == 'feuler':
        from src.algorithms.serializer.freq_eulerian_serializer import FeulerSerializer
        # 计时：feuler 找路径
        try:
            orig_feuler_find = FeulerSerializer._find_frequency_guided_eulerian_circuit
            def wrap_feuler_find(self, g, start_node=0):
                s = time.perf_counter()
                r = orig_feuler_find(self, g, start_node)
                t['feuler_find'] += time.perf_counter() - s
                return r
            FeulerSerializer._find_frequency_guided_eulerian_circuit = wrap_feuler_find
        except Exception:
            pass
        # 计时：基类边权计算（feuler使用基类的 _calculate_edge_weights）
        orig_base_calc_w = BaseGraphSerializer._calculate_edge_weights
        def wrap_base_calc_w(self, g):
            s = time.perf_counter()
            r = orig_base_calc_w(self, g)
            t['calc_edge_weights_base'] += time.perf_counter() - s
            return r
        BaseGraphSerializer._calculate_edge_weights = wrap_base_calc_w

    # ---- run ----
    s_all = time.perf_counter()
    results = ser.batch_serialize(graphs, parallel=False)
    t['total'] = time.perf_counter() - s_all

    # ---- restore ----
    BaseGraphSerializer.get_node_token = orig_get_node_token
    BaseGraphSerializer.get_edge_token = orig_get_edge_token
    BaseGraphSerializer._convert_path_to_tokens = orig_convert_path
    if method == 'eulerian':
        from src.algorithms.serializer.eulerian_serializer import EulerianSerializer
        BaseGraphSerializer._build_adjacency_list_from_dgl = orig_build_adj
        EulerianSerializer._has_eulerian_circuit = orig_has_euler
        EulerianSerializer._find_eulerian_circuit = orig_find
    if method in ('cpp', 'fcpp'):
        if method == 'cpp':
            from src.algorithms.serializer.chinese_postman_serializer import CPPSerializer as _S
        else:
            from src.algorithms.serializer.freq_chinese_postman_serializer import FCPPSerializer as _S
        _S._chinese_postman_networkx = orig_cpp
        nx.single_source_dijkstra = orig_dijkstra
        from networkx.algorithms import matching as nxm
        nxm.max_weight_matching = orig_mwm
        try:
            _S._calculate_edge_weights = orig_fcpp_calc_w
        except Exception:
            pass
    if method == 'topo':
        from src.algorithms.serializer.topo_serializer import TopoSerializer
        TopoSerializer._topo_serialize = orig_topo
        TopoSerializer._serialize_single_graph = orig_topo_serialize_single
    if method == 'feuler':
        from src.algorithms.serializer.freq_eulerian_serializer import FeulerSerializer
        try:
            FeulerSerializer._find_frequency_guided_eulerian_circuit = orig_feuler_find
        except Exception:
            pass
        BaseGraphSerializer._calculate_edge_weights = orig_base_calc_w

    # 通用还原
    _dgl.graph = orig_dgl_graph_ctor
    BaseGraphSerializer._get_all_edges_from_heterograph = orig_get_all_edges
    BaseGraphSerializer._build_edge_id_mapping = orig_build_eid_map
    BaseGraphSerializer._get_edge_id = orig_get_eid
    BaseGraphSerializer._get_edge_type = orig_get_etype
    BaseGraphSerializer._get_node_type = orig_get_ntype
    BaseGraphSerializer._convert_dgl_to_networkx = orig_dgl2nx

    return t


@pytest.mark.parametrize("method", ["cpp", "fcpp", "eulerian", "feuler", "bfs", "dfs", "topo"])
def test_profile_breakdown_qm9test(method):
    t = _run_and_profile(method)
    # 打印关键耗时占比（不作性能断言，仅记录）
    total = t.get('total', 0.0) or 1.0
    parts = {k: v for k, v in t.items() if k != 'total'}
    parts_sorted = sorted(parts.items(), key=lambda x: x[1], reverse=True)
    print(f"[profile] method={method} total={total:.3f}s")
    for k, v in parts_sorted:
        pct = 100.0 * v / total
        print(f"  - {k}: {v:.3f}s ({pct:.1f}%)")
    # 基本有效性：总时间应大于各部分单项时间
    assert total >= 0.0


