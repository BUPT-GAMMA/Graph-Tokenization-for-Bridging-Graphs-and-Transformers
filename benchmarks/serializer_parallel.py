import os
import time
import pytest

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.serializer.serializer_factory import SerializerFactory


def test_eulerian_batch_parallel_speed_and_equivalence_full_dataset():
    cfg = ProjectConfig()
    dataset = "qm9test"
    udi = UnifiedDataInterface(cfg, dataset)

    loader = udi.get_dataset_loader()
    graphs = udi.get_graphs()  # 全量 qm9test 图
    assert len(graphs) > 0

    serializer = SerializerFactory.create_serializer("eulerian")
    serializer.initialize_with_dataset(loader, None)

    t0 = time.perf_counter()
    serial_res = serializer.batch_serialize(graphs, desc="euler-serial", parallel=False)
    t1 = time.perf_counter()

    max_workers = min(16, os.cpu_count() or 1)
    t2 = time.perf_counter()
    parallel_res = serializer.batch_serialize(graphs, desc="euler-parallel", parallel=True, max_workers=max_workers)
    t3 = time.perf_counter()

    # 1) 结果一致性（严格保序）
    assert len(serial_res) == len(parallel_res) == len(graphs)
    for i, (rs, rp) in enumerate(zip(serial_res, parallel_res)):
        ts_s, es_s = rs.get_sequence(0)
        ts_p, es_p = rp.get_sequence(0)
        assert ts_s == ts_p, f"batch_serialize 第{i}个样本 token 不一致"
        assert es_s == es_p, f"batch_serialize 第{i}个样本 element 不一致"

    # 2) 打印耗时（不对速度作硬性断言，仅记录）
    print(f"[Eulerian][batch][qm9test-full] serial: {t1 - t0:.3f}s | parallel({max_workers}): {t3 - t2:.3f}s | n={len(graphs)}")


@pytest.mark.parametrize("num_samples", [8])
def test_eulerian_multiple_parallel_speed_and_equivalence(num_samples: int):
    cfg = ProjectConfig()
    dataset = "qm9test"
    udi = UnifiedDataInterface(cfg, dataset)

    loader = udi.get_dataset_loader()
    graphs = udi.get_graphs()
    splits = udi.get_split_indices()
    graph = graphs[splits["train"][0]]

    serializer = SerializerFactory.create_serializer("eulerian")
    serializer.initialize_with_dataset(loader, None)

    t0 = time.perf_counter()
    serial_res = serializer.multiple_serialize(graph, num_samples=num_samples, parallel=False)
    t1 = time.perf_counter()

    max_workers = min(4, os.cpu_count() or 1)
    t2 = time.perf_counter()
    parallel_res = serializer.multiple_serialize(graph, num_samples=num_samples, parallel=True, max_workers=max_workers)
    t3 = time.perf_counter()

    assert serial_res.get_sequence_count() == parallel_res.get_sequence_count() == min(num_samples, graph['dgl_graph'].num_nodes())
    for i in range(serial_res.get_sequence_count()):
        ts_s, es_s = serial_res.get_sequence(i)
        ts_p, es_p = parallel_res.get_sequence(i)
        assert ts_s == ts_p, f"multiple_serialize 第{i}个样本 token 不一致"
        assert es_s == es_p, f"multiple_serialize 第{i}个样本 element 不一致"

    print(f"[Eulerian][multiple] serial: {t1 - t0:.3f}s | parallel({max_workers}): {t3 - t2:.3f}s")


