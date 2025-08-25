#!/usr/bin/env python3
"""
序列化方法速度测试脚本

测试不同序列化方法在不同数据集和批处理大小下的速度性能
支持多线程batch_serialize，测试批大小包括32和128
"""

import os
import time
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.serializer.serializer_factory import SerializerFactory


def get_all_available_datasets(data_dir: str = "data", exclude: List[str] = None) -> List[str]:
    """
    自动发现所有可用的数据集
    
    Args:
        data_dir: 数据目录路径
        exclude: 要排除的数据集列表
        
    Returns:
        可用数据集名称列表
    """
    # 默认排除dd数据集（耗时太久）和已知有问题的数据集
    default_exclude = ['dd', 'code2', 'mnist', 'mnist_raw', 'molecules']
    exclude = (exclude or []) + default_exclude
    exclude = list(set(exclude))  # 去重
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠️  数据目录不存在: {data_dir}")
        return []
    
    datasets = []
    for item in data_path.iterdir():
        if item.is_dir() and item.name not in ['processed', 'small_backup'] and item.name not in exclude:
            # 简单检查是否包含数据文件
            if any(item.glob("*.pkl")) or any(item.glob("*.pt")) or any(item.glob("*.json")):
                datasets.append(item.name)
    
    # 确保qm9test排在前面（通常比较小，适合快速测试）
    if 'qm9test' in datasets:
        datasets.remove('qm9test')
        datasets.insert(0, 'qm9test')
    
    return sorted(datasets, key=lambda x: (x != 'qm9test', x))


@dataclass
class BenchmarkResult:
    """基准测试结果数据类"""
    method: str
    dataset: str
    batch_size: int
    num_samples: int
    total_time: float
    avg_time_per_sample: float
    parallel_enabled: bool
    max_workers: int
    threads_used: int
    samples_per_second: float
    success_count: int
    error_count: int


class SerializationBenchmark:
    """序列化方法速度基准测试"""
    
    def __init__(self, datasets: List[str] = None, methods: List[str] = None, 
                 batch_sizes: List[int] = None, max_workers: int = None, max_samples: int = None):
        """
        初始化基准测试
        
        Args:
            datasets: 要测试的数据集列表，默认为 ['qm9test', 'qm9']
            methods: 要测试的序列化方法列表，默认使用所有可用方法
            batch_sizes: 要测试的批处理大小列表，默认为 [32, 128]  
            max_workers: 最大工作线程数，默认为 CPU 核心数
            max_samples: 每个数据集最大测试样本数，None表示不限制
        """
        self.datasets = datasets or ['qm9test', 'qm9']
        self.methods = methods or SerializerFactory.get_available_serializers()
        self.batch_sizes = batch_sizes or [32, 128]
        self.max_workers = max_workers or min(16, os.cpu_count() or 1)
        self.max_samples = max_samples
        self.results: List[BenchmarkResult] = []
        self.config = ProjectConfig()
        
        # 过滤掉可能有问题的方法（根据已有测试代码的经验）
        self.methods = [m for m in self.methods if m not in ['smiles']]  # smiles可能需要特殊处理
        
        print(f"🧪 序列化速度基准测试初始化完成")
        print(f"   数据集: {self.datasets}")
        print(f"   方法: {self.methods}")
        print(f"   批处理大小: {self.batch_sizes}")
        print(f"   最大工作线程: {self.max_workers}")
        if self.max_samples:
            print(f"   最大测试样本数: {self.max_samples}")
    
    def _load_dataset_data(self, dataset: str) -> Tuple[Any, List[Dict[str, Any]]]:
        """加载数据集数据"""
        print(f"📊 加载数据集: {dataset}")
        
        # 更新配置
        self.config.dataset.name = dataset
        udi = UnifiedDataInterface(self.config, dataset)
        loader = udi.get_dataset_loader()
        graphs = udi.get_graphs()
        
        print(f"   ✅ 数据集 {dataset} 加载完成，图数量: {len(graphs)}")
        return loader, graphs
    
    def _initialize_serializer(self, method: str, loader: Any, graphs: List[Dict[str, Any]]) -> Any:
        """初始化序列化器"""
        serializer = SerializerFactory.create_serializer(method)
        
        # 根据方法类型进行不同的初始化
        if method in ("feuler", "fcpp"):
            # 频率引导方法需要全量图进行统计
            serializer.initialize_with_dataset(loader, graphs)
        else:
            try:
                serializer.initialize_with_dataset(loader, graphs)
            except Exception:
                # 回退到无图初始化
                serializer.initialize_with_dataset(loader, None)
        
        return serializer
    
    def _run_single_benchmark(self, method: str, dataset: str, batch_size: int, 
                            parallel: bool) -> BenchmarkResult:
        """运行单个基准测试"""
        try:
            # 加载数据
            loader, graphs = self._load_dataset_data(dataset)
            
            # 限制测试样本数量
            if self.max_samples is not None:
                max_samples = min(len(graphs), self.max_samples)
            else:
                max_samples = len(graphs)  # 使用全部样本
            test_graphs = graphs[:max_samples]
            
            # 初始化序列化器
            serializer = self._initialize_serializer(method, loader, graphs)
            
            # 批处理测试
            batches = [test_graphs[i:i+batch_size] for i in range(0, len(test_graphs), batch_size)]
            
            success_count = 0
            error_count = 0
            
            desc = f"{method}-{dataset}-bs{batch_size}-{'parallel' if parallel else 'serial'}"
            
            print(f"🚀 开始测试: {desc}")
            print(f"   样本总数: {len(test_graphs)}, 批次数: {len(batches)}")
            
            start_time = time.perf_counter()
            
            for batch_idx, batch_graphs in enumerate(batches):
                try:
                    if parallel:
                        batch_results = serializer.batch_serialize(
                            batch_graphs, 
                            desc=f"{desc}-batch{batch_idx}",
                            parallel=True,
                            max_workers=self.max_workers
                        )
                    else:
                        batch_results = serializer.batch_serialize(
                            batch_graphs,
                            desc=f"{desc}-batch{batch_idx}",
                            parallel=False
                        )
                    
                    # 验证结果
                    if batch_results and len(batch_results) == len(batch_graphs):
                        success_count += len(batch_graphs)
                    else:
                        error_count += len(batch_graphs)
                        
                except Exception as e:
                    print(f"   ❌ 批次 {batch_idx} 失败: {str(e)}")
                    error_count += len(batch_graphs)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # 计算性能指标
            avg_time_per_sample = total_time / len(test_graphs) if len(test_graphs) > 0 else 0
            samples_per_second = len(test_graphs) / total_time if total_time > 0 else 0
            
            result = BenchmarkResult(
                method=method,
                dataset=dataset,
                batch_size=batch_size,
                num_samples=len(test_graphs),
                total_time=total_time,
                avg_time_per_sample=avg_time_per_sample,
                parallel_enabled=parallel,
                max_workers=self.max_workers if parallel else 1,
                threads_used=min(self.max_workers, batch_size) if parallel else 1,
                samples_per_second=samples_per_second,
                success_count=success_count,
                error_count=error_count
            )
            
            print(f"   ✅ 测试完成: {samples_per_second:.2f} samples/sec, 成功: {success_count}, 失败: {error_count}")
            return result
            
        except Exception as e:
            print(f"   ❌ 测试失败 {method}-{dataset}-bs{batch_size}: {str(e)}")
            return BenchmarkResult(
                method=method,
                dataset=dataset, 
                batch_size=batch_size,
                num_samples=0,
                total_time=0,
                avg_time_per_sample=0,
                parallel_enabled=parallel,
                max_workers=self.max_workers if parallel else 1,
                threads_used=0,
                samples_per_second=0,
                success_count=0,
                error_count=1
            )
    
    def run_benchmark(self, test_parallel: bool = True) -> None:
        """运行完整的基准测试"""
        print(f"\n🎯 开始序列化方法速度基准测试")
        print(f"{'='*60}")
        
        total_tests = len(self.datasets) * len(self.methods) * len(self.batch_sizes)
        if test_parallel:
            total_tests *= 2  # 串行和并行各测试一次
        
        current_test = 0
        
        for dataset in self.datasets:
            for method in self.methods:
                for batch_size in self.batch_sizes:
                    # 串行测试
                    current_test += 1
                    print(f"\n📝 测试进度: {current_test}/{total_tests}")
                    result_serial = self._run_single_benchmark(method, dataset, batch_size, False)
                    self.results.append(result_serial)
                    
                    if test_parallel:
                        # 并行测试
                        current_test += 1  
                        print(f"\n📝 测试进度: {current_test}/{total_tests}")
                        result_parallel = self._run_single_benchmark(method, dataset, batch_size, True)
                        self.results.append(result_parallel)
        
        print(f"\n🎉 基准测试完成！总共完成 {len(self.results)} 个测试")
    
    def save_results(self, output_path: str = "serialization_benchmark_results.csv") -> None:
        """保存结果到CSV文件"""
        output_path = Path(output_path)
        
        print(f"💾 保存结果到: {output_path}")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            headers = [
                'method', 'dataset', 'batch_size', 'num_samples', 'total_time',
                'avg_time_per_sample', 'samples_per_second', 'parallel_enabled',
                'max_workers', 'threads_used', 'success_count', 'error_count',
                'success_rate'
            ]
            writer.writerow(headers)
            
            # 写入数据
            for result in self.results:
                success_rate = result.success_count / (result.success_count + result.error_count) if (result.success_count + result.error_count) > 0 else 0
                row = [
                    result.method,
                    result.dataset,
                    result.batch_size,
                    result.num_samples,
                    f"{result.total_time:.4f}",
                    f"{result.avg_time_per_sample:.6f}",
                    f"{result.samples_per_second:.2f}",
                    result.parallel_enabled,
                    result.max_workers,
                    result.threads_used,
                    result.success_count,
                    result.error_count,
                    f"{success_rate:.4f}"
                ]
                writer.writerow(row)
        
        print(f"   ✅ 结果已保存")
    
    def print_summary(self) -> None:
        """打印测试结果摘要"""
        print(f"\n📊 基准测试结果摘要")
        print(f"{'='*80}")
        
        # 按方法分组统计
        method_stats = defaultdict(list)
        for result in self.results:
            method_stats[result.method].append(result)
        
        print(f"{'方法':<12} {'数据集':<10} {'批大小':<8} {'并行':<6} {'样本/秒':<12} {'成功率':<8}")
        print(f"{'-'*80}")
        
        for method, results in sorted(method_stats.items()):
            for result in sorted(results, key=lambda x: (x.dataset, x.batch_size, x.parallel_enabled)):
                parallel_str = "是" if result.parallel_enabled else "否"
                success_rate = result.success_count / (result.success_count + result.error_count) if (result.success_count + result.error_count) > 0 else 0
                print(f"{method:<12} {result.dataset:<10} {result.batch_size:<8} {parallel_str:<6} {result.samples_per_second:<12.2f} {success_rate:<8.2%}")
        
        # 找出最快的方法
        if self.results:
            fastest_result = max(self.results, key=lambda x: x.samples_per_second)
            print(f"\n🏆 最快方法: {fastest_result.method} (数据集: {fastest_result.dataset}, "
                  f"批大小: {fastest_result.batch_size}, 并行: {'是' if fastest_result.parallel_enabled else '否'}, "
                  f"速度: {fastest_result.samples_per_second:.2f} samples/sec)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="序列化方法速度基准测试")
    parser.add_argument("--datasets", nargs="+", default=["qm9test"], 
                      help="要测试的数据集列表 (default: ['qm9test'])")
    parser.add_argument("--all-datasets", action="store_true",
                      help="测试所有可用数据集（排除dd数据集）")
    parser.add_argument("--methods", nargs="+", default=None,
                      help="要测试的序列化方法列表 (default: 所有可用方法)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 128],
                      help="要测试的批处理大小列表 (default: [32, 128])")
    parser.add_argument("--max-workers", type=int, default=None,
                      help="最大工作线程数 (default: min(16, CPU核心数))")
    parser.add_argument("--max-samples", type=int, default=None,
                      help="每个数据集最大测试样本数 (default: 无限制)")
    parser.add_argument("--output", type=str, default="serialization_benchmark_results.csv",
                      help="输出CSV文件路径 (default: serialization_benchmark_results.csv)")
    parser.add_argument("--no-parallel", action="store_true",
                      help="跳过并行测试，只进行串行测试")
    
    args = parser.parse_args()
    
    # 处理数据集参数
    if args.all_datasets:
        datasets = get_all_available_datasets(exclude=['dd'])
        print(f"🔍 自动发现的数据集: {datasets}")
        if not datasets:
            print("❌ 未找到任何可用数据集")
            return
    else:
        datasets = args.datasets
    
    # 创建基准测试实例
    benchmark = SerializationBenchmark(
        datasets=datasets,
        methods=args.methods,
        batch_sizes=args.batch_sizes,
        max_workers=args.max_workers,
        max_samples=args.max_samples
    )
    
    try:
        # 运行基准测试
        benchmark.run_benchmark(test_parallel=not args.no_parallel)
        
        # 保存结果
        benchmark.save_results(args.output)
        
        # 打印摘要
        benchmark.print_summary()
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
