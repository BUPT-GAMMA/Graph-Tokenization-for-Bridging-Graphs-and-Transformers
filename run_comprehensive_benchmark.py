#!/usr/bin/env python3
"""
全面的序列化方法基准测试脚本

测试所有可用的数据集（排除dd）和序列化方法，每个数据集限制1000个样本
包括批处理大小32和128，以及串行和并行处理对比
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse


def run_comprehensive_benchmark(max_samples: int = 1000, batch_sizes: str = "32 128", 
                               output_dir: str = "comprehensive_benchmark_results", 
                               max_workers: int = None):
    """运行全面的基准测试"""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🚀 开始全面的序列化方法基准测试")
    print("="*60)
    print(f"📊 每个数据集最大样本数: {max_samples}")
    print(f"📦 批处理大小: {batch_sizes}")
    print(f"📁 结果保存目录: {output_path}")
    print(f"🕒 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 构建基本命令
    base_cmd = [
        sys.executable, "benchmark_serialization_speed.py",
        "--all-datasets",
        "--batch-sizes"] + batch_sizes.split() + [
        "--max-samples", str(max_samples),
    ]
    
    if max_workers:
        base_cmd.extend(["--max-workers", str(max_workers)])
    
    results_files = []
    
    try:
        # 1. 完整测试（串行+并行）
        print("🔄 第一阶段: 完整测试（串行 + 并行）")
        print("-" * 40)
        full_output = output_path / f"full_benchmark_{timestamp}.csv"
        cmd_full = base_cmd + ["--output", str(full_output)]
        
        print(f"运行命令: {' '.join(cmd_full)}")
        result = subprocess.run(cmd_full, check=True, capture_output=False)
        results_files.append(full_output)
        print(f"✅ 完整测试完成，结果保存至: {full_output}")
        print()
        
        # 2. 仅串行测试（用于对比）
        print("🔄 第二阶段: 仅串行测试")
        print("-" * 40)
        serial_output = output_path / f"serial_only_benchmark_{timestamp}.csv"
        cmd_serial = base_cmd + ["--output", str(serial_output), "--no-parallel"]
        
        print(f"运行命令: {' '.join(cmd_serial)}")
        result = subprocess.run(cmd_serial, check=True, capture_output=False)
        results_files.append(serial_output)
        print(f"✅ 串行测试完成，结果保存至: {serial_output}")
        print()
        
        print("🎉 全面基准测试完成！")
        print("="*60)
        print("📄 生成的结果文件:")
        for rf in results_files:
            print(f"  - {rf}")
        
        # 自动生成分析报告
        print()
        print("📊 正在生成分析报告...")
        for result_file in results_files:
            analysis_cmd = [
                sys.executable, "analyze_benchmark_results.py", 
                str(result_file), 
                "--output-dir", str(output_path / result_file.stem)
            ]
            try:
                subprocess.run(analysis_cmd, check=True, capture_output=False)
                print(f"✅ 分析报告已生成: {output_path / result_file.stem}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  分析报告生成失败: {e}")
        
        print()
        print("🔍 使用建议:")
        print(f"  - 查看完整结果: cat {results_files[0] if results_files else 'N/A'}")
        print(f"  - 查看分析报告: ls {output_path}/*/benchmark_analysis_report.md")
        print(f"  - 查看图表: ls {output_path}/*/*.png")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 基准测试失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生意外错误: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="全面序列化基准测试")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="每个数据集最大测试样本数 (default: 1000)")
    parser.add_argument("--batch-sizes", type=str, default="32 128",
                       help="批处理大小，空格分隔 (default: '32 128')")
    parser.add_argument("--output-dir", type=str, default="comprehensive_benchmark_results",
                       help="结果输出目录 (default: comprehensive_benchmark_results)")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="最大工作线程数 (default: auto)")
    
    args = parser.parse_args()
    
    run_comprehensive_benchmark(
        max_samples=args.max_samples,
        batch_sizes=args.batch_sizes,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
