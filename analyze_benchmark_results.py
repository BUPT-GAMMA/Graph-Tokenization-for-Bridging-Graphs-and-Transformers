#!/usr/bin/env python3
"""
序列化基准测试结果分析脚本

分析benchmark_serialization_speed.py生成的CSV结果文件，
生成详细的性能对比图表和统计报告
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np


class BenchmarkAnalyzer:
    """基准测试结果分析器"""
    
    def __init__(self, csv_path: str):
        """
        初始化分析器
        
        Args:
            csv_path: CSV结果文件路径
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        print(f"📊 加载基准测试结果: {len(self.df)} 条记录")
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def print_basic_stats(self):
        """打印基本统计信息"""
        print(f"\n📈 基本统计信息")
        print(f"="*50)
        
        print(f"测试的序列化方法: {sorted(self.df['method'].unique())}")
        print(f"测试的数据集: {sorted(self.df['dataset'].unique())}")
        print(f"测试的批处理大小: {sorted(self.df['batch_size'].unique())}")
        print(f"并行测试: {self.df['parallel_enabled'].sum()} 个")
        print(f"串行测试: {(~self.df['parallel_enabled']).sum()} 个")
        
        # 性能统计
        print(f"\n⚡ 性能统计")
        print(f"平均处理速度: {self.df['samples_per_second'].mean():.2f} samples/sec")
        print(f"最高处理速度: {self.df['samples_per_second'].max():.2f} samples/sec")
        print(f"最低处理速度: {self.df['samples_per_second'].min():.2f} samples/sec")
        
        # 成功率统计
        print(f"\n✅ 成功率统计")
        print(f"总体成功率: {self.df['success_rate'].mean():.2%}")
        failed_tests = self.df[self.df['success_rate'] < 1.0]
        if not failed_tests.empty:
            print(f"失败的测试: {len(failed_tests)} 个")
            for _, row in failed_tests.iterrows():
                print(f"  - {row['method']} on {row['dataset']} (bs={row['batch_size']}): {row['success_rate']:.2%}")
    
    def create_performance_comparison_chart(self, output_dir: Path):
        """创建性能对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('序列化方法性能基准测试结果', fontsize=16, fontweight='bold')
        
        # 1. 方法性能对比（条形图）
        ax1 = axes[0, 0]
        method_perf = self.df.groupby('method')['samples_per_second'].mean().sort_values(ascending=False)
        bars = ax1.bar(range(len(method_perf)), method_perf.values)
        ax1.set_xlabel('序列化方法')
        ax1.set_ylabel('平均处理速度 (samples/sec)')
        ax1.set_title('各序列化方法平均性能对比')
        ax1.set_xticks(range(len(method_perf)))
        ax1.set_xticklabels(method_perf.index, rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, method_perf.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 2. 并行 vs 串行性能对比
        ax2 = axes[0, 1]
        parallel_data = []
        methods = self.df['method'].unique()
        for method in methods:
            method_df = self.df[self.df['method'] == method]
            parallel_perf = method_df[method_df['parallel_enabled']]['samples_per_second'].mean()
            serial_perf = method_df[~method_df['parallel_enabled']]['samples_per_second'].mean()
            if not pd.isna(parallel_perf) and not pd.isna(serial_perf):
                parallel_data.append({
                    'method': method,
                    'parallel': parallel_perf,
                    'serial': serial_perf,
                    'speedup': parallel_perf / serial_perf if serial_perf > 0 else 1
                })
        
        if parallel_data:
            parallel_df = pd.DataFrame(parallel_data)
            x = np.arange(len(parallel_df))
            width = 0.35
            ax2.bar(x - width/2, parallel_df['serial'], width, label='串行', alpha=0.8)
            ax2.bar(x + width/2, parallel_df['parallel'], width, label='并行', alpha=0.8)
            ax2.set_xlabel('序列化方法')
            ax2.set_ylabel('处理速度 (samples/sec)')
            ax2.set_title('并行 vs 串行性能对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels(parallel_df['method'], rotation=45)
            ax2.legend()
        
        # 3. 批处理大小影响
        ax3 = axes[1, 0]
        batch_perf = self.df.groupby(['method', 'batch_size'])['samples_per_second'].mean().unstack()
        batch_perf.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_xlabel('序列化方法')
        ax3.set_ylabel('处理速度 (samples/sec)')
        ax3.set_title('不同批处理大小的性能影响')
        ax3.legend(title='批处理大小', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 数据集性能对比
        ax4 = axes[1, 1]
        if len(self.df['dataset'].unique()) > 1:
            dataset_perf = self.df.groupby(['method', 'dataset'])['samples_per_second'].mean().unstack()
            dataset_perf.plot(kind='bar', ax=ax4, width=0.8)
            ax4.set_xlabel('序列化方法')
            ax4.set_ylabel('处理速度 (samples/sec)')
            ax4.set_title('不同数据集的性能对比')
            ax4.legend(title='数据集', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # 如果只有一个数据集，显示成功率
            success_rate = self.df.groupby('method')['success_rate'].mean().sort_values(ascending=False)
            bars = ax4.bar(range(len(success_rate)), success_rate.values)
            ax4.set_xlabel('序列化方法')
            ax4.set_ylabel('成功率')
            ax4.set_title('各序列化方法成功率')
            ax4.set_xticks(range(len(success_rate)))
            ax4.set_xticklabels(success_rate.index, rotation=45)
            ax4.set_ylim(0, 1.1)
            
            # 添加数值标签
            for bar, value in zip(bars, success_rate.values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.1%}', ha='center', va='bottom')
        
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = output_dir / f"serialization_benchmark_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 性能对比图表已保存: {output_path}")
        
        plt.show()
    
    def create_detailed_heatmap(self, output_dir: Path):
        """创建详细的性能热图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 性能热图（方法 x 数据集+批大小）
        ax1 = axes[0]
        pivot_data = self.df.pivot_table(
            values='samples_per_second',
            index='method',
            columns=['dataset', 'batch_size', 'parallel_enabled'],
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('序列化性能热图 (samples/sec)')
        ax1.set_xlabel('数据集-批大小-是否并行')
        ax1.set_ylabel('序列化方法')
        
        # 2. 成功率热图
        ax2 = axes[1]
        success_pivot = self.df.pivot_table(
            values='success_rate',
            index='method', 
            columns=['dataset', 'batch_size', 'parallel_enabled'],
            aggfunc='mean'
        )
        
        sns.heatmap(success_pivot, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax2, vmin=0, vmax=1)
        ax2.set_title('序列化成功率热图')
        ax2.set_xlabel('数据集-批大小-是否并行')
        ax2.set_ylabel('序列化方法')
        
        plt.tight_layout()
        
        # 保存热图
        output_path = output_dir / f"serialization_benchmark_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"🔥 性能热图已保存: {output_path}")
        
        plt.show()
    
    def generate_report(self, output_dir: Path):
        """生成详细的分析报告"""
        report_path = output_dir / "benchmark_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 序列化方法性能基准测试分析报告\n\n")
            f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**数据源**: {self.csv_path.name}\n\n")
            
            # 基本统计
            f.write("## 📊 基本统计信息\n\n")
            f.write(f"- **测试记录数**: {len(self.df)}\n")
            f.write(f"- **测试方法数**: {len(self.df['method'].unique())}\n")
            f.write(f"- **测试数据集**: {', '.join(sorted(self.df['dataset'].unique()))}\n")
            f.write(f"- **批处理大小**: {', '.join(map(str, sorted(self.df['batch_size'].unique())))}\n\n")
            
            # 性能排名
            f.write("## 🏆 性能排名\n\n")
            method_perf = self.df.groupby('method')['samples_per_second'].agg(['mean', 'std', 'max']).sort_values('mean', ascending=False)
            f.write("| 排名 | 方法 | 平均速度 (samples/sec) | 标准差 | 最高速度 |\n")
            f.write("|------|------|----------------------|--------|----------|\n")
            for i, (method, row) in enumerate(method_perf.iterrows(), 1):
                f.write(f"| {i} | {method} | {row['mean']:.2f} | {row['std']:.2f} | {row['max']:.2f} |\n")
            
            # 并行加速比
            f.write("\n## ⚡ 并行加速效果\n\n")
            methods = self.df['method'].unique()
            speedup_data = []
            for method in methods:
                method_df = self.df[self.df['method'] == method]
                parallel_perf = method_df[method_df['parallel_enabled']]['samples_per_second'].mean()
                serial_perf = method_df[~method_df['parallel_enabled']]['samples_per_second'].mean()
                if not pd.isna(parallel_perf) and not pd.isna(serial_perf) and serial_perf > 0:
                    speedup = parallel_perf / serial_perf
                    speedup_data.append((method, serial_perf, parallel_perf, speedup))
            
            if speedup_data:
                speedup_data.sort(key=lambda x: x[3], reverse=True)
                f.write("| 方法 | 串行速度 | 并行速度 | 加速比 |\n")
                f.write("|------|----------|----------|--------|\n")
                for method, serial, parallel, speedup in speedup_data:
                    f.write(f"| {method} | {serial:.2f} | {parallel:.2f} | {speedup:.2f}x |\n")
            
            # 失败案例分析
            failed_tests = self.df[self.df['success_rate'] < 1.0]
            if not failed_tests.empty:
                f.write("\n## ❌ 失败案例分析\n\n")
                for _, row in failed_tests.iterrows():
                    f.write(f"- **{row['method']}** 在数据集 **{row['dataset']}** "
                           f"(批大小={row['batch_size']}, 并行={row['parallel_enabled']}) "
                           f"成功率: {row['success_rate']:.1%}\n")
            
            # 推荐建议
            f.write("\n## 💡 推荐建议\n\n")
            best_method = method_perf.index[0]
            best_speed = method_perf.iloc[0]['mean']
            f.write(f"1. **最佳整体性能**: {best_method} 方法，平均速度 {best_speed:.2f} samples/sec\n")
            
            if speedup_data:
                best_parallel = max(speedup_data, key=lambda x: x[3])
                f.write(f"2. **最佳并行效果**: {best_parallel[0]} 方法，加速比 {best_parallel[3]:.2f}x\n")
            
            # 成功率最高的方法
            success_rate = self.df.groupby('method')['success_rate'].mean().sort_values(ascending=False)
            most_reliable = success_rate.index[0]
            f.write(f"3. **最高可靠性**: {most_reliable} 方法，成功率 {success_rate.iloc[0]:.1%}\n")
            
        print(f"📝 分析报告已生成: {report_path}")
    
    def analyze(self, output_dir: str = "benchmark_analysis"):
        """执行完整分析"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🔍 开始分析基准测试结果...")
        
        # 基本统计
        self.print_basic_stats()
        
        # 生成图表
        self.create_performance_comparison_chart(output_dir)
        self.create_detailed_heatmap(output_dir)
        
        # 生成报告
        self.generate_report(output_dir)
        
        print(f"\n✅ 分析完成！所有结果已保存到 {output_dir} 目录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析序列化基准测试结果")
    parser.add_argument("csv_file", help="基准测试结果CSV文件路径")
    parser.add_argument("--output-dir", default="benchmark_analysis",
                       help="输出目录 (default: benchmark_analysis)")
    
    args = parser.parse_args()
    
    try:
        analyzer = BenchmarkAnalyzer(args.csv_file)
        analyzer.analyze(args.output_dir)
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
