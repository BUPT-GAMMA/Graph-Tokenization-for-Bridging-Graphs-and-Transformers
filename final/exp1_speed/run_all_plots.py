#!/usr/bin/env python3
"""
效率对比实验 - 一键生成所有图表
===============================

运行所有三个子实验的画图脚本：
1. 序列化长度对比 (token_length/)
2. 序列化速度对比 (serialize_time/)
3. 训练效率对比 (train_time/)

支持多重采样对比和数据集筛选

使用方式：
python run_all_plots.py [--show] [--mult MULT...] [--datasets DATASETS...]

参数：
--show: 显示生成的图表（默认不显示，只保存）
--mult: 要比较的多重采样次数，如 --mult 1 5 10
--datasets: 要处理的数据集名称，如 --datasets qm9test zinc
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加plot_utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from plot_utils import setup_matplotlib_style

def run_script(script_path: str, description: str, show_plots: bool = False,
               mult_values: list = None, datasets: list = None) -> bool:
    """
    运行指定的画图脚本

    Args:
        script_path: 脚本路径
        description: 脚本描述
        show_plots: 是否显示图表
        mult_values: 要比较的多重采样次数列表（仅对plot_token_length.py有效）
        datasets: 要处理的数据集列表（仅对plot_token_length.py有效）

    Returns:
        是否成功运行
    """
    print(f"\n{'='*50}")
    print(f"运行 {description}")
    print(f"脚本路径: {script_path}")
    print('='*50)
    
    if not os.path.exists(script_path):
        print(f"❌ 脚本文件不存在: {script_path}")
        return False
    
    try:
        # 获取脚本所在目录
        script_dir = os.path.dirname(script_path)

        # 构建命令行参数
        cmd = [sys.executable, script_path]

        # 特殊处理plot_token_length.py，传递多重采样和数据集参数
        if 'plot_token_length.py' in script_path:
            if mult_values:
                cmd.extend(['--mult'] + [str(m) for m in mult_values])
            if datasets:
                cmd.extend(['--datasets'] + datasets)

        # 设置环境变量
        env = os.environ.copy()
        if not show_plots:
            env['MPLBACKEND'] = 'Agg'  # 使用非交互式后端

        # 运行脚本
        result = subprocess.run(cmd,
                              cwd=script_dir,
                              capture_output=True,
                              text=True,
                              env=env)
        
        if result.returncode == 0:
            print("✅ 脚本运行成功")
            # 打印成功信息，但不打印详细输出（避免中文字体警告信息过多）
            if "图片已保存到" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "图片已保存到" in line or "✓ 成功生成" in line or "所有图表绘制完成" in line:
                        print(f"  {line}")
            return True
        else:
            print("❌ 脚本运行失败")
            print(f"错误代码: {result.returncode}")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 运行脚本时出现异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='生成所有效率对比图表')
    parser.add_argument('--show', action='store_true',
                       help='显示生成的图表（默认只保存不显示）')
    parser.add_argument('--mult', type=int, nargs='+',
                       help='要比较的多重采样次数，如 --mult 1 5 10')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='要处理的数据集名称，如 --datasets qm9test zinc')
    args = parser.parse_args()
    
    # 设置matplotlib样式
    setup_matplotlib_style()
    
    # 获取当前脚本所在目录
    base_dir = Path(__file__).parent
    
    print("🚀 开始生成效率对比实验的所有图表...")
    print(f"基础目录: {base_dir}")

    if args.mult:
        print(f"多重采样次数: {args.mult}")
    if args.datasets:
        print(f"指定数据集: {args.datasets}")

    # 定义所有要运行的脚本
    scripts = [
        {
            'path': base_dir / 'token_length' / 'plot_token_length.py',
            'description': '序列化长度对比图'
        },
        {
            'path': base_dir / 'serialize_time' / 'plot_serialize_speed.py',
            'description': '序列化速度对比图'
        },
        {
            'path': base_dir / 'train_time' / 'plot_train_efficiency.py',
            'description': '训练效率对比图'
        }
    ]

    # 运行所有脚本
    success_count = 0
    total_count = len(scripts)

    for script_info in scripts:
        script_path = str(script_info['path'])
        description = script_info['description']

        # 只对plot_token_length.py传递额外参数，其他脚本使用默认参数
        if 'plot_token_length.py' in script_path:
            if run_script(script_path, description, args.show, args.mult, args.datasets):
                success_count += 1
        else:
            if run_script(script_path, description, args.show):
                success_count += 1
    
    # 总结报告
    print(f"\n{'='*60}")
    print("📊 图表生成完成总结")
    print('='*60)
    print(f"成功生成: {success_count}/{total_count} 个图表")
    
    if success_count == total_count:
        print("🎉 所有图表都已成功生成！")
        
        # 显示生成的文件位置
        print("\n📂 生成的图表文件位置：")
        output_dirs = [
            base_dir / 'token_length',
            base_dir / 'serialize_time', 
            base_dir / 'train_time'
        ]
        
        for output_dir in output_dirs:
            jpg_files = list(output_dir.glob('*.jpg'))
            if jpg_files:
                print(f"\n  {output_dir.name}/:")
                for jpg_file in jpg_files:
                    print(f"    - {jpg_file.name}")
        
        print(f"\n💡 提示:")
        print(f"  - 所有图表都已保存为JPG格式（300 DPI）")
        print(f"  - 如需显示图表，请使用 --show 参数")
        print(f"  - 要添加新数据集，请在各文件夹下添加对应的CSV文件")
        if args.mult:
            print(f"  - 多重采样对比: {args.mult}")
        if args.datasets:
            print(f"  - 数据集筛选: {args.datasets}")
        
    else:
        failed_count = total_count - success_count
        print(f"⚠️  有 {failed_count} 个图表生成失败，请检查错误信息")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
