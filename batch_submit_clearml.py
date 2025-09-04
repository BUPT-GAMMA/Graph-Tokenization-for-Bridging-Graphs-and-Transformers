#!/usr/bin/env python3
"""
ClearML批量任务提交脚本

从命令文件中读取任务，创建ClearML任务模板并加入队列，由Agent分布式执行
"""

import sys
import os
import argparse
import shlex
from pathlib import Path
from typing import List, Tuple
import time

# 设置项目路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ClearML 导入
from clearml import Task


class ClearMLBatchSubmitter:
    """ClearML批量任务提交器（创建任务模板，交由Agent执行）"""

    def __init__(self):
        self.working_directory = str(ROOT)
        print(f"Working directory: {self.working_directory}")

    def parse_args(self, args: List[str]) -> List[Tuple[str, str]]:
        """解析命令行参数为 (key, value) 对"""
        parsed_args = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith('--'):
                key = arg[2:]  # 移除 -- 前缀
                if i + 1 < len(args) and not args[i + 1].startswith('--'):
                    value = args[i + 1]
                    parsed_args.append((key, value))
                    i += 2
                else:
                    parsed_args.append((key, "True"))  # 布尔参数
                    i += 1
            else:
                i += 1
        return parsed_args

    def determine_queue(self, parsed_args: List[Tuple[str, str]]) -> str:
        """根据参数确定应该提交到的队列"""
        # 将参数转换为字典以便查询
        args_dict = {key: value for key, value in parsed_args}

        # 获取相关参数的值
        bpe_encode_rank_mode = args_dict.get('bpe_encode_rank_mode', '').lower()
        dataset = args_dict.get('datasets', args_dict.get('dataset', '')).lower()
        method = args_dict.get('methods', args_dict.get('method', '')).lower()

        # 条件判断
        is_raw_bpe = bpe_encode_rank_mode == 'raw'
        has_peptides = 'peptides' in dataset
        is_synthetic_or_dd = any(d in dataset for d in ['synthetic', 'dd'])
        is_eulerian_method = any(m in method for m in ['eulerian', 'feuler'])

        # 根据规则确定队列
        if is_raw_bpe and has_peptides:
            # 同时满足两个条件：raw BPE + peptides dataset
            return "mid"
        elif is_raw_bpe:
            # BPE模式是raw
            return "mid"
        elif has_peptides:
            # dataset含有peptides
            return "mid"
        elif is_synthetic_or_dd:
            # dataset是synthetic或dd
            return "mid"
        elif is_eulerian_method:
            # method是eulerian或feuler
            return "mid"
        else:
            # 默认情况
            return "default"

    def create_task_from_command(self, command_line: str) -> str:
        """从命令行创建ClearML任务，让Agent去执行"""
        parts = shlex.split(command_line.strip())

        # 处理不同类型的脚本调用
        if parts[0] in ['python', 'python3'] and len(parts) > 1:
            # Python脚本调用
            script_path = parts[1]
            args = parts[2:]
        elif parts[0].endswith('.sh') and len(parts) > 0:
            # Bash脚本调用
            script_path = parts[0]
            args = parts[1:]
        else:
            # 其他类型的脚本或命令
            script_path = parts[0]
            args = parts[1:]

        # 生成任务名称
        script_name = Path(script_path).stem
        # task_name = f"{script_name}_{int(time.time())}"

        # 解析参数
        parsed_args = self.parse_args(args)
        parsed_args_dict = {key: value for key, value in parsed_args}
        
        task_name =f'{parsed_args_dict["experiment_group"]}/{parsed_args_dict["experiment_name"]}'

        # 根据参数确定队列
        queue_name = self.determine_queue(parsed_args)

        # 创建任务模板（不执行代码）
        if script_path.endswith('.sh'):
            # 对于bash脚本，直接使用bash脚本作为script，并传递参数
            task = Task.create(
                project_name="TokenizerGraph",
                task_name=task_name,
                script=script_path,
                working_directory=self.working_directory,
                # 对于bash脚本，我们将参数传递给bash脚本
                argparse_args=parsed_args
            )
        else:
            # 对于Python脚本，使用标准模式
            task = Task.create(
                project_name="TokenizerGraph",
                task_name=task_name,
                script=script_path,
                working_directory=self.working_directory,
                argparse_args=parsed_args
            )

        # 加入队列，让Agent执行
        Task.enqueue(task, queue_name=queue_name)

        print(f"✅ 任务已加入队列: {task_name}")
        print(f"   脚本: {script_path}")
        print(f"   队列: {queue_name}")
        print(f"   参数数量: {len(parsed_args)}")
        return task.id

    def submit_from_file(self, task_file: str) -> List[str]:
        """从文件提交任务到ClearML队列"""
        submitted_tasks = []

        with open(task_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                print(f"🔄 处理命令: {line}")
                task_id = self.create_task_from_command(line)
                submitted_tasks.append(task_id)

        return submitted_tasks


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ClearML批量任务提交器")
    parser.add_argument("--file", type=str, default="test_single.txt", help="命令文件路径")

    args = parser.parse_args()

    if not args.file:
        parser.error("必须指定 --file")

    print("🚀 ClearML批量任务提交器")
    print(f"   文件: {args.file}")
    print("   任务将根据参数智能分配到合适的队列:")
    print("   - raw BPE + peptides dataset → high队列")
    print("   - raw BPE模式 → mid队列")
    print("   - 包含peptides的dataset → mid队列")
    print("   - synthetic/dd dataset → mid队列")
    print("   - eulerian/feuler方法 → mid队列")
    print("   - 其他情况 → default队列")
    print("   由Agent分布式执行")

    # 创建提交器
    submitter = ClearMLBatchSubmitter()

    try:
        submitted_tasks = submitter.submit_from_file(args.file)

        print("\n📋 已提交任务:")
        for i, task_id in enumerate(submitted_tasks, 1):
            print(f"   {i}: {task_id}")

        return 0
    except Exception as e:
        print(f"❌ 提交失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
