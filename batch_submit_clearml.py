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
        task_name = f"{script_name}_{int(time.time())}"

        # 解析参数
        parsed_args = self.parse_args(args)

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
        Task.enqueue(task, queue_name="default")

        print(f"✅ 任务已加入队列: {task_name}")
        print(f"   脚本: {script_path}")
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
    print("   任务将加入队列，由Agent分布式执行")

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
