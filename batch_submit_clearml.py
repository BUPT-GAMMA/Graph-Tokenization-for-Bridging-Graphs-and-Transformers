#!/usr/bin/env python3
"""
批量提交任务到 ClearML 平台的脚本

使用 Task.create() 方法创建任务模板，无需修改 run 脚本
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
    """ClearML 批量任务提交器"""

    def __init__(self, project_name: str = "TokenizerGraph", queue_name: str = "default"):
        self.project_name = project_name
        self.queue_name = queue_name
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
        """从命令行创建 ClearML 任务"""
        parts = shlex.split(command_line.strip())

        # 处理 python 命令的情况
        if parts[0] == 'python' and len(parts) > 1:
            script_path = parts[1]
            args = parts[2:]
        else:
            script_path = parts[0]
            args = parts[1:]

        # 生成任务名称
        script_name = Path(script_path).stem
        task_name = f"{script_name}_{int(time.time())}"

        # 解析参数 - ClearML需要完整的参数名
        parsed_args = self.parse_args(args)

        # 确保脚本路径是相对于repo的
        script_relative_path = os.path.relpath(script_path, self.working_directory)

        # 创建任务
        task = Task.create(
            project_name=self.project_name,
            task_name=task_name,
            repo=self.working_directory,  # 使用绝对路径作为repo
            script=script_relative_path,  # 相对于repo的脚本路径
            working_directory=self.working_directory,  # 工作目录
            argparse_args=parsed_args,  # 参数列表
            add_task_init_call=True  # 自动添加 Task.init() 调用
        )

        # 加入队列
        Task.enqueue(task, queue_name=self.queue_name)

        print(f"✅ 创建任务: {task_name} (ID: {task.id})")
        print(f"   脚本: {script_relative_path}")
        print(f"   参数: {parsed_args}")
        return task.id

    def submit_from_file(self, task_file: str) -> List[str]:
        """从文件提交任务"""
        submitted_tasks = []

        with open(task_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                print(f"🔄 处理: {line}")
                task_id = self.create_task_from_command(line)
                submitted_tasks.append(task_id)

        return submitted_tasks


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量提交任务到 ClearML 平台")
    parser.add_argument("--file", type=str,default="test_single.txt", help="任务指令文件路径")
    parser.add_argument("--project", type=str, default="TokenizerGraph", help="ClearML 项目名称")
    parser.add_argument("--queue", type=str, default="default", help="ClearML 队列名称")

    args = parser.parse_args()

    if not args.file:
        parser.error("必须指定 --file")

    print("🚀 ClearML 批量任务提交器")
    print(f"   项目: {args.project}")
    print(f"   队列: {args.queue}")
    print(f"   文件: {args.file}")

    # 创建提交器
    submitter = ClearMLBatchSubmitter(args.project, args.queue)

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
