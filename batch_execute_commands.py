#!/usr/bin/env python3
"""
批量执行命令脚本（每个命令自动创建ClearML任务）

直接执行命令，让每个run脚本自己处理ClearML任务创建和TensorBoard日志记录
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


class CommandBatchExecutor:
    """批量命令执行器（每个命令自动创建ClearML任务）"""

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

    def execute_command(self, command_line: str) -> str:
        """直接执行命令，让run脚本自己处理ClearML任务"""
        print(f"🔄 执行命令: {command_line}")

        # 直接执行命令，run脚本会自己调用Task.init()
        result = subprocess.run(
            command_line,
            shell=True,
            cwd=self.working_directory,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✅ 命令执行成功")
            # 从输出中提取任务ID（如果有的话）
            # 这里可以根据实际输出格式来解析
            task_id = f"executed_{int(time.time())}"
            return task_id
        else:
            print(f"❌ 命令执行失败: {result.stderr}")
            raise Exception(f"Command failed: {result.stderr}")

    def submit_from_file(self, task_file: str) -> List[str]:
        """从文件提交任务"""
        submitted_tasks = []

        with open(task_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                task_id = self.execute_command(line)
                submitted_tasks.append(task_id)

        return submitted_tasks


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量执行命令，每个命令会自动创建ClearML任务")
    parser.add_argument("--file", type=str, default="test_single.txt", help="命令文件路径")

    args = parser.parse_args()

    if not args.file:
        parser.error("必须指定 --file")

    print("🚀 批量命令执行器（自动创建ClearML任务）")
    print(f"   文件: {args.file}")

    # 创建执行器
    submitter = CommandBatchExecutor()

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
