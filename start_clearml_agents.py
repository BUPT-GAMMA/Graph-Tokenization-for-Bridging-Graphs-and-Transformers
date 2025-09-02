#!/usr/bin/env python3
import torch
import subprocess
import os
import sys


def get_gpu_count():
    """获取GPU数量"""
    return torch.cuda.device_count()


def start_agents():
    """为每张GPU启动一个agent"""
    gpu_count = get_gpu_count()
    print(f"检测到 {gpu_count} 张GPU")

    for i in range(gpu_count):
        cmd = f'CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/home/gzy/miniconda3/envs/pthgnn/bin/python clearml-agent daemon --queue default --gpus {i} --detached'
        print(f"启动GPU {i} 的agent")
        os.system(cmd)


def stop_agents():
    """停止所有ClearML agent"""
    print("停止所有ClearML agent...")
    os.system("pkill -f clearml-agent")
    print("已停止所有clearml-agent进程")


def status_agents():
    """查看ClearML agent状态"""
    print("ClearML Agent状态:")
    result = os.system("clearml-agent list 2>/dev/null")
    if result != 0:
        print("未发现运行中的agent")


def main():
    if len(sys.argv) < 2:
        print("用法: python start_clearml_agents.py <command>")
        print("命令:")
        print("  start   - 为每张GPU启动agent")
        print("  stop    - 停止所有agent")
        print("  status  - 查看agent状态")
        print()
        print("示例:")
        print("  python start_clearml_agents.py start")
        print("  python start_clearml_agents.py stop")
        return

    command = sys.argv[1].lower()

    if command == 'start':
        start_agents()
    elif command == 'stop':
        stop_agents()
    elif command == 'status':
        status_agents()
    else:
        print(f"未知命令: {command}")
        print("可用命令: start, stop, status")


if __name__ == "__main__":
    main()
