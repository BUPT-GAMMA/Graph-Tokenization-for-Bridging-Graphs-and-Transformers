#!/usr/bin/env python3
import torch
import subprocess
import os
import sys
import socket
from clearml_agent.config import get_config


def get_gpu_count():
    """获取GPU数量"""
    return torch.cuda.device_count()


def get_hostname():
    """获取主机名"""
    return socket.gethostname()


def get_queues_by_hostname(hostname):
    """根据主机名确定应该监听的队列列表"""
    hostname_lower = hostname.lower()

    if "2080" in hostname_lower:
        # 显存较低的机器，只监听default队列
        return ["default"]
    elif "a800" in hostname_lower:
        # 显存最高的机器，监听所有三个队列（按优先级顺序）
        return ["high"]
    else:
        # 其他机器，监听mid和default队列（按优先级顺序）
        return ["mid", "default"]


def get_cache_dir(hostname, gpu_id):
    """根据主机名和GPU ID生成缓存目录"""
    base_cache_dir = "/tmp/clearml_cache"
    return f"{base_cache_dir}/{hostname}_gpu_{gpu_id}"


def get_venv_dir(hostname, gpu_id):
    """根据主机名和GPU ID生成虚拟环境目录"""
    base_venv_dir = "/tmp/clearml_venv"
    return f"{base_venv_dir}/{hostname}_gpu_{gpu_id}"


def configure_clearml_agent(cache_dir, venv_dir):
    """配置ClearML Agent的缓存和虚拟环境路径"""
    try:
        # 使用环境变量来设置缓存路径
        os.environ['CLEARML_CACHE_DIR'] = cache_dir
        os.environ['CLEARML_VENV_DIR'] = venv_dir

        print(f"✅ 设置环境变量 CLEARML_CACHE_DIR = {cache_dir}")
        print(f"✅ 设置环境变量 CLEARML_VENV_DIR = {venv_dir}")

        return True
    except Exception as e:
        print(f"❌ ClearML配置失败: {e}")
        return False


def start_agents():
    """为每张GPU启动一个agent"""
    gpu_count = get_gpu_count()
    hostname = get_hostname()
    queues = get_queues_by_hostname(hostname)
    print(f"检测到 {gpu_count} 张GPU，主机名: {hostname}")
    print(f"根据主机名 '{hostname}' 确定监听队列: {queues}")

    for i in range(gpu_count):
        # 为每个agent设置独立的缓存和虚拟环境目录
        cache_dir = get_cache_dir(hostname, i)
        venv_dir = get_venv_dir(hostname, i)

        # 创建目录
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(venv_dir, exist_ok=True)

        print(f"\n🔧 配置GPU {i} 的ClearML Agent...")
        # 配置ClearML Agent
        if not configure_clearml_agent(cache_dir, venv_dir):
            print(f"⚠️  GPU {i} 的配置失败，但继续启动agent")

        # 构建启动命令，根据队列列表生成多个--queue参数
        queue_args = " ".join(queues)
        cmd = f'CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/home/gzy/miniconda3/envs/pthgnn/bin/python clearml-agent daemon  --queue {queue_args} --gpus {i} --detached'

        print(f"🚀 启动GPU {i} 的agent")
        print(f"   缓存目录: {cache_dir}")
        print(f"   虚拟环境目录: {venv_dir}")
        print(f"   监听队列: {queues}")
        print(f"   执行命令: {cmd}")

        result = os.system(cmd)

        if result == 0:
            print(f"✅ GPU {i} 的agent启动成功")
        else:
            print(f"❌ GPU {i} 的agent启动失败 (退出码: {result})")


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
        print("  start   - 为每张GPU启动agent (每个agent有独立的缓存和虚拟环境目录)")
        print("  stop    - 停止所有agent")
        print("  status  - 查看agent状态")
        print()
        print("功能特性:")
        print("  - 每个agent使用独立的缓存目录: /tmp/clearml_cache/{hostname}_gpu_{id}")
        print("  - 每个agent使用独立的虚拟环境目录: /tmp/clearml_venv/{hostname}_gpu_{id}")
        print("  - 通过环境变量 CLEARML_CACHE_DIR 和 CLEARML_VENV_DIR 配置")
        print("  - 避免NFS文件句柄冲突和并发访问问题")
        print("  - 智能队列分配:")
        print("    * 主机名包含 '2080' 的机器: 只监听 'default' 队列")
        print("    * 主机名包含 'A800' 的机器: 监听 'high', 'mid', 'default' 队列")
        print("    * 其他机器: 监听 'mid', 'default' 队列")
        print()
        print("示例:")
        print("  python start_clearml_agents.py start")
        print("  python start_clearml_agents.py stop")
        print("  python start_clearml_agents.py status")
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
