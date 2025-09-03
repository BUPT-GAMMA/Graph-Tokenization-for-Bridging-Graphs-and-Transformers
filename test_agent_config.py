#!/usr/bin/env python3
"""
测试ClearML Agent环境变量配置
"""

import os
import socket
import subprocess
import sys

def test_agent_config():
    """测试agent配置"""
    hostname = socket.gethostname()
    gpu_id = 0  # 测试GPU 0

    cache_dir = f"/tmp/clearml_cache/{hostname}_gpu_{gpu_id}"
    venv_dir = f"/tmp/clearml_venv/{hostname}_gpu_{gpu_id}"

    # 创建目录
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(venv_dir, exist_ok=True)

    # 设置环境变量
    os.environ['CLEARML_CACHE_DIR'] = cache_dir
    os.environ['CLEARML_VENV_DIR'] = venv_dir

    print(f"主机名: {hostname}")
    print(f"缓存目录: {cache_dir}")
    print(f"虚拟环境目录: {venv_dir}")
    print(f"CLEARML_CACHE_DIR: {os.environ.get('CLEARML_CACHE_DIR')}")
    print(f"CLEARML_VENV_DIR: {os.environ.get('CLEARML_VENV_DIR')}")

    # 测试环境变量是否传递给子进程
    cmd = ['env']
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if 'CLEARML_CACHE_DIR' in result.stdout and 'CLEARML_VENV_DIR' in result.stdout:
        print("✅ 环境变量正确传递给子进程")
    else:
        print("❌ 环境变量未传递给子进程")

    return True

if __name__ == "__main__":
    test_agent_config()
