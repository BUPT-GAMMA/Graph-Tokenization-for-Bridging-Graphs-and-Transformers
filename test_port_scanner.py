#!/usr/bin/env python3
"""
端口扫描脚本测试 - 只扫描几个IP来验证tqdm进度条功能
"""

import socket
import time
from tqdm import tqdm

def scan_port(ip, port, timeout=1):
    """扫描指定IP和端口"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def test_scanner():
    """测试函数，只扫描10.122.0.0到10.122.0.10这11个IP"""
    port = 20001
    test_ips = [f"10.122.0.{i}" for i in range(11)]  # 0-10

    print("测试端口扫描脚本 (只扫描10.122.0.0-10.122.0.10)")
    print("-" * 50)

    open_ports = []

    with tqdm(total=len(test_ips), desc="测试扫描进度", unit="IP",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              colour='blue') as pbar:

        for ip in test_ips:
            is_open = scan_port(ip, port)
            if is_open:
                open_ports.append(ip)
                pbar.set_description(f"测试扫描 (发现{len(open_ports)}个开放端口)")
                print(f"\n[发现开放端口] {ip}:{port}")

            pbar.update(1)
            pbar.set_postfix(current_ip=ip, open_ports=len(open_ports))
            time.sleep(0.1)  # 快速测试用0.1秒

    print("\n测试完成!")
    print(f"发现 {len(open_ports)} 个开放的端口")
    if open_ports:
        print("开放端口列表:")
        for ip in open_ports:
            print(f"  {ip}:{port}")

if __name__ == "__main__":
    test_scanner()
