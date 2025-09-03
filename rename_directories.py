#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目录重命名脚本
将指定目录下所有以 'default' 结尾的目录重命名为不带 'default' 的版本
"""

import os
import shutil
from pathlib import Path

def rename_default_directories(base_path):
    """
    将指定目录下所有以 'default' 结尾的目录重命名为不带 'default' 的版本

    Args:
        base_path (str): 基础目录路径
    """
    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"错误：目录 {base_path} 不存在")
        return

    if not base_dir.is_dir():
        print(f"错误：{base_path} 不是一个目录")
        return

    print(f"开始处理目录：{base_path}")
    renamed_count = 0

    # 遍历目录下的所有项目
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.endswith('_default'):
            # 构造新名称：去掉末尾的 '_default'
            new_name = item.name[:-8]  # 去掉 '_default' (8个字符)
            new_path = item.parent / new_name

            # 检查新名称是否已存在
            if new_path.exists():
                try:
                    # 删除已存在的目标目录
                    import shutil
                    shutil.rmtree(new_path)
                    print(f"删除已存在的目录：{new_path}")
                except Exception as e:
                    print(f"错误：删除目录失败 {new_path}: {str(e)}")
                    continue

            try:
                # 重命名目录
                item.rename(new_path)
                print(f"重命名：{item.name} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"错误：重命名失败 {item.name} -> {new_name}: {str(e)}")

    print(f"完成！共重命名了 {renamed_count} 个目录")

if __name__ == "__main__":
    # 指定目标目录
    target_directory = "/home/gzy/py/tokenizerGraph/model/pre_comp/mult/1"

    rename_default_directories(target_directory)
