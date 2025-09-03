#!/usr/bin/env python3
"""
测试批量脚本修改的验证脚本
"""

import subprocess
import sys
from pathlib import Path

def test_batch_scripts():
    """测试修改后的批量脚本"""
    print("🧪 测试批量脚本修改...")

    # 测试预训练脚本的命令生成
    print("\n📋 测试预训练脚本命令生成:")
    test_cmd = [
        "python", "batch_pretrain_simple.py",
        "--datasets", "qm9test",
        "--methods", "feuler",
        "--bpe_scenarios", "all",
        "--commands_stdout",
        "--exp_prefix", "test_"
    ]

    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, cwd="/home/gzy/py/tokenizerGraph")
        if result.returncode == 0:
            print("✅ 预训练脚本测试成功")
            # 检查输出中是否包含了包装脚本
            if "./pretrain_wrapper.sh" in result.stdout:
                print("✅ 预训练脚本正确使用了包装脚本")
            else:
                print("❌ 预训练脚本未使用包装脚本")
        else:
            print(f"❌ 预训练脚本测试失败: {result.stderr}")
    except Exception as e:
        print(f"❌ 预训练脚本测试异常: {e}")

    # 测试微调脚本的命令生成
    print("\n📋 测试微调脚本命令生成:")
    test_cmd = [
        "python", "batch_finetune_simple.py",
        "--datasets", "qm9test",
        "--methods", "feuler",
        "--bpe_scenarios", "all",
        "--commands_stdout",
        "--exp_prefix", "test_"
    ]

    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, cwd="/home/gzy/py/tokenizerGraph")
        if result.returncode == 0:
            print("✅ 微调脚本测试成功")
            # 检查输出中是否包含了包装脚本
            if "./finetune_wrapper.sh" in result.stdout:
                print("✅ 微调脚本正确使用了包装脚本")
            else:
                print("❌ 微调脚本未使用包装脚本")
        else:
            print(f"❌ 微调脚本测试失败: {result.stderr}")
    except Exception as e:
        print(f"❌ 微调脚本测试异常: {e}")

def test_clearml_script():
    """测试ClearML提交脚本的bash脚本支持"""
    print("\n📋 测试ClearML脚本bash支持:")

    # 创建测试命令文件
    test_commands = [
        "./pretrain_wrapper.sh --dataset qm9test --method feuler",
        "./finetune_wrapper.sh --dataset qm9test --method feuler",
        "python run_pretrain.py --dataset qm9test --method feuler"
    ]

    test_file = "/tmp/test_clearml_commands.txt"
    with open(test_file, 'w') as f:
        for cmd in test_commands:
            f.write(cmd + '\n')

    print(f"创建测试命令文件: {test_file}")

    # 这里我们只测试脚本的导入和基本功能，不实际提交到ClearML
    try:
        # 导入脚本的主要类
        sys.path.insert(0, "/home/gzy/py/tokenizerGraph")
        from batch_submit_clearml import ClearMLBatchSubmitter

        submitter = ClearMLBatchSubmitter()

        # 测试命令解析功能
        for cmd in test_commands:
            print(f"\n测试命令: {cmd}")
            task_id = submitter.create_task_from_command(cmd)
            print(f"生成的任务ID: {task_id}")

        print("✅ ClearML脚本测试完成")

    except Exception as e:
        print(f"❌ ClearML脚本测试异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理测试文件
        Path(test_file).unlink(missing_ok=True)

if __name__ == "__main__":
    test_batch_scripts()
    test_clearml_script()
    print("\n🎉 测试完成!")
