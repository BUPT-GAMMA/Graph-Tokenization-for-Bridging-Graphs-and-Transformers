#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
slurm_submit_simple.py — 为任务列表逐条提交单卡 sbatch 作业

用法示例：
  python slurm_submit_simple.py -f tasks.txt

特点：
- 不扫描资源，不指定节点
- 对于任务文件中的每一行（忽略空行与#注释），各自创建一个 sbatch 并提交
- 每个任务固定申请 1 GPU 和指定数量 CPU（默认 2）
"""

import argparse
import os
import re
import subprocess
import sys


SBATCH_SCRIPT_DIR = "sbatch_scripts"


def read_tasks_from_file(tasks_file_path: str):
    """从文本文件读取任务指令（每行一个），忽略空行和以#开头的行。"""
    if not os.path.isfile(tasks_file_path):
        print(f"FATAL: 任务文件不存在: {tasks_file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(tasks_file_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
    except OSError as e:
        print(f"FATAL: 读取任务文件失败: {tasks_file_path}", file=sys.stderr)
        print(f"错误详情: {e}", file=sys.stderr)
        sys.exit(1)

    tasks = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        tasks.append(stripped)

    if not tasks:
        print("FATAL: 任务文件在移除空行/注释后为空。", file=sys.stderr)
        sys.exit(1)

    return tasks


def sanitize_task_command(task: str) -> str:
    """移除命令中显式设置的 CUDA_VISIBLE_DEVICES，以避免与 Slurm 分配冲突。"""
    tokens = task.strip().split()
    filtered_tokens = [t for t in tokens if not t.startswith("CUDA_VISIBLE_DEVICES=")]
    sanitized = " ".join(filtered_tokens).strip()
    return sanitized


def generate_single_task_sbatch_script(task: str, cpus_per_task: int, job_name: str, partition: str = None) -> str:
    """生成单个任务的 sbatch 脚本内容。"""
    sanitized_task = sanitize_task_command(task)
    extra_directives = f"#SBATCH --partition={partition}\n" if partition else ""
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus_per_task}
{extra_directives}#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err


export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
source /home/fit/shichuan/WORK/miniconda3/envs/pthgnn/bin/activate
echo "INFO: Using Python at $(which python)"

mkdir -p logs
echo "INFO: Starting {job_name}..."
srun --exclusive -n1 {sanitized_task}
echo "INFO: Finished {job_name}."
"""
    return script_content


def parse_sbatch_job_id(sbatch_stdout: str):
    """从 sbatch 输出中提取 JobID，例如 'Submitted batch job 12345'。"""
    match = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
    if not match:
        return None
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description="按任务文件逐条提交单卡 sbatch 作业（不扫描资源不指定节点）")
    parser.add_argument("--tasks-file", "-f", default="commands.list", type=str, help="包含任务指令的文本文件（每行一个）")
    parser.add_argument("--cpus-per-task", type=int, default=2, help="每个任务申请的 CPU 数（默认 2）")
    parser.add_argument("--job-name-prefix", type=str, default="gzy_task", help="作业名前缀（默认 gzy_task）")
    parser.add_argument("--partition", "-p", type=str, default='a01', help="Slurm 分区名（如 a01）。若集群无默认分区，则必须指定")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要提交的任务，不真正提交")
    args = parser.parse_args()

    tasks = read_tasks_from_file(args.tasks_file)

    print("\n" + "=" * 40)
    print(" 提交计划预览")
    print("=" * 40)
    print(f"任务数量: {len(tasks)}")
    print(f"CPU/任务: {args.cpus_per_task}")
    print("每个任务固定申请 1 GPU")
    print("=" * 40 + "\n")

    if args.dry_run:
        for i, task in enumerate(tasks, start=1):
            job_name = f"{args.job_name_prefix}_{i}"
            batch_content = generate_single_task_sbatch_script(task, args.cpus_per_task, job_name, partition=args.partition)
            print(f"[{i}] {job_name}: {batch_content}")
        print("\nDRY RUN: 未提交任何作业。")
        return

    if not os.path.exists(SBATCH_SCRIPT_DIR):
        os.makedirs(SBATCH_SCRIPT_DIR)

    submitted = 0
    for i, task in enumerate(tasks, start=1):
        job_name = f"{args.job_name_prefix}-{i}"
        script_content = generate_single_task_sbatch_script(task, args.cpus_per_task, job_name, partition=args.partition)
        script_path = os.path.join(SBATCH_SCRIPT_DIR, f"submit_{job_name}.sh")

        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
        except OSError as e:
            print(f"FATAL: 写入 sbatch 脚本失败: {script_path}", file=sys.stderr)
            print(f"错误详情: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            proc = subprocess.run(["sbatch", script_path], check=True, capture_output=True, text=True)
            job_id = parse_sbatch_job_id(proc.stdout)
            print(f"-> Submitted {job_name}  JobID={job_id if job_id else 'UNKNOWN'}")
            submitted += 1
        except subprocess.CalledProcessError as e:
            print(f"FATAL: 提交 {job_name} 失败。", file=sys.stderr)
            print(f"stderr: {e.stderr}", file=sys.stderr)
            sys.exit(1)

    print(f"完成：成功提交 {submitted}/{len(tasks)} 个任务。")


if __name__ == "__main__":
    main()




