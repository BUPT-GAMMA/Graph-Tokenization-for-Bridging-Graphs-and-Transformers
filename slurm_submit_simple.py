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
import math


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
#SBATCH --time=12:00:00 
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
    # 两阶段依赖：将任务分为前后两批，后半批依赖前半批完成
    parser.add_argument("--phase-split", type=str, default=None,
                        help="两阶段拆分：整数=前一阶段任务数量；0-1小数=比例。示例: 10 或 0.5")
    parser.add_argument("--dependency-type", type=str, default="afterok", choices=["afterok", "afterany"],
                        help="第二阶段依赖类型：afterok=仅前一阶段全部成功后启动；afterany=前一阶段全部结束(含失败)后启动")
    args = parser.parse_args()

    tasks = read_tasks_from_file(args.tasks_file)

    print("\n" + "=" * 40)
    print(" 提交计划预览")
    print("=" * 40)
    print(f"任务数量: {len(tasks)}")
    print(f"CPU/任务: {args.cpus_per_task}")
    print("每个任务固定申请 1 GPU")
    print("=" * 40 + "\n")

    # 计算两阶段拆分索引
    split_idx = None
    if args.phase_split is not None:
        try:
            val = float(args.phase_split)
            if 0 < val < 1:
                split_idx = max(1, min(len(tasks) - 1, math.floor(len(tasks) * val)))
            else:
                split_idx = int(val)
                split_idx = max(1, min(len(tasks) - 1, split_idx))
        except ValueError:
            print(f"FATAL: --phase-split 参数无效: {args.phase_split}", file=sys.stderr)
            sys.exit(1)

    if args.dry_run:
        if split_idx is None:
            for i, task in enumerate(tasks, start=1):
                job_name = f"{args.job_name_prefix}_{i}"
                batch_content = generate_single_task_sbatch_script(task, args.cpus_per_task, job_name, partition=args.partition)
                print(f"[{i}] {job_name}: {batch_content}")
        else:
            print(f"[Phase 1] 任务数: {split_idx}")
            for i, task in enumerate(tasks[:split_idx], start=1):
                job_name = f"{args.job_name_prefix}_{i}"
                batch_content = generate_single_task_sbatch_script(task, args.cpus_per_task, job_name, partition=args.partition)
                print(f"[{i}] {job_name}: {batch_content}")
            print(f"\n[Phase 2] 任务数: {len(tasks) - split_idx}")
            dep_placeholder = f"{args.dependency_type}:<PHASE1_JOB_IDS_COLON_SEPARATED>"
            for j, task in enumerate(tasks[split_idx:], start=split_idx + 1):
                job_name = f"{args.job_name_prefix}_{j}"
                batch_content = generate_single_task_sbatch_script(task, args.cpus_per_task, job_name, partition=args.partition)
                print(f"[{j}] {job_name} (依赖: {dep_placeholder}): {batch_content}")
        print("\nDRY RUN: 未提交任何作业。")
        return

    if not os.path.exists(SBATCH_SCRIPT_DIR):
        os.makedirs(SBATCH_SCRIPT_DIR)

    submitted = 0
    phase1_job_ids = []
    def _submit_one(script_path: str, job_name: str, dependency_expr: str | None = None) -> str | None:
        try:
            cmd = ["sbatch"]
            if dependency_expr:
                cmd.append(f"--dependency={dependency_expr}")
            cmd.append(script_path)
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            job_id = parse_sbatch_job_id(proc.stdout)
            print(f"-> Submitted {job_name}  JobID={job_id if job_id else 'UNKNOWN'}{('  DEP='+dependency_expr) if dependency_expr else ''}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"FATAL: 提交 {job_name} 失败。", file=sys.stderr)
            print(f"stderr: {e.stderr}", file=sys.stderr)
            sys.exit(1)

    # 写文件+提交 的小工具
    def _write_script(job_name: str, content: str) -> str:
        script_path = os.path.join(SBATCH_SCRIPT_DIR, f"submit_{job_name}.sh")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            print(f"FATAL: 写入 sbatch 脚本失败: {script_path}", file=sys.stderr)
            print(f"错误详情: {e}", file=sys.stderr)
            sys.exit(1)
        return script_path

    if split_idx is None:
        for i, task in enumerate(tasks, start=1):
            job_name = f"{args.job_name_prefix}-{i}"
            content = generate_single_task_sbatch_script(task, args.cpus_per_task, job_name, partition=args.partition)
            path = _write_script(job_name, content)
            _ = _submit_one(path, job_name)
            submitted += 1
    else:
        # Phase 1
        for i, task in enumerate(tasks[:split_idx], start=1):
            job_name = f"{args.job_name_prefix}-{i}"
            content = generate_single_task_sbatch_script(task, args.cpus_per_task, job_name, partition=args.partition)
            path = _write_script(job_name, content)
            jid = _submit_one(path, job_name)
            if jid:
                phase1_job_ids.append(jid)
            submitted += 1

        # Phase 2 依赖 Phase 1 完成
        if not phase1_job_ids:
            print("WARNING: 第一阶段未获得有效 JobID，第二阶段将不设置依赖直接提交。")
            dep_expr = None
        else:
            dep_expr = f"{args.dependency_type}:{':'.join(phase1_job_ids)}"

        for j, task in enumerate(tasks[split_idx:], start=split_idx + 1):
            job_name = f"{args.job_name_prefix}-{j}"
            content = generate_single_task_sbatch_script(task, args.cpus_per_task, job_name, partition=args.partition)
            path = _write_script(job_name, content)
            _ = _submit_one(path, job_name, dependency_expr=dep_expr)
            submitted += 1

    print(f"完成：成功提交 {submitted}/{len(tasks)} 个任务。")


if __name__ == "__main__":
    main()




