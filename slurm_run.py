#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
distribute_tasks.py (v2.1) — read tasks from a text file

This script intelligently distributes single-GPU tasks across a Slurm cluster's
free resources.

Features:
- Reads all tasks from a user-provided plain text file (one task per line)
- Scans the cluster for all available free GPU slots
- Assigns tasks to nodes to maximize utilization of fragmented resources
- Generates and submits tailored sbatch scripts for each node
- Supports --dry-run to preview the execution plan without submitting any jobs
"""

import subprocess
import sys
import os
import time
import argparse
import re
from collections import defaultdict

# --- 配置区 ---
# 临时sbatch脚本存放的目录
SBATCH_SCRIPT_DIR = "sbatch_scripts"
# -----------------


def read_tasks_from_file(tasks_file_path):
    """Reads tasks from a text file, one per line. Lines starting with '#' are ignored.

    Raises SystemExit(1) if the file cannot be read or contains no valid tasks.
    """
    print(f"INFO: Reading tasks from file: {tasks_file_path}")
    if not os.path.isfile(tasks_file_path):
        print(f"FATAL: Tasks file does not exist: {tasks_file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(tasks_file_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
    except OSError as e:
        print(f"FATAL: Failed to read tasks file: {tasks_file_path}", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
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
        print("FATAL: Tasks file contained no valid tasks (after removing blanks/comments).", file=sys.stderr)
        sys.exit(1)

    print(f"SUCCESS: Loaded {len(tasks)} tasks from file.")
    return tasks


def get_available_gpus():
    """Runs the resource scanner to get a map of available GPUs."""
    print("INFO: Scanning cluster for available GPU slots using './scan.py'...")
    try:
        proc = subprocess.run(['./scan.py'], capture_output=True, text=True, check=True)
        lines = proc.stdout.strip().split('\n')
        
        node_gpu_map = defaultdict(int)
        header_found = False
        for line in lines:
            if line.strip().startswith("Node"):
                header_found = True
                continue
            if header_found and "---" in line:
                continue
            if header_found and line.strip():
                parts = line.split('|')
                if len(parts) < 3:
                    continue
                node_name = parts[0].strip()
                free_gpus_str = parts[2].split(':')[1].strip().split('/')[0]
                node_gpu_map[node_name] = int(free_gpus_str)
        
        if not node_gpu_map:
            print("WARNING: Scan finished, but no nodes with free GPU resources were found.")
            return {}

        print(f"SUCCESS: Found {sum(node_gpu_map.values())} total free GPU slots across {len(node_gpu_map)} nodes.")
        return dict(node_gpu_map)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("FATAL: Could not get available resources. Is './scan.py' in the same directory and executable?", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        sys.exit(1)


def generate_sbatch_script(node, tasks, total_cpus_on_node=104):
    """Generates a tailored sbatch script for a specific node."""
    num_tasks = len(tasks)
    cpus_per_task = max(1, total_cpus_on_node // 8)

    script_content = f"""#!/bin/bash
#SBATCH --job-name=dist-task-{node}
#SBATCH --nodelist={node}
#SBATCH --nodes=1
#SBATCH --ntasks={num_tasks}
#SBATCH --gres=gpu:{num_tasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output=logs/{node}_%j.out
#SBATCH --error=logs/{node}_%j.err

mkdir -p logs
echo "INFO: Starting {num_tasks} parallel tasks on node {node}..."
echo "--------------------------------------------------------"

"""
    for task in tasks:
        # srun will automatically handle GPU affinity. Each task gets its own GPU.
        script_content += f'srun --exclusive -n1 --gres=gpu:1 {task} &\n'
        
    script_content += "\nwait\n"
    script_content += f'echo "INFO: All tasks on node {node} finished."\n'
    return script_content


def generate_single_task_sbatch_script(task, cpus_per_task=4, job_name_prefix="dist-task"):
    """Generates an sbatch script for a single-GPU task without nodelist constraints.

    This avoids self-blocking by letting Slurm schedule each task independently
    as soon as any GPU becomes available.
    """
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name_prefix}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err

mkdir -p logs
echo "INFO: Starting single task..."
srun --exclusive -n1 --gres=gpu:1 {task}
echo "INFO: Single task finished."
"""
    return script_content


def parse_sbatch_job_id(sbatch_stdout):
    """Extracts the job ID from sbatch output like 'Submitted batch job 12345'."""
    match = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
    if not match:
        return None
    return match.group(1)


def query_job_states(job_ids):
    """Returns a dict of job_id -> state using squeue. Jobs not found are omitted."""
    if not job_ids:
        return {}
    jobs_arg = ",".join(job_ids)
    try:
        proc = subprocess.run([
            'squeue', '--jobs', jobs_arg, '--noheader', '-o', '%i %T %R'
        ], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("FATAL: Failed to query job states via squeue.", file=sys.stderr)
        print(f"Error details: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    states = {}
    for line in proc.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1]
        states[jid] = state
    return states


def wait_for_all_jobs_to_leave_pending(job_ids, timeout_seconds=30, poll_interval=1.0):
    """Waits until none of the jobs is in PENDING state, or timeout occurs.

    Returns True if successful (no PENDING jobs), False if timed out.
    """
    start_time = time.time()
    while True:
        states = query_job_states(job_ids)
        pending_job_ids = [jid for jid, st in states.items() if st.upper() == 'PENDING']
        if not pending_job_ids:
            return True
        if time.time() - start_time >= timeout_seconds:
            print("ERROR: Some jobs remain in PENDING after timeout:")
            for jid in pending_job_ids:
                print(f"  - Job {jid} is PENDING")
            return False
        time.sleep(poll_interval)


def cancel_jobs(job_ids):
    if not job_ids:
        return
    for jid in job_ids:
        try:
            subprocess.run(['scancel', jid], check=False, capture_output=True, text=True)
            print(f"Cancelled job {jid}")
        except Exception as e:
            print(f"WARNING: Failed to cancel job {jid}: {e}")


def main():
    """Main distributor logic with argument parsing."""
    parser = argparse.ArgumentParser(description="Smartly distribute single-GPU tasks across a Slurm cluster.")
    parser.add_argument('--tasks-file', '-t', required=True, type=str,
                        help="Path to a text file that contains all tasks, one per line. Lines starting with '#' are ignored.")
    parser.add_argument('--dry-run', action='store_true', help="Show the execution plan without submitting any jobs.")
    parser.add_argument('--mode', choices=['per-task', 'per-node'], default='per-task',
                        help="Submission mode: 'per-task' submits one sbatch per task (default, safest to avoid self-blocking); 'per-node' groups tasks per node.")
    parser.add_argument('--cpus-per-task', type=int, default=4, help="cpus-per-task for per-task submission mode.")
    parser.add_argument('--start-timeout', type=int, default=30, help="Max seconds to wait for all submitted jobs to leave PENDING state. If exceeded, all jobs are cancelled and the script exits with error.")
    args = parser.parse_args()

    if args.dry_run:
        print("\n*** DRY RUN MODE ACTIVATED ***")
        print("No sbatch files will be written and no jobs will be submitted.\n")

    # 1. Get all tasks
    all_tasks = read_tasks_from_file(args.tasks_file)
    if not all_tasks:
        return

    # 2. Get all available GPU slots
    node_gpu_map = get_available_gpus()
    total_free_gpus = sum(node_gpu_map.values()) if node_gpu_map else 0

    # 3. Evaluate capacity and prepare plan
    can_run_all_at_once = total_free_gpus >= len(all_tasks)
    print("\n" + "="*40)
    print(" Capacity Overview")
    print("="*40)
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Total free GPUs now: {total_free_gpus}")
    print(f"Can run all at once now: {'YES' if can_run_all_at_once else 'NO'}")
    print("="*40 + "\n")

    # Enforce: we expect to submit and run all tasks at once; otherwise, abort
    if not can_run_all_at_once:
        msg = "FATAL: Insufficient free GPUs to run all tasks at once. Aborting without submission."
        if args.dry_run:
            print(msg)
            sys.exit(2)
        else:
            print(msg, file=sys.stderr)
            sys.exit(2)

    if args.mode == 'per-node':
        if not node_gpu_map:
            print("INFO: No currently free GPUs detected. Nothing to submit in per-node mode.")
            return

        # Assign tasks to nodes greedily based on free GPUs (descending)
        task_assignments = defaultdict(list)
        task_idx = 0
        sorted_nodes = sorted(node_gpu_map.items(), key=lambda item: -item[1])

        for node, free_gpus in sorted_nodes:
            for _ in range(free_gpus):
                if task_idx < len(all_tasks):
                    task_assignments[node].append(all_tasks[task_idx])
                    task_idx += 1
                else:
                    break
            if task_idx >= len(all_tasks):
                break

        if not task_assignments:
            print("INFO: No tasks could be assigned (perhaps you have more free GPUs than tasks).")
            return

        # Print the per-node plan
        print("\n" + "="*30)
        print("      Execution Plan (per-node)")
        print("="*30)
        for node, tasks in task_assignments.items():
            print(f"\n[Node: {node}] - Will be assigned {len(tasks)} tasks.")
            for i, task in enumerate(tasks):
                print(f"  - GPU Slot {i+1}: {task}")
        print("="*30 + "\n")

        # Execute per-node plan
        if args.dry_run:
            print("DRY RUN: Plan displayed above. All tasks could start now (per-node). No submission performed.")
            print("--------------------------------")
            return

        print("--- Submitting Jobs to Slurm (per-node) ---")
        if not os.path.exists(SBATCH_SCRIPT_DIR):
            os.makedirs(SBATCH_SCRIPT_DIR)

        submitted_job_ids = []
        for node, tasks in task_assignments.items():
            print(f"INFO: Generating and submitting sbatch script for node {node}...")
            script_content = generate_sbatch_script(node, tasks)
            script_path = os.path.join(SBATCH_SCRIPT_DIR, f"submit_{node}.sh")
            with open(script_path, 'w') as f:
                f.write(script_content)

            try:
                proc = subprocess.run(['sbatch', script_path], check=True, capture_output=True, text=True)
                job_id = parse_sbatch_job_id(proc.stdout)
                if job_id:
                    submitted_job_ids.append(job_id)
                print(f"  -> SUCCESS: Job for node {node} submitted. JobID={job_id if job_id else 'UNKNOWN'}")
                time.sleep(0.1)
            except subprocess.CalledProcessError as e:
                print(f"  -> ERROR: Failed to submit job for node {node}.", file=sys.stderr)
                print(f"     Please check the generated script: {script_path}", file=sys.stderr)
                print(f"     Slurm Error: {e.stderr}", file=sys.stderr)
                cancel_jobs(submitted_job_ids)
                sys.exit(1)

        # Ensure none is pending after submission
        ok = wait_for_all_jobs_to_leave_pending(submitted_job_ids, timeout_seconds=args.start_timeout)
        if not ok:
            print("FATAL: At least one job remained in PENDING. Cancelling all submitted jobs.", file=sys.stderr)
            cancel_jobs(submitted_job_ids)
            sys.exit(1)

        print("All jobs started without entering PENDING.")
        print("--------------------------------")

    else:  # per-task mode
        print("Plan: Submit one sbatch per task (per-task mode). This avoids self-blocking and will utilize GPUs as they free up.")
        if args.dry_run:
            print("DRY RUN: All tasks could start now (per-task). Would submit the following tasks:")
            for i, task in enumerate(all_tasks):
                print(f"  [{i+1}] {task}")
            print("--------------------------------")
            return

        print("--- Submitting Jobs to Slurm (per-task) ---")
        if not os.path.exists(SBATCH_SCRIPT_DIR):
            os.makedirs(SBATCH_SCRIPT_DIR)
        submitted_job_ids = []
        for i, task in enumerate(all_tasks):
            job_name_prefix = f"dist-task-{i+1}"
            script_content = generate_single_task_sbatch_script(task, cpus_per_task=args.cpus_per_task, job_name_prefix=job_name_prefix)
            script_path = os.path.join(SBATCH_SCRIPT_DIR, f"submit_single_{i+1}.sh")
            with open(script_path, 'w') as f:
                f.write(script_content)
            try:
                proc = subprocess.run(['sbatch', script_path], check=True, capture_output=True, text=True)
                job_id = parse_sbatch_job_id(proc.stdout)
                if job_id:
                    submitted_job_ids.append(job_id)
                print(f"  -> Submitted job for task #{i+1} JobID={job_id if job_id else 'UNKNOWN'}")
                time.sleep(0.05)
            except subprocess.CalledProcessError as e:
                print(f"  -> ERROR: Failed to submit job for task #{i+1}.", file=sys.stderr)
                print(f"     Please check the generated script: {script_path}", file=sys.stderr)
                print(f"     Slurm Error: {e.stderr}", file=sys.stderr)
                cancel_jobs(submitted_job_ids)
                sys.exit(1)

        ok = wait_for_all_jobs_to_leave_pending(submitted_job_ids, timeout_seconds=args.start_timeout)
        if not ok:
            print("FATAL: At least one job remained in PENDING. Cancelling all submitted jobs.", file=sys.stderr)
            cancel_jobs(submitted_job_ids)
            sys.exit(1)

        print("All jobs started without entering PENDING.")
        print("--------------------------------")


if __name__ == "__main__":
    main()