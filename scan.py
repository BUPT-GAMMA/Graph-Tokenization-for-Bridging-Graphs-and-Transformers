#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_resources.py (Version 2.0 - Corrected)
This script now uses the '-N' flag with sinfo to ensure nodes are listed
individually, preventing the hostlist expression bug.
"""

import subprocess
import re
import sys

def parse_tres_string(tres_string, resource_key):
    """Parses a TRES string to find the value for a specific resource."""
    match = re.search(f'{resource_key}=(\\d+)', tres_string)
    if match:
        return int(match.group(1))
    return 0

def main():
    """Main function to execute the resource check."""
    print("==========================================================================")
    print("          Scanning Cluster for Nodes with Free GPU Resources... (v2)      ")
    print("==========================================================================")

    try:
        # THE FIX IS HERE: Added '-N' to list nodes one-per-line and prevent compression.
        sinfo_command = ['sinfo', '-N', '-h', '-o', '%N']
        sinfo_proc = subprocess.run(sinfo_command, capture_output=True, text=True, check=True)
        nodes = sinfo_proc.stdout.strip().split('\n')
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Fatal Error: Could not execute 'sinfo'. Is Slurm installed and in your PATH?", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    nodes_with_free_gpus = []

    for node in nodes:
        if not node:  # Skip empty lines if any
            continue
        try:
            scontrol_proc = subprocess.run(['scontrol', 'show', 'node', node], capture_output=True, text=True, check=True)
            node_info = scontrol_proc.stdout
            
            state_match = re.search(r'State=(\S+)', node_info)
            if not state_match: continue
            state = state_match.group(1)
            
            if any(s in state for s in ['DOWN', 'DRAIN', 'FAIL']):
                continue

            cfg_tres_match = re.search(r'CfgTRES=([^\s]+)', node_info)
            if not cfg_tres_match: continue
            cfg_tres_str = cfg_tres_match.group(1)

            alloc_tres_match = re.search(r'AllocTRES=([^\s]*)', node_info)
            alloc_tres_str = alloc_tres_match.group(1) if alloc_tres_match else ""

            total_gpus = parse_tres_string(cfg_tres_str, 'gres/gpu')
            if total_gpus == 0: continue
                
            alloc_gpus = parse_tres_string(alloc_tres_str, 'gres/gpu')
            free_gpus = total_gpus - alloc_gpus

            if free_gpus > 0:
                total_cpus = parse_tres_string(cfg_tres_str, 'cpu')
                alloc_cpus = parse_tres_string(alloc_tres_str, 'cpu')
                free_cpus = total_cpus - alloc_cpus
                
                nodes_with_free_gpus.append({
                    "name": node,
                    "state": state.replace('*',''), # Clean up states like 'MIXED*'
                    "free_gpus": free_gpus,
                    "total_gpus": total_gpus,
                    "free_cpus": free_cpus,
                    "total_cpus": total_cpus,
                })
        except (subprocess.CalledProcessError, AttributeError):
            continue
    
    if not nodes_with_free_gpus:
        print("INFO: No nodes with free GPU resources were found.")
    else:
        nodes_with_free_gpus.sort(key=lambda x: (x['state'] != 'IDLE', -x['free_gpus']))
        # Print header
        print("{:<12} | {:<12} | {:<15} | {:<15}".format("Node", "State", "Free GPUs", "Free CPUs"))
        print("-" * 65)
        for data in nodes_with_free_gpus:
             printf_template = "{name:<12} | {state:<12} | {free_gpus}/{total_gpus:<12d} | {free_cpus}/{total_cpus}"
             print(printf_template.format(**data))

    print("==========================================================================")
    print("                              Scan Finished.                              ")
    print("==========================================================================")

if __name__ == "__main__":
    main()
