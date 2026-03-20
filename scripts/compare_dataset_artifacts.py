#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.repro_compare import compare_dataset_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare cold-start dataset artifacts against a baseline directory.")
    parser.add_argument("--baseline", required=True, help="Baseline dataset directory")
    parser.add_argument("--candidate", required=True, help="Candidate dataset directory")
    parser.add_argument(
        "--files",
        nargs="*",
        default=["data.pkl", "train_index.json", "val_index.json", "test_index.json"],
        help="Required files to compare",
    )
    args = parser.parse_args()

    report = compare_dataset_artifacts(args.baseline, args.candidate, required_files=args.files)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
