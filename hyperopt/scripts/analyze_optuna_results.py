#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperopt.scripts.common import build_study_name, create_storage, get_all_study_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Optuna studies used by TokenizerGraph hyperopt.")
    parser.add_argument("--journal", default="hyperopt/journal/large_batch.db")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--bpe_mode", default="all", choices=["none", "all", "topk", "random", "gaussian"])
    parser.add_argument("--stage", default=None, choices=["pretrain", "finetune"])
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    return parser


def main() -> None:
    import optuna

    args = build_parser().parse_args()
    storage = create_storage(args.journal)

    if args.study_name:
        study_names = [args.study_name]
    elif args.dataset and args.stage:
        study_names = [build_study_name(args.stage, args.dataset, args.bpe_mode)]
    else:
        study_names = get_all_study_names(storage)

    if not study_names:
        raise RuntimeError(f"No Optuna studies found in journal: {args.journal}")

    for study_name in study_names:
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        pruned = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
        failed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL]

        print("=" * 80)
        print(study_name)
        print(f"trials={len(study.trials)} completed={len(completed)} pruned={len(pruned)} failed={len(failed)}")
        if not completed:
            continue

        best = min(completed, key=lambda trial: float(trial.value))
        print(f"best_trial={best.number} objective={float(best.value):.6f}")
        print(f"best_params={best.params}")

        top_trials = sorted(completed, key=lambda trial: float(trial.value))[: args.top_k]
        for index, trial in enumerate(top_trials, start=1):
            print(
                f"top{index}: trial={trial.number} objective={float(trial.value):.6f} "
                f"params={trial.params} attrs={{'method': {trial.user_attrs.get('method')}, "
                f"'metric_name': {trial.user_attrs.get('metric_name')}}}"
            )


if __name__ == "__main__":
    main()
