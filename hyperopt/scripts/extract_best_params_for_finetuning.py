#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperopt.scripts.common import (
    build_study_name,
    create_storage,
    dump_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract top pretrain trials for downstream finetune search.")
    parser.add_argument("--journal", default="hyperopt/journal/large_batch.db")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bpe_mode", default="all", choices=["none", "all", "topk", "random", "gaussian"])
    parser.add_argument("--study_name", default=None, help="Override the pretrain study name.")
    parser.add_argument("--top_k_overall", type=int, default=3)
    parser.add_argument("--top_k_per_method", type=int, default=1)
    parser.add_argument("--output_dir", default="hyperopt/results")
    return parser


def main() -> None:
    import optuna

    args = build_parser().parse_args()
    storage = create_storage(args.journal)
    study_name = args.study_name or build_study_name("pretrain", args.dataset, args.bpe_mode)
    study = optuna.load_study(study_name=study_name, storage=storage)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda trial: float(trial.value))
    if not completed:
        raise RuntimeError(f"No completed trials found in study: {study_name}")

    options: dict[str, dict] = {}
    for index, trial in enumerate(completed[: args.top_k_overall], start=1):
        options[f"top{index}_overall"] = {
            "trial_number": trial.number,
            "objective_value": float(trial.value),
            "method": trial.user_attrs.get("method") or trial.params.get("method"),
            "experiment_group": trial.user_attrs.get("experiment_group"),
            "experiment_name": trial.user_attrs.get("experiment_name"),
            "model_path": trial.user_attrs.get("model_path"),
            "hyperparameters": dict(trial.params),
            "training_time_minutes": trial.user_attrs.get("training_time_minutes"),
        }

    grouped: dict[str, list] = {}
    for trial in completed:
        method = trial.user_attrs.get("method") or trial.params.get("method")
        grouped.setdefault(str(method), []).append(trial)

    for method, trials in sorted(grouped.items()):
        for index, trial in enumerate(trials[: args.top_k_per_method], start=1):
            options[f"{method}_top{index}"] = {
                "trial_number": trial.number,
                "objective_value": float(trial.value),
                "method": method,
                "experiment_group": trial.user_attrs.get("experiment_group"),
                "experiment_name": trial.user_attrs.get("experiment_name"),
                "model_path": trial.user_attrs.get("model_path"),
                "hyperparameters": dict(trial.params),
                "training_time_minutes": trial.user_attrs.get("training_time_minutes"),
            }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_pretrain_params_for_finetuning.json"
    dump_json(
        output_path,
        {
            "metadata": {
                "dataset": args.dataset,
                "bpe_mode": args.bpe_mode,
                "study_name": study_name,
                "num_completed_trials": len(completed),
            },
            "options": options,
        },
    )
    print(output_path)


if __name__ == "__main__":
    main()
