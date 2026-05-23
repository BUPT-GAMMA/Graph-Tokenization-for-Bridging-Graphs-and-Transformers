#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperopt.scripts.common import (
    build_finetune_experiment_name,
    build_study_name,
    create_storage,
    ensure_runtime_prerequisites,
    extract_optimization_target,
    load_json,
    make_project_config,
    parse_csv_list,
    utc_timestamp,
    validate_bpe_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune hyperparameter search over extracted pretrain candidates.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bpe_mode", default="all", choices=["none", "all", "topk", "random", "gaussian"])
    parser.add_argument("--encoder", default="gte", choices=["bert", "gte"])
    parser.add_argument("--bpe_backend", default="cpp", choices=["cpp", "python"])
    parser.add_argument("--target_property", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_style", default="offline", choices=["online", "offline"])
    parser.add_argument("--config_json", default=None)
    parser.add_argument("--options_file", default="hyperopt/results/best_pretrain_params_for_finetuning.json")
    parser.add_argument("--journal_file", default="hyperopt/journal/finetune_with_options.db")
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--experiment_group", default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_sizes", default="128,256,512")
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=5e-4)
    parser.add_argument("--wd_min", type=float, default=0.0)
    parser.add_argument("--wd_max", type=float, default=0.3)
    parser.add_argument("--warmup_min", type=float, default=0.05)
    parser.add_argument("--warmup_max", type=float, default=0.2)
    parser.add_argument("--grad_norm_min", type=float, default=0.5)
    parser.add_argument("--grad_norm_max", type=float, default=3.0)
    return parser


def _make_group(args: argparse.Namespace) -> str:
    return args.experiment_group or f"hyperopt_finetune_{args.dataset}_{args.bpe_mode}"


def _load_options(path: str | Path) -> dict[str, dict]:
    payload = load_json(path)
    options = payload.get("options")
    if not isinstance(options, dict) or not options:
        raise RuntimeError(f"No finetune options found in: {path}")
    return options


def main() -> None:
    args = build_parser().parse_args()
    ensure_runtime_prerequisites()
    validate_bpe_backend(args.bpe_backend)
    if args.bpe_backend == "python" and args.bpe_mode not in {"all", "none"}:
        raise ValueError("--bpe_backend python only supports --bpe_mode all or none.")
    import optuna
    from src.training.finetune_pipeline import run_finetune

    batch_sizes = [int(item) for item in parse_csv_list(args.batch_sizes)]
    if not batch_sizes:
        raise ValueError("--batch_sizes must provide at least one batch size.")

    options = _load_options(args.options_file)
    for option_key, option in options.items():
        model_path = option.get("model_path")
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"Pretrained model for option '{option_key}' is missing: {model_path}"
            )

    storage = create_storage(args.journal_file)
    study = optuna.create_study(
        study_name=args.study_name or build_study_name("finetune", args.dataset, args.bpe_mode),
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=None,
            n_startup_trials=3,
            multivariate=True,
            constant_liar=True,
            warn_independent_sampling=False,
        ),
        pruner=optuna.pruners.PercentilePruner(
            percentile=25.0,
            n_startup_trials=3,
            n_warmup_steps=2,
            interval_steps=1,
            n_min_trials=2,
        ),
    )

    option_keys = sorted(options)

    def objective(trial):
        option_key = trial.suggest_categorical("pretrain_option", option_keys)
        option = options[option_key]
        method = option["method"]
        config = make_project_config(
            dataset=args.dataset,
            method=method,
            encoder=args.encoder,
            bpe_backend=args.bpe_backend,
            device=args.device,
            log_style=args.log_style,
            config_json=args.config_json,
            target_property=args.target_property,
        )
        config.serialization.bpe.engine.encode_rank_mode = args.bpe_mode
        config.experiment_group = _make_group(args)
        config.experiment_name = build_finetune_experiment_name(args.dataset, args.bpe_mode, option_key, trial.number)
        config.bert.finetuning.epochs = args.epochs
        config.bert.finetuning.learning_rate = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        config.bert.finetuning.batch_size = trial.suggest_categorical("bs", batch_sizes)
        config.bert.finetuning.weight_decay = trial.suggest_float("wd", args.wd_min, args.wd_max)
        config.bert.finetuning.max_grad_norm = trial.suggest_float("grad_norm", args.grad_norm_min, args.grad_norm_max)
        config.bert.finetuning.warmup_ratio = trial.suggest_float("warmup_ratio", args.warmup_min, args.warmup_max)
        config.optuna_trial = trial

        start = time.time()
        result = run_finetune(
            config,
            pretrained_dir=option["model_path"],
            pretrain_exp_name=option["experiment_name"],
        )
        training_minutes = (time.time() - start) / 60.0
        target = extract_optimization_target(result)

        trial.set_user_attr("dataset", args.dataset)
        trial.set_user_attr("method", method)
        trial.set_user_attr("bpe_mode", args.bpe_mode)
        trial.set_user_attr("bpe_backend", args.bpe_backend)
        trial.set_user_attr("pretrain_option", option_key)
        trial.set_user_attr("pretrain_trial_number", option["trial_number"])
        trial.set_user_attr("metric_name", target["metric_name"])
        trial.set_user_attr("metric_raw_value", target["raw_value"])
        trial.set_user_attr("metric_direction", target["direction"])
        trial.set_user_attr("training_time_minutes", training_minutes)
        trial.set_user_attr("timestamp_utc", utc_timestamp())
        return float(target["objective_value"])

    study.optimize(objective, n_trials=args.trials)


if __name__ == "__main__":
    main()
