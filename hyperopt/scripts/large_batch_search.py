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
    build_pretrain_experiment_name,
    build_study_name,
    create_storage,
    ensure_runtime_prerequisites,
    extract_optimization_target,
    make_project_config,
    parse_csv_list,
    utc_timestamp,
    validate_bpe_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna large-batch search for TokenizerGraph.")
    parser.add_argument("--dataset", default="zinc", help="Dataset name.")
    parser.add_argument("--methods", default="fcpp", help="Comma-separated serialization methods to search.")
    parser.add_argument("--bpe_mode", default="all", choices=["none", "all", "topk", "random", "gaussian"])
    parser.add_argument("--encoder", default="gte", choices=["bert", "gte"], help="Encoder type.")
    parser.add_argument("--bpe_backend", default="cpp", choices=["cpp", "python"], help="BPE encode backend.")
    parser.add_argument("--target_property", default=None, help="Regression target property when required.")
    parser.add_argument("--device", default="cuda:0", help="Training device.")
    parser.add_argument("--log_style", default="offline", choices=["online", "offline"])
    parser.add_argument("--config_json", default=None, help="Optional JSON override string or file path.")
    parser.add_argument("--journal_file", default="hyperopt/journal/large_batch.db")
    parser.add_argument("--stage", default="pretrain", choices=["pretrain", "finetune", "both"])
    parser.add_argument("--experiment_group", default=None, help="Override the pretrain experiment group.")
    parser.add_argument("--finetune_group", default=None, help="Override the finetune experiment group.")
    parser.add_argument("--pretrain_trials", type=int, default=20)
    parser.add_argument("--finetune_trials", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3, help="Top-K pretrain trials used for finetuning.")
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=50)
    parser.add_argument("--batch_sizes", default="128,256,512", help="Comma-separated batch sizes.")
    parser.add_argument("--finetune_batch_sizes", default=None, help="Comma-separated finetune batch sizes.")
    parser.add_argument("--lr_min", type=float, default=5e-5)
    parser.add_argument("--lr_max", type=float, default=5e-4)
    parser.add_argument("--finetune_lr_min", type=float, default=1e-5)
    parser.add_argument("--finetune_lr_max", type=float, default=5e-4)
    parser.add_argument("--wd_min", type=float, default=0.0)
    parser.add_argument("--wd_max", type=float, default=0.3)
    parser.add_argument("--mask_prob_min", type=float, default=0.08)
    parser.add_argument("--mask_prob_max", type=float, default=0.15)
    parser.add_argument("--warmup_min", type=float, default=0.05)
    parser.add_argument("--warmup_max", type=float, default=0.2)
    parser.add_argument("--grad_norm_min", type=float, default=0.5)
    parser.add_argument("--grad_norm_max", type=float, default=3.0)
    return parser


def _make_pretrain_group(args: argparse.Namespace) -> str:
    return args.experiment_group or f"hyperopt_pretrain_{args.dataset}_{args.bpe_mode}"


def _make_finetune_group(args: argparse.Namespace) -> str:
    return args.finetune_group or f"hyperopt_finetune_{args.dataset}_{args.bpe_mode}"


def _load_completed_trials(study):
    import optuna

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda trial: float(trial.value))
    return completed


def _build_pretrain_objective(args: argparse.Namespace, methods: list[str], batch_sizes: list[int]):
    from src.training.pretrain_pipeline import train_bert_mlm

    def objective(trial):
        method = methods[0] if len(methods) == 1 else trial.suggest_categorical("method", methods)
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
        config.experiment_group = _make_pretrain_group(args)
        config.experiment_name = build_pretrain_experiment_name(args.dataset, args.bpe_mode, trial.number)
        config.bert.pretraining.epochs = args.pretrain_epochs
        config.bert.pretraining.learning_rate = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        config.bert.pretraining.batch_size = trial.suggest_categorical("bs", batch_sizes)
        config.bert.pretraining.weight_decay = trial.suggest_float("wd", args.wd_min, args.wd_max)
        config.bert.pretraining.max_grad_norm = trial.suggest_float("grad_norm", args.grad_norm_min, args.grad_norm_max)
        config.bert.pretraining.mask_prob = trial.suggest_float("mask_prob", args.mask_prob_min, args.mask_prob_max)
        config.bert.pretraining.warmup_ratio = trial.suggest_float("warmup_ratio", args.warmup_min, args.warmup_max)
        config.optuna_trial = trial

        start = time.time()
        result = train_bert_mlm(config)
        training_minutes = (time.time() - start) / 60.0
        model_path = config.get_model_dir(run_i=0) / "best"

        trial.set_user_attr("stage", "pretrain")
        trial.set_user_attr("dataset", args.dataset)
        trial.set_user_attr("method", method)
        trial.set_user_attr("encoder", args.encoder)
        trial.set_user_attr("bpe_backend", args.bpe_backend)
        trial.set_user_attr("bpe_mode", args.bpe_mode)
        trial.set_user_attr("experiment_group", config.experiment_group)
        trial.set_user_attr("experiment_name", config.experiment_name)
        trial.set_user_attr("model_path", str(model_path))
        trial.set_user_attr("training_time_minutes", training_minutes)
        trial.set_user_attr("timestamp_utc", utc_timestamp())
        return float(result["best_val_loss"])

    return objective


def _build_finetune_objective(args: argparse.Namespace, candidates: dict[str, dict], batch_sizes: list[int]):
    from src.training.finetune_pipeline import run_finetune

    candidate_keys = sorted(candidates)

    def objective(trial):
        option_key = trial.suggest_categorical("pretrain_option", candidate_keys)
        option = candidates[option_key]
        config = make_project_config(
            dataset=args.dataset,
            method=option["method"],
            encoder=args.encoder,
            bpe_backend=args.bpe_backend,
            device=args.device,
            log_style=args.log_style,
            config_json=args.config_json,
            target_property=args.target_property,
        )
        config.serialization.bpe.engine.encode_rank_mode = args.bpe_mode
        config.experiment_group = _make_finetune_group(args)
        config.experiment_name = build_finetune_experiment_name(args.dataset, args.bpe_mode, option_key, trial.number)
        config.bert.finetuning.epochs = args.finetune_epochs
        config.bert.finetuning.learning_rate = trial.suggest_float("lr", args.finetune_lr_min, args.finetune_lr_max, log=True)
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

        trial.set_user_attr("stage", "finetune")
        trial.set_user_attr("dataset", args.dataset)
        trial.set_user_attr("method", option["method"])
        trial.set_user_attr("encoder", args.encoder)
        trial.set_user_attr("bpe_backend", args.bpe_backend)
        trial.set_user_attr("bpe_mode", args.bpe_mode)
        trial.set_user_attr("pretrain_option", option_key)
        trial.set_user_attr("pretrain_trial_number", option["trial_number"])
        trial.set_user_attr("pretrain_model_path", option["model_path"])
        trial.set_user_attr("metric_name", target["metric_name"])
        trial.set_user_attr("metric_raw_value", target["raw_value"])
        trial.set_user_attr("metric_direction", target["direction"])
        trial.set_user_attr("training_time_minutes", training_minutes)
        trial.set_user_attr("timestamp_utc", utc_timestamp())
        return float(target["objective_value"])

    return objective


def _create_pretrain_sampler():
    import optuna

    return optuna.samplers.TPESampler(
        seed=None,
        n_startup_trials=5,
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
    )


def _create_pretrain_pruner():
    import optuna

    return optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
        interval_steps=1,
        n_min_trials=2,
    )


def _create_finetune_sampler():
    import optuna

    return optuna.samplers.TPESampler(
        seed=None,
        n_startup_trials=3,
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
    )


def _create_finetune_pruner():
    import optuna

    return optuna.pruners.PercentilePruner(
        percentile=25.0,
        n_startup_trials=3,
        n_warmup_steps=2,
        interval_steps=1,
        n_min_trials=2,
    )


def main() -> None:
    args = build_parser().parse_args()
    ensure_runtime_prerequisites()
    validate_bpe_backend(args.bpe_backend)
    if args.bpe_backend == "python" and args.bpe_mode not in {"all", "none"}:
        raise ValueError("--bpe_backend python only supports --bpe_mode all or none.")

    import optuna

    methods = parse_csv_list(args.methods)
    if not methods:
        raise ValueError("--methods must provide at least one serialization method.")

    batch_sizes = [int(item) for item in parse_csv_list(args.batch_sizes)]
    if not batch_sizes:
        raise ValueError("--batch_sizes must provide at least one batch size.")

    finetune_batch_sizes = [int(item) for item in parse_csv_list(args.finetune_batch_sizes or args.batch_sizes)]
    if not finetune_batch_sizes:
        raise ValueError("--finetune_batch_sizes must provide at least one batch size.")

    storage = create_storage(args.journal_file)

    if args.stage in {"pretrain", "both"}:
        pretrain_study = optuna.create_study(
            study_name=build_study_name("pretrain", args.dataset, args.bpe_mode),
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=_create_pretrain_sampler(),
            pruner=_create_pretrain_pruner(),
        )
        pretrain_study.optimize(
            _build_pretrain_objective(args, methods, batch_sizes),
            n_trials=args.pretrain_trials,
        )

    if args.stage in {"finetune", "both"}:
        pretrain_study = optuna.load_study(
            study_name=build_study_name("pretrain", args.dataset, args.bpe_mode),
            storage=storage,
        )
        completed = _load_completed_trials(pretrain_study)
        if not completed:
            raise RuntimeError("No completed pretrain trials were found for finetune search.")

        top_trials = completed[: max(1, args.top_k)]
        candidates = {}
        for index, candidate_trial in enumerate(top_trials, start=1):
            label = f"top{index}_overall"
            method = candidate_trial.user_attrs.get("method") or candidate_trial.params.get("method")
            model_path = candidate_trial.user_attrs.get("model_path")
            experiment_name = candidate_trial.user_attrs.get("experiment_name")
            if not method or not model_path or not experiment_name:
                raise RuntimeError(
                    f"Pretrain trial {candidate_trial.number} is missing method/model metadata. "
                    "Re-run the maintained pretrain search first."
                )
            candidates[label] = {
                "trial_number": candidate_trial.number,
                "method": method,
                "model_path": model_path,
                "experiment_name": experiment_name,
            }

        finetune_study = optuna.create_study(
            study_name=build_study_name("finetune", args.dataset, args.bpe_mode),
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=_create_finetune_sampler(),
            pruner=_create_finetune_pruner(),
        )
        finetune_study.optimize(
            _build_finetune_objective(args, candidates, finetune_batch_sizes),
            n_trials=args.finetune_trials,
        )


if __name__ == "__main__":
    main()
