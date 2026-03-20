from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TARGET_PRIORITY = (
    "mae",
    "rmse",
    "roc_auc",
    "ap",
    "accuracy",
    "f1",
    "precision",
    "recall",
)
MAXIMIZE_METRICS = {"accuracy", "roc_auc", "ap", "f1", "precision", "recall"}
MINIMIZE_METRICS = {"mae", "rmse", "mse", "loss", "val_loss", "best_val_loss"}


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def sanitize_name_part(raw: str) -> str:
    lowered = raw.strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", lowered)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "item"


def build_study_name(stage: str, dataset: str, bpe_mode: str) -> str:
    return f"hyperopt_{sanitize_name_part(stage)}_{sanitize_name_part(dataset)}_{sanitize_name_part(bpe_mode)}"


def build_pretrain_experiment_name(dataset: str, bpe_mode: str, trial_number: int) -> str:
    return f"search_{sanitize_name_part(dataset)}_{sanitize_name_part(bpe_mode)}_pt_{trial_number:03d}"


def build_finetune_experiment_name(
    dataset: str,
    bpe_mode: str,
    pretrain_option_key: str,
    trial_number: int,
) -> str:
    option = sanitize_name_part(pretrain_option_key).replace("_", "-")
    return f"search_{sanitize_name_part(dataset)}_{sanitize_name_part(bpe_mode)}_ft_{option}_{trial_number:03d}"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def create_storage(journal_file: str | Path):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    del optuna  # imported to validate availability
    journal_path = Path(journal_file)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(journal_path)))


def get_all_study_names(storage) -> list[str]:
    import optuna

    summaries = optuna.study.get_all_study_summaries(storage=storage)
    return sorted(summary.study_name for summary in summaries)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_optimization_target(
    result: dict[str, Any],
    metric_priority: tuple[str, ...] = DEFAULT_TARGET_PRIORITY,
) -> dict[str, Any]:
    test_metrics = result.get("test_metrics")
    if isinstance(test_metrics, dict):
        for metric_name in metric_priority:
            metric_value = test_metrics.get(metric_name)
            if isinstance(metric_value, (int, float)):
                raw_value = float(metric_value)
                direction = "maximize" if metric_name in MAXIMIZE_METRICS else "minimize"
                objective_value = -raw_value if direction == "maximize" else raw_value
                return {
                    "metric_name": metric_name,
                    "raw_value": raw_value,
                    "objective_value": objective_value,
                    "direction": direction,
                }
        for metric_name, metric_value in sorted(test_metrics.items()):
            if isinstance(metric_value, (int, float)):
                raw_value = float(metric_value)
                direction = "maximize" if metric_name in MAXIMIZE_METRICS else "minimize"
                objective_value = -raw_value if direction == "maximize" else raw_value
                return {
                    "metric_name": metric_name,
                    "raw_value": raw_value,
                    "objective_value": objective_value,
                    "direction": direction,
                }

    best_val_loss = result.get("best_val_loss")
    if isinstance(best_val_loss, (int, float)):
        return {
            "metric_name": "best_val_loss",
            "raw_value": float(best_val_loss),
            "objective_value": float(best_val_loss),
            "direction": "minimize",
        }

    raise RuntimeError("Unable to extract an optimization target from result payload.")


def make_project_config(
    *,
    dataset: str,
    method: str,
    encoder: str,
    bpe_backend: str,
    device: str,
    log_style: str,
    config_json: str | None = None,
    target_property: str | None = None,
):
    from config import ProjectConfig
    from src.utils.config_override import apply_json_config

    config = ProjectConfig()
    if config_json:
        apply_json_config(config, config_json)

    config.dataset.name = dataset
    config.serialization.method = method
    config.encoder.type = encoder
    config.serialization.bpe.engine.encode_backend = bpe_backend
    config.system.device = device
    config.device = device
    config.system.log_style = log_style
    config.repeat_runs = 1

    if hasattr(config, "logging") and hasattr(config.logging, "use_wandb"):
        config.logging.use_wandb = False
    if target_property:
        config.task.target_property = target_property

    return config


def require_existing_dir(path: str | Path, description: str) -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"{description} does not exist: {target}")
    return target


def ensure_runtime_prerequisites() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the maintained hyperopt workflow.")


def validate_bpe_backend(backend: str) -> None:
    if backend != "cpp":
        return
    try:
        from src.algorithms.compression import _cpp_bpe  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "The C++ BPE backend was requested but the `_cpp_bpe` extension is not available. "
            "Run `python setup.py build_ext --inplace` first, or rerun the search with `--bpe_backend python`."
        ) from exc
