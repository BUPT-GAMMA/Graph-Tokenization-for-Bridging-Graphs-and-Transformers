from pathlib import Path

import pytest

from hyperopt.scripts.common import (
    build_finetune_experiment_name,
    build_pretrain_experiment_name,
    build_study_name,
    extract_optimization_target,
    parse_csv_list,
)
from src.algorithms.compression.bpe_engine import BPEEngine


DRIVE_URL = "https://drive.google.com/file/d/10etZF9OnV569_Fp7tpdMUVEH9eZECKdW/view?usp=sharing"


def test_parse_csv_list_strips_whitespace_and_empty_items():
    assert parse_csv_list(" feuler, fcpp ,, smiles ") == ["feuler", "fcpp", "smiles"]


def test_hyperopt_names_follow_stable_convention():
    assert build_study_name(stage="pretrain", dataset="qm9test", bpe_mode="all") == "hyperopt_pretrain_qm9test_all"
    assert build_study_name(stage="finetune", dataset="zinc", bpe_mode="random") == "hyperopt_finetune_zinc_random"
    assert build_pretrain_experiment_name(dataset="qm9test", bpe_mode="all", trial_number=7) == "search_qm9test_all_pt_007"
    assert (
        build_finetune_experiment_name(
            dataset="qm9test",
            bpe_mode="all",
            pretrain_option_key="top1_overall",
            trial_number=12,
        )
        == "search_qm9test_all_ft_top1-overall_012"
    )


def test_extract_optimization_target_prefers_test_metric_and_handles_minimize_metrics():
    result = {
        "test_metrics": {"mae": 0.123, "rmse": 0.456},
        "best_val_loss": 0.789,
    }
    target = extract_optimization_target(result)

    assert target["metric_name"] == "mae"
    assert target["raw_value"] == 0.123
    assert target["objective_value"] == 0.123
    assert target["direction"] == "minimize"


def test_extract_optimization_target_converts_maximize_metrics_to_minimize_objective():
    result = {
        "test_metrics": {"accuracy": 0.88},
        "best_val_loss": 0.5,
    }
    target = extract_optimization_target(result)

    assert target["metric_name"] == "accuracy"
    assert target["raw_value"] == 0.88
    assert target["objective_value"] == -0.88
    assert target["direction"] == "maximize"


def test_extract_optimization_target_falls_back_to_best_val_loss():
    result = {"best_val_loss": 1.25}
    target = extract_optimization_target(result)

    assert target["metric_name"] == "best_val_loss"
    assert target["raw_value"] == 1.25
    assert target["objective_value"] == 1.25
    assert target["direction"] == "minimize"


def test_python_bpe_backend_supports_deterministic_encode_modes():
    engine = BPEEngine(train_backend="python", encode_backend="python", encode_rank_mode="all")
    engine.merge_rules = [(1, 2, 4), (4, 3, 5)]
    engine.vocab_size = 6
    engine.build_encoder()
    assert engine.encode([1, 2, 3]) == [5]

    topk_engine = BPEEngine(train_backend="python", encode_backend="python", encode_rank_mode="topk")
    topk_engine.merge_rules = [(1, 2, 4)]
    topk_engine.vocab_size = 5
    with pytest.raises(ValueError, match="encode_backend='python' only supports"):
        topk_engine.build_encoder()


def test_dataset_download_link_is_published_in_main_docs():
    for path in ("README.md", "README_zh.md", "scripts/dataset_conversion/README.md"):
        text = Path(path).read_text(encoding="utf-8")
        assert DRIVE_URL in text


def test_dataset_docs_point_to_conversion_scripts_for_new_dataset_integration():
    text = Path("scripts/dataset_conversion/README.md").read_text(encoding="utf-8")
    assert "new dataset" in text.lower() or "新数据集" in text
    assert "scripts/dataset_conversion" in text


def test_hyperopt_docs_exist_and_reference_optuna():
    readme = Path("hyperopt/README.md").read_text(encoding="utf-8")

    assert "Optuna" in readme
    assert "JournalStorage" in readme
    assert "large_batch_search.py" in readme
    assert "--lr_min" in readme
    assert "--wd_min" in readme
    assert "--config_json" in readme
    assert "hyperopt/scripts/finetune_with_pretrain_options.py" in readme
