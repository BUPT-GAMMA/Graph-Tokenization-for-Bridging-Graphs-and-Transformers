"""
Repeat-run statistics aggregation.

Aggregates results from multiple repeated experiment runs and computes
statistical summaries (mean, variance, std, etc.).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
from config import ProjectConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def aggregate_experiment_results(
    config: ProjectConfig,
    experiment_name: str,
    repeat_runs: int,
    task_type: str = "finetune",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Aggregate results from multiple repeated runs.

    Args:
        config: Project config
        experiment_name: Experiment name
        repeat_runs: Number of repeated runs
        task_type: "pretrain" or "finetune"

    Returns:
        Aggregated statistics dict
    """
    logger.info(f"Aggregating {repeat_runs} runs for {experiment_name}")

    # Collect results from all runs
    all_results = []
    valid_runs = 0

    for run_i in range(repeat_runs):
        try:
            # Build result file path
            if task_type == "finetune":
                results_file = config.get_logs_dir(
                    exp_name=experiment_name,
                    run_i=run_i
                ) / "finetune" / "finetune_metrics.json"
            else:  # pretrain
                results_file = config.get_logs_dir(
                    exp_name=experiment_name,
                    run_i=run_i
                ) / "pretrain" / "pretrain_metrics.json"

            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    all_results.append(result)
                    valid_runs += 1
                    logger.info(f"Found run_{run_i} results")
            else:
                logger.warning(f"run_{run_i} result file not found: {results_file}")

        except Exception as e:
            logger.warning(f"Failed to read run_{run_i} results: {e}")

    if valid_runs == 0:
        logger.error("No valid run results found")
        return {}

    logger.info(f"Collected {valid_runs}/{repeat_runs} run results")

    # Aggregate
    aggregated = _aggregate_results(all_results, task_type)
    aggregated['config'] = config.to_dict()

    # Save aggregated results
    if output_file is None:
        output_file = config.get_logs_dir(exp_name=experiment_name, run_i=-1) / f"{task_type}_aggregated_stats.json"

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    logger.info(f"Aggregated results saved: {output_file}")
    return aggregated


def _aggregate_results(results: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
    """
    Compute statistical aggregation over a list of result dicts.

    Args:
        results: List of result dicts
        task_type: Task type

    Returns:
        Aggregated statistics
    """
    if not results:
        return {}

    aggregated = {
        "summary": {
            "total_runs": len(results),
            "task_type": task_type,
            "aggregation_timestamp": np.datetime64('now').astype(str)
        },
        "statistics": {},
        "individual_runs": []
    }
    
    def get_fair_best(result, metric):
      """Get fair best test metric (choose optimal between avg and learned)."""
      test_data = result['test']
      by_agg = test_data['by_aggregation']
      avg_data = by_agg['avg']
      learned_data = by_agg['learned']

      # Check if metric exists
      avg_has_metric = metric in avg_data
      learned_has_metric = metric in learned_data

      if not avg_has_metric and not learned_has_metric:
        raise ValueError(f"Metric '{metric}' not found in either avg or learned aggregation")

      if not avg_has_metric:
        return learned_data[metric]
      if not learned_has_metric:
        return avg_data[metric]

      # Use correct task type path
      task_type = result['config']['task']['type']

      # For classification: higher is better; for regression: lower is better
      if 'classification' in task_type:
        return max(avg_data[metric], learned_data[metric])
      else:  # regression
        return min(avg_data[metric], learned_data[metric])

    # Save basic info for each run
    for i, result in enumerate(results):
        run_info = {
            "run_id": i,
            "seed": result["config"]["system"]["seed"],
            "experiment_name": result["config"]["experiment_name"],
            "start_time": result.get("start_time"),
            "end_time": result.get("end_time")
        }

        # Add key metrics
        add={}
        if task_type == "finetune":
          # Select metrics based on task type
          task_type_config = result['config']['task']['type']
          if task_type_config == "classification":
            key_metrics = ["accuracy", "roc_auc", "ap", "precision", "recall", "f1"]
          else:  # regression or other
            key_metrics = ["mae", "rmse", "r2", "loss"]

          for metric in key_metrics:
            # Use fair best value (optimal between avg and learned)
            try:
              fair_value = get_fair_best(result, metric)
              add[f"test_{metric}"] = fair_value
            except (KeyError, ValueError) as e:
              # Skip missing metrics
              print(f"Skipping metric '{metric}': {e}")
              continue
          run_info.update(add)
        elif task_type == "pretrain":
            run_info.update({
                "best_val_loss": result["best_val_loss"],
                "total_train_time_sec": result["total_train_time_sec"],
                "effective_max_length": result["effective_max_length"]
            })

        aggregated["individual_runs"].append(run_info)

    if task_type == "finetune":
        aggregated["statistics"] = _aggregate_finetune_results(results)

        # Add pk stats for all three modes to summary
        if 'test' in aggregated["statistics"]:
            pk_summary = {}
            for mode in ['avg', 'best', 'learned']:
                if mode in aggregated["statistics"]['test'] and 'pk' in aggregated["statistics"]['test'][mode]:
                    pk_stats = aggregated["statistics"]['test'][mode]['pk']
                    pk_summary[mode] = {
                        'mean': pk_stats['mean'],
                        'std': pk_stats['std']
                    }
            if pk_summary:
                aggregated["summary"]["pk_stats"] = pk_summary

    elif task_type == "pretrain":
        aggregated["statistics"] = _aggregate_pretrain_results(results)

    return aggregated


def _aggregate_finetune_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate finetune results."""
    stats = {}

    # 1. Validation metrics
    val_keys = ['val_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap','mae', 'rmse', 'r2']
    stats['val'] = {}
    for key in val_keys:
        values = [r['val'].get(key, None) for r in results]
        values = [value for value in values if value is not None]
        if len(values) == 0:
          continue
        stats['val'][key] = _calculate_stats(values)

    # 2. Test metrics (all aggregation modes)
    test_modes = ['avg', 'best', 'learned']
    stats['test'] = {}

    # Determine metric set based on task type
    first_result = results[0]
    task_type = first_result['config']['task']['type']

    if 'classification' in task_type:
        test_metrics = ['val_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap']
    else:
        test_metrics = ['val_loss', 'mae', 'rmse', 'r2']
    test_metrics.append('pk')

    for mode in test_modes:
        stats['test'][mode] = {}
        for metric in test_metrics:
            values = [r['test']['by_aggregation'][mode].get(metric, None) for r in results]
            values = [value for value in values if value is not None]
            stats['test'][mode][metric] = _calculate_stats(values)

    # 3. Time metrics
    stats['time'] = {}
    time_keys = ['total_train_time_sec', 'avg_epoch_time_sec']
    for key in time_keys:
        values = [r['time'][key] for r in results]
        stats['time'][key] = _calculate_stats(values)

    # 4. Training metrics
    stats['train'] = {}
    train_keys = ['last_loss', 'learning_rate_last']
    for key in train_keys:
        values = [r['train'][key] for r in results]
        stats['train'][key] = _calculate_stats(values)

    return stats


def _aggregate_pretrain_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate pretrain results."""
    stats = {}

    # Key pretrain metrics
    keys = [
        'best_val_loss',
        'total_train_time_sec',
        'avg_epoch_time_sec',
        'effective_max_length'
    ]

    for key in keys:
        values = [r[key] for r in results]
        stats[key] = _calculate_stats(values)

    return stats


def _calculate_stats(values: List[float]) -> Dict[str, float]:
    """
    Compute basic statistics for a list of numeric values.

    Args:
        values: List of numeric values

    Returns:
        Statistics dict
    """
    if not values or len(values) == 0:
        return {}

    values = np.array(values)
    n = len(values)

    # Handle single-sample case
    if n == 1:
        val = float(values[0])
        stats = {
            'mean': val,
            'std': 0.0,
            'var': 0.0,
            'min': val,
            'max': val,
            'median': val,
            'count': 1
        }
    else:
        stats = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)),  # sample std
            'var': float(np.var(values, ddof=1)),   # sample var
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values)
        }

    # 95% confidence interval (normal distribution assumption)
    # if len(values) > 1:
    #     sem = stats['std'] / np.sqrt(len(values))
    #     confidence_interval = 1.96 * sem
    #     stats['ci_95'] = float(confidence_interval)
    #     stats['ci_95_lower'] = stats['mean'] - confidence_interval
    #     stats['ci_95_upper'] = stats['mean'] + confidence_interval

    return stats


def print_aggregated_stats(aggregated: Dict[str, Any], task_type: str):
    """
    Print aggregated statistics.

    Args:
        aggregated: Aggregated results
        task_type: Task type
    """
    if not aggregated:
        print("No aggregated results to display")
        return

    summary = aggregated.get('summary', {})
    stats = aggregated.get('statistics', {})

    print("\n" + "="*60)
    print(f"{task_type.upper()} aggregated statistics")
    print("="*60)
    print(f"Total runs: {summary.get('total_runs', 0)}")
    print(f"Task type: {summary.get('task_type', 'unknown')}")
    print(f"Timestamp: {summary.get('aggregation_timestamp', 'unknown')}")

    # PK stats (if available)
    if 'pk_stats' in summary:
        print(f"\nPK metric stats:")
        for mode, pk_data in summary['pk_stats'].items():
            print(f"  {mode.upper()}: {pk_data['mean']:.4f} ± {pk_data['std']:.4f}")

    if task_type == "finetune":
        _print_finetune_stats(stats)
    elif task_type == "pretrain":
        _print_pretrain_stats(stats)

    print("="*60)


def _print_finetune_stats(stats: Dict[str, Any]):
    """Print finetune statistics."""
    print("\nKey metrics:")

    # Validation metrics
    if 'val' in stats:
        val_keys = ['val_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap','mae', 'rmse', 'r2']
        print("\nValidation metrics:")
        for key in val_keys:
            if key in stats['val']:
                data = stats['val'][key]
                if data:
                    print(f"  {key}: {data['mean']:.4f} +/- {data['std']:.4f} "
                          f"(range: [{data['min']:.4f}, {data['max']:.4f}], n={data['count']})")

    # Test metrics (all aggregation modes)
    if 'test' in stats:
        test_modes = ['avg', 'best', 'learned']
        metrics = ['val_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap','mae', 'rmse', 'r2']

        print("\nTest metrics:")
        for mode in test_modes:
            if mode in stats['test']:
                print(f"\n  {mode.upper()} mode:")
                for metric in metrics:
                    if metric in stats['test'][mode]:
                        data = stats['test'][mode][metric]
                        if data:
                            print(f"    {metric}: {data['mean']:.4f} +/- {data['std']:.4f} "
                                  f"(range: [{data['min']:.4f}, {data['max']:.4f}], n={data['count']})")

    # Time stats
    if 'time' in stats:
        if 'total_train_time_sec' in stats['time']:
            data = stats['time']['total_train_time_sec']
            if data:
                print(f"\nTraining time: {data['mean']:.1f} +/- {data['std']:.1f} sec "
                      f"(range: [{data['min']:.1f}, {data['max']:.1f}])")

        if 'avg_epoch_time_sec' in stats['time']:
            data = stats['time']['avg_epoch_time_sec']
            if data:
                print(f"Avg epoch time: {data['mean']:.1f} +/- {data['std']:.1f} sec "
                      f"(range: [{data['min']:.1f}, {data['max']:.1f}])")

    # Training metrics
    if 'train' in stats:
        if 'last_loss' in stats['train'] or 'learning_rate_last' in stats['train']:
            print(f"\nTraining metrics:")

        if 'last_loss' in stats['train']:
            data = stats['train']['last_loss']
            if data:
                print(f"  Final loss: {data['mean']:.4f} +/- {data['std']:.4f} "
                      f"(range: [{data['min']:.4f}, {data['max']:.4f}], n={data['count']})")

        if 'learning_rate_last' in stats['train']:
            data = stats['train']['learning_rate_last']
            if data:
                print(f"  Final LR: {data['mean']:.6f} +/- {data['std']:.6f} "
                      f"(range: [{data['min']:.6f}, {data['max']:.6f}], n={data['count']})")


def _print_pretrain_stats(stats: Dict[str, Any]):
    """Print pretrain statistics."""
    print("\nKey metrics:")

    if 'best_val_loss' in stats:
        data = stats['best_val_loss']
        print(f"  best_val_loss: {data['mean']:.4f} +/- {data['std']:.4f} "
              f"(range: [{data['min']:.4f}, {data['max']:.4f}], n={data['count']})")

    if 'total_train_time_sec' in stats:
        data = stats['total_train_time_sec']
        print(f"\nTraining time: {data['mean']:.1f} +/- {data['std']:.1f} sec "
              f"(range: [{data['min']:.1f}, {data['max']:.1f}])")

    if 'avg_epoch_time_sec' in stats:
        data = stats['avg_epoch_time_sec']
        print(f"Avg epoch time: {data['mean']:.1f} +/- {data['std']:.1f} sec "
              f"(range: [{data['min']:.1f}, {data['max']:.1f}])")

    if 'effective_max_length' in stats:
        data = stats['effective_max_length']
        print(f"\nMax seq length: {data['mean']:.0f} +/- {data['std']:.0f} "
              f"(range: [{data['min']:.0f}, {data['max']:.0f}])")


def generate_detailed_report(aggregated: Dict[str, Any], task_type: str, output_file: Optional[str] = None) -> str:
    """
    Generate a detailed aggregation report.

    Args:
        aggregated: Aggregated results
        task_type: Task type
        output_file: Output file path; returns string if None

    Returns:
        Report string
    """
    if not aggregated:
        report = "No aggregated results to generate report"
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        return report

    summary = aggregated.get('summary', {})
    stats = aggregated.get('statistics', {})
    individual_runs = aggregated.get('individual_runs', [])

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"{task_type.upper()} Detailed Aggregation Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {summary.get('aggregation_timestamp', 'unknown')}")
    report_lines.append(f"Total runs: {summary.get('total_runs', 0)}")
    report_lines.append(f"Task type: {summary.get('task_type', 'unknown')}")
    report_lines.append("")

    # Statistics summary
    report_lines.append("Statistics Summary:")
    report_lines.append("-" * 40)

    if task_type == "finetune":
        _add_finetune_stats_to_report(stats, report_lines)
    elif task_type == "pretrain":
        _add_pretrain_stats_to_report(stats, report_lines)
    else:
        report_lines.append(f"Unknown task type: {task_type}")

    # Individual run details
    report_lines.append("")
    report_lines.append("Individual Run Details:")
    report_lines.append("-" * 40)

    for run in individual_runs:
        report_lines.append(f"Run {run['run_id']} (seed={run['seed']}):")
        if task_type == "finetune":
            if run.get('best_val_loss') is not None:
                report_lines.append(f"  Best val loss: {run['best_val_loss']:.4f}")
            if run.get('test_mae') is not None:
                report_lines.append(f"  Test MAE: {run['test_mae']:.4f}")
            if run.get('total_train_time_sec') is not None:
                report_lines.append(f"  Training time: {run['total_train_time_sec']:.1f}s")
        elif task_type == "pretrain":
            if run.get('best_val_loss') is not None:
                report_lines.append(f"  Best val loss: {run['best_val_loss']:.4f}")
            if run.get('total_train_time_sec') is not None:
                report_lines.append(f"  Training time: {run['total_train_time_sec']:.1f}s")
            if run.get('effective_max_length') is not None:
                report_lines.append(f"  Max seq length: {run['effective_max_length']}")
        report_lines.append("")

    report_lines.append("=" * 80)

    report = "\n".join(report_lines)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Detailed report saved: {output_file}")

    return report


def _add_finetune_stats_to_report(stats: Dict[str, Any], report_lines: List[str]):
    """Add finetune stats to report."""
    # Validation metrics
    val_keys = ['best_val_mae', 'best_val_rmse', 'best_val_r2', 'best_val_loss']
    report_lines.append("Validation metrics:")
    for key in val_keys:
        if key in stats:
            data = stats[key]
            metric_name = key.replace('best_val_', '')
            report_lines.append(f"  {metric_name}: {data['mean']:.4f} ± {data['std']:.4f}")
            report_lines.append(f"    Range: [{data['min']:.4f}, {data['max']:.4f}] (n={data['count']})")
            if 'ci_95' in data:
                report_lines.append(f"    95% CI: +/-{data['ci_95']:.4f}")

    # Test metrics - all aggregation modes
    test_modes = ['avg', 'best', 'learned']
    primary_metrics = ['mae', 'rmse', 'r2', 'loss']

    for mode in test_modes:
        report_lines.append(f"\n{mode.upper()} aggregation test metrics:")
        for metric in primary_metrics:
            key = f"test_{mode}_{metric}"
            if key in stats:
                data = stats[key]
                report_lines.append(f"  {metric}: {data['mean']:.4f} ± {data['std']:.4f}")
                report_lines.append(f"    Range: [{data['min']:.4f}, {data['max']:.4f}] (n={data['count']})")

    # Time stats
    if 'total_train_time_sec' in stats:
        data = stats['total_train_time_sec']
        report_lines.append("\nTime statistics:")
        report_lines.append(f"  Total training time: {data['mean']:.1f} +/- {data['std']:.1f} sec")
        report_lines.append(f"    Range: [{data['min']:.1f}, {data['max']:.1f}] sec")


def _add_pretrain_stats_to_report(stats: Dict[str, Any], report_lines: List[str]):
    """Add pretrain stats to report."""
    if 'best_val_loss' in stats:
        data = stats['best_val_loss']
        report_lines.append("Validation loss:")
        report_lines.append(f"  Mean: {data['mean']:.4f} +/- {data['std']:.4f}")
        report_lines.append(f"  Range: [{data['min']:.4f}, {data['max']:.4f}] (n={data['count']})")

    if 'total_train_time_sec' in stats:
        data = stats['total_train_time_sec']
        report_lines.append("\nTraining time:")
        report_lines.append(f"  Total: {data['mean']:.1f} +/- {data['std']:.1f} sec")

    if 'effective_max_length' in stats:
        data = stats['effective_max_length']
        report_lines.append("\nSequence length:")
        report_lines.append(f"  Max length: {data['mean']:.0f} +/- {data['std']:.0f}")
