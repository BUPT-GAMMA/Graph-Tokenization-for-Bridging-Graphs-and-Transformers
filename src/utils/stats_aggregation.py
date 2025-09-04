"""
重复运行统计聚合工具
====================

负责聚合多次重复运行的实验结果，计算统计指标（均值、方差、标准差等）。
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
    聚合多次重复运行的实验结果

    Args:
        config: 项目配置
        experiment_name: 实验名称
        repeat_runs: 重复运行次数
        task_type: 任务类型 ("pretrain" 或 "finetune")

    Returns:
        聚合后的统计结果字典
    """
    logger.info(f"📊 开始聚合 {experiment_name} 的 {repeat_runs} 次重复运行结果")

    # 收集所有run的结果
    all_results = []
    valid_runs = 0

    for run_i in range(repeat_runs):
        try:
            # 构建结果文件路径
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
                    logger.info(f"✅ 找到 run_{run_i} 的结果")
            else:
                logger.warning(f"⚠️ run_{run_i} 的结果文件不存在: {results_file}")

        except Exception as e:
            logger.warning(f"⚠️ 读取 run_{run_i} 结果失败: {e}")

    if valid_runs == 0:
        logger.error("❌ 未找到任何有效的运行结果")
        return {}

    logger.info(f"📈 成功收集 {valid_runs}/{repeat_runs} 次运行结果")

    # 聚合结果
    aggregated = _aggregate_results(all_results, task_type)

    # 保存聚合结果
    if output_file is None:
        output_file = config.get_logs_dir(exp_name=experiment_name, run_i=-1) / f"{task_type}_aggregated_stats.json"

    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 聚合结果已保存: {output_file}")
    return aggregated


def _aggregate_results(results: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
    """
    对结果列表进行统计聚合

    Args:
        results: 结果字典列表
        task_type: 任务类型

    Returns:
        聚合后的统计结果
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
        "individual_runs": []  # 保存每个运行的详细信息
    }
    
    def get_fair_best(result, metric):
      """获取公平的最佳测试指标（在avg和learned之间选择最优）"""
      test_data = result['test']
      by_agg = test_data['by_aggregation']
      avg_data = by_agg['avg']
      learned_data = by_agg['learned']

      # 使用正确的任务类型路径
      task_type = result['config']['task']['type']

      # 对于分类任务（包括多标签分类等），选择较高的指标；对于回归任务，选择较低的指标
      if 'classification' in task_type:
        return max(avg_data[metric], learned_data[metric])
      else:  # regression
        return min(avg_data[metric], learned_data[metric])

    # 保存每个运行的基本信息
    for i, result in enumerate(results):
        run_info = {
            "run_id": i,
            "seed": result["config"]["system"]["seed"],  # 存在于 config.system.seed
            "experiment_name": result["config"]["experiment_name"],  # 存在于 config.experiment_name
            "start_time": result.get("start_time"),  # 这个可能不存在
            "end_time": result.get("end_time")  # 这个可能不存在
        }

        # 添加关键指标
        add={}
        if task_type == "finetune":
          # 根据任务类型选择合适的指标
          task_type_config = result['config']['task']['type']
          if task_type_config == "classification":
            key_metrics = ["accuracy", "roc_auc", "ap", "precision", "recall", "f1"]
          else:  # regression or other
            key_metrics = ["mae", "rmse", "r2", "loss"]

          for metric in key_metrics:
            # 使用公平最佳值（avg和learned之间的最优选择）
            fair_value = get_fair_best(result, metric)
            add[f"test_{metric}"] = fair_value
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
    elif task_type == "pretrain":
        aggregated["statistics"] = _aggregate_pretrain_results(results)

    return aggregated


def _aggregate_finetune_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """聚合微调结果"""
    stats = {}

    # 1. 处理验证集指标 - 直接提取已知格式的数据
    val_keys = ['val_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap', 'best_val_roc_auc']
    for key in val_keys:
        values = [r['val'][key] for r in results]
        stats[f'val_{key}'] = _calculate_stats(values)

    # 2. 处理测试集指标（所有聚合模式）- 直接提取已知格式的数据
    test_modes = ['avg', 'best', 'learned']

    # 根据任务类型确定指标集合
    first_result = results[0]
    task_type = first_result['config']['task']['type']

    if 'classification' in task_type:
        test_metrics = ['val_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap']
    else:
        test_metrics = ['val_loss', 'mae', 'rmse', 'r2']

    for mode in test_modes:
        for metric in test_metrics:
            key = f"test_{mode}_{metric}"
            values = [r['test']['by_aggregation'][mode][metric] for r in results]
            stats[key] = _calculate_stats(values)

    # 3. 处理直接测试集指标（向后兼容）
    for metric in test_metrics:
        direct_key = f"test_{metric}"
        values = [r['test'][metric] for r in results]
        stats[direct_key] = _calculate_stats(values)

    # 4. 处理时间指标 - 直接提取
    time_keys = ['total_train_time_sec', 'avg_epoch_time_sec']
    for key in time_keys:
        values = [r['time'][key] for r in results]
        stats[key] = _calculate_stats(values)

    # 5. 处理训练指标 - 直接提取
    train_keys = ['last_loss', 'learning_rate_last']
    for key in train_keys:
        values = [r['train'][key] for r in results]
        stats[f'train_{key}'] = _calculate_stats(values)

    return stats


def _aggregate_pretrain_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """聚合预训练结果"""
    stats = {}

    # 预训练的关键指标
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
    计算数值列表的基本统计信息

    Args:
        values: 数值列表

    Returns:
        统计结果字典
    """
    if not values:
        return {}

    values = np.array(values)
    stats = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)),  # 使用样本标准差
        'var': float(np.var(values, ddof=1)),   # 使用样本方差
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'count': len(values)
    }

    # 计算95%置信区间（假设正态分布）
    # if len(values) > 1:
    #     sem = stats['std'] / np.sqrt(len(values))  # 标准误差
    #     confidence_interval = 1.96 * sem  # 95% CI
    #     stats['ci_95'] = float(confidence_interval)
    #     stats['ci_95_lower'] = stats['mean'] - confidence_interval
    #     stats['ci_95_upper'] = stats['mean'] + confidence_interval

    return stats


def print_aggregated_stats(aggregated: Dict[str, Any], task_type: str):
    """
    打印聚合统计结果

    Args:
        aggregated: 聚合结果
        task_type: 任务类型
    """
    if not aggregated:
        print("❌ 无聚合结果可显示")
        return

    summary = aggregated.get('summary', {})
    stats = aggregated.get('statistics', {})

    print("\n" + "="*60)
    print(f"📊 {task_type.upper()} 重复运行聚合统计")
    print("="*60)
    print(f"总运行次数: {summary.get('total_runs', 0)}")
    print(f"任务类型: {summary.get('task_type', 'unknown')}")
    print(f"聚合时间: {summary.get('aggregation_timestamp', 'unknown')}")

    if task_type == "finetune":
        _print_finetune_stats(stats)
    elif task_type == "pretrain":
        _print_pretrain_stats(stats)

    print("="*60)


def _print_finetune_stats(stats: Dict[str, Any]):
    """打印微调统计结果"""
    print("\n🎯 关键性能指标:")

    # 验证集指标 - 直接打印已知格式
    val_keys = ['val_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap', 'best_val_roc_auc']
    print("\n📊 验证集指标:")
    for key in val_keys:
        stat_key = f'val_{key}'
        if stat_key in stats:
            data = stats[stat_key]
            print(f"  {key}: {data['mean']:.4f} ± {data['std']:.4f} "
                  f"(范围: [{data['min']:.4f}, {data['max']:.4f}], n={data['count']})")

    # 测试集指标（显示所有聚合模式）
    test_modes = ['avg', 'best', 'learned']
    classification_metrics = ['val_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap']

    print("\n🔍 分类任务指标:")
    for mode in test_modes:
        print(f"\n  📊 {mode.upper()}模式:")
        for metric in classification_metrics:
            key = f"test_{mode}_{metric}"
            if key in stats:
                data = stats[key]
                print(f"    {metric}: {data['mean']:.4f} ± {data['std']:.4f} "
                      f"(范围: [{data['min']:.4f}, {data['max']:.4f}], n={data['count']})")

    # 显示直接测试集指标
    print("\n📋 直接测试指标:")
    for metric in classification_metrics:
        direct_key = f"test_{metric}"
        if direct_key in stats:
            data = stats[direct_key]
            print(f"  {metric}: {data['mean']:.4f} ± {data['std']:.4f} "
                  f"(范围: [{data['min']:.4f}, {data['max']:.4f}], n={data['count']})")

    # 时间统计
    if 'total_train_time_sec' in stats:
        data = stats['total_train_time_sec']
        print(f"\n⏱️ 训练时间统计: {data['mean']:.1f} ± {data['std']:.1f} 秒 "
              f"(范围: [{data['min']:.1f}, {data['max']:.1f}])")

    if 'avg_epoch_time_sec' in stats:
        data = stats['avg_epoch_time_sec']
        print(f"⏱️ 平均Epoch时间: {data['mean']:.1f} ± {data['std']:.1f} 秒 "
              f"(范围: [{data['min']:.1f}, {data['max']:.1f}])")

    # 显示训练指标统计
    if 'train_last_loss' in stats:
        data = stats['train_last_loss']
        print(f"\n📚 训练指标:")
        print(f"  最终损失: {data['mean']:.4f} ± {data['std']:.4f} "
              f"(范围: [{data['min']:.4f}, {data['max']:.4f}], n={data['count']})")

    if 'train_learning_rate_last' in stats:
        data = stats['train_learning_rate_last']
        print(f"  最终学习率: {data['mean']:.6f} ± {data['std']:.6f} "
              f"(范围: [{data['min']:.6f}, {data['max']:.6f}], n={data['count']})")


def _print_pretrain_stats(stats: Dict[str, Any]):
    """打印预训练统计结果"""
    print("\n🎯 关键性能指标:")

    if 'best_val_loss' in stats:
        data = stats['best_val_loss']
        print(f"  best_val_loss: {data['mean']:.4f} ± {data['std']:.4f} "
              f"(范围: [{data['min']:.4f}, {data['max']:.4f}], n={data['count']})")

    if 'total_train_time_sec' in stats:
        data = stats['total_train_time_sec']
        print(f"\n⏱️ 训练时间统计: {data['mean']:.1f} ± {data['std']:.1f} 秒 "
              f"(范围: [{data['min']:.1f}, {data['max']:.1f}])")

    if 'avg_epoch_time_sec' in stats:
        data = stats['avg_epoch_time_sec']
        print(f"⏱️ 平均Epoch时间: {data['mean']:.1f} ± {data['std']:.1f} 秒 "
              f"(范围: [{data['min']:.1f}, {data['max']:.1f}])")

    if 'effective_max_length' in stats:
        data = stats['effective_max_length']
        print(f"\n📏 最大序列长度: {data['mean']:.0f} ± {data['std']:.0f} "
              f"(范围: [{data['min']:.0f}, {data['max']:.0f}])")


def generate_detailed_report(aggregated: Dict[str, Any], task_type: str, output_file: Optional[str] = None) -> str:
    """
    生成详细的聚合报告

    Args:
        aggregated: 聚合结果
        task_type: 任务类型
        output_file: 输出文件路径，如果为None则返回字符串

    Returns:
        详细报告字符串
    """
    if not aggregated:
        report = "❌ 无聚合结果可生成报告"
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        return report

    summary = aggregated.get('summary', {})
    stats = aggregated.get('statistics', {})
    individual_runs = aggregated.get('individual_runs', [])

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"📊 {task_type.upper()} 重复运行详细聚合报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {summary.get('aggregation_timestamp', 'unknown')}")
    report_lines.append(f"总运行次数: {summary.get('total_runs', 0)}")
    report_lines.append(f"任务类型: {summary.get('task_type', 'unknown')}")
    report_lines.append("")

    # 统计摘要
    report_lines.append("📈 统计摘要:")
    report_lines.append("-" * 40)

    if task_type == "finetune":
        _add_finetune_stats_to_report(stats, report_lines)
    elif task_type == "pretrain":
        _add_pretrain_stats_to_report(stats, report_lines)
    else:
        report_lines.append(f"⚠️ 未知任务类型: {task_type}")

    # 单个运行详情
    report_lines.append("")
    report_lines.append("🔍 单个运行详情:")
    report_lines.append("-" * 40)

    for run in individual_runs:
        report_lines.append(f"运行 {run['run_id']} (seed={run['seed']}):")
        if task_type == "finetune":
            if run.get('best_val_loss') is not None:
                report_lines.append(f"  最佳验证损失: {run['best_val_loss']:.4f}")
            if run.get('test_mae') is not None:
                report_lines.append(f"  测试MAE: {run['test_mae']:.4f}")
            if run.get('total_train_time_sec') is not None:
                report_lines.append(f"  训练时间: {run['total_train_time_sec']:.1f}秒")
        elif task_type == "pretrain":
            if run.get('best_val_loss') is not None:
                report_lines.append(f"  最佳验证损失: {run['best_val_loss']:.4f}")
            if run.get('total_train_time_sec') is not None:
                report_lines.append(f"  训练时间: {run['total_train_time_sec']:.1f}秒")
            if run.get('effective_max_length') is not None:
                report_lines.append(f"  最大序列长度: {run['effective_max_length']}")
        report_lines.append("")

    report_lines.append("=" * 80)

    report = "\n".join(report_lines)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📝 详细报告已保存: {output_file}")

    return report


def _add_finetune_stats_to_report(stats: Dict[str, Any], report_lines: List[str]):
    """添加微调统计到报告"""
    # 验证集指标
    val_keys = ['best_val_mae', 'best_val_rmse', 'best_val_r2', 'best_val_loss']
    report_lines.append("🎯 验证集指标:")
    for key in val_keys:
        if key in stats:
            data = stats[key]
            metric_name = key.replace('best_val_', '')
            report_lines.append(f"  {metric_name}: {data['mean']:.4f} ± {data['std']:.4f}")
            report_lines.append(f"    范围: [{data['min']:.4f}, {data['max']:.4f}] (n={data['count']})")
            if 'ci_95' in data:
                report_lines.append(f"    95%置信区间: ±{data['ci_95']:.4f}")

    # 测试集指标 - 所有聚合模式
    test_modes = ['avg', 'best', 'learned']
    primary_metrics = ['mae', 'rmse', 'r2', 'loss']

    for mode in test_modes:
        report_lines.append(f"\n📊 {mode.upper()}聚合模式测试指标:")
        for metric in primary_metrics:
            key = f"test_{mode}_{metric}"
            if key in stats:
                data = stats[key]
                report_lines.append(f"  {metric}: {data['mean']:.4f} ± {data['std']:.4f}")
                report_lines.append(f"    范围: [{data['min']:.4f}, {data['max']:.4f}] (n={data['count']})")

    # 时间统计
    if 'total_train_time_sec' in stats:
        data = stats['total_train_time_sec']
        report_lines.append("\n⏱️ 时间统计:")
        report_lines.append(f"  总训练时间: {data['mean']:.1f} ± {data['std']:.1f} 秒")
        report_lines.append(f"    范围: [{data['min']:.1f}, {data['max']:.1f}] 秒")


def _add_pretrain_stats_to_report(stats: Dict[str, Any], report_lines: List[str]):
    """添加预训练统计到报告"""
    if 'best_val_loss' in stats:
        data = stats['best_val_loss']
        report_lines.append("🎯 验证损失:")
        report_lines.append(f"  均值: {data['mean']:.4f} ± {data['std']:.4f}")
        report_lines.append(f"  范围: [{data['min']:.4f}, {data['max']:.4f}] (n={data['count']})")

    if 'total_train_time_sec' in stats:
        data = stats['total_train_time_sec']
        report_lines.append("\n⏱️ 训练时间:")
        report_lines.append(f"  总时间: {data['mean']:.1f} ± {data['std']:.1f} 秒")

    if 'effective_max_length' in stats:
        data = stats['effective_max_length']
        report_lines.append("\n📏 序列长度:")
        report_lines.append(f"  最大长度: {data['mean']:.0f} ± {data['std']:.0f}")
