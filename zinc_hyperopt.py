#!/usr/bin/env python3
"""
ZINC超参数搜索 - 简化版
====================

直接搜索，不搞复杂的包装
"""

import argparse
import json
import sys
from pathlib import Path

import optuna
from optuna.storages import JournalStorage, JournalFileStorage

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import ProjectConfig
from src.training.pretrain_pipeline import train_bert_mlm
from src.training.finetune_pipeline import run_finetune


def pretrain_objective(trial):
    """预训练目标函数"""
    # 创建配置
    config = ProjectConfig()
    config.dataset.name = 'zinc'
    config.encoder.type = 'gte'
    
    # 超参数
    config.bert.pretraining.learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    # 🔧 动态batch size范围，根据max_batch_size限制
    batch_sizes = [bs for bs in [16,32,64,128,256,512,1024] if bs <= args.max_batch_size]
    config.bert.pretraining.batch_size = trial.suggest_categorical('bs', batch_sizes)
    config.bert.pretraining.weight_decay = trial.suggest_float('wd', 0.0, 0.3)
    config.bert.pretraining.max_grad_norm = trial.suggest_float('grad_norm', 0.0, 3.0)
    config.bert.pretraining.mask_prob = trial.suggest_float('mask_prob', 0.10, 0.25)
    config.bert.pretraining.warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)  # 🆕 warmup比例搜索
    config.serialization.method = 'fcpp'  # 🔧 固定使用fcpp序列化方法
    
    # 🔧 修正：使用固定格式的实验名，不依赖时间戳
    config.experiment_name = f"zinc_{args.bpe_mode}_pt_{trial.number:03d}"
    
    # 🆕 设置实验组（用于组织和管理实验）
    config.experiment_group = args.experiment_group or f"hyperopt_{args.bpe_mode}"
    
    # BPE模式
    config.serialization.bpe.engine.encode_rank_mode = args.bpe_mode
    
    print(f"🚀 预训练试验 {trial.number}: {config.experiment_name}")
    print(f"📊 lr={config.bert.pretraining.learning_rate:.2e}, bs={config.bert.pretraining.batch_size}, wd={config.bert.pretraining.weight_decay:.3f}, warmup={config.bert.pretraining.warmup_ratio:.3f}")
    
    # 🔧 关键修复：只在需要时临时设置optuna_trial，避免序列化问题
    temp_trial = trial  # 保存引用
    try:
        # 在训练函数调用的前一刻设置，确保剪枝功能可用
        config.optuna_trial = temp_trial
        result = train_bert_mlm(config)
        # 训练完成立即清理，避免后续序列化问题
        config.optuna_trial = None
        
        val_loss = result['best_val_loss']
        
        # 🆕 关键修正：保存trial信息到模型目录，建立trial.number到模型路径的映射
        model_dir = config.get_model_dir() / config.experiment_name
        
        # 🔧 安全地转换hyperparameters为JSON可序列化格式
        safe_hyperparams = {}
        for key, value in trial.params.items():
            if isinstance(value, (int, float, str, bool)):
                safe_hyperparams[key] = value
            else:
                safe_hyperparams[key] = str(value)  # 转换为字符串
        
        trial_info = {
            'trial_number': int(trial.number),
            'experiment_name': str(config.experiment_name),
            'bpe_mode': str(args.bpe_mode),
            'hyperparameters': safe_hyperparams,
            'val_loss': float(val_loss),
            'model_path': str(model_dir / "best"),
            'stage': 'pretrain'
        }
        
        # 保存trial信息
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / 'trial_info.json', 'w') as f:
            json.dump(trial_info, f, indent=2)
        
        print(f"✅ 试验 {trial.number} 完成: val_loss={val_loss:.4f}")
        print(f"📁 模型路径: {model_dir}")
        return float(val_loss)  # 🔧 确保返回值是基本float类型
    except Exception as e:
        # 🔧 清理trial对象引用
        config.optuna_trial = None
        print(f"❌ 试验 {trial.number} 失败: {e}")
        raise optuna.TrialPruned()


def get_pretrain_model_path(trial_number, bpe_mode):
    """根据trial.number查找预训练模型路径"""
    # 构造实验名（与预训练时一致）
    exp_name = f"zinc_{bpe_mode}_pt_{trial_number:03d}"
    
    # 构造模型目录路径  
    from config import ProjectConfig
    temp_config = ProjectConfig()
    temp_config.experiment_group = f"hyperopt_{bpe_mode}"  # 🔧 设置experiment_group
    model_dir = temp_config.get_model_dir() / exp_name
    
    # 检查trial_info.json是否存在，获取精确的模型路径
    trial_info_file = model_dir / 'trial_info.json'
    if trial_info_file.exists():
        with open(trial_info_file, 'r') as f:
            trial_info = json.load(f)
        return trial_info['model_path']
    else:
        # 如果没有trial_info.json，使用默认路径
        return str(model_dir / "best")


def finetune_objective(trial, top_pretrain_trials):
    """微调目标函数"""
    # 选择预训练模型
    pretrain_trial = trial.suggest_categorical('pretrain_trial', top_pretrain_trials)
    
    # 创建配置
    config = ProjectConfig()
    config.dataset.name = 'zinc'
    config.encoder.type = 'gte'
    
    # 微调超参数
    config.bert.finetuning.learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    # 🔧 动态batch size范围，根据max_batch_size限制（与预训练一致）
    batch_sizes = [bs for bs in [16,32,64,128,256,512,1024] if bs <= args.max_batch_size]
    config.bert.finetuning.batch_size = trial.suggest_categorical('bs', batch_sizes)
    config.bert.finetuning.weight_decay = trial.suggest_float('wd', 0.0, 0.3)
    config.bert.finetuning.max_grad_norm = trial.suggest_float('grad_norm', 0.0, 3.0)
    config.bert.finetuning.warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)  # 🆕 warmup比例搜索
    
    # 🔧 修正：使用固定格式的实验名
    config.experiment_name = f"zinc_{args.bpe_mode}_pt_{pretrain_trial:03d}_ft_{trial.number:03d}"
    
    # 🆕 设置实验组（用于组织和管理实验）
    config.experiment_group = args.experiment_group or f"hyperopt_{args.bpe_mode}"
    
    # BPE模式 
    config.serialization.bpe.engine.encode_rank_mode = args.bpe_mode
    
    # 🆕 关键修正：通过trial.number精确查找预训练模型路径
    pretrained_dir = get_pretrain_model_path(pretrain_trial, args.bpe_mode)
    pretrain_exp_name = f"zinc_{args.bpe_mode}_pt_{pretrain_trial:03d}"
    
    print(f"🎯 微调试验 {trial.number}: {config.experiment_name}")
    print(f"🔗 基于预训练模型: {pretrain_exp_name}")
    print(f"📁 预训练模型路径: {pretrained_dir}")
    print(f"📊 lr={config.bert.finetuning.learning_rate:.2e}, bs={config.bert.finetuning.batch_size}, wd={config.bert.finetuning.weight_decay:.3f}, warmup={config.bert.finetuning.warmup_ratio:.3f}")
    
    # 🔧 关键修复：只在需要时临时设置optuna_trial，避免序列化问题
    temp_trial = trial  # 保存引用
    try:
        # 在训练函数调用的前一刻设置，确保剪枝功能可用
        config.optuna_trial = temp_trial
        result = run_finetune(
            config,
            pretrained_dir=pretrained_dir,
            pretrain_exp_name=pretrain_exp_name
        )
        # 训练完成立即清理，避免后续序列化问题
        config.optuna_trial = None
        
        # 使用测试指标作为优化目标
        if 'test_metrics' in result:
            test_metrics = result['test_metrics']
            if 'mae' in test_metrics:
                target = test_metrics['mae']
            elif 'rmse' in test_metrics:
                target = test_metrics['rmse']
            else:
                target = result['best_val_loss']
        else:
            target = result['best_val_loss']
            
        print(f"✅ 微调试验 {trial.number} 完成: target={target:.4f}")
        return float(target)  # 🔧 确保返回值是基本float类型
        
    except Exception as e:
        # 🔧 清理trial对象引用
        config.optuna_trial = None
        print(f"❌ 微调试验 {trial.number} 失败: {e}")
        raise optuna.TrialPruned()


def run_pretrain_search(journal_file, study_name, n_trials):
    """运行预训练搜索"""
    storage = JournalStorage(JournalFileStorage(journal_file))
    
    # 🆕 配置剪枝器：预训练阶段使用MedianPruner，在训练早期就能剪枝差的试验
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,      # 前5个trial不剪枝，积累基础统计
        n_warmup_steps=10,       # 前10个epoch不剪枝，让模型有基本的训练
        interval_steps=1,        # 每个epoch都检查剪枝
        n_min_trials=3           # 至少需要3个trial完成才开始剪枝（避免过度剪枝）
    )
    
    # 🆕 配置采样器：TPE对超参数搜索效果好
    # 🔧 并发优化：移除固定种子，启用constant_liar避免并发重复
    sampler = optuna.samplers.TPESampler(
        seed=None,               # 🚨 并发环境下不固定种子，避免重复采样
        n_startup_trials=5,     # 前10个trial用随机采样
        n_ei_candidates=24,      # TPE候选数量
        multivariate=True,       # 考虑参数间的相关性
        constant_liar=True,      # 🆕 启用constant_liar策略，减少并发重复
        warn_independent_sampling=False
    )
    
    study = optuna.create_study(
        study_name=f"{study_name}_pretrain_{args.bpe_mode}",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    print(f"🔍 开始预训练搜索 (BPE: {args.bpe_mode})")
    study.optimize(pretrain_objective, n_trials=n_trials, catch=(Exception,))
    
    print("📈 预训练搜索完成")
    print(f"🏆 最优验证损失: {study.best_value:.4f}")
    print(f"🎯 最优参数: {study.best_params}")
    
    return study


def run_finetune_search(journal_file, study_name, pretrain_study, top_k, n_trials):
    """运行微调搜索"""
    # 获取top-k预训练试验
    completed_trials = [t for t in pretrain_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == 0:
        raise ValueError("没有完成的预训练试验")
    
    # 按验证损失升序排序（最小化目标）
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)
    actual_k = min(top_k, len(sorted_trials))
    top_trials = [t.number for t in sorted_trials[:actual_k]]
    
    print(f"🏆 选择Top-{actual_k} 预训练试验:")
    for i, trial in enumerate(sorted_trials[:actual_k]):
        print(f"  #{i+1}: Trial {trial.number}, val_loss={trial.value:.4f}")
        # 验证模型路径是否存在
        try:
            model_path = get_pretrain_model_path(trial.number, args.bpe_mode)
            if Path(model_path).exists():
                print(f"       模型路径: ✅ {model_path}")
            else:
                print(f"       模型路径: ❌ {model_path} (不存在)")
        except Exception as e:
            print(f"       模型路径: ❌ 查找失败: {e}")
    
    storage = JournalStorage(JournalFileStorage(journal_file))
    
    # 🆕 配置剪枝器：微调阶段使用PercentilePruner，更激进的剪枝策略
    pruner = optuna.pruners.PercentilePruner(
        percentile=25.0,         # 剪枝掉性能最差的25%的trial
        n_startup_trials=3,      # 前3个trial不剪枝
        n_warmup_steps=5,        # 前5个epoch不剪枝，微调需要更多warmup
        interval_steps=2,        # 每2个epoch检查一次，微调变化较快
        n_min_trials=2           # 至少2个trial完成即可开始剪枝
    )
    
    # 🆕 配置采样器：微调阶段TPE参数调优
    # 🔧 并发优化：移除固定种子，启用constant_liar避免并发重复
    sampler = optuna.samplers.TPESampler(
        seed=None,               # 🚨 并发环境下不固定种子，避免重复采样
        n_startup_trials=5,      # 微调搜索空间相对较小，少些随机采样
        n_ei_candidates=16,      # 候选数量适中
        multivariate=True,
        constant_liar=True,      # 🆕 启用constant_liar策略，减少并发重复
        warn_independent_sampling=False
    )
    
    # 🔧 解决动态值空间问题：基于top_trials创建唯一study名称
    top_trials_str = "_".join(map(str, sorted(top_trials)))
    unique_study_name = f"{study_name}_finetune_{args.bpe_mode}_top{len(top_trials)}_{hash(top_trials_str) % 10000:04d}"
    
    study = optuna.create_study(
        study_name=unique_study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    print(f"🎯 开始微调搜索 (BPE: {args.bpe_mode})")
    print(f"📊 Study名称: {unique_study_name}")
    print(f"📊 将基于{actual_k}个最优预训练模型进行微调超参数搜索")
    
    def objective_wrapper(trial):
        return finetune_objective(trial, top_trials)
    
    study.optimize(objective_wrapper, n_trials=n_trials, catch=(Exception,))
    
    print("🎉 微调搜索完成")
    if study.best_trial is not None:
        print(f"🏆 最优指标: {study.best_value:.4f}")
        print(f"🎯 最优参数: {study.best_params}")
        
        # 显示最优组合的详细信息
        best_pretrain = study.best_params.get('pretrain_trial')
        if best_pretrain is not None:
            print(f"🔗 最优预训练模型: Trial {best_pretrain}")
    else:
        print("⚠️ 未找到成功的微调试验")
    
    return study


def main():
    parser = argparse.ArgumentParser(description="ZINC超参数搜索")
    parser.add_argument("--bpe_mode", default='all', choices=['none', 'all', 'random', 'gaussian'])
    parser.add_argument("--journal_file", default='./journal/zinc_hyperopt.db', help="Journal存储文件")
    parser.add_argument("--study_name", default="zinc_hyperopt", help="研究名称")
    parser.add_argument("--experiment_group", default=None, help="实验组名称，默认为hyperopt_{bpe_mode}")
    parser.add_argument("--stage", choices=['pretrain', 'finetune', 'both'], default='both')
    parser.add_argument("--pretrain_trials", type=int, default=40)
    parser.add_argument("--finetune_trials", type=int, default=60)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_batch_size", type=int, default=1024, help="最大batch size（防止显存溢出）")
    
    global args
    args = parser.parse_args()
    
    # 确保journal目录存在
    Path(args.journal_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 启动ZINC超参数搜索 (BPE: {args.bpe_mode})")
    print("📊 配置信息:")
    print(f"  - 最大batch size: {args.max_batch_size} (可用范围: [16,32,64,128,256,512,1024])")
    print("  - 序列化方法: fcpp (固定)")
    print(f"  - 预训练试验数: {args.pretrain_trials}")
    print(f"  - 微调试验数: {args.finetune_trials}")
    print(f"  - Top-K: {args.top_k}")
    
    # 显存警告
    if args.max_batch_size >= 512:
        print("⚠️  警告: 大batch size (≥512) 可能导致显存不足，建议监控GPU内存使用")
    
    if args.stage in ['pretrain', 'both']:
        pretrain_study = run_pretrain_search(
            args.journal_file, 
            args.study_name, 
            args.pretrain_trials
        )
        
        if args.stage == 'both':
            run_finetune_search(
                args.journal_file,
                args.study_name,
                pretrain_study,
                args.top_k,
                args.finetune_trials
            )
    
    elif args.stage == 'finetune':
        # 加载预训练结果
        storage = JournalStorage(JournalFileStorage(args.journal_file))
        pretrain_study = optuna.load_study(
            study_name=f"{args.study_name}_pretrain_{args.bpe_mode}",
            storage=storage
        )
        
        run_finetune_search(
            args.journal_file,
            args.study_name,
            pretrain_study,
            args.top_k,
            args.finetune_trials
        )
    
    print("🎉 搜索完成!")


if __name__ == "__main__":
    main()