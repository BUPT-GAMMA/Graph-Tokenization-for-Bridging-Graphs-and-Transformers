#!/usr/bin/env python3
"""
大Batch Size专用超参数搜索 - 简化版
================================
专注于[128, 256, 512]，包含已知最优结果作为起始点
"""

import argparse
import json
import sys
import time
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

# 添加项目路径（假设从项目根目录运行）
sys.path.append('.')

from config import ProjectConfig
from src.training.pretrain_pipeline import train_bert_mlm
from src.training.finetune_pipeline import run_finetune


# 🌱 内置的种子数据（从现有搜索结果提取）
SEED_PARAMS = {
    128: [
        {'lr': 3.22e-04, 'bs': 128, 'wd': 0.243, 'grad_norm': 1.72, 'mask_prob': 0.159, 'warmup_ratio': 0.159},
        {'lr': 2.83e-04, 'bs': 128, 'wd': 0.248, 'grad_norm': 2.85, 'mask_prob': 0.157, 'warmup_ratio': 0.157},
        {'lr': 3.97e-04, 'bs': 128, 'wd': 0.199, 'grad_norm': 1.75, 'mask_prob': 0.175, 'warmup_ratio': 0.175}
    ],
    256: [
        {'lr': 3.15e-04, 'bs': 256, 'wd': 0.271, 'grad_norm': 1.72, 'mask_prob': 0.105, 'warmup_ratio': 0.105},
        {'lr': 3.12e-04, 'bs': 256, 'wd': 0.235, 'grad_norm': 2.23, 'mask_prob': 0.123, 'warmup_ratio': 0.123},
        {'lr': 3.61e-04, 'bs': 256, 'wd': 0.180, 'grad_norm': 1.50, 'mask_prob': 0.150, 'warmup_ratio': 0.150}
    ],
    512: [
        {'lr': 3.01e-04, 'bs': 512, 'wd': 0.252, 'grad_norm': 2.85, 'mask_prob': 0.161, 'warmup_ratio': 0.130},
        {'lr': 5.11e-04, 'bs': 512, 'wd': 0.006, 'grad_norm': 2.91, 'mask_prob': 0.291, 'warmup_ratio': 0.291},
        {'lr': 1.70e-04, 'bs': 512, 'wd': 0.261, 'grad_norm': 2.03, 'mask_prob': 0.030, 'warmup_ratio': 0.030}
    ]
}


def pretrain_objective(trial):
    """预训练目标函数"""
    config = ProjectConfig()
    config.dataset.name = 'zinc'
    config.encoder.type = 'gte'
    
    # 大batch size范围
    config.bert.pretraining.learning_rate = trial.suggest_float('lr', 8e-5, 5e-4, log=True)
    config.bert.pretraining.batch_size = trial.suggest_categorical('bs', [128, 256, 512])
    config.bert.pretraining.weight_decay = trial.suggest_float('wd', 0.05, 0.25)
    config.bert.pretraining.max_grad_norm = trial.suggest_float('grad_norm', 1.5, 5.0)
    config.bert.pretraining.mask_prob = trial.suggest_float('mask_prob', 0.05, 0.15)
    config.bert.pretraining.warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.15)
    config.serialization.method = 'fcpp'
    
    # 实验管理
    config.experiment_name = f"large_bs_{args.bpe_mode}_pt_{trial.number:03d}"
    config.experiment_group = f"large_bs_hyperopt_{args.bpe_mode}"
    config.serialization.bpe.engine.encode_rank_mode = args.bpe_mode
    
    print(f"🚀 Trial {trial.number}: lr={config.bert.pretraining.learning_rate:.2e}, bs={config.bert.pretraining.batch_size}, wd={config.bert.pretraining.weight_decay:.3f}")
    
    try:
        config.optuna_trial = trial
        start_time = time.time()
        result = train_bert_mlm(config)
        training_time_minutes = (time.time() - start_time) / 60.0
        config.optuna_trial = None
        
        val_loss = result['best_val_loss']
        
        # 记录时间信息
        trial.set_user_attr('training_time_minutes', training_time_minutes)
        trial.set_user_attr('search_type', 'large_batch')
        
        print(f"✅ Trial {trial.number} 完成: val_loss={val_loss:.4f}, time={training_time_minutes:.1f}min")
        return float(val_loss)
        
    except optuna.TrialPruned:
        config.optuna_trial = None
        raise
    except Exception as e:
        config.optuna_trial = None
        print(f"❌ Trial {trial.number} 失败: {e}")
        raise


def finetune_objective(trial, top_pretrain_trials):
    """微调目标函数 - 与原版相同"""
    pretrain_trial = trial.suggest_categorical('pretrain_trial', top_pretrain_trials)
    
    config = ProjectConfig()
    config.dataset.name = 'zinc'
    config.encoder.type = 'gte'
    
    config.bert.finetuning.learning_rate = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    config.bert.finetuning.batch_size = trial.suggest_categorical('bs', [128, 256, 512])
    config.bert.finetuning.weight_decay = trial.suggest_float('wd', 0.0, 0.3)
    config.bert.finetuning.max_grad_norm = trial.suggest_float('grad_norm', 0.5, 5.0)
    config.bert.finetuning.warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)
    
    config.experiment_name = f"large_bs_{args.bpe_mode}_pt_{pretrain_trial:03d}_ft_{trial.number:03d}"
    config.experiment_group = f"large_bs_hyperopt_{args.bpe_mode}"
    config.serialization.bpe.engine.encode_rank_mode = args.bpe_mode
    
    # 查找预训练模型
    temp_config = ProjectConfig()
    temp_config.experiment_name = f"large_bs_{args.bpe_mode}_pt_{pretrain_trial:03d}"
    temp_config.experiment_group = f"large_bs_hyperopt_{args.bpe_mode}"
    pretrained_dir = temp_config.get_model_dir() / temp_config.experiment_name / "best"
    
    print(f"🎯 Finetune Trial {trial.number}: bs={config.bert.finetuning.batch_size}, 基于PT-{pretrain_trial}")
    
    try:
        config.optuna_trial = trial
        start_time = time.time()
        result = run_finetune(config, pretrained_dir=pretrained_dir, 
                             pretrain_exp_name=f"large_bs_{args.bpe_mode}_pt_{pretrain_trial:03d}")
        training_time_minutes = (time.time() - start_time) / 60.0
        config.optuna_trial = None
        
        # 🎯 使用测试集指标作为优化目标
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
        
        # 记录时间信息
        trial.set_user_attr('training_time_minutes', training_time_minutes)
        trial.set_user_attr('search_type', 'large_batch')
        trial.set_user_attr('pretrain_trial', pretrain_trial)
        
        print(f"✅ Finetune Trial {trial.number} 完成: target={target:.4f}, time={training_time_minutes:.1f}min")
        return float(target)
        
    except optuna.TrialPruned:
        config.optuna_trial = None
        raise
    except Exception as e:
        config.optuna_trial = None
        print(f"❌ Finetune Trial {trial.number} 失败: {e}")
        raise


def add_seeds_to_study(study):
    """将种子参数加入搜索队列（并发安全）"""
    added = 0
    print("🌱 添加种子起始点:")
    for bs, params_list in SEED_PARAMS.items():
        for i, params in enumerate(params_list, 1):
            try:
                # 添加用户属性标记种子来源，方便识别
                study.enqueue_trial(params, user_attrs={'seed_source': f'BS{bs}_top{i}'})
                print(f"  BS={bs} #{i}: lr={params['lr']:.2e}, wd={params['wd']:.3f}")
                added += 1
            except Exception as e:
                # 如果其他主机已添加相同参数，跳过
                print(f"  BS={bs} #{i}: 跳过 (可能已被其他主机添加)")
    print(f"🚀 已添加 {added} 个种子参数")


def main():
    global args
    parser = argparse.ArgumentParser(description="大Batch Size专用ZINC搜索")
    parser.add_argument("--bpe_mode", default='all', choices=['none', 'all', 'random', 'gaussian'])
    parser.add_argument("--journal_file", default='hyperopt/journal/large_batch.db')
    parser.add_argument("--stage", choices=['pretrain', 'finetune', 'both'], default='pretrain')
    parser.add_argument("--pretrain_trials", type=int, default=30)
    parser.add_argument("--finetune_trials", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=3)
    
    args = parser.parse_args()
    
    print(f"🎯 大Batch Size搜索 [128, 256, 512] - BPE: {args.bpe_mode}")
    
    storage = JournalStorage(JournalFileBackend(args.journal_file))
    
    if args.stage in ['pretrain', 'both']:
        print(f"\n{'='*30} 预训练搜索 {'='*30}")
        
        # 🔧 配置剪枝器和采样器，优化并发性能
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=8, 
            n_warmup_steps=25,
            interval_steps=1,
            n_min_trials=3
        )
        
        # 🔧 并发优化配置：与zinc_hyperopt.py保持一致
        sampler = optuna.samplers.TPESampler(
            seed=None,               # 🚨 并发环境下不固定种子，避免重复采样
            n_startup_trials=10,     # 前10个trial用随机采样
            n_ei_candidates=24,      # TPE候选数量
            multivariate=True,       # 考虑参数间的相关性
            constant_liar=True,      # 🆕 启用constant_liar策略，减少并发重复
            warn_independent_sampling=False
        )
        
        study = optuna.create_study(
            study_name=f"large_batch_pretrain_{args.bpe_mode}",
            storage=storage, direction="minimize", load_if_exists=True,
            pruner=pruner,
            sampler=sampler
        )
        
        # 🔒 并发安全的种子添加逻辑
        seed_trials = [t for t in study.trials if 'seed_source' in t.user_attrs]
        expected_seeds = sum(len(params_list) for params_list in SEED_PARAMS.values())  # 动态计算总种子数
        if len(seed_trials) < expected_seeds:
            print(f"🌱 检测到种子数量不足 ({len(seed_trials)}/{expected_seeds})，尝试添加...")
            add_seeds_to_study(study)
        else:
            print(f"✅ 种子已完整 ({len(seed_trials)}/{expected_seeds})，跳过添加")
        
        study.optimize(pretrain_objective, n_trials=args.pretrain_trials)
        print(f"🏆 预训练最优: {study.best_value:.4f}")
    
    if args.stage in ['finetune', 'both']:
        print(f"\n{'='*30} 微调搜索 {'='*30}")
        
        # 加载预训练结果
        pretrain_study = optuna.load_study(f"large_batch_pretrain_{args.bpe_mode}", storage)
        completed = [t for t in pretrain_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed.sort(key=lambda x: x.value)
        top_trials = [t.number for t in completed[:args.top_k]]
        
        print(f"📊 基于top-{len(top_trials)}预训练模型: {top_trials}")
        
        # 🔧 微调阶段的并发优化配置
        ft_pruner = optuna.pruners.PercentilePruner(
            percentile=25.0,
            n_startup_trials=3,
            n_warmup_steps=5,
            interval_steps=2,
            n_min_trials=2
        )
        
        ft_sampler = optuna.samplers.TPESampler(
            seed=None,               # 🚨 并发环境下不固定种子
            n_startup_trials=5,      # 微调阶段少一些随机试验
            n_ei_candidates=16,
            multivariate=True,
            constant_liar=True,      # 🆕 并发优化
            warn_independent_sampling=False
        )
        
        finetune_study = optuna.create_study(
            study_name=f"large_batch_finetune_{args.bpe_mode}_{hash(str(top_trials)) % 1000:03d}",
            storage=storage, direction="minimize", load_if_exists=True,
            pruner=ft_pruner,
            sampler=ft_sampler
        )
        
        finetune_study.optimize(lambda trial: finetune_objective(trial, top_trials), n_trials=args.finetune_trials)
        print(f"🏆 微调最优: {finetune_study.best_value:.4f}")
    
    print("✅ 搜索完成！")


if __name__ == "__main__":
    main()