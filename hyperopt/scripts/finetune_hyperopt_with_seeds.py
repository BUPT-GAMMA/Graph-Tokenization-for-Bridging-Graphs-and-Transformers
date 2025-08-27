#!/usr/bin/env python3
"""
基于最佳预训练参数的微调超参数搜索
==================================
使用从large_batch.db提取的最佳预训练参数作为种子起始点，
进行微调阶段的超参数优化搜索。

🎯 种子来源：
- Top3 overall结果（都是feuler）
- eulerian、feuler、cpp、fcpp的最佳参数
- 默认配置参数（如有预训练模型）

🔧 微调搜索空间：
- Learning rate: [1e-5, 1e-3] (log scale)
- Batch size: [32, 64, 128] (微调用较小batch)  
- Weight decay: [0.0, 0.3]
- Gradient norm: [0.5, 3.0]
- Warmup ratio: [0.0, 0.3]
- Epochs: [3, 5, 10, 15]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

# 添加项目路径
sys.path.append('.')

from config import ProjectConfig
from src.training.finetune_pipeline import run_finetune


def load_seed_params(seed_file):
    """加载种子参数"""
    with open(seed_file, 'r', encoding='utf-8') as f:
        seed_data = json.load(f)
    
    # 转换为适合enqueue的格式，排除默认配置（因为可能没有预训练模型）
    seed_params = []
    for key, params in seed_data.items():
        if key == 'default_config' and params['loss'] is None:
            print(f"⚠️ 跳过默认配置 (没有预训练模型)")
            continue
            
        # 转换为微调参数（预训练参数需要适配到微调）
        finetune_params = {
            'ft_lr': params['lr'] * 0.1,  # 微调学习率通常比预训练小
            'ft_bs': min(params['bs'], 128),  # 微调批次不要太大
            'ft_wd': params['wd'] * 0.5,  # 微调权重衰减适度减小
            'ft_grad_norm': params['grad_norm'],
            'ft_warmup_ratio': params['warmup_ratio'] * 0.5,  # 微调预热比例较小
            'ft_epochs': 5,  # 默认epochs
            'method': params['method'],
            'seed_source': key,
            'original_loss': params.get('loss')
        }
        seed_params.append(finetune_params)
        print(f"🌱 加载种子 {key}: method={params['method']}, loss={params.get('loss', 'N/A')}")
    
    print(f"✅ 总共加载 {len(seed_params)} 个种子参数")
    return seed_params


def find_pretrained_model_for_seed(seed_source, method, bpe_mode, original_loss, search_journal_file):
    """根据种子信息查找对应的预训练模型路径"""
    try:
        # 加载原始搜索结果，找到对应的trial
        from optuna.storages.journal import JournalFileBackend
        storage = JournalStorage(JournalFileBackend(search_journal_file))
        study = optuna.load_study('methods_large_batch_pretrain_all', storage)
        
        # 查找匹配的trial
        for trial in study.trials:
            if (trial.state == optuna.trial.TrialState.COMPLETE and 
                trial.params.get('method') == method and
                abs(trial.value - original_loss) < 1e-6):  # 匹配loss值
                
                # 构建预训练模型路径
                config = ProjectConfig()
                config.experiment_name = f"large_bs_{bpe_mode}_pt_{trial.number:03d}"
                config.experiment_group = f"large_bs_hyperopt_{bpe_mode}"
                pretrained_dir = config.get_model_dir() / config.experiment_name / "best"
                
                if pretrained_dir.exists():
                    print(f"🎯 找到预训练模型: Trial {trial.number} -> {pretrained_dir}")
                    return str(pretrained_dir), f"large_bs_{bpe_mode}_pt_{trial.number:03d}"
                else:
                    print(f"⚠️ 预训练模型路径不存在: {pretrained_dir}")
        
        print(f"❌ 未找到匹配的预训练模型: {seed_source} (method={method}, loss={original_loss})")
        return None, None
        
    except Exception as e:
        print(f"❌ 查找预训练模型失败: {e}")
        return None, None


def finetune_objective(trial):
    """微调目标函数"""
    config = ProjectConfig()
    config.dataset.name = 'zinc'
    config.encoder.type = 'gte'
    
    # 微调超参数空间
    config.bert.finetuning.learning_rate = trial.suggest_float('ft_lr', 1e-5, 1e-3, log=True)
    config.bert.finetuning.batch_size = trial.suggest_categorical('ft_bs', [32, 64, 128])
    config.bert.finetuning.weight_decay = trial.suggest_float('ft_wd', 0.0, 0.3)
    config.bert.finetuning.max_grad_norm = trial.suggest_float('ft_grad_norm', 0.5, 3.0)
    config.bert.finetuning.warmup_ratio = trial.suggest_float('ft_warmup_ratio', 0.0, 0.3)
    config.bert.finetuning.num_epochs = trial.suggest_categorical('ft_epochs', [3, 5, 10, 15])
    
    # 使用固定的序列化方法（从种子或搜索中选择）
    if 'method' in trial.params:
        config.serialization.method = trial.params['method']
    else:
        config.serialization.method = trial.suggest_categorical('method', ['feuler', 'eulerian', 'cpp', 'fcpp'])
    
    # 实验管理
    config.experiment_name = f"ft_seed_{args.bpe_mode}_{trial.number:03d}"
    config.experiment_group = f"finetune_with_seeds_{args.bpe_mode}"
    config.serialization.bpe.engine.encode_rank_mode = args.bpe_mode
    
    print(f"🎯 Finetune Trial {trial.number}: lr={config.bert.finetuning.learning_rate:.2e}, "
          f"bs={config.bert.finetuning.batch_size}, epochs={config.bert.finetuning.num_epochs}, "
          f"method={config.serialization.method}")
    
    try:
        config.optuna_trial = trial
        start_time = time.time()
        
        # 查找对应的预训练模型
        pretrained_dir = None
        pretrain_exp_name = None
        
        if hasattr(trial, 'user_attrs') and 'seed_source' in trial.user_attrs:
            # 如果是种子trial，需要找到对应的预训练模型
            seed_source = trial.user_attrs['seed_source']
            method = trial.user_attrs.get('method', config.serialization.method)
            original_loss = trial.user_attrs.get('original_loss')
            
            print(f"📦 基于种子源: {seed_source} (method={method})")
            
            # 查找预训练模型路径
            search_journal = args.pretrain_journal
            pretrained_dir, pretrain_exp_name = find_pretrained_model_for_seed(
                seed_source, method, args.bpe_mode, original_loss, search_journal
            )
            
            if pretrained_dir is None:
                print(f"⚠️ 未找到种子对应的预训练模型，跳过此trial")
                raise optuna.TrialPruned("No pretrained model found")
        
        # 运行微调
        result = run_finetune(config, pretrained_dir=pretrained_dir, 
                             pretrain_exp_name=pretrain_exp_name)
        training_time_minutes = (time.time() - start_time) / 60.0
        config.optuna_trial = None
        
        # 使用测试集指标作为优化目标
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
        trial.set_user_attr('search_type', 'finetune_with_seeds')
        
        print(f"✅ Finetune Trial {trial.number} 完成: target={target:.4f}, time={training_time_minutes:.1f}min")
        return float(target)
        
    except optuna.TrialPruned:
        config.optuna_trial = None
        raise
    except Exception as e:
        config.optuna_trial = None
        print(f"❌ Finetune Trial {trial.number} 失败: {e}")
        raise


def add_seeds_to_study(study, seed_params):
    """将种子参数加入搜索队列（并发安全）"""
    added = 0
    print("🌱 添加种子起始点:")
    
    for i, params in enumerate(seed_params, 1):
        try:
            # 移除非参数字段，但保留method作为参数
            enqueue_params = {k: v for k, v in params.items() 
                            if k not in ['seed_source', 'original_loss']}
            
            # 添加用户属性
            user_attrs = {
                'seed_source': params['seed_source'],
                'method': params['method'],
                'original_loss': params.get('original_loss')
            }
            
            study.enqueue_trial(enqueue_params, user_attrs=user_attrs)
            print(f"  种子 {i}: {params['seed_source']} (method={params['method']})")
            added += 1
            
        except Exception as e:
            print(f"  种子 {i}: 跳过 (可能已被其他进程添加) - {e}")
    
    print(f"🚀 已添加 {added} 个种子参数")


def main():
    global args
    parser = argparse.ArgumentParser(description="基于种子的微调超参数搜索")
    parser.add_argument("--bpe_mode", default='all', choices=['none', 'all', 'random', 'gaussian'])
    parser.add_argument("--journal_file", default='hyperopt/journal/finetune_with_seeds.db')
    parser.add_argument("--seed_file", default='hyperopt/results/best_pretrain_params_for_finetuning.json')
    parser.add_argument("--pretrain_journal", default='hyperopt/journal/large_batch.db')
    parser.add_argument("--trials", type=int, default=50)
    
    args = parser.parse_args()
    
    print(f"🎯 基于种子的微调搜索 - BPE: {args.bpe_mode}")
    
    # 加载种子参数
    if not Path(args.seed_file).exists():
        print(f"❌ 种子文件不存在: {args.seed_file}")
        print("请先运行 extract_best_params_for_finetuning.py 提取种子参数")
        return
    
    seed_params = load_seed_params(args.seed_file)
    if not seed_params:
        print("❌ 没有可用的种子参数")
        return
    
    # 创建存储和研究
    storage = JournalStorage(JournalFileBackend(args.journal_file))
    
    # 配置剪枝器和采样器（针对微调优化）
    pruner = optuna.pruners.PercentilePruner(
        percentile=30.0,  # 稍微保守一些
        n_startup_trials=5,
        n_warmup_steps=3,  # 微调很快，不需要太长预热
        interval_steps=1,
        n_min_trials=2
    )
    
    sampler = optuna.samplers.TPESampler(
        seed=None,  # 并发环境下不固定种子
        n_startup_trials=len(seed_params) + 2,  # 种子数量 + 一些随机试验
        n_ei_candidates=16,
        multivariate=True,
        constant_liar=True,  # 并发优化
        warn_independent_sampling=False
    )
    
    study = optuna.create_study(
        study_name=f"finetune_with_seeds_{args.bpe_mode}",
        storage=storage, 
        direction="minimize", 
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    # 并发安全的种子添加逻辑
    seed_trials = [t for t in study.trials if 'seed_source' in t.user_attrs]
    if len(seed_trials) < len(seed_params):
        print(f"🌱 检测到种子数量不足 ({len(seed_trials)}/{len(seed_params)})，添加种子...")
        add_seeds_to_study(study, seed_params)
    else:
        print(f"✅ 种子已完整 ({len(seed_trials)}/{len(seed_params)})，跳过添加")
    
    # 开始优化
    print(f"\n{'='*40} 微调搜索 {'='*40}")
    study.optimize(finetune_objective, n_trials=args.trials)
    
    print(f"🏆 微调最优结果: {study.best_value:.4f}")
    print(f"🏆 最优参数: {study.best_params}")
    
    print("✅ 微调搜索完成！")


if __name__ == "__main__":
    main()
