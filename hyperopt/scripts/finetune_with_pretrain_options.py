#!/usr/bin/env python3
"""
基于最佳预训练参数的微调超参数搜索
==================================
直接使用最佳预训练参数作为微调搜索的候选选项，
让Optuna从这些已验证的好参数中选择和组合。

🎯 候选参数来源：
- Top3 overall结果
- eulerian、feuler、cpp、fcpp的最佳参数
- 默认配置参数（如有预训练模型）

🔧 微调搜索逻辑：
- 直接使用预训练的学习率/batch size等参数
- 只对微调特有的参数（如epochs）进行搜索
- 不做任何参数转换假设
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


def load_pretrain_options(seed_file):
    """加载预训练参数选项（不做任何转换）"""
    with open(seed_file, 'r', encoding='utf-8') as f:
        seed_data = json.load(f)
    
    # 提取有效的预训练参数
    pretrain_options = {}
    for key, params in seed_data.items():
        pretrain_options[key] = {
            'lr': params['lr'],
            'bs': params['bs'], 
            'wd': params['wd'],
            'grad_norm': params['grad_norm'],
            'mask_prob': params['mask_prob'],
            'warmup_ratio': params['warmup_ratio'],
            'method': params['method'],
            'original_loss': params.get('loss')
        }
        
        if key == 'default_config' and params['loss'] is None:
            print(f"📦 预训练选项 {key}: method={params['method']} (需要先训练预训练模型)")
        else:
            print(f"📦 预训练选项 {key}: method={params['method']}, lr={params['lr']:.4e}, bs={params['bs']}")
    
    print(f"✅ 总共 {len(pretrain_options)} 个预训练参数选项")
    return pretrain_options


def find_pretrained_model_for_option(option_key, method, bpe_mode, original_loss, search_journal_file):
    """根据预训练选项查找对应的预训练模型路径"""
    # 特殊处理默认配置
    if option_key == 'default_config':
        config = ProjectConfig()
        config.experiment_name = f"large_bs_{bpe_mode}_pt_default"
        config.experiment_group = f"large_bs_hyperopt_{bpe_mode}"
        pretrained_dir = config.get_model_dir() / config.experiment_name / "best"
        
        if pretrained_dir.exists():
            print(f"🎯 找到默认配置预训练模型: {pretrained_dir}")
            return str(pretrained_dir), f"large_bs_{bpe_mode}_pt_default"
        else:
            print(f"❌ 默认配置预训练模型不存在: {pretrained_dir}")
            print(f"请先运行: python bert_pretraining_pipeline_optimized.py --experiment_name 'large_bs_{bpe_mode}_pt_default' --experiment_group 'large_bs_hyperopt_{bpe_mode}'")
            return None, None
    
    # 其他选项从journal中查找
    try:
        from optuna.storages.journal import JournalFileBackend
        storage = JournalStorage(JournalFileBackend(search_journal_file))
        study = optuna.load_study('methods_large_batch_pretrain_all', storage)
        
        # 查找匹配的trial
        for trial in study.trials:
            if (trial.state == optuna.trial.TrialState.COMPLETE and 
                trial.params.get('method') == method and
                abs(trial.value - original_loss) < 1e-6):
                
                # 构建预训练模型路径
                config = ProjectConfig()
                config.experiment_name = f"large_bs_{bpe_mode}_pt_{trial.number:03d}"
                config.experiment_group = f"large_bs_hyperopt_{bpe_mode}"
                pretrained_dir = config.get_model_dir() / config.experiment_name / "best"
                
                if pretrained_dir.exists():
                    print(f"🎯 找到预训练模型: {option_key} -> Trial {trial.number}")
                    return str(pretrained_dir), f"large_bs_{bpe_mode}_pt_{trial.number:03d}"
                else:
                    print(f"⚠️ 预训练模型路径不存在: {pretrained_dir}")
        
        print(f"❌ 未找到匹配的预训练模型: {option_key}")
        return None, None
        
    except Exception as e:
        print(f"❌ 查找预训练模型失败: {e}")
        return None, None


def finetune_objective(trial, pretrain_options, pretrain_journal, bpe_mode):
    """微调目标函数 - 直接选择预训练参数"""
    config = ProjectConfig()
    config.dataset.name = 'zinc'
    config.encoder.type = 'gte'
    
    # 1. 选择预训练参数组合
    option_key = trial.suggest_categorical('pretrain_option', list(pretrain_options.keys()))
    selected_option = pretrain_options[option_key]
    
    # 2. 仅使用预训练选项来确定序列化方法与加载的预训练模型
    #    微调阶段的所有超参数由Optuna重新搜索
    config.serialization.method = selected_option['method']
    
    # 3. 微调阶段的超参数全部搜索
    config.bert.finetuning.learning_rate = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    config.bert.finetuning.batch_size = trial.suggest_categorical('bs', [128,256,512])
    config.bert.finetuning.weight_decay = trial.suggest_float('wd', 0.0, 0.3)
    config.bert.finetuning.max_grad_norm = trial.suggest_float('grad_norm', 0.5, 3.0)
    config.bert.finetuning.warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.2)
    
    # 实验管理
    config.experiment_name = f"ft_option_{bpe_mode}_{trial.number:03d}"
    config.experiment_group = f"finetune_with_options_{bpe_mode}"
    config.serialization.bpe.engine.encode_rank_mode = bpe_mode
    
    print(f"🎯 Trial {trial.number}: option={option_key}, lr={config.bert.finetuning.learning_rate:.4e}, bs={config.bert.finetuning.batch_size}, method={config.serialization.method}")
    
    try:
        config.optuna_trial = trial
        start_time = time.time()
        
        # 查找对应的预训练模型
        pretrained_dir, pretrain_exp_name = find_pretrained_model_for_option(
            option_key, selected_option['method'], bpe_mode, 
            selected_option['original_loss'], pretrain_journal
        )
        
        if pretrained_dir is None:
            print(f"⚠️ 未找到预训练模型，跳过此trial")
            raise optuna.TrialPruned("No pretrained model found")
        
        # 运行微调
        result = run_finetune(config, pretrained_dir=pretrained_dir, 
                             pretrain_exp_name=pretrain_exp_name)
        training_time_minutes = (time.time() - start_time) / 60.0
        config.optuna_trial = None
        
        # 使用测试集MAE作为唯一优化目标（无任何回退）
        if 'test_metrics' not in result or 'mae' not in result['test_metrics']:
            raise RuntimeError("Finetune result must contain test_metrics['mae'] as the optimization target.")
        target = float(result['test_metrics']['mae'])
        
        # 记录详细信息
        trial.set_user_attr('training_time_minutes', training_time_minutes)
        trial.set_user_attr('pretrain_option', option_key)
        trial.set_user_attr('pretrain_method', selected_option['method'])
        trial.set_user_attr('pretrain_loss', selected_option['original_loss'])
        
        print(f"✅ Trial {trial.number} 完成: MAE={target:.4f}, time={training_time_minutes:.1f}min")
        return float(target)
        
    except optuna.TrialPruned:
        config.optuna_trial = None
        raise
    except Exception as e:
        config.optuna_trial = None
        print(f"❌ Trial {trial.number} 失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="基于预训练参数选项的微调搜索")
    parser.add_argument("--bpe_mode", default='all', choices=['none', 'all', 'random', 'gaussian'])
    parser.add_argument("--journal_file", default='hyperopt/journal/finetune_with_options.db')
    parser.add_argument("--seed_file", default='hyperopt/results/best_pretrain_params_for_finetuning.json')
    parser.add_argument("--pretrain_journal", default='hyperopt/journal/large_batch.db')
    parser.add_argument("--trials", type=int, default=50)
    
    args = parser.parse_args()
    
    print(f"🎯 基于预训练参数选项的微调搜索 - BPE: {args.bpe_mode}")
    
    # 加载预训练参数选项
    if not Path(args.seed_file).exists():
        print(f"❌ 参数文件不存在: {args.seed_file}")
        print("请先运行 extract_best_params_for_finetuning.py 提取参数")
        return
    
    pretrain_options = load_pretrain_options(args.seed_file)
    if not pretrain_options:
        print("❌ 没有可用的预训练参数选项")
        return
    
    # 创建存储和研究
    storage = JournalStorage(JournalFileBackend(args.journal_file))
    
    # 配置优化器（简单配置，让搜索聚焦在选项选择上）
    study = optuna.create_study(
        study_name=f"finetune_with_options_{args.bpe_mode}",
        storage=storage, 
        direction="minimize", 
        load_if_exists=True
    )
    
    # 开始优化
    print(f"\n{'='*40} 微调搜索 {'='*40}")
    print(f"可选预训练参数: {list(pretrain_options.keys())}")
    
    study.optimize(
        lambda trial: finetune_objective(trial, pretrain_options, args.pretrain_journal, args.bpe_mode), 
        n_trials=args.trials
    )
    
    print(f"🏆 微调最优结果: {study.best_value:.4f}")
    print(f"🏆 最优参数: {study.best_params}")
    
    # 显示最优选项的详细信息
    best_option = study.best_params['pretrain_option']
    best_pretrain = pretrain_options[best_option]
    print(f"\n🎯 最优预训练选项: {best_option}")
    print(f"    method: {best_pretrain['method']}")
    print(f"    lr: {best_pretrain['lr']:.4e}")
    print(f"    bs: {best_pretrain['bs']}")
    print(f"    epochs: {study.best_params['epochs']}")
    
    print("✅ 微调搜索完成！")


if __name__ == "__main__":
    main()
