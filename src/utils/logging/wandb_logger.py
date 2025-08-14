from __future__ import annotations

from typing import Dict, Any, Optional
import os


class WandbLogger:
    """
    WandB 封装：支持离线模式和自动项目名称生成
    - 环境变量：WANDB_PROJECT（可选），WANDB_ENTITY（可选）
    - 组名/名称/标签由 config.compose_wandb_metadata() 提供
    - 支持离线模式：当无法连接WandB时自动切换到离线模式
    """

    def __init__(self, project: Optional[str] = None, entity: Optional[str] = None, offline=True):
        import wandb  # noqa: F401 (延迟导入)
        self._wandb = wandb
        self._run = None
        
        # 项目名称：优先级 参数 > 环境变量 > 默认值
        self._project = project or os.environ.get("WANDB_PROJECT") or "tokenizerGraph"
        self._entity = entity or os.environ.get("WANDB_ENTITY")
        
        # 离线模式：优先级 参数 > 环境变量 > 自动检测
        if offline is not None:
            self._offline = offline
        elif os.environ.get("WANDB_MODE") == "offline":
            self._offline = True
        else:
            self._offline = False  # 默认尝试在线模式，失败时自动切换到离线

    def init(self, config) -> None:
        try:
            meta = config.compose_wandb_metadata()
            
            # 构建详细的配置字典，重点突出BPE参数
            wandb_config = config.to_dict()
            
            # 添加BPE详细信息到顶级（便于在W&B界面查看）
            if config.serialization.bpe.num_merges > 0:
                bpe_engine = config.serialization.bpe.engine
                wandb_config["bpe_enabled"] = True
                wandb_config["bpe_num_merges"] = config.serialization.bpe.num_merges
                wandb_config["bpe_encode_backend"] = bpe_engine.encode_backend
                wandb_config["bpe_encode_rank_mode"] = bpe_engine.encode_rank_mode
                wandb_config["bpe_encode_rank_k"] = bpe_engine.encode_rank_k
                wandb_config["bpe_encode_rank_min"] = bpe_engine.encode_rank_min
                wandb_config["bpe_encode_rank_max"] = bpe_engine.encode_rank_max
                wandb_config["bpe_encode_rank_dist"] = bpe_engine.encode_rank_dist
            else:
                wandb_config["bpe_enabled"] = False
            
            # 构建初始化参数
            init_kwargs = {
                "project": self._project,
                "entity": self._entity,
                "group": meta.get("group"),
                "name": meta.get("name"),
                "tags": meta.get("tags"),
                "config": wandb_config,
                "settings": self._wandb.Settings(start_method="fork")  # 避免多进程问题
            }
            
            if self._offline:
                # 显式离线模式
                init_kwargs["mode"] = "offline"
                self._run = self._wandb.init(**init_kwargs)
                print(f"📴 W&B离线模式初始化成功: 项目={self._project}")
            else:
                # 尝试在线模式，失败时自动切换到离线模式
                try:
                    self._run = self._wandb.init(**init_kwargs)
                    print(f"🌐 W&B在线模式初始化成功: 项目={self._project}")
                except Exception as online_error:
                    print(f"⚠️ W&B在线模式失败，切换到离线模式: {online_error}")
                    try:
                        init_kwargs["mode"] = "offline"
                        self._run = self._wandb.init(**init_kwargs)
                        self._offline = True
                        print(f"📴 W&B离线模式初始化成功: 项目={self._project}")
                    except Exception as offline_error:
                        raise RuntimeError(f"W&B初始化失败（在线和离线模式均失败）: 在线错误={online_error}, 离线错误={offline_error}")

            # 统一定义不同系列的横轴：
            # - train/*        使用 global_step（批级）
            # - train_epoch/*  使用 epoch      （epoch级）
            # - val/* 与 test/* 使用 epoch
            try:
                self._wandb.define_metric("train/*", step_metric="global_step")
                self._wandb.define_metric("train_epoch/*", step_metric="epoch")
                self._wandb.define_metric("val/*", step_metric="epoch")
                self._wandb.define_metric("test/*", step_metric="epoch")
            except Exception:
                pass
                        
        except Exception as meta_error:
            # 如果元数据构建失败，使用更简单的配置尝试初始化
            print(f"⚠️ W&B元数据构建失败，使用简化配置: {meta_error}")
            try:
                simple_kwargs = {
                    "project": self._project,
                    "entity": self._entity,
                    "mode": "offline",  # 元数据失败时直接使用离线模式
                    "settings": self._wandb.Settings(start_method="fork")
                }
                self._run = self._wandb.init(**simple_kwargs)
                self._offline = True
                print(f"📴 W&B简化离线模式初始化成功: 项目={self._project}")
            except Exception as simple_error:
                raise RuntimeError(f"W&B完全初始化失败: 元数据错误={meta_error}, 简化模式错误={simple_error}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is None:
            raise RuntimeError("WandB 尚未初始化")
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None





