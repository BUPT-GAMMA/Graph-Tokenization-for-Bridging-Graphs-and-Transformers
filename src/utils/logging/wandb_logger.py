from __future__ import annotations

from typing import Dict, Any, Optional
import os


class WandbLogger:
    """
    WandB wrapper with offline mode and automatic project naming.
    - Env vars: WANDB_PROJECT (optional), WANDB_ENTITY (optional)
    - Group/name/tags provided by config.compose_wandb_metadata()
    - Auto-fallback to offline mode when WandB server is unreachable
    """

    def __init__(self, project: Optional[str] = None, entity: Optional[str] = None, offline=True):
        import wandb  # noqa: F401 (lazy import)
        self._wandb = wandb
        self._run = None
        
        # Project name priority: arg > env var > default
        self._project = project or os.environ.get("WANDB_PROJECT") or "tokenizerGraph"
        self._entity = entity or os.environ.get("WANDB_ENTITY")
        
        # Offline mode priority: arg > env var > auto-detect
        if offline is not None:
            self._offline = offline
        elif os.environ.get("WANDB_MODE") == "offline":
            self._offline = True
        else:
            self._offline = False  # Default: try online, fallback to offline

    def init(self, config) -> None:
        try:
            # meta = config.compose_wandb_metadata()
            
            # Build detailed config dict, highlighting BPE params
            wandb_config = config.to_dict()
            
            # Promote BPE details to top level for W&B dashboard
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
            
            # Build init kwargs
            init_kwargs = {
                "project": self._project,
                "entity": self._entity,
                # "group": meta.get("group"),
                # "name": meta.get("name"),
                # "tags": meta.get("tags"),
                "config": wandb_config,
                "settings": self._wandb.Settings(start_method="fork")  # Avoid multiprocess issues
            }
            
            if self._offline:
                # Explicit offline mode
                init_kwargs["mode"] = "offline"
                self._run = self._wandb.init(**init_kwargs)
                print(f"W&B offline mode initialized: project={self._project}")
            else:
                # Try online, auto-fallback to offline
                try:
                    self._run = self._wandb.init(**init_kwargs)
                    print(f"W&B online mode initialized: project={self._project}")
                except Exception as online_error:
                    print(f"W&B online mode failed, switching to offline: {online_error}")
                    try:
                        init_kwargs["mode"] = "offline"
                        self._run = self._wandb.init(**init_kwargs)
                        self._offline = True
                        print(f"W&B offline mode initialized: project={self._project}")
                    except Exception as offline_error:
                        raise RuntimeError(f"W&B init failed (both online and offline): online={online_error}, offline={offline_error}")

            # Define x-axis for each metric series:
            # - train/*        -> global_step (batch-level)
            # - train_epoch/*  -> epoch       (epoch-level)
            # - val/* & test/* -> epoch
            try:
                self._wandb.define_metric("train/*", step_metric="global_step")
                self._wandb.define_metric("train_epoch/*", step_metric="epoch")
                self._wandb.define_metric("val/*", step_metric="epoch")
                self._wandb.define_metric("test/*", step_metric="epoch")
            except Exception:
                pass
                        
        except Exception as meta_error:
            # Metadata build failed; try simplified init
            print(f"W&B metadata build failed, using simplified config: {meta_error}")
            try:
                simple_kwargs = {
                    "project": self._project,
                    "entity": self._entity,
                    "mode": "offline",  # Fall back to offline when metadata fails
                    "settings": self._wandb.Settings(start_method="fork")
                }
                self._run = self._wandb.init(**simple_kwargs)
                self._offline = True
                print(f"W&B simplified offline mode initialized: project={self._project}")
            except Exception as simple_error:
                raise RuntimeError(f"W&B init completely failed: metadata_error={meta_error}, simplified_error={simple_error}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is None:
            raise RuntimeError("WandB not initialized")
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None





