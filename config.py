
import os as _os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import os
import yaml
from datetime import datetime
def _as_bool(val: str | None, default: bool = True) -> bool:
    if val is None:
        return default
    v = val.strip().lower()
    return v in ("1", "true", "yes", "y", "on", "force")


_force = _as_bool(_os.environ.get("TG_THREADS_FORCE"), default=True)

if _force:
    # Set env vars first (affects oneDNN/BLAS/OMP/TBB/DGL etc.)
    print("Setting environment variables for single-threaded execution")
    _env_map = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TBB_NUM_THREADS": "1",
        "DGL_NUM_THREADS": "1",
    }
    for _k, _v in _env_map.items():
        _os.environ[_k] = _v

    # Then limit PyTorch thread count
    try:
        import importlib
        torch = importlib.import_module('torch')  # dynamic import; throws if unavailable
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print("Set PyTorch threads to 1")
        print("get_num_threads:", torch.get_num_threads())
        print("get_num_interop_threads:", torch.get_num_interop_threads())
    except Exception as e:
        # Silently skip if torch not available at early startup
        print("PyTorch not found, skipping: ", e)



"""
TokenizerGraph unified configuration management.

Single source of truth for all config parameters.
No fallbacks — missing or invalid config raises errors immediately.
"""
class ConfigNode:
    """Config node supporting dot-access and automatic type conversion."""
    def __init__(self, data: Dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, self._convert_value(value))
    
    def _convert_value(self, value):
        """Auto type conversion for string values."""
        if isinstance(value, str):
            # Try numeric conversion
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            elif value.lower() == 'null' or value.lower() == 'none':
                return None
            else:
                # Try number (including scientific notation)
                try:
                    # Try float first (handles scientific notation)
                    float_val = float(value)
                    # Return int if it's a whole number
                    if float_val.is_integer():
                        return int(float_val)
                    else:
                        return float_val
                except ValueError:
                    # Keep as string if conversion fails
                    return value
        return value
    
    def __getattr__(self, name):
        """Raise on missing attribute."""
        raise AttributeError(f"Config key '{name}' does not exist. Check your config path.")
    
    def to_dict(self) -> Dict:
        """Convert back to dict."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

@dataclass
class ProjectConfig:
    """
    Project-wide configuration.
    Loads from YAML, supports dot-access, CLI override, and JSON override.
    No fallbacks — errors are raised immediately on invalid config.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config from YAML file.
        
        Args:
            config_path: Path to YAML config file (default: config/default_config.yml)
        """
        # Load YAML config
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "default_config.yml"

        self._config_path = Path(config_path).resolve()

        with open(self._config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Create config nodes
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)
        
        # Process derived config values
        self._process_special_configs()
        
        # Post-init setup
        self.__post_init__()
        setup_global_seeds(self.system.seed)
        
        # Optuna hyperparameter search support
        self.optuna_trial = None  # stores optuna trial object for pruning
    
    def _process_special_configs(self):
        """
        Derive convenience aliases from the raw YAML config.
        Covers: device, paths, BERT architecture, special tokens, split ratios.
        """
        # Device config
        if self.system.device == 'auto':
            import importlib
            _t = importlib.import_module('torch')
            assert _t.cuda.is_available()
            self.device = 'cuda'

            # try:
            #     import importlib
            #     _t = importlib.import_module('torch')
            #     self.device = 'cuda' if _t.cuda.is_available() else 'cpu'
            # except Exception:
            #     self.device = 'cpu'
        else:
            self.device = self.system.device
        
        # Resolve all paths relative to project root (avoid CWD dependency)
        config_base_dir = self._config_path.parent if hasattr(self, '_config_path') else (Path(__file__).parent / "config")
        project_root_cfg = Path(self.paths.project_root)
        if project_root_cfg.is_absolute():
            project_root = project_root_cfg
        else:
            project_root = (config_base_dir / project_root_cfg).resolve()

        def _as_abs(p: str | Path) -> Path:
            p = Path(p)
            return (p if p.is_absolute() else (project_root / p)).resolve()

        self.data_dir = _as_abs(self.paths.data_dir)
        self.cache_dir = _as_abs(self.paths.cache_dir)
        self.processed_data_dir = _as_abs(self.paths.processed_dir)
        self.model_dir = _as_abs(self.paths.model_dir)
        self.log_dir = _as_abs(self.paths.log_dir)
        
        # BERT architecture aliases (standard names matching the paper)
        self.d_model = self.bert.architecture.hidden_size
        self.n_heads = self.bert.architecture.num_attention_heads
        self.n_layers = self.bert.architecture.num_hidden_layers
        self.d_ff = self.bert.architecture.intermediate_size
        self.vocab_size = self.bert.architecture.vocab_size
        
        # Special token config (fixed)
        self.pad_token = self.special_tokens.pad
        self.unk_token = self.special_tokens.unk
        self.mask_token = self.special_tokens.mask
        self.cls_token = self.special_tokens.cls
        self.sep_token = self.special_tokens.sep
        self.node_start_token = self.special_tokens.node_start
        self.node_end_token = self.special_tokens.node_end
        self.component_sep_token = self.special_tokens.component_sep
        
        # Token ID config (fixed)
        self.pad_token_id = self.special_tokens.ids.pad
        self.unk_token_id = self.special_tokens.ids.unk
        self.mask_token_id = self.special_tokens.ids.mask
        self.cls_token_id = self.special_tokens.ids.cls
        self.sep_token_id = self.special_tokens.ids.sep
        self.node_start_token_id = self.special_tokens.ids.node_start
        self.node_end_token_id = self.special_tokens.ids.node_end
        self.component_sep_token_id = self.special_tokens.ids.component_sep
        
        # Dataset split ratios
        self.train_split = self.dataset.splits.train
        self.val_split = self.dataset.splits.val
        self.test_split = self.dataset.splits.test
        
        # Experiment identifiers
        self.experiment_name = None   # user-specified experiment name (optional)
        self.experiment_group = None  # experiment group (optional, supports nesting)

        # Repeated runs config
        self.repeat_runs = 1  # number of repeated runs (default: 1, no repetition)

        # Timestamps fixed at config creation time
        self._run_simple_ts = datetime.now().strftime("%m%d_%H%M")    # for exp_name suffix
        self._run_full_ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # for experiment_id
    
    def __post_init__(self):
        """Post-init: ensure directories exist."""
        # Ensure required directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.cache_dir / "serialization").mkdir(exist_ok=True)
        (self.cache_dir / "bpe").mkdir(exist_ok=True)
        (self.cache_dir / "bert").mkdir(exist_ok=True)

        # Load repeat runs config
        if hasattr(self, 'repeat_runs'):
            self.repeat_runs = int(self.repeat_runs)
    
    @classmethod
    def from_args(cls, args) -> 'ProjectConfig':
        """Create config from parsed CLI args."""
        config = cls()
        
        # Override only non-None values
        for key, value in vars(args).items():
            if value is not None:
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    assert hasattr(obj, part), f"Invalid config path: '{key}'"
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        
        return config
    
    def validate(self):
        """Validate config consistency."""
        # Basic validation (read architecture fields directly to avoid alias staleness)
        d_model_now = int(self.bert.architecture.hidden_size)
        n_heads_now = int(self.bert.architecture.num_attention_heads)
        if d_model_now % n_heads_now != 0:
            raise AssertionError(
                f"d_model({d_model_now}) must be divisible by n_heads({n_heads_now})"
            )
        assert self.bert.architecture.max_seq_length > 0, "max_seq_length must be positive"
        assert 0 < self.bert.pretraining.mask_prob < 1, "mask_prob must be in (0,1)"
        
        # Split ratio validation
        total_split = (self.dataset.splits.train + 
                      self.dataset.splits.val + 
                      self.dataset.splits.test)
        assert abs(total_split - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        # Vocab validation
        if self.vocab_size is not None:
            assert self.vocab_size > 5, f"vocab_size({self.vocab_size}) must be > 5 (number of special tokens)"
        
        # Task type validation
        supported_tasks = ["mlm", "regression", "classification", "multi_label_classification", "multi_target_regression"]
        assert self.task.type in supported_tasks, \
            f"task_type must be one of {supported_tasks}, got '{self.task.type}'"
        
        # Dataset limit validation
        if self.dataset.limit is not None:
            assert 0 < self.dataset.limit <= 130831, \
                f"dataset_limit({self.dataset.limit}) must be in (0, 130831]"

        # BPE engine config validation
        bpe_cfg = getattr(self.serialization, 'bpe', None)
        if bpe_cfg is not None:
            engine_cfg = getattr(bpe_cfg, 'engine', None)
            if engine_cfg is not None:
                mode = str(engine_cfg.encode_rank_mode)
                assert mode in {"none", "all", "topk", "random", "gaussian"}, f"Invalid encode_rank_mode: {mode}"
                # topk mode requires non-negative k
                if mode == "topk" and getattr(engine_cfg, 'encode_rank_k', None) is not None:
                    assert int(engine_cfg.encode_rank_k) >= 0, "encode_rank_k must be non-negative"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def get_cache_key(self, split: str = None) -> str:
        """Generate cache key."""
        key_parts = [
            self.serialization.method,
            str(self.dataset.limit or "full"),
            str(self.serialization.bpe.num_merges),
            str(self.serialization.bpe.min_frequency)
        ]
        
        if split:
            key_parts.append(split)
            
        return "_".join(key_parts)
    
    def get_experiment_name(self, task: str = None, pipeline: str = None) -> str:
        """
        Generate experiment name.
        
        Args:
            task: Task name (e.g. 'pretrain', 'finetune')
            pipeline: Pipeline type ('bert'); auto-inferred if None
        """
        if self.experiment_name:
            return self.experiment_name
        
        # Auto-infer pipeline type
        if pipeline is None:
            pipeline = 'bert'
        
        if pipeline == 'bert':
            # BERT pipeline: dataset_serialization_method (BPE is a runtime transform, not in name)
            method_part = f"{self.dataset.name}_{self.serialization.method}"
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline}")
        
        if task:
            name_parts = [task, method_part]
        else:
            name_parts = [
                datetime.now().strftime("%m%d_%H%M"),
                method_part
            ]
        
        return "/".join(name_parts)

    # ================= Path helpers for new layout =================
    def _compute_method_dir(self) -> str:
        """Return method directory name (just the serialization method name)."""
        return self.serialization.method

    def build_suffix(self) -> str:
        """Build exp_name suffix from seed and timestamp (e.g. 'seed42-0808_1325')."""
        seed = self.system.seed
        seed_part = f"seed{seed}" if seed is not None else ""
        ts_part = self._run_simple_ts
        if seed_part and ts_part:
            return f"{seed_part}-{ts_part}"
        return seed_part or ts_part or ""

    def build_exp_name(self, user_name: Optional[str] = None) -> str:
        """Build exp_name. Uses user_name if given, otherwise falls back to suffix or 'exp'."""
        if user_name and len(str(user_name).strip()) > 0:
            return str(user_name)
        suffix = self.build_suffix()
        return suffix if suffix else "exp"

    def get_logs_dir(self,
                     group: Optional[str] = None,
                     exp_name: Optional[str] = None,
                     run_i: Optional[int] = None,
                     dataset: Optional[str] = None,
                     method: Optional[str] = None) -> Path:
        """Get standard log directory: log/<group>/<exp_name>/run_{i}.

        run_i=None → default run_1; run_i=-1 → aggregation dir (no run_ suffix).
        Raises ValueError if experiment_group is not set.
        """
        # Require group to be set, no silent fallback
        group = group if group is not None else self.experiment_group
        if group is None:
            raise ValueError("experiment_group is not set")
        if exp_name is None:
            # Derive exp_name from user-provided name or suffix
            exp_name = self.build_exp_name(self.experiment_name)

        if run_i is None:
          return self.log_dir / group / exp_name / f"run_1"  # default to first run
        elif run_i == -1:  # aggregation dir
            return self.log_dir / group / exp_name
        else:
          return self.log_dir / group / exp_name / f"run_{run_i}"

    def get_model_dir(self,
                      group: Optional[str] = None,
                      exp_name: Optional[str] = None,
                      run_i: Optional[int] = None,
                      dataset: Optional[str] = None,
                      method: Optional[str] = None) -> Path:
        """Get standard model directory: model/<group>/<exp_name>/run_{i}.

        run_i=None → default run_0; run_i=-1 → not supported (raises).
        Raises ValueError if experiment_group is not set.
        """
        group = group if group is not None else self.experiment_group
        if group is None:
            raise ValueError("experiment_group is not set")
        if exp_name is None:
            exp_name = self.build_exp_name(self.experiment_name)

        if run_i is None:
          return self.model_dir / group / exp_name / f"run_0"  # default to first run
        elif run_i == -1:  # aggregation dir not supported for models
          assert False, "Aggregation model directory is not supported"
        else:
          return self.model_dir / group / exp_name / f"run_{run_i}"

    # ================= BPE codebook path =================
    def get_bpe_model_path(self, dataset_name: str, method: str) -> Path:
        """Return BPE codebook save path: model/bpe/<dataset>/<method>/bpe_codebook.pkl."""
        # exp_name = self.build_exp_name(self.experiment_name)
        out_dir = self.model_dir / "bpe" / dataset_name / method
        return out_dir / "bpe_codebook.pkl"

    # ================= Experiment directory helpers =================

    def ensure_experiment_dirs(self, run_i: Optional[int] = None) -> tuple[Path, Path]:
        """Ensure experiment directories (logs & model) exist. Returns (logs_dir, model_dir)."""
        logs_dir = self.get_logs_dir(run_i=run_i)
        model_dir = self.get_model_dir(run_i=run_i)
        logs_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir, model_dir

    
    def _get_bpe_identifier(self) -> str:
        """Generate BPE config identifier string."""
        if self.serialization.bpe.num_merges <= 0:
            return "raw"
        
        engine = self.serialization.bpe.engine
        mode = engine.encode_rank_mode
        num_merges = self.serialization.bpe.num_merges
        
        if mode == "all":
            return f"bpe_all_{num_merges}"
        elif mode == "topk":
            k = engine.encode_rank_k or "auto"
            return f"bpe_topk{k}_{num_merges}"
        elif mode == "random":
            return f"bpe_random_{num_merges}"
        elif mode == "gaussian":
            return f"bpe_gauss_{num_merges}"
        else:
            return f"bpe_{mode}_{num_merges}"
    
    def get_bert_model_path(self, model_type: str = "pretrained") -> Path:
        """Get BERT model path (BPE is a runtime transform, doesn't affect model path)."""
        experiment_name = self.get_experiment_name(pipeline='bert')
        method = self.serialization.method
        
        if model_type == "pretrained":
            model_dir = self.model_dir / "pretrain_bert" / self.dataset.name / experiment_name / method
            return model_dir / "model.pkl"
        elif model_type == "finetuned":
            model_dir = self.model_dir / "finetune_bert" / self.dataset.name / experiment_name / method
            return model_dir / "model.pkl"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_bert_output_dir(self, task: str = None) -> Path:
        """Get BERT output directory."""
        experiment_name = self.get_experiment_name(task, pipeline='bert')
        # Legacy compatibility: use log_dir/bert_pretrain/
        return self.log_dir / "bert_pretrain" / experiment_name
    
    def get_output_dir(self, task: str = None, pipeline: str = None) -> Path:
        """Get output directory."""
        experiment_name = self.get_experiment_name(task, pipeline)
        # Legacy compatibility: fall back to log_dir/
        return self.log_dir / experiment_name
    

# ===========================================
# Global config utilities
# ===========================================

def create_default_config() -> ProjectConfig:
    """Create default config instance."""
    return ProjectConfig()

def create_config_from_args(args) -> ProjectConfig:
    """Create config from CLI args."""
    config = ProjectConfig.from_args(args)
    config.validate()
    return config

def setup_global_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch_local  # local import to avoid global dependency
        _torch_local.manual_seed(seed)
        if _torch_local.cuda.is_available():
            _torch_local.cuda.manual_seed_all(seed)
        if hasattr(_torch_local.backends, 'cudnn'):
            _torch_local.backends.cudnn.deterministic = True
            _torch_local.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)
