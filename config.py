
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
    # 先设置环境变量（影响 oneDNN/BLAS/OMP/TBB/DGL 等）
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

    # 再限制 PyTorch 线程数
    try:
        import importlib
        torch = importlib.import_module('torch')  # 动态导入，若不存在则抛异常
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print("Set PyTorch threads to 1")
        print("get_num_threads:", torch.get_num_threads())
        print("get_num_interop_threads:", torch.get_num_interop_threads())
    except Exception as e:
        # 启动早期不可用时静默跳过；后续首次导入 torch 后，由调用方自设更改
        print("PyTorch not found, skipping: ", e)



"""
TokenizerGraph 统一配置管理
============================

科研代码的单一配置源，避免配置冲突和fallback处理
所有配置参数都在此文件中定义，确保实验结果的可重现性
"""
class ConfigNode:
    """配置节点，支持点号访问和自动类型转换"""
    def __init__(self, data: Dict):
        for key, value in data.items():
            if isinstance(value, dict):
                # 递归创建子节点
                setattr(self, key, ConfigNode(value))
            else:
                # 自动类型转换
                setattr(self, key, self._convert_value(value))
    
    def _convert_value(self, value):
        """自动类型转换"""
        if isinstance(value, str):
            # 尝试转换为数值类型
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            elif value.lower() == 'null' or value.lower() == 'none':
                return None
            else:
                # 尝试转换为数字（包括科学计数法）
                try:
                    # 先尝试转换为浮点数（支持科学计数法）
                    float_val = float(value)
                    # 如果是整数，返回整数
                    if float_val.is_integer():
                        return int(float_val)
                    else:
                        return float_val
                except ValueError:
                    # 如果转换失败，保持字符串
                    return value
        return value
    
    def __getattr__(self, name):
        """处理属性不存在的情况"""
        raise AttributeError(f"配置项 '{name}' 不存在，请检查配置路径")
    
    def to_dict(self) -> Dict:
        """转换回字典"""
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
    项目统一配置类
    
    设计原则：
    1. 单一配置源，避免多套配置系统冲突
    2. 基于经过验证的参数（BERT论文、前人实验）
    3. 统一的命令行覆盖方式
    4. 确保科研实验的可重现性
    5. 不提供fallback，及时暴露配置问题
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: YAML配置文件路径（可选，默认使用default_config.yml）
        """
        # 加载YAML配置
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "default_config.yml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建配置节点
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)
        
        # 处理特殊配置
        self._process_special_configs()
        
        # 初始化后处理
        self.__post_init__()
        setup_global_seeds(self.system.seed)
        
        # 🆕 Optuna超参数搜索支持
        self.optuna_trial = None  # 存储optuna trial对象，用于剪枝支持
    
    def _process_special_configs(self):
        """
        处理特殊配置
        
        只保留最基础的、不会变动的别名：
        1. 特殊token相关
        2. BERT模型架构相关
        3. 数据集分割相关
        4. 系统配置相关
        """
        # 处理设备配置
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
        
        # 基础路径配置（统一解析为以项目根为基准的绝对路径，避免依赖当前工作目录）
        project_root = Path(self.paths.project_root)

        def _as_abs(p: str | Path) -> Path:
            p = Path(p)
            return (p if p.is_absolute() else (project_root / p)).resolve()

        self.data_dir = _as_abs(self.paths.data_dir)
        self.cache_dir = _as_abs(self.paths.cache_dir)
        self.processed_data_dir = _as_abs(self.paths.processed_dir)
        self.model_dir = _as_abs(self.paths.model_dir)
        self.log_dir = _as_abs(self.paths.log_dir)
        
        # BERT模型架构配置（这些是与论文对应的标准名称）
        self.d_model = self.bert.architecture.hidden_size
        self.n_heads = self.bert.architecture.num_attention_heads
        self.n_layers = self.bert.architecture.num_hidden_layers
        self.d_ff = self.bert.architecture.intermediate_size
        self.vocab_size = self.bert.architecture.vocab_size
        
        # 特殊Token配置（这些是固定的，不会变的）
        self.pad_token = self.special_tokens.pad
        self.unk_token = self.special_tokens.unk
        self.mask_token = self.special_tokens.mask
        self.cls_token = self.special_tokens.cls
        self.sep_token = self.special_tokens.sep
        self.node_start_token = self.special_tokens.node_start
        self.node_end_token = self.special_tokens.node_end
        self.component_sep_token = self.special_tokens.component_sep
        
        # Token ID配置（这些也是固定的）
        self.pad_token_id = self.special_tokens.ids.pad
        self.unk_token_id = self.special_tokens.ids.unk
        self.mask_token_id = self.special_tokens.ids.mask
        self.cls_token_id = self.special_tokens.ids.cls
        self.sep_token_id = self.special_tokens.ids.sep
        self.node_start_token_id = self.special_tokens.ids.node_start
        self.node_end_token_id = self.special_tokens.ids.node_end
        self.component_sep_token_id = self.special_tokens.ids.component_sep
        
        # 数据集分割配置（这些是标准的分割比例）
        self.train_split = self.dataset.splits.train
        self.val_split = self.dataset.splits.val
        self.test_split = self.dataset.splits.test
        
        # 实验标识（基础功能）
        self.experiment_name = None  # 用户可指定的实验名（可为空）
        self.experiment_group = None  # 实验分组（可为空，支持多级）

        # 🆕 重复运行配置
        self.repeat_runs = 1  # 重复运行次数，默认1次（不重复）
        self.current_run_i = None  # 当前运行编号（运行时设置）

        # 运行时间戳（在配置创建时固定下来，全局统一使用）
        self._run_simple_ts = datetime.now().strftime("%m%d_%H%M")  # 用于exp_name后缀（月日_时分）
        self._run_full_ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # 用于experiment_id（年月至秒）
    
    def __post_init__(self):
        """初始化后的自动配置"""
        # 确保必要目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.cache_dir / "serialization").mkdir(exist_ok=True)
        (self.cache_dir / "bpe").mkdir(exist_ok=True)
        (self.cache_dir / "bert").mkdir(exist_ok=True)

        # 🆕 加载重复运行配置
        if hasattr(self, 'repeat_runs'):
            self.repeat_runs = int(self.repeat_runs)
    
    @classmethod
    def from_args(cls, args) -> 'ProjectConfig':
        """从命令行参数创建配置"""
        config = cls()
        
        # 遍历所有命令行参数，只覆盖非None的值
        for key, value in vars(args).items():
            if value is not None:
                # 处理嵌套属性
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    assert hasattr(obj, part), f"配置路径错误: '{key}'"
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        
        return config
    
    def validate(self):
        """配置验证，确保参数合理性"""
        # 基本验证（直接读取当前架构字段，避免与缓存别名不同步）
        d_model_now = int(self.bert.architecture.hidden_size)
        n_heads_now = int(self.bert.architecture.num_attention_heads)
        if d_model_now % n_heads_now != 0:
            raise AssertionError(
                f"d_model({d_model_now})必须能被n_heads({n_heads_now})整除"
            )
        assert self.bert.architecture.max_seq_length > 0, "max_seq_length必须为正数"
        assert 0 < self.bert.pretraining.mask_prob < 1, "mask_prob必须在(0,1)区间内"
        
        # 数据分割验证
        total_split = (self.dataset.splits.train + 
                      self.dataset.splits.val + 
                      self.dataset.splits.test)
        assert abs(total_split - 1.0) < 1e-6, "数据分割比例之和必须等于1.0"
        
        # 词汇表验证
        if self.vocab_size is not None:
            assert self.vocab_size > 5, f"vocab_size({self.vocab_size})必须大于特殊token数量(5)"
        
        # 任务类型验证 - 🆕 添加MLM支持
        supported_tasks = ["mlm", "regression", "classification", "multi_label_classification", "multi_target_regression"]
        assert self.task.type in supported_tasks, \
            f"task_type必须是{supported_tasks}之一，当前为'{self.task.type}'"
        
        # QM9限制验证
        if self.dataset.limit is not None:
            assert 0 < self.dataset.limit <= 130831, \
                f"dataset_limit({self.dataset.limit})必须在(0, 130831]范围内"

        # BPEEngine 配置验证（无隐式回退，仅检查已声明字段的基本一致性）
        bpe_cfg = getattr(self.serialization, 'bpe', None)
        if bpe_cfg is not None:
            engine_cfg = getattr(bpe_cfg, 'engine', None)
            if engine_cfg is not None:
                mode = str(engine_cfg.encode_rank_mode)
                assert mode in {"none", "all", "topk", "random", "gaussian"}, f"无效的 encode_rank_mode: {mode}"
                # 当 topk 模式时，若提供了 k 则需为非负
                if mode == "topk" and getattr(engine_cfg, 'encode_rank_k', None) is not None:
                    assert int(engine_cfg.encode_rank_k) >= 0, "encode_rank_k 必须为非负整数"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
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
        """生成缓存键"""
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
        生成实验名称
        
        Args:
            task: 任务名称（如'pretrain', 'finetune'等）
            pipeline: 管道类型（'bert'），如果为None则自动推断
        """
        if self.experiment_name:
            return self.experiment_name
        
        # 自动推断pipeline类型
        if pipeline is None:
            pipeline = 'bert'
        
        if pipeline == 'bert':
            # BERT pipeline: 数据集_序列好化方法（BPE作为运行时transform，不影响实验名称）
            method_part = f"{self.dataset.name}_{self.serialization.method}"
        else:
            raise ValueError(f"不支持的pipeline类型: {pipeline}")
        
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
        """构建方法目录名。

        说明：
          - 目录名格式为序列化方法名（BPE作为运行时transform，不影响目录结构）
          - 该值作为 logs/model 下的目录层级之一

        示例：
          - "feuler"  # 使用feuler序列化方法
          - "eulerian"  # 使用eulerian序列化方法

        返回：
          str: 方法目录名。
        """
        return self.serialization.method

    def build_suffix(self) -> str:
        """构建 exp_name 的后缀。

        说明：
          - 不包含 dataset、method、BPE/RAW 或训练超参数
          - 包含随机种子信息与简单时间戳（例如："seed42-0808_1325"）

        返回：
          str: 后缀字符串（如 "seed42"）。若未设置种子则返回空字符串。
        """
        seed = self.system.seed
        seed_part = f"seed{seed}" if seed is not None else ""
        ts_part = self._run_simple_ts
        if seed_part and ts_part:
            return f"{seed_part}-{ts_part}"
        return seed_part or ts_part or ""

    def build_exp_name(self, user_name: Optional[str] = None) -> str:
        """根据用户名称与后缀构建 exp_name。

        说明：
          - 当提供 user_name 时，exp_name = "{user_name}"
          - 当未提供 user_name 时，exp_name = "{suffix}"
          - 当二者均空时，回退为 "exp"
          - 后缀包含 seed 与简单时间戳（不含 dataset/method/BPE/训练超参）

        参数：
          user_name (Optional[str]): 用户指定的实验名称，可为 None。

        返回：
          str: 组合后的 exp_name。
        """
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
        """获取标准日志目录。

        目录层级："log/<group>/<exp_name>/run_{i}" 或兼容旧格式。

        参数：
          group (Optional[str]): 实验分组；为 None 时使用 self.experiment_group。必须存在，否则抛出异常。
          exp_name (Optional[str]): 显式指定的 exp_name；为 None 时根据用户名称与后缀自动生成。
          run_i (Optional[int]): 重复运行编号，为 None 时使用传统路径格式。
          dataset (Optional[str]): 数据集层名称；默认使用 self.dataset.name（仅兼容旧格式）。
          method (Optional[str]): 方法层名称；默认使用 _compute_method_dir()（仅兼容旧格式）。

        返回：
          Path: 日志目录路径。

        异常：
          ValueError: 当实验分组未设置时抛出。

        示例：
          - 重复运行：group=g1, exp_name=testX, run_i=1 → log/g1/testX/run_1
          - 聚合日志目录：group=g1, exp_name=testX, run_i=-1 → log/g1/testX
          - 兼容旧格式：group=g1, exp_name=testX, run_i=None → log/g1/testX/qm9/feuler-BPE
        """
        # 严格要求必要字段存在，不做静默回退
        group = group if group is not None else self.experiment_group
        if group is None:
            raise ValueError("experiment_group 未设置")
        if exp_name is None:
            # 始终根据用户提供的名称派生 exp_name（user_name-suffix 或纯 suffix）
            exp_name = self.build_exp_name(self.experiment_name)

        # 🆕 支持重复运行的简化路径结构
        if run_i is not None:
            if run_i == -1: #聚合日志目录
                return self.log_dir / group / exp_name
            return self.log_dir / group / exp_name / f"run_{run_i}"
        else:
            # 🔄 兼容旧格式（带dataset/method层级）
            dataset = dataset if dataset is not None else self.dataset.name
            method = method if method is not None else self._compute_method_dir()
            return self.log_dir / group / exp_name / dataset / method

    def get_model_dir(self,
                      group: Optional[str] = None,
                      exp_name: Optional[str] = None,
                      run_i: Optional[int] = None,
                      dataset: Optional[str] = None,
                      method: Optional[str] = None) -> Path:
        """获取标准模型目录。

        目录层级："model/<group>/<exp_name>/run_{i}" 或兼容旧格式。

        参数：
          group (Optional[str]): 实验分组；为 None 时使用 self.experiment_group。必须存在，否则抛出异常。
          exp_name (Optional[str]): 显式指定的 exp_name；为 None 时根据用户名称与后缀自动生成。
          run_i (Optional[int]): 重复运行编号，为 None 时使用传统路径格式。
          dataset (Optional[str]): 数据集层名称；默认使用 self.dataset.name（仅兼容旧格式）。
          method (Optional[str]): 方法层名称；默认使用 _compute_method_dir()（仅兼容旧格式）。

        返回：
          Path: 模型目录路径。

        异常：
          ValueError: 当实验分组未设置时抛出。

        示例：
          - 重复运行：group=g1, exp_name=testX, run_i=1 → model/g1/testX/run_1
          - 兼容旧格式：group=g1, exp_name=testX, run_i=None → model/g1/testX/qm9/feuler-BPE
        """
        group = group if group is not None else self.experiment_group
        if group is None:
            raise ValueError("experiment_group 未设置")
        if exp_name is None:
            exp_name = self.build_exp_name(self.experiment_name)

        # 🆕 支持重复运行的简化路径结构
        if run_i is not None:
            return self.model_dir / group / exp_name / f"run_{run_i}"
        else:
            # 🔄 兼容旧格式（带dataset/method层级）
            dataset = dataset if dataset is not None else self.dataset.name
            method = method if method is not None else self._compute_method_dir()
            return self.model_dir / group / exp_name / dataset / method

    # ================= BPE 码本保存路径 =================
    def get_bpe_model_path(self, dataset_name: str, method: str) -> Path:
        """返回 BPE 码本（codebook）保存路径。

        目录层级：model/bpe/<dataset>/<exp_name>/<method>/bpe_codebook.pkl
        不创建目录，仅返回 Path；调用方负责创建目录。
        """
        # exp_name = self.build_exp_name(self.experiment_name)
        out_dir = self.model_dir / "bpe" / dataset_name / method
        return out_dir / "bpe_codebook.pkl"

    # ================= 实验ID与快照/目录辅助 =================

    def ensure_experiment_dirs(self, run_i: Optional[int] = None) -> tuple[Path, Path]:
        """确保实验目录存在（logs与model）。

        参数：
          run_i (Optional[int]): 重复运行编号，为 None 时使用传统路径格式。

        返回：
          (Path, Path): (logs_dir, model_dir)
        """
        logs_dir = self.get_logs_dir(run_i=run_i)
        model_dir = self.get_model_dir(run_i=run_i)
        logs_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir, model_dir

    def compose_wandb_metadata(self) -> Dict[str, Any]:
        """组合WandB元数据（不直接导入/依赖wandb）。

        返回：
          Dict[str, Any]: 包含 project(需外部指定)、group、name、tags 的字典。
            - group: "{group}/{exp_name}/run_{i}" 或兼容旧格式
            - name : experiment_id（当前时间戳）
            - tags : [dataset, method, bpe_config, seed, run_i]（包含BPE配置信息和run_i）
        """
        group = self.experiment_group or "default"  # 提供默认组名
        exp_name = self.build_exp_name(self.experiment_name)
        dataset = self.dataset.name
        method = self._compute_method_dir()
        seed = self.system.seed
        run_i = getattr(self, 'current_run_i', None)
        # 可选：由调用方在运行期设置实验阶段（如 "pretrain"、"finetune"）
        phase = getattr(self, "experiment_phase", None)

        # 添加BPE配置信息
        bpe_tag = self._get_bpe_identifier()

        # 🆕 支持重复运行的简化路径结构
        if run_i is not None:
            # 简化路径：group/exp_name/run_i
            detailed_name = f"{group}/{exp_name}/run_{run_i}_{bpe_tag}"
            if phase is not None:
                detailed_name = f"{detailed_name}__{phase}"

            group_path = f"{group}/{exp_name}/run_{run_i}_{bpe_tag}"
            if phase is not None:
                group_path = f"{group_path}/{phase}"

            tags = [bpe_tag, f"seed{seed}", f"run{run_i}"]
            if phase is not None:
                tags.append(str(phase))
        else:
            # 🔄 兼容旧格式
            detailed_name = f"{group}/{exp_name}__{dataset}/{method}_{bpe_tag}"
            if phase is not None:
                detailed_name = f"{detailed_name}__{phase}"

            group_path = f"{group}/{exp_name}/{dataset}/{method}_{bpe_tag}"
            if phase is not None:
                group_path = f"{group_path}/{phase}"

            tags = [dataset, method, bpe_tag, f"seed{seed}"]
            if phase is not None:
                tags.append(str(phase))

        return {
            "group": group_path,
            "name": detailed_name,
            "tags": tags,
        }
    
    def _get_bpe_identifier(self) -> str:
        """生成BPE配置标识符"""
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
        """获取BERT模型路径（BPE作为运行时transform，不影响模型路径）"""
        experiment_name = self.get_experiment_name(pipeline='bert')
        method = self.serialization.method
        
        if model_type == "pretrained":
            model_dir = self.model_dir / "pretrain_bert" / self.dataset.name / experiment_name / method
            return model_dir / "model.pkl"
        elif model_type == "finetuned":
            model_dir = self.model_dir / "finetune_bert" / self.dataset.name / experiment_name / method
            return model_dir / "model.pkl"
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def get_bert_output_dir(self, task: str = None) -> Path:
        """获取BERT输出目录"""
        experiment_name = self.get_experiment_name(task, pipeline='bert')
        # 兼容旧接口：改为使用 log_dir 下的 bert_pretrain 目录
        return self.log_dir / "bert_pretrain" / experiment_name
    
    def get_output_dir(self, task: str = None, pipeline: str = None) -> Path:
        """获取输出目录"""
        experiment_name = self.get_experiment_name(task, pipeline)
        # 兼容旧接口：统一回到 log_dir 下
        return self.log_dir / experiment_name
    

# ===========================================
# 全局配置实例和工具函数
# ===========================================

def create_default_config() -> ProjectConfig:
    """创建默认配置实例"""
    return ProjectConfig()

def create_config_from_args(args) -> ProjectConfig:
    """从命令行参数创建配置"""
    config = ProjectConfig.from_args(args)
    config.validate()
    return config

def setup_global_seeds(seed: int):
    """设置所有随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch_local  # 局部导入以避免全局依赖
        _torch_local.manual_seed(seed)
        if _torch_local.cuda.is_available():
            _torch_local.cuda.manual_seed_all(seed)
        if hasattr(_torch_local.backends, 'cudnn'):
            _torch_local.backends.cudnn.deterministic = True
            _torch_local.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)
