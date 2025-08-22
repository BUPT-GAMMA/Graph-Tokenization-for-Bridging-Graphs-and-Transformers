import argparse
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface


def compute_stats(lengths: List[int]) -> Dict[str, Any]:
    if len(lengths) == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0,
            'max': 0,
            'u1': 0,
            'u2': 0,
            'u3': 0,
            'cov1': 0.0,
            'cov2': 0.0,
            'cov3': 0.0,
        }
    arr = np.asarray(lengths, dtype=np.int64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    mn = int(arr.min())
    mx = int(arr.max())
    u1 = int(math.ceil(mean + 1.0 * std))
    u2 = int(math.ceil(mean + 2.0 * std))
    u3 = int(math.ceil(mean + 3.0 * std))
    # 覆盖率（只看上界 <= u±kσ），下界对长度无意义
    cov1 = float((arr <= u1).mean())
    cov2 = float((arr <= u2).mean())
    cov3 = float((arr <= u3).mean())
    return {
        'count': int(arr.size),
        'mean': mean,
        'std': std,
        'min': mn,
        'max': mx,
        'u1': u1,
        'u2': u2,
        'u3': u3,
        'cov1': cov1,
        'cov2': cov2,
        'cov3': cov3,
    }


def choose_recommended_max(stats: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
    """
    返回：(recommended_max_len, policy, extra)
    policy in {"u+3sigma", "max"}
    extra: 额外指标（如截断比例）
    """
    mx = int(stats['max'])
    u3 = int(stats['u3'])
    # 若 max 明显大于 u+3σ，则报告使用 u+3σ 的截断比例
    # 经验阈值：若 >u3 的样本比例 < 0.1%，可用 u+3σ；否则取 max（更安全）
    # 覆盖率 cov3 即 <=u3 的比例
    cov3 = float(stats['cov3'])
    tail_ratio = max(0.0, 1.0 - cov3)
    if u3 >= mx:
        return u3, "u+3sigma", {"tail_ratio": 0.0}
    if tail_ratio <= 0.001:
        return u3, "u+3sigma", {"tail_ratio": tail_ratio}
    return mx, "max", {"tail_ratio": tail_ratio}


def load_lengths_from_udi(dataset_name: str, method: str,
                          ms_enabled: bool, ms_k: int | None,
                          by_split: bool = False) -> Dict[str, List[int]] | List[int]:
    """
    从 UDI 已落盘的序列结果中读取序列长度。
    返回：
      - by_split=False: List[int]
      - by_split=True:  {"train": [...], "val": [...], "test": [...], "all": [...]}
    """
    config = ProjectConfig()
    # 强制使用 single，避免误指向 multi_100 目录（除非用户显式指定）
    try:
        config.serialization.multiple_sampling.enabled = bool(ms_enabled)
        if ms_enabled and ms_k is not None:
            config.serialization.multiple_sampling.num_realizations = int(ms_k)
    except Exception:
        # 忽略失败，按默认
        pass

    udi = UnifiedDataInterface(config=config, dataset=dataset_name)

    def _to_lengths(seq_list: List[Tuple[int, List[int]]]) -> List[int]:
        arr = seq_list if isinstance(seq_list, list) else []
        return [len(seq) for _, seq in arr]

    if not by_split:
        sequences_with_id, _ = udi.get_sequences(method)
        return _to_lengths(sequences_with_id)
    else:
        tr, trp, va, vap, te, tep = udi.get_sequences_by_splits(method)
        out = {
            "train": _to_lengths(tr),
            "val": _to_lengths(va),
            "test": _to_lengths(te),
        }
        # all 合并
        all_list = tr + va + te
        out["all"] = _to_lengths(all_list)
        return out


def main():
    ap = argparse.ArgumentParser(description="统计各数据集/方法的序列长度分布及1-3σ覆盖率")
    ap.add_argument("--datasets", type=str, default="aqsol",
                    help="以逗号分隔，如: aqsol,qm9test,qm9,zinc")
    ap.add_argument("--methods", type=str,
                    default="feuler,eulerian,cpp,fcpp,topo,dfs,bfs,smiles",
                    help="以逗号分隔的方法列表")
    ap.add_argument("--ms_enabled", action="store_true",
                    help="选择 UDI 的 multi_<k> 缓存键（默认 single）")
    ap.add_argument("--ms_k", type=int, default=None,
                    help="当 --ms_enabled 打开时，指定 realizations 数（例如 100），用于选择 multi_k 目录")
    ap.add_argument("--by_split", action="store_true",
                    help="按 train/val/test 分别统计并汇总")
    args = ap.parse_args()

    datasets = [s.strip() for s in args.datasets.split(',') if s.strip()]
    methods = [s.strip() for s in args.methods.split(',') if s.strip()]
    print(f"Datasets: {datasets}")
    print(f"Methods : {methods}")
    print(f"UDI cache key: {'multi_'+str(args.ms_k) if args.ms_enabled else 'single'}")
    print()

    # 数据集级建议：取各方法推荐值的最大值作为该数据集的建议上限
    for ds in datasets:
        print(f"=== Dataset: {ds} ===")
        ds_recommended_values: List[int] = []
        ds_policy_votes: Dict[str, int] = {"u+3sigma": 0, "max": 0}

        for m in methods:
            try:
                if args.by_split:
                    split_lengths = load_lengths_from_udi(ds, m,
                                                          ms_enabled=args.ms_enabled, ms_k=args.ms_k,
                                                          by_split=True)
                    lengths = split_lengths["all"]
                else:
                    lengths = load_lengths_from_udi(ds, m,
                                                    ms_enabled=args.ms_enabled, ms_k=args.ms_k,
                                                    by_split=False)
            except Exception as e:
                print(f"  [Warn] {ds}/{m} 序列化失败: {e}")
                continue

            stats = compute_stats(lengths)
            rec_val, policy, extra = choose_recommended_max(stats)
            ds_recommended_values.append(int(rec_val))
            ds_policy_votes[policy] = ds_policy_votes.get(policy, 0) + 1

            cov1 = stats['cov1']
            cov2 = stats['cov2']
            cov3 = stats['cov3']

            if args.by_split:
                # 简要展示各 split 的 u+3σ 覆盖情况
                tr_stats = compute_stats(split_lengths["train"]) if split_lengths.get("train") is not None else None
                va_stats = compute_stats(split_lengths["val"]) if split_lengths.get("val") is not None else None
                te_stats = compute_stats(split_lengths["test"]) if split_lengths.get("test") is not None else None
                split_line = (
                    f"    [train] n={tr_stats['count'] if tr_stats else 0}, cov@u+3σ={tr_stats['cov3'] if tr_stats else 0:.4f}; "
                    f"[val] n={va_stats['count'] if va_stats else 0}, cov@u+3σ={va_stats['cov3'] if va_stats else 0:.4f}; "
                    f"[test] n={te_stats['count'] if te_stats else 0}, cov@u+3σ={te_stats['cov3'] if te_stats else 0:.4f}"
                )
            else:
                split_line = None

            msg = (
                "  - Method: {m}\n"
                "    count={count}, min={min}, max={max}\n"
                "    mean={mean:.2f}, std={std:.2f}\n"
                "    u+1σ={u1} (cov={cov1:.4f}), u+2σ={u2} (cov={cov2:.4f}), u+3σ={u3} (cov={cov3:.4f})\n"
                "    recommended={rec} ({policy}, tail>{u3}={tail:.4%})" + ("\n" + split_line if split_line else "")
            ).format(
                m=m,
                count=stats['count'],
                min=stats['min'],
                max=stats['max'],
                mean=stats['mean'],
                std=stats['std'],
                u1=stats['u1'], u2=stats['u2'], u3=stats['u3'],
                cov1=cov1, cov2=cov2, cov3=cov3,
                rec=int(rec_val), policy=policy, tail=float(extra.get('tail_ratio', 0.0))
            )
            print(msg)

        if ds_recommended_values:
            ds_overall = int(max(ds_recommended_values))
            # 简单多数投票决定倾向策略
            policy_choice = max(ds_policy_votes.items(), key=lambda kv: kv[1])[0]
            print(
                f"  -> Dataset-level suggestion: max_seq_length={ds_overall} (policy majority: {policy_choice})"
            )
        print()


if __name__ == "__main__":
    main()


