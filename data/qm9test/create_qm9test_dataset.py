#!/usr/bin/env python3
"""Create QM9Test by replaying a subset from the current QM9 baseline."""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_replay_spec(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    indices_path = Path(path)
    if not indices_path.exists():
        raise FileNotFoundError(f"original_indices_path not found: {indices_path}")
    payload = json.loads(indices_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"original_indices": [int(x) for x in payload]}
    if not isinstance(payload, dict):
        raise ValueError("original_indices_path must contain a JSON list or object")
    selected = payload.get("original_indices", [])
    if not isinstance(selected, list):
        raise ValueError("original_indices must be a JSON list")
    replay_spec = dict(payload)
    replay_spec["original_indices"] = [int(x) for x in selected]
    return replay_spec


def _load_qm9_source(source_dir: Path) -> Dict[str, Any]:
    with (source_dir / "data.pkl").open("rb") as f:
        data = pickle.load(f)
    smiles = {
        "smiles_1": (source_dir / "smiles_1_direct.txt").read_text(encoding="utf-8").splitlines(),
        "smiles_2": (source_dir / "smiles_2_explicit_h.txt").read_text(encoding="utf-8").splitlines(),
        "smiles_3": (source_dir / "smiles_3_addhs.txt").read_text(encoding="utf-8").splitlines(),
        "smiles_4": (source_dir / "smiles_4_addhs_explicit_h.txt").read_text(encoding="utf-8").splitlines(),
    }
    if any(len(values) != len(data) for values in smiles.values()):
        raise ValueError("QM9 source data and SMILES side files have inconsistent lengths")
    return {"data": data, "smiles": smiles}


def _write_lines(path: Path, values: List[str]) -> None:
    path.write_text("\n".join(values), encoding="utf-8")


def create_qm9test_dataset(
    test_ratio: float = 0.1,
    random_state: int = 42,
    original_indices_path: Optional[str] = None,
    source_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    logger.info(f"🚀 开始创建QM9Test数据集 (比例: {test_ratio*100}%)")

    qm9_source_dir = Path(source_dir).resolve() if source_dir else (REPO_ROOT / "data" / "qm9")
    qm9test_dir = Path(output_dir).resolve() if output_dir else (REPO_ROOT / "data" / "qm9test")
    qm9test_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📂 从 {qm9_source_dir} 加载QM9基线数据...")
    source_payload = _load_qm9_source(qm9_source_dir)
    all_data = source_payload["data"]
    all_smiles = source_payload["smiles"]

    logger.info(f"✅ 成功加载 {len(all_data)} 个QM9样本")

    replay_spec = _load_replay_spec(original_indices_path)
    if replay_spec is None:
        np.random.seed(random_state)
        total_size = len(all_data)
        subset_size = int(total_size * test_ratio)
        selected_indices = np.random.choice(total_size, subset_size, replace=False)
        selected_indices = sorted(selected_indices.tolist())
        train_size = int(subset_size * 0.8)
        effective_random_state = random_state
        logger.info(f"🎯 随机选择 {subset_size} 个样本 (索引范围: {min(selected_indices)} - {max(selected_indices)})")
    else:
        selected_indices = replay_spec["original_indices"]
        subset_size = len(selected_indices)
        train_size = int(replay_spec.get("train_samples", int(subset_size * 0.8)))
        effective_random_state = int(replay_spec.get("random_state", random_state))
        logger.info(
            f"🎯 使用显式原始索引文件，共 {len(selected_indices)} 个样本 "
            f"(索引范围: {min(selected_indices)} - {max(selected_indices)})"
        )

    test_data = [all_data[i] for i in selected_indices]

    logger.info("📊 创建新的数据划分 (8:1:1)...")
    from sklearn.model_selection import train_test_split

    train_indices, temp_indices = train_test_split(
        selected_indices,
        train_size=train_size,
        random_state=effective_random_state,
        shuffle=True,
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=0.5,
        random_state=effective_random_state,
        shuffle=True,
    )

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)

    train_new_indices = [selected_indices.index(i) for i in train_indices]
    val_new_indices = [selected_indices.index(i) for i in val_indices]
    test_new_indices = [selected_indices.index(i) for i in test_indices]

    logger.info(f"📈 划分完成:")
    logger.info(f"   - 训练集: {len(train_new_indices)} 个样本")
    logger.info(f"   - 验证集: {len(val_new_indices)} 个样本")
    logger.info(f"   - 测试集: {len(test_new_indices)} 个样本")

    logger.info("💾 保存统一数据文件...")
    data_file = qm9test_dir / "data.pkl"
    with data_file.open("wb") as f:
        pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("💾 保存索引文件...")
    with (qm9test_dir / "train_index.json").open("w") as f:
        json.dump(train_new_indices, f)
    with (qm9test_dir / "val_index.json").open("w") as f:
        json.dump(val_new_indices, f)
    with (qm9test_dir / "test_index.json").open("w") as f:
        json.dump(test_new_indices, f)

    logger.info("💾 保存SMILES文件...")
    _write_lines(qm9test_dir / "smiles_1_direct.txt", [all_smiles["smiles_1"][i] for i in selected_indices])
    _write_lines(qm9test_dir / "smiles_2_explicit_h.txt", [all_smiles["smiles_2"][i] for i in selected_indices])
    _write_lines(qm9test_dir / "smiles_3_addhs.txt", [all_smiles["smiles_3"][i] for i in selected_indices])
    _write_lines(qm9test_dir / "smiles_4_addhs_explicit_h.txt", [all_smiles["smiles_4"][i] for i in selected_indices])

    metadata = {
        "dataset_name": "qm9test",
        "source_dataset": "qm9",
        "test_ratio": test_ratio,
        "total_samples": len(test_data),
        "train_samples": len(train_new_indices),
        "val_samples": len(val_new_indices),
        "test_samples": len(test_new_indices),
        "original_indices": selected_indices,
        "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": effective_random_state,
        "original_indices_path": original_indices_path,
    }
    if replay_spec is not None:
        metadata.update({k: v for k, v in replay_spec.items() if k != "original_indices"})
        metadata["original_indices"] = selected_indices
        metadata["train_samples"] = len(train_new_indices)
        metadata["val_samples"] = len(val_new_indices)
        metadata["test_samples"] = len(test_new_indices)
        metadata["total_samples"] = len(test_data)
        metadata["source_dataset"] = "qm9"
        metadata["dataset_name"] = "qm9test"
        if "original_indices_path" in replay_spec:
            metadata["original_indices_path"] = replay_spec["original_indices_path"]
        else:
            metadata.pop("original_indices_path", None)

    with (qm9test_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("✅ QM9Test数据集创建完成！")
    logger.info(f"📁 数据目录: {qm9test_dir}")
    logger.info(f"📊 数据集统计:")
    logger.info(f"   - 总样本数: {len(test_data)}")
    logger.info(f"   - 训练集: {len(train_new_indices)}")
    logger.info(f"   - 验证集: {len(val_new_indices)}")
    logger.info(f"   - 测试集: {len(test_new_indices)}")
    logger.info(f"   - 原始QM9索引范围: {min(selected_indices)} - {max(selected_indices)}")


def verify_qm9test_dataset(output_dir: Optional[str] = None) -> None:
    logger.info("🔍 验证QM9Test数据集...")

    qm9test_dir = Path(output_dir).resolve() if output_dir else (REPO_ROOT / "data" / "qm9test")
    required_files = [
        "data.pkl",
        "train_index.json",
        "val_index.json",
        "test_index.json",
        "smiles_1_direct.txt",
        "smiles_2_explicit_h.txt",
        "smiles_3_addhs.txt",
        "smiles_4_addhs_explicit_h.txt",
        "metadata.json"
    ]
    
    for file_name in required_files:
        file_path = qm9test_dir / file_name
        if not file_path.exists():
            logger.error(f"❌ 缺少文件: {file_name}")
            return
        logger.info(f"✅ 文件存在: {file_name}")

    try:
        with (qm9test_dir / "data.pkl").open("rb") as f:
            data = pickle.load(f)
        with (qm9test_dir / "train_index.json").open("r") as f:
            train_indices = json.load(f)
        with (qm9test_dir / "val_index.json").open("r") as f:
            val_indices = json.load(f)
        with (qm9test_dir / "test_index.json").open("r") as f:
            test_indices = json.load(f)

        all_indices = set(train_indices) | set(val_indices) | set(test_indices)
        assert len(all_indices) == len(train_indices) + len(val_indices) + len(test_indices), "索引有重复"
        assert max(all_indices) < len(data), "索引超出范围"
        assert train_indices == sorted(train_indices), "训练集索引未按顺序排列"
        assert val_indices == sorted(val_indices), "验证集索引未按顺序排列"
        assert test_indices == sorted(test_indices), "测试集索引未按顺序排列"

        with (qm9test_dir / "smiles_1_direct.txt").open("r") as f:
            smiles_1 = f.read().strip().split('\n')
        assert len(smiles_1) == len(data), "SMILES数量与数据不匹配"

        logger.info("✅ QM9Test数据集验证通过！")
        logger.info(f"📊 验证结果:")
        logger.info(f"   - 数据样本数: {len(data)}")
        logger.info(f"   - 训练集索引: {len(train_indices)}")
        logger.info(f"   - 验证集索引: {len(val_indices)}")
        logger.info(f"   - 测试集索引: {len(test_indices)}")
        logger.info(f"   - SMILES数量: {len(smiles_1)}")
    except Exception as e:
        logger.error(f"❌ 数据集验证失败: {e}")
        raise


def main():
    logger.info("🚀 QM9Test数据集创建工具")
    parser = argparse.ArgumentParser(description="Create QM9Test by replaying a subset from data/qm9.")
    parser.add_argument("--original-indices-path", type=str, default=None, help="JSON list or metadata.json containing original QM9 indices")
    parser.add_argument("--source-dir", type=str, default=None, help="QM9 source directory, defaults to data/qm9")
    parser.add_argument("--output-dir", type=str, default=None, help="QM9Test output directory, defaults to data/qm9test")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    create_qm9test_dataset(
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        original_indices_path=args.original_indices_path,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
    verify_qm9test_dataset(args.output_dir)
    logger.info("🎉 QM9Test数据集创建和验证完成！")


if __name__ == "__main__":
    main()
