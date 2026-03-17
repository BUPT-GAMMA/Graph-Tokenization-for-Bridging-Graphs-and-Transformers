from __future__ import annotations

import hashlib
import gzip
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


def _update_digest_with_value(digest: "hashlib._Hash", value) -> None:
    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value.keys(), key=lambda x: repr(x)):
            _update_digest_with_value(digest, key)
            _update_digest_with_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}:{len(value)}".encode())
        for item in value:
            _update_digest_with_value(digest, item)
        return
    if isinstance(value, (str, int, float, bool, type(None))):
        digest.update(repr(value).encode())
        return
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "numpy"):
        arr = value.numpy()
        digest.update(str(tuple(arr.shape)).encode())
        digest.update(str(arr.dtype).encode())
        digest.update(arr.tobytes())
        return
    if isinstance(value, np.ndarray):
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
        return
    if hasattr(value, "ndata") and hasattr(value, "edata") and hasattr(value, "num_nodes") and hasattr(value, "num_edges"):
        digest.update(f"graph:{value.num_nodes()}:{value.num_edges()}".encode())
        for key in sorted(value.ndata.keys()):
            digest.update(f"ndata:{key}".encode())
            _update_digest_with_value(digest, value.ndata[key])
        for key in sorted(value.edata.keys()):
            digest.update(f"edata:{key}".encode())
            _update_digest_with_value(digest, value.edata[key])
        return
    digest.update(repr(value).encode())


def semantic_digest_pickle(path: Path) -> str:
    if path.name.endswith(".pkl.gz"):
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
    else:
        with path.open("rb") as f:
            payload = pickle.load(f)
    digest = hashlib.sha256()
    _update_digest_with_value(digest, payload)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_json_files(baseline: Path, candidate: Path) -> dict:
    baseline_obj = json.loads(baseline.read_text(encoding="utf-8"))
    candidate_obj = json.loads(candidate.read_text(encoding="utf-8"))
    return {
        "match": baseline_obj == candidate_obj,
        "baseline_sha256": sha256_file(baseline),
        "candidate_sha256": sha256_file(candidate),
        "baseline_length": len(baseline_obj) if hasattr(baseline_obj, "__len__") else None,
        "candidate_length": len(candidate_obj) if hasattr(candidate_obj, "__len__") else None,
    }


def _compare_binary_files(baseline: Path, candidate: Path) -> dict:
    baseline_sha = sha256_file(baseline)
    candidate_sha = sha256_file(candidate)
    semantic_match = None
    if baseline.name.endswith(".pkl") or baseline.name.endswith(".pkl.gz"):
        semantic_match = semantic_digest_pickle(baseline) == semantic_digest_pickle(candidate)
    return {
        "match": baseline_sha == candidate_sha,
        "semantic_match": semantic_match,
        "baseline_sha256": baseline_sha,
        "candidate_sha256": candidate_sha,
    }


def compare_dataset_artifacts(
    baseline_dir: Path | str,
    candidate_dir: Path | str,
    *,
    required_files: Iterable[str] | None = None,
) -> dict:
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    required_files = list(required_files or ["data.pkl", "train_index.json", "val_index.json", "test_index.json"])

    report = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "required_files": required_files,
        "files": {},
        "all_match": True,
    }

    for filename in required_files:
        baseline_path = baseline_dir / filename
        candidate_path = candidate_dir / filename
        if not baseline_path.exists() or not candidate_path.exists():
            report["files"][filename] = {
                "match": False,
                "baseline_exists": baseline_path.exists(),
                "candidate_exists": candidate_path.exists(),
            }
            report["all_match"] = False
            continue

        if baseline_path.suffix == ".json":
            file_report = _compare_json_files(baseline_path, candidate_path)
        else:
            file_report = _compare_binary_files(baseline_path, candidate_path)
        report["files"][filename] = file_report
        if not file_report["match"]:
            report["all_match"] = False

    return report
