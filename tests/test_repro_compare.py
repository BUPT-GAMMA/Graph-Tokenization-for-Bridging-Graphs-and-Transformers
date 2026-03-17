import json
import gzip
import pickle
from pathlib import Path

from src.data.repro_compare import compare_dataset_artifacts, semantic_digest_pickle, sha256_file


def _write_pickle(path: Path, payload) -> None:
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sha256_file_is_stable(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("abc", encoding="utf-8")
    digest1 = sha256_file(target)
    digest2 = sha256_file(target)
    assert digest1 == digest2


def test_compare_dataset_artifacts_reports_identical_json_and_pickle(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()

    payload = [([1, 2], {"y": 1.0}), ([3, 4], {"y": 2.0})]
    _write_pickle(baseline / "data.pkl", payload)
    _write_pickle(candidate / "data.pkl", payload)
    _write_json(baseline / "train_index.json", [0])
    _write_json(candidate / "train_index.json", [0])
    _write_json(baseline / "val_index.json", [1])
    _write_json(candidate / "val_index.json", [1])
    _write_json(baseline / "test_index.json", [])
    _write_json(candidate / "test_index.json", [])

    report = compare_dataset_artifacts(baseline, candidate)
    assert report["all_match"] is True
    assert report["files"]["data.pkl"]["match"] is True
    assert report["files"]["train_index.json"]["match"] is True


def test_compare_dataset_artifacts_reports_json_mismatch(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()

    payload = [([1, 2], {"y": 1.0})]
    _write_pickle(baseline / "data.pkl", payload)
    _write_pickle(candidate / "data.pkl", payload)
    _write_json(baseline / "train_index.json", [0])
    _write_json(candidate / "train_index.json", [1])

    report = compare_dataset_artifacts(baseline, candidate, required_files=["data.pkl", "train_index.json"])
    assert report["all_match"] is False
    assert report["files"]["train_index.json"]["match"] is False


def test_semantic_digest_pickle_can_ignore_protocol_differences(tmp_path):
    baseline = tmp_path / "a.pkl"
    candidate = tmp_path / "b.pkl"
    payload = [({"x": [1, 2, 3]}, 1), ({"x": [4, 5, 6]}, 0)]
    with baseline.open("wb") as f:
        pickle.dump(payload, f, protocol=4)
    with candidate.open("wb") as f:
        pickle.dump(payload, f, protocol=5)

    assert sha256_file(baseline) != sha256_file(candidate)
    assert semantic_digest_pickle(baseline) == semantic_digest_pickle(candidate)


def test_compare_dataset_artifacts_supports_pickle_gzip(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    payload = [({"x": [1, 2]}, 1), ({"x": [3, 4]}, 0)]
    with gzip.open(baseline / "data.pkl.gz", "wb") as f:
        pickle.dump(payload, f, protocol=4)
    with gzip.open(candidate / "data.pkl.gz", "wb") as f:
        pickle.dump(payload, f, protocol=5)

    report = compare_dataset_artifacts(baseline, candidate, required_files=["data.pkl.gz"])
    assert report["files"]["data.pkl.gz"]["match"] is False
    assert report["files"]["data.pkl.gz"]["semantic_match"] is True
