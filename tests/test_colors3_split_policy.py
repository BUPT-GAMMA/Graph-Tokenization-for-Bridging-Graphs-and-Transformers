from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path("data/colors3/preprocess_colors3.py").resolve()
SPEC = spec_from_file_location("colors3_preprocess", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_splits = MODULE.build_splits


def test_colors3_build_splits_matches_current_baseline_policy():
    labels = [i % 11 for i in range(10500)]
    train, val, test = build_splits(len(labels), labels)
    assert train[:5] == [0, 1, 2, 3, 4]
    assert train[-1] == 8399
    assert val[:5] == [8400, 8401, 8402, 8403, 8404]
    assert test[:5] == [9450, 9451, 9452, 9453, 9454]
