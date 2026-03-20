from pathlib import Path


RELEASE_SCOPE_SCRIPT_PATHS = [
    "data/coildel/preprocess_coil_del.py",
    "data/colors3/preprocess_colors3.py",
    "data/dblp/preprocess_dblp_v1.py",
    "data/dd/preprocess_dd.py",
    "data/mnist_raw/prepare.py",
    "data/molhiv/preprocess_molhiv.py",
    "data/mutagenicity/preprocess_mutagenicity.py",
    "data/peptides_func/prepare_lrgb_data.py",
    "data/peptides_struct/prepare_lrgb_data.py",
    "data/proteins/preprocess_proteins.py",
    "data/qm9/prepare_qm9_raw.py",
    "data/qm9test/create_qm9test_dataset.py",
    "data/synthetic/preprocess_synthetic.py",
    "data/twitter/preprocess_twitter_real_graph_partial.py",
]

DEV_ONLY_EXPERIMENTAL_SCRIPT_PATHS = [
    "data/code2/preprocess_code2.py",
    "data/aqsol/prepare_aqsol_raw.py",
    "data/zinc/prepare_zinc_raw.py",
    "data/zinc/usage_example_new.py",
    "data/mnist/convert_mnist_to_dgl.py",
]

DEV_ONLY_HELPER_SCRIPT_PATHS = [
    "data/qm9/process_qm9_dataset.py",
    "data/mnist/convert_test.py",
    "data/mnist/test_2slic.py",
]


def test_release_scope_preprocess_scripts_exist():
    for script_path in RELEASE_SCOPE_SCRIPT_PATHS:
        assert Path(script_path).exists(), f"missing release-scope script: {script_path}"


def test_dev_only_experimental_and_helper_scripts_exist():
    for script_path in DEV_ONLY_EXPERIMENTAL_SCRIPT_PATHS + DEV_ONLY_HELPER_SCRIPT_PATHS:
        assert Path(script_path).exists(), f"missing documented dev-only script: {script_path}"


def test_dataset_conversion_readme_mentions_secondary_and_experimental_scripts():
    readme = Path("scripts/dataset_conversion/README.md").read_text(encoding="utf-8")
    assert "secondary helpers" in readme
    assert "experimental scripts" in readme
    assert "data/qm9/process_qm9_dataset.py" in readme
    assert "`code2`" in readme
    assert "`zinc`" in readme
    assert "`aqsol`" in readme
    assert "`qm9test`" in readme
