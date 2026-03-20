from pathlib import Path


QM9_KEYS = {
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
    "u0_atom",
    "u298_atom",
    "h298_atom",
    "g298_atom",
}


def test_audit_doc_records_qm9_current_baseline_format():
    audit = Path("docs/datasets_overview.md").read_text(encoding="utf-8")
    script = Path("data/qm9/prepare_qm9_raw.py").read_text(encoding="utf-8")
    assert "## 12. `qm9`" in audit
    assert "list[tuple[DGLGraph, dict]]" in audit
    assert "Label 全集键值" in audit
    assert "节点特征 (ndata):** `['attr', 'pos']`" in audit
    assert "边特征 (edata):** `['edge_attr']`" in audit
    for key in sorted(QM9_KEYS):
        assert f'"{key}"' in script


def test_audit_doc_records_zinc_current_baseline_format():
    audit = Path("docs/datasets_overview.md").read_text(encoding="utf-8")
    assert "## 16. `zinc`" in audit
    assert "list[tuple[DGLGraph, torch.Tensor]]" in audit
    assert "shape=(1,), dtype=torch.float32" in audit
    assert "节点特征 (ndata):** `['feat']`" in audit


def test_audit_doc_records_aqsol_current_baseline_format():
    audit = Path("docs/datasets_overview.md").read_text(encoding="utf-8")
    assert "## 1. `aqsol`" in audit
    assert "list[tuple[DGLGraph, float]]" in audit
    assert "Label 类型:** `float`" in audit
    assert "节点特征 (ndata):** `['feat']`" in audit
