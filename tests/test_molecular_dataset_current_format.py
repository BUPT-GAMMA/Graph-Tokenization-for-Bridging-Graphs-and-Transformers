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
    audit = Path("docs/reproducibility/dataset-cold-start-audit.md").read_text(encoding="utf-8")
    script = Path("data/qm9/prepare_qm9_raw.py").read_text(encoding="utf-8")
    assert "sample type: `(DGLGraph, dict)`" in audit
    assert "label payload: 16-property dictionary" in audit
    assert "graph fields: `ndata['pos']`, `ndata['attr']`, `edata['edge_attr']`" in audit
    for key in sorted(QM9_KEYS):
        assert f'"{key}"' in script


def test_audit_doc_records_zinc_current_baseline_format():
    audit = Path("docs/reproducibility/dataset-cold-start-audit.md").read_text(encoding="utf-8")
    assert "sample type: `(DGLGraph, torch.Tensor)`" in audit
    assert "label payload: scalar tensor with shape `(1,)`" in audit
    assert "graph fields: `ndata['feat']`, `edata['feat']`" in audit


def test_audit_doc_records_aqsol_current_baseline_format():
    audit = Path("docs/reproducibility/dataset-cold-start-audit.md").read_text(encoding="utf-8")
    assert "sample type: `(DGLGraph, float)`" in audit
    assert "label payload: Python `float`" in audit
    assert "graph fields: `ndata['feat']`, `edata['feat']`" in audit
