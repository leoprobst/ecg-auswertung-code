from vcg_biomarker.mimic.workflow import _workflow_backend_hash


def test_gui_edits_leave_workflow_identity_stable_but_backend_edits_do_not(tmp_path):
    for directory in ("mimic", "core", "gui", "tests"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "module.py").write_text("value = 1\n")
    baseline = _workflow_backend_hash(tmp_path)
    (tmp_path / "gui/module.py").write_text("value = 2\n")
    (tmp_path / "tests/module.py").write_text("value = 3\n")
    assert _workflow_backend_hash(tmp_path) == baseline
    (tmp_path / "mimic/module.py").write_text("value = 2\n")
    assert _workflow_backend_hash(tmp_path) != baseline
    baseline = _workflow_backend_hash(tmp_path)
    (tmp_path / "core/module.py").write_text("value = 2\n")
    assert _workflow_backend_hash(tmp_path) != baseline
