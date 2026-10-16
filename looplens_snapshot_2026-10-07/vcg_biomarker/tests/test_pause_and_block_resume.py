"""Pauses are control flow, and finished screen blocks survive them (04.10.2026).

* ``JobPauseRequested`` was a RuntimeError. The broad-axis sweep wraps every
  axis in ``except Exception`` so that one broken axis does not stop the
  others -- and so a pause raised from a progress callback recorded the axis
  as failed and the stage completed without it.
* The sweep kept no intermediate state; a pause late in an axis discarded
  every screened block.
"""

from __future__ import annotations

import inspect

import pytest

from vcg_biomarker.mimic import broad_axis_stage as stage
from vcg_biomarker.mimic.jobs import JobCancellationRequested, JobPauseRequested


@pytest.mark.parametrize("signal", [JobPauseRequested, JobCancellationRequested])
def test_pause_and_cancel_pass_through_a_generic_handler(signal):
    def sweep():
        try:
            raise signal("Pause requested at an atomic checkpoint")
        except Exception:  # what the per-axis guard does
            return "axis recorded as failed"

    with pytest.raises(signal):
        sweep()
    assert not issubclass(signal, Exception)


def test_axis_block_round_trip_and_key_binding(tmp_path):
    records = [{"target_id": "50912", "feature_id": "qrs", "rho": 0.12, "note": None}]
    key = stage._axis_block_key("stage", "laboratory", "continuous", ("qrs",), 10, ["50912"])
    stage._save_axis_block(tmp_path, "laboratory", "continuous", 3, key, records)
    assert stage._load_axis_block(tmp_path, "laboratory", "continuous", 3, key) == records
    other = stage._axis_block_key("stage", "laboratory", "continuous", ("qrs",), 11, ["50912"])
    assert stage._load_axis_block(tmp_path, "laboratory", "continuous", 3, other) is None
    assert stage._load_axis_block(tmp_path, "laboratory", "binary", 3, key) is None
    assert stage._load_axis_block(None, "laboratory", "continuous", 3, key) is None
    # An empty block is a finished block too.
    stage._save_axis_block(tmp_path, "laboratory", "continuous", 4, key, [])
    assert stage._load_axis_block(tmp_path, "laboratory", "continuous", 4, key) == []


def test_key_changes_with_targets_markers_and_stage():
    base = stage._axis_block_key("s", "a", "binary", ("m1", "m2"), 5, ["t1", "t2"])
    assert base == stage._axis_block_key("s", "a", "binary", ("m1", "m2"), 5, ["t2", "t1"])
    for changed in (
        stage._axis_block_key("s2", "a", "binary", ("m1", "m2"), 5, ["t1", "t2"]),
        stage._axis_block_key("s", "a", "binary", ("m1",), 5, ["t1", "t2"]),
        stage._axis_block_key("s", "a", "binary", ("m1", "m2"), 5, ["t1"]),
    ):
        assert changed != base


def test_both_block_loops_consult_the_checkpoint_before_computing():
    source = inspect.getsource(stage._screen_one_axis)
    for kind in ("continuous", "binary"):
        load = source.index(f'_load_axis_block(checkpoint_root, axis, "{kind}"')
        save = source.index(f'_save_axis_block(checkpoint_root, axis, "{kind}"')
        assert load < save


# --- Result-equivalence attestations -----------------------------------------


def _tree(root, files):
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_an_attested_fix_keeps_the_scientific_hash_and_an_unattested_one_does_not(tmp_path):
    import json

    from vcg_biomarker.mimic.code_scope import scientific_source_hash
    from vcg_biomarker.mimic.storage import file_sha256

    tree = _tree(tmp_path, {"trajectory.py": "x = 1\n", "validation.py": "y = 1\n"})
    before = scientific_source_hash(tree)
    old_sha = file_sha256(tree / "trajectory.py")
    (tree / "trajectory.py").write_text("x = 1  # columnar, same result\n", encoding="utf-8")
    assert scientific_source_hash(tree) != before
    (tree / "result_equivalence.json").write_text(json.dumps([{
        "module": "trajectory.py", "from_sha256": old_sha,
        "to_sha256": file_sha256(tree / "trajectory.py"),
        "tests": ["vcg_biomarker/tests/test_x.py::test_same"], "reason": "memory"}]), encoding="utf-8")
    assert scientific_source_hash(tree) == before
    (tree / "validation.py").write_text("y = 2\n", encoding="utf-8")
    assert scientific_source_hash(tree) != before


def test_attestation_without_tests_is_rejected(tmp_path):
    import json

    from vcg_biomarker.mimic.code_scope import load_attestations

    (tmp_path / "result_equivalence.json").write_text(json.dumps([{
        "module": "a.py", "from_sha256": "1", "to_sha256": "2", "tests": [], "reason": "x"}]),
        encoding="utf-8")
    with pytest.raises(ValueError):
        load_attestations(tmp_path)


def test_operational_fix_accepts_attested_analysis_modules_but_never_marker_code():
    from audit_tools.apply_operational_fix import classify_changes

    attest = [{"module": "trajectory.py", "from_sha256": "old", "to_sha256": "new",
               "tests": ["t::x"], "reason": "r"},
              {"module": "classical.py", "from_sha256": "old", "to_sha256": "new",
               "tests": ["t::x"], "reason": "r"}]
    old = {"mimic/trajectory.py": "old", "mimic/classical.py": "old", "mimic/validation.py": "old"}
    allowed, refused = classify_changes(old, dict(old, **{"mimic/trajectory.py": "new"}), attest)
    assert allowed == ["mimic/trajectory.py"] and refused == []
    _, refused = classify_changes(old, dict(old, **{"mimic/classical.py": "new"}), attest)
    assert refused == ["mimic/classical.py"]
    _, refused = classify_changes(old, dict(old, **{"mimic/validation.py": "new"}), attest)
    assert refused == ["mimic/validation.py"]
