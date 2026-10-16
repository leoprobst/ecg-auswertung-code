"""Form errors are reported; only mathematics and logic stop a run.

A143, A150 and A151 were each repeated for a defect in how the run was
measured or recorded, not in what it computed: a rewritten timestamp, a
recovered 3.8 s stage failure, a reuse verdict that was never logged. These
tests pin the line between the two: run machinery, telemetry, resource
settings and the host environment are recorded and reported, while marker
definitions, scientific code, configuration, computation libraries and
upstream content keep refusing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcg_biomarker.mimic import code_scope, marker_freeze
from vcg_biomarker.mimic.code_scope import (
    OPERATIONAL_MODULES,
    operational_source_hash,
    scientific_source_hash,
)
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.marker_freeze import (
    assert_marker_freeze_current,
    freeze_marker_inventory,
)
from vcg_biomarker.mimic.pilot_go_no_go import FORM_GATES, pilot_technical_go

MIMIC_ROOT = Path(code_scope.__file__).resolve().parent


def _config(tmp_path, *, workers=1):
    return MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "echo.csv",
        echo_study_list_path=tmp_path / "echo-study.csv",
        echo_record_list_path=tmp_path / "echo-record.csv",
        work_root=tmp_path / "work",
        workers=workers,
        duckdb_memory_limit_gb=2,
    )


# --- Which code is run machinery --------------------------------------------


def test_every_operational_module_exists_and_none_computes_markers():
    present = {path.name for path in MIMIC_ROOT.glob("*.py")}
    assert OPERATIONAL_MODULES <= present
    # The record-level marker computation and anything that builds results
    # must never be classified as run machinery.
    for scientific in ("feature_record.py", "feature_extraction.py", "classical.py",
                       "validation.py", "labels.py", "evidence.py", "statistics.py",
                       "workflow.py", "marker_freeze.py", "run_lineage.py", "storage.py"):
        assert scientific not in OPERATIONAL_MODULES


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_operational_edit_keeps_the_scientific_hash(tmp_path):
    tree = _tree(tmp_path, {"supervisor.py": "a = 1\n", "validation.py": "b = 1\n"})
    scientific = scientific_source_hash(tree)
    operational = operational_source_hash(tree)
    (tree / "supervisor.py").write_text("a = 2  # race fixed\n", encoding="utf-8")
    assert scientific_source_hash(tree) == scientific
    assert operational_source_hash(tree) != operational


def test_scientific_or_unknown_module_moves_the_scientific_hash(tmp_path):
    tree = _tree(tmp_path, {"supervisor.py": "a = 1\n", "validation.py": "b = 1\n"})
    scientific = scientific_source_hash(tree)
    (tree / "validation.py").write_text("b = 2\n", encoding="utf-8")
    assert scientific_source_hash(tree) != scientific
    changed = scientific_source_hash(tree)
    # A module nobody classified yet counts as scientific.
    (tree / "brand_new_stage.py").write_text("c = 1\n", encoding="utf-8")
    assert scientific_source_hash(tree) != changed


def test_run_identity_ignores_run_machinery(monkeypatch, tmp_path):
    from vcg_biomarker.mimic import run_lineage

    config = _config(tmp_path)
    before = run_lineage.compute_fingerprint(config)
    real = code_scope.operational_source_hash
    monkeypatch.setattr(code_scope, "operational_source_hash", lambda root: "x" * 64)
    after = run_lineage.compute_fingerprint(config)
    assert after.fingerprint == before.fingerprint
    assert after.components["operational"] == "x" * 64
    assert before.components["operational"] == real(MIMIC_ROOT)


# --- Pilot go/no-go ----------------------------------------------------------


def _report(**gates):
    blocking = [name for name in gates if name not in FORM_GATES]
    failed = [name for name in blocking if not gates[name]]
    return {"decision": "NO_GO" if failed else "TECHNICAL_GO_SCIENTIFIC_REVIEW_REQUIRED",
            "gates": gates, "blocking_gates": blocking}


def test_measurement_gaps_do_not_block_a_technical_go():
    report = _report(pilot_10000_validation_complete=True, marker_freeze_current=True,
                     runtime_query_telemetry_well_formed=False,
                     pilot_feature_throughput_measured=False)
    assert pilot_technical_go(report)


def test_scientific_gates_still_block():
    for blocking in ("marker_freeze_current", "pilot_10000_validation_complete",
                     "locked_test_unread", "no_uncaptured_warnings"):
        assert blocking not in FORM_GATES
        report = _report(**{blocking: False, "runtime_query_telemetry_well_formed": True})
        assert not pilot_technical_go(report)


def test_reports_before_form_classification_keep_every_gate_binding():
    legacy = {"decision": "TECHNICAL_GO_SCIENTIFIC_REVIEW_REQUIRED",
              "gates": {"marker_freeze_current": True,
                        "runtime_query_telemetry_well_formed": False}}
    assert not pilot_technical_go(legacy)


# --- Marker freeze snapshot ---------------------------------------------------


def _frozen(tmp_path, **kwargs):
    config = _config(tmp_path, **kwargs)
    config.ensure_directories()
    (config.work_root / "inventory.manifest.json").write_text(json.dumps({
        "stage": "inventory", "input_hash": "abc", "created_at": "2026-10-01T00:00:00",
        "notes": ["operational_config_hash=aaa", "icu_coverage_status=complete"],
        "output_integrity": [{"path": "E:/old/place.json", "mtime_ns": 1, "sha256": "f00"}],
    }), encoding="utf-8")
    freeze_marker_inventory(config, run_id="R001", finalize_inputs=True)
    return config


def test_rewritten_timestamps_paths_and_resource_notes_are_not_input_changes(tmp_path):
    config = _frozen(tmp_path)
    (config.work_root / "inventory.manifest.json").write_text(json.dumps({
        "stage": "inventory", "input_hash": "abc", "created_at": "2026-10-02T09:00:00",
        "notes": ["operational_config_hash=bbb", "icu_coverage_status=complete"],
        "output_integrity": [{"path": "D:/new/place.json", "mtime_ns": 2, "sha256": "f00"}],
    }), encoding="utf-8")
    assert_marker_freeze_current(config, run_id="R001")


def test_changed_manifest_content_still_refuses_and_names_the_input(tmp_path):
    config = _frozen(tmp_path)
    (config.work_root / "inventory.manifest.json").write_text(json.dumps({
        "stage": "inventory", "input_hash": "CHANGED", "created_at": "2026-10-01T00:00:00",
        "notes": ["operational_config_hash=aaa", "icu_coverage_status=complete"],
        "output_integrity": [{"path": "E:/old/place.json", "mtime_ns": 1, "sha256": "f00"}],
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="inputs.inventory_manifest"):
        assert_marker_freeze_current(config, run_id="R001")


def test_resource_settings_and_host_drift_are_logged_not_refused(tmp_path, monkeypatch):
    config = _frozen(tmp_path)
    moved = _config(tmp_path, workers=7)
    monkeypatch.setattr(marker_freeze.platform, "platform", lambda: "Windows-11-new-build")
    first = assert_marker_freeze_current(moved, run_id="R001")
    second = assert_marker_freeze_current(moved, run_id="R001")
    assert {"operational_config_hash", "platform"} <= set(first["form_drift"])
    assert second["form_drift"] == first["form_drift"]
    lines = (config.runs_root / "R001" / "form_drift.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # the same drift is recorded once, not per stage


def test_computation_library_change_still_refuses(tmp_path, monkeypatch):
    config = _frozen(tmp_path)
    real = marker_freeze._dependency_versions

    def bumped(names=marker_freeze.COMPUTATION_DISTRIBUTIONS):
        versions = real(names)
        if "numpy" in versions:
            versions["numpy"] = "0.0.0-other"
        return versions

    monkeypatch.setattr(marker_freeze, "_dependency_versions", bumped)
    with pytest.raises(RuntimeError, match="dependencies"):
        assert_marker_freeze_current(config, run_id="R001")


def test_gui_toolkit_is_recorded_but_not_enforced():
    assert "PyQt6" in marker_freeze.DEPENDENCY_DISTRIBUTIONS
    assert "PyQt6" not in marker_freeze.COMPUTATION_DISTRIBUTIONS
    assert "numpy" in marker_freeze.COMPUTATION_DISTRIBUTIONS


# --- Mid-run operational fixes (audit_tools) ---------------------------------


def test_reboot_guard_ignores_unrelated_packages(monkeypatch):
    """`pip install pypdf` by another session blocked the A147 restart."""
    import importlib.metadata

    from audit_tools import reboot_guard

    before = reboot_guard.environment_identity()
    real = importlib.metadata.version

    def version(name):
        if name == "pypdf":
            return "6.19.0"
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert reboot_guard.environment_identity() == before
    assert "pypdf" not in reboot_guard.COMPUTATION_DISTRIBUTIONS


def test_reboot_guard_tracks_the_same_libraries_as_the_marker_freeze():
    from audit_tools import reboot_guard

    assert set(reboot_guard.COMPUTATION_DISTRIBUTIONS) == set(
        marker_freeze.COMPUTATION_DISTRIBUTIONS
    )


def test_operational_fix_accepts_only_run_machinery_tests_gui_and_docs():
    from audit_tools.apply_operational_fix import classify_changes

    old = {"mimic/supervisor.py": "a", "mimic/validation.py": "b", "tests/test_x.py": "c",
           "gui/view.py": "d", "core/biomarkers/qt.py": "e", "RUNTIME_OPERATIONS.md": "f",
           "mimic_job_worker.py": "g"}
    operational = dict(old, **{"mimic/supervisor.py": "A", "tests/test_x.py": "C", "gui/view.py": "D",
                               "RUNTIME_OPERATIONS.md": "F", "mimic_job_worker.py": "G"})
    allowed, refused = classify_changes(old, operational)
    assert refused == []
    assert set(allowed) == {"mimic/supervisor.py", "tests/test_x.py", "gui/view.py",
                            "RUNTIME_OPERATIONS.md", "mimic_job_worker.py"}
    for scientific in ("mimic/validation.py", "core/biomarkers/qt.py", "config.py"):
        _, refused = classify_changes(old, dict(old, **{scientific: "changed"}))
        assert refused == [scientific]


def test_reboot_guard_accepts_a_changed_worker_count_but_not_a_scientific_setting(tmp_path):
    from audit_tools import reboot_guard
    from vcg_biomarker.mimic.storage import scientific_config_hash

    config = _config(tmp_path, workers=2)
    project = tmp_path / "project.json"
    config.save(project) if hasattr(config, "save") else project.write_text(
        json.dumps(config.to_dict()), encoding="utf-8")
    policy = {"project": str(project), "project_sha256": "stale",
              "project_scientific_config_hash": scientific_config_hash(MimicProjectConfig.load(project))}
    assert reboot_guard._only_operational_project_change(policy)
    changed = dict(json.loads(project.read_text(encoding="utf-8")), random_seed=999)
    project.write_text(json.dumps(changed), encoding="utf-8")
    assert not reboot_guard._only_operational_project_change(policy)
    assert not reboot_guard._only_operational_project_change({"project": str(project)})
