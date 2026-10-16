from __future__ import annotations

import pytest
import json
from types import SimpleNamespace

from vcg_biomarker.mimic.capacity_planning import GIB, build_capacity_plan
from vcg_biomarker.mimic.capacity_planning import require_capacity_at_start
from vcg_biomarker.mimic.storage import file_sha256


def _master():
    return {
        "splits": [
            {"split": "train", "ecg_count": 600},
            {"split": "validation", "ecg_count": 200},
            {"split": "locked_test", "ecg_count": 200},
        ]
    }


def _pilot():
    return {
        "locked_test_read": False,
        "feature_partition_rows_not_verified_unique_patients": 100,
        "feature_partition_logical_bytes": 1_000_000,
        "feature_partitions": [
            {"rows": 50, "columns": 1943},
            {"rows": 50, "columns": 1943},
        ],
    }


def test_capacity_plan_scales_only_feature_components_by_records():
    report = build_capacity_plan(
        _master(),
        _pilot(),
        free_bytes=100 * GIB,
        external_records=100,
        concurrent_duckdb_spill_bytes=10 * GIB,
    )
    assert report["development_records"] == 800
    assert report["components_bytes"]["development_feature_partitions"] == 8_000_000
    assert (
        report["components_bytes"][
            "marker_endpoint_results_and_compressed_checkpoints_envelope"
        ]
        == 12 * GIB
    )
    assert report["capacity_gate_passed"] is True


def test_missing_spill_budget_cannot_grant_capacity_even_with_abundant_space():
    report = build_capacity_plan(
        _master(), _pilot(), free_bytes=1000 * GIB, external_records=0
    )
    assert report["status"] == "missing_spill_budget"
    assert not report["capacity_gate_passed"]
    assert report["required_free_bytes_is_lower_bound"]


def test_concurrent_spill_allowance_is_included_in_required_capacity():
    report = build_capacity_plan(
        _master(),
        _pilot(),
        free_bytes=100 * GIB,
        external_records=0,
        concurrent_duckdb_spill_bytes=80 * GIB,
    )
    assert report["components_bytes"]["concurrent_duckdb_spill_envelope"] == 80 * GIB
    assert report["status"] == "insufficient_capacity"


def test_capacity_plan_fails_when_free_space_cannot_preserve_residual():
    report = build_capacity_plan(
        _master(), _pilot(), free_bytes=30 * GIB, external_records=100
    )
    assert report["capacity_gate_passed"] is False
    assert report["margin_bytes"] < 0


@pytest.mark.parametrize(
    "change",
    (
        {"locked_test_read": None},
        {"feature_partitions": [{"rows": 100, "columns": 459}]},
    ),
)
def test_capacity_plan_rejects_unproven_or_narrow_pilot(change):
    pilot = {**_pilot(), **change}
    with pytest.raises(ValueError):
        build_capacity_plan(
            _master(), pilot, free_bytes=100 * GIB, external_records=100
        )


def bound_plan(tmp_path):
    plan = build_capacity_plan(
        _master(),
        _pilot(),
        free_bytes=200 * GIB,
        external_records=100,
        concurrent_duckdb_spill_bytes=10 * GIB,
    )
    plan.update(run_fingerprint="frozen-source", work_root=str(tmp_path))
    path = tmp_path / "capacity.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    release = {
        "run_fingerprint": "frozen-source",
        "gate_evidence": {
            "resource_scaling_and_storage": {
                "relative_path": path.name,
                "sha256": file_sha256(path),
            }
        },
    }
    (tmp_path / "scientific_release.json").write_text(
        json.dumps(release), encoding="utf-8"
    )
    return plan


def test_live_capacity_recheck_rejects_deteriorated_free_space(tmp_path, monkeypatch):
    from vcg_biomarker.mimic import capacity_planning as module

    plan = bound_plan(tmp_path)
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * GIB)
    )
    require_capacity_at_start(tmp_path, tmp_path, "frozen-source")
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=plan["required_free_bytes"] - 1),
    )
    with pytest.raises(RuntimeError, match="insufficient_current_space"):
        require_capacity_at_start(tmp_path, tmp_path, "frozen-source")


def test_capacity_recheck_rejects_stale_source_and_changed_evidence(tmp_path):
    bound_plan(tmp_path)
    with pytest.raises(RuntimeError, match="stale"):
        require_capacity_at_start(tmp_path, tmp_path, "different-source")
    (tmp_path / "capacity.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not_bound"):
        require_capacity_at_start(tmp_path, tmp_path, "frozen-source")


def test_scientific_gate_and_capacity_recheck_accept_same_document(tmp_path, monkeypatch):
    from vcg_biomarker.mimic import capacity_planning as module
    from vcg_biomarker.mimic.scientific_release import (
        REQUIRED_GATE_CHECKS, SCIENTIFIC_RELEASE_VERSION, scientific_release_valid,
    )
    plan = bound_plan(tmp_path)
    release = dict(version=SCIENTIFIC_RELEASE_VERSION, run_id=tmp_path.name,
                   run_fingerprint='frozen-source', status='approved_for_development_full_run',
                   clinical_approval=False, locked_test_read=False, gates={}, gate_evidence={})
    for name, checks in REQUIRED_GATE_CHECKS.items():
        payload = dict(gate=name, status='PASS', run_fingerprint='frozen-source',
                       checks={check: True for check in checks})
        if name == 'resource_scaling_and_storage':
            payload['capacity_plan'] = plan
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        release['gates'][name] = True
        release['gate_evidence'][name] = dict(relative_path=path.name, sha256=file_sha256(path))
    (tmp_path / 'scientific_release.json').write_text(json.dumps(release), encoding='utf-8')
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=200 * GIB))
    assert scientific_release_valid(tmp_path, 'frozen-source')
    require_capacity_at_start(tmp_path, tmp_path, 'frozen-source')
    path = tmp_path / 'resource_scaling_and_storage.json'
    payload = json.loads(path.read_text())
    payload['checks']['resume_verified'] = False
    path.write_text(json.dumps(payload), encoding='utf-8')
    release['gate_evidence']['resource_scaling_and_storage']['sha256'] = file_sha256(path)
    (tmp_path / 'scientific_release.json').write_text(json.dumps(release), encoding='utf-8')
    assert not scientific_release_valid(tmp_path, 'frozen-source')
    with pytest.raises(RuntimeError, match='resource_gate'):
        require_capacity_at_start(tmp_path, tmp_path, 'frozen-source')


@pytest.mark.parametrize(
    "changes",
    [
        {"uncertainty_multiplier": 0.1},
        {"minimum_residual_gib": -1},
        {"external_records": -100},
        {"free_bytes": -1},
    ],
)
def test_invalid_budget_cannot_create_a_pass(changes):
    kwargs = dict(
        free_bytes=200 * GIB,
        external_records=100,
        concurrent_duckdb_spill_bytes=10 * GIB,
    )
    kwargs.update(changes)
    with pytest.raises(ValueError):
        build_capacity_plan(_master(), _pilot(), **kwargs)


def test_full_workflow_checks_live_capacity_before_preparation(tmp_path, monkeypatch):
    from vcg_biomarker.mimic import workflow, scientific_release, services
    from vcg_biomarker.mimic.config import MimicProjectConfig

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "db", work_root=tmp_path / "work"
    )
    monkeypatch.setattr(
        workflow, "_assert_generated_run_is_current", lambda *a: "frozen-source"
    )
    monkeypatch.setattr(workflow, "_assert_run_preflight_gates", lambda *a: None)
    monkeypatch.setattr(scientific_release, "scientific_release_valid", lambda *a: True)
    monkeypatch.setattr(
        services, "run_service_stage", lambda *a, **k: pytest.fail("started a stage")
    )
    with pytest.raises(RuntimeError, match="live capacity preflight"):
        workflow.run_guided_study(config, {"run_id": "new-development"})
    assert not config.runs_root.exists()


def test_capacity_recheck_credits_the_runs_own_feature_partitions(tmp_path, monkeypatch):
    # A resumed run must not be refused for the space its own finished
    # feature partitions occupy; the credit is capped at the planned component.
    from vcg_biomarker.mimic import capacity_planning as module

    plan = bound_plan(tmp_path)
    required = plan["required_free_bytes"]
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda _: SimpleNamespace(free=required - 5 * 1024)
    )
    with pytest.raises(RuntimeError, match="insufficient_current_space"):
        require_capacity_at_start(tmp_path, tmp_path, "frozen-source")
    part = tmp_path / "features" / "panel" / "development" / "split=train"
    part.mkdir(parents=True)
    (part / "part-000000.parquet").write_bytes(b"x" * 6 * 1024)
    require_capacity_at_start(tmp_path, tmp_path, "frozen-source")
    # Files elsewhere are not credited.
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda _: SimpleNamespace(free=required - 20 * 1024)
    )
    other = tmp_path / "features" / "panel" / "smoke-development-1000" / "split=train"
    other.mkdir(parents=True)
    (other / "part-000000.parquet").write_bytes(b"x" * 50 * 1024)
    with pytest.raises(RuntimeError, match="already written"):
        require_capacity_at_start(tmp_path, tmp_path, "frozen-source")
