from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vcg_biomarker.mimic_job_worker import (
    _apply_job_resource_environment,
    _sync_guided_manifest_status,
)
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.jobs import JobControl, JobStore
from vcg_biomarker.mimic.models import JobState, ResourceProfile
from vcg_biomarker.mimic.resources import (
    MemoryStatus,
    stage_memory_safety_policy,
    stage_worker_memory_scope,
)
from vcg_biomarker.mimic.supervisor import (
    MEMORY_PRESSURE_SAMPLES,
    LocalJobSupervisor,
)


def _config(tmp_path: Path) -> MimicProjectConfig:
    return MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic-iv",
        mimic_ed_root=tmp_path / "mimic-ed",
        mimic_ecg_root=tmp_path / "mimic-ecg",
        echo_structured_path=tmp_path / "echo.csv",
        echo_study_list_path=tmp_path / "echo-study.csv",
        echo_record_list_path=tmp_path / "echo-record.csv",
        work_root=tmp_path / "work",
        workers=2,
        duckdb_memory_limit_gb=20,
        os_memory_reserve_gb=6.0,
    )


def test_study_worker_failure_updates_guided_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.ensure_directories()
    run_id = "current-run"
    path = config.runs_root / run_id / "guided_workflow.manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"status": "running", "completed_stages": ["inventory"]}),
        encoding="utf-8",
    )
    spec = SimpleNamespace(stage="study-run", arguments={"run_id": run_id})

    _sync_guided_manifest_status(
        config,
        spec,
        status="failed",
        diagnostic="OutOfMemoryException",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["terminal_diagnostic"] == "OutOfMemoryException"
    assert payload["completed_stages"] == ["inventory"]


def test_stage_memory_policy_separates_allocation_from_duckdb_budget() -> None:
    streaming = stage_memory_safety_policy(
        "ingest-events",
        reserve_gb=6.0,
        requested_memory_gb=14.0,
    )
    waveform = stage_memory_safety_policy(
        "extract-features",
        reserve_gb=6.0,
        requested_memory_gb=14.0,
    )

    assert streaming.duckdb_memory_limit_gb == 3.0
    assert streaming.non_duckdb_headroom_gb == 11.0
    assert waveform.duckdb_memory_limit_gb < 14.0
    assert waveform.non_duckdb_headroom_gb >= 3.5
    assert waveform.critical_available_gb == streaming.critical_available_gb == 6.0
    assert waveform.recovery_available_gb > waveform.critical_available_gb


def test_stage_memory_scope_is_bounded_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCG_JOB_MEMORY_ALLOCATION_GB", "14")
    monkeypatch.setenv("VCG_JOB_MEMORY_GB", "10")
    monkeypatch.setenv("VCG_JOB_OS_RESERVE_GB", "6")

    with stage_worker_memory_scope("ingest-events") as policy:
        assert policy.duckdb_memory_limit_gb == 3.0
        assert os.environ["VCG_JOB_MEMORY_GB"] == "3.0"
        assert os.environ["VCG_JOB_ACTIVE_STAGE"] == "ingest-events"

    assert os.environ["VCG_JOB_MEMORY_GB"] == "10"
    assert "VCG_JOB_ACTIVE_STAGE" not in os.environ


def test_worker_exports_allocation_but_clamps_duckdb_for_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "VCG_JOB_CPU_TOKENS",
        "VCG_JOB_MEMORY_GB",
        "VCG_JOB_MEMORY_ALLOCATION_GB",
        "VCG_JOB_OS_RESERVE_GB",
    ):
        monkeypatch.delenv(name, raising=False)

    policy = _apply_job_resource_environment(
        ResourceProfile(cpu_tokens=4, memory_gb=14.0),
        stage="extract-features",
        os_reserve_gb=6.0,
    )

    assert os.environ["VCG_JOB_CPU_TOKENS"] == "4"
    assert os.environ["VCG_JOB_MEMORY_ALLOCATION_GB"] == "14.0"
    assert float(os.environ["VCG_JOB_MEMORY_GB"]) == (
        policy.duckdb_memory_limit_gb
    )
    assert policy.duckdb_memory_limit_gb < policy.requested_memory_gb


def test_supervisor_uses_guided_runs_active_stage_for_pause_policy(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.save()
    store = JobStore.for_project(config)
    supervisor = LocalJobSupervisor(
        config,
        memory_probe=lambda: MemoryStatus(32.0, 7.0, 78.1, 0.75),
    )
    record = store.create(
        stage="study-run",
        project_file=config.project_file,
        job_id="stage-aware-memory",
        resource_profile=ResourceProfile(
            cpu_tokens=1,
            memory_gb=14.0,
            writer_namespace="guided-study",
        ),
    )
    store.claim(record.spec.job_id, os.getpid())
    control = JobControl(store, record.spec.job_id)

    light_policy = stage_memory_safety_policy(
        "report",
        reserve_gb=6.0,
        requested_memory_gb=14.0,
    )
    control.set_runtime_stage(
        "report",
        memory_policy=light_policy.to_dict(),
    )
    for _ in range(MEMORY_PRESSURE_SAMPLES + 1):
        supervisor.run_once()
    assert store.get(record.spec.job_id).state == JobState.RUNNING

    supervisor._memory_probe = lambda: MemoryStatus(32.0, 3.1, 90.3, 0.75)
    waveform_policy = stage_memory_safety_policy(
        "extract-features",
        reserve_gb=6.0,
        requested_memory_gb=14.0,
    )
    control.set_runtime_stage(
        "extract-features",
        memory_policy=waveform_policy.to_dict(),
    )
    for _ in range(MEMORY_PRESSURE_SAMPLES):
        supervisor.run_once()

    assert store.get(record.spec.job_id).state == JobState.PAUSING
    event = [
        item
        for item in store.events(record.spec.job_id)
        if item["event_type"] == "memory_pressure_pause_requested"
    ][0]
    assert event["payload"]["active_stage"] == "extract-features"
    assert event["payload"]["workload_class"] == "waveform"
    assert event["payload"]["swap_used_gb"] == 0.75
    assert event["payload"]["duckdb_memory_limit_gb"] < 14.0
