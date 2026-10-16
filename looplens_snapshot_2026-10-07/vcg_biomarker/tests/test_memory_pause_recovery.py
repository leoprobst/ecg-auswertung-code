"""A memory pause has to end by itself, and only the machine's own pause.

Pausing under memory pressure is a safety feature and works. Staying paused
afterwards is not: the resume was only ever wired to the interface button and
the CLI, so a month-long run stopped at the first pressure spike and waited
for somebody to notice. On 2026-08-26 that happened five times before ten in
the morning, and the machine had 16 GB free while the run sat idle.

The two ways this can go wrong are opposite, and both are tested here:

* not resuming leaves the run idle, which is the whole problem;
* resuming a pause a *person* asked for overrides them, which is worse than
  the problem.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vcg_biomarker.mimic.supervisor import (
    MEMORY_PAUSE_MARKER,
    MEMORY_READY_SAMPLES,
    LocalJobSupervisor,
)


class _Store:
    def __init__(self, records):
        self._records = list(records)
        self.resumed: list[str] = []
        self.events: list[tuple[str, str]] = []

    def list(self, states=None):
        return list(self._records)

    def request_resume(self, job_id):
        self.resumed.append(job_id)
        return SimpleNamespace(job_id=job_id)

    def append_event(self, job_id, event_type, message, payload):
        self.events.append((job_id, event_type))


def _record(job_id, message, *, memory_gb=10.0):
    return SimpleNamespace(
        spec=SimpleNamespace(
            job_id=job_id,
            stage="validate",
            arguments={},
            resource_profile=SimpleNamespace(
                memory_gb=memory_gb,
                cpu_tokens=4,
                interactive=False,
            ),
        ),
        error_message=message,
        state="paused",
    )


def _supervisor(records, monkeypatch):
    supervisor = LocalJobSupervisor.__new__(LocalJobSupervisor)
    supervisor.store = _Store(records)
    supervisor.config = SimpleNamespace(
        os_memory_reserve_gb=6.0, project_file="p.json"
    )
    supervisor._memory_resume_samples = {}
    # __new__ deliberately bypasses the constructor; represent the initial
    # coordinator state (no outstanding drain), rather than omitting it.
    supervisor._resource_grants = SimpleNamespace(drain_in_progress=lambda *args: False)
    monkeypatch.setattr(
        LocalJobSupervisor, "_active_runtime_stage", lambda self, r: "validate"
    )
    return supervisor


def _memory(available_gb):
    return SimpleNamespace(
        available_gb=available_gb,
        total_gb=32.0,
        used_percent=50.0,
        swap_used_gb=0.0,
    )


def _drive(supervisor, available_gb, times=MEMORY_READY_SAMPLES):
    for _ in range(times):
        supervisor._resume_memory_paused_jobs(_memory(available_gb))


# ------------------------------------------------------------ the problem


def test_an_automatic_pause_is_taken_back_once_memory_returns(monkeypatch):
    record = _record("j1", f"{MEMORY_PAUSE_MARKER}: 2.10 GB verfuegbar")
    supervisor = _supervisor([record], monkeypatch)
    _drive(supervisor, 20.0)
    assert supervisor.store.resumed == ["j1"]
    assert ("j1", "memory_pressure_resume_requested") in supervisor.store.events


def test_resuming_waits_for_several_consecutive_samples(monkeypatch):
    """One lucky reading must not restart a job into the same pressure."""

    record = _record("j1", f"{MEMORY_PAUSE_MARKER}: 2.10 GB verfuegbar")
    supervisor = _supervisor([record], monkeypatch)
    _drive(supervisor, 20.0, times=MEMORY_READY_SAMPLES - 1)
    assert supervisor.store.resumed == []
    _drive(supervisor, 20.0, times=1)
    assert supervisor.store.resumed == ["j1"]


def test_a_dip_below_the_threshold_restarts_the_count(monkeypatch):
    record = _record("j1", f"{MEMORY_PAUSE_MARKER}: 2.10 GB verfuegbar")
    supervisor = _supervisor([record], monkeypatch)
    _drive(supervisor, 20.0, times=MEMORY_READY_SAMPLES - 1)
    _drive(supervisor, 1.0, times=1)          # pressure again
    _drive(supervisor, 20.0, times=MEMORY_READY_SAMPLES - 1)
    assert supervisor.store.resumed == []
    _drive(supervisor, 20.0, times=1)
    assert supervisor.store.resumed == ["j1"]


# ------------------------------------------------------- the worse failure


def test_a_pause_a_person_asked_for_is_never_taken_back(monkeypatch):
    """The machine must not overrule the operator."""

    record = _record("j1", "Pause requested at an atomic checkpoint")
    supervisor = _supervisor([record], monkeypatch)
    _drive(supervisor, 30.0, times=MEMORY_READY_SAMPLES * 2)
    assert supervisor.store.resumed == []


def test_a_pause_without_any_reason_is_left_alone(monkeypatch):
    record = _record("j1", "")
    supervisor = _supervisor([record], monkeypatch)
    _drive(supervisor, 30.0, times=MEMORY_READY_SAMPLES * 2)
    assert supervisor.store.resumed == []


# ------------------------------------------------------------- robustness


def test_pressure_still_present_does_not_resume(monkeypatch):
    record = _record("j1", f"{MEMORY_PAUSE_MARKER}: 2.10 GB verfuegbar")
    supervisor = _supervisor([record], monkeypatch)
    _drive(supervisor, 2.5, times=MEMORY_READY_SAMPLES * 2)
    assert supervisor.store.resumed == []


def test_an_unreadable_memory_probe_resumes_nothing(monkeypatch):
    record = _record("j1", f"{MEMORY_PAUSE_MARKER}: 2.10 GB verfuegbar")
    supervisor = _supervisor([record], monkeypatch)
    for _ in range(MEMORY_READY_SAMPLES * 2):
        supervisor._resume_memory_paused_jobs(None)
    assert supervisor.store.resumed == []


def test_a_failing_resume_does_not_stop_the_supervisor(monkeypatch):
    """A job cancelled meanwhile must not take the supervisor down."""

    record = _record("j1", f"{MEMORY_PAUSE_MARKER}: 2.10 GB verfuegbar")
    supervisor = _supervisor([record], monkeypatch)

    def boom(job_id):
        raise RuntimeError("job is gone")

    supervisor.store.request_resume = boom
    _drive(supervisor, 20.0)
    assert supervisor.store.events == []


def test_the_pause_message_and_the_resume_check_use_one_marker():
    """Matching on free text would drift; both sides read the constant."""

    import inspect

    from vcg_biomarker.mimic import supervisor as module

    source = inspect.getsource(module)
    # The literal appears once, in the constant itself.
    assert source.count('"Automatische Pause wegen kritischem') == 1
    assert "MEMORY_PAUSE_MARKER" in source


def test_recovery_resumes_only_one_job_then_requires_fresh_samples(monkeypatch):
    records = [_record(name, MEMORY_PAUSE_MARKER) for name in ("j1", "j2")]
    supervisor = _supervisor(records, monkeypatch)
    supervisor.config.os_memory_reserve_gb = 2.0
    _drive(supervisor, 3.0)
    assert supervisor.store.resumed == ["j1"]
    supervisor.store._records = records[1:]
    _drive(supervisor, 3.0, times=MEMORY_READY_SAMPLES - 1)
    assert supervisor.store.resumed == ["j1"]
    _drive(supervisor, 3.0, times=1)
    assert supervisor.store.resumed == ["j1", "j2"]


def test_pending_memory_pause_does_not_cascade_to_other_jobs(monkeypatch):
    from vcg_biomarker.mimic.jobs import JobState

    record = _record("j1", MEMORY_PAUSE_MARKER)
    record.state = JobState.PAUSING
    other = _record("j2", "")
    other.state = JobState.RUNNING
    supervisor = _supervisor([record, other], monkeypatch)
    supervisor._low_memory_samples = {"j2": 100}
    for _ in range(10):
        assert supervisor._guard_running_jobs_for_memory([record, other], _memory(2.0))
    assert supervisor._low_memory_samples == {}


def test_pressure_selects_one_job_at_exact_floor_then_resamples(monkeypatch):
    from vcg_biomarker.mimic.models import JobState
    from vcg_biomarker.mimic.supervisor import MEMORY_PRESSURE_SAMPLES

    records = [_record(name, "") for name in ("j1", "j2")]
    for record in records:
        record.state = JobState.RUNNING
        record.spec.priority = 0
    supervisor = _supervisor(records, monkeypatch)
    supervisor.config.os_memory_reserve_gb = 2.0
    supervisor._low_memory_samples = {}
    supervisor._trace = lambda message: None
    pauses = []
    supervisor.store.request_pause = lambda job_id, **kwargs: pauses.append(job_id)
    for _ in range(MEMORY_PRESSURE_SAMPLES):
        supervisor._guard_running_jobs_for_memory(records, _memory(2.0))
    assert pauses == ["j1"]
    assert supervisor._low_memory_samples == {}
    remaining = records[1:]
    for _ in range(MEMORY_PRESSURE_SAMPLES - 1):
        supervisor._guard_running_jobs_for_memory(remaining, _memory(2.0))
    assert pauses == ["j1"]
    supervisor._guard_running_jobs_for_memory(remaining, _memory(2.5))
    assert supervisor._low_memory_samples == {}


@pytest.mark.parametrize("stage", ["extract-features", "validate", "build-evidence", "study-run", "ingest", "unknown"])
def test_two_gib_reserve_is_literal_even_for_large_allocations(stage):
    from vcg_biomarker.mimic.resources import stage_memory_safety_policy

    policy = stage_memory_safety_policy(stage, reserve_gb=2.0, requested_memory_gb=24.0)
    assert policy.critical_available_gb == 2.0
    assert policy.recovery_available_gb == 3.0
