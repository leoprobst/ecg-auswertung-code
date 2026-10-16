"""Two defects that stopped the A147 full run (27.-30.09.2026).

1. A worker that paused itself exits with 75. When a resume re-queued the job
   before the supervisor reaped that exit, the reap treated it as a crash and
   put the queued job back to PAUSED, where nothing resumed it.
2. Count columns were written as int64 in partitions without a missing value
   and as double elsewhere; the differing schemas forced every full read into
   DuckDB's memory-hungry ``union_by_name`` path.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vcg_biomarker.mimic import supervisor as supervisor_module
from vcg_biomarker.mimic.feature_extraction import _cast_integer_counts_to_float
from vcg_biomarker.mimic.models import JobState
from vcg_biomarker.mimic.supervisor import WORKER_PAUSE_EXIT_CODE, LocalJobSupervisor


class _Process:
    def __init__(self, pid, exit_code):
        self.pid = pid
        self._exit_code = exit_code

    def poll(self):
        return self._exit_code


class _Store:
    def __init__(self, state):
        self.state = state
        self.transitions: list[tuple[str, JobState]] = []
        self.events: list[str] = []

    def get(self, job_id):
        return SimpleNamespace(state=self.state)

    def transition(self, job_id, state, **kwargs):
        self.transitions.append((job_id, state))
        self.state = state

    def append_event(self, job_id, event_type, message, payload):
        self.events.append(event_type)


def _reap(state, exit_code, monkeypatch):
    monkeypatch.setattr(supervisor_module, "_terminate_process_pool_children", lambda pid: [])
    supervisor = LocalJobSupervisor.__new__(LocalJobSupervisor)
    supervisor.store = _Store(state)
    supervisor._children = {"job": _Process(4242, exit_code)}
    supervisor._reap_children()
    return supervisor.store


def test_paused_exit_after_resume_keeps_the_job_queued(monkeypatch):
    store = _reap(JobState.QUEUED, WORKER_PAUSE_EXIT_CODE, monkeypatch)
    assert store.transitions == []
    assert store.state == JobState.QUEUED
    assert "worker_paused_exit_after_resume" in store.events


def test_a_real_crash_of_a_queued_or_running_job_is_still_recovered_as_paused(monkeypatch):
    assert _reap(JobState.QUEUED, 1, monkeypatch).transitions == [("job", JobState.PAUSED)]
    assert _reap(JobState.RUNNING, WORKER_PAUSE_EXIT_CODE, monkeypatch).transitions == [
        ("job", JobState.PAUSED)
    ]


def test_partitions_with_and_without_missing_counts_share_one_schema(tmp_path):
    complete = pd.DataFrame({"study_id": [1, 2], "subject_id": [7, 8],
                             "detected_beat_count": [10, 12], "flag": [True, False]})
    gapped = pd.DataFrame({"study_id": [3, 4], "subject_id": [9, 9],
                           "detected_beat_count": [11, np.nan], "flag": [False, True]})
    schemas = []
    for index, frame in enumerate((complete, gapped)):
        _cast_integer_counts_to_float(frame)
        path = tmp_path / f"part-{index}.parquet"
        frame.to_parquet(path)
        schemas.append({field.name: str(field.type) for field in pq.read_schema(path)
                        if not field.name.startswith("__")})
    assert schemas[0] == schemas[1]
    assert schemas[0]["detected_beat_count"] == "double"
    assert schemas[0]["study_id"] == "int64" and schemas[0]["subject_id"] == "int64"
    assert schemas[0]["flag"] == "bool"
