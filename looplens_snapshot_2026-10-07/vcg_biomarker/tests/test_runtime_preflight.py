import json
from types import SimpleNamespace

import psutil
import pytest

from vcg_biomarker.mimic import runtime_preflight as guard
from vcg_biomarker.mimic.jobs import JobStore
from vcg_biomarker.mimic.models import JobSpec, JobState
from vcg_biomarker.mimic.runtime_snapshot import create_runtime_snapshot


def mock_process(monkeypatch, folder):
    monkeypatch.setattr(guard.psutil, "Process", lambda _: SimpleNamespace(
        cwd=lambda: str(folder),
        cmdline=lambda: ["python", "-m", "vcg_biomarker.mimic_supervisor"],
    ))


def test_snapshot_integrity_and_legacy_rejection(tmp_path, monkeypatch):
    package = tmp_path / "source" / "vcg_biomarker"
    package.mkdir(parents=True)
    for name in ("mimic_supervisor.py", "mimic_job_worker.py"):
        (package / name).write_text("# frozen\n")
    root = tmp_path / "snapshots"
    runtime = create_runtime_snapshot(package.parent, root)
    mock_process(monkeypatch, runtime)
    guard.require_isolated_study_supervisor({"pid": 123}, root)
    (runtime / "vcg_biomarker/mimic_supervisor.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        guard.require_isolated_study_supervisor({"pid": 123}, root)
    mock_process(monkeypatch, package.parent)
    with pytest.raises(RuntimeError, match="development code"):
        guard.require_isolated_study_supervisor({"pid": 123}, root)


def test_missing_or_unreadable_process(tmp_path, monkeypatch):
    guard.require_isolated_study_supervisor({}, tmp_path)

    def missing(_):
        raise psutil.NoSuchProcess(123)

    monkeypatch.setattr(guard.psutil, "Process", missing)
    guard.require_isolated_study_supervisor({"pid": 123}, tmp_path)

    def denied(_):
        raise psutil.AccessDenied(123)

    monkeypatch.setattr(guard.psutil, "Process", denied)
    with pytest.raises(RuntimeError, match="Studienstart gesperrt"):
        guard.require_isolated_study_supervisor({"pid": 123}, tmp_path)


@pytest.mark.parametrize("payload", [[], {}, {"version": 1, "files": {}},
    {"version": 1, "files": {"arbitrary.py": "fake"}}])
def test_invalid_manifest_abstains(tmp_path, monkeypatch, payload):
    runtime = tmp_path / "session"
    runtime.mkdir()
    (runtime / "runtime_manifest.json").write_text(json.dumps(payload))
    mock_process(monkeypatch, runtime)
    with pytest.raises(RuntimeError):
        guard.require_isolated_study_supervisor({"pid": 123}, tmp_path)


def test_rejection_precedes_queue_write_and_retry_resume(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.sqlite")
    spec = JobSpec("study", "study-run", str(tmp_path / "project.json"))
    mock_process(monkeypatch, tmp_path / "development")
    monkeypatch.setattr(store, "supervisor_status", lambda: {"pid": 123})
    with pytest.raises(RuntimeError, match="Studienstart gesperrt"):
        store.enqueue(spec)
    assert not store.list()
    monkeypatch.setattr(store, "supervisor_status", lambda: {})
    store.enqueue(spec)
    store.transition("study", JobState.RUNNING)
    store.transition("study", JobState.FAILED)
    monkeypatch.setattr(store, "supervisor_status", lambda: {"pid": 123})
    with pytest.raises(RuntimeError):
        store.retry("study")
    assert store.get("study").state == JobState.FAILED
    store.transition("study", JobState.QUEUED)
    store.transition("study", JobState.PAUSED)
    with pytest.raises(RuntimeError):
        store.request_resume("study")
    assert store.get("study").state == JobState.PAUSED
    # An ongoing event-import workflow is not subjected to a study-start guard.
    store.enqueue(JobSpec("import", "ingest-events", spec.project_file))
    assert store.get("import").state == JobState.QUEUED
