from types import SimpleNamespace

import pytest

from vcg_biomarker.gui.widgets.mimic_project_widget import _job_progress_text, _job_eta_text
from vcg_biomarker.mimic.models import JobState


def record(stage, current=4, total=4, state=JobState.RUNNING):
    return SimpleNamespace(spec=SimpleNamespace(stage=stage), progress_current=current,
        progress_total=total, state=state, progress_message="Baue Kontextmatrix",
        started_at="2026-09-12T03:54:28+00:00")


@pytest.mark.parametrize("stage", ["attest-event-evidence", "build-evidence", "validate", "unknown"])
def test_completed_subcounter_never_implies_completed_job(stage):
    job = record(stage)
    progress = _job_progress_text(job)
    assert "4/4" in progress and "unbekannt" in progress
    assert "%" not in progress
    assert "unbekannt" in _job_eta_text(job)
    job.state = JobState.COMPLETED
    assert "100.0 %" in _job_progress_text(job)
    assert _job_eta_text(job) == "fertig"


def test_uniform_records_keep_fraction_but_end_counter_is_not_completion_eta():
    job = record("extract-features", current=2)
    assert "50.0 %" in _job_progress_text(job)
    job.progress_current = 4
    assert "%" not in _job_progress_text(job)
    assert "ausstehend" in _job_progress_text(job)
    assert "Abschlussdauer" in _job_eta_text(job)


def test_queue_state_does_not_display_stale_subcounter():
    assert _job_progress_text(record("attest-event-evidence", state=JobState.QUEUED)) == "Noch nicht begonnen"
