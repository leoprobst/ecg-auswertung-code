from types import SimpleNamespace

from vcg_biomarker.gui.run_progress import CalibratedJobProgress
from vcg_biomarker.gui.widgets.mimic_project_widget import _job_eta_text
from vcg_biomarker.mimic.live_progress import RunProgress, OPEN_ENDED_STAGES
from vcg_biomarker.mimic.models import JobState


def test_unknown_calibrated_eta_never_falls_back_to_endpoint_fraction():
    record = SimpleNamespace(state=JobState.RUNNING, progress_current=29,
        progress_total=34, started_at="2026-09-08T00:00:00+00:00")
    estimate = CalibratedJobProgress(0., None, 0, 1, estimate_has_unmodelled_work=True)
    assert _job_eta_text(record, estimate) == "Gesamtrestarbeit unbekannt; noch nicht belastbar"


def test_partial_eta_is_not_presented_as_a_lower_bound():
    record = SimpleNamespace(state=JobState.RUNNING)
    estimate = CalibratedJobProgress(0., 300., 0, 2, estimate_has_unmodelled_work=True)
    text = _job_eta_text(record, estimate)
    assert "Planungswert" in text and "unbekannte Restarbeit" in text
    assert "mindestens" not in text


def test_backend_discloses_missing_work_without_inventing_interval():
    progress = RunProgress("synthetic", ["ingest", sorted(OPEN_ENDED_STAGES)[0]])
    payload = progress.to_dict()
    assert payload["estimate_has_unmodelled_work"]
    assert not payload["estimate_is_lower_bound"]
    assert payload["remaining_low_seconds"] is None
    assert payload["remaining_high_seconds"] is None
    assert payload["estimate_status"] == "partial_planning_estimate"
