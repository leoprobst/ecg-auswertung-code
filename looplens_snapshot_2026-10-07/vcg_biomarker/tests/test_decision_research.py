from datetime import datetime, date, timezone

import pytest

from vcg_biomarker.mimic.decision_research import (
    TimeInterval, temporal_relation, available_at_decision, reference_label, tracking_requirements,
)


def instant(hour):
    dt = datetime(2026, 1, 1, hour, tzinfo=timezone.utc)
    return TimeInterval(dt, dt)


def test_same_day_procedure_does_not_establish_after_ecg():
    day = TimeInterval.calendar_day(date(2026, 1, 1), timezone.utc)
    assert temporal_relation(day, instant(12)) == "ambiguous"
    assert temporal_relation(instant(10), instant(12)) == "before"


def test_result_availability_not_collection_controls_prediction():
    assert not available_at_decision(instant(13), instant(12))
    assert not available_at_decision(None, instant(12))
    assert available_at_decision(instant(11), instant(12))


@pytest.mark.parametrize("kind", ["ecg_annotation", "machine_report", "administrative_code"])
def test_dependent_sources_never_become_independent_truth(kind):
    result = reference_label(observed_state="positive", reference_type=kind,
                             evidence_id="id", independent=True, verified=True)
    assert result["state"] == "unknown"


def test_no_procedure_is_not_negative_occlusion():
    result = reference_label(observed_state="unknown", reference_type="angiography",
                             evidence_id="", independent=True, verified=False)
    assert result["state"] == "unknown"


def test_tracking_does_not_claim_missing_echo_or_drug_data():
    profiles = tracking_requirements(["ecg_time", "dose"])
    assert "culprit_adjudication" in profiles["pci"]["missing_fields"]
    assert "actual_administration_time" in profiles["pretreatment"]["missing_fields"]
    assert all(not p["clinical_claim_allowed"] for p in profiles.values())


def test_future_outcome_is_allowed_but_future_predictor_is_not():
    args = dict(observed_state="positive", reference_type="imaging", evidence_id="actual_id",
                independent=True, verified=True, design="prediction",
                result_available=instant(13), decision=instant(12))
    assert reference_label(**args)["state"] == "positive"
    assert reference_label(**args, as_predictor=True)["state"] == "unknown"
