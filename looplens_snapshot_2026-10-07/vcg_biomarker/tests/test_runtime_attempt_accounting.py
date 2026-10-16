import json

import pytest

from vcg_biomarker.mimic.pilot_go_no_go import _runtime_summary
from vcg_biomarker.gui.run_progress import _calibration


def test_failed_attempt_cost_is_not_a_future_execution_prior(tmp_path):
    path = tmp_path / "run.jsonl"
    events = []
    for event, duration in [("failed", 900), ("end", 100), ("end", 120)]:
        events.extend([{"kind": "stage", "stage": "validate", "event": "begin"},
            {"kind": "reuse", "stage": "validate", "reused": False},
            {"kind": "stage", "stage": "validate", "event": event, "duration_seconds": duration}])
    path.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    result = _runtime_summary(path)
    assert result["stage_duration_seconds"]["validate"] == 1120
    assert result["stage_successful_attempt_seconds"]["validate"] == [100, 120]
    assert result["stage_failed_attempt_seconds"]["validate"] == [900]
    assert _calibration({"runtime_and_memory": result})["validate"] == 110


def test_only_failures_do_not_supply_an_execution_prior(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"kind": "stage", "stage": "validate", "event": "failed",
        "duration_seconds": 200}), encoding="utf-8")
    assert _calibration({"runtime_and_memory": _runtime_summary(path)}) == {}


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan"), "bad"])
def test_invalid_durations_are_not_calibration(value):
    assert _calibration({"runtime_and_memory": {"stage_successful_attempt_seconds": {"stage": [value]}}}) == {}


def test_reuse_and_unknown_attempts_do_not_shrink_compute_prior(tmp_path):
    path = tmp_path / "run.jsonl"
    events = []
    for reused, duration in [(False, 100), (True, 1), (None, 2)]:
        events.append({"kind": "stage", "stage": "validate", "event": "begin"})
        if reused is not None:
            events.append({"kind": "reuse", "stage": "validate", "reused": reused})
        events.append({"kind": "stage", "stage": "validate", "event": "end", "duration_seconds": duration})
    path.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    result = _runtime_summary(path)
    assert result["stage_reuse_reported_attempt_seconds"] == {"validate": [1]}
    assert result["stage_unknown_cache_attempt_seconds"] == {"validate": [2]}
    assert _calibration({"runtime_and_memory": result}) == {"validate": 100}


def test_reuse_decisions_do_not_cross_attempt_identity(tmp_path):
    path = tmp_path / "run.jsonl"
    events = [{"kind": "stage", "event": "begin", "attempt_id": "new"},
        {"kind": "reuse", "reused": False, "attempt_id": "old"},
        {"kind": "stage", "event": "end", "duration_seconds": 4, "attempt_id": "new"}]
    path.write_text("\n".join(json.dumps({"stage": "validate", **x}) for x in events), encoding="utf-8")
    assert _calibration({"runtime_and_memory": _runtime_summary(path)}) == {}
