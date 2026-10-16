import copy
import json
from pathlib import Path
import pytest
from vcg_biomarker.core.delineation import benchmark_suite as m


def test_qtdb_record_cache_bound_to_source_and_implementation(tmp_path, monkeypatch):
    source = tmp_path / "record.dat"
    source.write_bytes(b"synthetic waveform fixture")
    calls = []
    def compute(root, *, record_names):
        calls.append(record_names)
        return {"records_completed": 1, "metric": 7.}
    monkeypatch.setattr(m, "_run_qtdb_boundary_records", compute)
    task = (tmp_path, "record", "implementation-A", tmp_path / "checkpoints")
    first, reused = m._qtdb_record_task(task)
    assert not reused
    second, reused = m._qtdb_record_task(task)
    assert reused and first == second and len(calls) == 1
    source.write_bytes(b"changed waveform fixture")
    assert not m._qtdb_record_task(task)[1]
    assert not m._qtdb_record_task((*task[:2], "implementation-B", task[3]))[1]
    assert len(calls) == 3
    for path in task[3].glob("*.json"):
        payload = json.loads(path.read_text())
        payload["result"]["metric"] = 99
        path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="integrity"):
        m._qtdb_record_task(task)


def test_record_aggregation_preserves_raw_errors_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_record_names", lambda root: ["one", "two"])
    template = m._run_qtdb_boundary_records(tmp_path, record_names=())
    record = copy.deepcopy(template)
    record.update(records_completed=1, windows_processed=3, signal_seconds_processed=12.)
    record["raw_boundary_metrics"]["qrs"].update(reference_events=1, predicted_events=1,
        matched_events=1, peak_errors_ms=[5.])
    monkeypatch.setattr(m, "_qtdb_record_task", lambda task: (copy.deepcopy(record), True))
    result = m.run_qtdb_boundary_benchmark(tmp_path)
    assert result["records_completed"] == 2
    assert result["record_checkpoint_reused"] == 2
    assert result["windows_processed"] == 6
    assert result["signal_seconds_processed"] == 24
    assert result["waves"]["qrs"]["positive_predictive_value"] is None
    assert result["status"] == "complete"
    assert "raw_boundary_metrics" not in result


def test_record_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        m._qtdb_record_task((tmp_path, "../outside", "code", tmp_path / "checkpoints"))
