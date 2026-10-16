import json
import sys
from types import SimpleNamespace

import pytest

from vcg_biomarker.gui.run_progress import calibrated_job_progress
from vcg_biomarker.gui.widgets.mimic_project_widget import _job_eta_text, _job_progress_text
from vcg_biomarker.mimic.models import JobState
from vcg_biomarker.mimic.resources import recommend_dynamic_workers


def test_gui_inventory_uses_actual_development_counts_only(tmp_path):
    from vcg_biomarker.gui.run_progress import development_record_count
    assert development_record_count(tmp_path) is None
    payload = {"splits": [{"split": "train", "ecg_count": 120},
                          {"split": "validation", "ecg_count": 35},
                          {"split": "locked_test", "ecg_count": 999}]}
    (tmp_path / "ecg_master_index_summary.json").write_text(json.dumps(payload))
    assert development_record_count(tmp_path) == 155


def test_cache_hits_do_not_inflate_compute_rate_and_gui_uses_estimate(tmp_path):
    record = SimpleNamespace(state=JobState.RUNNING, heartbeat_at=None,
        spec=SimpleNamespace(job_id="test", arguments={"study_spec": {"stage_ids": ["extract-features"]}}))
    payload = {"job_id": "test", "current_stage": "extract-features",
        "current_stage_fraction": .9,
        "current_stage_message": "Features 900/1,000; Fehler=0, Reused=800, ETA=x",
        "stages": [{"stage": "extract-features", "state": "running", "elapsed_seconds": 100}]}
    (tmp_path / "live_progress.json").write_text(json.dumps(payload))
    estimate = calibrated_job_progress(record, tmp_path)
    assert estimate.fraction == pytest.approx(.9)
    assert estimate.remaining_seconds == pytest.approx(100.)  # not 11 seconds
    assert "Gesamt: 90.00 %" in _job_progress_text(record, estimate)
    assert "Planungswert" in _job_eta_text(record, estimate)
    payload["current_stage_message"] = "Features 900/1,000; Reused=900"
    (tmp_path / "live_progress.json").write_text(json.dumps(payload))
    estimate = calibrated_job_progress(record, tmp_path)
    assert estimate.remaining_seconds is None
    assert "unbekannt" in _job_eta_text(record, estimate)


def test_capacity_down_then_up_respects_reserve_and_admitted_ceiling(monkeypatch):
    telemetry = SimpleNamespace(available=14 * 1024**3, load=0.)
    fake = SimpleNamespace(Error=RuntimeError, cpu_count=lambda **kw: 16,
        cpu_percent=lambda **kw: telemetry.load,
        virtual_memory=lambda: telemetry,
        Process=lambda: SimpleNamespace(children=lambda **kw: []))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    from vcg_biomarker.mimic import resources
    monkeypatch.setattr(resources, "commit_available_gb", lambda: None)  # physical path only
    assert recommend_dynamic_workers(6, current_workers=0, reserve_gb=6) == 4
    telemetry.available = 8 * 1024**3
    assert recommend_dynamic_workers(6, current_workers=0, reserve_gb=6) == 1
    telemetry.available = 30 * 1024**3
    assert recommend_dynamic_workers(6, current_workers=0) == 6
    telemetry.load = 95.
    assert recommend_dynamic_workers(6, current_workers=0) == 1


def test_missing_ram_telemetry_does_not_authorize_expansion(monkeypatch):
    def failed():
        raise OSError("sample failed")
    fake = SimpleNamespace(Error=RuntimeError, cpu_count=lambda **kw: 16,
        cpu_percent=lambda **kw: 0., Process=failed)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert recommend_dynamic_workers(6, current_workers=2) == 1


def test_own_cpu_load_is_not_mistaken_for_foreign_contention(monkeypatch):
    from vcg_biomarker.mimic import resources

    state = {"seconds": 0.0, "cpu": 0.0}
    process = SimpleNamespace(pid=123, create_time=lambda: 1.0,
        children=lambda **kwargs: [],
        cpu_times=lambda: SimpleNamespace(user=state["cpu"], system=0.0))
    def cpu_percent(**kwargs):
        state["seconds"] += 2.0
        state["cpu"] += 16.0  # Eight logical CPUs, all belonging to this job.
        return 50.0
    fake = SimpleNamespace(Error=RuntimeError, cpu_count=lambda **kwargs: 16,
        cpu_percent=cpu_percent, Process=lambda: process,
        virtual_memory=lambda: SimpleNamespace(available=40 * 1024**3))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(resources, "commit_available_gb", lambda: None)  # physical path only
    monkeypatch.setattr(resources.time, "monotonic", lambda: state["seconds"])
    assert recommend_dynamic_workers(15, current_workers=0, reserve_gb=2) == 15


def test_reused_validation_pool_really_changes_active_fit_count(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import time
    import pandas as pd
    from vcg_biomarker.mimic import resources, validation
    state = {"active": 0, "peak": 0, "limit": 1}
    lock = Lock()
    def fit(**arguments):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(.02)
            return arguments["feature_id"]
        finally:
            with lock:
                state["active"] -= 1
    monkeypatch.setattr(validation, "_evaluate_single_marker", fit)
    monkeypatch.setattr(resources, "recommend_dynamic_workers", lambda *a, **kw: state["limit"])
    features = tuple(f"marker_{i}" for i in range(8))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for limit in (1, 3, 2):
            state.update(limit=limit, peak=0)
            result = validation._evaluate_endpoint_markers(endpoint_id="synthetic",
                feature_ids=features, panel_map=dict.fromkeys(features, "sentinel"),
                train=pd.DataFrame(index=range(3)), validation=pd.DataFrame(index=range(3)),
                specification=SimpleNamespace(), bootstrap_iterations=0,
                worker_count=6, executor=pool)
            assert result == list(features)
            assert state["peak"] == limit


def test_commit_limit_caps_available_memory_for_admission(monkeypatch):
    """R4: free RAM behind an exhausted commit limit is not available."""
    from vcg_biomarker.mimic import resources
    telemetry = SimpleNamespace(available=30 * 1024**3)
    fake = SimpleNamespace(Error=RuntimeError, cpu_count=lambda **kw: 16,
        cpu_percent=lambda **kw: 0., virtual_memory=lambda: telemetry,
        Process=lambda: SimpleNamespace(children=lambda **kw: []))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(resources, "commit_available_gb", lambda: 6.0)
    # 6 GB commit - 2 GB reserve = 4 GB -> two 2-GB shares, not twelve.
    assert recommend_dynamic_workers(15, current_workers=0, reserve_gb=2) == 2


def test_memory_status_reports_the_smaller_of_ram_and_commit(monkeypatch):
    from vcg_biomarker.mimic import resources
    monkeypatch.setattr(resources, "commit_available_gb", lambda: 1.5)
    status = resources.detect_memory_status()
    assert status is not None
    assert status.available_gb == min(status.physical_available_gb, 1.5)
    assert status.commit_available_gb == 1.5


def test_commit_probe_returns_a_plausible_value_on_windows():
    import os
    from vcg_biomarker.mimic import resources
    value = resources.commit_available_gb()
    if os.name == "nt":
        assert value is not None and 0 < value < 4096
    else:
        assert value is None
