from types import SimpleNamespace

import pytest

from vcg_biomarker.mimic.run_log import active_log, profile_query, read_records
from vcg_biomarker.mimic_job_worker import _standalone_job_telemetry


def test_standalone_attempts_append_with_distinct_identity_and_stop_sampler(tmp_path):
    config = SimpleNamespace(jobs_root=tmp_path)
    spec = SimpleNamespace(stage="attest-event-evidence", job_id="fixture")
    ids = []
    for _ in range(2):
        with _standalone_job_telemetry(config, spec) as log:
            assert active_log() is log
            assert log._sampler.is_alive()
            with profile_query("synthetic-context") as result:
                result["rows"] = 7
            ids.append(log.attempt_id)
        assert log._sampler is None and active_log() is None
    records = read_records(log.path)
    assert len(set(ids)) == 2
    assert {r["attempt_id"] for r in records} == set(ids)
    assert all(r["job_id"] == "fixture" for r in records)
    assert len([r for r in records if r["kind"] == "query" and r["event"] == "end"]) == 2


def test_standalone_failure_is_logged_and_propagated(tmp_path):
    config = SimpleNamespace(jobs_root=tmp_path)
    spec = SimpleNamespace(stage="build-evidence", job_id="../untrusted")
    with pytest.raises(ValueError, match="synthetic"):
        with _standalone_job_telemetry(config, spec) as log:
            assert log.path.resolve().is_relative_to(tmp_path.resolve())
            raise ValueError("synthetic")
    assert log._sampler is None and active_log() is None
    assert any(row.get("event") == "failed" for row in read_records(log.path))


def test_guided_run_does_not_create_a_second_sampler(tmp_path):
    config = SimpleNamespace(jobs_root=tmp_path)
    spec = SimpleNamespace(stage="study-run", job_id="fixture")
    with _standalone_job_telemetry(config, spec) as log:
        assert log is None and active_log() is None
    assert not list(tmp_path.iterdir())
