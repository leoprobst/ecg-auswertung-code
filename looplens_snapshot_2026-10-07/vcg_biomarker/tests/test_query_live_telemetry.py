import threading

import pytest

from vcg_biomarker.mimic.run_log import RunLog, read_records, summarize


def test_begin_is_visible_before_query_finishes_and_summary_counts_rows_once(tmp_path):
    log = RunLog.for_run(tmp_path)
    log.stage = "build-evidence"
    with log.query_scope("context.parquet") as outcome:
        records = read_records(log.path)
        assert [r["event"] for r in records] == ["begin"]
        assert "duration_seconds" not in records[0]
        running = log.active_queries()
        assert len(running) == 1 and running[0]["elapsed_seconds"] >= 0
        assert running[0]["query_id"] == records[0]["query_id"]
        running[0]["label"] = "external_mutation"
        assert log.active_queries()[0]["label"] == "context.parquet"
        outcome["rows"] = 42
    records = read_records(log.path)
    assert records[0]["query_id"] == records[1]["query_id"]
    assert log.active_queries() == []
    assert summarize(records)["stages"][0]["rows"] == 42


def test_same_label_concurrent_queries_have_distinct_ids_and_clear_on_failure(tmp_path):
    log = RunLog.for_run(tmp_path)
    ready = threading.Event()
    release = threading.Event()
    errors = []

    def worker():
        try:
            with log.query_scope("same"):
                ready.set()
                assert release.wait(5)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert ready.wait(5)
        with pytest.raises(ValueError, match="synthetic"):
            with log.query_scope("same"):
                queries = log.active_queries()
                assert len({query["query_id"] for query in queries}) == 2
                raise ValueError("synthetic")
        assert len(log.active_queries()) == 1
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert not log.active_queries()
    records = read_records(log.path)
    for query_id in {record["query_id"] for record in records}:
        events = [record["event"] for record in records if record["query_id"] == query_id]
        assert events in (["begin", "end"], ["begin", "failed"])


def test_periodic_sample_identifies_running_query(tmp_path, monkeypatch):
    log = RunLog.for_run(tmp_path)
    observed = threading.Event()
    original_write = log.write

    def capture(kind, **fields):
        original_write(kind, **fields)
        if kind == "sample" and fields.get("active_queries"):
            observed.set()

    monkeypatch.setattr(log, "write", capture)
    with log.query_scope("long-context"):
        log.start_sampling(.01)
        try:
            assert observed.wait(5)
        finally:
            log.stop_sampling()
    sample = next(row for row in read_records(log.path) if row["kind"] == "sample")
    assert sample["active_queries"][0]["label"] == "long-context"
    assert "eta_seconds" not in sample["active_queries"][0]
