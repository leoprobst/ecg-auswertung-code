"""The run log has one job: still be there, and still be readable, after.

A month-long study wrote 31,414 events of which 31,133 were free text with no
measurements, and ``supervisor.log`` was empty. When one query then spent 25.9
hours producing 3,357 rows, file modification times were the only evidence
left. So the properties tested here are the ones that failure mode needs:

* it survives process restarts, because a paused-and-resumed run is several
  processes;
* a crash mid-write leaves the earlier records readable;
* it can never end the run it is describing.
"""

from __future__ import annotations

import json
import time

import pytest

from vcg_biomarker.mimic.run_log import (
    RUN_LOG_FILENAME,
    RunLog,
    active,
    note_reuse,
    profile_query,
    read_records,
    render_summary,
    set_active_log,
    summarize,
)
from vcg_biomarker.mimic.storage import StageManifest


def test_stage_records_bracket_the_work(tmp_path):
    log = RunLog.for_run(tmp_path)
    with log.stage_scope("validate", decision="computed", reason="key moved"):
        pass
    records = read_records(tmp_path / RUN_LOG_FILENAME)
    events = [r for r in records if r.get("kind") == "stage"]
    assert [r["event"] for r in events] == ["begin", "end"]
    assert events[0]["decision"] == "computed"
    assert events[0]["reason"] == "key moved"
    assert events[1]["duration_seconds"] is not None


def test_profile_query_uses_active_guided_log(tmp_path, monkeypatch):
    monkeypatch.setenv("VCG_JOB_MEMORY_GB", "7.5")
    log = RunLog.for_run(tmp_path)
    with active(log), log.stage_scope("build-evidence"):
        with profile_query("evidence:labs.parquet") as outcome:
            outcome["rows"] = 123

    queries = [
        item
        for item in read_records(tmp_path / RUN_LOG_FILENAME)
        if item.get("kind") == "query" and item.get("event") == "end"
    ]
    assert len(queries) == 1
    assert queries[0]["stage"] == "build-evidence"
    assert queries[0]["label"] == "evidence:labs.parquet"
    assert queries[0]["rows"] == 123
    assert queries[0]["memory_limit_gb"] == 7.5


def test_the_reuse_reason_is_recorded(tmp_path):
    """The sentence that would have explained a walk-back over 11 stages."""

    log = RunLog.for_run(tmp_path)
    with log.stage_scope(
        "extract-features", decision="skipped", reason="reuse key identical"
    ):
        pass
    summary = summarize(read_records(tmp_path / RUN_LOG_FILENAME))
    entry = summary["stages"][0]
    assert entry["decision"] == "skipped"
    assert entry["reason"] == "reuse key identical"
    assert "reuse key identical" in render_summary(summary)


def test_a_query_records_its_duration_and_rows(tmp_path):
    log = RunLog.for_run(tmp_path)
    with log.query_scope("Laborfenster", memory_limit_gb=8.0) as outcome:
        outcome["rows"] = 3357
    record = [
        r for r in read_records(tmp_path / RUN_LOG_FILENAME)
        if r.get("kind") == "query" and r.get("event") == "end"
    ][0]
    assert record["rows"] == 3357
    assert record["memory_limit_gb"] == 8.0
    assert record["duration_seconds"] is not None


def test_a_failing_stage_records_the_traceback_and_reraises(tmp_path):
    log = RunLog.for_run(tmp_path)
    with pytest.raises(ValueError):
        with log.stage_scope("therapy-response"):
            raise ValueError("heart_rate_bpm not found")
    records = read_records(tmp_path / RUN_LOG_FILENAME)
    errors = [r for r in records if r.get("kind") == "error"]
    assert errors and "heart_rate_bpm" in errors[0]["error"]
    assert "Traceback" in errors[0]["traceback"]
    assert errors[0]["stage"] == "therapy-response"
    assert any(
        r.get("kind") == "stage" and r.get("event") == "failed"
        for r in records
    )


def test_a_failing_query_records_and_reraises(tmp_path):
    log = RunLog.for_run(tmp_path)
    with pytest.raises(RuntimeError):
        with log.query_scope("Paare"):
            raise RuntimeError("binder error")
    record = [
        r for r in read_records(tmp_path / RUN_LOG_FILENAME)
        if r.get("kind") == "query" and r.get("event") == "failed"
    ][0]
    assert record["event"] == "failed"
    assert "binder error" in record["error"]


# ----------------------------------------------------------- survivability


def test_a_second_process_appends_rather_than_truncating(tmp_path):
    """A resumed run is a new process; the first one's record must remain."""

    first = RunLog.for_run(tmp_path)
    with first.stage_scope("ingest"):
        pass
    second = RunLog.for_run(tmp_path)
    with second.stage_scope("build-evidence"):
        pass
    stages = {
        r["stage"]
        for r in read_records(tmp_path / RUN_LOG_FILENAME)
        if r.get("kind") == "stage"
    }
    assert stages == {"ingest", "build-evidence"}


def test_a_truncated_last_line_does_not_hide_the_rest(tmp_path):
    """A crash mid-write is when this file matters most."""

    log = RunLog.for_run(tmp_path)
    with log.stage_scope("ingest"):
        pass
    path = tmp_path / RUN_LOG_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "stage", "eve')
    records = read_records(path)
    assert [r["event"] for r in records if r.get("kind") == "stage"] == [
        "begin",
        "end",
    ]


def test_logging_never_ends_the_run(tmp_path, monkeypatch):
    """A full disk must cost a record, not a month of computation."""

    log = RunLog.for_run(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(type(log.path), "open", boom, raising=False)
    with log.stage_scope("validate"):
        pass  # must not raise
    log.write("sample", rss_gb=1.0)


def test_an_unwritable_directory_disables_the_log_quietly(tmp_path, monkeypatch):
    import pathlib

    def boom(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(pathlib.Path, "mkdir", boom)
    log = RunLog.for_run(tmp_path / "neu")
    assert not log.enabled
    with log.stage_scope("validate"):
        pass


# ---------------------------------------------------------------- sampling


def test_sampling_produces_a_curve_not_a_single_point(tmp_path):
    """One number at the moment of failure does not show what it climbed."""

    log = RunLog.for_run(tmp_path)
    log.start_sampling(interval_seconds=0.02)
    time.sleep(0.15)
    log.stop_sampling()
    samples = [
        r for r in read_records(tmp_path / RUN_LOG_FILENAME)
        if r.get("kind") == "sample"
    ]
    assert len(samples) >= 2
    assert all("rss_gb" in r for r in samples)


def test_stopping_twice_is_harmless(tmp_path):
    log = RunLog.for_run(tmp_path)
    log.start_sampling(interval_seconds=0.02)
    log.stop_sampling()
    log.stop_sampling()


def test_the_summary_names_the_worst_query_and_the_peak(tmp_path):
    log = RunLog.for_run(tmp_path)
    log.write("query", label="schnell", duration_seconds=1.0, rows=10)
    log.write("query", label="langsam", duration_seconds=93_000.0, rows=3357)
    log.write("sample", rss_gb=2.0, available_gb=9.0)
    log.write("sample", rss_gb=27.5, available_gb=1.8)
    summary = summarize(read_records(tmp_path / RUN_LOG_FILENAME))
    assert summary["slowest_queries"][0]["label"] == "langsam"
    assert summary["peak_rss_gb"] == 27.5
    assert summary["lowest_available_gb"] == 1.8
    text = render_summary(summary)
    assert "langsam" in text and "27.5" in text


# --- Why a stage recomputed --------------------------------------------------


def _manifest(tmp_path, **overrides):
    output = tmp_path / "out.json"
    output.write_text("{}", encoding="utf-8")
    fields = {
        "stage": "validate",
        "stage_version": "1.0.0",
        "input_hash": "in-a",
        "config_hash": "cfg-a",
        "output_paths": (str(output),),
        "row_counts": {},
        "created_at": "2026-08-20T00:00:00+00:00",
    }
    fields.update(overrides)
    return StageManifest(**fields)


def test_a_reused_artifact_says_so(tmp_path):
    manifest = _manifest(tmp_path)
    log = RunLog.for_run(tmp_path / "run")
    with active(log):
        assert manifest.matches("in-a", "cfg-a", "1.0.0") is True
    records = [r for r in read_records(log.path) if r["kind"] == "reuse"]
    assert len(records) == 1
    assert records[0]["reused"] is True
    assert records[0]["decided_stage"] == "validate"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (("in-b", "cfg-a", "1.0.0"), "input_hash"),
        (("in-a", "cfg-b", "1.0.0"), "config_hash"),
        (("in-a", "cfg-a", "1.1.0"), "Stufenversion"),
    ],
)
def test_the_record_names_the_key_that_moved(tmp_path, call, expected):
    """A bool could not explain a walk-back over eleven stages."""

    manifest = _manifest(tmp_path)
    log = RunLog.for_run(tmp_path / "run")
    with active(log):
        assert manifest.matches(*call) is False
    record = [r for r in read_records(log.path) if r["kind"] == "reuse"][0]
    assert record["reused"] is False
    assert expected in record["reason"]


def test_a_missing_output_is_named(tmp_path):
    manifest = _manifest(tmp_path)
    (tmp_path / "out.json").unlink()
    reused, reason = manifest.reuse_decision("in-a", "cfg-a", "1.0.0")
    assert reused is False
    assert "out.json" in reason


def test_an_incomplete_previous_run_is_not_reused(tmp_path):
    manifest = _manifest(tmp_path, status="source_missing")
    reused, reason = manifest.reuse_decision("in-a", "cfg-a", "1.0.0")
    assert reused is False
    assert "unvollstaendig" in reason


def test_reuse_notes_are_silent_without_a_log(tmp_path):
    """Twenty call sites must stay safe in tests and single-stage runs."""

    set_active_log(None)
    assert _manifest(tmp_path).matches("in-a", "cfg-a", "1.0.0") is True
    note_reuse("validate", reused=True, reason="nichts hoert zu")


def test_logging_failure_never_reaches_the_caller(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    log = RunLog.for_run(tmp_path / "run")

    def explode(*_args, **_kwargs):
        raise OSError("Datentraeger voll")

    monkeypatch.setattr(RunLog, "write", explode)
    with active(log):
        assert manifest.matches("in-a", "cfg-a", "1.0.0") is True


# --- The summary a person actually reads -------------------------------------


def test_the_summary_carries_duration_peak_rows_and_reason():
    records = [
        {"kind": "stage", "stage": "gap", "event": "begin", "decision": "offered"},
        {"kind": "reuse", "decided_stage": "gap", "reused": False,
         "reason": "Eingangsdaten geaendert (input_hash)"},
        {"kind": "sample", "stage": "gap", "rss_gb": 2.1, "available_gb": 9.0},
        {"kind": "sample", "stage": "gap", "rss_gb": 3.7, "available_gb": 6.0},
        {"kind": "query", "stage": "gap", "label": "population",
         "rows": 640104, "duration_seconds": 812.0},
        {"kind": "stage", "stage": "gap", "event": "end",
         "duration_seconds": 900.0},
    ]
    summary = summarize(records)
    entry = summary["stages"][0]
    assert entry["duration_seconds"] == 900.0
    assert entry["peak_rss_gb"] == 3.7
    assert entry["rows"] == 640104
    assert entry["decision"] == "neu gerechnet"
    assert "input_hash" in entry["reason"]
    assert summary["lowest_available_gb"] == 6.0

    text = render_summary(summary)
    assert "gap" in text
    assert "640,104" in text
    assert "neu gerechnet" in text


def test_a_skipped_stage_is_visible_as_skipped():
    summary = summarize(
        [
            {
                "kind": "stage",
                "stage": "ingest",
                "event": "skipped",
                "decision": "skipped",
                "reason": "im Laufmanifest als abgeschlossen",
            }
        ]
    )
    assert summary["stages"][0]["state"] == "skipped"
    assert "abgeschlossen" in render_summary(summary)


def test_the_discarded_completed_list_is_reported():
    """The event that cost eleven stages, now stated instead of inferred."""

    summary = summarize(
        [
            {
                "kind": "workflow",
                "event": "completed_list_discarded",
                "reason": "Vertragshash des Quellstands weicht ab",
                "discarded_stage_count": 11,
            }
        ]
    )
    text = render_summary(summary)
    assert "11" in text
    assert "Vertragshash" in text


def test_a_stage_shorter_than_the_sampling_interval_still_reports_memory():
    """therapy-response finishes in six seconds; the 30 s sampler misses it."""

    summary = summarize(
        [
            {"kind": "stage", "stage": "therapy-response", "event": "begin",
             "rss_gb": 0.27, "available_gb": 20.4},
            {"kind": "stage", "stage": "therapy-response", "event": "end",
             "duration_seconds": 6.1, "rss_gb": 0.31, "available_gb": 20.37},
        ]
    )
    entry = summary["stages"][0]
    assert entry["peak_rss_gb"] == 0.31
    assert summary["peak_rss_gb"] == 0.31
    assert summary["lowest_available_gb"] == 20.37


# --- One stage, several attempts ---------------------------------------------


def test_a_repeated_stage_shows_its_latest_attempt_only():
    """The log spans every attempt a run ever made.

    On 2026-08-29 the summary merged them: ``repeatability`` read as "failed"
    hours after the defect was fixed, and the gap stage as "done, 171 min"
    while it was recomputing. Records are chronological, so the last begin
    wins.
    """

    records = [
        {"kind": "stage", "stage": "gap", "event": "begin", "at": "T1",
         "decision": "verification"},
        {"kind": "sample", "stage": "gap", "rss_gb": 6.6, "at": "T1a"},
        {"kind": "stage", "stage": "gap", "event": "end",
         "duration_seconds": 10260.0, "at": "T2"},
        {"kind": "stage", "stage": "gap", "event": "begin", "at": "T3",
         "decision": "offered"},
        {"kind": "sample", "stage": "gap", "rss_gb": 1.2, "at": "T3a"},
    ]
    entry = summarize(records)["stages"][0]
    assert entry["state"] == "running"
    assert entry["duration_seconds"] is None
    assert entry["peak_rss_gb"] == 1.2, "die Spitze des alten Laufs zaehlt nicht"
    assert entry["decision"] == "offered"
    assert entry["passes"] == 2


def test_the_repeat_is_visible_in_the_table():
    records = [
        {"kind": "stage", "stage": "gap", "event": "begin", "at": "T1"},
        {"kind": "stage", "stage": "gap", "event": "end",
         "duration_seconds": 5.0, "at": "T2"},
        {"kind": "stage", "stage": "gap", "event": "begin", "at": "T3"},
        {"kind": "stage", "stage": "gap", "event": "end",
         "duration_seconds": 7.0, "at": "T4"},
    ]
    assert "2. Durchlauf" in render_summary(summarize(records))


def test_an_error_before_the_latest_start_is_not_reported_as_current():
    """A binder error from the previous evening is not a live failure."""

    records = [
        {"kind": "stage", "stage": "repeat", "event": "begin", "at": "2026-08-29T08:24:00"},
        {"kind": "error", "stage": "repeat", "error": "BinderException: alt",
         "at": "2026-08-29T08:25:00"},
        {"kind": "stage", "stage": "repeat", "event": "failed", "at": "2026-08-29T08:25:01"},
        {"kind": "stage", "stage": "repeat", "event": "begin", "at": "2026-08-29T13:00:00"},
    ]
    summary = summarize(records)
    assert summary["errors"] == []
    assert len(summary["earlier_errors"]) == 1
    text = render_summary(summary)
    assert "frueheren Anlauf" in text
    assert "Fehler im laufenden Anlauf" not in text


def test_an_error_after_the_latest_start_is_reported_as_current():
    records = [
        {"kind": "stage", "stage": "repeat", "event": "begin", "at": "2026-08-29T13:00:00"},
        {"kind": "error", "stage": "repeat", "error": "BinderException: neu",
         "at": "2026-08-29T13:00:05"},
    ]
    summary = summarize(records)
    assert len(summary["errors"]) == 1
    assert summary["earlier_errors"] == []
    assert "Fehler im laufenden Anlauf" in render_summary(summary)


def test_a_killed_stage_is_not_reported_as_still_running():
    """A killed process cannot write its own end record.

    One stage runs at a time, so a stage still marked running when the next
    one begins did not finish. Reading that gap as progress is how a stage
    abandoned hours ago keeps claiming the machine.
    """

    records = [
        {"kind": "stage", "stage": "repeat", "event": "begin", "at": "T1"},
        {"kind": "sample", "stage": "repeat", "rss_gb": 13.7, "at": "T2"},
        {"kind": "stage", "stage": "gap", "event": "begin", "at": "T3"},
    ]
    states = {e["stage"]: e["state"] for e in summarize(records)["stages"]}
    assert states["repeat"] == "abgebrochen"
    assert states["gap"] == "running"


def test_the_stage_that_is_actually_running_keeps_saying_so():
    records = [
        {"kind": "stage", "stage": "a", "event": "begin", "at": "T1"},
        {"kind": "stage", "stage": "a", "event": "end",
         "duration_seconds": 1.0, "at": "T2"},
        {"kind": "stage", "stage": "b", "event": "begin", "at": "T3"},
    ]
    states = {e["stage"]: e["state"] for e in summarize(records)["stages"]}
    assert states["a"] == "done"
    assert states["b"] == "running"


def test_a_reuse_record_attaches_to_the_stage_that_was_running():
    """A manifest may call itself something other than its stage.

    ``ingest-events`` checks a manifest named ``ingest-clinical-events``, and
    ``parse-reports`` one named ``parse-machine-reports``. Keying the summary
    on the manifest name invented two stage rows no run ever had -- which then
    appeared as "abgebrochen" once abandoned stages were detected, i.e. as two
    failures that never happened.
    """

    records = [
        {"kind": "stage", "stage": "ingest-events", "event": "begin", "at": "T1"},
        {"kind": "reuse", "stage": "ingest-events",
         "decided_stage": "ingest-clinical-events", "reused": True,
         "reason": "Schluessel identisch", "at": "T2"},
        {"kind": "stage", "stage": "ingest-events", "event": "end",
         "duration_seconds": 66.0, "at": "T3"},
        {"kind": "stage", "stage": "build-index", "event": "begin", "at": "T4"},
    ]
    summary = summarize(records)
    names = {entry["stage"] for entry in summary["stages"]}
    assert names == {"ingest-events", "build-index"}
    entry = next(e for e in summary["stages"] if e["stage"] == "ingest-events")
    assert entry["state"] == "done"
    assert entry["decision"] == "wiederverwendet"
    # The manifest name is information, not noise: it says what was checked.
    assert "ingest-clinical-events" in entry["reason"]


def test_a_reuse_record_without_an_active_stage_still_lands_somewhere():
    """Single-stage runs open no envelope; the verdict must not vanish."""

    records = [
        {"kind": "reuse", "decided_stage": "validate", "reused": False,
         "reason": "Eingangsdaten geaendert (input_hash)", "at": "T1"},
    ]
    entry = summarize(records)["stages"][0]
    assert entry["stage"] == "validate"
    assert entry["decision"] == "neu gerechnet"


def test_the_overall_figures_describe_the_current_pass():
    """The 14.4 GB of an abandoned hand-run is not the run's high-water mark.

    Every line of the table above already showed the latest pass; the totals
    underneath were still taken from every record in the file, so they
    contradicted it.
    """

    records = [
        {"kind": "stage", "stage": "repeat", "event": "begin", "at": "T1"},
        {"kind": "sample", "stage": "repeat", "rss_gb": 14.4,
         "available_gb": 4.3, "at": "T2"},
        {"kind": "stage", "stage": "gap", "event": "begin", "at": "T3"},
        {"kind": "sample", "stage": "gap", "rss_gb": 5.9,
         "available_gb": 17.5, "at": "T4"},
        {"kind": "stage", "stage": "repeat", "event": "begin", "at": "T5"},
        {"kind": "sample", "stage": "repeat", "rss_gb": 2.1,
         "available_gb": 19.0, "at": "T6"},
    ]
    summary = summarize(records)
    assert summary["peak_rss_gb"] == 5.9, "die 14.4 GB gehoeren zum alten Anlauf"
    assert summary["lowest_available_gb"] == 17.5


def test_a_measurement_outside_any_stage_is_not_dropped():
    """A run that opened no envelope still has a sampler."""

    records = [
        {"kind": "sample", "rss_gb": 27.5, "available_gb": 1.8},
        {"kind": "sample", "rss_gb": 2.0, "available_gb": 9.0},
    ]
    summary = summarize(records)
    assert summary["peak_rss_gb"] == 27.5
    assert summary["lowest_available_gb"] == 1.8


def test_a_shortened_error_says_that_it_was_shortened():
    """A hard slice cut ``subject_id`` to ``subject_i`` and looked broken."""

    from vcg_biomarker.mimic.run_log import _short_error

    long = (
        'BinderException: Binder Error: Table "a" does not have a column '
        'named "subject_id"\n\nCandidate bindings: : "extraction_status"\n\n'
        "LINE 3580:               ON a.subject_id = c.subject_id"
    )
    shown = _short_error(long)
    assert "\n" not in shown, "nur die erste Zeile gehoert in eine Tabelle"
    assert shown.endswith("\u2026") or len(shown) < 96
    assert "Candidate bindings" not in shown


def test_a_short_error_is_left_alone():
    from vcg_biomarker.mimic.run_log import _short_error

    assert _short_error("ValueError: zu wenig Paare") == (
        "ValueError: zu wenig Paare"
    )
    assert _short_error("") == ""
    assert _short_error(None) == ""


def test_the_totals_are_formatted_like_the_table():
    """13.668 GB under a table that says 13.7 GB reads as two measurements."""

    text = render_summary(
        summarize(
            [
                {"kind": "stage", "stage": "a", "event": "begin", "at": "T1"},
                {"kind": "sample", "stage": "a", "rss_gb": 13.66789,
                 "available_gb": 4.35412, "at": "T2"},
            ]
        )
    )
    assert "13.7 GB" in text
    assert "13.668" not in text
    assert "4.4 GB" in text


def test_the_full_traceback_is_still_reachable():
    """The table abbreviates; nothing may be lost from the file itself."""

    import inspect

    from vcg_biomarker.mimic import run_log as module

    assert "--errors" in inspect.getsource(module.render_summary)


def test_a_stage_reports_what_it_produced():
    """The Zeilen column existed and was always empty."""

    records = [
        {"kind": "stage", "stage": "extract-features", "event": "begin",
         "at": "T1", "position": 18},
        {"kind": "stage", "stage": "extract-features", "event": "rows",
         "at": "T2", "position": 18, "rows": 630044,
         "row_counts": {"successful_ecgs": 630044}},
        {"kind": "stage", "stage": "extract-features", "event": "end",
         "duration_seconds": 12.0, "at": "T3"},
    ]
    entry = summarize(records)["stages"][0]
    assert entry["rows"] == 630044
    assert "630,044" in render_summary(summarize(records))


def test_a_new_pass_forgets_the_previous_row_count():
    records = [
        {"kind": "stage", "stage": "a", "event": "begin", "at": "T1"},
        {"kind": "stage", "stage": "a", "event": "rows", "rows": 999, "at": "T2"},
        {"kind": "stage", "stage": "a", "event": "end",
         "duration_seconds": 1.0, "at": "T3"},
        {"kind": "stage", "stage": "a", "event": "begin", "at": "T4"},
    ]
    assert summarize(records)["stages"][0]["rows"] is None


def test_the_stages_are_listed_in_pipeline_order():
    """A stage run by hand beforehand used to sit above step one."""

    records = [
        {"kind": "stage", "stage": "spaet", "event": "begin", "at": "T0"},
        {"kind": "stage", "stage": "spaet", "event": "end",
         "duration_seconds": 1.0, "at": "T1", "position": 19},
        {"kind": "stage", "stage": "frueh", "event": "begin", "at": "T2"},
        {"kind": "stage", "stage": "frueh", "event": "policy", "position": 1,
         "at": "T3"},
        {"kind": "stage", "stage": "frueh", "event": "end",
         "duration_seconds": 1.0, "at": "T4"},
    ]
    names = [entry["stage"] for entry in summarize(records)["stages"]]
    assert names == ["frueh", "spaet"]


def test_a_stage_without_a_position_is_listed_last_not_dropped():
    records = [
        {"kind": "stage", "stage": "ohne", "event": "begin", "at": "T0"},
        {"kind": "stage", "stage": "mit", "event": "begin", "at": "T1",
         "position": 3},
    ]
    names = [entry["stage"] for entry in summarize(records)["stages"]]
    assert names == ["mit", "ohne"]
