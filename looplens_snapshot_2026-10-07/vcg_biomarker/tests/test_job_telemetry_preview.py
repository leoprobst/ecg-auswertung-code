import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from vcg_biomarker.gui.job_telemetry import telemetry_path, telemetry_preview, TAIL_BYTES


def fixture(tmp_path):
    config = SimpleNamespace(jobs_root=tmp_path / "jobs", runs_root=tmp_path / "runs")
    spec = SimpleNamespace(job_id="test", stage="build-evidence", arguments={})
    path = telemetry_path(config, spec)
    path.parent.mkdir(parents=True)
    return config, spec, path


def sample(**fields):
    return dict(job_id="test", attempt_id="first", kind="sample",
                at="2026-09-12T00:00:00+00:00", rss_gb=2,
                tree_cpu_core_percent=250, tree_read_bytes_per_second=1048576,
                active_queries=[dict(label="context", elapsed_seconds=120)], **fields)


def test_snapshot_is_explicitly_historical_and_has_correct_units(tmp_path):
    config, spec, path = fixture(tmp_path)
    path.write_text(json.dumps(sample()) + "\n", encoding="utf-8")
    before = path.read_bytes()
    text = telemetry_preview(config, spec, now=datetime(2026, 9, 12, 0, 2, tzinfo=timezone.utc))
    assert "VERALTET" in text and "kein Nachweis" in text and "keine ETA" in text
    assert "2.50" in text and "1.00" in text and "context: 120 s" in text
    assert path.read_bytes() == before


def test_new_attempt_never_inherits_old_resources(tmp_path):
    config, spec, path = fixture(tmp_path)
    lines = [sample(), dict(job_id="test", attempt_id="second", kind="stage", event="begin")]
    path.write_text("\n".join(map(json.dumps, lines)), encoding="utf-8")
    text = telemetry_preview(config, spec)
    assert "Versuch: second" in text and "Noch keine" in text
    assert "context" not in text


def test_partial_line_large_prefix_and_foreign_job_are_not_displayed(tmp_path):
    config, spec, path = fixture(tmp_path)
    foreign = {**sample(), "job_id": "other", "attempt_id": "foreign"}
    path.write_bytes(b"x" * (TAIL_BYTES * 2) + b"\n" + json.dumps(sample()).encode()
                     + b"\n" + json.dumps(foreign).encode() + b'\n{"partial":')
    text = telemetry_preview(config, spec)
    assert "Versuch: first" in text and "foreign" not in text
    assert len(text) < 2000


@pytest.mark.parametrize("run_id", ["..", "../outside", "."])
def test_guided_path_cannot_escape_run_directory(tmp_path, run_id):
    config, spec, _ = fixture(tmp_path)
    spec.stage = "study-run"
    spec.arguments = {"run_id": run_id}
    assert "ausserhalb" in telemetry_preview(config, spec)


def test_missing_legacy_and_invalid_values_remain_unknown(tmp_path):
    config, spec, path = fixture(tmp_path)
    assert "Keine lesbare" in telemetry_preview(config, spec)
    row = {**sample(), "rss_gb": 10**400, "tree_cpu_core_percent": True,
           "at": "bad-date", "active_queries": None}
    path.write_text(json.dumps(row), encoding="utf-8")
    text = telemetry_preview(config, spec)
    assert "Zeitbezug unbekannt" in text and "RAM (GiB): unbekannt" in text
    assert "Prozesse): unbekannt" in text and "nicht aufgezeichnet" in text


def test_study_run_uses_guided_log_and_standalone_matches_worker(tmp_path):
    from vcg_biomarker.mimic_job_worker import _standalone_job_telemetry
    config, spec, path = fixture(tmp_path)
    with _standalone_job_telemetry(config, spec) as log:
        assert log.path == path
    spec.stage = "study-run"
    spec.arguments = {"run_id": "A-test"}
    assert telemetry_path(config, spec) == config.runs_root / "A-test" / "run.jsonl"


def test_gui_action_opens_read_only_snapshot_dialog(tmp_path):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QWidget, QTextEdit
    from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget

    app = QApplication.instance() or QApplication([])
    config, spec, path = fixture(tmp_path)
    path.write_text(json.dumps(sample()), encoding="utf-8")
    parent = QWidget()
    parent.config = config
    parent._selected_job_id = spec.job_id
    parent.job_store = SimpleNamespace(get=lambda _: SimpleNamespace(spec=spec))
    observed = []

    def inspect():
        dialog = app.activeModalWidget()
        preview = dialog.findChild(QTextEdit)
        observed.append((preview.isReadOnly(), preview.toPlainText()))
        dialog.accept()

    QTimer.singleShot(50, inspect)
    MimicProjectWidget._show_selected_job_telemetry(parent)
    assert observed[0][0]
    assert "context: 120 s" in observed[0][1]
    parent.close()
