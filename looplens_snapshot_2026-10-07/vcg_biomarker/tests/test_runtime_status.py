from types import SimpleNamespace

import psutil
import pytest

from vcg_biomarker.gui import runtime_status


@pytest.mark.parametrize("location,manifest,expected", [
    ("snapshots/session-1", True, "Snapshot-Verzeichnis aktiv"),
    ("development", True, "WARNUNG"),
    ("snapshots/session-1", False, "WARNUNG"),
    ("snapshots", True, "WARNUNG"),
])
def test_runtime_location_is_not_confused_with_arbitrary_manifest(
    tmp_path, monkeypatch, location, manifest, expected
):
    folder = tmp_path / location
    folder.mkdir(parents=True)
    if manifest:
        (folder / "runtime_manifest.json").write_text("{}")
    monkeypatch.setattr(runtime_status.psutil, "Process", lambda _: SimpleNamespace(
        cwd=lambda: str(folder), cmdline=lambda: ["python", "-m", "vcg_biomarker.mimic_supervisor"]
    ))
    assert expected in runtime_status.supervisor_runtime_notice(123, tmp_path / "snapshots")


def test_pid_reuse_and_access_denied_do_not_claim_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_status.psutil, "Process", lambda _: SimpleNamespace(
        cmdline=lambda: ["unrelated"]
    ))
    assert "PID gehoert nicht" in runtime_status.supervisor_runtime_notice(123, tmp_path)

    def denied(_):
        raise psutil.AccessDenied(123)

    monkeypatch.setattr(runtime_status.psutil, "Process", denied)
    assert "nicht pruefbar" in runtime_status.supervisor_runtime_notice(123, tmp_path)
    assert "unbekannt" in runtime_status.supervisor_runtime_notice(None, tmp_path)
