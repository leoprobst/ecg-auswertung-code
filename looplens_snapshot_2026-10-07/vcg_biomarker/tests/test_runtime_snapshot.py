import json
import subprocess
import sys

import pytest

from vcg_biomarker.mimic.runtime_snapshot import create_runtime_snapshot, runtime_environment


def test_snapshot_is_independent_and_importable_after_source_edit(tmp_path):
    source = tmp_path / "source"
    package = source / "vcg_biomarker"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 'frozen'", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "old.pyc").write_bytes(b"old")
    runtime = create_runtime_snapshot(source, tmp_path / "runtimes")
    (package / "__init__.py").write_text("VALUE = 'edited'", encoding="utf-8")
    result = subprocess.run([sys.executable, "-c",
                             "import vcg_biomarker; print(vcg_biomarker.VALUE)"],
                            cwd=runtime, env=runtime_environment(), capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == "frozen"
    assert not (runtime / "vcg_biomarker" / "__pycache__" / "old.pyc").exists()
    manifest = json.loads((runtime / "runtime_manifest.json").read_text())
    assert "vcg_biomarker/__init__.py" in manifest["files"]
    assert manifest["dependency_isolation"] is False


def test_runtime_environment_removes_development_import_override(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "development")
    assert "PYTHONPATH" not in runtime_environment()
    assert runtime_environment()["PYTHONNOUSERSITE"] == "1"


def test_snapshot_rejects_recursive_destination(tmp_path):
    package = tmp_path / "vcg_biomarker"
    package.mkdir()
    with pytest.raises(ValueError, match="inside"):
        create_runtime_snapshot(tmp_path, package / "snapshots")


def test_supervisor_launch_uses_private_runtime_and_detaches(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    from vcg_biomarker.mimic import supervisor

    config = SimpleNamespace(jobs_root=tmp_path, project_file=tmp_path / "project.json")
    monkeypatch.setattr(supervisor.JobStore, "for_project", lambda _: SimpleNamespace(supervisor_status=lambda: {}))
    runtime = tmp_path / "private"
    monkeypatch.setattr(supervisor, "create_runtime_snapshot", lambda *_: runtime)
    captured = {}

    def launch(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    assert supervisor.ensure_supervisor(config) == 123
    assert captured["cwd"] == str(runtime)
    assert captured["stdin"] == subprocess.DEVNULL
    assert "PYTHONPATH" not in captured["env"]
    assert captured["stdout"].closed
    if os.name == "nt":
        assert captured["creationflags"] & subprocess.DETACHED_PROCESS
    else:
        assert captured["start_new_session"]


def test_active_supervisor_is_not_replaced(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vcg_biomarker.mimic import supervisor

    monkeypatch.setattr(supervisor.JobStore, "for_project", lambda _: SimpleNamespace(
        supervisor_status=lambda: {"pid": 123, "heartbeat_at": "present"}))
    monkeypatch.setattr(supervisor, "_pid_alive", lambda _: True)
    monkeypatch.setattr(supervisor, "_heartbeat_recent", lambda _: True)
    monkeypatch.setattr(supervisor, "create_runtime_snapshot", lambda *_: pytest.fail("active session must not change"))
    assert supervisor.ensure_supervisor(SimpleNamespace()) == 123


def test_detached_snapshot_worker_survives_launcher_exit_and_late_import(tmp_path):
    """Real subprocess check without starting a production job or reading data."""
    import time

    source = tmp_path / "source"
    package = source / "vcg_biomarker"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "late.py").write_text("VALUE = 'frozen'", encoding="utf-8")
    runtime = create_runtime_snapshot(source, tmp_path / "runtimes")
    ready, release, result = [tmp_path / name for name in ("ready", "release", "result")]
    worker = (
        "from pathlib import Path; import time; "
        f"Path({str(ready)!r}).touch(); deadline=time.monotonic()+10\n"
        f"while not Path({str(release)!r}).exists():\n"
        " if time.monotonic()>deadline: raise TimeoutError('release not received')\n"
        " time.sleep(.02)\n"
        "from vcg_biomarker.late import VALUE\n"
        f"Path({str(result)!r}).write_text(VALUE)\n"
    )
    launcher = (
        "import os, subprocess, sys\n"
        "kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL, close_fds=True)\n"
        "if os.name == 'nt': kwargs['creationflags'] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW\n"
        "else: kwargs['start_new_session'] = True\n"
        f"subprocess.Popen([sys.executable, '-c', {worker!r}], **kwargs)\n"
    )
    subprocess.run([sys.executable, "-c", launcher], cwd=runtime,
                   env=runtime_environment(), check=True, timeout=10)
    # Launcher has exited. Edit the live source before the worker's late import.
    (package / "late.py").write_text("VALUE = 'development-edit'", encoding="utf-8")
    release.touch()
    deadline = time.monotonic() + 10
    while not result.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert ready.exists()
    assert result.read_text() == "frozen"
