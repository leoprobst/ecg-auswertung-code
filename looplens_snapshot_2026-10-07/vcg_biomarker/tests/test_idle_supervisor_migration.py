from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from audit_tools import migrate_idle_supervisor as migration


@pytest.mark.parametrize("blocker", ["jobs", "children", "identity", None])
def test_migration_only_stops_verified_idle_supervisor(monkeypatch, blocker):
    config = SimpleNamespace(project_file=Path("project.json"))
    store = MagicMock()
    store.supervisor_status.return_value = {"pid": 121 if blocker == "identity" else 120}
    connection = store.connect.return_value.__enter__.return_value
    connection.execute.return_value.fetchall.return_value = [object()] if blocker == "jobs" else []
    process = MagicMock()
    process.cmdline.return_value = ["python", "-m", "vcg_biomarker.mimic_supervisor", "--project", "project.json"]
    process.children.return_value = [object()] if blocker == "children" else []
    monkeypatch.setattr(migration.MimicProjectConfig, "load", lambda _: config)
    monkeypatch.setattr(migration.JobStore, "for_project", lambda _: store)
    monkeypatch.setattr(migration.psutil, "Process", lambda _: process)
    launch = MagicMock(return_value=130)
    monkeypatch.setattr(migration, "ensure_supervisor", launch)
    if blocker:
        with pytest.raises(RuntimeError):
            migration.migrate(config.project_file, 120)
        process.terminate.assert_not_called()
        launch.assert_not_called()
    else:
        result = migration.migrate(config.project_file, 120)
        assert result["analysis_jobs_enqueued"] == 0
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)
        connection.commit.assert_called_once()
        launch.assert_called_once_with(config)


@pytest.mark.parametrize("kind", ["owned", "wrong_parent", "wrong_exe", "has_children", "denied"])
def test_only_actual_direct_console_host_is_exempt(tmp_path, monkeypatch, kind):
    windows = tmp_path / "Windows"
    monkeypatch.setattr(migration, "os", SimpleNamespace(name="nt", environ={"SystemRoot": str(windows)}))
    child = MagicMock()
    child.ppid.return_value = 999 if kind == "wrong_parent" else 120
    child.exe.return_value = str(windows / "System32" / ("python.exe" if kind == "wrong_exe" else "conhost.exe"))
    child.children.return_value = [object()] if kind == "has_children" else []
    if kind == "denied":
        child.exe.side_effect = migration.psutil.AccessDenied(130)
    assert migration._is_owned_console_host(child, 120) is (kind == "owned")
