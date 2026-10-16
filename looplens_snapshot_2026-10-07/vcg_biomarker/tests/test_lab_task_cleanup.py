"""Inject resource failures around real disposable laboratory subprocesses."""
from types import SimpleNamespace
import sys

import psutil
import pytest


@pytest.mark.parametrize("failure", ["memory", "access_denied", "timeout"])
def test_task_reports_failure_after_stopping_owned_worker(tmp_path, monkeypatch, failure):
    from vcg_biomarker.gui.widgets import pilot_workspace_widget as gui

    original_launch = gui.subprocess.Popen
    launched = []

    def launch(command, **kwargs):
        assert "vcg_biomarker.mimic.pilot_lab_worker" in command
        worker = original_launch([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        launched.append(worker)
        return worker

    def memory(_):
        if failure == "access_denied":
            raise psutil.AccessDenied(123)
        return 0 if failure == "timeout" else 3 * 1024**3

    monkeypatch.setattr(gui.subprocess, "Popen", launch)
    monkeypatch.setattr(gui, "live_tree_rss", memory)
    if failure == "timeout":
        ticks = iter((0., 181.))
        monkeypatch.setattr(gui, "time", SimpleNamespace(
            monotonic=lambda: next(ticks), sleep=lambda _: None))
    task = gui.LabTask(SimpleNamespace(root=tmp_path), 1, "kors")
    errors, results = [], []
    task.failed.connect(errors.append)
    task.result_ready.connect(results.append)
    try:
        task.run()
        assert len(launched) == 1
        assert launched[0].poll() is not None
        assert len(errors) == 1 and results == []
        assert ("AccessDenied" if failure == "access_denied" else "180-s-Limit") in errors[0]
    finally:
        for worker in launched:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)
