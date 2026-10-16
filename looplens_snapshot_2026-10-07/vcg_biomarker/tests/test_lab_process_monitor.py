from types import SimpleNamespace

import psutil
import pytest

from vcg_biomarker.gui.lab_process_monitor import live_tree_rss, stop_owned_lab_tree


def test_disappearing_child_does_not_discard_live_memory():
    def gone():
        raise psutil.NoSuchProcess(123)

    parent = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100),
        children=lambda **_: [SimpleNamespace(memory_info=gone),
            SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=200))])
    assert live_tree_rss(parent) == 300


def test_disappeared_root_has_no_live_memory():
    def gone(**_):
        raise psutil.NoSuchProcess(123)

    assert live_tree_rss(SimpleNamespace(children=gone)) == 0


def test_access_denied_is_not_reported_as_zero_memory():
    def denied():
        raise psutil.AccessDenied(123)

    with pytest.raises(psutil.AccessDenied):
        live_tree_rss(SimpleNamespace(children=lambda **_: [], memory_info=denied))


def test_cleanup_refuses_current_host():
    import os
    with pytest.raises(ValueError, match="host"):
        stop_owned_lab_tree(psutil.Process(os.getpid()))


def test_real_owned_worker_and_child_are_stopped(tmp_path):
    import os
    import subprocess
    import sys
    import time

    ready = tmp_path / "ready.txt"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    worker = subprocess.Popen([sys.executable, "-c", code, str(ready)], creationflags=flags)
    child = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), "Test child did not initialize"
        child = psutil.Process(int(ready.read_text()))
        assert child.is_running()
        stop_owned_lab_tree(psutil.Process(worker.pid))
        worker.wait(timeout=5)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if worker.poll() is None:
            stop_owned_lab_tree(psutil.Process(worker.pid))
        worker.wait(timeout=5)
        if child is not None and child.is_running():
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
