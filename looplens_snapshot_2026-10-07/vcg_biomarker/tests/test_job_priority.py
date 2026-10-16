"""R6: job workers run below normal priority, and so do their pool children."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows priority classes")
def test_pool_children_inherit_below_normal_priority(tmp_path):
    import psutil

    code = (
        "import concurrent.futures, os, psutil\n"
        "def child_priority():\n"
        "    return int(psutil.Process().nice())\n"
        "if __name__ == '__main__':\n"
        "    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:\n"
        "        print(pool.submit(child_priority).result())\n"
    )
    script = tmp_path / "priority_probe.py"
    script.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip().splitlines()[-1]) == psutil.BELOW_NORMAL_PRIORITY_CLASS


def test_supervisor_launch_requests_below_normal_priority(tmp_path, monkeypatch):
    from vcg_biomarker.mimic import supervisor as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "subprocess.BELOW_NORMAL_PRIORITY_CLASS" in source
    # The I/O helper is best effort and never raises for a missing process.
    module._lower_io_priority(2**31 - 7)
