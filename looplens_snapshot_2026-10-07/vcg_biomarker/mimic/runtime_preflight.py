"""Read-only study-start guard; never replace a running supervisor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import psutil


def require_isolated_study_supervisor(status: Mapping, snapshots_root: Path) -> None:
    """Check runtime integrity, not dependency isolation or source freshness."""
    pid = status.get("pid")
    if not pid:
        return
    try:
        if isinstance(pid, bool) or int(pid) <= 0:
            raise ValueError("invalid supervisor PID")
        process = psutil.Process(int(pid))
        if "vcg_biomarker.mimic_supervisor" not in process.cmdline():
            raise ValueError("supervisor PID belongs to another process")
        runtime = Path(process.cwd()).resolve()
        root = snapshots_root.resolve()
        if runtime == root or not runtime.is_relative_to(root):
            raise ValueError("supervisor uses development code")
        manifest = json.loads((runtime / "runtime_manifest.json").read_text(encoding="utf-8"))
        files = manifest.get("files")
        if manifest.get("version") != 1 or not isinstance(files, dict) or not files:
            raise ValueError("invalid runtime manifest")
        required = {"vcg_biomarker/mimic_supervisor.py", "vcg_biomarker/mimic_job_worker.py"}
        if not required.issubset(files):
            raise ValueError("runtime entry points missing")
        for relative, digest in files.items():
            file = (runtime / relative).resolve()
            if not file.is_relative_to(runtime) or not file.is_file():
                raise ValueError("runtime file missing or outside snapshot")
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise ValueError("runtime source hash mismatch")
    except psutil.NoSuchProcess:
        return
    except (psutil.Error, OSError, ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            "Studienstart gesperrt: Supervisor-Codeisolation nicht nachgewiesen. "
            "Laufende Jobs unveraendert fertigstellen; danach Supervisor kontrolliert "
            "aus neuer Codekopie starten. Kein automatischer Prozessabbruch. "
            f"Detail: {exc}"
        ) from exc
