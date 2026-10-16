"""Read-only runtime location diagnostic, not a source-integrity attestation."""
from __future__ import annotations

from pathlib import Path

import psutil


def supervisor_runtime_notice(pid: object, snapshots_root: Path) -> str:
    """Inspect the live process; never replace a supervisor or touch its jobs."""
    try:
        if isinstance(pid, bool) or int(pid or 0) <= 0:
            return "Codeisolation unbekannt"
        process = psutil.Process(int(pid))
        command = process.cmdline()
        if "vcg_biomarker.mimic_supervisor" not in command:
            return "Codeisolation unbekannt: PID gehoert nicht zum Supervisor"
        runtime = Path(process.cwd()).resolve()
        root = snapshots_root.resolve()
        if runtime != root and runtime.is_relative_to(root):
            if (runtime / "runtime_manifest.json").is_file():
                return (
                    "Snapshot-Verzeichnis aktiv (keine erneute Hashpruefung). "
                    "GUI darf geschlossen und bearbeitet werden; gemeinsame Python-Umgebung "
                    "und Abhaengigkeiten waehrend der Analyse nicht aktualisieren."
                )
        return "WARNUNG: keine Codeisolation - Supervisor vor Voll-Lauf erneuern"
    except (psutil.Error, OSError, ValueError, TypeError):
        return "Codeisolation unbekannt: Prozess nicht pruefbar"
