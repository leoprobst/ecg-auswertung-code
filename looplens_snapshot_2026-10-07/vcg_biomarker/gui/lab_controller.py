"""Asynchronous bounded laboratory subprocess with cooperative cancellation."""
from __future__ import annotations

import json
import sys
import uuid
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, QLockFile, QCoreApplication, pyqtSignal
from ..mimic.storage import atomic_write_json
from .lab_process_monitor import live_tree_rss, stop_owned_lab_tree


class LabProcessController(QObject):
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)
    busyChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.process.started.connect(self._send)
        self.monitor = QTimer(self)
        self.monitor.setInterval(2000)
        self.monitor.timeout.connect(self._sample)
        self.output = ""
        self.request = None
        self.cancel_path = None
        self.lock = QLockFile(str(Path(tempfile.gettempdir()) / "vcg-expert-lab.lock"))
        self.lock.setStaleLockTime(0)
        self.cancel_at = None
        self.owned_process = None
        if QCoreApplication.instance():
            QCoreApplication.instance().aboutToQuit.connect(self.shutdown)

    @property
    def busy(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, config, case_id, powerline, *, ischemia_context=None):
        if self.busy:
            raise RuntimeError("Eine Laboranalyse läuft bereits")
        import psutil
        available = psutil.virtual_memory().available
        if available < 4 * 1024**3:
            raise RuntimeError(
                f"Laborstart zurückgestellt: {available / 1024**3:.2f} GiB RAM frei; "
                "benötigt werden 4 GiB (2 GiB Reserve + 2 GiB Laborbudget). "
                "Bitte später erneut starten. Die Produktionsanalyse bleibt unverändert."
            )
        from ..mimic.lab_workspace import LabReviewStore
        from ..mimic.ischemia_patterns import validate_lab_ischemia_context
        ischemia_context = validate_lab_ischemia_context(ischemia_context, case_id)
        store = LabReviewStore(config.cases_root, case_id)
        self.cancel_path = store.root / "jobs" / f"{uuid.uuid4().hex}.cancel.json"
        self.request = {"schema_version": "1.0.0", "config": config.to_dict(), "case_id": case_id, "powerline": powerline, "cancel_path": str(self.cancel_path)}
        self.request['ischemia_context'] = ischemia_context
        if not self.lock.tryLock(0):
            raise RuntimeError("Ein anderer Experten-Laborauftrag läuft bereits. Bitte später erneut starten.")
        self.output = ""
        self.cancel_at = None
        env = QProcessEnvironment.systemEnvironment()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.insert(key, "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        self.busyChanged.emit(True)
        self.status.emit("Lokaler Laborprozess startet · maximal 1 Auftrag / 1 nativer Thread")
        self.process.start(sys.executable, ["-m", "vcg_biomarker.mimic.lab_process"])

    def _send(self):
        import psutil
        self.owned_process = psutil.Process(int(self.process.processId()))
        self.process.write((json.dumps(self.request) + "\n").encode("utf-8"))
        self.process.closeWriteChannel()
        self.monitor.start()

    def cancel(self):
        if self.busy and self.cancel_path:
            if self.cancel_at is None:
                self.cancel_at = time.monotonic()
            self.cancel_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.cancel_path, {"cancel": True})
            self.status.emit("Laborabbruch angefordert; laufende Rechenphase endet kontrolliert.")

    def _sample(self):
        import psutil
        try:
            proc = self.owned_process
            if proc is None:
                return
            if self.cancel_at is not None and time.monotonic() - self.cancel_at > 10:
                stop_owned_lab_tree(proc, timeout=1)
                return
            rss = live_tree_rss(proc)
            free = psutil.virtual_memory().available
            if free < 2*1024**3 or rss > 2*1024**3:
                self.cancel()
                return
            if not self.cancel_path.is_file():
                self.status.emit(f"Labor rechnet · RAM {rss/1024**3:.2f} GiB · frei {free/1024**3:.2f} GiB · Produktionslauf unverändert")
        except psutil.AccessDenied:
            self.cancel()
        except psutil.NoSuchProcess:
            return

    def _read(self):
        self.output = (self.output + bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace"))[-100_000:]

    def _finished(self, code, _status):
        self.monitor.stop()
        self.lock.unlock()
        self._read()
        self.busyChanged.emit(False)
        if code != 0:
            self.failed.emit(self.output[-5000:] or "Laborprozess beendet ohne Ergebnis")
            return
        try:
            line = next(line for line in reversed(self.output.splitlines()) if line.startswith("LAB_RESULT="))
            result = json.loads(line[len("LAB_RESULT="):])
            if result["case_id"] != self.request["case_id"]:
                raise ValueError("Case identity mismatch")
            analysis_id = result["analysis_id"]
            if not isinstance(analysis_id, str) or not all(c in "0123456789abcdef" for c in analysis_id) or len(analysis_id) < 16:
                raise ValueError("Invalid analysis identifier")
            path = Path(self.request["config"]["work_root"]) / "cases" / result["case_id"] / "analyses" / analysis_id[:16] / "analysis.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("analysis_id") != analysis_id or payload.get("case_id") != result["case_id"]:
                raise ValueError("Analysis identity mismatch")
            self.completed.emit(payload)
        except (StopIteration, ValueError, KeyError, OSError) as exc:
            self.failed.emit(f"Laborergebnis nicht lesbar: {exc}")

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.lock.unlock()
            self.busyChanged.emit(False)
            self.failed.emit(self.process.errorString())

    def shutdown(self):
        self.monitor.stop()
        if self.busy:
            self.cancel()
            if not self.process.waitForFinished(1000) and self.owned_process is not None:
                stop_owned_lab_tree(self.owned_process, timeout=1)
                self.process.waitForFinished(1000)
        self.lock.unlock()
