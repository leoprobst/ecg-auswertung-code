"""Asynchronous pilot-only coordinator export; no production JobStore."""
from pathlib import Path
import json
import sys
import uuid

from PyQt6.QtCore import QProcess, QProcessEnvironment
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog


class ReviewerExportWidget(QWidget):
    def __init__(self, workspace: Path, parent=None):
        super().__init__(parent)
        self.workspace = Path(workspace).resolve()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.output = bytearray()
        self.destination: Path | None = None
        self.completed = False
        layout = QVBoxLayout(self)
        note = QLabel(
            "Koordinatoransicht: vorhandene Stichprobe aus dem Pilot-Overlay auswählen. "
            "Keine automatische Auswahl günstiger Fälle. Der Export benötigt mindestens "
            "zwei Beurteiler und dokumentierte Auswahlwahrscheinlichkeiten.\n"
            "Nur den einzelnen Ordner reviewer_packets/slot-N weitergeben – niemals "
            "coordinator_private oder den gesamten Exportordner. EKGs bleiben sensible "
            "Patientendaten. Keine klinische Freigabe; noch keine ärztliche Nachprüfung."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.button = QPushButton("Koordinator-Stichprobe auswählen und Signalpakete exportieren")
        self.button.clicked.connect(self.choose_selection)
        layout.addWidget(self.button)
        self.status = QLabel("Noch kein Export gestartet.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()

    @property
    def busy(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def choose_selection(self):
        path, _ = QFileDialog.getOpenFileName(self, "Koordinator-Stichprobe", str(self.workspace/"overlay"), "Parquet (*.parquet)")
        if path:
            try:
                self.start_export(Path(path))
            except (ValueError, OSError) as exc:
                self.status.setText(f"Export nicht gestartet: {exc}")

    def start_export(self, selection: Path):
        if self.busy:
            raise ValueError("Ein Export läuft bereits")
        overlay = (self.workspace/"overlay").resolve()
        source = Path(selection).resolve()
        if not source.is_relative_to(overlay) or not source.is_file():
            raise ValueError("Stichprobe muss eine vorhandene Datei im Pilot-Overlay sein")
        name = "review-" + uuid.uuid4().hex
        self.destination = overlay/"review_exports"/name
        self.output.clear()
        self.completed = False
        self.button.setEnabled(False)
        self.status.setText("Signalpakete werden im Hintergrund exportiert. Fenster bis zum Abschluss offen lassen.")
        env = QProcessEnvironment.systemEnvironment()
        env.remove("PYTHONPATH")
        env.insert("PYTHONNOUSERSITE", "1")
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.insert(variable, "1")
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        self.process.start(sys.executable, ["-m", "vcg_biomarker.mimic.pilot_reviewer_export",
            "--workspace", str(self.workspace), "--selection", source.relative_to(overlay).as_posix(), "--export-name", name])

    def _read_output(self):
        self.output.extend(self.process.readAllStandardOutput().data())
        # Bound diagnostic memory even if a failed decoder is very verbose.
        self.output[:] = self.output[-16000:]

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.button.setEnabled(True)
            self.status.setText("Exportprozess konnte nicht gestartet werden: " + self.process.errorString())

    def _finished(self, code, exit_status):
        self._read_output()
        self.button.setEnabled(True)
        try:
            if code != 0 or exit_status != QProcess.ExitStatus.NormalExit or self.destination is None:
                raise ValueError(self.output.decode("utf-8", errors="replace")[-2500:] or "Prozess fehlgeschlagen")
            manifest = json.loads((self.destination/"export_complete.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise ValueError("Abschlussmanifest fehlt oder ist unvollständig")
            self.completed = True
            self.status.setText(f"Export abgeschlossen: {manifest['n_cases']} Fälle, {manifest['n_reviewers']} Beurteiler.\n"
                                f"Beurteilerpakete: {self.destination / 'reviewer_packets'}\nKoordinatorordner nicht weitergeben.")
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f"Export fehlgeschlagen; unvollständige Ordner nicht verwenden.\n{exc}")
