"""Read-only coordinator return review, isolated from production labels."""
from pathlib import Path
import logging
import json

import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTableView

from ...mimic.reviewer_import import import_review_responses, validate_export_packets
from ...mimic.reviewer_packet import ReviewerPacket


class _ReturnModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = pd.DataFrame()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self.frame.iat[index.row(), index.column()]
        return "—" if pd.isna(value) else str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return str(self.frame.columns[section])
        return None

    def set_frame(self, frame):
        self.beginResetModel()
        self.frame = frame
        self.endResetModel()


class _ImportTask(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, folders, files, parent=None):
        super().__init__(parent)
        self.folders, self.files = folders, files

    def run(self):
        try:
            manifest_path = self.folders[0].parent.parent/"export_complete.json"
            if manifest_path.is_symlink() or manifest_path.stat().st_size > 4*1024**2:
                raise ValueError("Invalid original export manifest")
            with manifest_path.open("rb") as handle:
                raw = handle.read(4*1024**2+1)
            if len(raw) > 4*1024**2:
                raise ValueError("Original export manifest exceeds size limit")
            manifest = json.loads(raw)
            packets = [ReviewerPacket(folder) for folder in self.folders]
            validate_export_packets(manifest, packets)
            records, coverage = import_review_responses(packets, self.files)
            if not records.empty:
                coverage = coverage.merge(records[["review_case_id", "reviewer_slot", "reviewer_id", "reviewer_verdict", "recorded_at"]],
                    on=["review_case_id", "reviewer_slot"], how="left", validate="one_to_one")
            self.ready.emit(coverage)
        except Exception as exc:
            logging.getLogger(__name__).exception("Reviewer response import failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ReviewerImportWidget(QWidget):
    def __init__(self, workspace: Path, parent=None):
        super().__init__(parent)
        self.workspace = Path(workspace).resolve()
        self.task: _ImportTask | None = None
        layout = QVBoxLayout(self)
        note = QLabel("Koordinatoransicht: Antworten gegen die ursprünglich ausgegebenen Pakete prüfen. "
            "Fehlende Antworten bleiben fehlend. Keine automatische Umetikettierung, keine klinische Validierung. "
            "Beurteileridentität, Unabhängigkeit und Verblindung sind organisatorisch zu prüfen. "
            "Diese Ansicht speichert keine Änderungen und berechnet keine populationsbezogene Genauigkeit.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.button = QPushButton("Ausgegebenen Export und Antwortdateien auswählen")
        self.button.clicked.connect(self.choose)
        layout.addWidget(self.button)
        self.status = QLabel("Noch keine Antworten geprüft.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.model = _ReturnModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        layout.addWidget(self.table)

    @property
    def busy(self):
        return self.task is not None and self.task.isRunning()

    def choose(self):
        folder = QFileDialog.getExistingDirectory(self, "Ursprünglich ausgegebener Export", str(self.workspace/"overlay"/"review_exports"))
        if not folder:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Zurückgegebene Antworten (keine Auswahl zeigt alle als fehlend)", str(self.workspace/"overlay"), "JSON (*.json)")
        try:
            self.start_import(Path(folder), [Path(name) for name in files])
        except (OSError, ValueError) as exc:
            self.model.set_frame(pd.DataFrame())
            self.status.setText(f"Nicht gestartet: {exc}")

    def start_import(self, export, files):
        if self.busy:
            raise ValueError("Eine Prüfung läuft bereits")
        self.model.set_frame(pd.DataFrame())
        root = (self.workspace/"overlay"/"review_exports").resolve()
        export = Path(export).resolve()
        if not export.is_relative_to(root) or not (export/"export_complete.json").is_file():
            raise ValueError("Abgeschlossener Originalexport im Pilot-Overlay erforderlich")
        folders = sorted((export/"reviewer_packets").iterdir())
        if len(folders) < 2 or any(not p.is_dir() or p.is_symlink() or not p.resolve().is_relative_to(export) for p in folders):
            raise ValueError("Mindestens zwei unveränderte Beurteilerpakete erforderlich")
        self.button.setEnabled(False)
        self.status.setText("Prüfung läuft im Hintergrund; Fenster bis zum Abschluss offen lassen.")
        self.task = _ImportTask(folders, list(files), self)
        self.task.ready.connect(self._ready)
        self.task.failed.connect(lambda message: self.status.setText("Import abgelehnt: " + message))
        self.task.finished.connect(lambda: self.button.setEnabled(True))
        self.task.start()

    def _ready(self, frame):
        self.model.set_frame(frame)
        received = int(frame.status.eq("received").sum())
        self.status.setText(f"{received} von {len(frame)} Zuweisungen beantwortet; {len(frame)-received} fehlen. "
                            "Provenienz konsistent, nicht klinisch validiert. Keine Labels verändert.")

    def closeEvent(self, event):
        if self.busy:
            event.ignore()
        else:
            super().closeEvent(event)
