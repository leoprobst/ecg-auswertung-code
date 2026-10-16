"""Blinded, local 12-lead ECG exercise using annotated Challenge training ECGs."""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QTextBrowser, QVBoxLayout,
)

from ...mimic.ecg_lab import ECGCaseLibrary
from ...mimic.storage import atomic_write_json
from ..practice_cases import choose_practice_case, practice_solution
from .expert_ecg_lab import ExpertECGLab


class ECGPracticeDialog(QDialog):
    selectionReady = pyqtSignal(object)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._busy = False
        self._seen: set[Path] = set()
        self._case = None
        self.setWindowTitle("EKG-Übungsmodus · verblindete Befundung")
        self.resize(1450, 900)
        layout = QVBoxLayout(self)
        self.status = QLabel(
            "Zufällige, befundete 12-Kanal-Aufnahme aus lokalen Challenge-Trainingsdaten. "
            "Die Auswahl ist für die Übung angereichert, nicht prävalenzrepräsentativ. "
            "Keine Locked-Test-Aufnahme wird geöffnet."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        self.next_button = QPushButton("Zufalls-EKG laden")
        self.next_button.setToolTip("Nur lokale unabhängige Challenge-Trainingsquellen; Hintergrundsuche blockiert die GUI nicht.")
        self.next_button.clicked.connect(self.load_random)
        actions.addWidget(self.next_button)
        self.reveal_button = QPushButton("Lösung und Hintergrund aufdecken")
        self.reveal_button.setEnabled(False)
        self.reveal_button.clicked.connect(self.reveal)
        actions.addWidget(self.reveal_button)
        actions.addStretch()
        layout.addLayout(actions)
        self.solution = QTextBrowser()
        self.solution.setVisible(False)
        self.solution.setMaximumHeight(190)
        layout.addWidget(self.solution)
        self.lab = ExpertECGLab()
        self.lab.blind.setChecked(True)
        self.lab.blind.setEnabled(False)
        self.viewport = QScrollArea()
        self.viewport.setWidgetResizable(True)
        self.viewport.setWidget(self.lab)
        layout.addWidget(self.viewport, 1)
        self.selectionReady.connect(self._selection_ready)

    def load_random(self) -> None:
        if self._busy:
            return
        root = self.config.external_ecg_root
        if root is None or not Path(root).is_dir():
            QMessageBox.warning(self, "Übungsmodus", "Lokale Challenge-Trainingsdaten sind nicht konfiguriert.")
            return
        self._busy = True
        self.next_button.setEnabled(False)
        self.reveal_button.setEnabled(False)
        self.solution.hide()
        self.status.setText("Übungsfall wird lokal ausgewählt und geladen …")
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "VCG" / "practice_sessions"
        workspace = base / uuid.uuid4().hex
        seen = frozenset(self._seen)

        def work() -> None:
            try:
                selected = choose_practice_case(Path(root), excluded_headers=seen)
                local_config = replace(self.config, work_root=workspace)
                library = ECGCaseLibrary(local_config)
                imported = library.import_case(
                    selected.header, pseudonym="Verblindetes Übungs-EKG",
                    retain_source_path=False, copy_original=False,
                )
                # A WFDB header includes Dx comments. Keep them only in the
                # solution held by this dialog, never in the displayed case.
                case_document = imported.to_dict()
                case_document["metadata"]["source_metadata"] = {
                    "record_name": "verblindet", "comments": [],
                }
                case_document["metadata"]["age"] = None
                case_document["metadata"]["sex"] = None
                atomic_write_json(library.root / imported.case_id / "case.json", case_document)
                self.selectionReady.emit((library, imported.case_id, selected, None))
            except Exception as exc:
                self.selectionReady.emit((None, None, None, f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=work, name="ecg-practice-load", daemon=True).start()

    def _selection_ready(self, result: tuple) -> None:
        self._busy = False
        self.next_button.setEnabled(True)
        library, case_id, selected, error = result
        if error:
            self.status.setText("Übungsfall nicht geladen")
            QMessageBox.warning(self, "Übungsmodus", error)
            return
        try:
            self.lab.blind.setChecked(True)
            self.lab.blind.setEnabled(False)
            self.lab.load_case(library, case_id)
        except Exception as exc:
            self.status.setText("Fallwechsel abgebrochen; bisheriger Fall bleibt bestehen")
            QMessageBox.warning(self, "Übungsmodus", str(exc))
            return
        self._case = selected
        self._seen.add(selected.header)
        self.solution.hide()
        self.lab._place_heart_overlay()
        self.reveal_button.setEnabled(True)
        self.status.setText(
            "Verblindetes EKG geladen. Befunde und automatische Forschungsansichten sind verborgen. "
            "Zirkel und manuelle Befundung bleiben verfügbar."
        )

    def reveal(self) -> None:
        if self._case is None:
            return
        answer = practice_solution(self._case)
        labels = ", ".join(str(item).replace("_", " ") for item in answer["ecg_annotation_labels"]) or "keine zugeordnet"
        unknown = ", ".join(answer["unmapped_snomed_codes"]) or "keine"
        age = f"{answer['age']:g}" if answer["age"] is not None else "unbekannt"
        self.solution.setPlainText(
            f"EKG-Annotationen: {labels}\n"
            f"Nicht übersetzte SNOMED-Codes: {unknown}\n"
            f"Quelle: {answer['source']} · Record: {answer['record_id']} · Alter: {age} · Geschlecht: {answer['sex']}\n"
            "Aussagegrenze: Befundinterpretationen desselben EKGs, kein unabhängiger Echo-/Labor-/Mappingnachweis. "
            "Nicht genannte Diagnosen sind unbekannt, nicht negativ."
        )
        self.solution.show()
        self.lab.blind.setChecked(False)
        self.lab.blind.setEnabled(True)
        self.status.setText("Lösung aufgedeckt · den eigenen Befund mit den dokumentierten EKG-Annotationen vergleichen.")
        self.reveal_button.setEnabled(False)
