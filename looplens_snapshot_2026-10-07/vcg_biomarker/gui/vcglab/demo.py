"""Prüf-GUI des VCG-Labors: Vorschaufenster + Laborfenster, Szenarien der Herzillustration oder WFDB-Datei.

``py -3.12 -m vcg_biomarker.gui.vcglab [--wfdb PFAD] [--scenario NAME] [--quick]``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from ..heart3d.demo import SCENARIOS
from .analysis import VCGRecord, analyze_signal, load_wfdb, quick_record
from .widgets import MiniVCGWidget


def record_from_scenario(name: str, quick: bool = False) -> VCGRecord:
    tl = SCENARIOS[name]().timeline()
    sig, fs, names = tl.signal, tl.fs, tl.lead_names
    if quick:
        return quick_record(sig, fs, names, name)
    try:
        return analyze_signal(sig, fs, names, name)
    except Exception:
        return quick_record(sig, fs, names, name)


def record_from_wfdb(path: str, quick: bool = False) -> VCGRecord:
    sig, fs, names, meta = load_wfdb(path)
    label = Path(path).stem
    if quick:
        return quick_record(sig, fs, names, label, meta.get("age"), meta.get("sex"))
    try:
        return analyze_signal(sig, fs, names, label, meta.get("age"), meta.get("sex"))
    except Exception as exc:
        rec = quick_record(sig, fs, names, label, meta.get("age"), meta.get("sex"))
        rec.notes.append(f"Hauptprojekt-Analyse fehlgeschlagen: {exc}")
        return rec


class DemoWindow(QMainWindow):
    def __init__(self, initial: Optional[VCGRecord] = None, dark: bool = True):
        super().__init__()
        self._dark = bool(dark)
        self.setWindowTitle("VCG-Labor – Prüf-GUI")
        central = QWidget()
        layout = QVBoxLayout(central)
        row = QHBoxLayout()
        row.addWidget(QLabel("Szenario"))
        self.combo = QComboBox()
        self.combo.addItems(list(SCENARIOS))
        row.addWidget(self.combo, 1)
        start = QPushButton("Laden")
        start.clicked.connect(self._load_scenario)
        row.addWidget(start)
        wfdb_button = QPushButton("WFDB-Datei …")
        wfdb_button.clicked.connect(self._pick_wfdb)
        row.addWidget(wfdb_button)
        layout.addLayout(row)
        hint = QLabel("Die kleine Ansicht entspricht dem geplanten Vorschaufenster im ECG-Labor; „Vergrößern“ öffnet das VCG-Labor "
                      "mit drei Ebenen, 3D-Schleife, Skalarspuren, Maßen und Schleifenlesung.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.mini = MiniVCGWidget(self, auto_analysis=False, dark=self._dark)
        layout.addWidget(self.mini, 0)
        layout.addStretch(1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.setCentralWidget(central)
        self.resize(560, 420)
        if initial is not None:
            self.mini.set_record(initial)
            self.status.setText(initial.label)
        else:
            self._load_scenario()

    def _load_scenario(self) -> None:
        name = self.combo.currentText()
        rec = record_from_scenario(name)
        self.mini.set_record(rec)
        self.status.setText(f"{rec.label} · {len(rec.beats)} Schläge · {rec.source} · {'; '.join(rec.notes) or 'ok'}")

    def _pick_wfdb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "WFDB-Header wählen", "", "WFDB (*.hea)")
        if not path:
            return
        try:
            rec = record_from_wfdb(path)
        except Exception as exc:
            self.status.setText(f"Laden fehlgeschlagen: {exc}")
            return
        self.mini.set_record(rec)
        self.status.setText(f"{rec.label} · {len(rec.beats)} Schläge · {rec.source}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prüf-GUI für das VCG-Labor")
    parser.add_argument("--wfdb", help="WFDB-Record (Pfad mit oder ohne .hea)")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default=None)
    parser.add_argument("--quick", action="store_true", help="nur Schnellschätzung, ohne Analysekette des Hauptprojekts")
    parser.add_argument("--light", action="store_true", help="helles Farbschema des Programms")
    args = parser.parse_args(argv)
    if QApplication.instance() is None:
        from PyQt6.QtCore import Qt
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication(sys.argv)
    from ..theme import generate_stylesheet
    app.setStyleSheet(generate_stylesheet(not args.light))  # gleiches Farbschema wie das Hauptprogramm
    initial = None
    if args.wfdb:
        initial = record_from_wfdb(args.wfdb, args.quick)
    elif args.scenario:
        initial = record_from_scenario(args.scenario, args.quick)
    window = DemoWindow(initial, dark=not args.light)
    window.show()
    window.mini.open_window()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
