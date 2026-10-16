"""Availability of independently referenced laboratory changes on identical pairs."""
from collections.abc import Mapping
import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

LABELS = (("potassium", "Kalium"), ("calcium_total", "Gesamtcalcium"),
          ("calcium_ionized", "Ionisiertes Calcium"), ("magnesium", "Magnesium"), ("ph", "pH"))


def _count(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(number) if math.isfinite(number) and number >= 0 and number.is_integer() else None


def coverage_rows(summary):
    """Unknown counts are not zeros; impossible counts invalidate that row."""
    if not isinstance(summary, Mapping) or not isinstance(summary.get("status"), str) or summary["status"] != "computed":
        return None, []
    total = _count(summary.get("n_pairs"))
    if total is None:
        return None, []
    analytes = summary.get("analytes")
    analytes = analytes if isinstance(analytes, Mapping) else {}
    rows = []
    records = [(label, analytes.get(name)) for name, label in LABELS]
    records.append(("K + ion. Ca + Mg + pH", dict(n_pairs=summary.get("n_pairs_complete_core"),
        n_patients=summary.get("n_patients_complete_core"))))
    for label, record in records:
        record = record if isinstance(record, Mapping) else {}
        pairs, patients = _count(record.get("n_pairs")), _count(record.get("n_patients"))
        valid = pairs is not None and pairs <= total and (patients is None or patients <= pairs)
        rows.append(dict(label=label, n_pairs=pairs if valid else None,
            n_patients=patients if valid else None,
            percent=100*pairs/total if valid and total else None))
    core = rows[-1]
    for row in (rows[0], rows[2], rows[3], rows[4]):
        for name in ("n_pairs", "n_patients"):
            if core[name] is not None and row[name] is not None and core[name] > row[name]:
                core.update(n_pairs=None, n_patients=None, percent=None)
    return total, rows


class JointLabCoverageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.note = QLabel("Gueltige Laborveraenderungen am selben Patienten-/EKG-Paar, nach Paarselektion. "
            "Gemeinsame Referenzen sind Voraussetzung fuer das gemeinsame Modell, nicht fuer sich ein Wirksamkeitsnachweis. "
            "Gesamtcalcium ersetzt kein ionisiertes Calcium.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Anteil der ausgewaehlten EKG-Paare [%]")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").setWidth(360)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        layout.addWidget(self.plot)
        self.rows = []
        self._palette()
        self.set_summary(None)

    def _palette(self):
        self.plot.setBackground(self.palette().color(QPalette.ColorRole.Base))
        for name in ("bottom", "left"):
            self.plot.getAxis(name).setTextPen(self.palette().color(QPalette.ColorRole.Text))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "plot"):
            self._palette()

    def set_summary(self, summary):
        total, self.rows = coverage_rows(summary)
        self.plot.clear()
        self.plot.getAxis("left").setTicks([])
        self.plot.setVisible(total is not None)
        if total is None:
            self.status.setText("Gemeinsame Laborverfuegbarkeit nicht berechnet oder Ergebnisvertrag unvollstaendig.")
            return
        missing = sum(row["n_pairs"] is None for row in self.rows)
        self.status.setText(f"{total:,} ausgewaehlte EKG-Paare. Balken: Referenzpaare, nicht unabhaengige Patienten. "
            + (f"{missing} Angaben fehlen oder sind widerspruechlich; keine Nullwerte unterstellt." if missing else
               "Patientenzahlen stehen getrennt bei jeder Referenz."))
        ticks = []
        for index, row in enumerate(self.rows):
            y = len(self.rows)-1-index
            pairs = str(row["n_pairs"]) if row["n_pairs"] is not None else "?"
            patients = str(row["n_patients"]) if row["n_patients"] is not None else "?"
            ticks.append((y, f"{row['label']} | {pairs} Paare / {patients} Pat."))
            if row["percent"] is not None:
                self.plot.addItem(pg.BarGraphItem(x0=0, y=[y], width=[row["percent"]], height=.55,
                    brush="#E69F00" if index == len(self.rows)-1 else "#56B4E9", pen=None))
        self.plot.getAxis("left").setTicks([ticks])
        self.plot.setXRange(0, 100, padding=.01)
        self.plot.setYRange(-.5, len(self.rows)-.5, padding=0)
