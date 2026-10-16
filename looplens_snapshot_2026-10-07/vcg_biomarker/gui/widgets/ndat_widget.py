"""Research downslope timings; scatter points are not anatomical conduction."""
import math

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QAbstractItemView


class NdatWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Zeit nach QRS-Beginn", units="ms")
        self.plot.setLabel("left", "Brustwandableitung")
        self.plot.getAxis("left").setTicks([[(i, f"V{i+1}") for i in range(6)]])
        self.plot.setMinimumHeight(120)
        layout.addWidget(self.plot)
        self.table = QTableWidget(6, 5)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalHeaderLabels(["Ableitung", "Zeit [ms]", "Schläge", "MAD [ms]", "Qualität / Grund"])
        layout.addWidget(self.table)
        self.set_result({})

    def set_result(self, payload):
        self.plot.clear()
        self.note.setText(
            "NDAT: steilste negative QRS-Flanke, Kurz-EKG-Forschungsanpassung. "
            "Keine invasive Aktivierungszeit oder Monitoring-Empfehlung. "
            f"Status: {payload.get('status', 'nicht berechnet')}; "
            f"Grund: {payload.get('reason') or '—'}; "
            f"Pacing: {payload.get('pacing_status', 'unbekannt')}. "
            f"QRS-Qualität: {payload.get('n_qrs_quality_eligible', '—')}/"
            f"{payload.get('n_delineated_beats', '—')} Schläge. "
            "Unbekannte Gerätebandbreite gilt nicht als verifiziert. MAD beschreibt "
            "Schlagstreuung, kein Konfidenzintervall. Fehlend bedeutet nicht normal."
        )
        x, y = [], []
        for i in range(6):
            lead = f"V{i+1}"
            data = payload.get("leads", {}).get(lead, {})
            value = data.get("time_ms")
            finite = isinstance(value, (int, float)) and math.isfinite(value)
            if finite and data.get("status") == "evaluable":
                x.append(value)
                y.append(i)
            cells = [lead, f"{value:.1f}" if finite else "nicht messbar",
                f"{data.get('n_assessable', 0)}/{data.get('n_beats', 0)}",
                str(data.get("mad_ms", "—")),
                "; ".join(data.get("reasons", [])) or data.get("status", "nicht berechnet")]
            for column, text in enumerate(cells):
                self.table.setItem(i, column, QTableWidgetItem(text))
        if x:
            self.plot.plot(x, y, pen=None, symbol="o", symbolBrush="#0072B2", symbolSize=9)
        self.plot.setYRange(-.5, 5.5)
        self.table.resizeColumnsToContents()
