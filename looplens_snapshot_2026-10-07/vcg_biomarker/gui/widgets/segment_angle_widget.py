"""Labelled fixed 3x3 temporal decomposition, no selected best cell."""
import math
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView


class SegmentAngleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        note = QLabel("Zeitdrittel, keine anatomischen Regionen. Zeilen: QRS; Spalten: J-Punkt bis T-Ende. "
            "Gewichtete paarweise Winkel im nativen 8-Lead-Raum, keine direkt ablesbaren 3D-Winkel. "
            "Forschungshypothese ohne klinische Grenzwerte. Blau wird mit größerem Winkel dunkler (0–180°); "
            "grau/— bedeutet nicht messbar, nicht normal. Alle neun Zellen bleiben sichtbar.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(3, 3)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalHeaderLabels(["Repol. früh", "Repol. mittel", "Repol. spät"])
        self.table.setVerticalHeaderLabels(["QRS früh", "QRS mittel", "QRS spät"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.set_result({})

    def set_result(self, result):
        if not isinstance(result, dict):
            result = {"status": "incompatible", "reason": "invalid_result_contract"}
        self.status.setText(f"Status: {result.get('status', 'nicht berechnet')}; "
                            f"Grund: {result.get('reason') or '—'}")
        for i in range(3):
            for j in range(3):
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(QColor("#dddddd"))
                item.setForeground(QColor("#222222"))
                self.table.setItem(i, j, item)
        cells = result.get("cells", [])
        if not isinstance(cells, list):
            self.status.setText("Inkompatibles Resultset: cells ist keine Liste")
            return
        seen = set()
        for cell in cells:
            if not isinstance(cell, dict):
                self.status.setText("Inkompatibles Resultset: ungültige Zelle")
                continue
            indices = (cell.get("qrs_third"), cell.get("repolarization_third"))
            if any(type(index) is not int for index in indices):
                self.status.setText("Inkompatibles Resultset: ungültige Zeitdrittel")
                continue
            i, j = indices[0]-1, indices[1]-1
            if not (0 <= i < 3 and 0 <= j < 3):
                continue
            item = self.table.item(i, j)
            if indices in seen:
                item.setText("mehrdeutig")
                item.setBackground(QColor("#dddddd"))
                item.setForeground(QColor("#222222"))
                item.setToolTip("Doppelte Zeitdrittel im Ergebnisvertrag")
                self.status.setText("Inkompatibles Resultset: doppelte Zeitdrittel")
                continue
            seen.add(indices)
            value = cell.get("angle_deg")
            if (type(value) in (int, float) and math.isfinite(value)
                    and cell.get("status") == "evaluable" and 0 <= value <= 180
                    and result.get("status") in {"evaluable", "partial"}):
                item.setText(f"{value:.1f}°")
                fraction = value / 180
                item.setBackground(QColor(int(235-235*fraction), int(245-130*fraction), int(255-77*fraction)))
                item.setForeground(QColor("white" if fraction > .6 else "black"))
            weight = cell.get("magnitude_pair_fraction")
            weight_text = (f"{100*weight:.2f} %" if type(weight) in (int, float)
                           and math.isfinite(weight) and 0 <= weight <= 1 else "unbekannt")
            item.setToolTip(f"Gewicht im Gesamtintegral: {weight_text}; "
                            f"Status: {cell.get('status')}; {cell.get('reason', '')}")
