"""Descriptive paired interval changes; no implied temporal or clinical verdict."""
import math
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
import pyqtgraph as pg


class PairedChangeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(240)
        self.plot.setLabel('bottom', 'Differenz aktuell − Vergleich', units='ms')
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        layout.addWidget(self.plot)
        self.displayed_rows = []
        self.set_rows([])

    def set_rows(self, rows):
        self.plot.clear()
        self.displayed_rows = [r for r in rows
            if r.get('status') == 'descriptive_only' and r.get('unit') == 'ms'
            and isinstance(r.get('delta_current_minus_comparator'), (int, float))
            and not isinstance(r['delta_current_minus_comparator'], bool)
            and math.isfinite(r['delta_current_minus_comparator'])]
        omitted = len(rows)-len(self.displayed_rows)
        self.note.setText(f'{len(self.displayed_rows)} vergleichbare Intervalle; {omitted} nicht dargestellt. '
            'Deskriptive Differenz, keine Signifikanz oder Diagnose. „Aktuell“ bezeichnet die Auswahl, '
            'nicht eine belegte zeitliche Reihenfolge. Keine Messfehler-Konfidenzintervalle vorhanden.')
        self.plot.getAxis('left').setTicks([[(i, r['measurement']) for i, r in enumerate(self.displayed_rows)]])
        if not self.displayed_rows:
            self.plot.setTitle('Keine vergleichbaren gepaarten Messwerte')
            return
        self.plot.setTitle('Gepaarte Intervalländerungen')
        self.plot.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen('#888888')))
        maximum = max(1., max(abs(r['delta_current_minus_comparator']) for r in self.displayed_rows))
        for i, row in enumerate(self.displayed_rows):
            delta = row['delta_current_minus_comparator']
            color = '#0072b2' if delta >= 0 else '#d55e00'
            self.plot.plot([0, delta], [i, i], pen=pg.mkPen(color, width=2))
            self.plot.plot([delta], [i], pen=None, symbol='o', symbolBrush=color, symbolSize=8)
        self.plot.setXRange(-maximum*1.15, maximum*1.15, padding=0)
        self.plot.setYRange(-.7, len(self.displayed_rows)-.3, padding=0)

    def set_dark(self, dark):
        self.plot.setBackground('#111923' if dark else '#fffdf9')
