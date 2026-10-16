"""Physical ruler calibration in Qt logical pixels, independent of signal data."""
import math
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QWidget, QDialog, QVBoxLayout, QLabel, QDoubleSpinBox, QDialogButtonBox

RULER_PIXELS = 200.0


def pixels_per_mm(measured_mm):
    if isinstance(measured_mm, bool) or not isinstance(measured_mm, (int, float)):
        raise ValueError('Gemessene Länge muss eine Zahl sein')
    if not math.isfinite(measured_mm) or not 10 <= measured_mm <= 200:
        raise ValueError('Gemessene Länge muss zwischen 10 und 200 mm liegen')
    return RULER_PIXELS / measured_mm


class ScreenRuler(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(260, 70)
        self.setAccessibleName('Kalibrierstrecke: 200 logische Pixel zwischen den äußeren Strichen')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(self.palette().windowText().color(), 1))
        painter.drawLine(QPointF(30, 30), QPointF(230, 30))
        for x in (30, 230):
            painter.drawLine(QPointF(x, 17), QPointF(x, 43))
        painter.drawText(30, 62, '← diese Strecke mit Lineal messen →')
        painter.end()


class ScreenCalibrationDialog(QDialog):
    def __init__(self, current_pixels_per_mm, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Bildschirmmaßstab kalibrieren')
        layout = QVBoxLayout(self)
        instructions = QLabel('Mit einem echten Lineal den Abstand zwischen den beiden äußeren '
            'senkrechten Strichen messen und unten in Millimetern eintragen. '
            'Nicht die Breite des Fensters messen. Auf dem Bildschirm kalibrieren, '
            'auf dem anschließend das EKG betrachtet wird.')
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.ruler = ScreenRuler(self)
        layout.addWidget(self.ruler, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.measured = QDoubleSpinBox()
        self.measured.setRange(10, 200)
        self.measured.setDecimals(2)
        self.measured.setSuffix(' mm gemessen')
        self.measured.setValue(RULER_PIXELS / current_pixels_per_mm)
        layout.addWidget(self.measured)
        note = QLabel('Übernehmen aktiviert die maßstäbliche Ansicht (Fensteranpassung aus). '
            'Bei Monitorwechsel oder geänderter DPI-Skalierung erneut kalibrieren. '
            'Gilt nur für diese Sitzung; PDF-Maßstab und Messwerte ändern sich nicht.')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Kalibrierung übernehmen')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(480, 300)

    def calibration(self):
        return pixels_per_mm(self.measured.value())
