"""Compact, read-only laboratory references for one episode/marker result."""
import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QHeaderView, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)


REFERENCES = (
    ("potassium", "Kalium"), ("total_calcium", "Gesamtcalcium"),
    ("ionized_calcium", "Ionisiertes Calcium"), ("magnesium", "Magnesium"),
    ("blood_gas_ph", "Blutgas-pH"),
)


def _text(value):
    if value is None or str(value) in {"nan", "NaT", "<NA>"}:
        return "—"
    return str(value)


def _number(value):
    try:
        number = float(value)
        return f"{number:.4g}" if math.isfinite(number) else "—"
    except (TypeError, ValueError):
        return "—"


class RecoveryReferenceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        note = QLabel(
            "Restabweichung = Markerwert beim ersten EKG nach beobachteter Labornormalisierung "
            "minus eigener Vor-Ereignis-Baseline, in der Einheit des ausgewählten Markers. "
            "Keine kausale Therapiewirkung; kein Beleg anhaltender Normalisierung. "
            "— bedeutet fehlend oder nicht auswertbar, nicht normal."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(5, 5)
        self.table.setHorizontalHeaderLabels([
            "Laborreferenz", "Laborzeitpunkt", "EKG-Abstand (h)", "Restabweichung", "Status / Grund",
        ])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        vertical = self.table.verticalHeader()
        horizontal = self.table.horizontalHeader()
        assert vertical is not None and horizontal is not None
        vertical.hide()
        horizontal.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        horizontal.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setWordWrap(True)
        layout.addWidget(self.table)
        self.set_result({})

    def set_result(self, result):
        row = result if isinstance(result, dict) else {}
        self.title.setText(
            f"Marker: {_text(row.get('marker_id'))} · Episode: {_text(row.get('episode_id'))} · "
            f"Verlauf: {_text(row.get('status'))}" if row else "Eine Episodenzeile auswählen"
        )
        for index, (group, label) in enumerate(REFERENCES):
            prefix = "" if group == "potassium" else group + "__"
            reference = "reference_" if group == "potassium" else f"reference_{group}_"
            status = _text(row.get(prefix + "normalization_status"))
            reference_status = _text(row.get(reference + "normalization_status"))
            measured = (_text(row.get("status")) == "evaluable" and status == "evaluable"
                        and reference_status == "observed")
            if prefix + "normalization_status" not in row:
                explanation = "Nicht berechnet / nicht im Resultset"
            elif measured:
                explanation = "Auswertbar · Forschungsbefund"
            else:
                reason = row.get(prefix + "normalization_reason")
                if reference_status != "observed":
                    reason = row.get(reference + "normalization_reason", reason)
                reason_text = _text(reason)
                if reason_text in {"", "—"}:
                    reason_text = _text(row.get("failure_reason"))
                explanation = f"Nicht auswertbar: {reason_text}"
            if measured and _number(row.get(prefix + "normalization_residual")) == "—":
                explanation = "Inkompatibles Ergebnis: endlicher Restwert fehlt"
            cells = [label, _text(row.get(prefix + "normalization_reference_time")),
                     _number(row.get(prefix + "normalization_ecg_lag_hours")),
                     _number(row.get(prefix + "normalization_residual")) if measured else "—",
                     explanation]
            tooltip = (
                f"Assay/itemid: {_text(row.get(reference + 'itemid'))}; "
                f"Einheit: {_text(row.get(reference + 'unit'))}; "
                f"berichteter Bereich: {_number(row.get(reference + 'lower'))} – "
                f"{_number(row.get(reference + 'upper'))}; "
                f"Ergebnis verfügbar ab: {_text(row.get(reference + 'result_available_at'))}. "
                "Laborzeitpunkt und Ergebnisverfügbarkeit sind nicht identisch."
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(tooltip)
                self.table.setItem(index, column, item)
        self.table.resizeRowsToContents()
