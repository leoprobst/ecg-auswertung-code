"""Selected paired result: separate samples, inference and limited adjustment."""
import math
import numpy as np
from collections.abc import Mapping

from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView, QAbstractItemView, QComboBox
import pyqtgraph as pg


def _finite(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _count(value):
    number = _finite(value)
    return str(int(number)) if number is not None and number >= 0 and number.is_integer() else "—"


def _label(value):
    return value if isinstance(value, str) else ""


class PairedEffectWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.model_selector = QComboBox()
        self.model_selector.setAccessibleName("Modell der longitudinalen Sensitivitaetsanalyse")
        self.model_selector.addItem("Timing / Signalqualitaet", "timing")
        self.model_selector.addItem("Ausgangsklinik + Timing / Signalqualitaet", "clinical")
        self.model_selector.addItem("Gemeinsame Laborantwort + Ausgangsklinik", "joint")
        layout.addWidget(self.model_selector)
        self._source_row = {}
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(100)
        self.plot.setMaximumHeight(150)
        self.plot.setLabel("bottom", "Steigung standardisierter empirischer Ränge")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").setTicks([[(2, "Alle Paare"), (1, "Gleiche Stichprobe"), (0, "Timing / Qualität")]])
        self.plot.getAxis("left").setWidth(140)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self._apply_palette()
        layout.addWidget(self.plot)
        self.table = QTableWidget(3, 5)
        self.table.setHorizontalHeaderLabels(["Analyse", "Effekt [95-%-KI]", "Paare / Patienten", "q global", "Einordnung"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(130)
        layout.addWidget(self.table)
        self.note = QLabel("Timing/Qualität berücksichtigt nur EKG-Abstand und beide Qualitätswerte. "
            "Keine vollständige klinische Adjustierung oder Kausalität. Primäre und Sensitivitäts-q-Werte "
            "gehören zu getrennten Testfamilien. Fehlendes KI wird nicht durch null ersetzt.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.set_result({})
        self.model_selector.currentIndexChanged.connect(lambda _: self.set_result(self._source_row))

    def _apply_palette(self):
        self.plot.setBackground(self.palette().color(QPalette.ColorRole.Base))
        color = self.palette().color(QPalette.ColorRole.Text)
        for name in ("bottom", "left"):
            self.plot.getAxis(name).setTextPen(color)
            self.plot.getAxis(name).setPen(color)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "plot"):
            self._apply_palette()

    def set_result(self, row):
        row = row if isinstance(row, Mapping) else {}
        self._source_row = dict(row)
        clinical = self.model_selector.currentData() == "clinical"
        joint = self.model_selector.currentData() == "joint"
        model = "baseline_clinical_timing_quality_sensitivity" if clinical else "timing_quality_sensitivity_only"
        model_label = "Ausgangsklinik + QC" if clinical else "Timing / Qualitaet"
        if joint:
            model = "joint_physiology_baseline_timing_quality_sensitivity"
            model_label = "Laborantwort + Klinik"
        if clinical or joint:
            prefix = "joint_" if joint else "clinical_"
            # Never retain timing-model values when a clinical field is absent.
            row = {key: value for key, value in row.items() if not key.startswith("adjusted_") and key != "unadjusted_same_sample_rank_slope"}
            row.update({key.removeprefix(prefix): value for key, value in self._source_row.items() if key.startswith(prefix)})
        self.plot.getAxis("left").setTicks([[(2, "Alle Paare"), (1, "Gleiche Stichprobe"), (0, model_label)]])
        self.note.setText(
            ("Ausgangsklinik: Alter, Geschlecht und Herzfrequenz des ersten EKGs; dazu EKG-Abstand und beide Qualitaetswerte. "
             if clinical else "Timing/Qualitaet: EKG-Abstand und beide Qualitaetswerte. ")
            + "Keine vollstaendige klinische Adjustierung oder Kausalitaet. Schweregrad, Begleittherapien und Indikation bleiben unkontrolliert. "
            "Separate q-Familien je Modell; kein nachtraegliches Auswaehlen des guenstigsten Modells. Fehlendes KI bleibt fehlend.")
        if joint:
            self.note.setText("Bedingte Laborantwort: andere zeitgleiche Aenderungen von K, ionisiertem Ca, Mg und pH plus Ausgangsklinik, Zeit und Qualitaet. "
                "Kein Ersatz von ionisiertem durch Gesamtcalcium. Keine isolierte kausale Elektrolytwirkung; Volumen und Therapie bleiben unkontrolliert. "
                "Eigene q-Familie, eigene vollstaendige Stichprobe; fehlende Referenzen werden nicht imputiert.")
        primary_ok = _label(row.get("association_status")) == "estimable"
        self.plot.clear()
        self.rendered_effects = []
        self.table.clearContents()
        self.title.setText(f"{row.get('design', '')} · {row.get('feature_id', '')}" if row else "Ergebniszeile für Effektvergleich auswählen.")
        self.plot.setVisible(bool(row))
        recognized = _label(row.get("adjusted_model")) == model
        adjusted_ok = recognized and _label(row.get("adjusted_status")) == "estimable"
        rows = [
            ("Unadjustiert · alle Paare", row.get("spearman_rho"),
             row.get("rank_slope_ci_low") if primary_ok else None,
             row.get("rank_slope_ci_high") if primary_ok else None,
             row.get("n_pairs"), row.get("n_patients"), row.get("global_q_value") if primary_ok else None,
             "Primärfamilie" if primary_ok else "Deskriptiv; Inferenz fehlt / nicht schätzbar"),
            ("Unadjustiert · gleiche Stichprobe", row.get("unadjusted_same_sample_rank_slope") if adjusted_ok else None,
             None, None, row.get("adjusted_n_complete") if recognized else None,
             row.get("adjusted_n_patients") if recognized else None, None, "Nur deskriptiver Vergleich; kein KI berechnet"),
            ("Sensitivität · Timing/Qualität", row.get("adjusted_rank_slope") if adjusted_ok else None,
             row.get("adjusted_ci_low") if adjusted_ok else None, row.get("adjusted_ci_high") if adjusted_ok else None,
             row.get("adjusted_n_complete") if recognized else None, row.get("adjusted_n_patients") if recognized else None,
             row.get("adjusted_global_q_value") if adjusted_ok else None,
             "Separate Sensitivitätsfamilie" if adjusted_ok else (_label(row.get("adjusted_abstention_reason")) or "Nicht berechnet") if recognized else "Nicht vorhanden oder unbekanntes Modell"),
        ]
        if clinical:
            rows[2] = ("Sensitivitaet: Ausgangsklinik + QC", *rows[2][1:])
        if joint:
            rows[2] = ("Sensitivitaet: gemeinsame Laborantwort", *rows[2][1:])
        colors = ("#888888", "#E69F00", "#0072B2")
        for i, (name, effect, low, high, pairs, patients, q, status) in enumerate(rows):
            effect, low, high, q = map(_finite, (effect, low, high, q))
            ci = effect is not None and low is not None and high is not None and low <= effect <= high
            text = "—" if effect is None else f"{effect:+.3f}"
            text += f" [{low:+.3f}; {high:+.3f}]" if ci else " [KI fehlt]"
            qtext = f"{q:.3g}" if q is not None and 0 <= q <= 1 else "—"
            for j, value in enumerate((name, text, f"{_count(pairs)} / {_count(patients)}", qtext, status)):
                self.table.setItem(i, j, QTableWidgetItem(value if row else "—"))
            if effect is not None:
                self.plot.plot([effect], [2-i], pen=None, symbol="o", symbolSize=8, symbolBrush=colors[i], symbolPen=colors[i])
                if ci:
                    self.plot.plot([low, high], [2-i, 2-i], pen=pg.mkPen(colors[i], width=2))
                self.rendered_effects.append(dict(row=i, effect=effect, ci=(low, high) if ci else None))
        if self.rendered_effects:
            self.plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#999999", style=Qt.PenStyle.DashLine)))
        self.plot.setYRange(-.5, 2.5, padding=0)
        self.plot.enableAutoRange(axis="x")
