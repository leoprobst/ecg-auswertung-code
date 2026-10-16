"""Read-only comparison of two already computed inverse-simulation hypotheses."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                             QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)

LEAD_GRID = (("I", "aVR", "V1", "V4"),
             ("II", "aVL", "V2", "V5"),
             ("III", "aVF", "V3", "V6"))
LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def common_display_limit(*signals: np.ndarray) -> float:
    """One full-range mV scale for measured and both simulated curves."""
    finite = [np.abs(np.asarray(signal)[np.isfinite(signal)]) for signal in signals]
    finite = [values for values in finite if values.size]
    return max(0.2, 1.1 * max(float(np.max(values)) for values in finite)) if finite else 0.2


def paired_qrs_residuals(time_ms: np.ndarray, measured: np.ndarray,
                         simulated_a: np.ndarray, simulated_b: np.ndarray,
                         qrs_samples: int | None) -> np.ndarray | None:
    """Early/late leadwise RMSE in mV on identical, jointly simulated QRS samples."""
    if not isinstance(qrs_samples, (int, np.integer)) or qrs_samples < 20:
        return None
    delta = np.diff(time_ms)
    if not np.all(np.isfinite(delta)) or np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-4):
        return None
    limit = int(qrs_samples) * float(delta[0])
    indices = np.flatnonzero((time_ms >= -float(delta[0]) / 2) & (time_ms < limit - float(delta[0]) / 2))
    if len(indices) < 20:
        return None
    halves = np.array_split(indices, 2)
    out = np.full((2, 12, 2), np.nan)
    for row, part in enumerate(halves):
        for lead in range(12):
            valid = (np.isfinite(measured[part, lead]) & np.isfinite(simulated_a[part, lead])
                     & np.isfinite(simulated_b[part, lead]))
            if np.count_nonzero(valid) < 5:
                continue
            samples = part[valid]
            out[row, lead, 0] = np.sqrt(np.mean((measured[samples, lead] - simulated_a[samples, lead]) ** 2))
            out[row, lead, 1] = np.sqrt(np.mean((measured[samples, lead] - simulated_b[samples, lead]) ** 2))
    return out


class HypothesisDuelDialog(QDialog):
    def __init__(self, *, time_ms: np.ndarray, measured: np.ndarray, simulated_a: np.ndarray,
                 simulated_b: np.ndarray, result_a, result_b, parent=None):
        super().__init__(parent)
        time_ms = np.asarray(time_ms, dtype=float)
        measured = np.asarray(measured, dtype=float)
        simulated_a = np.asarray(simulated_a, dtype=float)
        simulated_b = np.asarray(simulated_b, dtype=float)
        if time_ms.ndim != 1 or len(time_ms) < 2 or measured.shape != (len(time_ms), 12) or \
                simulated_a.shape != measured.shape or simulated_b.shape != measured.shape:
            raise ValueError("Hypothesenvergleich erfordert denselben Schlag und zwölf gleich lange Ableitungen")
        self.setWindowTitle("Hypothesen-Duell · identischer Schlag")
        self.resize(1450, 830)
        self.setStyleSheet("""
            QDialog { background: #0b111a; color: #e8f1f5; }
            QLabel { color: #dce8ee; }
            QPushButton { background: #203644; color: #e8f1f5; border: 1px solid #4a697a;
                          border-radius: 5px; padding: 6px 13px; }
        """)
        root = QVBoxLayout(self)
        title = QLabel("ZWEI MODELLHYPOTHESEN · EIN GEMESSENER SCHLAG")
        title.setStyleSheet("font-size: 16px; font-weight: 700; padding: 4px;")
        root.addWidget(title)
        legend = QLabel("Gemessen: weiß, durchgezogen   ·   A: türkis, gestrichelt   ·   B: bernstein, gepunktet")
        legend.setStyleSheet("font-size: 12px;")
        root.addWidget(legend)
        comparison = QHBoxLayout()
        self.summary_cards = []
        for letter, result, color in (("A", result_a, "#4fd1c5"), ("B", result_b, "#ffbc55")):
            card = QFrame(self)
            card.setStyleSheet(f"QFrame {{ background: #16232d; border-left: 3px solid {color}; border-radius: 4px; }}")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 6, 10, 6)
            heading = QLabel(f"{letter}  {result.hypothesis.label}  ·  Modellscore {result.score:.2f}")
            heading.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {color}; border: none;")
            heading.setWordWrap(True)
            card_layout.addWidget(heading)
            details = QLabel(f"12-Kanal-r {result.r_mean:.2f}  ·  QRS-Zeit ×{result.time_scale_factor:.2f}"
                             + ("  ·  Zeitskalierung am Limit" if result.time_scale_limited else ""))
            details.setStyleSheet("font-size: 11px; border: none;")
            details.setWordWrap(True)
            card_layout.addWidget(details)
            comparison.addWidget(card, 1)
            self.summary_cards.append(card)
        root.addLayout(comparison)
        residuals = paired_qrs_residuals(
            time_ms, measured, simulated_a, simulated_b,
            min(len(result_a.simulated_leads), len(result_b.simulated_leads))
            if hasattr(result_a, "simulated_leads") and hasattr(result_b, "simulated_leads") else None,
        )
        residual_title = QLabel("WO UNTERSCHEIDEN SICH DIE FITS? · QRS-Residuen je Ableitung, gleicher Schlag")
        residual_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #c7dbe4;")
        root.addWidget(residual_title)
        self.residual_table = None
        if residuals is None:
            unavailable = QLabel("Früher/später QRS-Vergleich nicht verfügbar: gespeicherte QRS-Länge oder gemeinsame Samples fehlen.")
            unavailable.setWordWrap(True)
            root.addWidget(unavailable)
        else:
            table = QTableWidget(2, 12, self)
            table.setHorizontalHeaderLabels(LEAD_ORDER)
            table.setVerticalHeaderLabels(("früh", "spät"))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            table.setFixedHeight(92)
            table.setStyleSheet("QTableWidget { background: #101b24; color: #e8f1f5; gridline-color: #344b58; }")
            for row in range(2):
                for col in range(12):
                    a, b = residuals[row, col]
                    if np.isfinite(a) and np.isfinite(b):
                        winner = "A" if a < b else "B" if b < a else "="
                        item = QTableWidgetItem(f"{winner} · {abs(a-b):.2f}")
                        item.setToolTip(f"Gleiche Samples; RMSE A {a:.3f} mV, B {b:.3f} mV; Differenz {a-b:+.3f} mV")
                        item.setForeground(pg.mkColor("#4fd1c5" if winner == "A" else "#ffbc55" if winner == "B" else "#e8f1f5"))
                    else:
                        item = QTableWidgetItem("–")
                        item.setToolTip("Zu wenige gemeinsam simulierte und gemessene Samples")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table.setItem(row, col, item)
            root.addWidget(table)
            self.residual_table = table
        residual_note = QLabel("A/B = geringerer QRS-RMSE; Zahl = absoluter Unterschied in mV. Ein Modell gewinnt nicht automatisch anatomisch.")
        residual_note.setStyleSheet("font-size: 10px; color: #aebfc8;")
        root.addWidget(residual_note)
        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setBackground("#10161e")
        self.plots.setMinimumHeight(370)
        root.addWidget(self.plots, 1)
        limit = common_display_limit(measured, simulated_a, simulated_b)
        for row, leads in enumerate(LEAD_GRID):
            for col, name in enumerate(leads):
                index = LEAD_ORDER.index(name)
                ra = result_a.r_leads.get(name)
                rb = result_b.r_leads.get(name)
                quality = f" · r A/B {ra:+.2f}/{rb:+.2f}" if ra is not None and rb is not None else ""
                plot = self.plots.addPlot(row=row, col=col, title=name + quality)
                plot.showGrid(x=True, y=True, alpha=0.1)
                plot.setMenuEnabled(False)
                plot.setMouseEnabled(x=False, y=False)
                if col:
                    plot.hideAxis("left")
                else:
                    plot.setLabel("left", "mV")
                if row < 2:
                    plot.hideAxis("bottom")
                else:
                    plot.setLabel("bottom", "ms")
                plot.plot(time_ms, measured[:, index], pen=pg.mkPen("#eef6f8", width=1.7), connect="finite")
                plot.plot(time_ms, simulated_a[:, index],
                          pen=pg.mkPen("#4fd1c5", width=1.5, style=Qt.PenStyle.DashLine), connect="finite")
                plot.plot(time_ms, simulated_b[:, index],
                          pen=pg.mkPen("#ffbc55", width=1.5, style=Qt.PenStyle.DotLine), connect="finite")
                plot.setXRange(float(time_ms[0]), float(time_ms[-1]), padding=0)
                plot.setYRange(-limit, limit, padding=0)
        note = QLabel("Relative Passung innerhalb eines begrenzten Modellraums; A und B sind keine kalibrierten "
                      "Wahrscheinlichkeiten oder anatomischen Nachweise. VCG stammt aus demselben EKG und ist "
                      "keine unabhängige Bestätigung. Gestrichelte Lücken sind nicht simulierte Signalabschnitte.")
        note.setWordWrap(True)
        root.addWidget(note)
        close_button = QPushButton("Schließen")
        close_button.clicked.connect(self.close)
        root.addWidget(close_button)
