"""Readable, case-local evidence chain for stored research marker measurements.

This view describes recorded provenance and abstains where an artifact is
missing. It never promotes a marker value to a diagnosis or a probability.
"""
from __future__ import annotations

import math
import json

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..lab_measurement_windows import validated_direct8_windows


def recorded_abstention(recorded: dict) -> str | None:
    """Explain only a reason actually stored with this marker measurement."""
    notes = recorded.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        return None
    raw = notes.strip()
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            reason = parsed.get("reason")
            if isinstance(reason, str) and reason.strip():
                explanations = {
                    "insufficient_quality_eligible_beats": "Zu wenige qualitätsgeeignete Schläge",
                    "missing_fiducials": "Erforderliche Wellengrenzen fehlen",
                    "missing_leads": "Erforderliche Ableitungen fehlen",
                }
                return explanations.get(reason, f"Gespeicherter Grund: {reason[:100]}")
    if raw.startswith("not_evaluable: "):
        if raw == "not_evaluable: p fiducials below the pre-specified confidence threshold (0.35)":
            return "P-Wellengrenzen unterschreiten die im Artefakt festgelegte Qualitätsgrenze (0,35)"
        return f"Gespeicherter Grund: {raw[len('not_evaluable: '):][:140]}"
    return f"Gespeicherte Notiz: {raw[:140]}"


def marker_evidence_steps(detail: dict, windows, arrays, *, signal_bound: bool) -> tuple[tuple[str, str, str], ...]:
    marker = str(detail.get("marker") or "Unbekannter Marker")
    analysis_id = detail.get("analysis_id")
    recorded = detail.get("recorded_definition")
    recorded = recorded if isinstance(recorded, dict) else {}
    value = detail.get("value")
    valid_value = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    unit = recorded.get("unit") if isinstance(recorded.get("unit"), str) else None
    formula = recorded.get("formula") if isinstance(recorded.get("formula"), str) else None
    short_id = f"{str(analysis_id)[:12]}…" if analysis_id and len(str(analysis_id)) > 12 else analysis_id
    signal_text = (f"Analyse {short_id} · signalgebunden" if signal_bound and analysis_id
                   else "Historische Signalbindung hier nicht vollständig belegt")
    value_text = (f"{value:.6g} {unit or '(Einheit im Artefakt fehlt)'}" if valid_value
                  else "Kein auswertbarer gespeicherter Wert")
    formula_text = ((formula[:110] + "… · vollständig im JSON darunter" if len(formula) > 110 else formula)
                    if formula else
                    "Formel im geladenen Fallartefakt nicht enthalten; das aktuelle Register ersetzt sie nicht.")

    window_status = "Keine exakten Messfenster im Fallartefakt"
    window_detail = "Ohne gespeicherte Koordinaten keine grafisch behaupteten P-/QRS-/T-Grenzen."
    abstention = recorded_abstention(recorded) if not valid_value else None
    if abstention:
        window_status = "Nicht auswertbar · dokumentierter Grund"
        window_detail = abstention
    if isinstance(windows, dict) and not abstention:
        signal_path = windows.get("signal_path")
        signal = arrays.get(signal_path) if hasattr(arrays, "get") else None
        if signal is None:
            window_status = "Messvertrag gespeichert · Signalpfad nicht verfügbar"
            window_detail = f"Pfad {signal_path or 'unbekannt'} kann lokal nicht mit Samples verknüpft werden."
        elif getattr(signal, "ndim", None) != 2 or signal.shape[1] != 12:
            window_status = "Messvertrag gespeichert · Signalformat inkompatibel"
            window_detail = "Keine Rekonstruktion der Grenzen aus einem anderen Signalformat."
        elif validated_direct8_windows(windows, len(signal)) is None:
            window_status = "Messvertrag gespeichert · Koordinaten inkompatibel"
            window_detail = "Schema, Ableitungen und Samplegrenzen müssen zum gespeicherten Signal passen."
        else:
            window_status = "Messfenster mit lokalem Signal prüfbar"
            window_detail = "Acht Ableitungen sowie Baseline, QRS und J–T-Ende aus gespeicherten Koordinaten; siehe Kurven darunter."

    return (
        ("01 · SIGNAL", signal_text, f"Fallmarker: {marker}. Herkunft ist keine klinische Referenz."),
        ("02 · FORMEL & WERT", value_text, formula_text),
        ("03 · MESSFENSTER", window_status, window_detail),
        ("04 · AUSSAGE", "Forschungswert · keine Diagnose", "Dieser Inspektor prüft keine unabhängige fallbezogene Echo-/Labor- oder Mappingreferenz."),
    )


class MarkerEvidenceTrail(QWidget):
    """Four small, non-interactive cards above the detailed JSON and signal view."""

    def __init__(self, parent: QWidget | None = None, *, dark: bool = True):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        title = QLabel("NACHWEISKETTE · gespeicherter Messstand, keine automatische Befundfreigabe")
        title.setStyleSheet("font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        root.addWidget(title)
        row = QHBoxLayout()
        self.cards = []
        self.summaries = []
        self.details = []
        for _ in range(4):
            card = QFrame(self)
            card.setObjectName("markerEvidenceStep")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(8, 6, 8, 6)
            heading = QLabel("—")
            heading.setStyleSheet("font-size: 10px; font-weight: 700;")
            summary = QLabel("Kein Marker gewählt")
            summary.setWordWrap(True)
            summary.setStyleSheet("font-size: 12px; font-weight: 600;")
            detail = QLabel("")
            detail.setWordWrap(True)
            detail.setStyleSheet("font-size: 10px;")
            layout.addWidget(heading)
            layout.addWidget(summary)
            layout.addWidget(detail)
            layout.addStretch(1)
            row.addWidget(card, 1)
            self.cards.append(card)
            self.summaries.append(summary)
            self.details.append(detail)
            card.heading = heading
        root.addLayout(row)
        self.set_dark(dark)

    def set_evidence(self, detail: dict | None, windows=None, arrays=None, *, signal_bound: bool = False) -> None:
        steps = (marker_evidence_steps(detail, windows, arrays, signal_bound=signal_bound)
                 if isinstance(detail, dict) else ())
        for index, card in enumerate(self.cards):
            heading, summary, explanation = (steps[index] if steps else ("—", "Kein Marker gewählt", ""))
            card.heading.setText(heading)
            self.summaries[index].setText(summary)
            self.details[index].setText(explanation)
            self.details[index].setToolTip(explanation)

    def set_dark(self, dark: bool) -> None:
        background, border, foreground = (("#142632", "#476473", "#e4f0f4") if dark else
                                          ("#f2f8fa", "#9dbac4", "#17313e"))
        self.setStyleSheet(f"QFrame#markerEvidenceStep {{ background: {background}; "
                           f"border: 1px solid {border}; border-radius: 5px; }} "
                           f"QFrame#markerEvidenceStep QLabel {{ color: {foreground}; border: none; }}")
