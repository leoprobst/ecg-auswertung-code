"""Case-local, read-only comparison of recorded ECG criteria.

An ECG laboratory finding is not an echo/labor reference for this patient.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class EndpointComparisonCard(QFrame):
    markerRequested = pyqtSignal(str)
    ruleRequested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *, dark: bool = True):
        super().__init__(parent)
        self.setObjectName("endpointComparisonCard")
        self._dark = dark
        self._review = None
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        self.title = QLabel("Klinische Frage auswählen")
        self.title.setStyleSheet("font-size: 16px; font-weight: 700;")
        root.addWidget(self.title)
        grid = QGridLayout()
        self.classic_heading = QLabel("KLASSISCHES EKG")
        self.research_heading = QLabel("FORSCHUNGSMARKER")
        self.reference_heading = QLabel("UNABHÄNGIGER NACHWEIS")
        for col, label in enumerate((self.classic_heading, self.research_heading, self.reference_heading)):
            label.setStyleSheet("font-size: 10px; font-weight: 700; letter-spacing: 1px;")
            grid.addWidget(label, 0, col)
        self.classic = QLabel("Keine Frage gewählt")
        self.research = QLabel("Keine Frage gewählt")
        self.reference = QLabel("Kein fallbezogener Echo-/Laborbefund in diesem Analyseartefakt")
        for col, label in enumerate((self.classic, self.research, self.reference)):
            label.setWordWrap(True)
            label.setMinimumHeight(66)
            label.setObjectName("endpointComparisonCell")
            grid.addWidget(label, 1, col)
            grid.setColumnStretch(col, 1)
        root.addLayout(grid)
        self.comparison = QLabel("Die drei Quellen werden nicht zu einer Diagnosewahrscheinlichkeit verrechnet.")
        self.comparison.setWordWrap(True)
        root.addWidget(self.comparison)
        actions = QHBoxLayout()
        self.rule_button = QPushButton("Regel am Signal prüfen")
        self.rule_button.clicked.connect(lambda: self.ruleRequested.emit(self._rule_id))
        self.marker_button = QPushButton("Marker und Messfenster prüfen")
        self.marker_button.clicked.connect(lambda: self.markerRequested.emit(self._marker_id))
        self._rule_id = ""
        self._marker_id = ""
        actions.addWidget(self.rule_button)
        actions.addWidget(self.marker_button)
        actions.addStretch(1)
        root.addLayout(actions)
        self.set_review(None, [])
        self.set_dark(dark)

    def set_review(self, review, findings: list[dict], *, available_rules=None, available_markers=None) -> None:
        self._review = review
        self._rule_id = ""
        self._marker_id = ""
        if not isinstance(review, dict):
            self.title.setText("Klinische Frage auswählen")
            self.classic.setText("Kein gespeichertes Kriterium gewählt")
            self.research.setText("Kein gespeichertes Kriterium gewählt")
            self.comparison.setText("Keine Aussage über einen klinischen Endpunkt.")
        else:
            self.title.setText(str(review.get("name") or review.get("endpoint_id") or "Unbenannter Endpunkt"))
            self.classic.setText(str(review.get("classical_text") or "Nicht beurteilbar"))
            self.research.setText(str(review.get("research_text") or "Nicht beurteilbar"))
            self.comparison.setText(str(review.get("comparison") or "Keine eindeutige Gegenüberstellung")
                                    + " · Kriterien sind keine Diagnose; fehlende Referenz ist nicht negativ.")
            for finding in findings:
                if not isinstance(finding, dict) or finding.get("endpoint_id") != review.get("endpoint_id"):
                    continue
                marker = finding.get("feature_id")
                if not self._marker_id and isinstance(marker, str) and marker:
                    self._marker_id = marker
                for comparator in finding.get("classical_comparators", []):
                    if isinstance(comparator, dict) and isinstance(comparator.get("criterion_id"), str):
                        self._rule_id = comparator["criterion_id"]
                        break
                if self._rule_id and self._marker_id:
                    break
        self.reference.setText("Für diesen Fall im EKG-Laborartefakt nicht vorhanden; "
                               "Kohortenreferenzen sind kein Einzelfallnachweis.")
        self.rule_button.setEnabled(bool(self._rule_id) and
                                    (available_rules is None or self._rule_id in available_rules))
        self.marker_button.setEnabled(bool(self._marker_id) and
                                      (available_markers is None or self._marker_id in available_markers))

    def set_dark(self, dark: bool) -> None:
        self._dark = bool(dark)
        if dark:
            self.setStyleSheet("""
                QFrame#endpointComparisonCard { background: #111e29; border: 1px solid #3d5968; border-radius: 8px; }
                QLabel#endpointComparisonCell { background: #1a2c37; border: 1px solid #34505d;
                                                border-radius: 6px; padding: 9px; color: #e9f2f5; }
                QLabel { color: #d9e8ef; }
                QPushButton { background: #234451; color: #eef8fa; border: 1px solid #5c8290;
                              border-radius: 5px; padding: 6px 10px; }
            """)
        else:
            self.setStyleSheet("""
                QFrame#endpointComparisonCard { background: #edf5f7; border: 1px solid #9cbac4; border-radius: 8px; }
                QLabel#endpointComparisonCell { background: #ffffff; border: 1px solid #b6cbd2;
                                                border-radius: 6px; padding: 9px; color: #173240; }
                QLabel { color: #173240; }
                QPushButton { background: #ffffff; color: #143c48; border: 1px solid #7aa2ae;
                              border-radius: 5px; padding: 6px 10px; }
            """)
