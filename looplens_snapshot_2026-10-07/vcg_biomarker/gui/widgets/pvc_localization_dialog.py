"""Research worksheet for source-linked outflow-tract PVC ECG indices."""

from __future__ import annotations

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QLabel, QPushButton, QTextBrowser, QVBoxLayout,
)

from ..pvc_localization import BETENSKY_URL, YOSHIDA_URL, OutflowPVCInputs, localize_outflow_pvc


class PVCLocalizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PVC-Ursprung · Ausflusstrakt-Regeln")
        self.resize(680, 690)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Forschungsworksheet für einen bestätigten, idiopathischen Ausflusstrakt-PVC "
            "mit 12 simultanen Ableitungen. R/S-Amplituden getrennt am selben PVC und an "
            "einem Sinusschlag messen; keine automatische Schlagklassifikation. "
            "Narbe, Schenkelblock, Mischmorphologie oder fehlende V3-R-Zacke → keine Anwendung."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        checks = (
            ("pvc", "Ventrikulären ektopen Schlag bestätigt"),
            ("inferior", "PVC-Achse inferior (II/III/aVF positiv)"),
            ("lbbb", "PVC mit LSB-artiger Morphologie"),
            ("normal_structure", "Strukturelle Herzerkrankung ausgeschlossen"),
            ("normal_sinus", "Sinus-QRS ohne Schenkelblock"),
            ("same_morphology", "Einzelne homogene PVC-Morphologie"),
        )
        for name, label in checks:
            checkbox = QCheckBox(label)
            setattr(self, name, checkbox)
            form.addRow(checkbox)
        self.pvc_transition = QComboBox()
        for label, value in (("nicht beurteilt", "unknown"), ("V3", "v3"), ("andere", "other")):
            self.pvc_transition.addItem(label, value)
        form.addRow("PVC-R/S-Transition", self.pvc_transition)
        self.sinus_transition = QComboBox()
        for label, value in (("nicht beurteilt", "unknown"),
                             ("Sinus früher als PVC", "earlier"),
                             ("gleich", "same"), ("Sinus später als PVC", "later")):
            self.sinus_transition.addItem(label, value)
        form.addRow("Sinus-Transition relativ zum PVC", self.sinus_transition)
        for name, label in (("pvc_v2_r", "PVC V2 R"), ("pvc_v2_s", "PVC V2 S"),
                            ("pvc_v3_r", "PVC V3 R"), ("sinus_v2_r", "Sinus V2 R"),
                            ("sinus_v2_s", "Sinus V2 S")):
            field = QDoubleSpinBox()
            field.setRange(-1, 20)
            field.setDecimals(3)
            field.setSingleStep(.1)
            field.setSuffix(" mV")
            field.setSpecialValueText("nicht gemessen")
            field.setValue(-1)
            setattr(self, name, field)
            form.addRow(label, field)
        layout.addLayout(form)
        button = QPushButton("Publizierte Indizes berechnen")
        button.clicked.connect(self.evaluate)
        layout.addWidget(button)
        self.output = QTextBrowser()
        self.output.setPlainText("Keine Region berechnet. Voraussetzungen und Messungen zunächst prüfen.")
        layout.addWidget(self.output, 1)
        for label, url in (("Betensky: V2-Transitionsverhältnis", BETENSKY_URL),
                           ("Yoshida: V2S/V3R", YOSHIDA_URL)):
            source = QPushButton(label + " · Originalarbeit")
            source.clicked.connect(lambda _checked=False, target=url: QDesktopServices.openUrl(QUrl(target)))
            layout.addWidget(source)

    def evaluate(self) -> None:
        def measured(name: str) -> float | None:
            field = getattr(self, name)
            return field.value() if field.value() >= 0 else None

        result = localize_outflow_pvc(OutflowPVCInputs(
            self.pvc.isChecked(), self.inferior.isChecked(), self.lbbb.isChecked(),
            self.normal_structure.isChecked(), self.normal_sinus.isChecked(),
            self.same_morphology.isChecked(), self.pvc_transition.currentData(),
            self.sinus_transition.currentData(), measured("pvc_v2_r"),
            measured("pvc_v2_s"), measured("pvc_v3_r"), measured("sinus_v2_r"),
            measured("sinus_v2_s"),
        ))
        title = (f"Orientierende Region: {result.region}" if result.status == "candidate"
                 else "Keine eindeutige Region")
        self.output.setPlainText(title + "\n" + result.reason + "\n" + "\n".join(result.criteria)
                                 + "\nKeine fallbezogene Wahrscheinlichkeit oder Ablationszielvorgabe.")
