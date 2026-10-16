"""Review-gated V1 P-wave clue for focal atrial tachycardia."""

from __future__ import annotations

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDialog, QFormLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout

from ..atrial_focus_clue import SOURCES, focal_at_v1_clue


class AtrialFocusDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fokale atriale Tachykardie · V1-P-Welle")
        self.resize(620, 440)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Nur für eine bestätigte fokale atriale Tachykardie mit frei sichtbarer ektoper P-Welle. "
            "Nicht auf Vorhofflimmern, Flattern, AVNRT oder eine gewöhnliche Sinus-P-Welle anwenden. "
            "Die P-Welle muss zum untersuchten Rhythmus gehören."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.focal = QCheckBox("Fokale atriale Tachykardie bestätigt")
        self.unobscured = QCheckBox("Ektopische P-Welle nicht in T/QRS verborgen")
        self.quality = QCheckBox("V1-P-Signalqualität und Elektrodenlage geprüft")
        for checkbox in (self.focal, self.unobscured, self.quality):
            form.addRow(checkbox)
        self.pattern = QComboBox()
        for label, value in (("unbeurteilt", "unclear"), ("negativ", "negative"),
                             ("positiv → negativ", "positive_negative"), ("positiv", "positive"),
                             ("negativ → positiv", "negative_positive"),
                             ("isoelektrisch → positiv", "isoelectric_positive"),
                             ("isoelektrisch", "isoelectric")):
            self.pattern.addItem(label, value)
        form.addRow("Ektopische P-Welle in V1", self.pattern)
        layout.addLayout(form)
        button = QPushButton("Breiten regionalen Hinweis prüfen")
        button.clicked.connect(self.evaluate)
        layout.addWidget(button)
        self.output = QTextBrowser()
        self.output.setPlainText("Noch kein Hinweis berechnet.")
        layout.addWidget(self.output, 1)
        for label, url in (("Kistler 2006", SOURCES[0]), ("Kistler 2021", SOURCES[1])):
            source = QPushButton(label + " · Originalarbeit")
            source.clicked.connect(lambda _checked=False, target=url: QDesktopServices.openUrl(QUrl(target)))
            layout.addWidget(source)

    def evaluate(self) -> None:
        result = focal_at_v1_clue(
            focal_at_confirmed=self.focal.isChecked(),
            p_wave_unobscured=self.unobscured.isChecked(),
            p_quality_accepted=self.quality.isChecked(),
            v1_pattern=self.pattern.currentData(),
        )
        title = result.region_hint if result.status == "clue" else "Kein regionaler Hinweis"
        self.output.setPlainText(
            title + "\n" + result.explanation
            + "\nOrientierung für Mapping, keine Fokusdiagnose und keine fallbezogene Wahrscheinlichkeit."
        )
