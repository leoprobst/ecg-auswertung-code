"""Transparent, clinician-gated accessory-pathway localization worksheet."""

from __future__ import annotations

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFormLayout, QLabel, QPushButton,
    QTextBrowser, QVBoxLayout,
)

from ..wpw_localization import (
    DELTA_LEADS, EASYWPWInputs, SOURCE_URL,
    easy_wpw_localize, propose_easy_wpw_inputs,
)


class WPWLocalizationDialog(QDialog):
    def __init__(self, lab, parent=None):
        super().__init__(parent or lab)
        self.lab = lab
        self.setWindowTitle("Akzessorische Bahn · EASY-WPW-Forschungsworksheet")
        self.resize(650, 590)
        root = QVBoxLayout(self)
        note = QLabel(
            "Nur für manifeste Präexzitation mit wahrscheinlich einem Leitungsweg. "
            "Aberrante QRS-Überleitung/Schenkelblock ist nicht dasselbe. "
            "Die Region ist eine präinterventionelle Hypothese, keine Mappingdiagnose."
        )
        note.setWordWrap(True)
        root.addWidget(note)
        form = QFormLayout()
        self.manifest = QCheckBox("Delta-Welle/manifeste Präexzitation von mir bestätigt")
        self.single = QCheckBox("Ein einzelner Leitungsweg plausibel")
        self.no_structural = QCheckBox("Strukturelle Herzerkrankung ausgeschlossen")
        self.no_ablation = QCheckBox("Keine vorausgegangene Ablation/Narbe")
        for checkbox in (self.manifest, self.single, self.no_structural, self.no_ablation):
            form.addRow(checkbox)
        self.v1 = QComboBox()
        for text, value in (("nicht beurteilt", "unknown"), ("positiv", "positive"),
                            ("negativ", "negative"), ("isoelektrisch", "isoelectric")):
            self.v1.addItem(text, value)
        self.transition = QComboBox()
        for text, value in (("nicht beurteilt", "unknown"),
                            ("bis einschließlich V3", "at_or_before_v3"),
                            ("nach V3", "after_v3")):
            self.transition.addItem(text, value)
        self.delta = QComboBox()
        self.delta.addItem("nicht beurteilt", "")
        for lead in DELTA_LEADS:
            self.delta.addItem(lead, lead)
        form.addRow("V1-QRS-Polarität", self.v1)
        form.addRow("Präkordiale QRS-Transition", self.transition)
        form.addRow("Meistpositive Delta-Welle", self.delta)
        root.addLayout(form)
        preview = QPushButton("Messvorschläge aus geprüften QRS-Grenzen")
        preview.setToolTip(
            "Vorbelegung aus frühen 20–40 ms; Messimplementation ist nicht als Ersatz der manuellen EASY-WPW-Lesung validiert."
        )
        preview.clicked.connect(self.preview)
        root.addWidget(preview)
        evaluate = QPushButton("Region anhand bestätigter Eingaben prüfen")
        evaluate.clicked.connect(self.evaluate)
        root.addWidget(evaluate)
        self.output = QTextBrowser()
        self.output.setPlainText("Keine Lokalisation berechnet. Alle klinischen Voraussetzungen bleiben zunächst ungeprüft.")
        root.addWidget(self.output, 1)
        source = QPushButton("Originalarbeit und Flussdiagramm öffnen")
        source.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(SOURCE_URL)))
        root.addWidget(source)

    def preview(self) -> None:
        lab = self.lab
        if lab.signal is None or lab.reference is None:
            self.output.setPlainText("Kein Signal geladen.")
            return
        details = lab.payload.get("delineation", {}).get("fiducial_details", {})
        verified = lab.manual_fiducials or (
            not lab.payload.get("delineation", {}).get("review_required", True)
            and all(isinstance(details.get(key), dict)
                    and details[key].get("wave_status") == "present"
                    and float(details[key].get("confidence") or 0) >= .8
                    for key in ("qrs_onset", "qrs_offset"))
        )
        if not verified:
            self.output.setPlainText(
                "QRS-Beginn/-Ende sind nicht ausreichend geprüft. Zuerst im Wellen-Close-up bestätigen; "
                "keine automatische Delta-Polarität aus unsicheren Grenzen."
            )
            return
        try:
            values = propose_easy_wpw_inputs(
                lab.signal, lab.reference.sampling_rate_hz,
                lab.fiducials["qrs_onset"], lab.fiducials["qrs_offset"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.output.setPlainText(f"Messvorschlag nicht verfügbar: {exc}")
            return
        self.v1.setCurrentIndex(self.v1.findData(values["v1_polarity_proposal"]))
        self.transition.setCurrentIndex(self.transition.findData(values["transition_proposal"]))
        if values["most_positive_delta_lead_proposal"]:
            self.delta.setCurrentIndex(self.delta.findData(values["most_positive_delta_lead_proposal"]))
        self.output.setPlainText(
            "Automatische MESSVORSCHLÄGE, keine bestätigte Lokalisation. "
            "Delta-Zeichen, QRS-Transition und Präexzitation am Original-EKG visuell prüfen.\n"
            + "Frühe signierte Ausschläge [mV]: "
            + ", ".join(f"{lead} {value:+.3f}" for lead, value in values["delta_signed_mv"].items())
            + f"\nMaximales Baseline-Rauschen: {values['baseline_noise_mv_max']:.3f} mV"
        )

    def evaluate(self) -> None:
        inputs = EASYWPWInputs(
            manifest_preexcitation_confirmed=self.manifest.isChecked(),
            single_pathway_assumed=self.single.isChecked(),
            structural_disease_excluded=self.no_structural.isChecked(),
            prior_ablation_excluded=self.no_ablation.isChecked(),
            v1_polarity=self.v1.currentData(),
            transition=self.transition.currentData(),
            most_positive_delta_lead=self.delta.currentData(),
        )
        result = easy_wpw_localize(inputs)
        if result.status == "abstained":
            self.output.setPlainText("Keine Lokalisation: " + result.reason)
        else:
            self.output.setPlainText(
                f"Forschungskandidat: {result.region}\n"
                + "\n".join(result.decision_path)
                + "\n" + result.reason
                + "\nKeine fallbezogene Wahrscheinlichkeit. Quelle: " + result.source
            )
