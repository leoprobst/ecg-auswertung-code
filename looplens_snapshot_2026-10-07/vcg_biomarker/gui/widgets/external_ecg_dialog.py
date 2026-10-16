from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QSpinBox, QDialogButtonBox, QGroupBox, QFormLayout,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from ...core.data_loader import LEAD_NAMES_12
from ...core.external_loader import ExternalECGMeta


class ExternalECGDialog(QDialog):
    """Ask for patient metadata and sampling rate when loading an external ECG."""

    def __init__(self, detected_fs: int = 500, parent=None):
        super().__init__(parent)
        self.setWindowTitle("External ECG – Patient Information")
        self.setMinimumWidth(320)
        self._setup_ui(detected_fs)

    def _setup_ui(self, detected_fs: int):
        layout = QVBoxLayout(self)

        group = QGroupBox("Patient / Recording Info")
        form = QFormLayout(group)

        self.age_edit = QSpinBox()
        self.age_edit.setRange(0, 120)
        self.age_edit.setSpecialValueText("Unknown")
        form.addRow("Age:", self.age_edit)

        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Unknown", "M", "F"])
        form.addRow("Sex:", self.sex_combo)

        self.fs_spin = QSpinBox()
        self.fs_spin.setRange(50, 10000)
        self.fs_spin.setSuffix(" Hz")
        self.fs_spin.setValue(detected_fs)
        form.addRow("Sampling rate:", self.fs_spin)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["mV", "uV", "V"])
        self.unit_combo.setToolTip(
            "Physikalische Einheit der importierten Kanäle. Eine explizite "
            "Einheit verhindert unbemerkte Amplitudenfehler."
        )
        form.addRow("Signal unit:", self.unit_combo)

        layout.addWidget(group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_meta(self) -> ExternalECGMeta:
        age = self.age_edit.value() if self.age_edit.value() > 0 else None
        sex_text = self.sex_combo.currentText()
        sex = sex_text if sex_text != "Unknown" else None
        return ExternalECGMeta(
            age=age,
            sex=sex,
            sampling_rate=self.fs_spin.value(),
            default_unit=self.unit_combo.currentText(),
        )


class LeadMappingDialog(QDialog):
    """Require an explicit one-to-one mapping for ambiguous ECG channels."""

    def __init__(
        self,
        available_channels: tuple[str, ...],
        *,
        base_meta: ExternalECGMeta,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("ECG-Kanäle eindeutig zuordnen")
        self.setMinimumWidth(480)
        self.available_channels = tuple(available_channels)
        self.base_meta = base_meta
        self.mapping_boxes: dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)
        description = QLabel(
            "Ordnen Sie jeden Pflichtkanal genau einem Quellkanal zu. "
            "Die Analyse bleibt gesperrt, bis Leadnamen, Einheit und "
            "Samplingrate eindeutig sind."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        group = QGroupBox("12-Kanal-Zuordnung")
        form = QFormLayout(group)
        for lead in LEAD_NAMES_12:
            combo = QComboBox()
            combo.addItem("Nicht zugeordnet", "")
            for channel in self.available_channels:
                combo.addItem(channel, channel)
            exact = combo.findText(lead, Qt.MatchFlag.MatchFixedString)
            if exact >= 0:
                combo.setCurrentIndex(exact)
            combo.setToolTip(
                f"Quellkanal, der den kanonischen Kanal {lead} repräsentiert"
            )
            self.mapping_boxes[lead] = combo
            form.addRow(lead, combo)
        layout.addWidget(group)

        recording = QGroupBox("Aufzeichnung")
        recording_form = QFormLayout(recording)
        self.fs_spin = QSpinBox()
        self.fs_spin.setRange(50, 10_000)
        self.fs_spin.setSuffix(" Hz")
        self.fs_spin.setValue(int(base_meta.sampling_rate or 500))
        recording_form.addRow("Samplingrate", self.fs_spin)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["mV", "uV", "V"])
        selected_unit = self.unit_combo.findText(base_meta.default_unit or "mV")
        if selected_unit >= 0:
            self.unit_combo.setCurrentIndex(selected_unit)
        self.unit_combo.setToolTip(
            "Physikalische Einheit aller gewählten Quellkanäle"
        )
        recording_form.addRow("Einheit", self.unit_combo)
        layout.addWidget(recording)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        selected = [
            str(box.currentData() or "")
            for box in self.mapping_boxes.values()
        ]
        if any(not value for value in selected):
            QMessageBox.warning(
                self,
                "Zuordnung unvollständig",
                "Alle zwölf Pflichtkanäle müssen zugeordnet sein.",
            )
            return
        if len(set(selected)) != len(selected):
            QMessageBox.warning(
                self,
                "Zuordnung mehrdeutig",
                "Jeder Quellkanal darf nur einmal verwendet werden.",
            )
            return
        super().accept()

    def get_meta(self) -> ExternalECGMeta:
        return ExternalECGMeta(
            age=self.base_meta.age,
            sex=self.base_meta.sex,
            sampling_rate=float(self.fs_spin.value()),
            lead_mapping={
                lead: str(box.currentData())
                for lead, box in self.mapping_boxes.items()
            },
            default_unit=self.unit_combo.currentText(),
        )
