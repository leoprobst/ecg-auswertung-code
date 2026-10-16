"""Explicit research assays; no silent mapping from patient serum to cell state."""
from dataclasses import asdict
import json

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QLabel, QTextBrowser, QVBoxLayout)


class MetabolismDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Energetik und Stoffwechsel · Forschungsversuch')
        self.resize(640, 520)
        form = QFormLayout(self)
        note = QLabel('Getrennte publizierte Referenzmodelle, kein vollständiger menschlicher Patientenzwilling. '
                      'EDGAR-EKGs bestimmen keine intrazellulären ATP- oder pH-Werte. '
                      'Alle Eingaben hier sind Versuchsbedingungen, keine Patientendiagnose.')
        note.setWordWrap(True)
        form.addRow(note)
        self.model = QComboBox()
        self.model.addItem('Cortassa 2006 · Erregung/Kontraktion/Energetik (Meerschweinchen)', 'ecme')
        self.model.addItem('Beard 2005 · isolierte Mitochondrien, O₂ / Matrix-pH / Mg', 'mitochondria')
        form.addRow('Referenzmodell', self.model)
        self.controls = {}
        for key, label, lo, hi, default, decimals in (
            ('duration', 'Beobachtungsdauer (s)', .1, 60., 8., 2),
            ('cycle', 'Pacing-Zyklus (s; nur ECME)', .25, 10., 4., 2),
            ('respiration', 'Atmungsketten-Kapazitätsfaktor (ECME)', .1, 3., 1., 2),
            ('leak', 'Protonenleck-Faktor', .1, 3., 1., 2),
            ('oxygen', 'Bad-O₂ (mmol/l; nur Mitochondrien)', .00001, .2, .026, 5),
            ('ph', 'Bad-pH (nicht intrazellulärer pH)', 6.5, 7.8, 7.1, 2),
            ('adp', 'Bad-ADP (mmol/l; Mitochondrien)', 0., 5., .1, 3),
        ):
            control = QDoubleSpinBox()
            control.setDecimals(decimals)
            control.setRange(lo, hi)
            control.setValue(default)
            self.controls[key] = control
            form.addRow(label, control)
        self.parameters = QComboBox()
        self.parameters.addItem('Unveränderte CellML-Protokollparameter', 'cellml_source')
        self.parameters.addItem('Parameter gemäß korrigierter Publikation (2006)', 'paper_2006_corrected')
        form.addRow('Mitochondrien-Parametersatz', self.parameters)
        self.acknowledge = QCheckBox('ECME: unvollständige Einheitenannotation der Quelle ist bekannt; nur Forschung')
        form.addRow(self.acknowledge)
        self.error = QLabel()
        self.error.setWordWrap(True)
        form.addRow(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_valid)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.model.currentIndexChanged.connect(self._update_fields)
        self._update_fields()

    def _update_fields(self):
        ecme = self.model.currentData() == 'ecme'
        for key in ('cycle', 'respiration'):
            self.controls[key].setEnabled(ecme)
        for key in ('oxygen', 'ph', 'adp'):
            self.controls[key].setEnabled(not ecme)
        self.parameters.setEnabled(not ecme)
        self.acknowledge.setVisible(ecme)

    def request(self):
        values = {k: c.value() for k, c in self.controls.items()}
        if self.model.currentData() == 'ecme':
            from ..simulation.ecme import ECMEExperiment
            spec = ECMEExperiment(duration_ms=values['duration']*1000, cycle_ms=values['cycle']*1000,
                output_ms=2., respiratory_capacity_scale=values['respiration'], proton_leak_scale=values['leak'],
                acknowledge_source_unit_limitations=self.acknowledge.isChecked())
            spec.validate()
            return {'kind': 'published_ecme', 'ecme_experiment': asdict(spec)}
        from ..simulation.mitochondria import MitochondrialExperiment
        spec = MitochondrialExperiment(duration_s=values['duration'], oxygen_mmol_l=values['oxygen'],
            bath_pH=values['ph'], bath_adp_mmol_l=values['adp'], proton_leak_scale=values['leak'],
            parameter_set=self.parameters.currentData())
        spec.validate()
        return {'kind': 'mitochondrial_assay', 'mitochondrial_experiment': asdict(spec)}

    def _accept_valid(self):
        try:
            self.request()
            self.accept()
        except ValueError as exc:
            self.error.setText(str(exc))


class MetabolismResult(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Berechnete Stoffwechselzustände · keine Patientendiagnose')
        self.resize(900, 680)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            schema = metadata.get('schema')
            if schema not in {'published-ecme-assay/1', 'published-mitochondrial-assay/1'}:
                raise ValueError('Unbekannter Stoffwechsel-Ergebnisvertrag')
            ecme = schema == 'published-ecme-assay/1'
            time_key = 'times_ms' if ecme else 'times_s'
            # Read the initial audit-export alias using the explicit model schema.
            self.times = data[time_key if time_key in data else 'time'].copy()
            self.states = data['states'].copy()
        names = metadata['state_names']
        if (self.states.shape != (len(self.times), len(names)) or not np.isfinite(self.states).all()
                or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)):
            raise ValueError('Ungültige Stoffwechselkurven')
        layout = QVBoxLayout(self)
        note = QLabel('Publizierter Referenzversuch · nicht am Menschen validiert. '
                      'Bad-O₂ bleibt im Mitochondrienmodell konstant. '
                      'ECME-Quellmodell besitzt unvollständige Einheiten; keine stillschweigende Umrechnung.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.quantity = QComboBox()
        self.quantity.addItems(names)
        layout.addWidget(self.quantity)
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Zeit', units='ms' if ecme else 's')
        layout.addWidget(self.plot, 3)
        browser = QTextBrowser()
        browser.setPlainText(json.dumps(metadata, ensure_ascii=False, indent=2))
        layout.addWidget(browser, 1)
        self.units = metadata.get('state_units', ['Quellmodell-Einheit (unvollständig)']*len(names))
        self.quantity.currentIndexChanged.connect(self._draw)
        self._draw(0)

    def _draw(self, index):
        self.plot.clear()
        self.plot.setLabel('left', self.quantity.itemText(index)+' · '+self.units[index])
        self.plot.plot(self.times, self.states[:, index], pen=pg.mkPen('#33cbbb', width=2))
