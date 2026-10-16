"""Manual rhythm hypothesis diagrams; never an inferred conduction diagnosis."""
import json
import uuid
from datetime import datetime, timezone

import pyqtgraph as pg
import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QPushButton, QLineEdit, QMessageBox, QCheckBox)

from ..mimic.storage import atomic_write_json
from ..core.data_loader import LEAD_NAMES_12

LEVELS = {'A': 2, 'AV': 1, 'V': 0}
KINDS = ('conduction_hypothesis', 'blocked_hypothesis')


def validate_diagram(document, reference):
    if document.get('schema_version') != '1.0.0' or document.get('signal_reference') != reference:
        raise ValueError('Diagrammversion oder Signalbezug passt nicht')
    if document.get('kind') != 'manual_laddergram' or document.get('clinical_claim_allowed') is not False:
        raise ValueError('Kein gültiges manuelles Forschungsdiagramm')
    events, edges = document.get('events'), document.get('edges')
    if not isinstance(events, list) or not isinstance(edges, list) or len(events) > 1000 or len(edges) > 2000:
        raise ValueError('Ungültiger Diagrammumfang')
    by_id = {}
    for item in events:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or item['id'] in by_id:
            raise ValueError('Ungültige oder doppelte Aktivierung')
        if item.get('level') not in LEVELS or type(item.get('sample')) is not int or not 0 <= item['sample'] < reference['sample_count']:
            raise ValueError('Aktivierung außerhalb der Signalzeit oder unbekannte Ebene')
        by_id[item['id']] = item
    for edge in edges:
        if not isinstance(edge, dict) or edge.get('source') not in by_id or edge.get('target') not in by_id:
            raise ValueError('Verbindung ohne gültige Aktivierungen')
        a, b = by_id[edge['source']], by_id[edge['target']]
        if a['id'] == b['id'] or b['sample'] < a['sample'] or edge.get('kind') not in KINDS:
            raise ValueError('Ungültige Verbindungsrichtung oder -art')
    return document


class LaddergramWidget(QWidget):
    timeSelected = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.reference = None
        self.root = None
        self.signal = None
        self._draft_ready = False
        self._dark = False
        self.events, self.edges, self.history = [], [], []
        layout = QVBoxLayout(self)
        note = QLabel('Manuelles Laddergram · A = Vorhof, AV = AV-Übergang, V = Ventrikel. '
            'Aktivierungen und Verbindungen sind Ihre zeitgebundene Hypothese, kein Mappingnachweis. '
            'Blockierte Verbindung endet vor dem Ziel. Rückleitung kann von V nach A eingezeichnet werden, '
            'aber nicht rückwärts in der Zeit. Klick ins Diagramm setzt die EKG-Zeit.')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.level = QComboBox()
        self.level.addItems(LEVELS)
        self.time = QDoubleSpinBox()
        self.time.setDecimals(6)
        self.time.setSuffix(' s')
        self.author = QLineEdit()
        self.author.setPlaceholderText('Bearbeiter')
        for widget in (self.level, self.time, self.author):
            row.addWidget(widget)
        add = QPushButton('Aktivierung setzen')
        add.clicked.connect(self.add_event)
        row.addWidget(add)
        layout.addLayout(row)
        self.lead = QComboBox()
        self.lead.addItems(LEAD_NAMES_12)
        self.lead.setCurrentText('II')
        self.lead.setToolTip('Natives Originalsignal in mV, keine aVR-Anzeigeinversion. Doppelklick setzt eine manuelle Aktivierung auf der gewählten Ebene.')
        self.lead.currentIndexChanged.connect(self.draw_ecg)
        layout.addWidget(self.lead)
        self.ecg_plot = pg.PlotWidget()
        self.ecg_plot.setMinimumHeight(160)
        self.ecg_plot.setLabel('left', 'Originalsignal', units='mV')
        self.ecg_plot.scene().sigMouseClicked.connect(lambda event: self.clicked(event, self.ecg_plot))
        layout.addWidget(self.ecg_plot)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(260)
        self.ecg_plot.setXLink(self.plot)
        self.ecg_plot.getAxis('left').setWidth(65)
        self.plot.getAxis('left').setWidth(65)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setLabel('bottom', 'Zeit im Originalsignal', units='s')
        self.plot.getAxis('left').setTicks([[(v, k) for k, v in LEVELS.items()]])
        self.plot.scene().sigMouseClicked.connect(self.clicked)
        self.time.valueChanged.connect(self.update_cursor)
        layout.addWidget(self.plot)
        links = QHBoxLayout()
        self.source, self.target, self.kind = QComboBox(), QComboBox(), QComboBox()
        self.kind.addItem('Überleitungshypothese', KINDS[0])
        self.kind.addItem('Blockierungshypothese', KINDS[1])
        for widget in (self.source, self.target, self.kind):
            links.addWidget(widget)
        connect = QPushButton('Verbinden')
        connect.clicked.connect(self.add_edge)
        links.addWidget(connect)
        delete = QPushButton('Quellaktivierung entfernen')
        delete.clicked.connect(self.remove_event)
        links.addWidget(delete)
        layout.addLayout(links)
        actions = QHBoxLayout()
        for label, fn in (('Rückgängig', self.undo), ('Revision speichern', self.save), ('Revision laden', self.load)):
            button = QPushButton(label)
            button.clicked.connect(fn)
            actions.addWidget(button)
        self.revisions = QComboBox()
        actions.addWidget(self.revisions, 1)
        layout.addLayout(actions)
        export_row = QHBoxLayout()
        self.export_format = QComboBox()
        self.export_format.addItems(['PDF', 'SVG', 'PNG'])
        export_row.addWidget(self.export_format)
        self.export_ecg = QCheckBox('Ausgewählte EKG-Ableitung mitexportieren')
        export_row.addWidget(self.export_ecg)
        export_button = QPushButton('Diagramm lokal exportieren')
        export_button.clicked.connect(self.export)
        export_row.addWidget(export_button)
        layout.addLayout(export_row)
        self.status = QLabel('Kein Fall geöffnet')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.set_dark(False)

    def set_dark(self, dark):
        self._dark = bool(dark)
        self.plot.setBackground('#111923' if dark else '#fffdf9')
        self.ecg_plot.setBackground('#111923' if dark else '#fffdf9')
        for name in ('bottom', 'left'):
            axis = self.plot.getAxis(name)
            axis.setPen('#dce5ef' if dark else '#263746')
            axis.setTextPen('#dce5ef' if dark else '#263746')
            ecg_axis = self.ecg_plot.getAxis(name)
            ecg_axis.setPen('#dce5ef' if dark else '#263746')
            ecg_axis.setTextPen('#dce5ef' if dark else '#263746')
        self.draw_ecg()
        self.draw()

    def update_cursor(self, value):
        if hasattr(self, 'cursor'):
            self.cursor.setValue(value)
        if hasattr(self, 'ecg_cursor'):
            self.ecg_cursor.setValue(value)

    def set_context(self, reference, root, signal=None):
        if signal is not None:
            signal = np.asarray(signal)
            if signal.shape != (reference['sample_count'], 12):
                raise ValueError('EKG-Signal und Diagrammreferenz passen nicht')
        self.signal = signal
        self.reference, self.root = dict(reference), root / 'laddergrams'
        self.events, self.edges, self.history = [], [], []
        self.time.setRange(0, (reference['sample_count']-1)/reference['sampling_rate_hz'])
        self.time.setValue(0)
        self.author.clear()
        self.status.setText('Leeres Diagramm · keine automatische Rhythmusinterpretation')
        self._draft_ready = False
        draft = self.root / 'draft_recovery.json'
        try:
            if draft.exists():
                document = validate_diagram(json.loads(draft.read_text(encoding='utf-8')), self.reference)
                self.events, self.edges = document['events'], document['edges']
                self.author.setText(document.get('author', ''))
                self.status.setText('Lokaler unbestätigter Diagrammentwurf wiederhergestellt')
            else:
                self.author.clear()
            self._draft_ready = True
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            self.status.setText('Entwurf nicht lesbar; automatisches Überschreiben gesperrt: ' + str(exc))
        self.refresh_revisions()
        self.draw_ecg()
        self.draw()

    def draw_ecg(self, *_):
        self.ecg_plot.clear()
        if self.signal is not None and self.reference:
            fs = self.reference['sampling_rate_hz']
            curve = self.ecg_plot.plot(np.arange(len(self.signal))/fs,
                self.signal[:, self.lead.currentIndex()], pen='#dce5ef' if self._dark else '#172333')
            curve.setDownsampling(auto=True, method='peak')
            curve.setClipToView(True)
        self.ecg_cursor = pg.InfiniteLine(self.time.value(), pen=pg.mkPen('#888888', style=Qt.PenStyle.DashLine))
        self.ecg_plot.addItem(self.ecg_cursor)

    def document(self):
        return {'schema_version': '1.0.0', 'kind': 'manual_laddergram',
            'signal_reference': self.reference, 'events': self.events, 'edges': self.edges,
            'author': self.author.text().strip(), 'created_at': datetime.now(timezone.utc).isoformat(),
            'clinical_claim_allowed': False}

    def remember(self):
        self.history.append(json.dumps([self.events, self.edges]))
        self.history = self.history[-100:]
        self.status.setText('Ungesicherte manuelle Diagrammänderungen')

    def add_event(self):
        if self.reference is None or not self._draft_ready:
            return
        if len(self.events) >= 1000:
            self.status.setText('Maximal 1000 Aktivierungen pro Diagramm')
            return
        self.remember()
        sample = min(self.reference['sample_count']-1, round(self.time.value()*self.reference['sampling_rate_hz']))
        self.events.append({'id': uuid.uuid4().hex, 'level': self.level.currentText(), 'sample': sample})
        self.draw()

    def add_edge(self):
        if self.reference is None or not self._draft_ready:
            return
        edge = {'source': self.source.currentData(), 'target': self.target.currentData(), 'kind': self.kind.currentData()}
        if edge in self.edges:
            return
        trial = dict(self.document(), edges=self.edges + [edge])
        try:
            validate_diagram(trial, self.reference)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.remember()
        self.edges.append(edge)
        self.draw()

    def remove_event(self):
        identifier = self.source.currentData()
        if identifier is None:
            return
        self.remember()
        self.events = [item for item in self.events if item['id'] != identifier]
        self.edges = [item for item in self.edges if identifier not in (item['source'], item['target'])]
        self.draw()

    def move_event(self, identifier, seconds):
        if not self._draft_ready or self.reference is None:
            return
        updated = [dict(item) for item in self.events]
        item = next((item for item in updated if item['id'] == identifier), None)
        if item is None:
            return
        item['sample'] = max(0, min(self.reference['sample_count']-1,
                                   round(seconds*self.reference['sampling_rate_hz'])))
        try:
            validate_diagram(dict(self.document(), events=updated), self.reference)
        except ValueError as exc:
            self.status.setText('Verschieben abgelehnt: ' + str(exc))
            self.draw()
            return
        if updated != self.events:
            self.remember()
            self.events = updated
        self.draw()

    def undo(self):
        if self.history:
            self.events, self.edges = json.loads(self.history.pop())
            self.status.setText('Ungesicherter Diagrammstand nach Rückgängig')
            self.draw()

    def clicked(self, event, plot=None):
        plot = self.plot if plot is None else plot
        if self.reference and plot.sceneBoundingRect().contains(event.scenePos()):
            point = plot.plotItem.vb.mapSceneToView(event.scenePos())
            if 0 <= point.x() <= self.time.maximum():
                self.time.setValue(point.x())
                self.timeSelected.emit(point.x())
                if event.double():
                    self.add_event()

    def draw(self):
        self.plot.clear()
        fs = self.reference['sampling_rate_hz'] if self.reference else 1
        duration = (self.reference['sample_count']-1)/fs if self.reference else 10
        for y in LEVELS.values():
            self.plot.plot([0, duration], [y, y], pen=pg.mkPen('#888888', style=Qt.PenStyle.DotLine))
        by_id = {item['id']: item for item in self.events}
        for item in self.events:
            t, y = item['sample']/fs, LEVELS[item['level']]
            line = pg.InfiniteLine(t, movable=True, span=((y+.5-.18)/3, (y+.5+.18)/3),
                                  bounds=(0, duration), pen=pg.mkPen('#56b4e9' if self._dark else '#0072b2', width=3))
            line.sigPositionChangeFinished.connect(lambda line, key=item['id']: self.move_event(key, line.value()))
            self.plot.addItem(line)
        for edge in self.edges:
            a, b = by_id[edge['source']], by_id[edge['target']]
            x0, x1, y0, y1 = a['sample']/fs, b['sample']/fs, LEVELS[a['level']], LEVELS[b['level']]
            if edge['kind'] == KINDS[1]:
                x1, y1 = (x0+x1)/2, (y0+y1)/2
                self.plot.plot([x1], [y1], pen=None, symbol='x', symbolPen='#d55e00', symbolSize=12)
            self.plot.plot([x0, x1], [y0, y1], pen=pg.mkPen('#d55e00', width=2))
        self.plot.setYRange(-.5, 2.5, padding=0)
        self.cursor = pg.InfiniteLine(self.time.value(), pen=pg.mkPen('#888888', style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.cursor)
        for combo in (self.source, self.target):
            previous = combo.currentData()
            combo.clear()
            for index, item in enumerate(self.events):
                combo.addItem(f'{index+1}: {item["level"]} · {item["sample"]/fs:.4f} s', item['id'])
            selected = combo.findData(previous)
            if selected >= 0:
                combo.setCurrentIndex(selected)
        try:
            self.persist_draft()
        except (OSError, ValueError) as exc:
            self.status.setText('Entwurfsicherung fehlgeschlagen: ' + str(exc))

    def persist_draft(self):
        if self.reference is None or not self._draft_ready:
            return
        if not self.events and not self.edges and not (self.root / 'draft_recovery.json').exists():
            return
        document = validate_diagram(self.document(), self.reference)
        document['status'] = 'recovery_not_reviewed'
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.root / 'draft_recovery.json', document)

    def refresh_revisions(self):
        self.revisions.clear()
        if self.root and self.root.exists():
            for path in sorted(self.root.glob('*.json')):
                if path.name == 'draft_recovery.json':
                    continue
                self.revisions.addItem(path.stem, str(path))

    def export(self):
        if self.reference is None or not self.author.text().strip():
            self.status.setText('Export benötigt Fall und Bearbeiter')
            return
        from .lab_laddergram_export import export_laddergram
        try:
            document = validate_diagram(self.document(), self.reference)
            directory = self.root / 'exports'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (uuid.uuid4().hex + '.' + self.export_format.currentText().lower())
            if self.export_ecg.isChecked() and self.signal is None:
                raise ValueError('Kein Originalsignal für den EKG-Streifen verfügbar')
            document['export_ecg'] = export_laddergram(
                path, document,
                signal=self.signal if self.export_ecg.isChecked() else None,
                lead=self.lead.currentText())
            atomic_write_json(path.with_suffix('.json'), document)
        except (ValueError, OSError) as exc:
            self.status.setText('Export fehlgeschlagen: ' + str(exc))
            return
        self.status.setText('Lokaler Forschungsdiagramm-Export: ' + str(path))

    def save(self):
        if self.reference is None:
            return
        if not self.author.text().strip():
            self.status.setText('Bearbeiter erforderlich')
            return
        try:
            document = validate_diagram(self.document(), self.reference)
            self.root.mkdir(parents=True, exist_ok=True)
            name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex
            atomic_write_json(self.root / (name + '.json'), document)
        except (ValueError, OSError) as exc:
            self.status.setText('Nicht gespeichert: ' + str(exc))
            return
        self.refresh_revisions()
        self.status.setText('Neue lokale Forschungsrevision gespeichert; keine Diagnosefreigabe')

    def load(self):
        path = self.revisions.currentData()
        if not path:
            return
        if QMessageBox.question(self, 'Revision laden', 'Aktuellen Diagrammentwurf durch gewählte Revision ersetzen?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            from pathlib import Path
            document = validate_diagram(json.loads(Path(path).read_text(encoding='utf-8')), self.reference)
        except (ValueError, TypeError, OSError, AttributeError) as exc:
            self.status.setText('Nicht geladen: ' + str(exc))
            return
        self.remember()
        self.events, self.edges = document['events'], document['edges']
        self.author.setText(document.get('author', ''))
        self.draw()
        self.status.setText('Gewählte historische manuelle Revision geladen')
