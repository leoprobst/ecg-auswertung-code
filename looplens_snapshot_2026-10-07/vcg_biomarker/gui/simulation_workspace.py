"""Dedicated electrophysiology bench. Patient viewer and production are not its engine."""
from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QProcess, QProcessEnvironment, QStandardPaths, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
    QSizePolicy,
)

from ..simulation.model import CELL_NAMES, PRESETS, UNSUPPORTED, Electrode, Experiment, electrode_preset
from ..simulation.solver import LIMITATIONS, export_lab_signal, load_result
from ..simulation.visualization import graph_slice, process_summary, surface_segments
from .motion import current_motion_policy

LOGGER = logging.getLogger(__name__)


def tube_mesh(start, end, radius=.8, sides=8):
    """A cable, not a floating decorative stroke. Coordinates stay in mm."""
    direction = np.asarray(end) - start
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return np.empty((0, 3)), np.empty((0, 3), int)
    direction /= length
    helper = np.array([0., 0., 1.]) if abs(direction[2]) < .9 else np.array([0., 1., 0.])
    a = np.cross(direction, helper)
    a /= np.linalg.norm(a)
    b = np.cross(direction, a)
    angle = np.arange(sides) * 2 * np.pi / sides
    ring = radius * (np.cos(angle)[:, None] * a + np.sin(angle)[:, None] * b)
    vertices = np.vstack([ring + start, ring + end])
    faces = []
    for j in range(sides):
        k = (j + 1) % sides
        faces.extend([(j, k, j + sides), (k, k + sides, j + sides)])
    return vertices, np.asarray(faces)


def update_mesh_colors(item, colors):
    # setMeshData(vertexColors=...) would replace the mesh and lose its faces!
    item.opts["meshdata"].setVertexColors(np.asarray(colors, np.float32))
    item.meshDataChanged()


class SimulationWorkspace(QWidget):
    labSignalReady = pyqtSignal(str)

    def __init__(self, parent=None, target_provider=None, workspace_root=None):
        super().__init__(parent)
        self.target_provider = target_provider
        self.root = Path(workspace_root) if workspace_root else Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
        ) / "cardiac_simulation"
        self.result = None
        self.result_path = None
        self.job_root = None
        self._stdout = ""
        self._stderr = ""
        self._closing = False
        self._gl = None
        self._gl_items = []
        self._gl_plane = None
        self._slice_cache = None
        self._process_summary = None
        self._play_clock = None
        self.selected_node = 0
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read_progress)
        self.process.readyReadStandardError.connect(self._read_error)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._advance)
        current_motion_policy().changed.connect(self._motion_changed)
        layout = QVBoxLayout(self)
        header = QLabel("CARDIAC EXPLORER   /   ELEKTROPHYSIOLOGIE")
        header.setStyleSheet("font-size: 23px; font-weight: 700; padding: 10px 6px; color: #299dba;")
        header.setWordWrap(True)
        layout.addWidget(header)
        note = QLabel("Eigenständige Geweberechnung · keine Animation eines Patientenbefunds. "
                      "Reaktions–Diffusionsmodell, generische Anatomie, ausschließlich Forschung.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.live_strip = QLabel("●  Bereit     ·     3D-Graphmodell     ·     Synchrone Schnitte und Signale")
        self.live_strip.setStyleSheet("padding: 9px 12px; border-radius: 8px; background: #173044; color: #dcecf4;")
        layout.addWidget(self.live_strip)
        split = QSplitter()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls = QWidget()
        self.controls_layout = QVBoxLayout(controls)
        self._make_controls()
        scroll.setWidget(controls)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(400)
        split.addWidget(scroll)
        self.views = QTabWidget()
        self.views.setMinimumWidth(480)
        self.views.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self._make_views()
        from .patient_twin_panel import IonicCellPanel, PatientTwinPanel
        self.patient_panel = PatientTwinPanel(self._launch)
        self.ionic_panel = IonicCellPanel(self._launch, self.patient_panel)
        self.views.addTab(self.patient_panel, "Patientenzwilling")
        self.views.addTab(self.ionic_panel, "Ionenlabor")
        split.addWidget(self.views)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1000])
        layout.addWidget(split, 1)
        layout.addWidget(self.transport)
        self.status = QLabel("Bereit. Noch kein Experiment berechnet.")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        descriptions = {
            "Experiment berechnen": "Aktuelle Parameter einfrieren und einen separaten Forschungsprozess starten.",
            "Abbrechen": "Nur dieses Simulationsexperiment am nächsten sicheren Punkt abbrechen.",
            "Gespeichertes Experiment öffnen …": "Ein versioniertes lokales Simulationsergebnis samt Provenienz öffnen.",
            "EKG exportieren / zum Labor …": "Explizite illustrative mV-Skala wählen; als synthetisches EKG im Labor öffnen.",
            "3D-Gewebe öffnen": "Anatomische Gewebeoberflächen mit den berechneten Potentialen zeigen.",
            "Abspielen": "Gespeicherte elektrische Aktivität mit 0,25-facher Geschwindigkeit abspielen.",
        }
        for button in self.findChildren(QPushButton):
            if button.text() in descriptions:
                button.setToolTip(descriptions[button.text()])
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(80)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for label in controls.findChildren(QLabel):
            label.setWordWrap(True)

    def _spin(self, form, label, lo, hi, value, decimals=1):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setValue(value)
        form.addRow(label, spin)
        return spin

    def _make_controls(self):
        box = QGroupBox("01  ·  Gewebe & Protokoll")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.preset = QComboBox()
        for key, title in PRESETS.items():
            self.preset.addItem(title, key)
        form.addRow("Experiment", self.preset)
        from ..simulation.conduction import VARIANTS
        self.conduction_variant = QComboBox()
        for key, title in VARIANTS.items():
            self.conduction_variant.addItem(title, key)
        form.addRow("Leitungsnetz (Hypothese)", self.conduction_variant)
        self.pathway = QComboBox()
        self.pathway.addItem("Keine zusätzliche Änderung", "")
        for name in ("bachmann", "coronary_sinus_musculature", "posterosuperior_connection", "lbb_anterior", "lbb_posterior",
                     "lbb_septal_from_lbb", "rbb", "av_slow_access", "accessory_av_connection"):
            self.pathway.addItem(name, name)
        form.addRow("Pfad-Kopplung ändern", self.pathway)
        self.pathway_scale = self._spin(form, "Pfad-Faktor (0 = Block)", 0, 3, 1, 2)
        self.region = self._spin(form, "AHA-Region (LV)", 1, 17, 7, 0)
        self.severity = self._spin(form, "Stärke (Mechanismus)", 0, 1, .8, 2)
        self.diffusion = self._spin(form, "Gewebekopplung ×", .05, 5, 1, 2)
        self.anisotropy = self._spin(form, "Längs/quer-Verhältnis", 1, 10, 3, 1)
        self.recovery = self._spin(form, "Repolarisationszeit ×", .25, 3, 1, 2)
        self.duration = self._spin(form, "Dauer (ms)", 10, 3000, 600, 0)
        self.output_step = self._spin(form, "Ausgabeabstand (ms)", .5, 10, 2, 1)
        self.cycle = self._spin(form, "S1-Zyklus (ms)", 150, 3000, 800, 0)
        self.stimulus = self._spin(form, "Stimulusdauer (ms)", .1, 10, 2, 1)
        self.strength = self._spin(form, "Stimulus (normiert/ms)", .01, 2, .8, 2)
        self.s2 = self._spin(form, "S2-Zeit (ms; 0 = aus)", 0, 2999, 0, 0)
        self.site = QComboBox()
        self.site.addItems(["his", "sa_node", "lv_apical", "rv_apical", "lv_lateral_base", "rv_outflow",
                            "lv_mid_septal", "rv_free_wall_base"])
        form.addRow("Stimulationsort", self.site)
        self.controls_layout.addWidget(box)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self._preset_changed()
        actions = QHBoxLayout()
        self.run_button = QPushButton("Experiment berechnen")
        self.run_button.clicked.connect(self.start_experiment)
        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        self.controls_layout.addLayout(actions)
        inverse = QGroupBox("02  ·  Vergleich mit Labor-EKG")
        f = QFormLayout(inverse)
        f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.target_start = self._spin(f, "Segmentbeginn im EKG (s)", 0, 86400, 0, 3)
        self.aligned = QCheckBox("Zeitbezug bewusst ausgerichtet")
        f.addRow(self.aligned)
        self.fit_button = QPushButton("3 Leitungshypothesen vergleichen")
        self.fit_button.setToolTip("Referenz, linker und rechter Leitungsastblock. Keine automatische Zeitstauchung; maximal drei Versuche.")
        self.fit_button.clicked.connect(self.start_fit)
        f.addRow(self.fit_button)
        self.controls_layout.addWidget(inverse)
        self.controls_layout.addWidget(QLabel("Ein Hintergrundprozess · ein nativer Thread · 2 GiB RAM-Reserve"))
        open_button = QPushButton("Gespeichertes Experiment öffnen …")
        open_button.clicked.connect(self.open_result)
        self.controls_layout.addWidget(open_button)
        self.export_button = QPushButton("EKG exportieren / zum Labor …")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.send_to_lab)
        self.controls_layout.addWidget(self.export_button)
        capture = QPushButton("Aktuelle Ansicht als PNG sichern …")
        capture.setToolTip("Aktuelle Ansicht als PNG-Datei speichern")
        capture.clicked.connect(self.save_view)
        self.controls_layout.addWidget(capture)
        self.controls_layout.addStretch()

    def _make_views(self):
        stage = QWidget()
        lay = QVBoxLayout(stage)
        tools = QHBoxLayout()
        self.field = QComboBox()
        self.field.addItems(["Potential u · normiert", "Erste Aktivierung · ms", "Gewebephänotyp",
                             "Erholung nach Aktivierung"])
        self.field.currentIndexChanged.connect(self.draw_frame)
        self.plane = QComboBox()
        self.plane.addItems(["Frontal (x/z)", "Transversal (x/y)", "Sagittal (y/z)"])
        self.plane.currentIndexChanged.connect(self._reset_slice_range)
        self.slice_mm = QDoubleSpinBox()
        self.slice_mm.setRange(-250, 250)
        self.slice_mm.setDecimals(1)
        self.slice_mm.setSingleStep(1.)
        self.slice_mm.valueChanged.connect(self._slice_changed)
        self.cut_slider = QSlider(Qt.Orientation.Horizontal)
        self.cut_slider.setRange(0, 1000)
        self.cut_slider.valueChanged.connect(self._cut_slider_changed)
        tools.addWidget(self.field)
        tools.addWidget(self.plane)
        tools.addWidget(QLabel("Schnittlage mm"))
        tools.addWidget(self.slice_mm)
        self.gl_button = QPushButton("3D aktivieren")
        self.gl_button.setToolTip("Dreidimensionale Ansicht öffnen, sofern eine anatomische Oberfläche vorliegt")
        self.gl_button.clicked.connect(self.show_3d)
        tools.addWidget(self.gl_button)
        lay.addLayout(tools)
        cut_controls = QHBoxLayout()
        cut_controls.addWidget(QLabel("Schnitt durch das Gewebe"))
        cut_controls.addWidget(self.cut_slider, 1)
        self.cut_label = QLabel("–")
        cut_controls.addWidget(self.cut_label)
        lay.addLayout(cut_controls)
        spatial = QSplitter(Qt.Orientation.Horizontal)
        self.slice_plot = pg.PlotWidget()
        self.slice_plot.setAspectLocked(True)
        self.slice_plot.showGrid(x=True, y=True, alpha=.15)
        self._slice_outline = pg.PlotDataItem(pen=pg.mkPen("#8fa9bb", width=1.5))
        self._slice_cloud = pg.ScatterPlotItem(size=4, pen=None)
        self._slice_electrodes = pg.ScatterPlotItem(size=13, symbol="+", pen=pg.mkPen("#edbd65", width=2))
        for item in (self._slice_outline, self._slice_cloud, self._slice_electrodes):
            self.slice_plot.addItem(item)
        self.slice_plot.scene().sigMouseClicked.connect(self._place_electrode)
        spatial.addWidget(self.slice_plot)
        self._gl_host = QWidget()
        self._gl_host_layout = QVBoxLayout(self._gl_host)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("3D-Deckkraft"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(65)
        self.opacity.valueChanged.connect(self._update_gl_colors)
        opacity_row.addWidget(self.opacity, 1)
        self._gl_host_layout.addLayout(opacity_row)
        self._gl_placeholder = QLabel("3D-Gewebe erscheint nach Laden eines berechneten Experiments.")
        self._gl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gl_placeholder.setWordWrap(True)
        self._gl_placeholder.setStyleSheet("background:#101a27;color:#dcecf4;border-radius:9px;padding:18px;")
        self._gl_host_layout.addWidget(self._gl_placeholder, 1)
        spatial.addWidget(self._gl_host)
        spatial.setStretchFactor(0, 1)
        spatial.setStretchFactor(1, 1)
        spatial.setSizes([520, 520])
        lay.addWidget(spatial, 1)
        self.legend = QLabel("Potential: dunkel = Ruhe, türkis = erregt. NaN-Aktivierung = nicht aktiviert, nicht 0 ms.")
        self.legend.setWordWrap(True)
        lay.addWidget(self.legend)
        row = QHBoxLayout()
        self.play = QPushButton("Abspielen")
        self.play.clicked.connect(self.toggle_play)
        self.previous_frame = QPushButton("-1 Bild")
        self.previous_frame.setToolTip("Ein berechnetes Bild zurück")
        self.previous_frame.clicked.connect(lambda: self.slider.setValue(max(0, self.slider.value()-1)))
        self.next_frame = QPushButton("+1 Bild")
        self.next_frame.setToolTip("Ein berechnetes Bild vorwärts")
        self.next_frame.clicked.connect(lambda: self.slider.setValue(min(self.slider.maximum(), self.slider.value()+1)))
        self.speed = QComboBox()
        for label, value in (("0,25×", .25), ("0,5×", .5), ("1×", 1.), ("2×", 2.)):
            self.speed.addItem(label, value)
        self.loop_play = QCheckBox("Schleife")
        self.loop_play.setChecked(True)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.draw_frame)
        self.time_label = QLabel("0 ms")
        row.addWidget(self.previous_frame)
        row.addWidget(self.play)
        row.addWidget(self.next_frame)
        row.addWidget(self.speed)
        row.addWidget(self.loop_play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.time_label)
        self.transport = QWidget()
        self.transport.setLayout(row)
        # The constructor places transport below the whole workspace.
        self.views.addTab(stage, "Gewebe · 2D / 3D")
        processes = QWidget()
        process_layout = QVBoxLayout(processes)
        process_note = QLabel("Berechnete Aktivierung und Erholung pro Gewebegruppe. "
                              "Die Kabel sind vereinfachte Leitungsbahnen; metabolische Einzelzellmodelle laufen im Ionenlabor.")
        process_note.setWordWrap(True)
        process_layout.addWidget(process_note)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Gewebegruppe"))
        self.process_group = QComboBox()
        for title, key in (("Alle Gruppen", "all"), ("Kammer", "ventricle"),
                           ("Vorhof", "atrium"), ("AV-Zone", "av_junction"),
                           ("Leitungskabel", "conduction_cable")):
            self.process_group.addItem(title, key)
        self.process_group.currentIndexChanged.connect(self._draw_processes)
        filter_row.addWidget(self.process_group)
        filter_row.addStretch(1)
        process_layout.addLayout(filter_row)
        self.process_plot = pg.PlotWidget()
        self.process_plot.setLabel("bottom", "Simulationszeit", units="ms")
        self.process_plot.setLabel("left", "Anteil der Knoten")
        self.process_plot.setYRange(0, 1)
        self.process_legend = self.process_plot.addLegend()
        self.process_plot.scene().sigMouseClicked.connect(self._seek_from_process)
        process_layout.addWidget(self.process_plot, 1)
        self.rate_plot = pg.PlotWidget()
        self.rate_plot.setLabel("bottom", "Simulationszeit", units="ms")
        self.rate_plot.setLabel("left", "Neue Erstaktivierung", units="Anteil/Bild")
        self.rate_legend = self.rate_plot.addLegend()
        self.rate_plot.scene().sigMouseClicked.connect(self._seek_from_rate)
        process_layout.addWidget(self.rate_plot, 1)
        self.process_readout = QLabel("Noch keine Geweberechnung geladen.")
        self.process_readout.setWordWrap(True)
        process_layout.addWidget(self.process_readout)
        metabolic_button = QPushButton("Einzelzell- und Stoffwechselmodelle im Ionenlabor öffnen")
        metabolic_button.setToolTip("Zum Ionenlabor mit Einzelzell- und Stoffwechselmodellen wechseln")
        metabolic_button.clicked.connect(lambda: self.views.setCurrentWidget(self.ionic_panel))
        process_layout.addWidget(metabolic_button)
        self.views.addTab(processes, "Prozesse & Aktivierung")
        traces = QWidget()
        tl = QVBoxLayout(traces)
        self.lead = QComboBox()
        self.lead.currentIndexChanged.connect(self.draw_trace)
        tl.addWidget(self.lead)
        self.trace = pg.PlotWidget()
        self.trace.setLabel("bottom", "Zeit", units="ms")
        self.trace.setLabel("left", "Pseudo-EKG", units="a.u.")
        tl.addWidget(self.trace, 1)
        self.cell_trace = pg.PlotWidget()
        self.cell_trace.setLabel("bottom", "Zeit", units="ms")
        self.cell_trace.setLabel("left", "Lokales Potential u", units="normiert")
        tl.addWidget(self.cell_trace, 1)
        self.cell_label = QLabel("Gewebeknoten im 2D-Schnitt anklicken: lokale Erregung und Repolarisation prüfen.")
        self.cell_label.setWordWrap(True)
        tl.addWidget(self.cell_label)
        self.views.addTab(traces, "Ableitungen")
        electrodes = QWidget()
        el = QVBoxLayout(electrodes)
        self.electrode_preset = QComboBox()
        self.electrode_preset.addItem("12-Kanal · Rumpfpositionen (Mason–Likar-artig, generisch)", "torso_12")
        self.electrode_preset.addItem("Zusätzlich rechts/posterior · generische Positionen", "extended")
        self.electrode_preset.addItem("Zusätzlich bipolares Katheterpaar", "intracardiac")
        self.electrode_preset.currentIndexChanged.connect(self.reset_electrodes)
        el.addWidget(self.electrode_preset)
        self.electrode_table = QTableWidget(0, 5)
        self.electrode_table.setHorizontalHeaderLabels(["Name", "x links (mm)", "y posterior", "z kranial", "Art"])
        self.electrode_table.horizontalHeader().setStretchLastSection(True)
        el.addWidget(self.electrode_table)
        self.placing = QCheckBox("Elektrode im Schnitt platzieren")
        el.addWidget(self.placing)
        note = QLabel("Koordinaten editierbar. Schnittklick setzt alle drei Koordinaten in der gewählten Ebene. "
                      "Oberflächenpositionen werden nicht anatomisch verifiziert. Intrakardial: CathA–CathB; "
                      "unipolare Ableitungen gegen Wilson-Terminal. Änderungen gelten erst für das nächste Experiment.")
        note.setWordWrap(True)
        el.addWidget(note)
        self.reset_electrodes()
        self.views.addTab(electrodes, "Elektroden & Katheter")
        self.fit_view = QTextBrowser()
        self.fit_view.setPlainText("Keine Suche gestartet. Eine gute Signalpassung ist kein Nachweis eines Ursprungs oder Blocks.")
        self.views.addTab(self.fit_view, "Hypothesenvergleich")
        self.provenance = QTextBrowser()
        self.provenance.setPlainText("MODELLGRENZEN\n\n" + "\n\n".join(LIMITATIONS)
                                     + "\n\nNOCH NICHT QUANTITATIV MODELLIERT\n" + "\n".join(UNSUPPORTED.values())
                                     + "\n\nIonenlabor: veröffentlichte Einzelzellmodelle (separate Rechnung). "
                                       "Das 3D-Gewebe bleibt ein reduziertes Modell, kein Ionen-/FEM-Ganzherzzwilling.")
        self.views.addTab(self.provenance, "Modell & Provenienz")

    def reset_electrodes(self):
        items = electrode_preset(self.electrode_preset.currentData())
        self.electrode_table.setRowCount(len(items))
        for row, e in enumerate(items):
            for col, value in enumerate((e.name, *e.xyz_mm, e.kind)):
                self.electrode_table.setItem(row, col, QTableWidgetItem(str(value)))

    def _preset_changed(self, *_):
        key = self.preset.currentData()
        self.severity.setEnabled(key in {"slow_zone", "short_ap", "long_ap"})
        self.region.setEnabled(key in {"scar", "slow_zone", "short_ap", "long_ap"})

    def specification(self):
        electrodes = []
        for row in range(self.electrode_table.rowCount()):
            items = [self.electrode_table.item(row, col).text() for col in range(5)]
            electrodes.append(Electrode(items[0], tuple(float(x) for x in items[1:4]), items[4]))
        out = Experiment(duration_ms=self.duration.value(), output_step_ms=self.output_step.value(),
                         cycle_ms=self.cycle.value(), stimulus_ms=self.stimulus.value(),
                         stimulus_strength=self.strength.value(), stimulus_site=self.site.currentText(),
                         diffusion_scale=self.diffusion.value(), anisotropy_ratio=self.anisotropy.value(),
                         recovery_scale=self.recovery.value(), preset=self.preset.currentData(),
                         region=int(self.region.value()), severity=self.severity.value(),
                         s2_ms=self.s2.value() or None, electrodes=tuple(electrodes))
        out = replace(out, conduction_variant=self.conduction_variant.currentData(),
                      pathway_scales={self.pathway.currentData(): self.pathway_scale.value()} if self.pathway.currentData() else {})
        out.validate()
        return out

    def _launch(self, request, target=None):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise ValueError("Ein Simulationsauftrag läuft bereits")
        request = dict(request)
        request.setdefault("patient_profile", self.patient_panel.profile().to_dict())
        self.root.mkdir(parents=True, exist_ok=True)
        self.job_root = self.root / ("experiment-" + uuid.uuid4().hex[:12])
        self.job_root.mkdir()
        (self.job_root / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
        if target is not None:
            np.savez_compressed(self.job_root / "target.npz", **target)
        env = QProcessEnvironment.systemEnvironment()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.insert(key, "1")
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        self._stdout = self._stderr = ""
        self.run_button.setEnabled(False)
        self.fit_button.setEnabled(False)
        self.ionic_panel.set_busy(True)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText("Experiment wird im eigenen, niedrig priorisierten Prozess vorbereitet …")
        self.process.start(sys.executable, ["-m", "vcg_biomarker.simulation.worker", str(self.job_root)])

    def start_experiment(self):
        try:
            self._launch({"kind": "forward", "experiment": self.specification().to_dict()})
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def start_fit(self):
        try:
            if not self.aligned.isChecked():
                raise ValueError("Zuerst das beobachtete Segment zeitlich ausrichten und bestätigen")
            if self.target_provider is None:
                raise ValueError("Kein EKG-Labor verbunden")
            signal, fs, names, source = self.target_provider()
            spec = self.specification()
            start = int(round(self.target_start.value() * fs))
            stop = start + int(np.ceil(spec.duration_ms / 1000 * fs)) + 1
            if stop > len(signal):
                raise ValueError("Das gewählte Segment liegt außerhalb des Labor-EKGs")
            target = {"signal": np.asarray(signal[start:stop]), "fs": fs, "lead_names": np.asarray(names)}
            candidates = [replace(spec, preset=k).to_dict() for k in ("reference", "lbbb", "rbbb")]
            self._launch({"kind": "inverse", "candidates": candidates, "source": source,
                          "target_start_seconds": start / fs}, target)
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))

    def cancel(self):
        if self.job_root and self.process.state() != QProcess.ProcessState.NotRunning:
            (self.job_root / "cancel").touch()
            self.status.setText("Abbruch angefordert; aktueller kleiner Rechenschritt läuft aus …")

    def shutdown(self):
        """Only this workspace's child process, never the background analysis."""
        self.play_timer.stop()
        if self.ionic_panel.tissue_gl is not None:
            self.ionic_panel.tissue_gl.close()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.cancel()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(1500)

    def _read_error(self):
        self._stderr += bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")

    def _process_error(self, error):
        self.status.setText("Simulationsprozess: " + self.process.errorString())
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.run_button.setEnabled(True)
            self.fit_button.setEnabled(True)
            self.ionic_panel.set_busy(False)
            self.cancel_button.setEnabled(False)
            self.progress.setRange(0, 1000)

    def _read_progress(self):
        self._stdout += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._stdout:
            line, self._stdout = self._stdout.split("\n", 1)
            try:
                item = json.loads(line)
            except ValueError:
                continue
            phase = item.get("phase")
            if phase == "preconditioning":
                self.progress.setRange(0, 0)
                self.status.setText(f"Vorkonditionierung: {item['done']} von höchstens {item['total']} Schlägen · "
                                    + ("Periodizitätskriterium erreicht; Messung folgt" if item.get("converged")
                                       else "Konvergenz wird geprüft; Restdauer offen"))
            elif phase == "computing":
                self.progress.setRange(0, 1000)
                self.progress.setValue(int(1000 * item["done"] / max(item["total"], 1)))
                eta = item.get("eta_seconds")
                self.status.setText(f"Berechnung {item['done']}/{item['total']} · "
                                    + (f"Restschätzung {eta:.0f} s" if eta is not None else "ETA noch unbekannt"))
            else:
                self.status.setText({"building_geometry": "Generisches Rechennetz wird aufgebaut …",
                                     "waiting_for_ram": "Wartet auf RAM · Produktionslauf hat Vorrang",
                                     "saving": "Ergebnis und Provenienz werden gespeichert …",
                                     "complete": "Experiment vollständig abgeschlossen",
                                     "cancelled": "Experiment abgebrochen"}.get(phase, item.get("message", str(phase))))

    def _finished(self, code, status):
        self._read_error()
        self._read_progress()
        self.run_button.setEnabled(True)
        self.fit_button.setEnabled(True)
        self.ionic_panel.set_busy(False)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 1000)
        if self.job_root:
            (self.job_root / "worker_stderr.log").write_text(self._stderr, encoding="utf-8")
        if code == 0:
            self.progress.setValue(1000)
            try:
                if (self.job_root / "result.npz").is_file():
                    self.display_result(self.job_root / "result.npz")
                elif (self.job_root / "ionic.npz").is_file():
                    self.ionic_panel.show_result(self.job_root / "ionic.npz")
                    self.views.setCurrentWidget(self.ionic_panel)
                elif (self.job_root / "ecme.npz").is_file():
                    self.ionic_panel.show_metabolism(self.job_root / "ecme.npz")
                    self.views.setCurrentWidget(self.ionic_panel)
                elif (self.job_root / "mitochondria.npz").is_file():
                    self.ionic_panel.show_metabolism(self.job_root / "mitochondria.npz")
                    self.views.setCurrentWidget(self.ionic_panel)
                elif (self.job_root / "tissue.npz").is_file():
                    self.ionic_panel.show_tissue(self.job_root / "tissue.npz")
                    self.views.setCurrentWidget(self.ionic_panel)
                elif (self.job_root / "ring.npz").is_file():
                    self.ionic_panel.show_tissue(self.job_root / "ring.npz")
                    self.views.setCurrentWidget(self.ionic_panel)
                elif (self.job_root / "fit.json").is_file():
                    rows = json.loads((self.job_root / "fit.json").read_text(encoding="utf-8"))["results"]
                    text = "SIGNALPASSUNG ≠ DIAGNOSE\nKeine unabhängige Validierung; alternative Ursachen bleiben möglich.\n\n"
                    for row in rows:
                        text += f"{row['preset']}: {row['status']} · relativer Fehler {row.get('relative_squared_error', '–')}\n"
                    self.fit_view.setPlainText(text + "\nVollständige Parameter/Hashes: " + str(self.job_root / "fit.json"))
                    self.views.setCurrentWidget(self.fit_view)
            except (OSError, ValueError, KeyError) as exc:
                self.status.setText("Ergebnis nicht darstellbar: " + str(exc))
        elif code != 2:
            self.status.setText("Experiment fehlgeschlagen · " + self._stderr[-1000:])

    def open_result(self):
        path, _ = QFileDialog.getOpenFileName(self, "Simulation öffnen", str(self.root), "Simulation (*.npz)")
        if path:
            try:
                with np.load(path, allow_pickle=False) as data:
                    schema = json.loads(str(data["metadata"])).get("schema")
                if schema in {"published-ecme-assay/1", "published-mitochondrial-assay/1"}:
                    self.ionic_panel.show_metabolism(Path(path))
                    self.views.setCurrentWidget(self.ionic_panel)
                    return
                self.display_result(Path(path))
            except (OSError, ValueError, KeyError) as exc:
                QMessageBox.warning(self, "Simulation", str(exc))

    def display_result(self, path):
        self.play_timer.stop()
        self.play.setText("Abspielen")
        result = load_result(path)
        self._gl_items.clear()  # old vertex mapping must never index the new tissue
        self._slice_cache = None
        self.result = result
        self._process_summary = process_summary(result["times_ms"], result["voltage"],
                                                result["activation_ms"], result["cells"])
        self.selected_node = 0
        self.result_path = Path(path)
        self.slider.setRange(0, len(self.result["times_ms"]) - 1)
        self.slider.setValue(0)
        self.lead.clear()
        self.lead.addItems(list(self.result["lead_names"]))
        self.provenance.setPlainText(json.dumps(self.result["metadata"], indent=2, ensure_ascii=False))
        self.export_button.setEnabled(True)
        self.status.setText(f"Experiment geladen · {self.result['metadata']['n_nodes']:,} Knoten · "
                            f"{100*self.result['metadata']['activated_fraction']:.1f} % erregbares Gewebe aktiviert · Forschungsmodell")
        self._draw_processes()
        if self._gl is not None:
            self._build_gl()
        self._reset_slice_range()
        self.draw_frame()
        self.draw_trace()
        if self.views.currentIndex() == 0:
            self.show_3d()

    def _reset_slice_range(self, *_):
        if self.result is not None:
            a, b, normal = ((0, 2, 1), (0, 1, 2), (1, 2, 0))[self.plane.currentIndex()]
            xyz = self.result["xyz"]
            self.slice_plot.setRange(xRange=(float(xyz[:, a].min()) - 10, float(xyz[:, a].max()) + 10),
                                     yRange=(float(xyz[:, b].min()) - 10, float(xyz[:, b].max()) + 10), padding=.05)
            minimum, maximum = float(xyz[:, normal].min()), float(xyz[:, normal].max())
            self.slice_mm.blockSignals(True)
            self.slice_mm.setRange(minimum, maximum)
            self.slice_mm.setValue(float(np.median(xyz[:, normal])))
            self.slice_mm.blockSignals(False)
            self._slice_changed()

    def _cut_slider_changed(self, position):
        if self.result is None:
            return
        low, high = self.slice_mm.minimum(), self.slice_mm.maximum()
        self.slice_mm.setValue(low + (high-low)*position/1000)

    def _slice_changed(self, *_):
        if self.result is None:
            return
        low, high = self.slice_mm.minimum(), self.slice_mm.maximum()
        progress = 0 if high == low else int(round(1000*(self.slice_mm.value()-low)/(high-low)))
        self.cut_slider.blockSignals(True)
        self.cut_slider.setValue(progress)
        self.cut_slider.blockSignals(False)
        axis_name = ("x", "y", "z")[(1, 2, 0)[self.plane.currentIndex()]]
        self.cut_label.setText(f"{axis_name} = {self.slice_mm.value():.1f} mm")
        self._slice_cache = None
        self.draw_frame()
        self._update_gl_plane()

    def _slice_geometry(self):
        data = self.result
        _, _, normal = ((0, 2, 1), (0, 1, 2), (1, 2, 0))[self.plane.currentIndex()]
        key = (id(data), normal, self.slice_mm.value())
        if self._slice_cache is not None and self._slice_cache[0] == key:
            return self._slice_cache[1:]
        section = graph_slice(data["xyz"], data["edges"], normal, self.slice_mm.value())
        outlines = []
        if data["metadata"].get("geometry_kind", "").startswith("generic_anatomical_graph"):
            from .heart3d.model import default_model
            model = default_model()
            for name in ("wall", "ra_wall", "la_wall"):
                part = model.parts[name]
                outlines.append(surface_segments(part.vertices, part.faces, normal, self.slice_mm.value()))
        outline = np.concatenate(outlines) if outlines else np.empty((0, 2, 3))
        self._slice_cache = (key, section, outline)
        return section, outline

    def _colors(self):
        data = self.result
        if self.field.currentIndex() == 0:
            values = data["voltage"][self.slider.value()]
            ratio = np.clip(values, 0, 1)
            text = "u 0 → 1 · normiertes Potential (keine mV). Farbsättigung nur Darstellung."
        elif self.field.currentIndex() == 1:
            values = data["activation_ms"]
            ratio = values / max(1, data["times_ms"][-1])
            text = f"Erste Aktivierung 0 → {data['times_ms'][-1]:.0f} ms · grau = nicht aktiviert."
        elif self.field.currentIndex() == 2:
            values = data["cells"]
            ratio = values / (len(CELL_NAMES) - 1)
            text = " · ".join(f"{i}: {name}" for i, name in enumerate(CELL_NAMES))
        else:
            current = data["times_ms"][self.slider.value()]
            activation = data["activation_ms"]
            values = np.where(np.isfinite(activation) & (activation <= current),
                              data["voltage"][self.slider.value()] < .13, np.nan)
            ratio = values.astype(float)
            text = "Erholung: hell = nach erster Aktivierung wieder unter u=0,13; grau = noch nicht erstmals aktiviert."
        colors = pg.colormap.get("viridis").map(np.nan_to_num(ratio), mode="float")
        colors[~np.isfinite(values)] = (.25, .28, .33, 1.)
        self.legend.setText(text)
        return colors

    def draw_frame(self, *_):
        if self.result is None:
            return
        axes = ((0, 2, 1), (0, 1, 2), (1, 2, 0))[self.plane.currentIndex()]
        a, b, normal = axes
        section, outline = self._slice_geometry()
        colors = self._colors()
        self.slice_plot.setLabel("bottom", ("links", "posterior", "kranial")[a], units="mm")
        self.slice_plot.setLabel("left", ("links", "posterior", "kranial")[b], units="mm")
        if len(outline):
            points = np.concatenate((outline[:, :, [a, b]], np.full((len(outline), 1, 2), np.nan)), axis=1).reshape(-1, 2)
            self._slice_outline.setData(points[:, 0], points[:, 1], connect="finite")
        else:
            self._slice_outline.setData([], [])
        if len(section.xyz):
            fraction = section.weight_end[:, None]
            cut_colors = colors[section.start]*(1-fraction) + colors[section.end]*fraction
            brushes = [pg.mkBrush(*(np.clip(color*255, 0, 255).astype(int))) for color in cut_colors]
            self._slice_cloud.setData(x=section.xyz[:, a], y=section.xyz[:, b], brush=brushes)
        else:
            self._slice_cloud.setData(x=[], y=[])
        self.time_label.setText(f"{self.result['times_ms'][self.slider.value()]:.0f} ms")
        self.live_strip.setText(f"●  {self.result['times_ms'][self.slider.value()]:.0f} ms  ·  "
                                f"{len(section.xyz):,} Gitterkanten im Schnitt  ·  {self.plane.currentText()}")
        if hasattr(self, "trace_cursor"):
            self.trace_cursor.setValue(self.result['times_ms'][self.slider.value()])
        if hasattr(self, "cell_cursor"):
            self.cell_cursor.setValue(self.result['times_ms'][self.slider.value()])
        electrode_positions = np.asarray([e["xyz_mm"] for e in
                                          self.result["metadata"]["experiment"]["electrodes"]], float)
        selected = np.abs(electrode_positions[:, normal]-self.slice_mm.value()) <= 5
        self._slice_electrodes.setData(x=electrode_positions[selected, a], y=electrode_positions[selected, b])
        if self._gl is not None and self.views.currentIndex() == 0:
            self._update_gl_colors()
        self._update_process_cursor()

    def draw_trace(self, *_):
        self.trace.clear()
        self.cell_trace.clear()
        if self.result is not None and self.lead.currentIndex() >= 0:
            self.trace.plot(self.result["times_ms"], self.result["ecg"][:, self.lead.currentIndex()],
                            pen=pg.mkPen("#35b3c6", width=2))
            self.trace_cursor = pg.InfiniteLine(pos=self.result['times_ms'][self.slider.value()], pen="#e4be6e")
            self.trace.addItem(self.trace_cursor)
            self.cell_trace.plot(self.result["times_ms"], self.result["voltage"][:, self.selected_node],
                                 pen=pg.mkPen("#c198d9", width=2))
            self.cell_cursor = pg.InfiniteLine(pos=self.result['times_ms'][self.slider.value()], pen="#e4be6e")
            self.cell_trace.addItem(self.cell_cursor)
            activation = self.result['activation_ms'][self.selected_node]
            activation_text = f"{activation:.1f} ms" if np.isfinite(activation) else "nicht aktiviert"
            self.cell_label.setText(f"Knoten {self.selected_node} · "
                                   f"{CELL_NAMES[int(self.result['cells'][self.selected_node])]} · "
                                   f"Erste Aktivierung: {activation_text} · Gewebephänotyp, kein Ionenkanalmodell")

    def _draw_processes(self):
        if self.result is None or self._process_summary is None:
            return
        self.process_plot.clear()
        self.rate_plot.clear()
        self.process_legend.clear()
        self.rate_legend.clear()
        times = self.result["times_ms"]
        colors = {"ventricle": "#0072b2", "atrium": "#e69f00",
                  "av_junction": "#cc79a7", "conduction_cable": "#009e73"}
        names = {"ventricle": "Kammer", "atrium": "Vorhof", "av_junction": "AV-Zone",
                 "conduction_cable": "Leitungskabel"}
        for key, metrics in self._process_summary.items():
            if self.process_group.currentData() not in ("all", key):
                continue
            color = colors[key]
            self.process_plot.plot(times, metrics["active_fraction"],
                                   pen=pg.mkPen(color, width=2), name=names[key]+" aktiv")
            self.process_plot.plot(times, metrics["recovered_fraction"],
                                   pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                                   name=names[key]+" erholt")
            self.rate_plot.plot(times, metrics["new_activation_fraction"],
                                pen=pg.mkPen(color, width=2), name=names[key])
        self.process_cursor = pg.InfiniteLine(pos=times[0], pen=pg.mkPen("#edbd65", width=2))
        self.rate_cursor = pg.InfiniteLine(pos=times[0], pen=pg.mkPen("#edbd65", width=2))
        self.process_plot.addItem(self.process_cursor)
        self.rate_plot.addItem(self.rate_cursor)
        self._update_process_cursor()

    def _update_process_cursor(self):
        if self.result is None or self._process_summary is None:
            return
        index = self.slider.value()
        instant = float(self.result["times_ms"][index])
        if hasattr(self, "process_cursor"):
            self.process_cursor.setValue(instant)
            self.rate_cursor.setValue(instant)
        names = {"ventricle": "Kammer", "atrium": "Vorhof", "av_junction": "AV-Zone",
                 "conduction_cable": "Leitungskabel"}
        phrases = [f"{names[key]}: {100*entry['active_fraction'][index]:.0f} % aktiv, "
                   f"{100*entry['recovered_fraction'][index]:.0f} % erholt"
                   for key, entry in self._process_summary.items()
                   if self.process_group.currentData() in ("all", key)]
        self.process_readout.setText(f"{instant:.0f} ms  ·  " + "   |   ".join(phrases))

    def _seek_plot(self, plot, event):
        if self.result is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if not plot.plotItem.vb.sceneBoundingRect().contains(event.scenePos()):
            return
        t = plot.plotItem.vb.mapSceneToView(event.scenePos()).x()
        index = int(np.clip(np.searchsorted(self.result["times_ms"], t), 0, self.slider.maximum()))
        self.slider.setValue(index)
        self._play_time_ms = float(self.result["times_ms"][index])
        self._play_clock = time.monotonic()

    def _seek_from_process(self, event):
        self._seek_plot(self.process_plot, event)

    def _seek_from_rate(self, event):
        self._seek_plot(self.rate_plot, event)

    def _motion_changed(self, reduced):
        if reduced and self.play_timer.isActive():
            self.play_timer.stop()
            self.play.setText("Abspielen")

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play.setText("Abspielen")
        elif self.result is not None:
            if current_motion_policy().reduced:
                self.status.setText("Reduzierte Bewegung aktiv: Zeitleiste manuell bewegen.")
                return
            self._play_time_ms = float(self.result["times_ms"][self.slider.value()])
            self._play_clock = time.monotonic()
            self.play_timer.start()
            self.play.setText("Pause")

    def _advance(self):
        if self.result is None or self._play_clock is None:
            return
        now = time.monotonic()
        self._play_time_ms += (now-self._play_clock)*1000*float(self.speed.currentData())
        self._play_clock = now
        times = self.result["times_ms"]
        if self._play_time_ms > times[-1]:
            if self.loop_play.isChecked():
                self._play_time_ms = float(times[0]) + (self._play_time_ms-times[0]) % (times[-1]-times[0])
            else:
                self.slider.setValue(self.slider.maximum())
                self.toggle_play()
                return
        self.slider.setValue(int(np.clip(np.searchsorted(times, self._play_time_ms, side="right")-1,
                                          0, self.slider.maximum())))

    def _place_electrode(self, event):
        if self.result is None or event.button() != Qt.MouseButton.LeftButton:
            return
        p = self.slice_plot.plotItem.vb.mapSceneToView(event.scenePos())
        a, b, n = ((0, 2, 1), (0, 1, 2), (1, 2, 0))[self.plane.currentIndex()]
        if not self.placing.isChecked():
            section, _ = self._slice_geometry()
            if len(section.xyz):
                distance = (section.xyz[:, a] - p.x()) ** 2 + (section.xyz[:, b] - p.y()) ** 2
                nearest = int(np.argmin(distance))
                if distance[nearest] > 12**2:
                    return
                self.selected_node = int(section.start[nearest] if section.weight_end[nearest] < .5
                                         else section.end[nearest])
                self.draw_trace()
                self.status.setText(self.cell_label.text())
            return
        row = self.electrode_table.currentRow()
        if row < 0:
            self.status.setText("Zuerst eine Elektrode in der Tabelle wählen")
            return
        for index, value in ((a, p.x()), (b, p.y()), (n, self.slice_mm.value())):
            self.electrode_table.item(row, index + 1).setText(f"{value:.2f}")
        self.status.setText("Elektrode im Entwurf versetzt · neues Experiment für neue Ableitungen starten")

    def show_3d(self):
        if self.result is None:
            self.status.setText("Zuerst ein Experiment berechnen oder öffnen")
            return
        if not self.result["metadata"].get("geometry_kind", "").startswith("generic_anatomical_graph"):
            self.status.setText("Für dieses Rechennetz liegt keine passende anatomische Oberfläche vor; der exakte Graphschnitt bleibt verfügbar.")
            return
        if self._gl is None:
            if QApplication.platformName().lower() == "offscreen":
                self._gl_placeholder.setText("3D-OpenGL ist im Offscreen-Testmodus nicht verfügbar. Der 2D-Schnitt bleibt bedienbar.")
                return
            try:
                import pyqtgraph.opengl as gl
                self._gl = gl.GLViewWidget()
                self._gl.setBackgroundColor("#101a27")
                self._gl.setCameraPosition(distance=380, elevation=15, azimuth=-85)
                self._gl_host_layout.addWidget(self._gl, 1)
                self._gl_placeholder.hide()
                self._build_gl()
                self.gl_button.setText("3D aktiv")
                self.gl_button.setEnabled(False)
            except Exception as exc:  # OpenGL driver boundary: keep the analytic 2D view available.
                LOGGER.exception("Simulation 3D renderer failed")
                if self._gl is not None:
                    self._gl.deleteLater()
                    self._gl = None
                self._gl_placeholder.setText("3D-OpenGL konnte nicht geöffnet werden: " + str(exc))
                self.status.setText("3D-Renderer nicht verfügbar; 2D-Schnitt und Prozesse sind weiter nutzbar.")
                return
        self.views.setCurrentIndex(0)

    def _update_gl_colors(self, *_):
        if self._gl is not None:
            colors = self._colors()
            colors[:, 3] = self.opacity.value() / 100.
            for item, indices in self._gl_items:
                update_mesh_colors(item, colors[indices])

    def _build_gl(self):
        import pyqtgraph.opengl as gl
        from scipy.spatial import cKDTree
        from .heart3d.model import default_model
        self._gl.clear()
        self._gl_items.clear()
        self._gl_plane = None
        model = default_model()
        tree = cKDTree(self.result["xyz"])
        colors = self._colors()
        for name in ("wall", "ra_wall", "la_wall"):
            part = model.parts[name]
            indices = tree.query(part.vertices)[1]
            item = gl.GLMeshItem(vertexes=part.vertices, faces=part.faces,
                                 vertexColors=colors[indices], smooth=True, shader="shaded", drawEdges=False,
                                 glOptions="translucent")
            self._gl.addItem(item)
            self._gl_items.append((item, indices))
        edges = self.result["edges"]
        cells = self.result["cells"]
        xyz = self.result["xyz"]
        cables = edges[(cells[edges[:, 0]] == 5) & (cells[edges[:, 1]] == 5)]
        vertices, faces, offset = [], [], 0
        for a, b in cables:
            v, f = tube_mesh(xyz[a], xyz[b])
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        if vertices:
            self._gl.addItem(gl.GLMeshItem(vertexes=np.vstack(vertices), faces=np.vstack(faces),
                                          color=(.95, .75, .35, 1.), shader="shaded", smooth=True))
        electrodes = np.asarray([e["xyz_mm"] for e in self.result["metadata"]["experiment"]["electrodes"]])
        self._gl.addItem(gl.GLScatterPlotItem(pos=electrodes, size=8, color=(.9, .9, .95, 1.)))
        vertices = self._plane_vertices()
        self._gl_plane = gl.GLMeshItem(vertexes=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]),
                                     color=(.22, .72, .95, .17), shader="balloon",
                                     drawEdges=True, glOptions="translucent")
        self._gl.addItem(self._gl_plane)
        self._update_gl_colors()

    def _plane_vertices(self):
        xyz = self.result["xyz"]
        a, b, normal = ((0, 2, 1), (0, 1, 2), (1, 2, 0))[self.plane.currentIndex()]
        bounds = ((float(xyz[:, a].min()-8), float(xyz[:, a].max()+8)),
                  (float(xyz[:, b].min()-8), float(xyz[:, b].max()+8)))
        out = np.zeros((4, 3), float)
        out[:, a] = [bounds[0][0], bounds[0][1], bounds[0][1], bounds[0][0]]
        out[:, b] = [bounds[1][0], bounds[1][0], bounds[1][1], bounds[1][1]]
        out[:, normal] = self.slice_mm.value()
        return out

    def _update_gl_plane(self):
        if self._gl_plane is not None and self.result is not None:
            self._gl_plane.setMeshData(vertexes=self._plane_vertices(),
                                       faces=np.array([[0, 1, 2], [0, 2, 3]]))

    def send_to_lab(self):
        if self.result_path is None:
            return
        from PyQt6.QtWidgets import QInputDialog
        gain, ok = QInputDialog.getDouble(self, "Explizite illustrative Amplitudenskala",
                                         "mV pro a.u. (nicht physiologisch kalibriert):", 1., .000001, 1e6, 6)
        if not ok:
            return
        try:
            target = self.result_path.parent / ("SYNTHETISCH-" + uuid.uuid4().hex[:8] + ".npz")
            export_lab_signal(self.result_path, target, gain)
            self.labSignalReady.emit(str(target))
            self.status.setText("Synthetisches EKG exportiert: " + str(target))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Simulationsexport", str(exc))

    def save_view(self):
        if self.result is None:
            self.status.setText("Zuerst ein Ergebnis laden, dann die Ansicht speichern.")
            return
        target, _ = QFileDialog.getSaveFileName(self, "Forschungsansicht als PNG",
                                                str(self.root / "simulation_view.png"), "PNG (*.png)")
        if not target:
            return
        if not self.views.currentWidget().grab().save(target, "PNG"):
            QMessageBox.warning(self, "Simulation", "PNG konnte nicht gespeichert werden.")
            return
        self.status.setText("Forschungsansicht gespeichert: " + target)
