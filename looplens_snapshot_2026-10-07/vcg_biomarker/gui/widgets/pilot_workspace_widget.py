"""Isolated real-pilot inspector. Never instantiates a production JobStore."""
from __future__ import annotations

from pathlib import Path
import traceback
import logging
import json
import os
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QThread, pyqtSignal, QSize
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QTabWidget, QWidget, QTableWidget, QTableWidgetItem, QSlider, QSizePolicy)
from PyQt6.QtCore import Qt

from ...mimic.pilot_viewmodel import PilotViewModel
from ..lab_process_monitor import live_tree_rss, stop_owned_lab_tree


class CompactPlotWidget(pg.PlotWidget):
    """PlotWidget's large default minimum hint must not force a tall dialog."""
    def minimumSizeHint(self):
        return QSize(100, 60)

    def sizeHint(self):
        return QSize(500, 180)


class LabTask(QThread):
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model: PilotViewModel, study_id: int, method: str):
        super().__init__()
        self.model, self.study_id, self.method = model, study_id, method

    def run(self):
        try:
            import psutil
            token = uuid.uuid4().hex
            relative = f"scratch/lab-{token}.json"
            result = self.model.root / "overlay" / relative
            result.parent.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = "1"
            command = [sys.executable, "-m", "vcg_biomarker.mimic.pilot_lab_worker", "--workspace", str(self.model.root),
                       "--study-id", str(self.study_id), "--method", self.method, "--result", relative]
            flags = subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS if os.name == "nt" else 0
            with result.with_suffix(".log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                           creationflags=flags, cwd=Path(__file__).resolve().parents[3])
                child = None
                try:
                    child = psutil.Process(process.pid)
                    if os.name != "nt": child.nice(10)
                    started = time.monotonic()
                    while process.poll() is None:
                        try:
                            rss = live_tree_rss(child)
                        except psutil.Error:
                            # Stop our owned worker before reporting an
                            # unavailable resource check to the GUI.
                            stop_owned_lab_tree(child)
                            process.wait(timeout=5)
                            raise
                        if rss > 2*1024**3 or time.monotonic()-started > 180:
                            stop_owned_lab_tree(child)
                            process.wait(timeout=5)
                            raise RuntimeError("Isolierte Laboranalyse abgebrochen: 2-GiB-RSS- oder 180-s-Limit überschritten.")
                        time.sleep(.1)
                    code = process.wait()
                finally:
                    # No Popen context manager: its implicit unbounded wait
                    # could hide failed monitoring/cleanup from the GUI.
                    if process.poll() is None:
                        if child is not None:
                            stop_owned_lab_tree(child)
                        else:
                            process.kill()
                        process.wait(timeout=5)
            if code:
                raise RuntimeError("Laborprozess fehlgeschlagen: " + result.with_suffix(".log").read_text(encoding="utf-8")[-4000:])
            self.result_ready.emit(json.loads(result.read_text(encoding="utf-8")))
        except Exception:
            self.failed.emit(traceback.format_exc())


class PilotWorkspaceDialog(QDialog):
    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.model = PilotViewModel(root)
        self.task = None
        self.xyz = np.empty((0, 3))
        self.array_data = {}
        self.gl = None
        self.setWindowTitle(f"Pilot-Testarbeitsbereich · {self.model.run_id}")
        self.resize(1280, 820)
        layout = QVBoxLayout(self)
        warning = QLabel("Unveränderte Pilot-Originale · Hypothesen, keine Diagnosen · "
                         "Neuberechnungen nur im Overlay · Produktionslauf und Locked Test bleiben getrennt")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        controls = QHBoxLayout()
        self.size = QComboBox(); self.size.addItems(["10000", "1000"])
        self.endpoint = QComboBox(); self.endpoint.setMinimumWidth(250)
        self.marker = QComboBox(); self.marker.setMinimumWidth(330)
        for item in (self.size, self.endpoint, self.marker): controls.addWidget(item)
        layout.addLayout(controls)
        self.tabs = QTabWidget(); layout.addWidget(self.tabs)
        results = QWidget(); results.setObjectName("pilot_results"); results_layout = QVBoxLayout(results)
        self.summary = QLabel(); self.summary.setWordWrap(True); results_layout.addWidget(self.summary)
        self.plot = CompactPlotWidget(); self.plot.setLabel("bottom", "Markerwert")
        self.plot.setLabel("left", "Dichte"); self.plot.addLegend()
        results_layout.addWidget(self.plot)
        self.availability_plot = CompactPlotWidget(); self.availability_plot.setMaximumHeight(180)
        results_layout.addWidget(self.availability_plot)
        self.result_table = QTableWidget(); self.result_table.setMaximumHeight(85); results_layout.addWidget(self.result_table)
        export = QPushButton("Auswahl als CSV und Diagramm als PNG/SVG exportieren")
        export.clicked.connect(self.export); results_layout.addWidget(export)
        self.tabs.addTab(results, "Ergebnisse & Messverfügbarkeit")
        lab = QWidget(); lab.setObjectName("pilot_lab"); lab_layout = QVBoxLayout(lab)
        bar = QHBoxLayout(); self.case = QComboBox()
        from ...core.vcg_transform import VCGTransformRegistry
        self.method = QComboBox(); self.method.addItems(VCGTransformRegistry.available())
        self.method.setToolTip("Transformation der VCG-Marker. Die separate SVD-Ansicht verändert diese Marker nicht.")
        self.coordinates = QComboBox()
        self.coordinates.addItems(["Anatomische VCG-Achsen", "SVD-Komponenten (nicht anatomisch)"])
        self.coordinates.setMaximumWidth(260)
        self.coordinates.currentIndexChanged.connect(self.draw_loop)
        self.segment = QComboBox(); self.segment.addItems(["Gesamtsignal", "P", "QRS", "T"])
        self.segment.currentIndexChanged.connect(self.draw_loop)
        self.plane = QComboBox()
        for title, axes in (("Frontal (X–Y)", (0, 1)), ("Horizontal (X–Z)", (0, 2)), ("Sagittal (Y–Z)", (1, 2))):
            self.plane.addItem(title, axes)
        self.plane.setToolTip("Unveränderte Transformationsachsen; Drehrichtung hängt von Achsenreihenfolge und Betrachtungsrichtung ab. Keine automatische klinische Befundung.")
        self.plane.currentIndexChanged.connect(self.draw_loop)
        self.analyze = QPushButton("Im isolierten Labor neu berechnen")
        self.analyze.clicked.connect(self.analyze_case)
        for item in (self.case, self.method, self.coordinates, self.segment, self.plane, self.analyze): bar.addWidget(item)
        lab_layout.addLayout(bar)
        self.lab_status = QLabel("Aktuelle Engine, keine Übernahme von Pilotkriterien.")
        self.lab_status.setMaximumHeight(78)
        self.lab_status.setWordWrap(True); lab_layout.addWidget(self.lab_status)
        self.projection_status = QLabel("")
        self.projection_status.setMaximumHeight(52)
        self.projection_status.setWordWrap(True); lab_layout.addWidget(self.projection_status)
        self.ecg_plot = CompactPlotWidget(); self.ecg_plot.setLabel("bottom", "Zeit", units="s")
        lab_layout.addWidget(self.ecg_plot)
        self.cursor = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen("#ee9b00", width=2))
        self.cursor.sigPositionChanged.connect(self.cursor_moved)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.valueChanged.connect(self.time_changed)
        lab_layout.addWidget(self.time_slider)
        loop_layout = QHBoxLayout()
        self.loop_plot = CompactPlotWidget(); self.loop_plot.setLabel("bottom", "VCG X", units="mV")
        self.loop_plot.setLabel("left", "VCG Y", units="mV"); self.loop_plot.setAspectLocked(True)
        # The analysis arrays are already in mV. Automatic SI prefixes would
        # label small amplitudes as "mmV", which is needlessly ambiguous.
        self.loop_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.loop_plot.getAxis("left").enableAutoSIPrefix(False)
        loop_layout.addWidget(self.loop_plot)
        self.loop_cursor = pg.ScatterPlotItem(size=12, brush="#ee9b00")
        try:
            from PyQt6.QtWidgets import QApplication
            if QApplication.platformName() in {"offscreen", "minimal"}:
                raise RuntimeError("This Qt platform cannot provide an auditable 3D OpenGL view")
            import pyqtgraph.opengl as gl
            from .loop_camera import loop_camera_distance
            class FittedLoopView(gl.GLViewWidget):
                loop_radius = None

                def fit_loop(self, radius=None):
                    if radius is not None:
                        self.loop_radius = radius
                    if self.loop_radius is not None and self.width() > 0 and self.height() > 0:
                        self.setCameraPosition(distance=loop_camera_distance(
                            self.loop_radius, self.width(), self.height(), self.opts["fov"]))

                def resizeEvent(self, event):
                    super().resizeEvent(event)
                    self.fit_loop()
            self.gl = gl
            self.loop3d = FittedLoopView()
            self.loop3d.setMinimumHeight(180)
            loop_layout.addWidget(self.loop3d)
        except (ImportError, RuntimeError):
            logging.getLogger(__name__).warning("3D unavailable; explicit 2D pilot fallback", exc_info=True)
            fallback = QLabel("3D in dieser Umgebung nicht verfügbar.\n2D-Projektion links; kein bestandener 3D-Darstellungstest.")
            fallback.setWordWrap(True)
            loop_layout.addWidget(fallback)
        lab_layout.addLayout(loop_layout)
        time_key = QHBoxLayout()
        self.loop_time_label = QLabel("3D-Zeitfarbe: noch keine Loop-Daten" if self.gl else "3D-Zeitfarbe in dieser Umgebung nicht verfügbar")
        time_key.addWidget(self.loop_time_label)
        self.loop_time_gradient = QLabel()
        self.loop_time_gradient.setFixedHeight(12)
        self.loop_time_gradient.setMinimumWidth(120)
        cmap = pg.colormap.get("viridis")
        stops = ", ".join(
            f"stop:{value:.3f} {cmap.map(value, mode='qcolor').name()}"
            for value in np.linspace(0, 1, 17)
        )
        self.loop_time_gradient.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops});"
        )
        self.loop_time_gradient.setVisible(False)
        time_key.addWidget(self.loop_time_gradient, 1)
        lab_layout.addLayout(time_key)
        self.timeline_plot = CompactPlotWidget(); self.timeline_plot.setMaximumHeight(160)
        self.timeline_plot.setLabel("bottom", "Stunden relativ zum ECG; positive Zeiten sind spätere Referenzen")
        self.tabs.addTab(lab, "ECG-Labor & klinischer Zeitstrahl")
        context_page = QWidget(); context_page.setObjectName("pilot_context"); context_layout = QVBoxLayout(context_page)
        self.context_title = QLabel("Klinischer Kontext des im Labor ausgewählten ECGs. Zeitangaben sind keine Synchronisationsbestätigung.")
        self.context_title.setWordWrap(True); context_layout.addWidget(self.context_title)
        context_layout.addWidget(self.timeline_plot)
        self.context_table = QTableWidget(); context_layout.addWidget(self.context_table)
        self.tabs.addTab(context_page, "Echo / Labor · Zeitbezug")
        lead_page = QWidget(); lead_layout = QVBoxLayout(lead_page)
        lead_note = QLabel("Direkte Einzelableitungs-Messwerte aus dem diagnostischen Signal. "
            "Gemeinsame 12-Kanal-Fiduzialpunkte; PR-Basislinie oder prä-QRS-Fallback. "
            "Keine Pathologierangfolge und keine Freigabe zur Monitoring-Ableitungswahl. "
            "Fehlende Werte bedeuten Abstention, nicht Normalbefund.")
        lead_note.setWordWrap(True); lead_layout.addWidget(lead_note)
        self.lead_table = QTableWidget(); lead_layout.addWidget(self.lead_table, 3)
        comparison_note = QLabel("QRS-T-Methodenvergleich auf demselben Morphologiesignal: gleiche PR-Baseline und QRS/J–T-Grenzen. "
            "Area = Winkel der vorzeichenbehafteten Integrale; Integral = gewichtetes Mittel der paarweisen Winkel. "
            "SVD hier PR-zentriert; keine Austauschbarkeit oder klinischen Grenzwerte belegt.")
        comparison_note.setWordWrap(True); lead_layout.addWidget(comparison_note)
        self.method_comparison_table = QTableWidget(); lead_layout.addWidget(self.method_comparison_table, 2)
        self.method_comparison_table.horizontalHeader().setStretchLastSection(True)
        from .ndat_widget import NdatWidget
        self.ndat_view = NdatWidget()
        self.tabs.addTab(self.ndat_view, "NDAT · Flankenzeiten")
        self.ndat_view.setObjectName("pilot_ndat")
        from .segment_angle_widget import SegmentAngleWidget
        self.segment_angle_view = SegmentAngleWidget()
        self.tabs.addTab(self.segment_angle_view, "QRS–T · Zeitmatrix")
        self.segment_angle_view.setObjectName("pilot_segment_matrix")
        from .reviewer_export_widget import ReviewerExportWidget
        self.reviewer_export = ReviewerExportWidget(self.model.root)
        self.tabs.addTab(self.reviewer_export, "Expertenreview · Export")
        from .reviewer_import_widget import ReviewerImportWidget
        self.reviewer_import = ReviewerImportWidget(self.model.root)
        self.tabs.addTab(self.reviewer_import, "Expertenreview · Antworten")
        self.tabs.addTab(lead_page, "Einzelableitungen · Messwerte")
        lead_page.setObjectName("pilot_lead_measurements")
        for plot in (self.plot, self.ecg_plot, self.loop_plot, self.timeline_plot, self.availability_plot):
            plot.setMinimumSize(0, 60)
            plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if self.gl:
            self.loop3d.setMinimumSize(160, 90)
            self.loop3d.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.timeline_plot.setFixedHeight(220)
        context_layout.setStretch(1, 0); context_layout.setStretch(2, 1)
        results_layout.setStretch(1, 3)
        results_layout.setStretch(2, 1)
        lab_layout.setStretch(3, 3)
        lab_layout.setStretch(5, 2)
        cases = self.model.cases()
        for study in cases["study_id"]: self.case.addItem(f"EKG {study}", int(study))
        self.case.currentIndexChanged.connect(self.show_case)
        self.size.currentIndexChanged.connect(self.load_results)
        self.endpoint.currentIndexChanged.connect(self.select_endpoint)
        self.marker.currentIndexChanged.connect(self.show_result)
        self.load_results(); self.show_case()

    def load_results(self):
        self.frame = self.model.results(int(self.size.currentText()))
        self.endpoint.blockSignals(True); self.endpoint.clear()
        self.endpoint.addItems(sorted(self.frame["endpoint_id"].unique()))
        idx = self.endpoint.findText("reduced_lvef_strict")
        if idx >= 0: self.endpoint.setCurrentIndex(idx)
        self.endpoint.blockSignals(False); self.select_endpoint()

    def select_endpoint(self):
        self.scoped = self.frame.loc[self.frame["endpoint_id"].eq(self.endpoint.currentText())]
        self.marker.blockSignals(True); self.marker.clear()
        self.marker.addItems(self.scoped.sort_values("roc_auc", ascending=False)["feature_id"].tolist())
        idx = self.marker.findText("qrs_t_area_angle_direct_8lead_deg")
        if idx >= 0: self.marker.setCurrentIndex(idx)
        self.marker.blockSignals(False); self.show_result()

    def show_result(self):
        self._clear_result_view()
        try:
            self._draw_result_view()
        except Exception as exc:
            logging.getLogger(__name__).exception("Pilot result display failed")
            self._clear_result_view()
            self.summary.setText(f"Pilotergebnis nicht darstellbar: {type(exc).__name__}: {exc}")

    def _clear_result_view(self):
        self.plot.clear(); self.availability_plot.clear()
        self.availability_plot.getAxis("bottom").setTicks([])
        self.result_table.setRowCount(0)
        self.export_frame = pd.DataFrame()
        self.summary.setText("Keine darstellbaren Ergebnisse fuer diese Auswahl; Export nicht verfuegbar.")

    def _draw_result_view(self):
        selected = self.scoped.loc[self.scoped["feature_id"].eq(self.marker.currentText())]
        if selected.empty: return
        row = selected.iloc[0]
        self.summary.setText(f"Pilot-Hypothese · {row['endpoint_id']} · AUC {row['roc_auc']:.3f} · "
            f"Fälle/Kontrollen: {row.get('n_positive', '?')}/{row.get('n_negative', '?')}\n"
            "Keine adjustierte Bestätigung, keine klinische Kalibration, kein belegter Zusatznutzen. "
            "Grafik unten: gesamte ausgewählte Development-Stichprobe; AUC oben: Validation.")
        counts, paired = self.model.availability(int(self.size.currentText()), self.endpoint.currentText(), self.marker.currentText())
        measured = paired.loc[paired.measurement_eligible]
        numeric = pd.to_numeric(measured[self.marker.currentText()], errors="coerce")
        edges = np.histogram_bin_edges(numeric[np.isfinite(numeric)], bins=30)
        for label, color, title in [("negative", "#0072B2", "Referenz negativ"), ("positive", "#D55E00", "Referenz positiv")]:
            values = pd.to_numeric(measured.loc[measured.label_state.eq(label), self.marker.currentText()], errors="coerce")
            values = values[np.isfinite(values)]
            if len(values):
                y, _ = np.histogram(values, bins=edges, density=True)
                self.plot.plot(edges, y, stepMode="center", fillLevel=0, brush=pg.mkBrush(color + "50"), pen=color, name=title)
        self.availability_plot.addItem(pg.BarGraphItem(x=np.arange(4), height=counts.n.to_numpy(), width=.65, brush="#168aad"))
        self.availability_plot.getAxis("bottom").setTicks([list(enumerate(counts.step))])
        self.availability_plot.setLabel("left", "ECGs")
        # Leave room for device-scaled labels in the short availability panel.
        self.availability_plot.setYRange(0, max(1, counts.n.max()*1.4), padding=0)
        for index, number in enumerate(counts.n):
            text = pg.TextItem(str(number), anchor=(.5, 1), color="#657b83")
            text.setPos(index, number); self.availability_plot.addItem(text)
        self.export_frame = selected
        columns = [x for x in ("feature_id", "reference_kind", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "n_positive", "n_negative", "evidence_grade_de",
            "independent_modality_observed_n", "clock_alignment_confirmed_n", "timing_evidence_note") if x in selected]
        self.result_table.setColumnCount(len(columns)); self.result_table.setHorizontalHeaderLabels(columns)
        self.result_table.setRowCount(1)
        for index, name in enumerate(columns):
            value = row.get(name)
            self.result_table.setItem(0, index, QTableWidgetItem("nicht berechnet" if pd.isna(value) else str(value)))
        self.result_table.resizeColumnsToContents()

    def show_case(self):
        self._clear_case_views()
        if self.case.currentData() is None:
            return
        try:
            self._load_case_views()
        except Exception as exc:
            logging.getLogger(__name__).exception("Pilot case display failed")
            self._clear_case_views()
            self.lab_status.setText(f"Pilot-EKG nicht darstellbar: {type(exc).__name__}: {exc}")

    def _clear_case_views(self):
        self.ecg_plot.clear()
        self.loop_plot.clear()
        self.timeline_plot.clear()
        self.timeline_plot.getAxis("left").setTicks([])
        self.timeline_plot.setToolTip("")
        self.context_table.setRowCount(0)
        self.context_title.setText("Kein klinischer Kontext fuer die aktuelle Auswahl geladen.")
        self.method_comparison_table.setRowCount(0)
        self.lead_table.setRowCount(0)
        self.ndat_view.set_result({})
        self.segment_angle_view.set_result({})
        self.svd_projection_metadata = {}
        self.xyz = np.empty((0, 3))
        self.array_data = {}
        self.time_slider.setRange(0, 0)
        self.loop_time_gradient.setVisible(False)
        self.loop_time_label.setText("Keine Loop-Daten fuer die aktuelle Auswahl")
        self.projection_status.clear()
        self.lab_status.setText("Keine Analyse fuer die aktuelle Auswahl geladen.")
        self.lab_status.setToolTip("")
        if self.gl:
            self.loop3d.clear()

    def _load_case_views(self):
        from ...core.external_loader import load_external_ecg_result
        if self.case.currentData() is None: return
        study = int(self.case.currentData())
        raw = load_external_ecg_result(self.model.waveform(study))
        signal = raw.signal_millivolts; times = np.arange(len(signal)) / raw.sampling_rate_hz
        self.ecg_plot.clear(); self.loop_plot.clear()
        self.xyz = np.empty((0, 3)); self.array_data = {}
        self.loop_time_label.setText("3D-Zeitfarbe: noch keine Loop-Daten" if self.gl else "3D-Zeitfarbe in dieser Umgebung nicht verfügbar")
        self.loop_time_gradient.setVisible(False)
        self.lead_table.setRowCount(0)
        if self.gl: self.loop3d.clear()
        self.lab_status.setText("Rohsignal — für synchronisierte Loops bitte im Overlay analysieren. Keine klinische Freigabe.")
        self.fs = float(raw.sampling_rate_hz)
        self.time_slider.setRange(0, max(0, len(signal)-1))
        for lead in range(min(12, signal.shape[1])):
            self.ecg_plot.plot(times, signal[:, lead] + (11-lead)*2, pen=self.signal_pen())
        self.ecg_plot.getAxis("left").setTicks([[(22-2*i, name) for i, name in enumerate(("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"))]])
        self.draw_ecg_calibration()
        self.ecg_plot.addItem(self.cursor)
        timeline_all = self.model.timeline(study)
        timeline = timeline_all.loc[timeline_all.hours_from_ecg.abs().nsmallest(6).index].sort_values("hours_from_ecg")
        self.timeline_plot.clear()
        self.timeline_plot.plot(timeline.hours_from_ecg.to_numpy(), np.arange(len(timeline)), pen=None, symbol="o", symbolBrush="#d1495b")
        self.timeline_plot.getAxis("left").setTicks([[(i, str(row.event)[:32]) for i, row in enumerate(timeline.itertuples())]])
        self.timeline_plot.setToolTip("Nächste 6 Zeitpunkte; vollständiger Kontext im Export.\n" + timeline_all.to_string(index=False))
        self.timeline_plot.setLabel("bottom", "Stunden zum ECG (Geräteuhr unsicher); nächste 6 Referenzzeitpunkte")
        self.context_title.setText(f"EKG {study} · {len(timeline_all)-1} verfügbare Referenzzeitpunkte. "
            "Positive Zeiten: nach dem ECG; nicht als Prädiktoren verwenden. Geräteuhr nicht extern bestätigt.")
        self.context_table.setColumnCount(3); self.context_table.setHorizontalHeaderLabels(["Ereignis", "Stunden zum ECG", "Wert (Einheit siehe Datenwörterbuch)"])
        self.context_table.setRowCount(len(timeline_all))
        for index, row in enumerate(timeline_all.itertuples()):
            for column, value in enumerate((row.event, f"{row.hours_from_ecg:.2f}", row.value)):
                self.context_table.setItem(index, column, QTableWidgetItem(str(value)))
        self.context_table.resizeColumnsToContents()
        self.timeline_plot.addLine(x=0, pen=pg.mkPen("#778388"))

    def analyze_case(self):
        if self.task is not None and self.task.isRunning(): return
        import psutil
        if psutil.virtual_memory().available < 8 * 1024**3:
            self.lab_status.setText("Labor wartet: mindestens 8 GiB freier Systemspeicher benötigt (6 GiB Reserve + 2 GiB Labor).")
            return
        for control in (self.analyze, self.case, self.method): control.setEnabled(False)
        self.lab_status.setText("Einzelne Laboranalyse läuft im Overlay; Pilotresultate bleiben unverändert.")
        self.task = LabTask(self.model, int(self.case.currentData()), self.method.currentText())
        self.task.result_ready.connect(self.lab_ready)
        self.task.failed.connect(lambda text: self.lab_status.setText(text))
        self.task.finished.connect(self.lab_finished)
        self.task.start(QThread.Priority.LowPriority)

    def lab_finished(self):
        for control in (self.analyze, self.case, self.method): control.setEnabled(True)

    def lab_ready(self, result):
        if result["study_id"] != self.case.currentData():
            self.lab_status.setText("Analyse gehört zu einem anderen ECG; Anzeige verworfen.")
            return
        payload = result["payload"]
        self.svd_projection_metadata = payload.get("svd_component_projection", {})
        comparisons = payload.get("qrs_t_method_comparison", [])
        self.ndat_view.set_result(payload.get("ndat", {}))
        self.segment_angle_view.set_result(payload.get("qrs_t_segment_matrix", {}))
        columns = ("method", "area_angle_deg", "integral_angle_deg", "status", "reason")
        self.method_comparison_table.setColumnCount(len(columns))
        self.method_comparison_table.setHorizontalHeaderLabels(["Methode", "Area [°]", "Integral [°]", "Status", "Abstention"])
        self.method_comparison_table.setRowCount(len(comparisons))
        for row_index, comparison in enumerate(comparisons):
            for column_index, name in enumerate(columns):
                value = comparison.get(name)
                text = "nicht messbar" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.method_comparison_table.setItem(row_index, column_index, item)
        self.method_comparison_table.resizeColumnsToContents()
        from ...core.data_loader import LEAD_NAMES_12
        from ...mimic.lead_measurements import LEAD_MEASUREMENT_SPEC, measurement_column
        measurements = payload.get("lead_measurements", {})
        self.lead_table.setColumnCount(1 + len(LEAD_MEASUREMENT_SPEC))
        self.lead_table.setHorizontalHeaderLabels(["Ableitung", "R [mV]", "S-Tiefe [mV]", "P+ [mV]",
            "P− [mV]", "P terminal− [mV·s]", "J [mV]", "T max [mV]", "T min [mV]"])
        self.lead_table.setRowCount(len(LEAD_NAMES_12))
        for row_index, lead in enumerate(LEAD_NAMES_12):
            self.lead_table.setItem(row_index, 0, QTableWidgetItem(lead))
            for column, spec in enumerate(LEAD_MEASUREMENT_SPEC, start=1):
                value = measurements.get(measurement_column(lead, *spec))
                text = f"{value:.4g}" if isinstance(value, (int, float)) and np.isfinite(value) else "nicht messbar"
                self.lead_table.setItem(row_index, column, QTableWidgetItem(text))
        self.lead_table.resizeColumnsToContents()
        with np.load(result["arrays"], allow_pickle=False) as saved:
            self.array_data = {name: saved[name] for name in saved.files}
        self.xyz = self.array_data.get("record_vcg_xyz", np.empty((0, 3)))
        self.fs = float(self.array_data.get("sampling_rate_hz", 500))
        signal = self.array_data["morphology_ecg"]
        self.ecg_plot.clear()
        for lead in range(min(12, signal.shape[1])):
            self.ecg_plot.plot(np.arange(len(signal))/self.fs, signal[:, lead]+(11-lead)*2,
                               pen=self.signal_pen())
        self.draw_ecg_calibration()
        self.ecg_plot.addItem(self.cursor)
        self.time_slider.setRange(0, max(0, len(signal)-1))
        features = payload.get("feature_values", {})
        angles = []
        for kind in ("area", "integral"):
            value = features.get(f"qrs_t_{kind}_angle_direct_8lead_deg")
            display = f"{value:.1f}°" if isinstance(value, (int, float)) and np.isfinite(value) else "nicht messbar (Abstention)"
            angles.append(f"Direct-8 {kind}: {display}")
        delineation = payload.get("delineation", {})
        quality = payload.get("signal_quality", {})
        self.lab_status.setToolTip(str(quality) + "\n" + str(delineation))
        self.lab_status.setText(f"{result['method']} · gefiltertes Analysesignal · " + " | ".join(angles)
            + "\nForschungsansicht, keine Diagnose. Review erforderlich: " + str(delineation.get("review_required", True))
            + " · " + "; ".join(map(str, delineation.get("review_reasons", [])))[:180]
            + " · Details im Tooltip")
        self.draw_loop()

    def draw_ecg_calibration(self):
        """Reference pulse in data coordinates; remains correct during zoom."""
        self.ecg_plot.plot(
            [-0.35, -0.30, -0.30, -0.10, -0.10, -0.05],
            [22, 22, 23, 23, 22, 22],
            pen=pg.mkPen("#e9a400", width=2),
        )
        # Reserve layout space for the label instead of tying its pixel height
        # to a data-coordinate anchor that can be clipped by autorange/DPI.
        # The calibration pulse itself remains in real seconds and mV.
        self.ecg_plot.setTitle("Referenzpuls: 1 mV / 200 ms")

    def draw_loop(self):
        self.loop_plot.clear()
        self.loop_time_label.setText("3D-Zeitfarbe: noch keine Loop-Daten" if self.gl else "3D-Zeitfarbe in dieser Umgebung nicht verfügbar")
        self.loop_time_gradient.setVisible(False)
        if self.gl: self.loop3d.clear()
        component_view = self.coordinates.currentIndex() == 1
        self.xyz = getattr(self, "array_data", {}).get(
            "record_svd_components" if component_view else "record_vcg_xyz", np.empty((0, 3)))
        labels = ("C1", "C2", "C3") if component_view else ("X", "Y", "Z")
        self.plane.blockSignals(True)
        for index, title in enumerate(("C1–C2", "C1–C3", "C2–C3") if component_view else
                ("Frontal (X–Y)", "Horizontal (X–Z)", "Sagittal (Y–Z)")):
            self.plane.setItemText(index, title)
        self.plane.blockSignals(False)
        metadata = getattr(self, "svd_projection_metadata", {})
        energy = metadata.get("retained_energy_fraction")
        energy_text = f" · erfasste Signalenergie {energy:.1%}" if isinstance(energy, (int, float)) else ""
        self.projection_status.setText(
            "SVD: datenabhängige Komponenten, keine anatomischen Richtungen. VCG-Marker bleiben unverändert."
            + energy_text + (" · Achsen bei ähnlichen Singulärwerten instabil" if metadata.get("near_degenerate_axes") else "")
            + (" · nicht verfügbar: " + str(metadata.get("reason", "noch nicht berechnet")) if metadata.get("status") != "available" else "")
            if component_view else "")
        if not len(self.xyz): return
        mask = np.isfinite(self.xyz).all(axis=1)
        segment = self.segment.currentText()
        from ...mimic.pilot_viewmodel import laboratory_segment_mask
        mask &= laboratory_segment_mask(len(mask), self.array_data, segment)
        positions = self.xyz.copy(); positions[~mask] = np.nan
        self.loop_visible_mask = mask
        horizontal, vertical = self.plane.currentData()
        axes_name = labels[horizontal] + "–" + labels[vertical]
        self.loop_plot.setLabel("bottom", ("SVD " if component_view else "VCG ") + labels[horizontal], units="mV")
        self.loop_plot.setLabel("left", ("SVD " if component_view else "VCG ") + labels[vertical], units="mV")
        # NaN gaps prevent artificial connections between different beats.
        self.loop_plot.plot(positions[:, horizontal], positions[:, vertical], pen=self.signal_pen(), connect="finite")
        visible = np.flatnonzero(mask)
        colors = pg.colormap.get("viridis").map(np.arange(len(mask))/max(1, len(mask)-1), mode="float")
        if len(visible) and self.gl:
            self.loop_time_label.setText(
                f"3D-Zeitfarbe: 0 bis {(len(mask)-1)/self.fs:.3f} s (gesamte Aufnahme)"
            )
            self.loop_time_gradient.setVisible(True)
        self.loop_plot.addItem(self.loop_cursor)
        if self.gl:
            if len(visible):
                starts = np.flatnonzero(mask[:-1] & mask[1:])
                pair_index = np.column_stack((starts, starts+1)).ravel()
                self.loop3d.addItem(self.gl.GLLinePlotItem(pos=self.xyz[pair_index], color=colors[pair_index],
                                                          width=2, antialias=True, mode="lines"))
                scale = max(.5, float(np.nanmax(np.linalg.norm(positions[visible], axis=1))))
                # Fit the full signal extent, retaining outliers and equal
                # spatial scale instead of enlarging each axis independently.
                self.loop3d.fit_loop(scale)
                axis = self.gl.GLAxisItem(); axis.setSize(scale, scale, scale); self.loop3d.addItem(axis)
                self.loop_time_label.setText(self.loop_time_label.text()
                    + f" · Achsenmarken {','.join(labels)}: jeweils {.45 * scale:.3g} mV")
                for coordinate, name in enumerate(labels):
                    endpoint = np.zeros(3)
                    # Keep labels inside the shallow landscape viewport;
                    # the text reports this actual coordinate, not axis length.
                    endpoint[coordinate] = .45 * scale
                    axis_text = self.gl.GLTextItem(
                        pos=endpoint, text=name,
                        color=pg.mkColor(pg.getConfigOption("foreground")),
                        font=self.font(),
                    )
                    # QPainter labels must follow the native GL geometry.
                    axis_text.setDepthValue(100)
                    self.loop3d.addItem(axis_text)
            self.cursor3d = self.gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, .65, 0, 1), size=11)
            self.loop3d.addItem(self.cursor3d)
        self.loop_plot.setTitle(f"{segment} · {axes_name} [mV] / rechts {'C1/C2/C3' if component_view else 'XYZ'} · 3D-Farbe: Anfang violett → Ende gelb")
        if not self.gl:
            self.loop_plot.setTitle(f"{segment} · {axes_name} [mV] · 2D-Projektion; 3D nicht verfügbar")
        if not len(visible):
            self.loop_plot.setTitle(f"{segment} · {axes_name}: kein freigegebenes, messbares Segment")
        self.time_changed(self.time_slider.value())

    def time_changed(self, sample):
        if not hasattr(self, "fs"): return
        self.cursor.blockSignals(True); self.cursor.setValue(sample/self.fs); self.cursor.blockSignals(False)
        if 0 <= sample < len(self.xyz):
            point = self.xyz[sample]
            horizontal, vertical = self.plane.currentData()
            visible = bool(getattr(self, "loop_visible_mask", np.zeros(len(self.xyz), dtype=bool))[sample])
            self.loop_cursor.setData([point[horizontal]] if visible else [], [point[vertical]] if visible else [])
            if self.gl and hasattr(self, "cursor3d"):
                self.cursor3d.setData(pos=point.reshape(1, 3) if visible else np.empty((0, 3)))

    def cursor_moved(self):
        if hasattr(self, "fs"):
            self.time_slider.setValue(int(round(self.cursor.value()*self.fs)))

    def export(self):
        if getattr(self, "export_frame", pd.DataFrame()).empty:
            self.summary.setText("Kein Export: Fuer die aktuelle Auswahl liegen keine geladenen Ergebnisse vor.")
            return
        from ..pilot_export import export_pilot_bundle, render_readable_plot_bundle
        try:
            study = self.case.currentData()
            timeline = self.model.timeline(int(study)) if study is not None else None
            def render(bundle):
                render_readable_plot_bundle(self.plot.plotItem, bundle,
                    endpoint=self.endpoint.currentText(), marker=self.marker.currentText())
            bundle = export_pilot_bundle(self.model.root,
                dict(pilot_size=int(self.size.currentText()), endpoint=self.endpoint.currentText(),
                     marker=self.marker.currentText(), context_study_id=study,
                     context_scope="separate_selected_lab_case_not_cohort_effect"),
                self.export_frame, timeline, render)
            self.summary.setText(self.summary.text() + f"\nExport: {bundle}")
        except Exception as exc:
            logging.getLogger(__name__).exception("Pilot export failed")
            self.summary.setText(self.summary.text() + f"\nExport fehlgeschlagen: {exc}")

    def signal_pen(self):
        dark = self.palette().color(self.backgroundRole()).lightness() < 128
        return pg.mkPen("#56B4E9" if dark else "#0072B2", width=1.3)

    def closeEvent(self, event):
        if self.reviewer_import.busy:
            self.reviewer_import.status.setText("Bitte den laufenden Antwortenimport vor dem Schließen abschließen lassen.")
            self.tabs.setCurrentWidget(self.reviewer_import)
            event.ignore()
            return
        if self.reviewer_export.busy:
            self.reviewer_export.status.setText("Bitte den laufenden Signalexport vor dem Schließen abschließen lassen.")
            self.tabs.setCurrentWidget(self.reviewer_export)
            event.ignore()
            return
        if self.task is not None and self.task.isRunning():
            self.lab_status.setText("Bitte die laufende Laboranalyse vor dem Schließen abschließen lassen.")
            event.ignore()
        else:
            super().closeEvent(event)

    def reject(self):
        if self.reviewer_import.busy or self.reviewer_export.busy or (self.task is not None and self.task.isRunning()):
            return
        super().reject()
