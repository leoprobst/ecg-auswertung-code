from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QPushButton, QCheckBox, QSlider, QGridLayout,
    QSplitter, QFrame, QProgressBar, QStackedWidget, QDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from ...config import AppConfig
from ...core.vcg_transform import VCGResult
from ...core.delineation.base import BeatDelineation, RecordDelineation
from ...core.reference import ReferenceLoopStore, ReferenceLoop
from ...core.delineation.base import measurement_confidence_threshold


OCTANT_LABELS = [
    "+X+Y+Z", "+X+Y-Z", "+X-Y+Z", "+X-Y-Z",
    "-X+Y+Z", "-X+Y-Z", "-X-Y+Z", "-X-Y-Z",
]
OCTANT_COLORS = [
    (1.0, 0.4, 0.4, 0.7),
    (1.0, 0.6, 0.2, 0.7),
    (0.4, 1.0, 0.4, 0.7),
    (0.2, 0.8, 0.8, 0.7),
    (0.4, 0.4, 1.0, 0.7),
    (0.8, 0.2, 0.8, 0.7),
    (0.8, 0.8, 0.2, 0.7),
    (0.6, 0.6, 0.6, 0.7),
]


def _time_color_array(n: int, colormap_name: str = "viridis") -> np.ndarray:
    if n <= 0:
        return np.empty((0, 4), dtype=np.float32)

    cmap = pg.colormap.get(colormap_name)
    t = np.linspace(0, 1, n)
    colors = np.asarray(cmap.map(t), dtype=np.float32)
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    if colors.shape[1] == 3:
        colors = np.column_stack([colors, np.ones(len(colors), dtype=np.float32)])
    if np.max(colors[:, :3]) > 1.0:
        colors[:, :3] /= 255.0
    if np.max(colors[:, 3]) > 1.0:
        colors[:, 3] /= 255.0
    colors = np.clip(colors[:, :4], 0.0, 1.0)
    colors[:, 3] = 0.85
    return colors.astype(np.float32)


def _rgba_to_qcolor(rgba: np.ndarray) -> QColor:
    rgba = np.clip(np.asarray(rgba, dtype=float), 0.0, 1.0)
    return QColor.fromRgbF(float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


def _compute_octant_dwell(xyz: np.ndarray) -> dict[str, float]:
    signs = np.sign(xyz)
    signs[signs == 0] = 1
    total = len(xyz)
    if total == 0:
        return {label: 0.0 for label in OCTANT_LABELS}
    dwell = {}
    for i, label in enumerate(OCTANT_LABELS):
        # label format: "+X+Y+Z" → [0]=X-sign, [2]=Y-sign, [4]=Z-sign
        match = (
            (signs[:, 0] == (1 if label[0] == "+" else -1)) &
            (signs[:, 1] == (1 if label[2] == "+" else -1)) &
            (signs[:, 2] == (1 if label[4] == "+" else -1))
        )
        dwell[label] = float(np.sum(match)) / total * 100.0
    return dwell


def _compute_velocity_field(xyz: np.ndarray, step: int = 5) -> tuple[np.ndarray, np.ndarray]:
    n = len(xyz)
    indices = np.arange(0, n - step, step)
    origins = xyz[indices]
    deltas = xyz[indices + step] - xyz[indices]
    return origins, deltas


class OctantPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dwell_data = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Octant Dwell Time (%)")
        header.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(header)

        self._bars = {}
        grid = QGridLayout()
        for i, (label, color) in enumerate(zip(OCTANT_LABELS, OCTANT_COLORS)):
            row, col = divmod(i, 2)
            frame = QFrame()
            frame.setStyleSheet(f"background-color: rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},40); border-radius: 3px;")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(2, 2, 2, 2)
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 9px;")
            fl.addWidget(lbl)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(True)
            fl.addWidget(bar)
            self._bars[label] = bar
            grid.addWidget(frame, row, col)

        layout.addLayout(grid)

    def update_dwell(self, dwell: dict[str, float]):
        self._dwell_data = dwell
        for label, bar in self._bars.items():
            val = dwell.get(label, 0.0)
            bar.setValue(int(round(val * 10)))   # range 0..1000 → 0.1% resolution
            bar.setFormat(f"{val:.1f}%")


class VCGWidget(QWidget):
    time_index_selected = pyqtSignal(int)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._current_vcg: VCGResult | None = None
        self._current_delineation: RecordDelineation | None = None
        self._current_fs: int = 500
        self._velocity_visible: bool = False
        self._velocity_step: int = 5
        self._reference_store: ReferenceLoopStore | None = None
        self._show_reference: bool = False
        self._show_ellipse: bool = False
        self._current_age: int | None = None
        self._current_sex: str | None = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("Projection:"))
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["3D", "Frontal (X-Y)", "Horizontal (X-Z)", "Sagittal (Y-Z)"])
        self.projection_combo.currentTextChanged.connect(self._on_projection_changed)
        ctrl_layout.addWidget(self.projection_combo)

        ctrl_layout.addWidget(QLabel("Segment:"))
        self.segment_combo = QComboBox()
        self.segment_combo.addItems(["Full", "QRS Loop", "T Loop", "P Loop"])
        self.segment_combo.currentTextChanged.connect(self._on_segment_changed)
        ctrl_layout.addWidget(self.segment_combo)

        ctrl_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "coolwarm", "rainbow"])
        self.cmap_combo.currentTextChanged.connect(self._refresh)
        ctrl_layout.addWidget(self.cmap_combo)

        self.velocity_cb = QCheckBox("Velocity")
        self.velocity_cb.stateChanged.connect(self._on_velocity_toggle)
        ctrl_layout.addWidget(self.velocity_cb)

        ctrl_layout.addWidget(QLabel("Vel Step:"))
        self.vel_step_slider = QSlider(Qt.Orientation.Horizontal)
        self.vel_step_slider.setRange(2, 20)
        self.vel_step_slider.setValue(5)
        self.vel_step_slider.setMaximumWidth(80)
        self.vel_step_slider.valueChanged.connect(self._on_vel_step_changed)
        ctrl_layout.addWidget(self.vel_step_slider)

        self.ref_cb = QCheckBox("Reference")
        self.ref_cb.stateChanged.connect(self._on_ref_toggle)
        ctrl_layout.addWidget(self.ref_cb)

        self.ellipse_cb = QCheckBox("2sd Ellipse")
        self.ellipse_cb.stateChanged.connect(self._on_ellipse_toggle)
        ctrl_layout.addWidget(self.ellipse_cb)

        self.temporal_views_button = QPushButton("QRS-Zeitstruktur")
        self.temporal_views_button.setToolTip("Forschungsansicht des aktuellen QRS mit Qualitätsprüfung; keine Aktivierungskarte")
        self.temporal_views_button.clicked.connect(self._show_temporal_views)
        ctrl_layout.addWidget(self.temporal_views_button)

        ctrl_layout.addStretch()
        main_layout.addLayout(ctrl_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        plot_frame = QWidget()
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=3)

        self._2d_widget = pg.GraphicsLayoutWidget()
        self._2d_widget.setBackground("w")

        self._plot_stack = QStackedWidget()
        self._plot_stack.addWidget(self.gl_widget)   # index 0 → 3D
        self._plot_stack.addWidget(self._2d_widget)  # index 1 → 2D
        self._plot_stack.setCurrentIndex(0)

        plot_layout.addWidget(self._plot_stack)

        xyz_group = QGroupBox("VCG Leads X, Y, Z")
        xyz_layout = QVBoxLayout(xyz_group)
        self.xyz_plot = pg.GraphicsLayoutWidget()
        self.xyz_plot.setBackground("w")
        xyz_layout.addWidget(self.xyz_plot)
        plot_layout.addWidget(xyz_group)

        splitter.addWidget(plot_frame)

        self.octant_panel = OctantPanel()
        self.octant_panel.setMaximumWidth(280)
        splitter.addWidget(self.octant_panel)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self._vcf_line = None
        self._scatter_item = None

    def _show_temporal_views(self):
        """Render on demand, never recompute features or mutate a running analysis."""
        from ...core.views import render_views
        from .temporal_view_data import qrs_view_points
        beat = self._current_beat()
        try:
            points = qrs_view_points(self._current_vcg, beat, self._current_fs)
        except ValueError as exc:
            QMessageBox.information(self, "Zeitstruktur nicht auswertbar", str(exc))
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        figure = render_views(points, None, self._current_fs, title="Aktueller repräsentativer QRS")
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("QRS-Zeitstruktur – Forschungsansicht")
        dialog.resize(1150, 740)
        layout = QVBoxLayout(dialog)
        warning = QLabel("Momentaufnahme des aktuellen Signals. Segmentqualität geprüft; absolute SNR und Beat-Homogenität nicht validiert. Kein lokales Aktionspotential und keine Therapieempfehlung.")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(NavigationToolbar2QT(canvas, dialog))
        layout.addWidget(canvas)
        dialog.show()

    def display_vcg(self, vcg_result: VCGResult, delineation=None, fs: int = 500,
                    age: int | None = None, sex: str | None = None):
        self._current_vcg = vcg_result
        self._current_delineation = delineation
        self._current_fs = fs
        self._current_age = age
        self._current_sex = sex
        self._refresh()

    def get_segment_indices(self) -> tuple[int, int]:
        if self._current_vcg is None:
            return 0, 0

        beat = self._current_beat()
        beat_start, beat_end = self._single_beat_indices(beat)
        segment = self.segment_combo.currentText()
        if segment == "Full" or beat is None:
            return beat_start, beat_end

        if segment == "QRS Loop":
            if beat.segment_confidence("qrs") < measurement_confidence_threshold("qrs"):
                return 0, 0
            start = beat.r_onset.sample
            end = beat.r_offset.sample
        elif segment == "T Loop":
            if beat.segment_confidence("t") < measurement_confidence_threshold("t"):
                return 0, 0
            start = beat.t_onset.sample
            end = beat.t_offset.sample
        elif segment == "P Loop":
            if beat.segment_confidence("p") < measurement_confidence_threshold("p"):
                return 0, 0
            start = beat.p_onset.sample
            end = beat.p_offset.sample
        else:
            return beat_start, beat_end

        start = max(beat_start, start)
        end = min(beat_end, end)
        return self._clip_indices(start, end)

    def _current_beat(self) -> BeatDelineation | None:
        if self._current_delineation is None:
            return None
        return self._current_delineation.representative_beat

    def _single_beat_indices(self, beat: BeatDelineation | None = None) -> tuple[int, int]:
        if self._current_vcg is None:
            return 0, 0

        n = len(self._current_vcg.vx)
        if n <= 1:
            return 0, max(0, n - 1)

        if beat is None:
            return 0, min(n - 1, max(1, int(self._current_fs)))

        rp = int(beat.r_peak.sample)
        start = beat.p_onset.sample if beat.p_onset and beat.p_onset.sample < rp else None
        end = beat.t_offset.sample if beat.t_offset and beat.t_offset.sample > rp else None

        if start is None or end is None:
            r_samples = sorted(
                int(b.r_peak.sample)
                for b in (self._current_delineation.beats if self._current_delineation else [])
                if 0 <= int(b.r_peak.sample) < n
            )
            if r_samples:
                idx = min(range(len(r_samples)), key=lambda i: abs(r_samples[i] - rp))
                if start is None:
                    start = (r_samples[idx - 1] + r_samples[idx]) // 2 if idx > 0 else rp - int(0.35 * self._current_fs)
                if end is None:
                    end = (r_samples[idx] + r_samples[idx + 1]) // 2 if idx < len(r_samples) - 1 else rp + int(0.65 * self._current_fs)

        if start is None:
            start = rp - int(0.35 * self._current_fs)
        if end is None:
            end = rp + int(0.65 * self._current_fs)

        return self._clip_indices(start, end)

    def _clip_indices(self, start: int, end: int) -> tuple[int, int]:
        if self._current_vcg is None:
            return 0, 0

        n = len(self._current_vcg.vx)
        if n <= 1:
            return 0, max(0, n - 1)

        start = max(0, min(n - 1, int(start)))
        end = max(0, min(n - 1, int(end)))
        if end <= start:
            end = min(n - 1, start + 1)
        return start, end

    def _refresh(self):
        if self._current_vcg is None:
            return

        projection = self.projection_combo.currentText()
        vx, vy, vz = self._current_vcg.vx, self._current_vcg.vy, self._current_vcg.vz
        xyz_data = np.column_stack([vx, vy, vz])

        start, end = self.get_segment_indices()
        xyz_seg = xyz_data[start:end + 1] if end >= start else xyz_data[:0]

        if projection == "3D":
            self._draw_3d(xyz_seg, start)
        else:
            self._draw_2d(xyz_seg, start, projection)

        self._draw_xyz_traces(start, end)

        dwell = _compute_octant_dwell(xyz_seg)
        self.octant_panel.update_dwell(dwell)

    def _draw_3d(self, data: np.ndarray, sample_offset: int = 0):
        self._plot_stack.setCurrentIndex(0)
        data = np.asarray(data, dtype=np.float32)

        for item in list(self.gl_widget.items):
            self.gl_widget.removeItem(item)

        axis_len = float(np.max(np.abs(data)) * 1.3) if len(data) > 0 else 1.0
        axis_len = max(axis_len, 0.1)

        # auto-scale camera so the loop fills the viewport
        self.gl_widget.setCameraPosition(distance=axis_len * 5)

        grid = gl.GLGridItem()
        grid.setSize(axis_len * 2, axis_len * 2, 1)
        grid.setSpacing(axis_len / 4, axis_len / 4, axis_len / 4)
        self.gl_widget.addItem(grid)

        for axis_vec, color in [
            (np.array([[0, 0, 0], [axis_len, 0, 0]], dtype=np.float32), (1, 0, 0, 0.8)),
            (np.array([[0, 0, 0], [0, axis_len, 0]], dtype=np.float32), (0, 0.8, 0, 0.8)),
            (np.array([[0, 0, 0], [0, 0, axis_len]], dtype=np.float32), (0, 0.4, 1, 0.8)),
        ]:
            self.gl_widget.addItem(gl.GLLinePlotItem(pos=axis_vec, color=color, width=2))

        if len(data) < 2:
            return

        cmap_name = self.cmap_combo.currentText()
        colors = _time_color_array(len(data), cmap_name)

        # colored line connecting all sample points (time-gradient)
        try:
            line_colors = colors.copy()
            line_colors[:, 3] = 0.9
            loop_plot = gl.GLLinePlotItem(pos=data, color=line_colors, width=2.5, antialias=True)
        except Exception:
            loop_plot = gl.GLLinePlotItem(pos=data, color=(0.3, 0.6, 1.0, 0.9), width=2.5, antialias=True)
        self.gl_widget.addItem(loop_plot)

        # scatter overlay for start/end markers
        self._scatter_item = gl.GLScatterPlotItem(pos=data, color=colors, size=5, pxMode=True)
        self.gl_widget.addItem(self._scatter_item)

        if data.shape[0] > 0:
            start_pt = gl.GLScatterPlotItem(
                pos=data[0:1],
                color=(0, 1, 0, 1),
                size=12,
                pxMode=True,
            )
            self.gl_widget.addItem(start_pt)

            end_pt = gl.GLScatterPlotItem(
                pos=data[-1:],
                color=(1, 0, 0, 1),
                size=12,
                pxMode=True,
            )
            self.gl_widget.addItem(end_pt)

        if self._velocity_visible and len(data) > self._velocity_step:
            self._draw_velocity_arrows_3d(data)

        if self._show_reference:
            self._draw_reference_3d()

    def _draw_reference_3d(self):
        if self._reference_store is None:
            return
        ref = self._reference_store.get_reference(self._current_age, self._current_sex)
        if ref is None:
            return

        if ref.qrs_mean is not None and len(ref.qrs_mean) > 1:
            qrs_ref = gl.GLLinePlotItem(
                pos=ref.qrs_mean,
                color=(0.5, 0.5, 0.5, 0.5),
                width=2,
                antialias=True,
            )
            self.gl_widget.addItem(qrs_ref)

        if ref.t_mean is not None and len(ref.t_mean) > 1:
            t_ref = gl.GLLinePlotItem(
                pos=ref.t_mean,
                color=(0.7, 0.7, 0.4, 0.5),
                width=2,
                antialias=True,
            )
            self.gl_widget.addItem(t_ref)

    def _draw_velocity_arrows_3d(self, data: np.ndarray):
        origins, deltas = _compute_velocity_field(data, self._velocity_step)
        scale = 0.3 / (np.max(np.linalg.norm(deltas, axis=1)) + 1e-10)

        for i in range(0, len(origins), max(1, len(origins) // 30)):
            o = origins[i]
            d = deltas[i] * scale
            end = o + d
            speed = np.linalg.norm(deltas[i])
            max_speed = np.max(np.linalg.norm(deltas, axis=1)) + 1e-10
            intensity = speed / max_speed
            color = (intensity, 0.3, 1.0 - intensity, 0.7)

            arrow = gl.GLLinePlotItem(
                pos=np.array([o, end]),
                color=color,
                width=2.5,
                antialias=True,
            )
            self.gl_widget.addItem(arrow)

    def _draw_2d(self, data: np.ndarray, sample_offset: int, projection: str):
        self._plot_stack.setCurrentIndex(1)
        self._2d_widget.clear()

        if projection == "Frontal (X-Y)":
            x_idx, y_idx = 0, 1
            xlabel, ylabel = "X (mV)", "Y (mV)"
        elif projection == "Horizontal (X-Z)":
            x_idx, y_idx = 0, 2
            xlabel, ylabel = "X (mV)", "Z (mV)"
        elif projection == "Sagittal (Y-Z)":
            x_idx, y_idx = 1, 2
            xlabel, ylabel = "Y (mV)", "Z (mV)"
        else:
            return

        plot = self._2d_widget.addPlot()
        plot.setLabel("bottom", xlabel)
        plot.setLabel("left", ylabel)
        plot.setAspectLocked(False)

        if len(data) < 2:
            return

        x = data[:, x_idx]
        y = data[:, y_idx]

        colors = _time_color_array(len(data), self.cmap_combo.currentText())
        for i in range(len(data) - 1):
            plot.plot(
                [x[i], x[i + 1]], [y[i], y[i + 1]],
                pen=pg.mkPen(_rgba_to_qcolor(colors[i]), width=1.8),
            )

        plot.plot([0], [0], pen=None, symbol="o", symbolSize=8, symbolBrush="k")

        start_brush = pg.mkBrush(0, 255, 0, 200)
        end_brush = pg.mkBrush(255, 0, 0, 200)
        plot.plot([x[0]], [y[0]], pen=None, symbol="o", symbolSize=10, symbolBrush=start_brush)
        plot.plot([x[-1]], [y[-1]], pen=None, symbol="o", symbolSize=10, symbolBrush=end_brush)

        if self._velocity_visible and len(data) > self._velocity_step:
            origins, deltas = _compute_velocity_field(data, self._velocity_step)
            max_speed = np.max(np.linalg.norm(deltas, axis=1)) + 1e-10
            scale = (np.max(x) - np.min(x)) * 0.05 / (max_speed + 1e-10)

            for i in range(0, len(origins), max(1, len(origins) // 40)):
                ox = origins[i][x_idx]
                oy = origins[i][y_idx]
                dx = deltas[i][x_idx] * scale
                dy = deltas[i][y_idx] * scale
                speed = np.linalg.norm(deltas[i])
                intensity = speed / max_speed
                color = QColor(
                    int(255 * intensity),
                    int(80),
                    int(255 * (1 - intensity)),
                    180,
                )
                arrow = pg.PlotCurveItem(
                    [ox, ox + dx], [oy, oy + dy],
                    pen=pg.mkPen(color, width=2),
                )
                plot.addItem(arrow)

        if self._show_reference:
            self._draw_reference_2d(plot, x_idx, y_idx)

        if self._show_ellipse:
            self._draw_confidence_ellipse_2d(plot, x_idx, y_idx)

    def _draw_reference_2d(self, plot, x_idx: int, y_idx: int):
        if self._reference_store is None:
            return
        ref = self._reference_store.get_reference(self._current_age, self._current_sex)
        if ref is None:
            return

        for loop_data, color in [
            (ref.qrs_mean, pg.mkPen((128, 128, 128, 128), width=2, style=Qt.PenStyle.DashLine)),
            (ref.t_mean, pg.mkPen((180, 180, 80, 128), width=2, style=Qt.PenStyle.DashLine)),
        ]:
            if loop_data is not None and len(loop_data) > 1:
                rx = loop_data[:, x_idx]
                ry = loop_data[:, y_idx]
                plot.plot(rx, ry, pen=color)

    def _draw_confidence_ellipse_2d(self, plot, x_idx: int, y_idx: int):
        if self._reference_store is None:
            return
        ref = self._reference_store.get_reference(self._current_age, self._current_sex)
        if ref is None:
            return

        plane_map = {"Frontal": "frontal", "Horizontal": "horizontal", "Sagittal": "sagittal"}
        projection = self.projection_combo.currentText()
        plane_key = None
        for key, val in plane_map.items():
            if key in projection:
                plane_key = val
                break
        if plane_key is None:
            return

        for loop_type, color in [("qrs", (128, 128, 128, 60)), ("t", (180, 180, 80, 60))]:
            ellipse = self._reference_store.compute_confidence_ellipse(
                ref, loop_type=loop_type, n_std=2.0, plane=plane_key,
            )
            if ellipse is not None:
                plot.plot(ellipse[:, 0], ellipse[:, 1],
                          pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine),
                          fillLevel=None)

    def _draw_xyz_traces(self, highlight_start: int = 0, highlight_end: int = -1):
        self.xyz_plot.clear()

        if self._current_vcg is None:
            return

        vcg = self._current_vcg
        fs = self._current_fs
        n = len(vcg.vx)
        if n == 0:
            return

        highlight_start, highlight_end = self._clip_indices(highlight_start, highlight_end if highlight_end >= 0 else n - 1)
        t = np.arange(highlight_start, highlight_end + 1) / fs

        xyz_plots = []
        for i, (signal, name, color) in enumerate([
            (vcg.vx, "X", "r"),
            (vcg.vy, "Y", "g"),
            (vcg.vz, "Z", "b"),
        ]):
            plot = self.xyz_plot.addPlot(row=i, col=0)
            xyz_plots.append(plot)
            plot.plot(t, signal[highlight_start:highlight_end + 1], pen=pg.mkPen(color, width=1.5))

            if self._velocity_visible and n > 1:
                sv = self._current_vcg.spatial_velocity
                sv_start = min(highlight_start, len(sv))
                sv_end = min(highlight_end, len(sv))
                sv_t = np.arange(sv_start, sv_end) / fs
                sv_plot = self.xyz_plot.addPlot(row=3, col=0) if i == 2 else None
                if sv_plot is not None:
                    sv_plot.plot(sv_t, sv[sv_start:sv_end], pen=pg.mkPen("m", width=1))
                    sv_plot.setLabel("left", "SV", units="mV/s")

            plot.setLabel("left", name, units="mV")
            if i < 2:
                plot.hideAxis("bottom")

        if xyz_plots:
            xyz_plots[-1].showAxis("bottom")
            xyz_plots[-1].setLabel("bottom", "Time", units="s")

    def _on_projection_changed(self, text):
        self._refresh()

    def _on_segment_changed(self, text):
        self._refresh()

    def _on_velocity_toggle(self, state):
        self._velocity_visible = state == Qt.CheckState.Checked.value
        self._refresh()

    def _on_vel_step_changed(self, value):
        self._velocity_step = value
        self._refresh()

    def _on_ref_toggle(self, state):
        self._show_reference = state == Qt.CheckState.Checked.value
        self._refresh()

    def _on_ellipse_toggle(self, state):
        self._show_ellipse = state == Qt.CheckState.Checked.value
        self._refresh()

    def set_reference_store(self, store: ReferenceLoopStore):
        self._reference_store = store

    def set_patient_info(self, age: int | None = None, sex: str | None = None):
        self._current_age = age
        self._current_sex = sex

    def highlight_sample(self, sample_idx: int):
        if self._current_vcg is None:
            return

        vcg = self._current_vcg
        if sample_idx < 0 or sample_idx >= len(vcg.vx):
            return

        pt = np.array([[vcg.vx[sample_idx], vcg.vy[sample_idx], vcg.vz[sample_idx]]])

        if self.projection_combo.currentText() == "3D":
            if self._vcf_line is not None:
                try:
                    self.gl_widget.removeItem(self._vcf_line)
                except Exception:
                    pass

            self._vcf_line = gl.GLScatterPlotItem(
                pos=pt,
                color=(1, 1, 0, 1),
                size=15,
                pxMode=True,
            )
            self.gl_widget.addItem(self._vcf_line)
