from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox,
    QPushButton, QComboBox, QGroupBox, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QLineEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
import pyqtgraph as pg

from ...config import AppConfig
from ...core.data_loader import LEAD_NAMES_12
from ...core.delineation.base import (
    BeatDelineation, MarkerPoint, RecordDelineation, WaveStatus, WaveType,
)
from .caliper_widget import CaliperManager


MARKER_COLORS = {
    "p_onset": "#00CC00", "p_peak": "#00FF00", "p_offset": "#00CC00",
    "q_peak": "#CC00CC", "r_onset": "#FF4444", "r_peak": "#FF0000",
    "r_offset": "#FF4444", "s_peak": "#CC00CC",
    "j_point": "#FF8800",
    "t_onset": "#0088CC", "t_peak": "#00AAFF", "t_offset": "#0088CC",
}

MARKER_LABELS = {
    "p_onset": "Pon", "p_peak": "P", "p_offset": "Poff",
    "q_peak": "Q", "r_onset": "Ron", "r_peak": "R",
    "r_offset": "Roff", "s_peak": "S",
    "j_point": "J",
    "t_onset": "Ton", "t_peak": "T", "t_offset": "Toff",
}

WAVE_REGION_COLORS = {
    "P": "#42B883",
    "PR": "#E3B341",
    "QRS": "#E55353",
    "ST": "#F08C46",
    "T": "#3B82C4",
}


class DraggableMarker(pg.InfiniteLine):
    def __init__(self, marker_type: str, sample: int, fs: int, parent_plot=None):
        color = MARKER_COLORS.get(marker_type, "#FFFFFF")
        super().__init__(
            pos=sample / fs,
            angle=90,
            pen=pg.mkPen(color, width=2),
            movable=True,
        )
        self.marker_type = marker_type
        self._sample = sample
        self._fs = fs
        self._parent_plot = parent_plot

        label_text = MARKER_LABELS.get(marker_type, marker_type)
        self.label = pg.TextItem(label_text, color=color, anchor=(0, 1))
        self.label.setPos(sample / fs, 0)

    @property
    def sample(self) -> int:
        return int(self.value() * self._fs)

    @sample.setter
    def sample(self, val: int):
        self._sample = val
        self.setValue(val / self._fs)

    def update_label_position(self, y_pos: float):
        self.label.setPos(self.value(), y_pos)


class AnnotationWidget(QWidget):
    markers_changed = pyqtSignal()
    biomarker_recompute_requested = pyqtSignal()
    sample_selected = pyqtSignal(int)
    caliper_updated = pyqtSignal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._current_signal: np.ndarray | None = None
        self._current_fs: int = 500
        self._current_delineation: RecordDelineation | None = None
        self._markers: list[DraggableMarker] = []
        self._marker_items_per_plot: dict[int, list] = {}
        self._current_beat_idx: int = 0
        self._crosshair_lines: list[pg.InfiniteLine] = []
        self._caliper_manager: CaliperManager | None = None
        self._caliper_regions: list[pg.LinearRegionItem] = []
        self._wave_region_items: list[tuple[pg.PlotItem, pg.LinearRegionItem]] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Beat:"))
        self.beat_spin = QSpinBox()
        self.beat_spin.setRange(0, 0)
        self.beat_spin.valueChanged.connect(self._on_beat_changed)
        ctrl.addWidget(self.beat_spin)

        self.show_markers_cb = QCheckBox("Show Markers")
        self.show_markers_cb.setChecked(True)
        self.show_markers_cb.stateChanged.connect(self._on_show_markers)
        ctrl.addWidget(self.show_markers_cb)

        self.show_regions_cb = QCheckBox("Show Wave Regions")
        self.show_regions_cb.setChecked(True)
        self.show_regions_cb.setToolTip(
            "Highlights detected P, PR, QRS, ST and T regions. "
            "Opacity reflects confidence; dashed borders require review "
            "and are excluded from biomarker calculation."
        )
        self.show_regions_cb.stateChanged.connect(self._on_show_regions)
        ctrl.addWidget(self.show_regions_cb)

        ctrl.addWidget(QLabel("Snap:"))
        self.snap_combo = QComboBox()
        self.snap_combo.addItems(["Off", "Peak", "Trough", "Zero-Crossing"])
        ctrl.addWidget(self.snap_combo)

        self.recompute_btn = QPushButton("Recompute Biomarkers")
        self.recompute_btn.clicked.connect(self.biomarker_recompute_requested.emit)
        ctrl.addWidget(self.recompute_btn)

        ctrl.addWidget(QLabel("Lead:"))
        self.lead_combo = QComboBox()
        self.lead_combo.addItems(LEAD_NAMES_12)
        self.lead_combo.setCurrentIndex(1)
        ctrl.addWidget(self.lead_combo)

        self.snap_btn = QPushButton("Snap All")
        self.snap_btn.clicked.connect(self._snap_all_markers)
        ctrl.addWidget(self.snap_btn)

        self.caliper_btn = QPushButton("Add Caliper")
        self.caliper_btn.setCheckable(True)
        self.caliper_btn.toggled.connect(self._on_caliper_toggle)
        ctrl.addWidget(self.caliper_btn)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("w")
        layout.addWidget(self.plot_widget)

        interval_group = QGroupBox("Intervals (auto-computed from markers)")
        interval_layout = QVBoxLayout(interval_group)
        self.interval_table = QTableWidget()
        self.interval_table.setColumnCount(5)
        self.interval_table.setHorizontalHeaderLabels(
            [
                "Interval",
                "Value (ms)",
                "Uncertainty (+/- ms)",
                "Reference",
                "Confidence",
            ]
        )
        self.interval_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        interval_layout.addWidget(self.interval_table)
        layout.addWidget(interval_group)

        self._plots = []
        self._curves = []

    def display_ecg(self, signal: np.ndarray, fs: int, delineation: RecordDelineation | None = None):
        self._current_signal = signal
        self._current_fs = fs
        self._current_delineation = delineation

        if delineation and delineation.beats:
            self.beat_spin.setRange(0, len(delineation.beats) - 1)
        else:
            self.beat_spin.setRange(0, 0)

        self._draw_ecg()

    def _draw_ecg(self):
        if self._current_signal is None:
            return

        signal = self._current_signal
        fs = self._current_fs
        n_leads = min(signal.shape[1], 12)

        self.plot_widget.clear()
        self._plots = []
        self._curves = []
        self._crosshair_lines = []
        self._clear_markers()

        t = np.arange(signal.shape[0]) / fs

        for i in range(n_leads):
            plot = self.plot_widget.addPlot(row=i, col=0)
            lead_data = signal[:, i]

            spacing = 1.5
            offset = -i * spacing * np.max(np.abs(lead_data)) * 2 if i > 0 else 0
            display_data = lead_data + offset

            curve = plot.plot(t, display_data, pen=pg.mkPen("b", width=1))
            self._curves.append(curve)

            text = pg.TextItem(LEAD_NAMES_12[i], color="k")
            text.setPos(t[0], display_data[0] if len(display_data) > 0 else 0)
            plot.addItem(text)

            plot.hideAxis("bottom")
            plot.hideAxis("left")
            plot.setMouseEnabled(x=True, y=False)

            self._plots.append(plot)

        if self._plots:
            self._plots[-1].showAxis("bottom")
            self._plots[-1].setLabel("bottom", "Time", units="s")

        if self._current_delineation and self.show_regions_cb.isChecked():
            self._draw_wave_regions()

        if self._current_delineation and self.show_markers_cb.isChecked():
            self._draw_markers()

        self._redraw_calipers()
        self._update_interval_table()

    def _draw_wave_regions(self):
        if not self._current_delineation or not self._current_delineation.beats:
            return

        beat_idx = self.beat_spin.value()
        if beat_idx >= len(self._current_delineation.beats):
            return

        beat = self._current_delineation.beats[beat_idx]
        fs = self._current_fs
        definitions = (
            ("P", beat.p_onset, beat.p_offset, beat.segment_confidence("p")),
            ("PR", beat.p_offset, beat.r_onset, beat.segment_confidence("pr")),
            ("QRS", beat.r_onset, beat.r_offset, beat.segment_confidence("qrs")),
            (
                "ST",
                beat.j_point or beat.r_offset,
                beat.t_onset,
                beat.segment_confidence("st"),
            ),
            ("T", beat.t_onset, beat.t_offset, beat.segment_confidence("t")),
        )

        for name, start, end, confidence in definitions:
            if start is None or end is None or end.sample <= start.sample:
                continue
            if confidence < 0.20:
                continue
            color = QColor(WAVE_REGION_COLORS[name])
            alpha = int(24 + 46 * min(1.0, confidence))
            brush = QColor(color)
            brush.setAlpha(alpha)
            for plot in self._plots:
                region = pg.LinearRegionItem(
                    values=(start.sample / fs, end.sample / fs),
                    movable=False,
                    brush=pg.mkBrush(brush),
                    pen=pg.mkPen(
                        color,
                        width=0.8,
                        style=(
                            Qt.PenStyle.SolidLine
                            if confidence >= 0.35
                            else Qt.PenStyle.DashLine
                        ),
                    ),
                )
                region.setZValue(-5)
                region.setToolTip(
                    f"{name}: {start.sample / fs:.3f}-{end.sample / fs:.3f} s; "
                    f"confidence {confidence:.0%}; "
                    f"boundary uncertainty +/-"
                    f"{np.hypot(start.uncertainty_ms, end.uncertainty_ms):.1f} ms; "
                    + (
                        "evaluable for biomarkers"
                        if confidence >= 0.35
                        else "manual review required; excluded from biomarkers"
                    )
                )
                plot.addItem(region)
                self._wave_region_items.append((plot, region))

    def _draw_markers(self):
        if not self._current_delineation or not self._current_delineation.beats:
            return

        beat_idx = self.beat_spin.value()
        if beat_idx >= len(self._current_delineation.beats):
            return

        beat = self._current_delineation.beats[beat_idx]
        fs = self._current_fs
        signal = self._current_signal
        n_leads = min(signal.shape[1], 12)

        marker_attrs = [
            "p_onset", "p_peak", "p_offset",
            "q_peak", "r_onset", "r_peak", "r_offset", "s_peak",
            "j_point",
            "t_onset", "t_peak", "t_offset",
        ]

        for attr in marker_attrs:
            marker = getattr(beat, attr, None)
            if marker is None:
                continue

            sample = marker.sample
            dm = DraggableMarker(attr, sample, fs)
            dm.setToolTip(
                f"{MARKER_LABELS.get(attr, attr)}: {sample / fs:.3f} s; "
                f"confidence {marker.confidence:.0%}; "
                f"uncertainty +/-{marker.uncertainty_ms:.1f} ms; "
                f"source {marker.source}"
            )

            for i, plot in enumerate(self._plots):
                line_clone = pg.InfiniteLine(
                    pos=sample / fs,
                    angle=90,
                    pen=pg.mkPen(MARKER_COLORS.get(attr, "#FFF"), width=1.5, style=Qt.PenStyle.DashLine),
                    movable=False,
                )
                plot.addItem(line_clone)

                if i not in self._marker_items_per_plot:
                    self._marker_items_per_plot[i] = []
                self._marker_items_per_plot[i].append(line_clone)

            if self._plots:
                lead_idx = self.lead_combo.currentIndex()
                if lead_idx < len(self._plots):
                    self._plots[lead_idx].addItem(dm)
                    self._plots[lead_idx].addItem(dm.label)

                    spacing = 1.5
                    lead_data = signal[:, lead_idx]
                    offset = -lead_idx * spacing * np.max(np.abs(lead_data)) * 2 if lead_idx > 0 else 0
                    dm.update_label_position(lead_data[sample] + offset if sample < len(lead_data) else offset)

            dm.sigPositionChanged.connect(self._on_marker_moved)
            self._markers.append(dm)

    def _clear_markers(self):
        for dm in self._markers:
            try:
                dm.label.setParent(None)
            except Exception:
                pass

        for plot_idx, items in self._marker_items_per_plot.items():
            for item in items:
                try:
                    if plot_idx < len(self._plots):
                        self._plots[plot_idx].removeItem(item)
                except Exception:
                    pass

        for plot, region in self._wave_region_items:
            try:
                plot.removeItem(region)
            except Exception:
                pass

        self._markers.clear()
        self._marker_items_per_plot.clear()
        self._wave_region_items.clear()

    def _on_marker_moved(self, line: DraggableMarker):
        new_time = line.value()
        new_sample = line.sample

        for plot_idx, items in self._marker_items_per_plot.items():
            for item in items:
                try:
                    item.setValue(new_time)
                except Exception:
                    pass

        self._update_delineation_from_markers()
        self._update_interval_table()
        self.markers_changed.emit()

    def _update_delineation_from_markers(self):
        if not self._current_delineation or not self._current_delineation.beats:
            return

        beat_idx = self.beat_spin.value()
        if beat_idx >= len(self._current_delineation.beats):
            return

        beat = self._current_delineation.beats[beat_idx]

        for dm in self._markers:
            marker = getattr(beat, dm.marker_type, None)
            if marker is not None:
                new_marker = MarkerPoint(
                    sample=dm.sample,
                    confidence=0.95,
                    source="manual",
                    uncertainty_ms=2.0,
                    support_count=1,
                    lead_names=(self.lead_combo.currentText(),),
                    evidence_sources=("manual_review",),
                )
                setattr(beat, dm.marker_type, new_marker)
                if dm.marker_type.startswith("p_"):
                    beat.p_wave_status = WaveStatus.PRESENT
                    beat.p_wave_absent = False
                elif dm.marker_type.startswith("t_"):
                    beat.t_wave_status = WaveStatus.PRESENT
                    beat.t_wave_absent = False
        self._current_delineation.invalidate_beat_cache()

    def _snap_all_markers(self):
        snap_mode = self.snap_combo.currentText()
        if snap_mode == "Off" or self._current_signal is None:
            return

        for dm in self._markers:
            new_sample = self._snap_sample(dm.sample, dm.marker_type, snap_mode)
            dm.sample = new_sample
            new_time = new_sample / self._current_fs
            for plot_idx, items in self._marker_items_per_plot.items():
                for item in items:
                    try:
                        item.setValue(new_time)
                    except Exception:
                        pass

        self._update_delineation_from_markers()
        self._update_interval_table()
        self.markers_changed.emit()

    def _snap_sample(self, current_sample: int, marker_type: str, mode: str) -> int:
        if self._current_signal is None:
            return current_sample

        lead_idx = self.lead_combo.currentIndex()
        signal = self._current_signal[:, lead_idx]
        fs = self._current_fs

        search_radius = int(0.05 * fs)
        lo = max(0, current_sample - search_radius)
        hi = min(len(signal), current_sample + search_radius)

        if lo >= hi:
            return current_sample

        segment = signal[lo:hi]

        if mode == "Peak":
            local_max = np.argmax(segment)
            return lo + local_max
        elif mode == "Trough":
            local_min = np.argmin(segment)
            return lo + local_min
        elif mode == "Zero-Crossing":
            for j in range(len(segment) - 1):
                if segment[j] * segment[j + 1] <= 0:
                    return lo + j

        return current_sample

    def _update_interval_table(self):
        if not self._current_delineation or not self._current_delineation.beats:
            self.interval_table.setRowCount(0)
            return

        beat_idx = self.beat_spin.value()
        if beat_idx >= len(self._current_delineation.beats):
            return

        beat = self._current_delineation.beats[beat_idx]
        fs = self._current_fs

        intervals: list[tuple[str, float, float, str, float]] = []

        def add_interval(
            name: str,
            start: MarkerPoint | None,
            end: MarkerPoint | None,
            reference: str,
            segment: str,
        ) -> None:
            if start is None or end is None or end.sample <= start.sample:
                return
            confidence = min(
                start.confidence,
                end.confidence,
                beat.segment_confidence(segment),
            )
            intervals.append(
                (
                    name,
                    (end.sample - start.sample) / fs * 1000.0,
                    float(
                        np.hypot(
                            start.uncertainty_ms,
                            end.uncertainty_ms,
                        )
                    ),
                    reference,
                    confidence,
                )
            )

        add_interval("PR Interval", beat.p_onset, beat.r_onset, "120-200", "pr")
        add_interval("QRS Duration", beat.r_onset, beat.r_offset, "60-120", "qrs")
        add_interval("P Duration", beat.p_onset, beat.p_offset, "80-120", "p")
        add_interval("QT Interval", beat.r_onset, beat.t_offset, "rate-dependent", "qrst")
        add_interval("JT Interval", beat.r_offset, beat.t_offset, "rate-dependent", "t")
        add_interval("T Duration", beat.t_onset, beat.t_offset, "context-dependent", "t")
        add_interval(
            "ST Segment",
            beat.j_point or beat.r_offset,
            beat.t_onset,
            "context-dependent",
            "st",
        )

        self.interval_table.setRowCount(len(intervals))
        for i, (
            name,
            val_ms,
            uncertainty_ms,
            reference,
            confidence,
        ) in enumerate(intervals):
            self.interval_table.setItem(i, 0, QTableWidgetItem(name))
            self.interval_table.setItem(i, 1, QTableWidgetItem(f"{val_ms:.1f}"))
            self.interval_table.setItem(
                i,
                2,
                QTableWidgetItem(f"{uncertainty_ms:.1f}"),
            )
            self.interval_table.setItem(
                i,
                4,
                QTableWidgetItem(f"{confidence:.0%}"),
            )

            normal_parts = reference.split("-")
            if len(normal_parts) == 2:
                try:
                    lo_n, hi_n = float(normal_parts[0]), float(normal_parts[1])
                    lower = val_ms - uncertainty_ms
                    upper = val_ms + uncertainty_ms
                    if confidence < 0.35:
                        self.interval_table.setItem(
                            i,
                            3,
                            QTableWidgetItem(f"{reference} [?]"),
                        )
                        self.interval_table.item(i, 1).setBackground(
                            QColor(238, 226, 184)
                        )
                        self.interval_table.item(i, 1).setToolTip(
                            "Below the biomarker measurement confidence "
                            "threshold; manual review required."
                        )
                    elif upper < lo_n or lower > hi_n:
                        self.interval_table.setItem(
                            i,
                            3,
                            QTableWidgetItem(f"{reference} [!]"),
                        )
                        self.interval_table.item(i, 1).setBackground(
                            QColor(255, 200, 200)
                        )
                    elif lower >= lo_n and upper <= hi_n:
                        self.interval_table.setItem(
                            i,
                            3,
                            QTableWidgetItem(reference),
                        )
                        self.interval_table.item(i, 1).setBackground(
                            QColor(200, 255, 200)
                        )
                    else:
                        self.interval_table.setItem(
                            i,
                            3,
                            QTableWidgetItem(f"{reference} [?]"),
                        )
                        self.interval_table.item(i, 1).setBackground(
                            QColor(238, 226, 184)
                        )
                        self.interval_table.item(i, 1).setToolTip(
                            "The uncertainty interval overlaps a reference "
                            "boundary; no categorical claim is made."
                        )
                except ValueError:
                    self.interval_table.setItem(
                        i,
                        3,
                        QTableWidgetItem(reference),
                    )
            else:
                self.interval_table.setItem(
                    i,
                    3,
                    QTableWidgetItem(reference),
                )

    def get_modified_delineation(self) -> RecordDelineation | None:
        return self._current_delineation

    def highlight_sample(self, sample_idx: int):
        if self._current_signal is None or not self._plots:
            return

        time_s = sample_idx / self._current_fs

        if not self._crosshair_lines:
            for plot in self._plots:
                line = pg.InfiniteLine(
                    pos=time_s, angle=90,
                    pen=pg.mkPen("y", width=2, style=Qt.PenStyle.DashLine),
                )
                plot.addItem(line)
                self._crosshair_lines.append(line)
        else:
            for line in self._crosshair_lines:
                line.setValue(time_s)

    def _on_beat_changed(self, value):
        self._clear_markers()
        self._draw_ecg()

    def _on_show_markers(self, state):
        self._clear_markers()
        self._draw_ecg()

    def _on_show_regions(self, state):
        self._clear_markers()
        self._draw_ecg()

    def set_caliper_manager(self, manager: CaliperManager):
        self._caliper_manager = manager

    def _on_caliper_toggle(self, checked: bool):
        if not checked:
            return
        if self._caliper_manager is None or self._current_signal is None or not self._plots:
            self.caliper_btn.setChecked(False)
            return
        self._start_caliper_placement()

    def _start_caliper_placement(self):
        if self._caliper_manager is None or self._current_signal is None or not self._plots:
            if hasattr(self, "caliper_btn"):
                self.caliper_btn.setChecked(False)
            return

        fs = self._current_fs
        lead_idx = self.lead_combo.currentIndex()
        plot = self._plots[lead_idx] if lead_idx < len(self._plots) else self._plots[0]

        # Place new caliper at the current view center, 200 ms wide
        view_range = plot.viewRange()[0]
        center = (view_range[0] + view_range[1]) / 2
        start_s = max(0.0, center - 0.1)
        end_s = start_s + 0.2

        measurement = self._caliper_manager.add_measurement(
            start_sample=int(start_s * fs),
            end_sample=int(end_s * fs),
            lead_index=lead_idx,
        )

        region = self._caliper_manager.create_region_item(measurement)
        plot.addItem(region)
        self._caliper_regions.append(region)

        region.sigRegionChanged.connect(
            lambda: self._on_caliper_region_changed(measurement, region)
        )

        if hasattr(self, "caliper_btn"):
            self.caliper_btn.setChecked(False)
        self.caliper_updated.emit()

    def _redraw_calipers(self):
        """Re-add caliper regions to the plot after a full ECG redraw."""
        if not self._caliper_regions or not self._plots:
            return
        lead_idx = self.lead_combo.currentIndex()
        plot = self._plots[lead_idx] if lead_idx < len(self._plots) else self._plots[0]
        for region in self._caliper_regions:
            plot.addItem(region)

    def _on_caliper_region_changed(self, measurement, region: pg.LinearRegionItem):
        values = region.getRegion()
        measurement.start_sample = int(values[0] * self._current_fs)
        measurement.end_sample = int(values[1] * self._current_fs)
        self.caliper_updated.emit()

    def clear_calipers(self):
        for region in self._caliper_regions:
            try:
                for plot in self._plots:
                    try:
                        plot.removeItem(region)
                    except Exception:
                        pass
            except Exception:
                pass
        self._caliper_regions.clear()
        if self._caliper_manager:
            self._caliper_manager.clear()
        self.caliper_updated.emit()
