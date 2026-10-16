from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox,
    QSplitter, QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

from ...config import AppConfig
from ...core.data_loader import LEAD_NAMES_12


class ECGWidget(QWidget):
    sample_selected = pyqtSignal(int)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._current_signal: np.ndarray | None = None
        self._current_fs: int = 500
        self._current_delineation = None
        self._vline: pg.InfiniteLine | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Display Beat:"))
        self.beat_spin = QSpinBox()
        self.beat_spin.setRange(0, 100)
        self.beat_spin.setValue(0)
        self.beat_spin.valueChanged.connect(self._on_beat_changed)
        ctrl_layout.addWidget(self.beat_spin)

        self.show_delineation_cb = QCheckBox("Show Delineation")
        self.show_delineation_cb.setChecked(True)
        self.show_delineation_cb.stateChanged.connect(self._on_beat_changed)
        ctrl_layout.addWidget(self.show_delineation_cb)

        self.show_vcg_ref_cb = QCheckBox("VCG Cross-Ref")
        self.show_vcg_ref_cb.setChecked(True)
        ctrl_layout.addWidget(self.show_vcg_ref_cb)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("w")
        layout.addWidget(self.plot_widget)

        self._plots = []
        self._curves = []
        self._delineation_items = []
        self._signal_proxy = None

    def display_ecg(self, signal: np.ndarray, fs: int, delineation=None):
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
        self._delineation_items = []

        t = np.arange(signal.shape[0]) / fs
        spacing = 1.5

        for i in range(n_leads):
            plot = self.plot_widget.addPlot(row=i, col=0)
            lead_data = signal[:, i]

            offset = -i * spacing * np.max(np.abs(lead_data)) * 2 if i > 0 else 0
            display_data = lead_data + offset

            curve = plot.plot(t, display_data, pen=pg.mkPen("b", width=1))
            self._curves.append(curve)

            text = pg.TextItem(LEAD_NAMES_12[i], color="k")
            text.setPos(t[0], display_data[0] if len(display_data) > 0 else 0)
            plot.addItem(text)

            plot.hideAxis("bottom")
            plot.hideAxis("left")
            plot.setMouseEnabled(False, False)

            vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("y", width=2, style=Qt.PenStyle.DashLine))
            vline.setVisible(False)
            plot.addItem(vline)
            self._delineation_items.append(vline)

            if i == 0:
                self._vline = vline

            self._plots.append(plot)

        if self._plots:
            self._plots[-1].showAxis("bottom")
            self._plots[-1].setLabel("bottom", "Time", units="s")

        self._setup_crosshair()

        if self._current_delineation and self.show_delineation_cb.isChecked():
            self._draw_delineation_markers()

    def _setup_crosshair(self):
        if not self._plots:
            return

        proxy = pg.SignalProxy(
            self._plots[0].scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )
        self._signal_proxy = proxy

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if not self._plots or self._current_signal is None:
            return

        plot = self._plots[0]
        mouse_point = plot.vb.mapSceneToView(pos)
        time_s = mouse_point.x()

        if time_s < 0 or time_s > len(self._current_signal) / self._current_fs:
            for vline in self._delineation_items:
                vline.setVisible(False)
            return

        sample_idx = int(time_s * self._current_fs)
        sample_idx = max(0, min(sample_idx, len(self._current_signal) - 1))

        if self.show_vcg_ref_cb.isChecked():
            for vline in self._delineation_items:
                vline.setValue(time_s)
                vline.setVisible(True)

            self.sample_selected.emit(sample_idx)

    def _draw_delineation_markers(self):
        if not self._current_delineation or not self._current_delineation.beats:
            return

        beat_idx = self.beat_spin.value()
        if beat_idx >= len(self._current_delineation.beats):
            return

        beat = self._current_delineation.beats[beat_idx]
        fs = self._current_fs

        markers = {
            "p_onset": ("g", "Pon"),
            "p_peak": ("g", "P"),
            "p_offset": ("g", "Poff"),
            "r_onset": ("r", "Ron"),
            "r_peak": ("r", "R"),
            "r_offset": ("r", "Roff"),
            "q_peak": ("m", "Q"),
            "s_peak": ("m", "S"),
            "t_onset": ("c", "Ton"),
            "t_peak": ("c", "T"),
            "t_offset": ("c", "Toff"),
        }

        for attr, (color, label) in markers.items():
            marker = getattr(beat, attr, None)
            if marker is None:
                continue

            sample = marker.sample
            time_s = sample / fs

            for plot in self._plots:
                line = pg.InfiniteLine(
                    pos=time_s, angle=90,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                )
                plot.addItem(line)

    def _on_beat_changed(self, value=None):
        self._draw_ecg()

    def highlight_sample(self, sample_idx: int):
        if self._current_signal is None or not self._delineation_items:
            return

        time_s = sample_idx / self._current_fs
        for vline in self._delineation_items:
            vline.setValue(time_s)
            vline.setVisible(True)
