from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg


@dataclass
class CaliperMeasurement:
    name: str
    start_sample: int
    end_sample: int
    fs: int
    lead_index: int = 0

    @property
    def duration_ms(self) -> float:
        return (self.end_sample - self.start_sample) / self.fs * 1000

    @property
    def start_time_s(self) -> float:
        return self.start_sample / self.fs

    @property
    def end_time_s(self) -> float:
        return self.end_sample / self.fs

    @property
    def interval_samples(self) -> int:
        return self.end_sample - self.start_sample


class CaliperManager:
    def __init__(self, fs: int = 500):
        self._fs = fs
        self._measurements: list[CaliperMeasurement] = []
        self._active_regions: dict[str, pg.LinearRegionItem] = {}
        self._counter = 0

    @property
    def fs(self) -> int:
        return self._fs

    @fs.setter
    def fs(self, value: int):
        self._fs = value

    def add_measurement(self, start_sample: int, end_sample: int,
                        lead_index: int = 0, name: Optional[str] = None) -> CaliperMeasurement:
        self._counter += 1
        if name is None:
            name = f"Measurement {self._counter}"
        m = CaliperMeasurement(
            name=name,
            start_sample=start_sample,
            end_sample=end_sample,
            fs=self._fs,
            lead_index=lead_index,
        )
        self._measurements.append(m)
        return m

    def remove_measurement(self, index: int):
        if 0 <= index < len(self._measurements):
            name = self._measurements[index].name
            self._measurements.pop(index)
            if name in self._active_regions:
                del self._active_regions[name]

    def clear(self):
        self._measurements.clear()
        self._active_regions.clear()
        self._counter = 0

    @property
    def measurements(self) -> list[CaliperMeasurement]:
        return self._measurements

    def create_region_item(self, measurement: CaliperMeasurement) -> pg.LinearRegionItem:
        region = pg.LinearRegionItem(
            values=[measurement.start_time_s, measurement.end_time_s],
            movable=True,
            brush=pg.mkBrush(255, 255, 0, 40),
        )
        region.sigRegionChanged.connect(
            lambda: self._on_region_changed(measurement.name, region)
        )
        self._active_regions[measurement.name] = region
        return region

    def _on_region_changed(self, name: str, region: pg.LinearRegionItem):
        for m in self._measurements:
            if m.name == name:
                values = region.getRegion()
                m.start_sample = int(values[0] * self._fs)
                m.end_sample = int(values[1] * self._fs)
                break


class CaliperWidget(QWidget):
    """
    Displays caliper measurements placed on the ECG Annotation tab.

    'Add Caliper' places a new draggable region on the ECG plot.
    'Clear All'  removes all regions from the plot and this table.
    The table updates automatically whenever a caliper is added, dragged,
    or removed.
    """
    add_caliper_requested = pyqtSignal()
    measurements_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._manager: Optional[CaliperManager] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "Calipers are placed on the ECG Annotation tab. "
            "Click 'Add Caliper' to insert a new draggable region, "
            "then drag its edges to measure any interval."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-style: italic;")
        layout.addWidget(info)

        ctrl = QHBoxLayout()
        self.add_btn = QPushButton("Add Caliper")
        self.add_btn.setToolTip("Place a new caliper region on the ECG plot")
        self.add_btn.clicked.connect(self._on_add_caliper)
        ctrl.addWidget(self.add_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setToolTip("Remove all caliper regions from the ECG plot")
        self.clear_btn.clicked.connect(self._on_clear)
        ctrl.addWidget(self.clear_btn)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.results_table = pg.TableWidget(sortable=True)
        layout.addWidget(self.results_table)

    def set_manager(self, manager: CaliperManager):
        self._manager = manager
        self._update_table()

    def _on_add_caliper(self):
        self.add_caliper_requested.emit()

    def _on_clear(self):
        if self._manager:
            self._manager.clear()
        self._update_table()
        self.measurements_cleared.emit()

    def _update_table(self):
        if not self._manager:
            return
        rows = []
        for m in self._manager.measurements:
            rows.append({
                "Name": m.name,
                "Start (s)": f"{m.start_time_s:.3f}",
                "End (s)": f"{m.end_time_s:.3f}",
                "Duration (ms)": f"{m.duration_ms:.1f}",
                "Samples": m.interval_samples,
            })
        self.results_table.setData(rows)
