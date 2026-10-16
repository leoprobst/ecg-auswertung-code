from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QCheckBox, QComboBox, QListWidget, QListWidgetItem,
    QProgressBar, QFileDialog, QMessageBox, QGridLayout, QFrame,
    QSplitter, QSpinBox, QLineEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush
import pyqtgraph as pg

from ...config import AppConfig
from ...core.analysis.results import AnalysisRun
from ...core.analysis.threshold_sweep import SweepResult


TRAFFIC_LIGHT_COLORS = {
    "normal": QColor(76, 175, 80),
    "borderline": QColor(255, 193, 7),
    "abnormal": QColor(244, 67, 54),
    "unknown": QColor(189, 189, 189),
}


def _traffic_light_style(color: QColor) -> str:
    return (
        f"background-color: rgb({color.red()},{color.green()},{color.blue()}); "
        f"border-radius: 8px; min-width: 16px; max-width: 16px; min-height: 16px; max-height: 16px;"
    )


def _classify_biomarker(value: Optional[float], normal_range: tuple) -> str:
    if value is None or not np.isfinite(value) or normal_range[0] is None:
        return "unknown"
    lo, hi = normal_range
    if lo is not None and hi is not None:
        margin = (hi - lo) * 0.1
        if value < lo - margin or value > hi + margin:
            return "abnormal"
        elif value < lo or value > hi:
            return "borderline"
        else:
            return "normal"
    return "unknown"


class TrafficLightWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.biomarker_name = ""
        self.category = ""
        self.status = "unknown"
        self.setStyleSheet("border: 1px solid #ccc; border-radius: 4px; padding: 2px;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.indicator = QLabel()
        self.indicator.setFixedSize(16, 16)
        self.indicator.setStyleSheet(_traffic_light_style(TRAFFIC_LIGHT_COLORS["unknown"]))
        layout.addWidget(self.indicator)

        self.name_label = QLabel("")
        self.name_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.name_label)

        self.value_label = QLabel("")
        self.value_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        layout.addWidget(self.value_label)

        self.range_label = QLabel("")
        self.range_label.setStyleSheet("font-size: 8px; color: #666;")
        layout.addWidget(self.range_label)

        layout.addStretch()

    def set_biomarker(self, name: str, value: Optional[float], unit: str,
                      normal_range: tuple, category: str = ""):
        self.biomarker_name = name
        self.category = category
        self.name_label.setText(name)
        val_str = f"{value:.2f}" if value is not None else "N/A"
        self.value_label.setText(f"{val_str} {unit}")

        if normal_range[0] is not None and normal_range[1] is not None:
            self.range_label.setText(f"[{normal_range[0]:.0f}-{normal_range[1]:.0f}]")
        else:
            self.range_label.setText("")

        classification = _classify_biomarker(value, normal_range)
        self.status = classification
        color = TRAFFIC_LIGHT_COLORS.get(classification, TRAFFIC_LIGHT_COLORS["unknown"])
        self.indicator.setStyleSheet(_traffic_light_style(color))


class BiomarkerWidget(QWidget):
    DASHBOARD_LIMIT = 48
    TABLE_ROW_LIMIT = 5000

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._biomarker_data = []
        self._traffic_lights: list[TrafficLightWidget] = []
        self._filter_refreshing = False
        self._setup_ui()
        self.display_biomarkers([])

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Age:"))
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 120)
        self.age_spin.setValue(50)
        self.age_spin.setSpecialValueText("Unknown")
        self.age_spin.valueChanged.connect(self._apply_filters)
        ctrl_layout.addWidget(self.age_spin)

        ctrl_layout.addWidget(QLabel("Sex:"))
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Unknown", "Male", "Female"])
        self.sex_combo.currentTextChanged.connect(self._apply_filters)
        ctrl_layout.addWidget(self.sex_combo)

        ctrl_layout.addWidget(QLabel("Filter:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(["All"])
        self.category_combo.setMaxVisibleItems(25)
        self.category_combo.currentTextChanged.connect(self._apply_filters)
        ctrl_layout.addWidget(self.category_combo)

        ctrl_layout.addWidget(QLabel("Family:"))
        self.family_combo = QComboBox()
        self.family_combo.addItems(["All"])
        self.family_combo.setMaxVisibleItems(25)
        self.family_combo.currentTextChanged.connect(self._apply_filters)
        ctrl_layout.addWidget(self.family_combo)

        ctrl_layout.addWidget(QLabel("Status:"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(["All", "implemented", "candidate", "normal", "borderline", "abnormal", "unknown"])
        self.status_combo.setMaxVisibleItems(25)
        self.status_combo.currentTextChanged.connect(self._apply_filters)
        ctrl_layout.addWidget(self.status_combo)

        self.computed_only_check = QCheckBox("Computed only")
        self.computed_only_check.setChecked(False)
        self.computed_only_check.stateChanged.connect(self._apply_filters)
        ctrl_layout.addWidget(self.computed_only_check)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search biomarker, alias, formula, family...")
        self.search_edit.textChanged.connect(self._apply_filters)
        ctrl_layout.addWidget(self.search_edit, 2)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        dashboard_frame = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_frame)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Biomarker Dashboard")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        dashboard_layout.addWidget(header)

        self._dashboard_grid = QGridLayout()
        self._dashboard_grid.setSpacing(4)
        dashboard_layout.addLayout(self._dashboard_grid)
        dashboard_layout.addStretch()

        splitter.addWidget(dashboard_frame)

        detail_frame = QWidget()
        detail_layout = QVBoxLayout(detail_frame)
        detail_layout.setContentsMargins(0, 0, 0, 0)

        summary_group = QGroupBox("Biomarker Summary Table")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(12)
        self.summary_table.setHorizontalHeaderLabels([
            "Biomarker", "Family", "Category", "Value", "Unit", "Confidence",
            "Reference/Range", "Status", "Pathology Rule", "Description", "Formula",
            "Qualität / Abstention"
        ])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.summary_table.horizontalHeader().setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        self.summary_table.setSortingEnabled(True)
        summary_layout.addWidget(self.summary_table)
        self.summary_count_label = QLabel("")
        self.summary_count_label.setStyleSheet("color: #666; font-size: 10px;")
        summary_layout.addWidget(self.summary_count_label)
        detail_layout.addWidget(summary_group)

        self.detail_plot = pg.GraphicsLayoutWidget()
        self.detail_plot.setBackground("w")
        detail_layout.addWidget(self.detail_plot)

        splitter.addWidget(detail_frame)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    def display_biomarkers(self, biomarker_results: list):
        self._biomarker_data = list(biomarker_results or [])
        self._populate_filter_combos()
        self._apply_filters()

    def _on_category_filter(self, category: str):
        self._apply_filters()

    def _sex_value(self) -> Optional[str]:
        sex = self.sex_combo.currentText()
        return None if sex == "Unknown" else sex[0]

    def _normal_range_for(self, name: str, fallback: tuple) -> tuple:
        from ...core.biomarkers.base import BiomarkerRegistry

        try:
            biomarker = BiomarkerRegistry.get(name)
            return biomarker.normal_range(
                self.age_spin.value() if self.age_spin.value() > 0 else None,
                self._sex_value(),
            )
        except Exception:
            return fallback

    def _build_records(self) -> list[dict]:
        from ...core.biomarkers.base import BiomarkerRegistry

        result_by_name = {result.name: result for result in self._biomarker_data}
        consumed_results: set[str] = set()
        records: list[dict] = []

        for definition in BiomarkerRegistry.definitions():
            result = result_by_name.get(definition.name)
            if result is None:
                for alias in definition.aliases:
                    if alias in result_by_name:
                        result = result_by_name[alias]
                        break

            computed = result is not None
            if computed:
                consumed_results.add(result.name)
                value = result.value
                unit = result.unit or definition.unit
                category = result.category or definition.category
                confidence = result.confidence
                name = result.name
                normal_range = self._normal_range_for(result.name, definition.normal_range)
                classification = _classify_biomarker(value, normal_range)
                status = classification
            else:
                value = None
                unit = definition.unit
                category = definition.category
                confidence = 0.0
                name = definition.name
                normal_range = definition.normal_range
                status = definition.implementation_status

            records.append({
                "name": name,
                "family": definition.family,
                "category": category,
                "value": value,
                "unit": unit,
                "confidence": confidence,
                "normal_range": normal_range,
                "status": status,
                "description": definition.description,
                "formula": definition.formula,
                "pathology_rule": definition.pathological_note,
                "computed": computed,
                "computation_notes": (result.computation_notes or "") if computed else "Nicht berechnet",
                "aliases": definition.aliases,
                "source_index": definition.source_index,
            })

        for result in self._biomarker_data:
            if result.name in consumed_results:
                continue
            normal_range = self._normal_range_for(result.name, (None, None))
            records.append({
                "name": result.name,
                "family": result.family or result.category,
                "category": result.category,
                "value": result.value,
                "unit": result.unit,
                "confidence": result.confidence,
                "normal_range": normal_range,
                "status": _classify_biomarker(result.value, normal_range),
                "description": result.description,
                "formula": result.formula,
                "pathology_rule": result.pathological_note or "Data-driven threshold in pathology-specific analysis.",
                "computed": True,
                "computation_notes": result.computation_notes or "",
                "aliases": [],
                "source_index": None,
            })

        return records

    def _populate_filter_combos(self):
        records = self._build_records()
        self._filter_refreshing = True
        for combo, values in [
            (self.category_combo, sorted({record["category"] for record in records if record["category"]})),
            (self.family_combo, sorted({record["family"] for record in records if record["family"]})),
            (self.status_combo, sorted({record["status"] for record in records if record["status"]})),
        ]:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All")
            combo.addItems(values)
            if current and current in ["All", *values]:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        self._filter_refreshing = False

    def _record_matches(self, record: dict) -> bool:
        if self.computed_only_check.isChecked() and not record["computed"]:
            return False

        category = self.category_combo.currentText()
        if category != "All" and record["category"] != category:
            return False

        family = self.family_combo.currentText()
        if family != "All" and record["family"] != family:
            return False

        status = self.status_combo.currentText()
        if status != "All" and record["status"] != status:
            return False

        query = self.search_edit.text().strip().lower()
        if query:
            haystack = " ".join([
                str(record.get("name", "")),
                str(record.get("family", "")),
                str(record.get("category", "")),
                str(record.get("status", "")),
                str(record.get("description", "")),
                str(record.get("formula", "")),
                str(record.get("computation_notes", "")),
                " ".join(record.get("aliases", [])),
            ]).lower()
            if query not in haystack:
                return False
        return True

    def _range_text(self, normal_range: tuple) -> str:
        lo, hi = normal_range
        if lo is None and hi is None:
            return "Data-driven cutoff"
        if lo is None:
            return f"<= {hi:g}"
        if hi is None:
            return f">= {lo:g}"
        return f"{lo:g}-{hi:g}"

    def _clear_dashboard(self):
        while self._dashboard_grid.count():
            item = self._dashboard_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._traffic_lights.clear()

    def _fill_dashboard(self, records: list[dict]):
        self._clear_dashboard()
        computed_records = [record for record in records if record["computed"]][:self.DASHBOARD_LIMIT]
        if not computed_records:
            label = QLabel("No computed biomarkers in the current filter. Disable 'Computed only' to browse candidate definitions.")
            label.setWordWrap(True)
            label.setStyleSheet("color: #666; font-size: 11px;")
            self._dashboard_grid.addWidget(label, 0, 0)
            return

        row = 0
        col = 0
        max_cols = 2
        for record in computed_records:
            tl = TrafficLightWidget()
            tl.set_biomarker(
                name=record["name"],
                value=record["value"],
                unit=record["unit"],
                normal_range=record["normal_range"],
                category=record["category"],
            )
            self._dashboard_grid.addWidget(tl, row, col)
            tl.setToolTip(record.get("computation_notes", ""))
            self._traffic_lights.append(tl)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        if len([record for record in records if record["computed"]]) > self.DASHBOARD_LIMIT:
            more = QLabel(f"Showing first {self.DASHBOARD_LIMIT} computed biomarkers. Use the table filters for the full list.")
            more.setStyleSheet("color: #666; font-size: 10px;")
            self._dashboard_grid.addWidget(more, row + 1, 0, 1, max_cols)

    def _fill_table(self, records: list[dict]):
        visible_records = records[:self.TABLE_ROW_LIMIT]
        if len(records) > len(visible_records):
            self.summary_count_label.setText(
                f"Showing {len(visible_records):,}/{len(records):,} biomarkers; refine filters"
            )
        else:
            self.summary_count_label.setText(f"{len(records):,} biomarkers")

        self.summary_table.setSortingEnabled(False)
        self.summary_table.setRowCount(len(visible_records))

        for row, record in enumerate(visible_records):
            value = record["value"]
            value_text = f"{value:.4f}" if value is not None else "N/A"
            values = [
                record["name"],
                record["family"],
                record["category"],
                value_text,
                record["unit"],
                f"{record['confidence']:.2f}" if record["computed"] else "-",
                self._range_text(record["normal_range"]),
                record["status"].upper(),
                record["pathology_rule"],
                record["description"],
                record["formula"],
                record.get("computation_notes", ""),
            ]
            for col, value_text in enumerate(values):
                item = QTableWidgetItem(value_text)
                item.setToolTip(value_text)
                if col == 7:
                    color = TRAFFIC_LIGHT_COLORS.get(record["status"], TRAFFIC_LIGHT_COLORS["unknown"])
                    item.setBackground(QBrush(color))
                    item.setForeground(QBrush(QColor(255, 255, 255) if record["status"] in ("abnormal", "normal") else QColor(0, 0, 0)))
                self.summary_table.setItem(row, col, item)

        self.summary_table.setSortingEnabled(True)
        self.summary_table.resizeColumnsToContents()
        self.summary_table.setColumnWidth(11, 320)
        self.summary_table.horizontalHeader().setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)

    def _apply_filters(self, *_args):
        if self._filter_refreshing:
            return
        records = [record for record in self._build_records() if self._record_matches(record)]
        records.sort(key=lambda record: (record["source_index"] is None, record["source_index"] or 10**9, record["name"]))
        self._fill_dashboard(records)
        self._fill_table(records)
