from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QComboBox, QSplitter, QLineEdit, QProgressBar, QTextEdit,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush
import pyqtgraph as pg

from ...config import AppConfig
from ...core.diagnostic import DiagnosticEngine, DiagnosticReport, PathologyScore
from ...core.diagnostic import LEGACY_DIAGNOSTIC_NOTICE
from ...core.biomarkers.base import BiomarkerResult, BiomarkerRegistry
from ...core.analysis.results import AnalysisRun


TRAFFIC_COLORS = {
    "normal": QColor(76, 175, 80),
    "borderline": QColor(255, 193, 7),
    "abnormal": QColor(244, 67, 54),
    "unknown": QColor(189, 189, 189),
}


class DiagnosticWidget(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._diagnostic_engine = DiagnosticEngine()
        self._current_report: Optional[DiagnosticReport] = None
        self._biomarker_results: list[BiomarkerResult] = []
        self._age: Optional[int] = None
        self._sex: Optional[str] = None
        self._setup_ui()

    @property
    def diagnostic_engine(self) -> DiagnosticEngine:
        return self._diagnostic_engine

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Age:"))
        self.age_spin = QLineEdit()
        self.age_spin.setPlaceholderText("Age")
        self.age_spin.setMaximumWidth(60)
        ctrl.addWidget(self.age_spin)

        ctrl.addWidget(QLabel("Sex:"))
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Unknown", "M", "F"])
        ctrl.addWidget(self.sex_combo)

        self.run_diag_btn = QPushButton("Run Diagnosis")
        self.run_diag_btn.clicked.connect(self._run_diagnosis)
        ctrl.addWidget(self.run_diag_btn)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        bm_frame = QWidget()
        bm_layout = QVBoxLayout(bm_frame)
        bm_layout.setContentsMargins(0, 0, 0, 0)

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter biomarkers...")
        self.search_edit.textChanged.connect(self._filter_biomarker_table)
        search_layout.addWidget(self.search_edit)

        self.cat_combo = QComboBox()
        self.cat_combo.addItems(["All", "ecg", "vcg", "loop"])
        self.cat_combo.currentTextChanged.connect(self._filter_biomarker_table)
        search_layout.addWidget(self.cat_combo)
        bm_layout.addLayout(search_layout)

        self.biomarker_table = QTableWidget()
        self.biomarker_table.setColumnCount(8)
        self.biomarker_table.setHorizontalHeaderLabels([
            "Biomarker", "Value", "Unit", "Normal Range", "Status",
            "Confidence", "Exceeded Pathology Thresholds", "Category"
        ])
        self.biomarker_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.biomarker_table.setSortingEnabled(True)
        self.biomarker_table.cellClicked.connect(self._on_biomarker_clicked)
        bm_layout.addWidget(self.biomarker_table)

        splitter.addWidget(bm_frame)

        diag_frame = QWidget()
        diag_layout = QVBoxLayout(diag_frame)
        diag_layout.setContentsMargins(0, 0, 0, 0)

        diag_header = QLabel("Auto-Diagnostic Results")
        diag_header.setStyleSheet("font-weight: bold; font-size: 13px;")
        diag_layout.addWidget(diag_header)

        self.diag_table = QTableWidget()
        self.diag_table.setColumnCount(6)
        self.diag_table.setHorizontalHeaderLabels([
            "Pathology", "Probability", "Confidence", "Superclass",
            "Best AUC", "Evidence Biomarkers"
        ])
        self.diag_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.diag_table.setSortingEnabled(True)
        diag_layout.addWidget(self.diag_table)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(120)
        diag_layout.addWidget(self.summary_text)

        splitter.addWidget(diag_frame)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        violation_group = QGroupBox("Threshold Violations → Pathology Mapping")
        violation_layout = QVBoxLayout(violation_group)
        self.violation_table = QTableWidget()
        self.violation_table.setColumnCount(6)
        self.violation_table.setHorizontalHeaderLabels([
            "Biomarker", "Value", "Pathology", "Threshold",
            "AUC", "PPV"
        ])
        self.violation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        violation_layout.addWidget(self.violation_table)
        layout.addWidget(violation_group)

        # Retire the complete legacy claim surface, including normal-range
        # colours and PPV tables; retaining a visible notice avoids silent loss.
        splitter.hide()
        violation_group.hide()
        self.run_diag_btn.setEnabled(False)
        self.run_diag_btn.setText("Historischer Pfad deaktiviert")
        self.age_spin.setEnabled(False)
        self.sex_combo.setEnabled(False)
        notice = QLabel(LEGACY_DIAGNOSTIC_NOTICE)
        notice.setWordWrap(True)
        layout.addWidget(notice)

    def set_biomarker_results(self, results: list[BiomarkerResult],
                              age: Optional[int] = None, sex: Optional[str] = None):
        self._biomarker_results = results
        self._age = age
        self._sex = sex

        if age is not None:
            self.age_spin.setText(str(age))
        if sex:
            idx = self.sex_combo.findText(sex)
            if idx >= 0:
                self.sex_combo.setCurrentIndex(idx)

        self._run_diagnosis()

    def learn_from_analysis(self, run: AnalysisRun):
        self._diagnostic_engine.learn_from_analysis(run, min_auc=0.6)

    def _populate_biomarker_table(self):
        age = self._get_age()
        sex = self._get_sex()
        filter_text = self.search_edit.text().lower()
        filter_cat = self.cat_combo.currentText()

        results = self._biomarker_results
        self.biomarker_table.setRowCount(len(results))

        bm_pathology_map = self._diagnostic_engine.get_biomarker_pathology_map()

        for i, br in enumerate(results):
            if filter_cat != "All" and br.category != filter_cat:
                self.biomarker_table.setRowHidden(i, True)
                continue

            if filter_text and filter_text not in br.name.lower() and filter_text not in br.description.lower():
                self.biomarker_table.setRowHidden(i, True)
                continue

            self.biomarker_table.setRowHidden(i, False)

            self.biomarker_table.setItem(i, 0, QTableWidgetItem(br.name))

            val_str = f"{br.value:.4f}" if br.value is not None else "N/A"
            self.biomarker_table.setItem(i, 1, QTableWidgetItem(val_str))
            self.biomarker_table.setItem(i, 2, QTableWidgetItem(br.unit))

            try:
                bm_obj = BiomarkerRegistry.get(br.name)
                nr = bm_obj.normal_range(age, sex)
                range_str = f"{nr[0]:.0f}-{nr[1]:.0f}" if nr[0] is not None else "N/A"
                status = self._classify_value(br.value, nr)
            except Exception:
                range_str = "N/A"
                status = "unknown"

            self.biomarker_table.setItem(i, 3, QTableWidgetItem(range_str))

            status_item = QTableWidgetItem(status.upper())
            color = TRAFFIC_COLORS.get(status, TRAFFIC_COLORS["unknown"])
            status_item.setBackground(QBrush(color))
            status_item.setForeground(QBrush(QColor(255, 255, 255) if status in ("abnormal", "normal") else QColor(0, 0, 0)))
            self.biomarker_table.setItem(i, 4, status_item)

            self.biomarker_table.setItem(i, 5, QTableWidgetItem(f"{br.confidence:.2f}"))

            pathology_thresholds = ""
            if br.name in bm_pathology_map:
                entries = bm_pathology_map[br.name]
                for entry in entries[:3]:
                    pname, thresh, auc = entry[:3]
                    direction = entry[3] if len(entry) > 3 else "above"
                    exceeded = (
                        br.value is not None and
                        ((br.value >= thresh) if direction == "above" else (br.value <= thresh))
                    )
                    marker = " [!]" if exceeded else ""
                    op = ">" if direction == "above" else "<"
                    pathology_thresholds += f"{pname}({op}{thresh:.3f}, AUC={auc:.2f}){marker}; "
            self.biomarker_table.setItem(i, 6, QTableWidgetItem(pathology_thresholds.strip("; ")))

            self.biomarker_table.setItem(i, 7, QTableWidgetItem(br.category))

    def _classify_value(self, value: Optional[float], normal_range: tuple) -> str:
        if value is None or normal_range[0] is None:
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

    def _run_diagnosis(self):
        self._current_report = None
        self.diag_table.setRowCount(0)
        self.violation_table.setRowCount(0)
        self.summary_text.setPlainText(LEGACY_DIAGNOSTIC_NOTICE)

    def _populate_diagnostic_table(self):
        # Also reject directly supplied historical reports, not only new fits.
        self.diag_table.setRowCount(0)
        self.summary_text.setPlainText(LEGACY_DIAGNOSTIC_NOTICE)

    def _populate_violation_table(self):
        if self._current_report is None:
            return

        violations = self._current_report.threshold_violations
        self.violation_table.setRowCount(len(violations))

        for i, v in enumerate(violations):
            self.violation_table.setItem(i, 0, QTableWidgetItem(v["biomarker"]))
            self.violation_table.setItem(i, 1, QTableWidgetItem(f"{v['value']:.4f}"))
            self.violation_table.setItem(i, 2, QTableWidgetItem(v["pathology"]))
            self.violation_table.setItem(i, 3, QTableWidgetItem(f"{v['threshold']:.4f}"))
            self.violation_table.setItem(i, 4, QTableWidgetItem(f"{v['auc']:.3f}"))
            self.violation_table.setItem(i, 5, QTableWidgetItem(f"{v['ppv']:.3f}"))

    def _filter_biomarker_table(self):
        self._populate_biomarker_table()

    def _on_biomarker_clicked(self, row, col):
        pass

    def _get_age(self) -> Optional[int]:
        try:
            return int(self.age_spin.text())
        except (ValueError, AttributeError):
            return self._age

    def _get_sex(self) -> Optional[str]:
        val = self.sex_combo.currentText()
        return val if val != "Unknown" else None
