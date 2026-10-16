from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QSplitter, QTextEdit, QGroupBox, QFormLayout,
)
import pyqtgraph as pg

from ...config import AppConfig
from ...core.analysis.results import AnalysisRun
from ...core.analysis.threshold_sweep import SweepResult


def _fmt(value, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        if not np.isfinite(value):
            return "N/A"
    except TypeError:
        return str(value)
    return f"{float(value):.{digits}f}{suffix}"


def _ci(value: float, bounds: Optional[tuple[float, float]], digits: int = 3) -> str:
    if bounds is None:
        return _fmt(value, digits)
    return f"{value:.{digits}f} [{bounds[0]:.{digits}f}-{bounds[1]:.{digits}f}]"


def _result_title(result: SweepResult) -> str:
    op = ">=" if result.threshold_direction == "above" else "<="
    return f"{result.pathology_name} | {result.biomarker_name} {op} {_fmt(result.optimal_threshold, 4)}"


class EvidenceExplorerWidget(QWidget):
    TABLE_ROW_LIMIT = 5000

    """
    Research-first evidence dashboard for candidate VCG biomarkers.

    It keeps the existing AnalysisRun/SweepResult pipeline intact, but presents
    the results as evidence objects: ranking, drilldowns, threshold tradeoffs,
    provenance and publication-style narratives.
    """

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._current_run: Optional[AnalysisRun] = None
        self._current_result: Optional[SweepResult] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Min AUC:"))
        self.min_auc_spin = QDoubleSpinBox()
        self.min_auc_spin.setRange(0.0, 1.0)
        self.min_auc_spin.setSingleStep(0.05)
        self.min_auc_spin.setDecimals(2)
        self.min_auc_spin.setValue(0.60)
        self.min_auc_spin.valueChanged.connect(self._refresh_all)
        controls.addWidget(self.min_auc_spin)

        controls.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Single", "Combined", "All"])
        self.type_combo.setMaxVisibleItems(20)
        self.type_combo.currentTextChanged.connect(self._refresh_all)
        controls.addWidget(self.type_combo)

        controls.addWidget(QLabel("Pathology:"))
        self.pathology_filter = QComboBox()
        self.pathology_filter.setMaxVisibleItems(25)
        self.pathology_filter.currentTextChanged.connect(self._refresh_all)
        controls.addWidget(self.pathology_filter)

        controls.addWidget(QLabel("Biomarker:"))
        self.biomarker_filter = QComboBox()
        self.biomarker_filter.setMaxVisibleItems(25)
        self.biomarker_filter.currentTextChanged.connect(self._refresh_all)
        controls.addWidget(self.biomarker_filter)

        controls.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("biomarker or pathology")
        self.search_edit.textChanged.connect(self._refresh_all)
        controls.addWidget(self.search_edit)
        self.result_count_label = QLabel("")
        self.result_count_label.setStyleSheet("color: #666;")
        controls.addWidget(self.result_count_label)
        layout.addLayout(controls)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._setup_evidence_tab()
        self._setup_pathology_tab()
        self._setup_biomarker_tab()
        self._setup_workbench_tab()
        self._setup_provenance_tab()
        self._setup_narrative_tab()

    def _setup_evidence_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.evidence_table = QTableWidget()
        self.evidence_table.setColumnCount(19)
        self.evidence_table.setHorizontalHeaderLabels([
            "Rank", "Pathology", "Biomarker", "Dir", "Cutoff", "AUC", "AUPRC",
            "Sens", "Spec", "PPV", "NPV", "LR+", "LR-", "Prev", "N",
            "Missing", "q", "Stability", "Status",
        ])
        self.evidence_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.evidence_table.setSortingEnabled(True)
        self.evidence_table.cellClicked.connect(self._on_evidence_clicked)
        splitter.addWidget(self.evidence_table)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        self.detail_summary = QTextEdit()
        self.detail_summary.setReadOnly(True)
        self.detail_summary.setMaximumHeight(170)
        detail_layout.addWidget(self.detail_summary)

        self.detail_plot = pg.GraphicsLayoutWidget()
        self.detail_plot.setBackground("w")
        detail_layout.addWidget(self.detail_plot)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Evidence Explorer")

    def _setup_pathology_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QHBoxLayout()
        top.addWidget(QLabel("Pathology:"))
        self.pathology_combo = QComboBox()
        self.pathology_combo.setMaxVisibleItems(25)
        self.pathology_combo.currentTextChanged.connect(self._populate_pathology_drilldown)
        top.addWidget(self.pathology_combo)
        top.addStretch()
        layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.pathology_table = QTableWidget()
        self.pathology_table.setColumnCount(10)
        self.pathology_table.setHorizontalHeaderLabels([
            "Biomarker", "Dir", "Cutoff", "AUC", "Sens", "Spec", "LR+", "LR-",
            "Stability", "Status",
        ])
        self.pathology_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pathology_table.setSortingEnabled(True)
        self.pathology_table.cellClicked.connect(self._on_pathology_row_clicked)
        splitter.addWidget(self.pathology_table)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.pathology_plot = pg.GraphicsLayoutWidget()
        self.pathology_plot.setBackground("w")
        right_layout.addWidget(self.pathology_plot)
        self.pathology_text = QTextEdit()
        self.pathology_text.setReadOnly(True)
        self.pathology_text.setMaximumHeight(120)
        right_layout.addWidget(self.pathology_text)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Pathology Drilldown")

    def _setup_biomarker_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QHBoxLayout()
        top.addWidget(QLabel("Biomarker:"))
        self.biomarker_combo = QComboBox()
        self.biomarker_combo.setMaxVisibleItems(25)
        self.biomarker_combo.currentTextChanged.connect(self._populate_biomarker_drilldown)
        top.addWidget(self.biomarker_combo)
        top.addStretch()
        layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.biomarker_table = QTableWidget()
        self.biomarker_table.setColumnCount(10)
        self.biomarker_table.setHorizontalHeaderLabels([
            "Pathology", "Dir", "Cutoff", "AUC", "AUPRC", "Sens", "Spec",
            "q", "N", "Status",
        ])
        self.biomarker_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.biomarker_table.setSortingEnabled(True)
        self.biomarker_table.cellClicked.connect(self._on_biomarker_row_clicked)
        splitter.addWidget(self.biomarker_table)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.biomarker_plot = pg.GraphicsLayoutWidget()
        self.biomarker_plot.setBackground("w")
        right_layout.addWidget(self.biomarker_plot)
        self.biomarker_text = QTextEdit()
        self.biomarker_text.setReadOnly(True)
        self.biomarker_text.setMaximumHeight(130)
        right_layout.addWidget(self.biomarker_text)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Biomarker Drilldown")

    def _setup_workbench_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QHBoxLayout()
        top.addWidget(QLabel("Candidate:"))
        self.workbench_result_combo = QComboBox()
        self.workbench_result_combo.setMaxVisibleItems(25)
        self.workbench_result_combo.currentIndexChanged.connect(self._update_workbench)
        top.addWidget(self.workbench_result_combo, stretch=1)

        top.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Youden-balanced", "Screening", "Confirmatory", "Manual"])
        self.preset_combo.currentTextChanged.connect(self._update_workbench)
        top.addWidget(self.preset_combo)
        layout.addLayout(top)

        knobs = QHBoxLayout()
        knobs.addWidget(QLabel("Manual cutoff:"))
        self.manual_threshold_spin = QDoubleSpinBox()
        self.manual_threshold_spin.setRange(-1000000, 1000000)
        self.manual_threshold_spin.setDecimals(4)
        self.manual_threshold_spin.valueChanged.connect(self._update_workbench)
        knobs.addWidget(self.manual_threshold_spin)

        knobs.addWidget(QLabel("Target prevalence:"))
        self.prevalence_spin = QDoubleSpinBox()
        self.prevalence_spin.setRange(0.0, 1.0)
        self.prevalence_spin.setDecimals(3)
        self.prevalence_spin.setSingleStep(0.01)
        self.prevalence_spin.valueChanged.connect(self._update_workbench)
        knobs.addWidget(self.prevalence_spin)
        knobs.addStretch()
        layout.addLayout(knobs)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        metrics_group = QGroupBox("Threshold Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.workbench_metrics = QTableWidget()
        self.workbench_metrics.setColumnCount(2)
        self.workbench_metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.workbench_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        metrics_layout.addWidget(self.workbench_metrics)
        splitter.addWidget(metrics_group)

        matrix_group = QGroupBox("2x2 Matrix")
        matrix_layout = QVBoxLayout(matrix_group)
        self.matrix_table = QTableWidget()
        self.matrix_table.setColumnCount(3)
        self.matrix_table.setRowCount(2)
        self.matrix_table.setHorizontalHeaderLabels(["", "Reference +", "Reference -"])
        self.matrix_table.setVerticalHeaderLabels(["Test +", "Test -"])
        self.matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        matrix_layout.addWidget(self.matrix_table)
        splitter.addWidget(matrix_group)

        narrative_group = QGroupBox("Interpretation")
        narrative_layout = QVBoxLayout(narrative_group)
        self.workbench_narrative = QTextEdit()
        self.workbench_narrative.setReadOnly(True)
        narrative_layout.addWidget(self.workbench_narrative)
        splitter.addWidget(narrative_group)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        layout.addWidget(splitter)

        self.subgroup_group = QGroupBox("Stratifizierte Schwellenwerte (Subgruppen)")
        subgroup_layout = QVBoxLayout(self.subgroup_group)
        self.subgroup_table = QTableWidget()
        self.subgroup_table.setColumnCount(7)
        self.subgroup_table.setHorizontalHeaderLabels(["Stratum", "N", "N+", "AUC", "Schwellenwert", "Sens", "Spez"])
        self.subgroup_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.subgroup_table.setSortingEnabled(True)
        subgroup_layout.addWidget(self.subgroup_table)
        self.subgroup_group.setVisible(False)
        layout.addWidget(self.subgroup_group)

        self.tabs.addTab(tab, "Threshold Workbench")

    def _setup_provenance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.provenance_text = QTextEdit()
        self.provenance_text.setReadOnly(True)
        layout.addWidget(self.provenance_text)
        self.tabs.addTab(tab, "Cohort & Provenance")

    def _setup_narrative_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.narrative_scope_combo = QComboBox()
        self.narrative_scope_combo.addItems(["Selected Candidate", "Top 10 Candidates"])
        self.narrative_scope_combo.currentTextChanged.connect(self._update_narrative)
        form.addRow("Scope:", self.narrative_scope_combo)
        layout.addLayout(form)
        self.narrative_text = QTextEdit()
        self.narrative_text.setReadOnly(True)
        layout.addWidget(self.narrative_text)
        self.tabs.addTab(tab, "Evidence Narrative")

    def set_analysis_run(self, run: Optional[AnalysisRun]):
        self._current_run = run
        self._current_result = None
        self._populate_filter_options()
        self._refresh_all()

    def _populate_filter_options(self):
        pathologies = ["All"]
        biomarkers = ["All"]
        if self._current_run is not None:
            pathologies += sorted(self._current_run.pathology_names)
            biomarkers += sorted(self._current_run.biomarker_names)

        for combo, values in [
            (self.pathology_filter, pathologies),
            (self.pathology_combo, pathologies[1:]),
            (self.biomarker_filter, biomarkers),
            (self.biomarker_combo, biomarkers[1:]),
        ]:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(values)
            if current in values:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def _filtered_results(self) -> list[SweepResult]:
        if self._current_run is None:
            return []
        results = [r for r in self._current_run.results if r.auc >= self.min_auc_spin.value()]
        show_type = self.type_combo.currentText()
        if show_type == "Single":
            results = [r for r in results if "+" not in r.biomarker_name]
        elif show_type == "Combined":
            results = [r for r in results if "+" in r.biomarker_name]

        pathology = self.pathology_filter.currentText()
        if pathology and pathology != "All":
            results = [r for r in results if r.pathology_name == pathology]

        biomarker = self.biomarker_filter.currentText()
        if biomarker and biomarker != "All":
            results = [r for r in results if r.biomarker_name == biomarker]

        text = self.search_edit.text().strip().lower()
        if text:
            results = [
                r for r in results
                if text in r.biomarker_name.lower() or text in r.pathology_name.lower()
            ]
        return sorted(results, key=lambda r: (r.auc, r.stability_score, r.youden_j), reverse=True)

    def _refresh_all(self, *args):
        self._populate_evidence_table()
        self._populate_pathology_drilldown()
        self._populate_biomarker_drilldown()
        self._populate_workbench_candidates()
        self._update_provenance()
        self._update_narrative()

    def _populate_evidence_table(self):
        results = self._filtered_results()
        visible_results = results[:self.TABLE_ROW_LIMIT]
        if len(results) > len(visible_results):
            self.result_count_label.setText(
                f"Showing {len(visible_results):,}/{len(results):,}; refine filters"
            )
        else:
            self.result_count_label.setText(f"{len(results):,} results")
        self.evidence_table.setSortingEnabled(False)
        self.evidence_table.setRowCount(len(visible_results))
        for row, r in enumerate(visible_results):
            values = [
                row + 1,
                r.pathology_name,
                r.biomarker_name,
                r.threshold_direction,
                _fmt(r.optimal_threshold, 4),
                _ci(r.auc, r.ci_auc),
                _fmt(r.auprc),
                _ci(r.optimal_sensitivity, r.ci_sensitivity),
                _ci(r.optimal_specificity, r.ci_specificity),
                _fmt(r.optimal_ppv),
                _fmt(r.optimal_npv),
                _fmt(r.lr_positive, 2),
                _fmt(r.lr_negative, 2),
                _fmt(r.prevalence, 3),
                r.n_total,
                _fmt(r.missing_rate, 3),
                _fmt(r.p_value_corrected if r.p_value_corrected is not None else r.p_value, 4),
                _fmt(r.stability_score, 2),
                r.threshold_status,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, r)
                if r.auc >= 0.8 and col in (1, 2, 5):
                    item.setBackground(QBrush(QColor(210, 238, 221)))
                elif r.auc >= 0.7 and col in (1, 2, 5):
                    item.setBackground(QBrush(QColor(235, 244, 214)))
                self.evidence_table.setItem(row, col, item)
        self.evidence_table.setSortingEnabled(True)
        if visible_results and self._current_result is None:
            self._select_result(visible_results[0])

    def _on_evidence_clicked(self, row: int, col: int):
        item = self.evidence_table.item(row, col)
        if item is not None:
            result = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(result, SweepResult):
                self._select_result(result)

    def _select_result(self, result: SweepResult):
        self._current_result = result
        self.detail_summary.setText(result.evidence_narrative())
        self._draw_selected_result(result)
        self._select_workbench_result(result)
        self._update_narrative()

    def _draw_selected_result(self, result: SweepResult):
        self.detail_plot.clear()
        roc = self.detail_plot.addPlot(row=0, col=0, title="ROC")
        roc.setLabel("bottom", "False Positive Rate")
        roc.setLabel("left", "True Positive Rate")
        roc.setXRange(0, 1)
        roc.setYRange(0, 1)
        roc.plot([0, 1], [0, 1], pen=pg.mkPen("k", style=Qt.PenStyle.DashLine))
        if len(result.roc_fpr) > 0:
            roc.plot(result.roc_fpr, result.roc_tpr, pen=pg.mkPen("#1565c0", width=2))
            idx = int(np.argmax(result.roc_tpr - result.roc_fpr))
            roc.plot([result.roc_fpr[idx]], [result.roc_tpr[idx]], pen=None,
                     symbol="o", symbolSize=9, symbolBrush="#c62828")

        tradeoff = self.detail_plot.addPlot(row=1, col=0, title="Threshold Tradeoff")
        tradeoff.setLabel("bottom", "Threshold")
        tradeoff.setLabel("left", "Metric")
        if len(result.roc_thresholds) > 0:
            specificity = 1.0 - result.roc_fpr
            tradeoff.plot(result.roc_thresholds, result.roc_tpr,
                          pen=pg.mkPen("#2e7d32", width=2), name="Sensitivity")
            tradeoff.plot(result.roc_thresholds, specificity,
                          pen=pg.mkPen("#ef6c00", width=2), name="Specificity")
            threshold_line = pg.InfiniteLine(
                pos=result.optimal_threshold,
                angle=90,
                pen=pg.mkPen("#c62828", width=1, style=Qt.PenStyle.DashLine),
            )
            tradeoff.addItem(threshold_line)

    def _populate_pathology_drilldown(self, *args):
        if self._current_run is None or not self.pathology_combo.currentText():
            self.pathology_table.setRowCount(0)
            self.pathology_plot.clear()
            self.pathology_text.clear()
            return
        pname = self.pathology_combo.currentText()
        results = [r for r in self._current_run.ranked_results(self.min_auc_spin.value()) if r.pathology_name == pname]
        visible_results = results[:self.TABLE_ROW_LIMIT]
        self.pathology_table.setSortingEnabled(False)
        self.pathology_table.setRowCount(len(visible_results))
        for row, r in enumerate(visible_results):
            values = [
                r.biomarker_name, r.threshold_direction, _fmt(r.optimal_threshold, 4),
                _ci(r.auc, r.ci_auc), _fmt(r.optimal_sensitivity),
                _fmt(r.optimal_specificity), _fmt(r.lr_positive, 2),
                _fmt(r.lr_negative, 2), _fmt(r.stability_score, 2), r.threshold_status,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, r)
                self.pathology_table.setItem(row, col, item)
        self.pathology_table.setSortingEnabled(True)
        self._draw_pathology_overlay(results[:5], pname)
        if results:
            self.pathology_text.setText(
                f"{pname}: {len(results)} candidates above current filters. "
                f"Table shows {len(visible_results)}. "
                f"Best candidate is {results[0].biomarker_name} with AUC {results[0].auc:.3f}. "
                "Distribution plots require persisted per-record biomarker values and are reserved for the next data-layer step."
            )

    def _draw_pathology_overlay(self, results: list[SweepResult], pathology: str):
        self.pathology_plot.clear()
        plot = self.pathology_plot.addPlot(title=f"Top ROC Overlay: {pathology}")
        plot.setLabel("bottom", "False Positive Rate")
        plot.setLabel("left", "True Positive Rate")
        plot.setXRange(0, 1)
        plot.setYRange(0, 1)
        plot.plot([0, 1], [0, 1], pen=pg.mkPen("k", style=Qt.PenStyle.DashLine))
        colors = ["#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#c62828"]
        for idx, r in enumerate(results):
            if len(r.roc_fpr) > 0:
                plot.plot(r.roc_fpr, r.roc_tpr, pen=pg.mkPen(colors[idx % len(colors)], width=2))

    def _on_pathology_row_clicked(self, row: int, col: int):
        item = self.pathology_table.item(row, col)
        if item is not None and isinstance(item.data(Qt.ItemDataRole.UserRole), SweepResult):
            self._select_result(item.data(Qt.ItemDataRole.UserRole))

    def _populate_biomarker_drilldown(self, *args):
        if self._current_run is None or not self.biomarker_combo.currentText():
            self.biomarker_table.setRowCount(0)
            self.biomarker_plot.clear()
            self.biomarker_text.clear()
            return
        bname = self.biomarker_combo.currentText()
        results = [r for r in self._current_run.ranked_results(self.min_auc_spin.value()) if r.biomarker_name == bname]
        visible_results = results[:self.TABLE_ROW_LIMIT]
        self.biomarker_table.setSortingEnabled(False)
        self.biomarker_table.setRowCount(len(visible_results))
        for row, r in enumerate(visible_results):
            values = [
                r.pathology_name, r.threshold_direction, _fmt(r.optimal_threshold, 4),
                _ci(r.auc, r.ci_auc), _fmt(r.auprc),
                _fmt(r.optimal_sensitivity), _fmt(r.optimal_specificity),
                _fmt(r.p_value_corrected if r.p_value_corrected is not None else r.p_value, 4),
                r.n_total, r.threshold_status,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, r)
                self.biomarker_table.setItem(row, col, item)
        self.biomarker_table.setSortingEnabled(True)
        self._draw_biomarker_auc(results, bname)
        if results:
            self.biomarker_text.setText(
                f"{bname}: discriminates {len(results)} pathologies above current filters. "
                f"Table shows {len(visible_results)}. "
                "Correlation and redundancy analysis will become active once the per-record biomarker matrix is retained with the analysis run."
            )

    def _draw_biomarker_auc(self, results: list[SweepResult], biomarker: str):
        self.biomarker_plot.clear()
        plot = self.biomarker_plot.addPlot(title=f"AUC by Pathology: {biomarker}")
        plot.setLabel("bottom", "Rank")
        plot.setLabel("left", "AUC")
        plot.setYRange(0.45, 1.0)
        y = [r.auc for r in results[:20]]
        x = list(range(1, len(y) + 1))
        if y:
            plot.plot(x, y, pen=pg.mkPen("#1565c0", width=2), symbol="o", symbolBrush="#1565c0")

    def _on_biomarker_row_clicked(self, row: int, col: int):
        item = self.biomarker_table.item(row, col)
        if item is not None and isinstance(item.data(Qt.ItemDataRole.UserRole), SweepResult):
            self._select_result(item.data(Qt.ItemDataRole.UserRole))

    def _populate_workbench_candidates(self):
        current = _result_title(self._current_result) if self._current_result else ""
        results = self._filtered_results()
        self.workbench_result_combo.blockSignals(True)
        self.workbench_result_combo.clear()
        for r in results[:200]:
            self.workbench_result_combo.addItem(_result_title(r), r)
        if current:
            idx = self.workbench_result_combo.findText(current)
            if idx >= 0:
                self.workbench_result_combo.setCurrentIndex(idx)
        self.workbench_result_combo.blockSignals(False)
        self._update_workbench()

    def _select_workbench_result(self, result: SweepResult):
        title = _result_title(result)
        idx = self.workbench_result_combo.findText(title)
        if idx >= 0:
            self.workbench_result_combo.blockSignals(True)
            self.workbench_result_combo.setCurrentIndex(idx)
            self.workbench_result_combo.blockSignals(False)
        self.manual_threshold_spin.blockSignals(True)
        self.manual_threshold_spin.setValue(result.optimal_threshold)
        self.manual_threshold_spin.blockSignals(False)
        self.prevalence_spin.blockSignals(True)
        self.prevalence_spin.setValue(result.prevalence)
        self.prevalence_spin.blockSignals(False)
        self._update_workbench()

    def _workbench_result(self) -> Optional[SweepResult]:
        data = self.workbench_result_combo.currentData()
        return data if isinstance(data, SweepResult) else self._current_result

    def _threshold_metrics(self, result: SweepResult) -> dict:
        if len(result.roc_thresholds) == 0:
            return {
                "threshold": result.optimal_threshold,
                "sensitivity": result.optimal_sensitivity,
                "specificity": result.optimal_specificity,
            }

        sensitivity = np.asarray(result.roc_tpr, dtype=float)
        specificity = 1.0 - np.asarray(result.roc_fpr, dtype=float)
        thresholds = np.asarray(result.roc_thresholds, dtype=float)
        preset = self.preset_combo.currentText()

        if preset == "Screening":
            candidates = np.where(sensitivity >= 0.90)[0]
            idx = int(candidates[np.argmax(specificity[candidates])]) if len(candidates) else int(np.argmax(sensitivity + specificity - 1.0))
        elif preset == "Confirmatory":
            candidates = np.where(specificity >= 0.90)[0]
            idx = int(candidates[np.argmax(sensitivity[candidates])]) if len(candidates) else int(np.argmax(sensitivity + specificity - 1.0))
        elif preset == "Manual":
            manual = self.manual_threshold_spin.value()
            idx = int(np.argmin(np.abs(thresholds - manual)))
        else:
            idx = int(np.argmax(sensitivity + specificity - 1.0))

        return {
            "threshold": float(thresholds[idx]),
            "sensitivity": float(sensitivity[idx]),
            "specificity": float(specificity[idx]),
        }

    def _update_workbench(self, *args):
        result = self._workbench_result()
        if result is None:
            self.workbench_metrics.setRowCount(0)
            self.matrix_table.clearContents()
            self.workbench_narrative.clear()
            return

        metrics = self._threshold_metrics(result)
        counts = result.confusion_counts(metrics["sensitivity"], metrics["specificity"])
        prevalence = self.prevalence_spin.value()
        sens = metrics["sensitivity"]
        spec = metrics["specificity"]
        ppv_den = sens * prevalence + (1.0 - spec) * (1.0 - prevalence)
        npv_den = spec * (1.0 - prevalence) + (1.0 - sens) * prevalence
        adj_ppv = sens * prevalence / ppv_den if ppv_den else 0.0
        adj_npv = spec * (1.0 - prevalence) / npv_den if npv_den else 0.0
        lr_pos = sens / (1.0 - spec) if spec < 1.0 else float("inf")
        lr_neg = (1.0 - sens) / spec if spec > 0.0 else float("inf")

        rows = [
            ("Preset", self.preset_combo.currentText()),
            ("Direction", result.threshold_direction),
            ("Cutoff", _fmt(metrics["threshold"], 4)),
            ("Sensitivity", _fmt(sens)),
            ("Specificity", _fmt(spec)),
            ("LR+", _fmt(lr_pos, 2)),
            ("LR-", _fmt(lr_neg, 2)),
            ("Adjusted PPV", _fmt(adj_ppv)),
            ("Adjusted NPV", _fmt(adj_npv)),
        ]
        self.workbench_metrics.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.workbench_metrics.setItem(row, 0, QTableWidgetItem(name))
            self.workbench_metrics.setItem(row, 1, QTableWidgetItem(str(value)))

        self.matrix_table.setItem(0, 0, QTableWidgetItem("Positive"))
        self.matrix_table.setItem(0, 1, QTableWidgetItem(str(counts["tp"])))
        self.matrix_table.setItem(0, 2, QTableWidgetItem(str(counts["fp"])))
        self.matrix_table.setItem(1, 0, QTableWidgetItem("Negative"))
        self.matrix_table.setItem(1, 1, QTableWidgetItem(str(counts["fn"])))
        self.matrix_table.setItem(1, 2, QTableWidgetItem(str(counts["tn"])))

        op = ">=" if result.threshold_direction == "above" else "<="
        self.workbench_narrative.setText(
            f"For {result.pathology_name}, classify {result.biomarker_name} as positive when "
            f"the value is {op} {_fmt(metrics['threshold'], 4)}. In this cohort-level estimate, "
            f"the preset yields sensitivity {_fmt(sens)}, specificity {_fmt(spec)}, LR+ {_fmt(lr_pos, 2)} "
            f"and LR- {_fmt(lr_neg, 2)}. At a target prevalence of {prevalence:.1%}, expected PPV is "
            f"{_fmt(adj_ppv)} and expected NPV is {_fmt(adj_npv)}. Treat this as exploratory until the "
            "cutoff is validated on a held-out MIMIC or external cohort."
        )

        subgroups = result.subgroup_metrics or {}
        if subgroups:
            _STRATUM_LABELS = {
                "sex_M": "Männer (M)", "sex_F": "Frauen (F)",
                "age_18-29": "18–29 J.", "age_30-49": "30–49 J.",
                "age_50-69": "50–69 J.", "age_70+": "70+ J.",
                "sex_M_age_18-29": "M / 18–29 J.", "sex_M_age_30-49": "M / 30–49 J.",
                "sex_M_age_50-69": "M / 50–69 J.", "sex_M_age_70+": "M / 70+ J.",
                "sex_F_age_18-29": "F / 18–29 J.", "sex_F_age_30-49": "F / 30–49 J.",
                "sex_F_age_50-69": "F / 50–69 J.", "sex_F_age_70+": "F / 70+ J.",
            }
            self.subgroup_table.setSortingEnabled(False)
            self.subgroup_table.setRowCount(len(subgroups))
            for row, (stratum, sm) in enumerate(sorted(subgroups.items())):
                thr = sm.get("threshold")
                direction = sm.get("threshold_direction", "above")
                thr_op = "≥" if direction == "above" else "≤"
                thr_text = f"{thr_op} {_fmt(thr, 4)}" if thr is not None else "N/A"
                values = [
                    _STRATUM_LABELS.get(stratum, stratum),
                    sm.get("n", ""),
                    sm.get("n_pos", ""),
                    _fmt(sm.get("auc")),
                    thr_text,
                    _fmt(sm.get("sensitivity")),
                    _fmt(sm.get("specificity")),
                ]
                for col, value in enumerate(values):
                    self.subgroup_table.setItem(row, col, QTableWidgetItem(str(value)))
            self.subgroup_table.setSortingEnabled(True)
            self.subgroup_group.setVisible(True)
        else:
            self.subgroup_group.setVisible(False)

    def _update_provenance(self):
        if self._current_run is None:
            self.provenance_text.setText("No analysis run loaded.")
            return
        summary = self._current_run.cohort_summary()
        lines = [
            "Cohort & Provenance",
            "",
            f"Run ID: {summary['run_id']}",
            f"VCG method: {summary['vcg_method']}",
            f"Analysis mode: {summary['analysis_mode']}",
            f"Split strategy: {summary['split_strategy']}",
            f"Max records represented: {summary['n_records_max']:,}",
            f"Biomarkers: {summary['n_biomarkers']}",
            f"Pathologies: {summary['n_pathologies']}",
            f"Pairs tested: {summary['n_pairs']}",
            f"Significant pairs: {summary['n_significant']}",
            "",
            "Dataset versions:",
        ]
        for key, value in self._current_run.dataset_versions.items():
            lines.append(f"  {key}: {value}")
        lines += ["", "Cohort filters:"]
        for key, value in self._current_run.cohort_filter.items():
            lines.append(f"  {key}: {value}")
        lines += ["", "MIMIC-ready linkage rules:"]
        for key, value in self._current_run.linkage_rules.items():
            lines.append(f"  {key}: {value}")
        lines += [
            "",
            "Reporting checklist hooks:",
            "  STARD: index test, reference standard, cutoff rationale, missing data, precision.",
            "  TRIPOD+AI: data sources, preprocessing, model output, subgroup performance, fairness.",
            "  BEST/FDA: biomarker description, measurement method, context of use.",
        ]
        self.provenance_text.setText("\n".join(lines))

    def _update_narrative(self, *args):
        if self._current_run is None:
            self.narrative_text.setText("No analysis run loaded.")
            return
        if self.narrative_scope_combo.currentText() == "Top 10 Candidates":
            results = self._filtered_results()[:10]
            text = "\n\n".join(r.evidence_narrative() for r in results)
        elif self._current_result is not None:
            text = self._current_result.evidence_narrative()
        else:
            results = self._filtered_results()
            text = results[0].evidence_narrative() if results else "No candidate selected."
        self.narrative_text.setText(text)
