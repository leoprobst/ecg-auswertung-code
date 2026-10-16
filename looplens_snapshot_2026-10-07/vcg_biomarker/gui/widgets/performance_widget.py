from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QSplitter,
    QTextEdit,
)
import pyqtgraph as pg

from ...config import AppConfig
from ...core.analysis.clinical_baseline import ClinicalRuleResult
from ...core.analysis.results import AnalysisRun
from ...core.analysis.threshold_sweep import SweepResult


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        if not np.isfinite(value):
            return "Inf" if value == float("inf") else "N/A"
    except TypeError:
        return str(value)
    return f"{float(value):.{digits}f}"


class PerformanceDashboardWidget(QWidget):
    """
    Pathology-centered performance dashboard.

    This replaces the old full biomarker x pathology matrix with ranked,
    interpretable comparisons against fixed clinical ECG reference rules.
    """

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._current_run: Optional[AnalysisRun] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Pathology:"))
        self.pathology_combo = QComboBox()
        self.pathology_combo.setMaxVisibleItems(30)
        self.pathology_combo.currentTextChanged.connect(self._refresh_detail)
        controls.addWidget(self.pathology_combo)

        controls.addWidget(QLabel("Biomarker:"))
        self.biomarker_combo = QComboBox()
        self.biomarker_combo.setMaxVisibleItems(30)
        self.biomarker_combo.currentTextChanged.connect(self._refresh_biomarker_view)
        controls.addWidget(self.biomarker_combo)

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
        self.type_combo.currentTextChanged.connect(self._refresh_all)
        controls.addWidget(self.type_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._setup_overview_tab()
        self._setup_pathology_tab()
        self._setup_biomarker_tab()

    def _setup_overview_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.overview_table = QTableWidget()
        self.overview_table.setColumnCount(8)
        self.overview_table.setHorizontalHeaderLabels([
            "Pathology", "Best Biomarker", "AUC", "Clinical Rule",
            "Clinical BalAcc", "Delta", "N", "Status",
        ])
        self.overview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.overview_table.setSortingEnabled(True)
        self.overview_table.cellClicked.connect(self._on_overview_clicked)
        splitter.addWidget(self.overview_table)

        self.overview_plot = pg.GraphicsLayoutWidget()
        self.overview_plot.setBackground("w")
        splitter.addWidget(self.overview_plot)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Pathology Overview")

    def _setup_pathology_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Vertical)

        top = QSplitter(Qt.Orientation.Horizontal)
        self.ranking_plot = pg.GraphicsLayoutWidget()
        self.ranking_plot.setBackground("w")
        top.addWidget(self.ranking_plot)

        self.roc_plot = pg.GraphicsLayoutWidget()
        self.roc_plot.setBackground("w")
        top.addWidget(self.roc_plot)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 1)
        splitter.addWidget(top)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.clinical_table = QTableWidget()
        self.clinical_table.setColumnCount(14)
        self.clinical_table.setHorizontalHeaderLabels([
            "Clinical Marker", "Biomarker", "Cutoff", "Sens", "Spec",
            "PPV", "NPV", "BalAcc", "Lit Sens", "Lit Spec",
            "Lit BalAcc", "N", "Source", "Status",
        ])
        self.clinical_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.clinical_table.setSortingEnabled(True)
        bottom.addWidget(self.clinical_table)

        self.pathology_note = QTextEdit()
        self.pathology_note.setReadOnly(True)
        bottom.addWidget(self.pathology_note)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Selected Pathology")

    def _setup_biomarker_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.biomarker_table = QTableWidget()
        self.biomarker_table.setColumnCount(9)
        self.biomarker_table.setHorizontalHeaderLabels([
            "Pathology", "AUC", "Sens", "Spec", "PPV", "NPV", "q",
            "Stability", "N",
        ])
        self.biomarker_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.biomarker_table.setSortingEnabled(True)
        splitter.addWidget(self.biomarker_table)

        self.biomarker_plot = pg.GraphicsLayoutWidget()
        self.biomarker_plot.setBackground("w")
        splitter.addWidget(self.biomarker_plot)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "Biomarker View")

    def set_analysis_run(self, run: Optional[AnalysisRun]):
        self._current_run = run
        self._populate_controls()
        self._refresh_all()

    def _populate_controls(self):
        pathologies = []
        biomarkers = []
        if self._current_run is not None:
            pathologies = sorted(set(self._current_run.pathology_names) | {
                item.pathology_name for item in self._baseline_results()
            })
            biomarkers = sorted(self._current_run.biomarker_names)

        for combo, values in [
            (self.pathology_combo, pathologies),
            (self.biomarker_combo, biomarkers),
        ]:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(values)
            if current in values:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def _baseline_results(self) -> list[ClinicalRuleResult]:
        if self._current_run is None:
            return []
        items = getattr(self._current_run, "clinical_baseline_results", []) or []
        return [
            item if isinstance(item, ClinicalRuleResult) else ClinicalRuleResult.from_dict(item)
            for item in items
        ]

    def _filtered_results(self) -> list[SweepResult]:
        if self._current_run is None:
            return []
        results = [r for r in self._current_run.results if r.auc >= self.min_auc_spin.value()]
        show_type = self.type_combo.currentText()
        if show_type == "Single":
            results = [r for r in results if "+" not in r.biomarker_name]
        elif show_type == "Combined":
            results = [r for r in results if "+" in r.biomarker_name]
        return sorted(results, key=lambda r: (r.auc, r.stability_score, r.n_total), reverse=True)

    def _best_biomarkers(self) -> dict[str, SweepResult]:
        best = {}
        for result in self._filtered_results():
            current = best.get(result.pathology_name)
            if current is None or result.auc > current.auc:
                best[result.pathology_name] = result
        return best

    def _best_baselines(self) -> dict[str, ClinicalRuleResult]:
        best = {}
        for item in self._baseline_results():
            if item.status != "ok" or item.balanced_accuracy is None:
                continue
            current = best.get(item.pathology_name)
            if current is None or (item.balanced_accuracy or 0) > (current.balanced_accuracy or 0):
                best[item.pathology_name] = item
        return best

    def _clinical_biomarker_names(self) -> set[str]:
        names = set()
        for item in self._baseline_results():
            for part in str(item.biomarker_name).split(" + "):
                if part:
                    names.add(part)
        return names

    def _refresh_all(self, *args):
        self._refresh_overview()
        self._refresh_detail()
        self._refresh_biomarker_view()

    def _refresh_overview(self):
        best_biomarkers = self._best_biomarkers()
        best_baselines = self._best_baselines()
        pathologies = sorted(set(best_biomarkers) | set(self._current_run.pathology_names if self._current_run else []))

        self.overview_table.setSortingEnabled(False)
        self.overview_table.setRowCount(len(pathologies))
        rows_for_plot = []
        for row, pathology in enumerate(pathologies):
            biomarker = best_biomarkers.get(pathology)
            baseline = best_baselines.get(pathology)
            delta = None
            if biomarker is not None and baseline is not None and baseline.balanced_accuracy is not None:
                delta = biomarker.auc - baseline.balanced_accuracy

            values = [
                pathology,
                biomarker.biomarker_name if biomarker else "N/A",
                _fmt(biomarker.auc if biomarker else None),
                baseline.display_name if baseline else "Keine klinische Referenzregel konfiguriert",
                _fmt(baseline.balanced_accuracy if baseline else None),
                _fmt(delta),
                str(biomarker.n_total if biomarker else (baseline.n_evaluable if baseline else 0)),
                "ok" if biomarker or baseline else "no_results",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col in (1, 2) and biomarker is not None and biomarker.auc >= 0.75:
                    item.setBackground(QBrush(QColor(210, 238, 221)))
                if col in (3, 4) and baseline is not None:
                    item.setBackground(QBrush(QColor(232, 240, 254)))
                self.overview_table.setItem(row, col, item)
            rows_for_plot.append((pathology, biomarker.auc if biomarker else None, baseline.balanced_accuracy if baseline else None))
        self.overview_table.setSortingEnabled(True)
        self._draw_overview_plot(rows_for_plot)

    def _draw_overview_plot(self, rows: list[tuple[str, Optional[float], Optional[float]]]):
        self.overview_plot.clear()
        plot = self.overview_plot.addPlot(title="Best Biomarker AUC vs Clinical Rule Balanced Accuracy")
        plot.setLabel("bottom", "Pathology")
        plot.setLabel("left", "Performance")
        plot.setYRange(0.0, 1.0)
        rows = rows[:30]
        x = np.arange(len(rows), dtype=float)
        biomarker_y = np.array([0.0 if auc is None else auc for _, auc, _ in rows], dtype=float)
        clinical_y = np.array([0.0 if bal is None else bal for _, _, bal in rows], dtype=float)
        plot.addItem(pg.BarGraphItem(x=x - 0.18, height=biomarker_y, width=0.32, brush="#1565c0"))
        plot.addItem(pg.BarGraphItem(x=x + 0.18, height=clinical_y, width=0.32, brush="#ef6c00"))
        plot.getAxis("bottom").setTicks([[(idx, name) for idx, (name, _, _) in enumerate(rows)]])

    def _refresh_detail(self, *args):
        pathology = self.pathology_combo.currentText()
        if not pathology:
            self.ranking_plot.clear()
            self.roc_plot.clear()
            self.clinical_table.setRowCount(0)
            self.pathology_note.clear()
            return
        results = [r for r in self._filtered_results() if r.pathology_name == pathology]
        baselines = [r for r in self._baseline_results() if r.pathology_name == pathology]
        self._draw_ranking_plot(results[:20])
        self._draw_roc_overlay(results[:5], [r for r in baselines if r.status == "ok"])
        self._populate_clinical_table(pathology, baselines)
        self._populate_pathology_note(pathology, results, baselines)

    def _draw_ranking_plot(self, results: list[SweepResult]):
        self.ranking_plot.clear()
        plot = self.ranking_plot.addPlot(title="Top Biomarker AUC Ranking")
        plot.setLabel("bottom", "Rank")
        plot.setLabel("left", "AUC")
        plot.setYRange(0.45, 1.0)
        if not results:
            return
        clinical_names = self._clinical_biomarker_names()
        for idx, result in enumerate(results):
            if result.biomarker_name in clinical_names:
                brush = "#ef6c00"
            elif "+" in result.biomarker_name:
                brush = "#6a1b9a"
            else:
                brush = "#1565c0"
            plot.addItem(pg.BarGraphItem(x=[idx + 1], height=[result.auc], width=0.75, brush=brush))
        plot.getAxis("bottom").setTicks([[(idx + 1, str(idx + 1)) for idx in range(len(results))]])

    def _draw_roc_overlay(self, results: list[SweepResult], baselines: list[ClinicalRuleResult]):
        self.roc_plot.clear()
        plot = self.roc_plot.addPlot(title="Top ROC Curves + Clinical Fixed-Rule Points")
        plot.setLabel("bottom", "False Positive Rate")
        plot.setLabel("left", "True Positive Rate")
        plot.setXRange(0.0, 1.0)
        plot.setYRange(0.0, 1.0)
        plot.plot([0, 1], [0, 1], pen=pg.mkPen("k", style=Qt.PenStyle.DashLine))
        colors = ["#1565c0", "#2e7d32", "#6a1b9a", "#c62828", "#455a64"]
        for idx, result in enumerate(results):
            if len(result.roc_fpr) > 0:
                plot.plot(result.roc_fpr, result.roc_tpr, pen=pg.mkPen(colors[idx % len(colors)], width=2))
        for item in baselines:
            if item.sensitivity is None or item.specificity is None:
                continue
            plot.plot(
                [1.0 - item.specificity],
                [item.sensitivity],
                pen=None,
                symbol="x",
                symbolSize=13,
                symbolBrush="#ef6c00",
                symbolPen="#ef6c00",
            )

    def _populate_clinical_table(self, pathology: str, baselines: list[ClinicalRuleResult]):
        if not baselines:
            self.clinical_table.setSortingEnabled(False)
            self.clinical_table.setRowCount(1)
            values = [
                "Keine klinische Referenzregel konfiguriert",
                "", "", "", "", "", "", "", "", "", "", "", "", "not_configured",
            ]
            for col, value in enumerate(values):
                self.clinical_table.setItem(0, col, QTableWidgetItem(value))
            self.clinical_table.setSortingEnabled(True)
            return

        self.clinical_table.setSortingEnabled(False)
        self.clinical_table.setRowCount(len(baselines))
        for row, item in enumerate(sorted(baselines, key=lambda r: (r.status == "ok", r.balanced_accuracy or -1), reverse=True)):
            ref_sens = getattr(item, "reference_sensitivity", None)
            ref_spec = getattr(item, "reference_specificity", None)
            ref_bal = getattr(item, "reference_balanced_accuracy", None)
            values = [
                item.display_name,
                item.biomarker_name,
                item.threshold_label,
                _fmt(item.sensitivity),
                _fmt(item.specificity),
                _fmt(item.ppv),
                _fmt(item.npv),
                _fmt(item.balanced_accuracy),
                _fmt(ref_sens),
                _fmt(ref_spec),
                _fmt(ref_bal),
                str(item.n_evaluable),
                item.source_label,
                item.status,
            ]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if item.status == "ok" and col in (0, 7):
                    cell.setBackground(QBrush(QColor(232, 240, 254)))
                self.clinical_table.setItem(row, col, cell)
        self.clinical_table.setSortingEnabled(True)

    def _populate_pathology_note(
        self,
        pathology: str,
        results: list[SweepResult],
        baselines: list[ClinicalRuleResult],
    ):
        lines = [f"Pathology: {pathology}", ""]
        if results:
            best = results[0]
            lines.append(
                f"Best biomarker: {best.biomarker_name} with AUC {_fmt(best.auc)}, "
                f"sensitivity {_fmt(best.optimal_sensitivity)}, specificity {_fmt(best.optimal_specificity)}."
            )
        else:
            lines.append("No biomarker result passes the current filters.")
        ok_baselines = [b for b in baselines if b.status == "ok"]
        if ok_baselines:
            best_rule = max(ok_baselines, key=lambda item: item.balanced_accuracy or 0)
            lines.append(
                f"Best clinical fixed rule: {best_rule.display_name} "
                f"({best_rule.threshold_label}) with balanced accuracy {_fmt(best_rule.balanced_accuracy)}."
            )
            if results and best_rule.balanced_accuracy is not None:
                lines.append(
                    f"Context gap: best biomarker AUC minus clinical balanced accuracy = "
                    f"{_fmt(results[0].auc - best_rule.balanced_accuracy)}."
                )
        else:
            lines.append("Keine klinische Referenzregel konfiguriert")
        if baselines:
            lines.append("")
            lines.append("Clinical-rule notes:")
            for item in baselines:
                lines.append(f"- {item.display_name}: {item.note}")
                ref_bits = []
                if getattr(item, "reference_sensitivity", None) is not None:
                    ref_bits.append(f"published sensitivity {_fmt(item.reference_sensitivity)}")
                if getattr(item, "reference_specificity", None) is not None:
                    ref_bits.append(f"published specificity {_fmt(item.reference_specificity)}")
                if ref_bits:
                    lines.append(f"  Literature benchmark: {', '.join(ref_bits)}.")
                if getattr(item, "reference_note", ""):
                    lines.append(f"  Reference note: {item.reference_note}")
        self.pathology_note.setText("\n".join(lines))

    def _refresh_biomarker_view(self, *args):
        biomarker = self.biomarker_combo.currentText()
        if not biomarker or self._current_run is None:
            self.biomarker_table.setRowCount(0)
            self.biomarker_plot.clear()
            return
        results = [
            r for r in self._current_run.results
            if r.biomarker_name == biomarker and r.auc >= self.min_auc_spin.value()
        ]
        results = sorted(results, key=lambda r: (r.auc, r.stability_score), reverse=True)
        self.biomarker_table.setSortingEnabled(False)
        self.biomarker_table.setRowCount(len(results))
        for row, result in enumerate(results):
            q_value = result.p_value_corrected if result.p_value_corrected is not None else result.p_value
            values = [
                result.pathology_name,
                _fmt(result.auc),
                _fmt(result.optimal_sensitivity),
                _fmt(result.optimal_specificity),
                _fmt(result.optimal_ppv),
                _fmt(result.optimal_npv),
                _fmt(q_value, 4),
                _fmt(result.stability_score, 2),
                str(result.n_total),
            ]
            for col, value in enumerate(values):
                self.biomarker_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.biomarker_table.setSortingEnabled(True)
        self._draw_biomarker_plot(results[:30], biomarker)

    def _draw_biomarker_plot(self, results: list[SweepResult], biomarker: str):
        self.biomarker_plot.clear()
        plot = self.biomarker_plot.addPlot(title=f"{biomarker} across pathologies")
        plot.setLabel("bottom", "Pathology rank")
        plot.setLabel("left", "AUC")
        plot.setYRange(0.45, 1.0)
        if not results:
            return
        x = np.arange(1, len(results) + 1, dtype=float)
        y = np.array([r.auc for r in results], dtype=float)
        plot.plot(x, y, pen=pg.mkPen("#1565c0", width=2), symbol="o", symbolBrush="#1565c0")

    def _on_overview_clicked(self, row: int, col: int):
        item = self.overview_table.item(row, 0)
        if item is None:
            return
        pathology = item.text()
        idx = self.pathology_combo.findText(pathology)
        if idx >= 0:
            self.pathology_combo.setCurrentIndex(idx)
            self.tabs.setCurrentIndex(1)
