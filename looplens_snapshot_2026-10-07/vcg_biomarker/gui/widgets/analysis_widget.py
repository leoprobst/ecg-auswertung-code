from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QComboBox, QSpinBox, QFileDialog, QMessageBox, QTabWidget,
    QSplitter, QCheckBox,
)
from PyQt6.QtCore import Qt, QSettings
import pyqtgraph as pg

from ...config import AppConfig
from ...core.analysis.results import AnalysisRun
from ...core.analysis.threshold_sweep import SweepResult


class AnalysisWidget(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._current_run: AnalysisRun | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        config_group = QGroupBox("Analysis Configuration")
        config_layout = QVBoxLayout(config_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Correction:"))
        self.correction_combo = QComboBox()
        self.correction_combo.addItems(["fdr_bh", "bonferroni", "none"])
        self.correction_combo.setCurrentText(self.config.multiple_testing_correction)
        row1.addWidget(self.correction_combo)

        row1.addWidget(QLabel("Bootstrap:"))
        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(10, 10000)
        self.bootstrap_spin.setValue(self.config.bootstrap_iterations)
        row1.addWidget(self.bootstrap_spin)

        row1.addWidget(QLabel("Combinations:"))
        self.combo_strategy_combo = QComboBox()
        self.combo_strategy_combo.addItems([
            "None (single only)",
            "Greedy Forward Selection",
            "Brute Force (small datasets)",
        ])
        self.combo_strategy_combo.setToolTip(
            "None: only single biomarkers are tested.\n"
            "Greedy: for each pathology, starts from the top-K single biomarkers\n"
            "  and greedily adds the marker that best improves AUC.\n"
            "  Fast (no bootstrap during screening), scales to 1000+ biomarkers.\n"
            "Brute Force: tests all pairs/triplets — only feasible for small sets (<200 markers)."
        )
        self.combo_strategy_combo.currentIndexChanged.connect(self._on_combo_strategy_changed)
        row1.addWidget(self.combo_strategy_combo)
        config_layout.addLayout(row1)

        self._greedy_row = QHBoxLayout()
        self._greedy_row.addWidget(QLabel("Seeds/Pathologie:"))
        self.greedy_seeds_spin = QSpinBox()
        self.greedy_seeds_spin.setRange(1, 5)
        self.greedy_seeds_spin.setValue(3)
        self.greedy_seeds_spin.setToolTip(
            "How many top single-biomarker starting points to try per pathology.\n"
            "Set to 2-3 to also test the 2nd/3rd best marker as a seed."
        )
        self._greedy_row.addWidget(self.greedy_seeds_spin)

        self._greedy_row.addWidget(QLabel("Max Tiefe:"))
        self.greedy_depth_spin = QSpinBox()
        self.greedy_depth_spin.setRange(1, 3)
        self.greedy_depth_spin.setValue(2)
        self.greedy_depth_spin.setToolTip(
            "How many extra markers to add greedily (max combination size = depth + 1).\n"
            "Depth 1 → pairs (2 biomarkers), depth 2 → up to triplets."
        )
        self._greedy_row.addWidget(self.greedy_depth_spin)
        self._greedy_row.addStretch()

        self._greedy_widget = QWidget()
        self._greedy_widget.setLayout(self._greedy_row)
        self._greedy_widget.setVisible(False)
        config_layout.addWidget(self._greedy_widget)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Min AUC:"))
        self.min_auc_spin = QSpinBox()
        self.min_auc_spin.setRange(0, 100)
        self.min_auc_spin.setValue(60)
        row2.addWidget(self.min_auc_spin)

        row2.addWidget(QLabel("Min Sens:"))
        self.min_sens_spin = QSpinBox()
        self.min_sens_spin.setRange(0, 100)
        self.min_sens_spin.setValue(50)
        row2.addWidget(self.min_sens_spin)

        row2.addWidget(QLabel("Min Spec:"))
        self.min_spec_spin = QSpinBox()
        self.min_spec_spin.setRange(0, 100)
        self.min_spec_spin.setValue(50)
        row2.addWidget(self.min_spec_spin)
        config_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.subgroups_check = QCheckBox("Subgruppenanalyse (Geschlecht + Altersgruppe)")
        self.subgroups_check.setToolTip(
            "Berechnet Schwellenwerte getrennt für Männer/Frauen und Altersgruppen\n"
            "(18–29, 30–49, 50–69, 70+) für Biomarker mit AUC ≥ 0.65.\n"
            "Verlängert die Analyse um ~5–15 Minuten."
        )
        row3.addWidget(self.subgroups_check)
        row3.addStretch()
        config_layout.addLayout(row3)

        layout.addWidget(config_group)

        self.result_tabs = QTabWidget()

        single_widget = QWidget()
        single_layout = QVBoxLayout(single_widget)
        self.single_table = self._create_results_table()
        single_layout.addWidget(self.single_table)
        self.result_tabs.addTab(single_widget, "Single Biomarkers")

        combined_widget = QWidget()
        combined_layout = QVBoxLayout(combined_widget)
        self.combined_table = self._create_results_table()
        combined_layout.addWidget(self.combined_table)
        self.result_tabs.addTab(combined_widget, "Combined Biomarkers")

        layout.addWidget(self.result_tabs)

        roc_group = QGroupBox("ROC Curve")
        roc_layout = QVBoxLayout(roc_group)
        self.roc_plot = pg.PlotWidget()
        self.roc_plot.setBackground("w")
        self.roc_plot.setLabel("bottom", "False Positive Rate")
        self.roc_plot.setLabel("left", "True Positive Rate")
        self.roc_plot.setRange(xRange=(0, 1), yRange=(0, 1))
        self.roc_plot.addLine(x=0, pen=pg.mkPen("k", width=1))
        self.roc_plot.addLine(x=1, pen=pg.mkPen("k", width=1))
        self.roc_plot.addLine(y=0, pen=pg.mkPen("k", width=1))
        self.roc_plot.addLine(y=1, pen=pg.mkPen("k", width=1))
        diag = pg.InfiniteLine(pos=(0, 0), angle=45, pen=pg.mkPen("k", width=1, style=Qt.PenStyle.DashLine))
        self.roc_plot.addItem(diag)
        self.roc_curve_item = None
        roc_layout.addWidget(self.roc_plot)

        self.roc_info_label = QLabel("Click a result row to view ROC curve")
        roc_layout.addWidget(self.roc_info_label)
        layout.addWidget(roc_group)

        export_layout = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(self._on_export_csv)
        export_layout.addWidget(self.export_csv_btn)
        self.export_excel_btn = QPushButton("Export Excel")
        self.export_excel_btn.clicked.connect(self._on_export_excel)
        export_layout.addWidget(self.export_excel_btn)
        layout.addLayout(export_layout)

    def _create_results_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(13)
        table.setHorizontalHeaderLabels([
            "Biomarker", "Pathology", "N", "N+", "N-",
            "AUC", "Threshold", "Sens", "Spec",
            "PPV", "NPV", "p(corr)", "Signif"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setSortingEnabled(True)
        table.cellClicked.connect(self._on_result_clicked)
        return table

    def _on_combo_strategy_changed(self, index: int):
        strategy = self._strategy_from_index(index)
        self._greedy_widget.setVisible(strategy == "greedy")

    def _strategy_from_index(self, index: int) -> str:
        return ["none", "greedy", "brute_force"][index]

    def get_combination_strategy(self) -> str:
        return self._strategy_from_index(self.combo_strategy_combo.currentIndex())

    def get_greedy_seeds(self) -> int:
        return self.greedy_seeds_spin.value()

    def get_greedy_max_depth(self) -> int:
        return self.greedy_depth_spin.value()

    def get_max_combination(self) -> int:
        """Legacy compatibility — returns 1 for none/greedy, 2 for brute_force."""
        strategy = self.get_combination_strategy()
        return 2 if strategy == "brute_force" else 1

    def get_correction_method(self) -> str:
        return self.correction_combo.currentText()

    def get_bootstrap_iterations(self) -> int:
        return self.bootstrap_spin.value()

    def get_compute_subgroups(self) -> bool:
        return self.subgroups_check.isChecked()

    def display_results(self, run: AnalysisRun):
        self._current_run = run

        single_results = [r for r in run.results if "+" not in r.biomarker_name]
        combined_results = [r for r in run.results if "+" in r.biomarker_name]

        self._populate_table(self.single_table, single_results, run)
        self._populate_table(self.combined_table, combined_results, run)

        status = run.cohort_filter.get("status", "complete")
        expected = run.cohort_filter.get("n_expected_pairs")
        prefix = "Live: " if status == "partial" else ""
        expected_text = f"/{expected:,}" if expected else ""
        self.roc_info_label.setText(
            f"{prefix}Total: {len(run.results):,}{expected_text} "
            f"({len(single_results):,} single, {len(combined_results):,} combined)"
        )

    def _populate_table(self, table: QTableWidget, results: list[SweepResult], run: AnalysisRun):
        min_auc = self.min_auc_spin.value() / 100.0
        min_sens = self.min_sens_spin.value() / 100.0
        min_spec = self.min_spec_spin.value() / 100.0

        sorted_results = sorted(results, key=lambda r: r.auc, reverse=True)

        was_sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(len(sorted_results))
            for i, r in enumerate(sorted_results):
                items = [
                    str(r.biomarker_name),
                    str(r.pathology_name),
                    str(r.n_total),
                    str(r.n_positive),
                    str(r.n_negative),
                    f"{r.auc:.4f}",
                    f"{r.optimal_threshold:.4f}",
                    f"{r.optimal_sensitivity:.4f}",
                    f"{r.optimal_specificity:.4f}",
                    f"{r.optimal_ppv:.4f}",
                    f"{r.optimal_npv:.4f}",
                    f"{r.p_value_corrected:.6f}" if r.p_value_corrected is not None else "N/A",
                    "Yes" if r.is_significant else "No",
                ]

                for j, text in enumerate(items):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, r)
                    if r.is_significant and r.auc >= min_auc:
                        item.setBackground(Qt.GlobalColor.green if r.is_significant else Qt.GlobalColor.white)
                    table.setItem(i, j, item)
        finally:
            table.setUpdatesEnabled(True)
            table.setSortingEnabled(was_sorting)

    def _on_result_clicked(self, row, col):
        table = self.sender()
        if table is None:
            return
        item = table.item(row, 0)
        if item is None:
            return
        result = item.data(Qt.ItemDataRole.UserRole)
        if result is None or not isinstance(result, SweepResult):
            return

        self.roc_plot.removeItem(self.roc_curve_item) if self.roc_curve_item else None

        self.roc_curve_item = self.roc_plot.plot(
            result.roc_fpr, result.roc_tpr,
            pen=pg.mkPen("b", width=2),
        )

        ci_auc = f" ({result.ci_auc[0]:.3f}-{result.ci_auc[1]:.3f})" if result.ci_auc else ""
        self.roc_info_label.setText(
            f"{result.biomarker_name} vs {result.pathology_name} | "
            f"AUC={result.auc:.3f}{ci_auc} | "
            f"Thr={result.optimal_threshold:.4f} | "
            f"Sens={result.optimal_sensitivity:.3f} | "
            f"Spec={result.optimal_specificity:.3f} | "
            f"p(corr)={result.p_value_corrected:.6f}" if result.p_value_corrected else ""
        )

    def _on_export_csv(self):
        if not self._current_run:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if path:
            self._current_run.export_csv(Path(path))

    def _on_export_excel(self):
        if not self._current_run:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Excel", "", "Excel Files (*.xlsx)")
        if path:
            self._current_run.export_excel(Path(path))

    def export_csv(self, path: Path):
        if self._current_run:
            self._current_run.export_csv(path)

    def export_excel(self, path: Path):
        if self._current_run:
            self._current_run.export_excel(path)
