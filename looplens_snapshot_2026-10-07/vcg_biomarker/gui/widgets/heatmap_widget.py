from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

from ...config import AppConfig
from ...core.analysis.results import AnalysisRun


class HeatmapWidget(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._current_run: Optional[AnalysisRun] = None
        self._biomarker_names: list[str] = []
        self._pathology_names: list[str] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Color by:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["AUC", "Sensitivity", "Specificity", "PPV", "NPV", "p-value (-log10)"])
        self.metric_combo.currentTextChanged.connect(self._refresh)
        ctrl_layout = ctrl

        ctrl.addWidget(self.metric_combo)

        ctrl.addWidget(QLabel("Min AUC:"))
        self.min_auc_combo = QComboBox()
        self.min_auc_combo.addItems(["0.5", "0.6", "0.7", "0.8", "0.9"])
        self.min_auc_combo.setCurrentText("0.6")
        self.min_auc_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self.min_auc_combo)

        ctrl.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Single", "Combined", "All"])
        self.type_combo.setCurrentText("Single")
        self.type_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self.type_combo)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.heatmap_plot = pg.GraphicsLayoutWidget()
        self.heatmap_plot.setBackground("w")
        layout.addWidget(self.heatmap_plot)

        self.info_label = QLabel("Load analysis results to view heatmap")
        layout.addWidget(self.info_label)

    def set_analysis_run(self, run: AnalysisRun):
        self._current_run = run
        self._refresh()

    def _refresh(self):
        if self._current_run is None:
            return

        metric = self.metric_combo.currentText()
        min_auc = float(self.min_auc_combo.currentText())
        show_type = self.type_combo.currentText()

        results = self._current_run.results

        if show_type == "Single":
            results = [r for r in results if "+" not in r.biomarker_name]
        elif show_type == "Combined":
            results = [r for r in results if "+" in r.biomarker_name]

        results = [r for r in results if r.auc >= min_auc]

        if not results:
            self.info_label.setText("No results match the current filters")
            return

        biomarker_names = sorted(set(r.biomarker_name for r in results))
        pathology_names = sorted(set(r.pathology_name for r in results))

        bm_idx = {name: i for i, name in enumerate(biomarker_names)}
        pt_idx = {name: i for i, name in enumerate(pathology_names)}

        matrix = np.full((len(pathology_names), len(biomarker_names)), np.nan)

        for r in results:
            bi = bm_idx.get(r.biomarker_name)
            pi = pt_idx.get(r.pathology_name)
            if bi is None or pi is None:
                continue

            if metric == "AUC":
                matrix[pi, bi] = r.auc
            elif metric == "Sensitivity":
                matrix[pi, bi] = r.optimal_sensitivity
            elif metric == "Specificity":
                matrix[pi, bi] = r.optimal_specificity
            elif metric == "PPV":
                matrix[pi, bi] = r.optimal_ppv
            elif metric == "NPV":
                matrix[pi, bi] = r.optimal_npv
            elif metric == "p-value (-log10)":
                if r.p_value_corrected is not None and r.p_value_corrected > 0:
                    matrix[pi, bi] = -np.log10(r.p_value_corrected)
                else:
                    matrix[pi, bi] = 0

        self.heatmap_plot.clear()

        n_bm = len(biomarker_names)
        n_pt = len(pathology_names)

        if n_bm == 0 or n_pt == 0:
            self.info_label.setText("Empty matrix")
            return

        max_cols = 30
        max_rows = 20

        if n_bm > max_cols:
            biomarker_names = biomarker_names[:max_cols]
            matrix = matrix[:, :max_cols]
            n_bm = max_cols
        if n_pt > max_rows:
            pathology_names = pathology_names[:max_rows]
            matrix = matrix[:max_rows, :]
            n_pt = max_rows

        plot = self.heatmap_plot.addPlot()

        img = pg.ImageItem()
        valid = matrix[~np.isnan(matrix)]
        if len(valid) == 0:
            self.info_label.setText("No valid data")
            return

        vmin = float(np.min(valid))
        vmax = float(np.max(valid))

        display = matrix.copy()
        nan_mask = np.isnan(display)
        display[nan_mask] = vmin - 1

        cmap = pg.colormap.get("viridis")
        img.setImage(display)
        img.setLevels([vmin, vmax])
        img.setLookupTable(cmap.getLookupTable(nPts=256))

        img_rect = pg.QtCore.QRectF(-0.5, -0.5, n_bm, n_pt)
        img.setRect(img_rect)
        plot.addItem(img)

        plot.getAxis("bottom").setTicks(
            [list(enumerate(biomarker_names))]
        )
        plot.getAxis("left").setTicks(
            [list(enumerate(pathology_names))]
        )

        color_map_item = pg.ColorBarItem(
            values=(vmin, vmax),
            colorMap=cmap,
        )
        color_map_item.setImageItem(img)
        self.heatmap_plot.addItem(color_map_item, row=1, col=0)

        n_significant = int(np.sum(matrix >= min_auc))
        self.info_label.setText(
            f"{metric} Heatmap: {n_bm} biomarkers x {n_pt} pathologies "
            f"| {n_significant}/{matrix.size} cells above {min_auc}"
        )
