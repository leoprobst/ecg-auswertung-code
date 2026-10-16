"""Manifest-verified, observational survival results for the research atlas."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultPage
from ...mimic.storage import StageManifest, verify_output_integrity
from ...mimic.survival import SURVIVAL_VERSION
from .export import choose_plot_export, choose_table_export


@dataclass(frozen=True)
class PrognosisSnapshot:
    status: str
    message: str
    fits: pd.DataFrame
    curves: pd.DataFrame
    run_id: str


def load_prognosis(page: ResultPage) -> PrognosisSnapshot:
    empty = pd.DataFrame()
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return PrognosisSnapshot("pending", "Kein kompatibler abgeschlossener Ergebnisstand.", empty, empty, page.run_id)
    root = page.artifact_path.parent / "survival"
    try:
        manifest = StageManifest.load(root / "time_to_event.manifest.json")
    except (OSError, TypeError, ValueError):
        return PrognosisSnapshot("pending", "Zeit-bis-Ereignis-Analyse noch nicht abgeschlossen.", empty, empty, page.run_id)
    paths = (root / "time_to_event.parquet", root / "time_to_event_curves.parquet")
    declared = {str(Path(path).resolve()) for path in manifest.output_paths}
    records = [record for record in manifest.output_integrity
               if str(Path(str(record.get("path", ""))).resolve()) in
               {str(path.resolve()) for path in paths}]
    if (manifest.status != "complete" or manifest.stage != "time-to-event"
            or manifest.stage_version != SURVIVAL_VERSION
            or not all(str(path.resolve()) in declared for path in paths)
            or len(records) != 2 or not verify_output_integrity(records, mode="sha256")):
        return PrognosisSnapshot("incompatible", "Prognoseartefakte sind unvollständig, inkompatibel oder nicht hashgeprüft.", empty, empty, page.run_id)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        fits = connection.execute(
            "SELECT feature_id, population, status, n_patients, n_events, "
            "hazard_ratio_per_sd, hr_lower, hr_upper, q_value_global "
            "FROM read_parquet(?) WHERE population = 'first_ecg_per_patient' "
            "ORDER BY feature_id LIMIT 100", [str(paths[0])],
        ).fetchdf()
        marker_ids = fits.feature_id.dropna().astype(str).unique().tolist()
        curves = empty
        if marker_ids:
            placeholders = ",".join("?" for _ in marker_ids)
            curves = connection.execute(
                "WITH ordered AS (SELECT feature_id, \"group\", group_label, n, n_events, "
                "time_days, cumulative_incidence, lower_bound, upper_bound, "
                "ROW_NUMBER() OVER (PARTITION BY feature_id, \"group\" ORDER BY time_days) AS point_no, "
                "COUNT(*) OVER (PARTITION BY feature_id, \"group\") AS group_points "
                f"FROM read_parquet(?) WHERE feature_id IN ({placeholders})), "
                "sampled AS (SELECT * FROM ordered WHERE group_points <= 120 "
                "OR point_no = 1 OR point_no = group_points "
                "OR MOD(point_no - 1, GREATEST(1, CAST(CEIL(group_points / 120.0) AS INTEGER))) = 0) "
                "SELECT feature_id, \"group\", group_label, n, n_events, time_days, "
                "cumulative_incidence, lower_bound, upper_bound "
                "FROM sampled ORDER BY feature_id, \"group\", time_days",
                [str(paths[1]), *marker_ids],
            ).fetchdf()
    except duckdb.Error:
        return PrognosisSnapshot("invalid", "Prognosetabellen sind nicht lesbar oder haben ein inkompatibles Schema.", empty, empty, page.run_id)
    finally:
        connection.close()
    if fits.empty:
        return PrognosisSnapshot("empty", "Keine auswertbaren Prognosemodelle vorhanden.", fits, curves, page.run_id)
    return PrognosisSnapshot(
        "ok", "Beobachtete Sterblichkeit ab Index-EKG; keine Behandlungswirkung oder "
        "individuelle Risikokalibration. Cox-Assoziationen: Development, erstes EKG "
        "pro Patient; Kurven: Validation mit Train-fixierten Terzilen. "
        "Für die Anzeige bleiben je Terzil höchstens etwa 121 Zeitpunkte einschließlich "
        "Anfang und Ende erhalten; dies sind nicht alle Originalpunkte.",
        fits, curves, page.run_id,
    )


class AtlasPrognosisWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: PrognosisSnapshot | None = None
        layout = QVBoxLayout(self)
        self.note = QLabel("Prognoseergebnisse werden geprüft.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.marker = QComboBox()
        self.marker.currentIndexChanged.connect(self._draw)
        layout.addWidget(self.marker)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Kumulative Sterblichkeit", units="%")
        self.plot.setLabel("bottom", "Tage seit Index-EKG")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend()
        layout.addWidget(self.plot, 1)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(("Marker", "Patienten", "Ereignisse", "HR pro SD (95-%-CI)", "globale q"))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        self.limits = QLabel(
            "Kurven: Validation, train-fixierte Markerterzile, erstes EKG pro Patient; "
            "Darstellung zeitpunktreduziert. Cox-HR: Development Train+Validation, "
            "nicht unabhängig auf Validation nachgeschätzt. "
            "Bänder: punktweise 95-%-Intervalle; kein simultanes Band. "
            "Hazard Ratios: beobachtende Assoziationen, kein kausaler Effekt. "
            "Wenige Ereignisse und unterschiedliche Follow-up-Zeiten begrenzen die Aussage."
        )
        self.limits.setWordWrap(True)
        layout.addWidget(self.limits)
        exports = QHBoxLayout()
        export_plot = QPushButton("Prognosekurve als PNG/SVG")
        export_plot.setToolTip("Exportiert nur die ausgewählte Validation-Kurve mit Provenienzdatei.")
        export_plot.clicked.connect(self._export_plot)
        exports.addWidget(export_plot)
        export_curves = QPushButton("Kurvenpunkte als CSV")
        export_curves.setToolTip("Exportiert die angezeigten Kurvenpunkte und punktweisen Intervalle.")
        export_curves.clicked.connect(self._export_curves)
        exports.addWidget(export_curves)
        export_fits = QPushButton("Hazard Ratios als CSV")
        export_fits.setToolTip("Exportiert die beobachtenden Cox-Assoziationen des ausgewählten Runs.")
        export_fits.clicked.connect(self._export_fits)
        exports.addWidget(export_fits)
        exports.addStretch(1)
        layout.addLayout(exports)

    def set_snapshot(self, snapshot: PrognosisSnapshot) -> None:
        self._snapshot = snapshot
        self.note.setText(snapshot.message)
        self.marker.blockSignals(True)
        previous = self.marker.currentData()
        self.marker.clear()
        if snapshot.status == "ok":
            for marker in sorted(snapshot.fits.feature_id.astype(str).unique()):
                self.marker.addItem(marker, marker)
            index = self.marker.findData(previous)
            if index >= 0:
                self.marker.setCurrentIndex(index)
        self.marker.blockSignals(False)
        self.table.setRowCount(0)
        if snapshot.status == "ok":
            for _, row in snapshot.fits.iterrows():
                i = self.table.rowCount()
                self.table.insertRow(i)
                hr = pd.to_numeric(row.get("hazard_ratio_per_sd"), errors="coerce")
                lo = pd.to_numeric(row.get("hr_lower"), errors="coerce")
                hi = pd.to_numeric(row.get("hr_upper"), errors="coerce")
                effect = f"{hr:.2f} ({lo:.2f}–{hi:.2f})" if np.isfinite([hr, lo, hi]).all() else "nicht schätzbar"
                q = pd.to_numeric(row.get("q_value_global"), errors="coerce")
                values = (str(row.feature_id), str(row.n_patients), str(row.n_events), effect,
                          f"{q:.3g}" if np.isfinite(q) else "nicht berechnet")
                for j, value in enumerate(values):
                    self.table.setItem(i, j, QTableWidgetItem(value))
        self._draw()

    def _draw(self, _index: int = -1) -> None:
        self.plot.clear()
        snapshot = self._snapshot
        if snapshot is None or snapshot.status != "ok" or snapshot.curves.empty:
            return
        marker = self.marker.currentData()
        frame = snapshot.curves.loc[snapshot.curves.feature_id.astype(str).eq(str(marker))]
        colours = ("#2171b5", "#f0a202", "#b23a48")
        for group, block in frame.groupby("group", sort=True):
            block = block.sort_values("time_days")
            if block.empty:
                continue
            colour = colours[(int(group) - 1) % len(colours)]
            x = block.time_days.to_numpy(dtype=float)
            y = block.cumulative_incidence.to_numpy(dtype=float) * 100
            low = block.lower_bound.to_numpy(dtype=float) * 100
            high = block.upper_bound.to_numpy(dtype=float) * 100
            valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (y <= 100)
            if not valid.any():
                continue
            x, y, low, high = x[valid], y[valid], low[valid], high[valid]
            ci_valid = (np.isfinite(low) & np.isfinite(high) & (low >= 0)
                        & (low <= y) & (y <= high) & (high <= 100))
            if ci_valid.all():
                lower = self.plot.plot(
                    x, low, stepMode="left",
                    pen=pg.mkPen(colour, width=1, style=Qt.PenStyle.DashLine),
                )
                upper = self.plot.plot(
                    x, high, stepMode="left",
                    pen=pg.mkPen(colour, width=1, style=Qt.PenStyle.DashLine),
                )
                self.plot.addItem(pg.FillBetweenItem(
                    lower, upper, brush=pg.mkBrush(*pg.mkColor(colour).getRgb()[:3], 28),
                ))
            self.plot.plot(x, y, stepMode="left", pen=pg.mkPen(colour, width=2),
                           name=f"{block.group_label.iloc[0]} (n={int(block.n.iloc[0])}, Ereignisse={int(block.n_events.iloc[0])})")

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "stage": "time-to-event", "stage_version": SURVIVAL_VERSION,
            "curves_split": "validation", "fits_split": "development_train_and_validation",
            "unit": "first_ecg_per_patient", "curve_points": "display_decimated_max_about_121_per_group",
            "marker_id": self.marker.currentData(),
            "claim_scope": "observational_prognosis_not_causal_treatment_effect",
        }

    def _export_plot(self, *_args) -> None:
        if self._snapshot is not None and not self._snapshot.curves.empty:
            choose_plot_export(self, self.plot, self._metadata(),
                               proposed_name="prognosekurve.png")

    def _export_curves(self, *_args) -> None:
        if self._snapshot is not None and not self._snapshot.curves.empty:
            frame = self._snapshot.curves.loc[
                self._snapshot.curves.feature_id.astype(str).eq(str(self.marker.currentData()))
            ]
            choose_table_export(self, frame, self._metadata(),
                                proposed_name="prognosekurvenpunkte.csv")

    def _export_fits(self, *_args) -> None:
        if self._snapshot is not None and not self._snapshot.fits.empty:
            choose_table_export(self, self._snapshot.fits, self._metadata(),
                                proposed_name="prognosemodelle.csv")
