"""Cohort-specific external prognosis and measurement transport, never pooled."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultRepository
from ...mimic.storage import verify_output_integrity
from .export import choose_plot_export, choose_table_export


@dataclass(frozen=True)
class TransportSnapshot:
    status: str
    message: str
    coverage: pd.DataFrame
    mortality: pd.DataFrame
    run_id: str
    manifest_hash: str


def _declared_verified(root: Path, manifest: dict, name: str) -> Path | None:
    path = root / name
    resolved = str(path.resolve())
    if resolved not in {str(Path(item).resolve()) for item in manifest.get("output_paths", [])}:
        return None
    records = [item for item in manifest.get("output_integrity", [])
               if str(Path(str(item.get("path", ""))).resolve()) == resolved]
    return path if len(records) == 1 and verify_output_integrity(records, mode="sha256") else None


def load_transport_snapshot(repository: ResultRepository, run_id: str) -> TransportSnapshot:
    empty = pd.DataFrame()
    descriptor = next((item for item in repository.available_runs() if item.run_id == run_id), None)
    if descriptor is None or not descriptor.complete_compatible:
        return TransportSnapshot("pending", "Noch kein kompatibler abgeschlossener MIMIC-Ergebnisstand.", empty, empty, run_id, "")
    root = repository.config.external_validation_root / run_id / "population_cohorts"
    path = root / "population_cohorts.manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return TransportSnapshot("missing", "Für diesen Run liegt noch keine externe Populationsanalyse vor.", empty, empty, run_id, "")
    if (manifest.get("status") != "complete" or manifest.get("run_id") != run_id
            or manifest.get("feature_implementation_hash") != repository.expected_feature_code_hash
            or manifest.get("locked_test_read") is not False):
        return TransportSnapshot("incompatible", "Externe Analyse ist unvollständig oder nicht kompatibel.", empty, empty, run_id, "")
    coverage_path = _declared_verified(root, manifest, "coverage.parquet")
    mortality_path = _declared_verified(root, manifest, "mortality_associations.parquet")
    if coverage_path is None and mortality_path is None:
        return TransportSnapshot("missing", "Keine hashgeprüften externen Darstellungsartefakte.", empty, empty, run_id, str(manifest.get("input_hash", "")))
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        coverage = empty
        if coverage_path is not None:
            schema = {item[0] for item in connection.execute(
                "SELECT * FROM read_parquet(?) LIMIT 0", [str(coverage_path)]
            ).description}
            if {"cohort", "marker_id", "availability"}.issubset(schema):
                coverage = connection.execute(
                    "SELECT CAST(cohort AS VARCHAR) AS cohort, "
                    "COUNT(DISTINCT marker_id) AS marker_count, "
                    "MEDIAN(TRY_CAST(availability AS DOUBLE)) AS median_availability "
                    "FROM read_parquet(?) GROUP BY cohort ORDER BY cohort LIMIT 20",
                    [str(coverage_path)],
                ).fetchdf()
        mortality = empty
        if mortality_path is not None:
            schema = {item[0] for item in connection.execute(
                "SELECT * FROM read_parquet(?) LIMIT 0", [str(mortality_path)]
            ).description}
            required = {"cohort", "marker_id", "status", "hazard_ratio_per_sd", "ci_low", "ci_high", "n", "events", "outcome"}
            if required.issubset(schema):
                q = "q_value_cohort" if "q_value_cohort" in schema else "NULL AS q_value_cohort"
                q_sort = "TRY_CAST(q_value_cohort AS DOUBLE)" if "q_value_cohort" in schema else "NULL"
                mortality = connection.execute(
                    "WITH ranked AS (SELECT cohort, marker_id, outcome, status, "
                    f"hazard_ratio_per_sd, ci_low, ci_high, n, events, {q}, "
                    "ROW_NUMBER() OVER (PARTITION BY cohort "
                    f"ORDER BY {q_sort} ASC NULLS LAST, marker_id) AS cohort_rank "
                    "FROM read_parquet(?)) "
                    "SELECT cohort, marker_id, outcome, status, hazard_ratio_per_sd, "
                    "ci_low, ci_high, n, events, q_value_cohort "
                    "FROM ranked WHERE cohort_rank <= 100 "
                    "ORDER BY cohort, cohort_rank",
                    [str(mortality_path)],
                ).fetchdf()
        if coverage.empty and mortality.empty:
            return TransportSnapshot("empty", "Externe Artefakte sind vorhanden, aber keine darstellbaren Kohortenwerte.", empty, empty, run_id, str(manifest.get("input_hash", "")))
        return TransportSnapshot(
            "ok", "Externe Kohorten separat: Messübertragbarkeit und Prognose, keine unabhängige MIMIC-Diagnosebestätigung.",
            coverage, mortality, run_id, str(manifest.get("input_hash", "")),
        )
    finally:
        connection.close()


class AtlasTransportWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: TransportSnapshot | None = None
        layout = QVBoxLayout(self)
        self.note = QLabel("Externe Kohorten werden geprüft.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.coverage_table = QTableWidget(0, 3)
        self.coverage_table.setHorizontalHeaderLabels(("Kohorte", "Marker", "Mediane Messverfügbarkeit"))
        self.coverage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.coverage_table.setMaximumHeight(180)
        layout.addWidget(self.coverage_table)
        self.mortality_note = QLabel("Mortalitätsassoziationen: nicht geladen.")
        self.mortality_note.setWordWrap(True)
        layout.addWidget(self.mortality_note)
        self.cohort = QComboBox()
        self.cohort.setToolTip("Kohorten einzeln betrachten; Werte werden nicht zusammengepoolt.")
        self.cohort.currentIndexChanged.connect(self._draw_mortality)
        layout.addWidget(self.cohort)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "log₂ Hazard Ratio je SD")
        self.plot.setMinimumHeight(260)
        layout.addWidget(self.plot)
        self.mortality_table = QTableWidget(0, 7)
        self.mortality_table.setHorizontalHeaderLabels((
            "Kohorte", "Marker", "Ziel", "Status", "HR (95-%-CI)",
            "Personen / Ereignisse", "q je Kohorte",
        ))
        self.mortality_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.mortality_table)
        actions = QHBoxLayout()
        plot_button = QPushButton("Prognosegrafik als PNG/SVG")
        plot_button.setToolTip("Exportiert die kohortengetrennten HR-Punkte mit Provenienz; keine klinische Empfehlung.")
        plot_button.clicked.connect(self._export_plot)
        actions.addWidget(plot_button)
        mortality_button = QPushButton("Prognosewerte als CSV")
        mortality_button.setToolTip("Exportiert die geladenen externen Prognosewerte samt Herkunft.")
        mortality_button.clicked.connect(self._export_mortality)
        actions.addWidget(mortality_button)
        coverage_button = QPushButton("Messverfügbarkeit als CSV")
        coverage_button.setToolTip("Exportiert die kohortenweise aggregierte Messverfügbarkeit samt Herkunft.")
        coverage_button.clicked.connect(self._export_coverage)
        actions.addWidget(coverage_button)
        layout.addLayout(actions)

    def set_snapshot(self, snapshot: TransportSnapshot) -> None:
        self._snapshot = snapshot
        self.plot.clear()
        self.coverage_table.setRowCount(0)
        self.mortality_table.setRowCount(0)
        self.note.setText(snapshot.message)
        self.cohort.blockSignals(True)
        previous = self.cohort.currentData()
        self.cohort.clear()
        if snapshot.status != "ok":
            self.cohort.blockSignals(False)
            self.mortality_note.setText("Keine kompatiblen externen Prognosewerte für diesen Run.")
            return
        self.coverage_table.setRowCount(len(snapshot.coverage))
        for index, (_, row) in enumerate(snapshot.coverage.iterrows()):
            availability = pd.to_numeric(row.get("median_availability"), errors="coerce")
            values = (str(row["cohort"]), str(int(row["marker_count"])),
                      f"{availability:.1%}" if pd.notna(availability) else "nicht angegeben")
            for column, value in enumerate(values):
                self.coverage_table.setItem(index, column, QTableWidgetItem(value))
        for cohort in sorted(snapshot.mortality.cohort.dropna().astype(str).unique()):
            self.cohort.addItem(cohort, cohort)
        selected_index = self.cohort.findData(previous)
        if selected_index >= 0:
            self.cohort.setCurrentIndex(selected_index)
        self.cohort.blockSignals(False)
        self._draw_mortality()

    def _draw_mortality(self, _index: int = -1) -> None:
        self.plot.clear()
        self.mortality_table.setRowCount(0)
        snapshot = self._snapshot
        if snapshot is None or snapshot.status != "ok":
            return
        cohort = self.cohort.currentData()
        frame = snapshot.mortality.loc[snapshot.mortality.cohort.astype(str).eq(str(cohort))]
        rows = frame.head(25)
        self.mortality_table.setRowCount(len(rows))
        self.mortality_note.setText(
            f"{cohort}: {len(rows)} von {len(frame)} geladenen intern adjustierten "
            "Mortalitätsauswertungen; maximal 25 sichtbar. Nicht schätzbare Modelle "
            "bleiben in der Tabelle, aber nicht im Forest-Diagramm. "
            "HR ist weder individuelle Erkrankungswahrscheinlichkeit noch Behandlungsnutzen. "
            "Die Auswahl zeigt keine unabhängige Replikation eines MIMIC-Echo-/Labor-Endpunkts."
        )
        ticks = []
        for index, (_, row) in enumerate(rows.iterrows()):
            hr, low, high = (pd.to_numeric(row.get(name), errors="coerce")
                             for name in ("hazard_ratio_per_sd", "ci_low", "ci_high"))
            y = len(rows) - index
            ticks.append((y, f"{row['cohort']} · {row['marker_id']}"[:42]))
            valid = (str(row["status"]) == "estimable"
                     and all(pd.notna(value) and math.isfinite(value) and value > 0
                             for value in (hr, low, high)) and low <= hr <= high)
            if valid:
                x, left, right = math.log2(hr), math.log2(low), math.log2(high)
                self.plot.plot([left, right], [y, y], pen=pg.mkPen("#29a3a3", width=2))
                self.plot.plot([x], [y], pen=None, symbol="o", symbolBrush="#29a3a3")
            q = pd.to_numeric(row.get("q_value_cohort"), errors="coerce")
            values = (str(row["cohort"]), str(row["marker_id"]), str(row["outcome"]),
                      str(row["status"]),
                      f"{hr:.2f} ({low:.2f}–{high:.2f})" if valid else "nicht schätzbar",
                      f"{row['n']} / {row['events']}", f"{q:.3g}" if pd.notna(q) else "nicht berechnet")
            for column, value in enumerate(values):
                self.mortality_table.setItem(index, column, QTableWidgetItem(value))
        self.plot.getAxis("left").setTicks([ticks])
        if len(rows):
            self.plot.addLine(x=0, pen=pg.mkPen("#89939b"))
            self.plot.setYRange(.5, len(rows) + .5, padding=0)

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "external_manifest_input_hash": snapshot.manifest_hash if snapshot else None,
            "scope": "external_population_cohorts",
            "selected_cohort": self.cohort.currentData(),
            "note": "Kohortengetrennte Forschungsassoziationen; keine MIMIC-Diagnosebestätigung oder Behandlungsempfehlung.",
        }

    def _export_plot(self, *_args) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.mortality.empty:
            return
        choose_plot_export(self, self.plot, self._metadata(), proposed_name="externe_prognose.png")

    def _export_mortality(self, *_args) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.mortality.empty:
            return
        frame = snapshot.mortality.loc[
            snapshot.mortality.cohort.astype(str).eq(str(self.cohort.currentData()))
        ]
        if not frame.empty:
            choose_table_export(self, frame, self._metadata(), proposed_name="externe_prognose.csv")

    def _export_coverage(self, *_args) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.coverage.empty:
            return
        choose_table_export(self, snapshot.coverage, self._metadata(), proposed_name="externe_messverfuegbarkeit.csv")
