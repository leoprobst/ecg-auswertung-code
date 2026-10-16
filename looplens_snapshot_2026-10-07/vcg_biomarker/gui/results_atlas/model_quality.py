"""Manifest-bound, descriptive calibration and decision-curve atlas panel."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import duckdb
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultPage
from ...mimic.storage import StageManifest
from .export import choose_plot_export, choose_table_export
from .row_scope import validation_output_assurance


@dataclass(frozen=True)
class ModelQualitySnapshot:
    status: str
    message: str
    calibration: pd.DataFrame
    decision: pd.DataFrame
    endpoint_id: str
    run_id: str = ""
    snapshot_hash: str = ""
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


def _read_points(path: Path, endpoint_id: str, required: set[str], limit: int) -> pd.DataFrame:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        columns = {column[0] for column in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)]
        ).description}
        if not required.issubset(columns):
            return pd.DataFrame()
        # Never silently mix a training curve or another clinical question.
        count = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?) WHERE endpoint_id = ? "
            "AND evaluation_split = 'validation'", [str(path), endpoint_id]
        ).fetchone()[0]
        if count > limit:
            return pd.DataFrame()
        return connection.execute(
            "SELECT * FROM read_parquet(?) WHERE endpoint_id = ? "
            "AND evaluation_split = 'validation' ORDER BY model_id",
            [str(path), endpoint_id],
        ).fetchdf()
    finally:
        connection.close()


def load_model_quality(page: ResultPage, endpoint_id: str) -> ModelQualitySnapshot:
    empty = pd.DataFrame()
    if page.status not in {"ok", "empty"} or page.artifact_path is None or not endpoint_id:
        return ModelQualitySnapshot("unavailable", "Kein kompatibler abgeschlossener Endpunktstand.", empty, empty, endpoint_id)
    run_root = page.artifact_path.parent
    try:
        manifest = StageManifest.load(run_root / "validate.manifest.json")
    except (OSError, TypeError, ValueError):
        return ModelQualitySnapshot("unavailable", "Validierungsmanifest fehlt oder ist beschädigt.", empty, empty, endpoint_id)
    if manifest.status != "complete":
        return ModelQualitySnapshot("unavailable", "Validierung noch nicht abgeschlossen.", empty, empty, endpoint_id)
    calibration_path = run_root / "calibration_curve_points.parquet"
    decision_path = run_root / "decision_curve_results.parquet"
    metrics_path = run_root / "adjusted_model_results.parquet"
    calibration_assurance = validation_output_assurance(manifest, calibration_path, page.run_id)
    decision_assurance = validation_output_assurance(manifest, decision_path, page.run_id)
    metrics_assurance = validation_output_assurance(manifest, metrics_path, page.run_id)
    calibration = (
        _read_points(calibration_path, endpoint_id, {
            "endpoint_id", "model_id", "evaluation_split", "mean_predicted",
            "observed_rate", "observed_ci_low", "observed_ci_high", "n", "events",
        }, 5000)
        if calibration_assurance is not None else empty
    )
    decision = (
        _read_points(decision_path, endpoint_id, {
            "endpoint_id", "model_id", "evaluation_split", "threshold_probability",
            "net_benefit", "treat_all_net_benefit", "treat_none_net_benefit",
            "evidence_scope", "clinical_utility_evidence_eligible",
        }, 5000)
        if decision_assurance is not None else empty
    )
    metrics = (
        _read_points(metrics_path, endpoint_id, {
            "endpoint_id", "model_id", "evaluation_split", "n_subjects",
            "roc_auc", "brier_score", "calibration_intercept", "calibration_slope",
            "inference_status",
        }, 500)
        if metrics_assurance is not None else empty
    )
    if calibration.empty and decision.empty and metrics.empty:
        return ModelQualitySnapshot("missing", "Für diesen Endpunkt liegen keine manifestierten Validierungs-Kurven oder Modellkennzahlen vor.", empty, empty, endpoint_id)
    return ModelQualitySnapshot(
        "ok", "Deskriptive interne Validation; keine externe klinische Nutzenprüfung. "
        + ("Mindestens eine Datei nur im abgeschlossenen Manifest deklariert; Ausgabe-SHA-256 fehlt."
        if "manifest_only" in {calibration_assurance, decision_assurance, metrics_assurance} else
           "Angezeigte Dateien SHA-256-geprüft."),
        calibration, decision, endpoint_id, page.run_id, page.snapshot_hash, metrics,
    )


class AtlasModelQualityWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: ModelQualitySnapshot | None = None
        layout = QVBoxLayout(self)
        self.note = QLabel("Kalibration und Entscheidungsnutzen: noch nicht geladen.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.tabs = QTabWidget()
        self.calibration_plot = pg.PlotWidget()
        self.calibration_plot.setLabel("bottom", "Mittlere vorhergesagte Wahrscheinlichkeit")
        self.calibration_plot.setLabel("left", "Beobachteter Anteil")
        self.decision_plot = pg.PlotWidget()
        self.decision_plot.setLabel("bottom", "Schwellenwahrscheinlichkeit")
        self.decision_plot.setLabel("left", "Net Benefit")
        self.tabs.addTab(self.calibration_plot, "Kalibration")
        self.tabs.addTab(self.decision_plot, "Entscheidungsnutzen")
        self.tabs.currentChanged.connect(self._populate_table)
        layout.addWidget(self.tabs)
        self.points_table = QTableWidget(0, 0)
        self.points_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.points_table.setAlternatingRowColors(True)
        layout.addWidget(self.points_table)
        self.metrics_note = QLabel(
            "Modellkennzahlen: nicht berechnet. Brier ist ein Vorhersagefehler; "
            "Kalibrations-Intercept 0 und -Slope 1 sind theoretische Ideale der geprüften Population."
        )
        self.metrics_note.setWordWrap(True)
        layout.addWidget(self.metrics_note)
        self.metrics_table = QTableWidget(0, 7)
        self.metrics_table.setHorizontalHeaderLabels((
            "Modell", "Patienten", "AUC", "Brier", "Kal.-Intercept", "Kal.-Slope", "Status",
        ))
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setMinimumHeight(150)
        layout.addWidget(self.metrics_table)
        actions = QHBoxLayout()
        plot_export = QPushButton("Aktuelles Diagramm als PNG/SVG")
        plot_export.setToolTip("Exportiert nur tatsächlich berechnete Kurven mit separater Herkunftsdatei.")
        plot_export.clicked.connect(self._export_plot)
        actions.addWidget(plot_export)
        table_export = QPushButton("Kurvenpunkte als CSV")
        table_export.setToolTip("Exportiert die zum gewählten Diagramm gehörenden Validation-Punkte.")
        table_export.clicked.connect(self._export_table)
        actions.addWidget(table_export)
        metrics_export = QPushButton("Modellkennzahlen als CSV")
        metrics_export.setToolTip(
            "Exportiert manifestierte interne Validation-Kennzahlen mit Herkunftsdatei; "
            "keine externe Kalibration oder klinische Freigabe."
        )
        metrics_export.clicked.connect(self._export_metrics)
        actions.addWidget(metrics_export)
        actions.addStretch(1)
        layout.addLayout(actions)

    def set_snapshot(self, snapshot: ModelQualitySnapshot) -> None:
        self._snapshot = snapshot
        self.calibration_plot.clear()
        self.decision_plot.clear()
        self.note.setText(snapshot.message)
        self._populate_metrics(snapshot.metrics)
        self._populate_table()
        if snapshot.status != "ok":
            return
        calibration = snapshot.calibration
        if not calibration.empty:
            plotted = False
            for index, (model_id, rows) in enumerate(calibration.groupby("model_id", sort=True)):
                rows = rows.sort_values("mean_predicted")
                x = pd.to_numeric(rows["mean_predicted"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(rows["observed_rate"], errors="coerce").to_numpy(dtype=float)
                lo = pd.to_numeric(rows["observed_ci_low"], errors="coerce").to_numpy(dtype=float)
                hi = pd.to_numeric(rows["observed_ci_high"], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
                if not valid.any():
                    continue
                plotted = True
                color = pg.intColor(index, hues=8)
                self.calibration_plot.plot(x[valid], y[valid], pen=pg.mkPen(color, width=2),
                                           symbol="o", symbolBrush=color, name=str(model_id))
                ci_valid = valid & np.isfinite(lo) & np.isfinite(hi) & (0 <= lo) & (lo <= y) & (y <= hi) & (hi <= 1)
                if ci_valid.any():
                    self.calibration_plot.addItem(pg.ErrorBarItem(
                        x=x[ci_valid], y=y[ci_valid], bottom=y[ci_valid] - lo[ci_valid],
                        top=hi[ci_valid] - y[ci_valid], beam=.01, pen=pg.mkPen(color),
                    ))
            if plotted:
                self.calibration_plot.plot([0, 1], [0, 1],
                                           pen=pg.mkPen("#89939b", style=Qt.PenStyle.DashLine))
                self.calibration_plot.setXRange(0, 1, padding=0)
                self.calibration_plot.setYRange(0, 1, padding=0)
            else:
                self.calibration_plot.setTitle("Keine gültigen Kalibrationspunkte")
        else:
            self.calibration_plot.setTitle("Nicht berechnet")
        decision = snapshot.decision
        if not decision.empty:
            for index, (model_id, rows) in enumerate(decision.groupby("model_id", sort=True)):
                rows = rows.sort_values("threshold_probability")
                x = pd.to_numeric(rows["threshold_probability"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(rows["net_benefit"], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (x < 1)
                if valid.any():
                    self.decision_plot.plot(x[valid], y[valid],
                                            pen=pg.mkPen(pg.intColor(index, hues=8), width=2),
                                            name=str(model_id))
            reference = decision.sort_values("threshold_probability").drop_duplicates("threshold_probability")
            x = pd.to_numeric(reference["threshold_probability"], errors="coerce").to_numpy(dtype=float)
            all_values = pd.to_numeric(reference["treat_all_net_benefit"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(all_values) & (x > 0) & (x < 1)
            if valid.any():
                self.decision_plot.plot(x[valid], all_values[valid],
                                        pen=pg.mkPen("#89939b", style=Qt.PenStyle.DashLine),
                                        name="Alle behandeln")
                self.decision_plot.plot(x[valid], np.zeros(int(valid.sum())),
                                        pen=pg.mkPen("#89939b", style=Qt.PenStyle.DotLine),
                                        name="Niemand behandeln")
            self.note.setText(snapshot.message + " Die Net-Benefit-Kurve ist ausdrücklich keine Behandlungsempfehlung.")
        else:
            self.decision_plot.setTitle("Nicht berechnet")

    def _populate_metrics(self, frame: pd.DataFrame) -> None:
        self.metrics_table.setRowCount(min(len(frame), 50))
        if frame.empty:
            self.metrics_note.setText("Brier, Kalibrations-Intercept und -Slope sind hier nicht berechnet.")
            return
        self.metrics_note.setText(
            f"{len(frame)} Validation-Modellzeilen; höchstens 50 sichtbar. "
            "Brier ist ein Vorhersagefehler (kleiner günstiger), kein Nutzenmaß. "
            "Intercept 0 und Slope 1 sind Ideale für dieselbe Zielpopulation; "
            "interne Kalibration belegt keine externe Übertragbarkeit."
        )

        def number(value: object) -> str:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return "nicht berechnet"
            return f"{numeric:.3f}" if math.isfinite(numeric) else "nicht berechnet"

        for index, (_, row) in enumerate(frame.head(50).iterrows()):
            subjects = number(row.get("n_subjects"))
            if subjects != "nicht berechnet":
                subjects = str(int(float(row["n_subjects"])))
            values = (
                str(row.get("model_id", "?")), subjects, number(row.get("roc_auc")),
                number(row.get("brier_score")), number(row.get("calibration_intercept")),
                number(row.get("calibration_slope")), str(row.get("inference_status", "unbekannt")),
            )
            for column, value in enumerate(values):
                self.metrics_table.setItem(index, column, QTableWidgetItem(value))

    def _current_frame(self) -> pd.DataFrame:
        if self._snapshot is None or self._snapshot.status != "ok":
            return pd.DataFrame()
        return self._snapshot.calibration if self.tabs.currentIndex() == 0 else self._snapshot.decision

    def _populate_table(self, *_args) -> None:
        frame = self._current_frame()
        columns = (
            ("model_id", "calibration_bin", "n", "events", "mean_predicted",
             "observed_rate", "observed_ci_low", "observed_ci_high")
            if self.tabs.currentIndex() == 0 else
            ("model_id", "threshold_probability", "net_benefit",
             "treat_all_net_benefit", "treat_none_net_benefit", "evidence_scope")
        )
        available = [column for column in columns if column in frame]
        self.points_table.setColumnCount(len(available))
        self.points_table.setHorizontalHeaderLabels(available)
        shown = frame.head(100)
        self.points_table.setRowCount(len(shown))
        for row_index, (_, row) in enumerate(shown.iterrows()):
            for column_index, name in enumerate(available):
                self.points_table.setItem(row_index, column_index, QTableWidgetItem(str(row[name])))

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "snapshot_hash": snapshot.snapshot_hash if snapshot else None,
            "endpoint_id": snapshot.endpoint_id if snapshot else None,
            "split": "validation",
            "view": "calibration" if self.tabs.currentIndex() == 0 else "decision_curve",
            "note": "Interne deskriptive Forschungsvalidierung; keine Therapieempfehlung.",
        }

    def _export_plot(self, *_args) -> None:
        if self._current_frame().empty:
            return
        plot = self.calibration_plot if self.tabs.currentIndex() == 0 else self.decision_plot
        choose_plot_export(self, plot, self._metadata(), proposed_name="modellguete_validation.png")

    def _export_table(self, *_args) -> None:
        frame = self._current_frame()
        if frame.empty:
            return
        choose_table_export(self, frame, self._metadata(), proposed_name="modellguete_validation.csv")

    def _export_metrics(self, *_args) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.metrics.empty:
            return
        choose_table_export(self, snapshot.metrics, {
            **self._metadata(), "view": "validation_model_metrics",
        }, proposed_name="modellkennzahlen_validation.csv")
