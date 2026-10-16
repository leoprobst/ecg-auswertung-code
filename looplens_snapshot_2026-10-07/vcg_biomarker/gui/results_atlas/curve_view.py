"""Held-out ROC/PR curves and binned score distributions for one marker question."""
from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultPage
from ...mimic.storage import StageManifest
from .export import choose_plot_export, choose_table_export
from .row_scope import validation_output_assurance


@dataclass(frozen=True)
class CurveSnapshot:
    status: str
    note: str
    curves: pd.DataFrame
    distributions: pd.DataFrame
    endpoint_id: str
    marker_id: str
    operating_points: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_id: str = ""


def load_marker_curves(page: ResultPage, endpoint_id: str, marker_id: str) -> CurveSnapshot:
    empty = pd.DataFrame()
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return CurveSnapshot("pending", "Kein kompatibler abgeschlossener Ergebnisstand.",
                             empty, empty, endpoint_id, marker_id)
    root = page.artifact_path.parent
    curves_path = root / "performance_curve_points.parquet"
    bins_path = root / "score_distribution_bins.parquet"
    operating_path = root / "operating_point_results.parquet"
    try:
        manifest = StageManifest.load(root / "validate.manifest.json")
    except (OSError, TypeError, ValueError):
        return CurveSnapshot("pending", "Validierung noch nicht abgeschlossen.",
                             empty, empty, endpoint_id, marker_id)
    assurance = validation_output_assurance(manifest, curves_path, page.run_id)
    if assurance is None:
        return CurveSnapshot("incompatible", "ROC-/PR-Kurven sind nicht vollständig manifestiert oder ihre Integritätsprüfung ist fehlgeschlagen.",
                             empty, empty, endpoint_id, marker_id)
    bins_allowed = validation_output_assurance(manifest, bins_path, page.run_id) is not None
    operating_allowed = validation_output_assurance(manifest, operating_path, page.run_id) is not None
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='256MB'")
        connection.execute("SET threads=1")
        schema = {item[0] for item in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(curves_path)],
        ).description}
        required = {"endpoint_id", "feature_id", "evaluation_split", "point_kind",
                    "curve_type", "point_order", "x", "y", "n_positive", "n_negative"}
        if not required.issubset(schema):
            return CurveSnapshot("incompatible", "Kurventabelle hat keinen kompatiblen Vertrag.",
                                 empty, empty, endpoint_id, marker_id)
        curves = connection.execute(
            "SELECT curve_type, point_order, x, y, n_positive, n_negative, "
            "roc_auc, roc_auc_ci_low, roc_auc_ci_high, average_precision, "
            "average_precision_ci_low, average_precision_ci_high "
            "FROM read_parquet(?) WHERE endpoint_id = ? AND feature_id = ? "
            "AND evaluation_split = 'validation' "
            "AND point_kind = 'continuous_marker_curve' "
            "AND curve_type IN ('roc', 'precision_recall') "
            "ORDER BY curve_type, point_order LIMIT 1200",
            [str(curves_path), endpoint_id, marker_id],
        ).fetchdf()
        bins = empty
        if bins_allowed:
            bin_schema = {item[0] for item in connection.execute(
                "SELECT * FROM read_parquet(?) LIMIT 0", [str(bins_path)],
            ).description}
            needed = {"endpoint_id", "feature_id", "evaluation_split", "label",
                      "bin_left", "bin_right", "fraction_within_class", "class_total",
                      "display_transform"}
            if needed.issubset(bin_schema):
                bins = connection.execute(
                    "SELECT label, bin_left, bin_right, fraction_within_class, "
                    "class_total, display_transform FROM read_parquet(?) "
                    "WHERE endpoint_id = ? AND feature_id = ? "
                    "AND evaluation_split = 'validation' "
                    "ORDER BY label, bin_left LIMIT 100",
                    [str(bins_path), endpoint_id, marker_id],
                ).fetchdf()
        operating = empty
        if operating_allowed:
            operating_schema = {item[0] for item in connection.execute(
                "SELECT * FROM read_parquet(?) LIMIT 0", [str(operating_path)],
            ).description}
            needed = {"endpoint_id", "feature_id", "evaluation_split", "operating_point",
                      "threshold_source", "target_attainable", "sensitivity", "specificity",
                      "sensitivity_ci_low", "sensitivity_ci_high", "specificity_ci_low",
                      "specificity_ci_high", "ppv", "npv", "inference_status"}
            if needed.issubset(operating_schema):
                operating = connection.execute(
                    "SELECT operating_point, threshold_source, target_attainable, "
                    "sensitivity, specificity, sensitivity_ci_low, sensitivity_ci_high, "
                    "specificity_ci_low, specificity_ci_high, ppv, npv, inference_status "
                    "FROM read_parquet(?) WHERE endpoint_id = ? AND feature_id = ? "
                    "AND evaluation_split = 'validation' "
                    "AND operating_point IN ('sensitivity_at_90_specificity', "
                    "'sensitivity_at_95_specificity', 'specificity_at_90_sensitivity') "
                    "ORDER BY operating_point LIMIT 30",
                    [str(operating_path), endpoint_id, marker_id],
                ).fetchdf()
    except duckdb.Error:
        return CurveSnapshot("invalid", "Kurven- oder Verteilungstabelle nicht lesbar.",
                             empty, empty, endpoint_id, marker_id)
    finally:
        connection.close()
    if curves.empty:
        return CurveSnapshot("empty", "Für dieses Paar wurden keine Validation-Kurven gespeichert.",
                             empty, bins, endpoint_id, marker_id)
    if not bins.empty and not bins.display_transform.astype(str).eq(
            "direction_aligned_train_zscore_clipped_to_5sd").all():
        bins = empty
    return CurveSnapshot(
        "ok", "Validation-Kurven für den ausgewählten Marker; Punkte sind auf höchstens 300 je Kurventyp "
        "reduziert. AUC-/AP-Intervalle sind skalare Unsicherheit, keine Kurvenbänder. "
        + ("Datei SHA-256-geprüft." if assurance == "sha256" else
           "Datei nur im abgeschlossenen Validierungsmanifest deklariert; kein Ausgabe-SHA-256 vorhanden."),
        curves, bins, endpoint_id, marker_id,
        operating, page.run_id,
    )


class AtlasCurveWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: CurveSnapshot | None = None
        layout = QVBoxLayout(self)
        self.note = QLabel("ROC-/PR-Kurven werden geprüft.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.tabs = QTabWidget()
        self.roc = pg.PlotWidget()
        self.roc.setLabel("bottom", "1 − Spezifität")
        self.roc.setLabel("left", "Sensitivität")
        self.pr = pg.PlotWidget()
        self.pr.setLabel("bottom", "Sensitivität (Recall)")
        self.pr.setLabel("left", "Präzision")
        self.distribution = pg.PlotWidget()
        self.distribution.setLabel("bottom", "Train-standardisierter Markerscore (±5 SD gekappt)")
        self.distribution.setLabel("left", "Anteil innerhalb der Klasse")
        for plot, title in ((self.roc, "ROC"), (self.pr, "Präzision–Recall"),
                            (self.distribution, "Klassenverteilung")):
            plot.showGrid(x=True, y=True, alpha=0.15)
            self.tabs.addTab(plot, title)
        layout.addWidget(self.tabs)
        self.metric = QLabel("AUC, AP und Fallzahlen werden aus der Kurvendatei gelesen.")
        self.metric.setWordWrap(True)
        layout.addWidget(self.metric)
        self.operating_note = QLabel(
            "Betriebspunkte: Schwelle im Training festgelegt; tatsächliche Validation-Werte folgen."
        )
        self.operating_note.setWordWrap(True)
        layout.addWidget(self.operating_note)
        self.operating_table = QTableWidget(0, 6)
        self.operating_table.setHorizontalHeaderLabels((
            "Train-Ziel", "erreichte Sensitivität", "erreichte Spezifität",
            "PPV", "NPV", "Nachweis",
        ))
        self.operating_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.operating_table)
        exports = QHBoxLayout()
        for title, handler, tooltip in (
            ("Aktuelles Diagramm als PNG/SVG", self._export_plot,
             "Exportiert nur die sichtbare Kurven-/Verteilungsansicht mit Herkunftsdatei."),
            ("Kurvenpunkte als CSV", self._export_curves,
             "Exportiert die gespeicherten Validation-Punkte, keine erfundenen Kurvenbänder."),
            ("Betriebspunkte als CSV", self._export_operating,
             "Exportiert die auf Training festgelegten Schwellen und erreichten Validation-Werte."),
        ):
            button = QPushButton(title)
            button.setToolTip(tooltip)
            button.clicked.connect(handler)
            exports.addWidget(button)
        exports.addStretch(1)
        layout.addLayout(exports)

    def set_snapshot(self, snapshot: CurveSnapshot) -> None:
        self._snapshot = snapshot
        self.note.setText(snapshot.note)
        for plot in (self.roc, self.pr, self.distribution):
            plot.clear()
        self.metric.setText("Für dieses Paar liegen keine kompatiblen Kurvenmetriken vor.")
        self.operating_table.setRowCount(0)
        self.operating_note.setText(
            "Keine hashgeprüften Betriebspunkte für dieses Marker–Endpunkt-Paar vorhanden."
        )
        if snapshot.status != "ok":
            return
        for kind, plot in (("roc", self.roc), ("precision_recall", self.pr)):
            frame = snapshot.curves.loc[snapshot.curves.curve_type.eq(kind)]
            if frame.empty:
                continue
            frame = frame.sort_values("point_order")
            plot.plot(frame.x.to_numpy(dtype=float), frame.y.to_numpy(dtype=float),
                      pen=pg.mkPen("#238b9b", width=2))
            plot.setXRange(0, 1, padding=0)
            plot.setYRange(0, 1, padding=0)
        self.roc.plot([0, 1], [0, 1], pen=pg.mkPen("#999999", width=1, style=2))
        first = snapshot.curves.iloc[0]
        n_pos, n_neg = int(first.n_positive), int(first.n_negative)
        prevalence = n_pos / (n_pos + n_neg) if n_pos + n_neg else 0
        self.pr.plot([0, 1], [prevalence, prevalence],
                     pen=pg.mkPen("#999999", width=1, style=2))
        auc = pd.to_numeric(first.roc_auc, errors="coerce")
        ap = pd.to_numeric(first.average_precision, errors="coerce")
        self.metric.setText(
            f"Validation: {n_pos} Fälle, {n_neg} Kontrollen; "
            f"ROC-AUC {auc:.3f}, Average Precision {ap:.3f}. "
            f"PR-Referenz = Fallanteil {prevalence:.1%} in dieser ausgewerteten Stichprobe, "
            "nicht Bevölkerungsprävalenz. Skalar-CI siehe Markertabelle; keine Kurven-CI gespeichert."
        )
        bins = snapshot.distributions
        for label, colour in ((0, "#3182bd"), (1, "#d95f0e")):
            block = bins.loc[bins.label.eq(label)] if not bins.empty else bins
            if block.empty:
                continue
            x = ((block.bin_left + block.bin_right) / 2).to_numpy(dtype=float)
            y = block.fraction_within_class.to_numpy(dtype=float)
            self.distribution.plot(x, y, pen=pg.mkPen(colour, width=2),
                                   name="Fälle" if label else "Kontrollen")
        self._show_operating(snapshot.operating_points)

    @staticmethod
    def _percentage(value: object) -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        return f"{numeric:.1%}" if pd.notna(numeric) and 0 <= numeric <= 1 else "nicht berechnet"

    def _show_operating(self, points: pd.DataFrame) -> None:
        if points.empty:
            return
        self.operating_note.setText(
            "Train-Ziel und erreichte Validation-Spezifität sind verschieden. "
            "PPV/NPV gelten nur für die ausgewertete Stichprobe; keine Bevölkerungswahrscheinlichkeiten."
        )
        labels = {
            "sensitivity_at_90_specificity": "90 % Spezifität",
            "sensitivity_at_95_specificity": "95 % Spezifität",
            "specificity_at_90_sensitivity": "90 % Sensitivität",
        }
        self.operating_table.setRowCount(len(points))
        for i, (_, row) in enumerate(points.iterrows()):
            sensitivity = self._percentage(row.sensitivity)
            specificity = self._percentage(row.specificity)
            sens_ci = self._interval(row.sensitivity_ci_low, row.sensitivity_ci_high)
            spec_ci = self._interval(row.specificity_ci_low, row.specificity_ci_high)
            values = (
                labels.get(str(row.operating_point), str(row.operating_point)),
                sensitivity + (f" ({sens_ci})" if sens_ci else ""),
                specificity + (f" ({spec_ci})" if spec_ci else ""),
                self._percentage(row.ppv), self._percentage(row.npv),
                f"{row.threshold_source or 'unbekannt'}; {row.inference_status or 'unbekannt'}; "
                f"Train-Ziel {self._attainable(row.target_attainable)}",
            )
            for j, value in enumerate(values):
                self.operating_table.setItem(i, j, QTableWidgetItem(value))

    @classmethod
    def _interval(cls, low: object, high: object) -> str:
        a, b = (pd.to_numeric(value, errors="coerce") for value in (low, high))
        if pd.notna(a) and pd.notna(b) and 0 <= a <= b <= 1:
            return f"95-%-CI {cls._percentage(a)}–{cls._percentage(b)}"
        return ""

    @staticmethod
    def _attainable(value: object) -> str:
        if pd.isna(value):
            return "unklar"
        if isinstance(value, (bool,)) or type(value).__name__ == "bool_":
            return "erreichbar" if bool(value) else "nicht erreichbar"
        return "unklar"

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "endpoint_id": snapshot.endpoint_id if snapshot else None,
            "feature_id": snapshot.marker_id if snapshot else None,
            "split": "validation", "claim_scope": "descriptive_development_validation",
        }

    def _export_plot(self, *_args) -> None:
        if self._snapshot is not None and self._snapshot.status == "ok":
            choose_plot_export(self, self.tabs.currentWidget(), self._metadata(),
                               proposed_name="marker_kurve.png")

    def _export_curves(self, *_args) -> None:
        if self._snapshot is not None and not self._snapshot.curves.empty:
            choose_table_export(self, self._snapshot.curves, self._metadata(),
                                proposed_name="marker_kurvenpunkte.csv")

    def _export_operating(self, *_args) -> None:
        if self._snapshot is not None and not self._snapshot.operating_points.empty:
            choose_table_export(self, self._snapshot.operating_points, self._metadata(),
                                proposed_name="marker_betriebspunkte.csv")
