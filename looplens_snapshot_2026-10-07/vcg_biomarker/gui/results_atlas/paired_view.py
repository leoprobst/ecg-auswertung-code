"""Paired classical ECG comparisons, with fixed rules kept apart from AUCs."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import duckdb
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultPage
from ...mimic.storage import StageManifest
from .export import choose_plot_export, choose_table_export
from .row_scope import validation_output_assurance


@dataclass(frozen=True)
class PairedSnapshot:
    status: str
    note: str
    rows: pd.DataFrame
    endpoint_id: str = ""
    run_id: str = ""
    snapshot_hash: str = ""
    total_count: int = 0
    outcome_counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    focus_counts: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)


_COLUMNS = (
    "endpoint_id", "evaluation_split", "comparison_type", "vcg_feature_id",
    "baseline_feature_id", "baseline_registry_id", "inference_status", "paired_n",
    "paired_n_positive", "paired_n_negative", "delta_auc", "delta_auc_ci_low",
    "delta_auc_ci_high", "sensitivity_delta_vs_fixed_rule",
    "sensitivity_delta_ci_low", "sensitivity_delta_ci_high",
    "baseline_validation_specificity", "vcg_specificity_at_specificity_match",
    "comparison_verdict_de", "threshold_source", "reference_independence",
)
_REQUIRED = {
    "endpoint_id", "evaluation_split", "comparison_type", "vcg_feature_id",
    "baseline_feature_id", "inference_status", "paired_n",
}
_AUC_TYPES = {
    "paired_continuous_marker_discrimination",
    "paired_availability_marker_discrimination",
}
_FIXED_TYPE = "paired_fixed_rule_operating_point"


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def paired_effect(row: pd.Series) -> tuple[str, float | None, float | None, float | None]:
    """Never turn a fixed binary rule into a continuous AUC comparison."""
    comparison_type = str(row.get("comparison_type", ""))
    if comparison_type == _FIXED_TYPE:
        metric = "Sensitivitäts-Δ"
        fields = ("sensitivity_delta_vs_fixed_rule", "sensitivity_delta_ci_low",
                  "sensitivity_delta_ci_high")
    elif comparison_type in _AUC_TYPES:
        metric = "AUC-Δ"
        fields = ("delta_auc", "delta_auc_ci_low", "delta_auc_ci_high")
    else:
        return "Nicht unterstützter Vergleich", None, None, None
    effect, low, high = (_finite(row.get(name)) for name in fields)
    if (effect is None or low is None or high is None or
            not -1 <= low <= effect <= high <= 1):
        return metric, effect, None, None
    return metric, effect, low, high


def paired_summary(snapshot: PairedSnapshot | None, endpoint_id: str, marker_id: str) -> str:
    """A concise, non-cherry-picked comparison for the simple result cards."""
    if snapshot is None or snapshot.status not in {"ok", "empty"}:
        return "Klassischer EKG-Vergleich: noch kein belegter gepaarter Ergebnisstand."
    if snapshot.endpoint_id and snapshot.endpoint_id != endpoint_id:
        return "Klassischer EKG-Vergleich: für diese klinische Frage im Detail öffnen."
    focused = snapshot.focus_counts.get((endpoint_id, marker_id))
    if focused is not None:
        count, valid = focused
        if count == 0:
            return "Klassischer EKG-Vergleich: für diese Kombination nicht gepaart ausgewertet."
        return (
            f"Klassischer EKG-Vergleich: {count} gepaarte Vergleiche auf denselben "
            f"Validation-Fällen; {valid} mit formaler Schätzung und CI. Feste Regeln "
            "(Sensitivitäts-Δ) und kontinuierliche Marker (AUC-Δ) stehen getrennt im Detail. "
            "Ein nominaler Unterschied allein ist keine klinische Nutzenprüfung."
        )
    if snapshot.rows.empty:
        return "Klassischer EKG-Vergleich: noch kein belegter gepaarter Ergebnisstand."
    rows = snapshot.rows
    matched = rows.loc[
        rows["endpoint_id"].astype(str).eq(endpoint_id)
        & rows["vcg_feature_id"].astype(str).eq(marker_id)
    ]
    if matched.empty:
        return "Klassischer EKG-Vergleich: für diese Kombination nicht gepaart ausgewertet."
    valid = 0
    for _, row in matched.iterrows():
        _, effect, low, high = paired_effect(row)
        if (str(row.get("inference_status")) == "formal_inference"
                and effect is not None and low is not None and high is not None):
            valid += 1
    return (
        f"Klassischer EKG-Vergleich: {len(matched)} gepaarte Vergleiche auf denselben "
        f"Validation-Fällen; {valid} mit formaler Schätzung und CI. Feste Regeln "
        "(Sensitivitäts-Δ) und kontinuierliche Marker (AUC-Δ) stehen getrennt im Detail. "
        "Ein nominaler Unterschied allein ist keine klinische Nutzenprüfung."
    )


def paired_outcome_summary(snapshot: PairedSnapshot | None) -> str:
    """Describe all paired comparisons without equating a null CI with equivalence."""
    if snapshot is None or snapshot.status not in {"ok", "empty"} or snapshot.outcome_counts.empty:
        return "Nicht-Befunde: Noch keine vollständige gepaarte Validation-Zählung verfügbar."
    labels = {
        _FIXED_TYPE: "feste EKG-Regeln (Sensitivitäts-Δ)",
        "paired_continuous_marker_discrimination": "kontinuierliche EKG-Marker (AUC-Δ)",
        "paired_availability_marker_discrimination": "Verfügbarkeitsvarianten (AUC-Δ)",
    }
    parts = []
    for comparison_type, group in snapshot.outcome_counts.groupby("comparison_type", sort=True):
        counts = {str(row.category): int(row.n) for row in group.itertuples()}
        parts.append(
            f"{labels.get(str(comparison_type), str(comparison_type))}: "
            f"{counts.get('positive_delta', 0)} positives Δ-CI, "
            f"{counts.get('negative_delta', 0)} negatives Δ-CI, "
            f"{counts.get('inconclusive', 0)} CI über Null, "
            f"{counts.get('no_formal_ci', 0)} ohne formales CI"
        )
    return (
        "Gepaarte Validation-Vergleiche: " + "; ".join(parts) + ". "
        "CI über Null bedeutet unentschieden, nicht Gleichwertigkeit; Zeilen sind "
        "überlappende Vergleiche, keine unabhängigen Entdeckungen."
    )


def load_paired_snapshot(
    page: ResultPage, endpoint_id: str,
    *, focus_pairs: tuple[tuple[str, str], ...] = (),
) -> PairedSnapshot:
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return PairedSnapshot("pending", "Noch kein kompatibler abgeschlossener Validation-Stand.",
                              pd.DataFrame(), endpoint_id, page.run_id, page.snapshot_hash)
    path = page.artifact_path.parent / "vcg_vs_classical_comparison.parquet"
    try:
        manifest = StageManifest.load(page.artifact_path.parent / "validate.manifest.json")
    except (OSError, TypeError, ValueError):
        return PairedSnapshot("missing", "Validierungsmanifest fehlt oder ist unlesbar.",
                              pd.DataFrame(), endpoint_id, page.run_id, page.snapshot_hash)
    assurance = validation_output_assurance(manifest, path, page.run_id)
    if assurance is None:
        return PairedSnapshot("missing", "Gepaarte Vergleichsdatei ist nicht vollständig manifestiert oder ihre Integritätsprüfung ist fehlgeschlagen.",
                              pd.DataFrame(), endpoint_id, page.run_id, page.snapshot_hash)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        schema = {column[0] for column in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)],
        ).description}
        if not _REQUIRED.issubset(schema):
            return PairedSnapshot("incompatible", "Vergleichsartefakt hat nicht den benötigten Vertrag.",
                                  pd.DataFrame(), endpoint_id, page.run_id, page.snapshot_hash)
        projected = ", ".join(
            f'"{name}"' if name in schema else f'NULL AS "{name}"' for name in _COLUMNS
        )
        where = "evaluation_split = 'validation' AND comparison_type IN (?, ?, ?)"
        parameters: list[object] = [str(path), *_AUC_TYPES, _FIXED_TYPE]
        if endpoint_id:
            where += " AND endpoint_id = ?"
            parameters.append(endpoint_id)
        count = int(connection.execute(
            f"SELECT COUNT(*) FROM read_parquet(?) WHERE {where}", parameters,
        ).fetchone()[0])
        outcome_counts = pd.DataFrame()
        ci_columns = {
            "delta_auc", "delta_auc_ci_low", "delta_auc_ci_high",
            "sensitivity_delta_vs_fixed_rule", "sensitivity_delta_ci_low",
            "sensitivity_delta_ci_high",
        }
        if ci_columns.issubset(schema):
            outcome_counts = connection.execute(
                "WITH effects AS (SELECT comparison_type, inference_status, "
                "CASE WHEN comparison_type = 'paired_fixed_rule_operating_point' "
                "THEN TRY_CAST(sensitivity_delta_vs_fixed_rule AS DOUBLE) "
                "ELSE TRY_CAST(delta_auc AS DOUBLE) END AS effect, "
                "CASE WHEN comparison_type = 'paired_fixed_rule_operating_point' "
                "THEN TRY_CAST(sensitivity_delta_ci_low AS DOUBLE) "
                "ELSE TRY_CAST(delta_auc_ci_low AS DOUBLE) END AS ci_low, "
                "CASE WHEN comparison_type = 'paired_fixed_rule_operating_point' "
                "THEN TRY_CAST(sensitivity_delta_ci_high AS DOUBLE) "
                "ELSE TRY_CAST(delta_auc_ci_high AS DOUBLE) END AS ci_high "
                f"FROM read_parquet(?) WHERE {where}), "
                "classified AS (SELECT comparison_type, "
                "CASE WHEN inference_status = 'formal_inference' "
                "AND isfinite(effect) AND isfinite(ci_low) AND isfinite(ci_high) "
                "AND -1 <= ci_low AND ci_low <= effect AND effect <= ci_high "
                "AND ci_high <= 1 THEN "
                "CASE WHEN ci_low > 0 THEN 'positive_delta' "
                "WHEN ci_high < 0 THEN 'negative_delta' ELSE 'inconclusive' END "
                "ELSE 'no_formal_ci' END AS category FROM effects) "
                "SELECT comparison_type, category, COUNT(*) AS n "
                "FROM classified GROUP BY comparison_type, category "
                "ORDER BY comparison_type, category",
                parameters,
            ).fetchdf()
        focus_counts = {pair: (0, 0) for pair in focus_pairs} if ci_columns.issubset(schema) else {}
        if focus_pairs and ci_columns.issubset(schema):
            pair_where = " OR ".join(
                "(CAST(endpoint_id AS VARCHAR) = ? AND CAST(vcg_feature_id AS VARCHAR) = ?)"
                for _ in focus_pairs
            )
            pair_params = [value for pair in focus_pairs for value in pair]
            focused = connection.execute(
                "WITH effects AS (SELECT CAST(endpoint_id AS VARCHAR) AS endpoint_id, "
                "CAST(vcg_feature_id AS VARCHAR) AS vcg_feature_id, inference_status, "
                "CASE WHEN comparison_type = 'paired_fixed_rule_operating_point' "
                "THEN TRY_CAST(sensitivity_delta_vs_fixed_rule AS DOUBLE) "
                "ELSE TRY_CAST(delta_auc AS DOUBLE) END AS effect, "
                "CASE WHEN comparison_type = 'paired_fixed_rule_operating_point' "
                "THEN TRY_CAST(sensitivity_delta_ci_low AS DOUBLE) "
                "ELSE TRY_CAST(delta_auc_ci_low AS DOUBLE) END AS ci_low, "
                "CASE WHEN comparison_type = 'paired_fixed_rule_operating_point' "
                "THEN TRY_CAST(sensitivity_delta_ci_high AS DOUBLE) "
                "ELSE TRY_CAST(delta_auc_ci_high AS DOUBLE) END AS ci_high "
                f"FROM read_parquet(?) WHERE {where} AND ({pair_where})) "
                "SELECT endpoint_id, vcg_feature_id, COUNT(*) AS n, "
                "COUNT(*) FILTER (WHERE inference_status = 'formal_inference' "
                "AND isfinite(effect) AND isfinite(ci_low) AND isfinite(ci_high) "
                "AND -1 <= ci_low AND ci_low <= effect AND effect <= ci_high "
                "AND ci_high <= 1) AS formal_n FROM effects "
                "GROUP BY endpoint_id, vcg_feature_id",
                [*parameters, *pair_params],
            ).fetchdf()
            for row in focused.itertuples():
                focus_counts[(str(row.endpoint_id), str(row.vcg_feature_id))] = (
                    int(row.n), int(row.formal_n),
                )
        rows = connection.execute(
            f"SELECT {projected} FROM read_parquet(?) WHERE {where} "
            "ORDER BY endpoint_id, comparison_type, baseline_feature_id, vcg_feature_id LIMIT 100",
            parameters,
        ).fetchdf()
        return PairedSnapshot(
            "ok" if count else "empty",
            "Gepaart auf denselben Validation-Fällen; Zeilen ohne CI bleiben deskriptiv. "
            + ("Datei SHA-256-geprüft." if assurance == "sha256" else
               "Datei nur im abgeschlossenen Validierungsmanifest deklariert; kein Ausgabe-SHA-256 vorhanden.")
            if count else "Für diesen Endpunkt liegt kein gepaarter klassischer Vergleich vor.",
            rows, endpoint_id, page.run_id, page.snapshot_hash, count, outcome_counts,
            focus_counts,
        )
    finally:
        connection.close()


class _ComparisonPanel(QWidget):
    def __init__(self, comparison_type: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.comparison_type = comparison_type
        self._snapshot: PairedSnapshot | None = None
        self._frame = pd.DataFrame()
        self._chart_has_ci = False
        layout = QVBoxLayout(self)
        self.note = QLabel("Gepaarte Vergleiche werden geprüft …")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(220)
        self.plot.setLabel("bottom", "Differenz neuer Marker minus klassische Regel")
        self.plot.setXRange(-1, 1, padding=0)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels((
            "Marker", "Klassischer Vergleich", "Gepaarte Fälle", "Differenz (95-%-CI)",
            "Validation-Spezifität Regel / VCG", "Schätzstatus",
        ))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.display = QTabWidget()
        self.display.addTab(self.plot, "Forest-Diagramm")
        self.display.addTab(self.table, "Zahlen")
        layout.addWidget(self.display, 1)
        plot_export = QPushButton("Vergleich als PNG/SVG")
        plot_export.setToolTip("Exportiert nur gepaarte Unterschiede und tatsächlich berechnete CI mit Herkunftsdatei.")
        plot_export.clicked.connect(self._export_plot)
        layout.addWidget(plot_export)
        table_export = QPushButton("Vergleichstabelle als CSV")
        table_export.setToolTip("Exportiert die begrenzten gepaarten Validation-Vergleiche mit Herkunftsdatei.")
        table_export.clicked.connect(self._export_table)
        layout.addWidget(table_export)

    def set_snapshot(self, snapshot: PairedSnapshot) -> None:
        self._snapshot = snapshot
        frame = snapshot.rows
        if not frame.empty and "comparison_type" in frame:
            frame = frame.loc[frame["comparison_type"].astype(str).eq(self.comparison_type)]
        available_rows = len(frame)
        frame = frame.head(25)
        self._frame = frame.copy()
        self._chart_has_ci = False
        self.plot.clear()
        self.table.setRowCount(len(frame))
        if frame.empty:
            self.note.setText(snapshot.note)
            self.plot.setTitle("Kein Vergleich mit dieser Messdefinition vorhanden")
            self.display.setTabEnabled(0, False)
            self.display.setCurrentIndex(1)
            return
        metric = "Sensitivität" if self.comparison_type == _FIXED_TYPE else "AUC"
        self.plot.setTitle(f"Gepaarte Validation-Differenz der {metric}")
        self.plot.setLabel(
            "bottom",
            "Δ Sensitivität (VCG minus feste EKG-Regel)"
            if self.comparison_type == _FIXED_TYPE else
            "Δ AUC (VCG minus klassischer EKG-Marker)",
        )
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#88939b", width=1)))
        ticks = []
        ci_count = 0
        for index, (_, row) in enumerate(frame.iterrows()):
            label, effect, low, high = paired_effect(row)
            baseline = str(row.get("baseline_feature_id") or "unbekannt")
            marker = str(row.get("vcg_feature_id") or "unbekannt")
            formal_ci = (
                str(row.get("inference_status")) == "formal_inference"
                and effect is not None and low is not None and high is not None
            )
            if formal_ci:
                y = float(ci_count + 1)
                ticks.append((y, baseline[:30]))
                self.plot.plot([effect], [y], pen=None, symbol="o", symbolSize=8,
                               symbolBrush="#287f99")
                self.plot.plot([low, high], [y, y], pen=pg.mkPen("#287f99", width=2))
                ci_count += 1
            rule_spec = _finite(row.get("baseline_validation_specificity"))
            vcg_spec = _finite(row.get("vcg_specificity_at_specificity_match"))
            paired_n = _finite(row.get("paired_n"))
            if formal_ci:
                difference_text = f"{label} {effect:+.3f} ({low:+.3f} bis {high:+.3f})"
            elif effect is not None and low is not None and high is not None:
                difference_text = (
                    f"{label} {effect:+.3f} ({low:+.3f} bis {high:+.3f}); "
                    "nur deskriptiv"
                )
            elif effect is not None:
                difference_text = f"{label} {effect:+.3f}; CI nicht berechnet"
            else:
                difference_text = "nicht schätzbar"
            values = (
                marker, baseline,
                str(int(paired_n)) if paired_n is not None and paired_n >= 0
                and paired_n.is_integer() else "nicht angegeben",
                difference_text,
                f"{rule_spec:.3f} / {vcg_spec:.3f}"
                if rule_spec is not None and vcg_spec is not None else "nicht anwendbar",
                str(row.get("inference_status") or "unbekannt"),
            )
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(value))
        self.plot.getAxis("left").setTicks([ticks])
        if ci_count:
            self.plot.setYRange(.5, ci_count + .5, padding=0)
        self._chart_has_ci = ci_count > 0
        self.display.setTabEnabled(0, self._chart_has_ci)
        if not self._chart_has_ci:
            self.display.setCurrentIndex(1)
        self.note.setText(
            f"{available_rows} geladene {metric}-Vergleiche; höchstens 25 angezeigt, "
            f"davon {ci_count} mit berechnetem CI. "
            + ("Bei festen Regeln wurde die VCG-Schwelle auf die Trainingsspezifität "
               "der Regel gesetzt; die tatsächlich erreichten Validation-Spezifitäten "
               "stehen in der Tabelle. Sensitivitäts-Δ ist keine AUC-Differenz. "
               if self.comparison_type == _FIXED_TYPE else
               "AUC-Differenzen betreffen kontinuierliche Vergleichsmarker. ")
            + "Ohne formale Inferenz und CI keine Überlegenheitsaussage."
        )

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "snapshot_hash": snapshot.snapshot_hash if snapshot else None,
            "endpoint_id": snapshot.endpoint_id if snapshot else None,
            "comparison_type": self.comparison_type,
            "split": "validation",
            "note": "Gepaarte Forschungsvergleiche; kein klinischer Nutzenclaim.",
        }

    def _export_plot(self, *_args) -> None:
        if self._chart_has_ci:
            choose_plot_export(self, self.plot, self._metadata(),
                               proposed_name="gepaarter_ekg_vcg_vergleich.png")

    def _export_table(self, *_args) -> None:
        if not self._frame.empty:
            choose_table_export(self, self._frame, self._metadata(),
                                proposed_name="gepaarter_ekg_vcg_vergleich.csv")


class AtlasPairedWidget(QTabWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.fixed = _ComparisonPanel(_FIXED_TYPE)
        self.continuous = _ComparisonPanel("paired_continuous_marker_discrimination")
        self.availability = _ComparisonPanel("paired_availability_marker_discrimination")
        self.addTab(self.fixed, "Feste EKG-Regeln")
        self.addTab(self.continuous, "Kontinuierliche EKG-Marker")
        self.addTab(self.availability, "Verfügbarkeitsvarianten")

    def set_snapshot(self, snapshot: PairedSnapshot) -> None:
        for panel in (self.fixed, self.continuous, self.availability):
            panel.set_snapshot(snapshot)
