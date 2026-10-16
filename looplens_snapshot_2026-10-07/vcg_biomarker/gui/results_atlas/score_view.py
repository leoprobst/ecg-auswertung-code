"""Transparent, research-only score search summaries from sealed follow-up stages."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultRepository
from ...mimic.score_research import VERSION as SCORE_VERSION
from ...mimic.storage import StageManifest, content_hash, verify_output_integrity
from .export import choose_plot_export, choose_table_export


@dataclass(frozen=True)
class ScoreSnapshot:
    status: str
    note: str
    models: pd.DataFrame
    comparisons: pd.DataFrame
    run_id: str


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _interval(value: object) -> tuple[float | None, float | None]:
    if isinstance(value, list) and len(value) == 2:
        low, high = (_number(item) for item in value)
        if low is not None and high is not None and low <= high:
            return low, high
    return None, None


def _sealed_report(path: Path) -> tuple[dict, dict] | None:
    try:
        manifest = StageManifest.load(path)
    except (OSError, ValueError, TypeError):
        return None
    report_path = path.parent / "score_research_report.json"
    inputs_path = path.parent / "inputs.json"
    resolved = str(report_path.resolve())
    declared = {str(Path(item).resolve()) for item in manifest.output_paths}
    records = [item for item in manifest.output_integrity
               if str(Path(str(item.get("path", ""))).resolve()) in
               {resolved, str(inputs_path.resolve())}]
    if (manifest.stage != "score-discovery"
            or manifest.status not in {"complete", "complete_without_eligible_inputs"}
            or manifest.stage_version != SCORE_VERSION
            or not {resolved, str(inputs_path.resolve())}.issubset(declared)
            or len(records) != 2 or not verify_output_integrity(records, mode="sha256")):
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if (report.get("version") != SCORE_VERSION or report.get("input_hash") != manifest.input_hash
            or inputs.get("version") != SCORE_VERSION
            or content_hash(inputs) != manifest.input_hash
            or inputs.get("config_hash") != manifest.config_hash
            or report.get("clinical_claim_allowed") is not False
            or report.get("locked_test_read") is not False):
        return None
    if inputs.get("sampling_design") != "complete_clinical_phenome_union" and any(
        isinstance(result, dict) and bool(result.get("models"))
        for result in report.get("results", ())
    ):
        return None
    return report, inputs


def load_score_snapshot(repository: ResultRepository, run_id: str) -> ScoreSnapshot:
    empty = pd.DataFrame()
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        return ScoreSnapshot("incompatible", "Ungültige Run-Kennung.", empty, empty, run_id)
    descriptor = next((item for item in repository.available_runs()
                       if item.run_id == run_id), None)
    if descriptor is None or not descriptor.complete_compatible:
        return ScoreSnapshot("pending", "Noch kein kompatibler abgeschlossener Validation-Stand.",
                             empty, empty, run_id)
    root = (repository.config.runs_root / run_id / "score_discovery").resolve()
    if not root.is_relative_to(repository.config.runs_root.resolve()):
        return ScoreSnapshot("incompatible", "Scorepfad liegt außerhalb des Run-Bereichs.",
                             empty, empty, run_id)
    manifests = sorted(root.glob("*/score_discovery.manifest.json"))
    if not manifests:
        return ScoreSnapshot("pending", "Automatische Score-Suche noch nicht abgeschlossen.",
                             empty, empty, run_id)
    if len(manifests) > 32:
        return ScoreSnapshot("incompatible", "Mehr als 32 Score-Snapshots: Auswahl muss eingegrenzt werden.",
                             empty, empty, run_id)
    model_rows, comparison_rows = [], []
    accepted = 0
    for manifest_path in manifests:
        sealed = _sealed_report(manifest_path)
        if sealed is None:
            continue
        report, inputs = sealed
        accepted += 1
        source = str(report["input_hash"])[:20]
        design = str(inputs.get("sampling_design", "unverified"))
        max_patients = inputs.get("max_patients")
        for result in report.get("results", []):
            if not isinstance(result, dict):
                continue
            endpoint = str(result.get("endpoint_id", ""))
            for group, model in result.get("models", {}).items():
                if not isinstance(model, dict):
                    continue
                evaluation = model.get("evaluation", {})
                metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
                spec = model.get("score_spec", {})
                if not isinstance(spec, dict):
                    spec = {}
                ci_low, ci_high = _interval(metrics.get("auc_ci95"))
                folds = model.get("folds", ())
                if not isinstance(folds, list):
                    folds = []
                fold_components = [
                    set(str(component) for component in fold.get("components", ()))
                    for fold in folds if isinstance(fold, dict)
                ]
                final_components = tuple(str(item) for item in spec.get("components", ()))
                stability = {
                    component: sum(component in selected for selected in fold_components)
                    for component in final_components
                }
                point = model.get("point_score", {})
                point_metrics = point.get("metrics", {}) if isinstance(point, dict) else {}
                model_rows.append({
                    "snapshot": source, "endpoint_id": endpoint, "model_group": str(group),
                    "sampling_design": design, "max_patients": max_patients,
                    "status": str(metrics.get("status", model.get("status", "unknown"))),
                    "auc": _number(metrics.get("auc")), "auc_low": ci_low,
                    "auc_high": ci_high, "average_precision": _number(metrics.get("average_precision")),
                    "brier": _number(metrics.get("brier")),
                    "n_patients": metrics.get("n_patients"), "n_events": metrics.get("n_events"),
                    "n_predicted": metrics.get("n_predicted"),
                    "n_eligible": metrics.get("n_eligible"),
                    "point_auc": _number(point_metrics.get("auc")),
                    "point_brier": _number(point_metrics.get("brier")),
                    "calibration_status": str(metrics.get("calibration_status", "not_computed")),
                    "components": final_components,
                    "fold_count": len(fold_components), "component_fold_counts": stability,
                    "coefficients": tuple(spec.get("coefficients", ())),
                    "formula": str(spec.get("formula", "")),
                })
            for name, comparison in result.get("paired_comparisons", {}).items():
                if not isinstance(comparison, dict):
                    continue
                low, high = _interval(comparison.get("delta_auc_ci95"))
                comparison_rows.append({
                    "snapshot": source, "endpoint_id": endpoint,
                    "sampling_design": design, "max_patients": max_patients,
                    "comparison": str(name), "status": str(comparison.get("status", "unknown")),
                    "delta_auc": _number(comparison.get("delta_auc_second_minus_first")),
                    "ci_low": low, "ci_high": high,
                    "n_common": comparison.get("n_common"),
                    "n_patients": comparison.get("n_patients"),
                })
    if not accepted:
        return ScoreSnapshot("incompatible", "Kein versions- und hashgeprüfter Score-Snapshot.",
                             empty, empty, run_id)
    models = pd.DataFrame(model_rows)
    comparisons = pd.DataFrame(comparison_rows)
    return ScoreSnapshot(
        "ok" if not models.empty else "empty",
        f"{accepted} versiegelte Score-Snapshot(s). Die AUC beschreibt äußere Out-of-fold-"
        "Vorhersagen des Suchverfahrens innerhalb Development, keine unabhängige externe "
        "Validierung der endgültigen Formel. Alle Scores bleiben Forschung.",
        models, comparisons, run_id,
    )


class AtlasScoreWidget(QWidget):
    _GROUPS = {
        "ecg": "Klassisches EKG", "vcg": "VCG", "ecg_vcg": "EKG + VCG",
        "ecg_extended": "EKG + direkte Geometrie", "clinical_context": "Klinischer Kontext",
        "clinical_ecg": "Kontext + EKG", "clinical_vcg": "Kontext + VCG",
        "clinical_ecg_vcg": "Kontext + EKG + VCG",
    }

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: ScoreSnapshot | None = None
        layout = QVBoxLayout(self)
        self.note = QLabel("Score-Snapshots werden geprüft.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.snapshot_choice = QComboBox()
        self.snapshot_choice.currentIndexChanged.connect(self._draw)
        layout.addWidget(self.snapshot_choice)
        self.endpoint_choice = QComboBox()
        self.endpoint_choice.currentIndexChanged.connect(self._draw)
        layout.addWidget(self.endpoint_choice)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Gepaartes ΔAUC gegenüber Vergleichsmodell")
        self.plot.showGrid(x=True, y=False, alpha=.15)
        layout.addWidget(self.plot, 1)
        self.model_table = QTableWidget(0, 7)
        self.model_table.setHorizontalHeaderLabels((
            "Modellgruppe", "OOF-AUC", "Punktescore-AUC", "Personen / Ereignisse",
            "Vorhersagen / geeignet", "Brier", "Refit-Komponenten",
        ))
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.model_table)
        self.comparison_table = QTableWidget(0, 3)
        self.comparison_table.setHorizontalHeaderLabels((
            "Gepaarter Vergleich", "ΔAUC (95-%-CI)", "gemeinsame Personen",
        ))
        self.comparison_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.comparison_table)
        self.detail = QLabel(
            "OOF-AUC bewertet das gesamte Auswahlverfahren. Die gezeigten Komponenten und "
            "Koeffizienten stammen erst aus dem abschließenden Refit; ihre Fold-Häufigkeit "
            "steht im Tooltip. Punktescore-AUC bewertet die gerundete OOF-Variante getrennt. "
            "Keiner dieser Werte ist eine unabhängige Validierung des Refit-Scores."
        )
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        actions = QHBoxLayout()
        plot_export = QPushButton("Gepaarte Vergleiche als PNG/SVG")
        plot_export.setToolTip("Exportiert den ausgewählten Forschungsvergleich mit Score-Snapshot-Herkunft.")
        plot_export.clicked.connect(self._export_plot)
        actions.addWidget(plot_export)
        model_export = QPushButton("Modellgruppen als CSV")
        model_export.setToolTip("Exportiert AUC, Brier und Komponenten der ausgewählten Forschungsfrage.")
        model_export.clicked.connect(self._export_models)
        actions.addWidget(model_export)
        comparison_export = QPushButton("Gepaarte Vergleiche als CSV")
        comparison_export.setToolTip("Exportiert Delta-AUC auf denselben Patienten mit vorhandenen Intervallen.")
        comparison_export.clicked.connect(self._export_comparisons)
        actions.addWidget(comparison_export)
        actions.addStretch(1)
        layout.addLayout(actions)

    def set_snapshot(self, snapshot: ScoreSnapshot) -> None:
        self._snapshot = snapshot
        self.note.setText(snapshot.note)
        self.snapshot_choice.blockSignals(True)
        self.endpoint_choice.blockSignals(True)
        self.snapshot_choice.clear()
        self.endpoint_choice.clear()
        if not snapshot.models.empty:
            for source in sorted(snapshot.models.snapshot.astype(str).unique()):
                row = snapshot.models.loc[snapshot.models.snapshot.eq(source)].iloc[0]
                cap = row.max_patients
                cap_label = f"max. {int(cap):,} Personen" if _number(cap) is not None else "ohne Personenlimit"
                self.snapshot_choice.addItem(
                    f"Score-Snapshot {source} · {row.sampling_design} · {cap_label}", source,
                )
            for endpoint in sorted(snapshot.models.endpoint_id.astype(str).unique()):
                self.endpoint_choice.addItem(endpoint, endpoint)
        self.snapshot_choice.blockSignals(False)
        self.endpoint_choice.blockSignals(False)
        self._draw()

    def _draw(self, _index: int = -1) -> None:
        self.plot.clear()
        self.model_table.setRowCount(0)
        self.comparison_table.setRowCount(0)
        snapshot = self._snapshot
        if snapshot is None or snapshot.models.empty:
            return
        source, endpoint = self.snapshot_choice.currentData(), self.endpoint_choice.currentData()
        models = snapshot.models.loc[
            snapshot.models.snapshot.eq(source) & snapshot.models.endpoint_id.eq(endpoint)
        ]
        self.model_table.setRowCount(len(models))
        for i, (_, row) in enumerate(models.iterrows()):
            auc = _number(row.auc)
            low, high = _number(row.auc_low), _number(row.auc_high)
            auc_text = f"{auc:.3f}" if auc is not None else "nicht schätzbar"
            if auc is not None and low is not None and high is not None:
                auc_text += f" ({low:.3f}–{high:.3f})"
            components = ", ".join(row.components)
            point_auc = _number(row.point_auc)
            values = (
                self._GROUPS.get(row.model_group, row.model_group), auc_text,
                f"{point_auc:.3f}" if point_auc is not None else "nicht berechnet",
                f"{self._count(row.n_patients)} / {self._count(row.n_events)}",
                f"{self._count(row.n_predicted)} / {self._count(row.n_eligible)}",
                f"{row.brier:.3f}" if _number(row.brier) is not None else "nicht berechnet",
                components[:90] + ("…" if len(components) > 90 else ""),
            )
            for j, value in enumerate(values):
                item = QTableWidgetItem(value)
                if j == 6:
                    item.setToolTip(
                        f"Refit-Formel: {row.formula}\nRefit-Komponenten: {components}\n"
                        f"Refit-Koeffizienten: {row.coefficients}\n"
                        f"Auswahl je äußerem Fold ({row.fold_count} Folds): "
                        f"{row.component_fold_counts}\n"
                        f"Kalibrationsdiagnostik der OOF-Vorhersagen: {row.calibration_status}"
                    )
                self.model_table.setItem(i, j, item)
        comparisons = snapshot.comparisons
        if comparisons.empty:
            return
        comparisons = comparisons.loc[
            comparisons.snapshot.eq(source) & comparisons.endpoint_id.eq(endpoint)
        ]
        self.comparison_table.setRowCount(len(comparisons))
        ticks = []
        for i, (_, row) in enumerate(comparisons.iterrows()):
            y = float(len(comparisons) - i)
            delta, low, high = (_number(row[name]) for name in ("delta_auc", "ci_low", "ci_high"))
            valid = (delta is not None and low is not None and high is not None
                     and low <= delta <= high and row.status == "research_comparison")
            if valid:
                self.plot.plot([low, high], [y, y], pen=pg.mkPen("#228b8d", width=2))
            if delta is not None:
                self.plot.plot([delta], [y], pen=None, symbol="o", symbolSize=8,
                               symbolBrush="#228b8d" if valid else "#888888")
            label = str(row.comparison).replace("_vs_", " gegen ")
            ticks.append((y, label[:36]))
            value = (f"{delta:+.3f} ({low:+.3f}–{high:+.3f})" if valid
                     else f"{delta:+.3f}; CI fehlt" if delta is not None
                     else "nicht schätzbar")
            for j, text in enumerate((label, value, str(row.n_patients or "?"))):
                self.comparison_table.setItem(i, j, QTableWidgetItem(text))
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=90,
                                          pen=pg.mkPen("#999999", width=1)))
        self.plot.getAxis("left").setTicks([ticks])
        self.plot.getAxis("left").setWidth(220)

    @staticmethod
    def _count(value: object) -> str:
        number = _number(value)
        return str(int(number)) if number is not None and number >= 0 and number.is_integer() else "?"

    def _selected(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        snapshot = self._snapshot
        if snapshot is None or snapshot.models.empty:
            return pd.DataFrame(), pd.DataFrame()
        source, endpoint = self.snapshot_choice.currentData(), self.endpoint_choice.currentData()
        models = snapshot.models.loc[
            snapshot.models.snapshot.eq(source) & snapshot.models.endpoint_id.eq(endpoint)
        ]
        comparisons = snapshot.comparisons
        if not comparisons.empty:
            comparisons = comparisons.loc[
                comparisons.snapshot.eq(source) & comparisons.endpoint_id.eq(endpoint)
            ]
        return models, comparisons

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "score_snapshot": self.snapshot_choice.currentData(),
            "endpoint_id": self.endpoint_choice.currentData(),
            "stage_version": SCORE_VERSION,
            "evidence_scope": "development_outer_oof_search_procedure_not_external_validation",
            "clinical_claim_allowed": False,
        }

    def _export_plot(self, *_args) -> None:
        _, comparisons = self._selected()
        if not comparisons.empty:
            choose_plot_export(self, self.plot, self._metadata(),
                               proposed_name="score_gepaarte_vergleiche.png")

    def _export_models(self, *_args) -> None:
        models, _ = self._selected()
        if not models.empty:
            choose_table_export(self, models, self._metadata(),
                                proposed_name="score_modellgruppen.csv")

    def _export_comparisons(self, *_args) -> None:
        _, comparisons = self._selected()
        if not comparisons.empty:
            choose_table_export(self, comparisons, self._metadata(),
                                proposed_name="score_gepaarte_vergleiche.csv")
