"""Observed medication response, deliberately separate from treatment benefit."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultPage
from ...mimic.storage import StageManifest, verify_output_integrity
from ...mimic.therapy_response import THERAPY_RESPONSE_VERSION
from .export import choose_plot_export, choose_table_export


@dataclass(frozen=True)
class ResponseSnapshot:
    status: str
    note: str
    rows: pd.DataFrame
    run_id: str


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_response(page: ResultPage) -> ResponseSnapshot:
    empty = pd.DataFrame()
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return ResponseSnapshot("pending", "Kein kompatibler abgeschlossener Ergebnisstand.", empty, page.run_id)
    root = page.artifact_path.parent / "therapy_response"
    result_path = root / "therapy_response_results.parquet"
    try:
        manifest = StageManifest.load(root / "therapy_response.manifest.json")
    except (OSError, TypeError, ValueError):
        return ResponseSnapshot("pending", "Medikamenten-Response-Analyse noch nicht abgeschlossen.", empty, page.run_id)
    resolved = str(result_path.resolve())
    records = [record for record in manifest.output_integrity
               if str(Path(str(record.get("path", ""))).resolve()) == resolved]
    if (manifest.status != "complete" or manifest.stage != "therapy-response"
            or manifest.stage_version != THERAPY_RESPONSE_VERSION
            or resolved not in {str(Path(path).resolve()) for path in manifest.output_paths}
            or len(records) != 1 or not verify_output_integrity(records, mode="sha256")):
        return ResponseSnapshot("incompatible", "Response-Artefakt ist unvollständig oder nicht hashgeprüft.", empty, page.run_id)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='256MB'")
        connection.execute("SET threads=1")
        schema = {item[0] for item in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(result_path)],
        ).description}
        required = {"scope", "medication_class", "status", "effect", "ci_low", "ci_high",
                    "n_pairs", "n_subjects", "rate_before", "rate_after", "claim_scope", "run_id"}
        if not required.issubset(schema):
            return ResponseSnapshot("incompatible", "Response-Tabelle hat keinen kompatiblen Ergebnisvertrag.", empty, page.run_id)
        rows = connection.execute(
            "SELECT scope, medication_class, status, effect, ci_low, ci_high, "
            "n_pairs, n_subjects, rate_before, rate_after, claim_scope "
            "FROM read_parquet(?) WHERE run_id = ? ORDER BY scope, medication_class LIMIT 100",
            [str(result_path), page.run_id],
        ).fetchdf()
    except duckdb.Error:
        return ResponseSnapshot("invalid", "Response-Tabelle ist nicht lesbar.", empty, page.run_id)
    finally:
        connection.close()
    if not rows.empty and not rows.claim_scope.astype(str).eq(
            "physiological_response_association_not_causal_effect").all():
        return ResponseSnapshot("incompatible", "Aussagegrenze der Response-Daten ist unbekannt.", empty, page.run_id)
    return ResponseSnapshot(
        "ok" if not rows.empty else "empty",
        "Gepaarte EKGs vor/nach dokumentierter Gabe: beobachtete Änderung eines Tachykardie-Musters. "
        "Kein randomisierter Vergleich, kein kausaler Nutzen und keine Therapieempfehlung.",
        rows, page.run_id,
    )


class AtlasResponseWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: ResponseSnapshot | None = None
        layout = QVBoxLayout(self)
        self.note = QLabel("Therapie-Response wird geprüft.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Nachher minus vorher: Tachykardie-Anteil", units="Prozentpunkte")
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        layout.addWidget(self.plot, 1)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels((
            "Medikamentenklasse", "Paare", "Personen", "Vorher", "Nachher", "Änderung (95-%-CI)",
        ))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        self.limits = QLabel(
            "Zwei diskrete EKGs innerhalb der vordefinierten Zeitfenster sind kein kontinuierliches Monitoring. "
            "Paare einer Person sind statistisch abhängig; CI werden nur gezeigt, wenn der patientengeclusterte "
            "Bootstrap sie tatsächlich schätzen konnte. Eine Veränderung unter Therapie ist kein Behandlungseffekt."
        )
        self.limits.setWordWrap(True)
        layout.addWidget(self.limits)
        exports = QHBoxLayout()
        export_plot = QPushButton("Response-Diagramm als PNG/SVG")
        export_plot.setToolTip("Exportiert beobachtete Vorher-nachher-Differenzen, keinen Behandlungseffekt.")
        export_plot.clicked.connect(self._export_plot)
        exports.addWidget(export_plot)
        export_table = QPushButton("Response-Tabelle als CSV")
        export_table.setToolTip("Exportiert die hashgeprüfte Response-Zusammenfassung mit Herkunftsdatei.")
        export_table.clicked.connect(self._export_table)
        exports.addWidget(export_table)
        exports.addStretch(1)
        layout.addLayout(exports)

    def set_snapshot(self, snapshot: ResponseSnapshot) -> None:
        self._snapshot = snapshot
        self.note.setText(snapshot.note)
        self.plot.clear()
        self.table.setRowCount(len(snapshot.rows))
        ticks = []
        for index, (_, row) in enumerate(snapshot.rows.iterrows()):
            label = "Alle Klassen" if row.scope == "all_classes" else str(row.medication_class)
            effect = _finite(row.effect)
            low = _finite(row.ci_low)
            high = _finite(row.ci_high)
            has_ci = (effect is not None and low is not None and high is not None
                      and low <= effect <= high and str(row.status) == "estimable")
            before = _finite(row.rate_before)
            after = _finite(row.rate_after)
            values = (
                label, str(row.n_pairs), str(row.n_subjects),
                f"{100 * before:.1f} %" if before is not None else "nicht berechnet",
                f"{100 * after:.1f} %" if after is not None else "nicht berechnet",
                (f"{100 * effect:+.1f} ({100 * low:+.1f} bis {100 * high:+.1f}) pp" if has_ci
                 else f"{100 * effect:+.1f} pp; CI fehlt" if effect is not None
                 else "nicht schätzbar"),
            )
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(value))
            if not has_ci:
                continue
            y = float(len(snapshot.rows) - index)
            colour = "#2b8cbe"
            self.plot.plot([100 * low, 100 * high], [y, y], pen=pg.mkPen(colour, width=2))
            self.plot.plot([100 * effect], [y], pen=None, symbol="o", symbolSize=9,
                           symbolBrush=colour)
            ticks.append((y, label[:28]))
        if ticks:
            self.plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#777777", width=1)))
        self.plot.getAxis("left").setTicks([ticks])
        self.plot.getAxis("left").setWidth(160)

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "stage": "therapy-response", "stage_version": THERAPY_RESPONSE_VERSION,
            "claim_scope": "physiological_response_association_not_causal_effect",
        }

    def _export_plot(self, *_args) -> None:
        if self._snapshot is not None and self.plot.listDataItems():
            choose_plot_export(self, self.plot, self._metadata(),
                               proposed_name="therapie_response.png")

    def _export_table(self, *_args) -> None:
        if self._snapshot is not None and not self._snapshot.rows.empty:
            choose_table_export(self, self._snapshot.rows, self._metadata(),
                                proposed_name="therapie_response.csv")
