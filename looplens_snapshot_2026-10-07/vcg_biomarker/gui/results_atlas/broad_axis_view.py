"""Validation-wide density map of the prespecified broad-axis screen."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QRectF, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...mimic.broad_axis_stage import BROAD_AXIS_STAGE_VERSION
from ...mimic.result_query import ResultRepository
from ...mimic.storage import StageManifest, verify_output_integrity
from .export import choose_plot_export, choose_table_export


@dataclass(frozen=True)
class BroadAxisSnapshot:
    status: str
    note: str
    families: pd.DataFrame
    bins: pd.DataFrame
    run_id: str = ""
    manifest_hash: str = ""
    qq: pd.DataFrame = field(default_factory=pd.DataFrame)


_AXIS_NAMES = {
    "laboratory": "Labor", "medication": "Medikamente", "procedure": "Prozeduren",
    "outcome": "Ereignisse", "icu": "Intensivstation",
}
_RELATION_NAMES = {
    "before_ecg": "vor dem EKG", "after_ecg": "nach dem EKG",
    "same_day_as_ecg": "am EKG-Tag", "same_time_as_ecg": "zeitgleich",
    "undated_admission_context": "ohne Zeitbezug", "unknown": "Zeitbezug unklar",
}


def _family_label(axis: str, relation: str, kind: str) -> str:
    return " · ".join((
        _AXIS_NAMES.get(axis, axis.replace("_", " ")),
        _RELATION_NAMES.get(relation, relation.replace("_", " ")),
        "binär" if kind == "binary" else "kontinuierlich",
    ))


def _verified_output(manifest: StageManifest, path: Path) -> bool:
    resolved = str(path.resolve())
    if resolved not in {str(Path(item).resolve()) for item in manifest.output_paths}:
        return False
    records = [record for record in manifest.output_integrity
               if str(Path(str(record.get("path", ""))).resolve()) == resolved]
    return len(records) == 1 and verify_output_integrity(records, mode="sha256")


def load_broad_axis_snapshot(repository: ResultRepository, run_id: str) -> BroadAxisSnapshot:
    """Read only a completed, hash-verified stage of exactly the selected run."""
    empty = pd.DataFrame()
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        return BroadAxisSnapshot("incompatible", "Ungültige Run-Kennung.", empty, empty, run_id)
    runs_root = repository.config.runs_root.resolve()
    run_root = (runs_root / run_id).resolve()
    if not run_root.is_relative_to(runs_root):
        return BroadAxisSnapshot("incompatible", "Run-Pfad liegt außerhalb des Ergebnisbereichs.",
                                 empty, empty, run_id)
    root = run_root / "broad_axes"
    manifest_path = root / "broad_axes.manifest.json"
    if not manifest_path.is_file():
        return BroadAxisSnapshot("pending", "Der vollständige breite Achsenscan liegt noch nicht vor.",
                                 empty, empty, run_id)
    try:
        manifest = StageManifest.load(manifest_path)
    except (OSError, TypeError, ValueError):
        return BroadAxisSnapshot("incompatible", "Manifest des breiten Scans ist unlesbar.",
                                 empty, empty, run_id)
    if (manifest.stage != "screen-broad-axes" or manifest.status != "complete"
            or manifest.stage_version != BROAD_AXIS_STAGE_VERSION):
        return BroadAxisSnapshot("pending", "Breiter Scan ist unvollständig oder versionsfremd.",
                                 empty, empty, run_id)
    screening = root / "broad_axis_screening.parquet"
    summary_path = root / "broad_axis_summary.json"
    if not (_verified_output(manifest, screening) and _verified_output(manifest, summary_path)):
        return BroadAxisSnapshot("incompatible", "Achsenscan-Dateien sind nicht vollständig hashgeprüft.",
                                 empty, empty, run_id)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return BroadAxisSnapshot("incompatible", "Zusammenfassung des Achsenscans ist unlesbar.",
                                 empty, empty, run_id)
    if (summary.get("run_id") != run_id or summary.get("locked_test_read") is not False
            or summary.get("fdr_family_definition") != "per_axis_not_pooled_across_axes"):
        return BroadAxisSnapshot("incompatible", "Run-, Split- oder FDR-Herkunft ist nicht belegt.",
                                 empty, empty, run_id)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        schema = {column[0] for column in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(screening)],
        ).description}
        required = {"axis", "temporal_relation", "target_kind", "evaluation_split",
                    "effect", "q_value", "target_id", "feature_id"}
        if not required.issubset(schema):
            return BroadAxisSnapshot("incompatible", "Achsenscan besitzt nicht den benötigten Vertrag.",
                                     empty, empty, run_id)
        relation = "COALESCE(CAST(temporal_relation AS VARCHAR), 'unknown')"
        valid = (
            "evaluation_split = 'validation' AND target_kind IN ('binary', 'continuous') "
            "AND isfinite(TRY_CAST(effect AS DOUBLE)) "
            "AND TRY_CAST(q_value AS DOUBLE) BETWEEN 0 AND 1"
        )
        families = connection.execute(
            f"SELECT axis, {relation} AS temporal_relation, target_kind, "
            "COUNT(*) AS tested_rows, COUNT(DISTINCT target_id) AS targets, "
            "COUNT(DISTINCT feature_id) AS markers, "
            "COUNT(*) FILTER (WHERE TRY_CAST(q_value AS DOUBLE) <= 0.05) AS fdr_rows "
            f"FROM read_parquet(?) WHERE {valid} "
            "GROUP BY axis, temporal_relation, target_kind "
            "ORDER BY axis, temporal_relation, target_kind", [str(screening)],
        ).fetchdf()
        normalized_x = (
            "CASE WHEN target_kind = 'binary' THEN TRY_CAST(effect AS DOUBLE) "
            "ELSE (TRY_CAST(effect AS DOUBLE) + 1.0) / 2.0 END"
        )
        bins = connection.execute(
            "WITH valid_rows AS ("
            f"SELECT axis, {relation} AS temporal_relation, target_kind, "
            f"{normalized_x} AS x, "
            "LEAST(19.999, -log10(GREATEST(TRY_CAST(q_value AS DOUBLE), 1e-300))) AS y "
            f"FROM read_parquet(?) WHERE {valid}"
            "), binned AS ("
            "SELECT axis, temporal_relation, target_kind, "
            "CAST(FLOOR(LEAST(0.999999, GREATEST(0.0, x)) * 60) AS INTEGER) AS x_bin, "
            "CAST(FLOOR(GREATEST(0.0, y) * 2) AS INTEGER) AS y_bin "
            "FROM valid_rows WHERE x BETWEEN 0 AND 1"
            ") SELECT axis, temporal_relation, target_kind, x_bin, y_bin, "
            "COUNT(*) AS n FROM binned "
            "GROUP BY axis, temporal_relation, target_kind, x_bin, y_bin",
            [str(screening)],
        ).fetchdf()
        qq = empty
        if "p_value" in schema:
            fractions = np.unique(np.r_[np.geomspace(1e-5, .5, 70), .75, .9])
            fraction_sql = "[" + ",".join(f"{fraction:.12g}" for fraction in fractions) + "]"
            qq_aggregates = connection.execute(
                f"SELECT axis, {relation} AS temporal_relation, target_kind, "
                "COUNT(*) AS n_tests, "
                f"approx_quantile(GREATEST(TRY_CAST(p_value AS DOUBLE), 1e-300), "
                f"{fraction_sql}::FLOAT[]) "
                "AS observed_p FROM read_parquet(?) "
                f"WHERE {valid} "
                "AND TRY_CAST(p_value AS DOUBLE) BETWEEN 0 AND 1 "
                "GROUP BY axis, temporal_relation, target_kind",
                [str(screening)],
            ).fetchdf()
            rows = []
            for _, row in qq_aggregates.iterrows():
                for fraction, observed in zip(fractions, row.observed_p):
                    if float(fraction) < 1 / max(int(row.n_tests), 1):
                        continue
                    rows.append({
                        "axis": row.axis, "temporal_relation": row.temporal_relation,
                        "target_kind": row.target_kind, "n_tests": int(row.n_tests),
                        "expected_logp": float(-np.log10(fraction)),
                        "observed_logp": float(-np.log10(max(float(observed), 1e-300))),
                        "quantile_fraction": float(fraction),
                    })
            qq = pd.DataFrame(rows)
    finally:
        connection.close()
    return BroadAxisSnapshot(
        "ok" if not families.empty else "empty",
        "Alle gültigen Validation-Testzeilen des hashgeprüften Achsenscans; "
        "FDR-Familie = Achse × Zeitbezug × Split. Validation-FDR allein ist "
        "keine Replikation, Adjustierung oder klinische Entdeckung.",
        families, bins, run_id, str(manifest.input_hash), qq,
    )


class BroadAxisLoad(QThread):
    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, generation: int, repository: ResultRepository, run_id: str):
        super().__init__()
        self.generation = generation
        self.repository = repository
        self.run_id = run_id

    def run(self) -> None:
        try:
            self.ready.emit(self.generation, load_broad_axis_snapshot(self.repository, self.run_id))
        except Exception as exc:
            self.failed.emit(self.generation,
                             f"Achsenscan konnte nicht gelesen werden: {type(exc).__name__}: {exc}")


class AtlasBroadAxisWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: BroadAxisSnapshot | None = None
        self._selected_bins = pd.DataFrame()
        self._selected_qq = pd.DataFrame()
        layout = QVBoxLayout(self)
        self.note = QLabel("Der vollständige breite Achsenscan wird geprüft …")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        family_row = QHBoxLayout()
        family_row.addWidget(QLabel("Achse, Zeitbezug und Zieltyp"))
        self.family = QComboBox()
        self.family.currentIndexChanged.connect(self._draw)
        family_row.addWidget(self.family, 1)
        layout.addLayout(family_row)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(300)
        self.plot.setLabel("left", "−log10(q); Werte ≥20 gedeckelt")
        self.qq_plot = pg.PlotWidget()
        self.qq_plot.setMinimumHeight(300)
        self.qq_plot.setLabel("bottom", "Erwartetes −log10(p) unter uniformem Nullmodell")
        self.qq_plot.setLabel("left", "Beobachtetes −log10(p)")
        self.qq_plot.showGrid(x=True, y=True, alpha=.15)
        self.visual_tabs = QTabWidget()
        self.visual_tabs.addTab(self.plot, "Effekt × FDR-Dichte")
        self.visual_tabs.addTab(self.qq_plot, "QQ-Diagnostik")
        layout.addWidget(self.visual_tabs)
        self.detail = QLabel("Farbintensität = Zahl der Testzeilen pro Effekt-/q-Bin (logarithmisch skaliert).")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.qq_note = QLabel(
            "QQ-Punkte sind approximierte p-Wert-Quantile derselben Achse/Zeitscope/Zielart. "
            "Korrelation der Marker, echte Effekte und Selektion können Abweichungen erzeugen; "
            "dies ist kein formaler Inflations- oder Kalibrationstest."
        )
        self.qq_note.setWordWrap(True)
        layout.addWidget(self.qq_note)
        self.families_table = QTableWidget(0, 5)
        self.families_table.setHorizontalHeaderLabels((
            "Achse", "Zeitbezug / Zieltyp", "Tests", "Ziele / Marker", "FDR ≤ 0,05",
        ))
        self.families_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.families_table)
        export_row = QHBoxLayout()
        plot_export = QPushButton("Dichtekarte als PNG/SVG")
        plot_export.setToolTip("Exportiert die gewählte vollständige Validation-FDR-Familie mit Herkunftsdatei.")
        plot_export.clicked.connect(self._export_plot)
        export_row.addWidget(plot_export)
        qq_export = QPushButton("QQ-Diagnostik als PNG/SVG")
        qq_export.setToolTip("Exportiert approximierte p-Wert-Quantile, keinen formalen Nullmodelltest.")
        qq_export.clicked.connect(self._export_qq_plot)
        export_row.addWidget(qq_export)
        bins_export = QPushButton("Dichte-Bins als CSV")
        bins_export.setToolTip("Exportiert die gezählten Effekt-/q-Bins der gewählten FDR-Familie.")
        bins_export.clicked.connect(self._export_bins)
        export_row.addWidget(bins_export)
        qq_data_export = QPushButton("QQ-Punkte als CSV")
        qq_data_export.setToolTip("Exportiert die Quantile mit Testzahl und FDR-Familienbezug.")
        qq_data_export.clicked.connect(self._export_qq)
        export_row.addWidget(qq_data_export)
        summary_export = QPushButton("Familienübersicht als CSV")
        summary_export.setToolTip("Exportiert Testzahlen und FDR-Zählung aller belegten Familien.")
        summary_export.clicked.connect(self._export_families)
        export_row.addWidget(summary_export)
        export_row.addStretch(1)
        layout.addLayout(export_row)

    def set_snapshot(self, snapshot: BroadAxisSnapshot) -> None:
        self._snapshot = snapshot
        self.note.setText(snapshot.note)
        self.plot.clear()
        self.qq_plot.clear()
        self._selected_bins = pd.DataFrame()
        self._selected_qq = pd.DataFrame()
        previous = self.family.currentData()
        self.family.blockSignals(True)
        self.family.clear()
        self.families_table.setRowCount(len(snapshot.families))
        for index, (_, row) in enumerate(snapshot.families.iterrows()):
            key = (str(row["axis"]), str(row["temporal_relation"]), str(row["target_kind"]))
            self.family.addItem(_family_label(*key), key)
            values = (
                _AXIS_NAMES.get(key[0], key[0]),
                f"{_RELATION_NAMES.get(key[1], key[1])} / "
                f"{'binär' if key[2] == 'binary' else 'kontinuierlich'}",
                str(int(row["tested_rows"])),
                f"{int(row['targets'])} / {int(row['markers'])}", str(int(row["fdr_rows"])),
            )
            for column, value in enumerate(values):
                self.families_table.setItem(index, column, QTableWidgetItem(value))
        if previous is not None:
            index = self.family.findData(previous)
            if index >= 0:
                self.family.setCurrentIndex(index)
        self.family.blockSignals(False)
        self._draw()

    def _draw(self, *_args) -> None:
        self.plot.clear()
        self.qq_plot.clear()
        snapshot = self._snapshot
        key = self.family.currentData()
        if snapshot is None or key is None:
            self._selected_bins = pd.DataFrame()
            self._selected_qq = pd.DataFrame()
            self.detail.setText("Keine vollständige, darstellbare FDR-Familie vorhanden.")
            return
        axis, relation, kind = key
        self._draw_qq(snapshot, axis, relation, kind)
        if snapshot.bins.empty:
            self._selected_bins = pd.DataFrame()
            self.detail.setText("Keine gültigen Effekt-/q-Bins für diese Familie.")
            return
        frame = snapshot.bins.loc[
            snapshot.bins["axis"].astype(str).eq(axis)
            & snapshot.bins["temporal_relation"].astype(str).eq(relation)
            & snapshot.bins["target_kind"].astype(str).eq(kind)
        ]
        self._selected_bins = frame.copy()
        if frame.empty:
            self.detail.setText("Keine gültigen Effekt-/q-Bins für diese Familie.")
            return
        grid = np.zeros((60, 40), dtype=np.float64)
        for _, row in frame.iterrows():
            x, y = int(row["x_bin"]), int(row["y_bin"])
            if 0 <= x < 60 and 0 <= y < 40:
                grid[x, y] = float(row["n"])
        rgba = np.zeros((60, 40, 4), dtype=np.uint8)
        nonzero = grid > 0
        maximum = float(grid.max())
        lookup = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256, alpha=True)
        indices = np.clip(np.rint(np.log1p(grid) / math.log1p(maximum) * 255), 0, 255).astype(np.uint8)
        rgba[nonzero] = lookup[indices[nonzero]]
        rgba[~nonzero, 3] = 0
        image = pg.ImageItem(rgba)
        x_min, width = (0., 1.) if kind == "binary" else (-1., 2.)
        image.setRect(QRectF(x_min, 0., width, 20.))
        self.plot.addItem(image)
        self.plot.addItem(pg.InfiniteLine(pos=.5 if kind == "binary" else 0., angle=90,
                                         pen=pg.mkPen("#ffffff", width=1)))
        self.plot.addItem(pg.InfiniteLine(
            pos=-math.log10(.05), angle=0,
            pen=pg.mkPen("#cf7d25", width=1, style=Qt.PenStyle.DashLine),
        ))
        self.plot.setXRange(x_min, x_min + width, padding=0)
        self.plot.setYRange(0, 20, padding=0)
        self.plot.setLabel("bottom", "AUC (0,5 = neutral)" if kind == "binary"
                           else "Spearman-ρ (0 = neutral)")
        family = snapshot.families.loc[
            snapshot.families["axis"].astype(str).eq(axis)
            & snapshot.families["temporal_relation"].astype(str).eq(relation)
            & snapshot.families["target_kind"].astype(str).eq(kind)
        ].iloc[0]
        self.detail.setText(
            f"{int(family['tested_rows']):,} gültige Validation-Tests; "
            f"{int(family['fdr_rows']):,} mit q ≤ 0,05 in diesem Zieltyp-Ausschnitt. "
            "Die q-Werte wurden für Achse × Zeitbezug × Split über binäre und "
            "kontinuierliche Ziele gemeinsam korrigiert. "
            "Die gestrichelte waagrechte Linie markiert q = 0,05; "
            "Jede Zelle zählt Marker–Ziel-Testzeilen, keine Patienten. "
            f"Farbe zeigt log(1+n), von 1 bis höchstens {int(maximum):,} "
            "Testzeilen je Rasterzelle; leere Zellen sind transparent. "
            "Dies ist der vollständige gültige Zieltyp-Ausschnitt, keine vorgewählte Topliste."
        )

    def _draw_qq(self, snapshot: BroadAxisSnapshot, axis: str, relation: str, kind: str) -> None:
        frame = snapshot.qq
        if frame.empty:
            self._selected_qq = pd.DataFrame()
            self.qq_note.setText("Keine gültigen p-Wert-Quantile für diese Familie vorhanden.")
            return
        selected = frame.loc[
            frame.axis.astype(str).eq(axis)
            & frame.temporal_relation.astype(str).eq(relation)
            & frame.target_kind.astype(str).eq(kind)
        ].sort_values("expected_logp")
        self._selected_qq = selected.copy()
        if selected.empty:
            self.qq_note.setText("Für diese Familie sind keine QQ-Punkte berechenbar.")
            return
        x = selected.expected_logp.to_numpy(dtype=float)
        y = selected.observed_logp.to_numpy(dtype=float)
        limit = max(2., float(np.nanmax(np.r_[x, np.minimum(y, 15.)])))
        self.qq_plot.plot([0, limit], [0, limit],
                          pen=pg.mkPen("#999999", width=1, style=Qt.PenStyle.DashLine))
        self.qq_plot.plot(x, np.minimum(y, 15.), pen=pg.mkPen("#286d87", width=2),
                          symbol="o", symbolSize=4)
        self.qq_plot.setXRange(0, limit, padding=0)
        self.qq_plot.setYRange(0, max(limit, min(float(np.max(y)), 15.)), padding=0)
        capped = int((y > 15).sum())
        self.qq_note.setText(
            f"{int(selected.iloc[0].n_tests):,} p-Wert-Testzeilen, "
            f"{len(selected)} approximierte Quantile; {capped} Punkte bei −log10(p)=15 "
            "visuell gedeckelt. Korrelierte Marker, echte Effekte und Zielselektion "
            "verändern die QQ-Form; kein formaler Inflations- oder Kalibrationstest."
        )

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "manifest_input_hash": snapshot.manifest_hash if snapshot else None,
            "family": self.family.currentData(), "split": "validation",
            "fdr_family": "axis_x_temporal_relation_x_split",
            "locked_test_read": False,
            "note": "Gesamter breiter Achsenscan; explorativ, keine klinische Freigabe.",
        }

    def _export_plot(self, *_args) -> None:
        if not self._selected_bins.empty:
            choose_plot_export(self, self.plot, self._metadata(),
                               proposed_name="breiter_achsenscan_dichte.png")

    def _export_bins(self, *_args) -> None:
        if not self._selected_bins.empty:
            choose_table_export(self, self._selected_bins, self._metadata(),
                                proposed_name="breiter_achsenscan_bins.csv")

    def _export_qq_plot(self, *_args) -> None:
        if not self._selected_qq.empty:
            choose_plot_export(self, self.qq_plot, self._metadata(),
                               proposed_name="breiter_achsenscan_qq.png")

    def _export_qq(self, *_args) -> None:
        if not self._selected_qq.empty:
            choose_table_export(self, self._selected_qq, self._metadata(),
                                proposed_name="breiter_achsenscan_qq.csv")

    def _export_families(self, *_args) -> None:
        if self._snapshot is not None and not self._snapshot.families.empty:
            choose_table_export(self, self._snapshot.families, self._metadata(),
                                proposed_name="breiter_achsenscan_familien.csv")
