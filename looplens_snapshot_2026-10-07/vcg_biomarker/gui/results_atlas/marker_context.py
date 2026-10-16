"""Marker-specific hospital distributions and analytical robustness."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultPage
from ...mimic.reference_ranges import REFERENCE_RANGE_STAGE_VERSION
from ...mimic.repeatability_stage import REPEATABILITY_STAGE_VERSION
from ...mimic.storage import StageManifest, verify_output_integrity
from ...mimic.transform_agreement import TRANSFORM_AGREEMENT_VERSION
from .export import choose_plot_export, choose_table_export


@dataclass(frozen=True)
class MarkerContext:
    marker_id: str
    ranges: pd.DataFrame
    transforms: pd.DataFrame
    repeatability: pd.DataFrame
    source_notes: tuple[str, ...]
    run_id: str = ""
    snapshot_hash: str = ""


_SOURCES = (
    ("Krankenhaus-Verteilung", "reference_ranges/reference_ranges.manifest.json",
     "reference_ranges/marker_reference_ranges.parquet", "marker_id", False,
     "reference-ranges", REFERENCE_RANGE_STAGE_VERSION),
    ("Transformvergleich", "guideline/transform_agreement.manifest.json",
     "guideline/transform_agreement.parquet", "marker_id", False,
     "transform-agreement", TRANSFORM_AGREEMENT_VERSION),
    ("Wiederholbarkeit", "repeatability/repeatability.manifest.json",
     "repeatability/marker_repeatability.parquet", "marker_id", True,
     "repeatability", REPEATABILITY_STAGE_VERSION),
)


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: object, digits: int = 2) -> str:
    number = _finite(value)
    return f"{number:.{digits}f}" if number is not None else "nicht berechnet"


def _integer(value: object) -> str:
    number = _finite(value)
    return str(int(number)) if number is not None and number >= 0 and number.is_integer() else "?"


def _percentile_text(row: pd.Series, column: str) -> str:
    value = _finite(row.get(column))
    if value is None:
        return "nicht berechnet"
    low = _finite(row.get(f"{column}_ci_low"))
    high = _finite(row.get(f"{column}_ci_high"))
    if low is not None and high is not None and low <= value <= high:
        return f"{value:.2f} (CI {low:.2f}–{high:.2f})"
    return f"{value:.2f} (CI nicht berechnet)"


def _read_marker_table(path: Path, marker_id: str, key: str) -> pd.DataFrame:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        schema = {column[0] for column in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)],
        ).description}
        if key not in schema:
            raise ValueError(f"{path.name}: marker key missing")
        return connection.execute(
            f'SELECT * FROM read_parquet(?) WHERE "{key}" = ? LIMIT 200',
            [str(path), marker_id],
        ).fetchdf()
    finally:
        connection.close()


def load_marker_context(page: ResultPage, marker_id: str) -> MarkerContext:
    empty = pd.DataFrame()
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return MarkerContext(marker_id, empty, empty, empty,
                             ("Kein kompatibler abgeschlossener Ergebnisstand.",),
                             page.run_id, page.snapshot_hash)
    run_root = page.artifact_path.parent
    frames = []
    notes = []
    for title, manifest_rel, table_rel, key, require_hash, stage, version in _SOURCES:
        manifest_path = run_root / manifest_rel
        table_path = run_root / table_rel
        try:
            manifest = StageManifest.load(manifest_path)
        except (OSError, TypeError, ValueError):
            frames.append(empty)
            notes.append(f"{title}: Stage-Manifest fehlt oder ist unlesbar.")
            continue
        resolved = str(table_path.resolve())
        if (manifest.status != "complete" or manifest.stage != stage
                or manifest.stage_version != version or resolved not in {
            str(Path(path).resolve()) for path in manifest.output_paths
        } or not table_path.is_file()):
            frames.append(empty)
            notes.append(f"{title}: noch nicht vollständig oder versionspassend manifestiert.")
            continue
        records = [record for record in manifest.output_integrity
                   if str(Path(str(record.get("path", ""))).resolve()) == resolved]
        hashed = len(records) == 1 and verify_output_integrity(records, mode="sha256")
        if require_hash and not hashed:
            frames.append(empty)
            notes.append(f"{title}: Dateiintegrität nicht hashgeprüft; keine Anzeige.")
            continue
        if records and not hashed:
            frames.append(empty)
            notes.append(f"{title}: veränderter oder unlesbarer Stage-Output.")
            continue
        try:
            frame = _read_marker_table(table_path, marker_id, key)
        except (duckdb.Error, OSError, ValueError):
            frames.append(empty)
            notes.append(f"{title}: Datentabelle unlesbar oder Vertragsfeld fehlt.")
            continue
        frames.append(frame)
        notes.append(
            f"{title}: {'hashgeprüft' if hashed else 'älteres Manifest ohne Dateihash; deskriptiv'}; "
            f"{len(frame)} Zeilen für den Marker."
        )
    return MarkerContext(marker_id, frames[0], frames[1], frames[2],
                         tuple(notes), page.run_id, page.snapshot_hash)


class AtlasMarkerContextWidget(QTabWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._context: MarkerContext | None = None
        self._selected_ranges = pd.DataFrame()
        self._selected_transforms = pd.DataFrame()
        self.ranges_page = QWidget()
        ranges_layout = QVBoxLayout(self.ranges_page)
        self.ranges_note = QLabel("Krankenhaus-Verteilungen werden geprüft …")
        self.ranges_note.setWordWrap(True)
        ranges_layout.addWidget(self.ranges_note)
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Herzfrequenz-Gruppe"))
        self.rate = QComboBox()
        self.rate.currentIndexChanged.connect(self._draw_ranges)
        filters.addWidget(self.rate)
        filters.addWidget(QLabel("Geschlecht"))
        self.sex = QComboBox()
        self.sex.currentIndexChanged.connect(self._draw_ranges)
        filters.addWidget(self.sex)
        filters.addStretch(1)
        ranges_layout.addLayout(filters)
        self.ranges_plot = pg.PlotWidget()
        self.ranges_plot.setMinimumHeight(230)
        self.ranges_plot.setLabel("bottom", "Markerwert; Einheit laut Feature-Freeze")
        ranges_layout.addWidget(self.ranges_plot)
        self.ranges_table = QTableWidget(0, 7)
        self.ranges_table.setHorizontalHeaderLabels((
            "Alter", "Geschlecht", "Rate", "Menschen / Messungen",
            "P2,5 (95-%-CI)", "Median", "P97,5 (95-%-CI)",
        ))
        self.ranges_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ranges_layout.addWidget(self.ranges_table)
        ranges_export = QPushButton("Verteilungsansicht als PNG/SVG")
        ranges_export.setToolTip("Exportiert ausschließlich gemessene Krankenhaus-Perzentile mit Herkunftsdatei.")
        ranges_export.clicked.connect(self._export_ranges_plot)
        ranges_layout.addWidget(ranges_export)
        ranges_table_export = QPushButton("Verteilungswerte als CSV")
        ranges_table_export.setToolTip("Exportiert alle geladenen Alters-/Geschlechtsgruppen mit Herkunftsdatei.")
        ranges_table_export.clicked.connect(self._export_ranges_table)
        ranges_layout.addWidget(ranges_table_export)
        self.addTab(self.ranges_page, "Krankenhaus-Verteilung")

        self.transforms_page = QWidget()
        transform_layout = QVBoxLayout(self.transforms_page)
        self.transforms_note = QLabel("Transformationsvergleich wird geprüft …")
        self.transforms_note.setWordWrap(True)
        transform_layout.addWidget(self.transforms_note)
        self.transforms_plot = pg.PlotWidget()
        self.transforms_plot.setMinimumHeight(180)
        self.transforms_plot.setLabel("bottom", "Differenz: Vergleichstransform minus Primärtransform")
        transform_layout.addWidget(self.transforms_plot)
        self.transforms_table = QTableWidget(0, 6)
        self.transforms_table.setHorizontalHeaderLabels((
            "Methoden", "Paare", "Bias (CI)", "95-%-Differenzbereich", "ρ", "Aussagegrenze",
        ))
        self.transforms_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        transform_layout.addWidget(self.transforms_table)
        transforms_export = QPushButton("Methodenübersicht als CSV")
        transforms_export.setToolTip("Exportiert die Transformationskennzahlen; keine Einzelpaar-Bland-Altman-Punkte.")
        transforms_export.clicked.connect(self._export_transforms)
        transform_layout.addWidget(transforms_export)
        transforms_plot_export = QPushButton("Methodendifferenzen als PNG/SVG")
        transforms_plot_export.setToolTip("Exportiert Bias und Limits of Agreement als Kennzahlenübersicht, keine Einzelpaare.")
        transforms_plot_export.clicked.connect(self._export_transforms_plot)
        transform_layout.addWidget(transforms_plot_export)
        self.addTab(self.transforms_page, "Messmethoden")

        self.repeat_page = QWidget()
        repeat_layout = QVBoxLayout(self.repeat_page)
        self.repeat_note = QLabel("Wiederholbarkeit wird geprüft …")
        self.repeat_note.setWordWrap(True)
        repeat_layout.addWidget(self.repeat_note)
        self.repeat_table = QTableWidget(0, 6)
        self.repeat_table.setHorizontalHeaderLabels((
            "Status", "Replikatpaare", "ICC", "Wiederholbarkeitskoeffizient",
            "Kleinste detektierbare Änderung", "Aussagegrenze",
        ))
        self.repeat_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        repeat_layout.addWidget(self.repeat_table)
        repeat_export = QPushButton("Wiederholbarkeit als CSV")
        repeat_export.setToolTip("Exportiert nur hashgeprüfte Repeatability-Kennzahlen mit Herkunftsdatei.")
        repeat_export.clicked.connect(self._export_repeat)
        repeat_layout.addWidget(repeat_export)
        self.addTab(self.repeat_page, "Wiederholbarkeit")

    def set_context(self, context: MarkerContext) -> None:
        self._context = context
        self._set_ranges(context.ranges)
        self._set_transforms(context.transforms)
        self._set_repeatability(context.repeatability)
        for note, source in zip(
            (self.transforms_note, self.repeat_note), context.source_notes[1:],
        ):
            note.setText(source + " " + note.text())

    def _set_ranges(self, frame: pd.DataFrame) -> None:
        self.rate.blockSignals(True)
        self.sex.blockSignals(True)
        self.rate.clear()
        self.sex.clear()
        if not frame.empty and {"rate_band", "sex_band"}.issubset(frame):
            for value in sorted(frame["rate_band"].dropna().astype(str).unique()):
                self.rate.addItem(value, value)
            self.sex.addItem("Alle", "")
            for value in sorted(frame["sex_band"].dropna().astype(str).unique()):
                self.sex.addItem(value, value)
        self.rate.blockSignals(False)
        self.sex.blockSignals(False)
        self._draw_ranges()

    def _draw_ranges(self, *_args) -> None:
        self.ranges_plot.clear()
        self.ranges_table.setRowCount(0)
        context = self._context
        if context is None or context.ranges.empty or not self.rate.currentData():
            self._selected_ranges = pd.DataFrame()
            prefix = context.source_notes[0] + " " if context and context.source_notes else ""
            self.ranges_note.setText(prefix + "Keine messbaren Krankenhaus-Perzentile für diesen Marker vorhanden.")
            return
        frame = context.ranges
        selected = frame.loc[frame["rate_band"].astype(str).eq(str(self.rate.currentData()))]
        sex = str(self.sex.currentData() or "")
        if sex:
            selected = selected.loc[selected["sex_band"].astype(str).eq(sex)]
        self._selected_ranges = selected.copy()
        shown = selected.head(25)
        self.ranges_table.setRowCount(len(shown))
        ticks = []
        plotted = 0
        for index, (_, row) in enumerate(shown.iterrows()):
            age = str(row.get("age_band", "?"))
            sex_label = str(row.get("sex_band", "?"))
            values = (
                age, sex_label, str(row.get("rate_band", "?")),
                f"{_integer(row.get('subject_count'))} / {_integer(row.get('observed_count'))}",
                _percentile_text(row, "percentile_2_5"), _fmt(row.get("percentile_50")),
                _percentile_text(row, "percentile_97_5"),
            )
            for column, value in enumerate(values):
                self.ranges_table.setItem(index, column, QTableWidgetItem(value))
            low, median, high = (_finite(row.get(name)) for name in (
                "percentile_2_5", "percentile_50", "percentile_97_5"
            ))
            if str(row.get("estimable", "")).lower() != "true" or None in (low, median, high):
                continue
            if not low <= median <= high:
                continue
            y = float(len(shown) - index)
            ticks.append((y, f"{age} · {sex_label}"[:25]))
            self.ranges_plot.plot([low, high], [y, y], pen=pg.mkPen("#2a9195", width=2))
            self.ranges_plot.plot([median], [y], pen=None, symbol="o", symbolSize=7,
                                  symbolBrush="#2a9195")
            plotted += 1
        self.ranges_plot.getAxis("left").setTicks([ticks])
        self.ranges_plot.setYRange(.5, max(1.5, len(shown) + .5), padding=0)
        self.ranges_note.setText(
            context.source_notes[0] + " "
            f"{len(selected)} Alters-/Geschlechtsgruppen in dieser Frequenzgruppe, "
            f"{plotted} mit berechenbaren Perzentilen (höchstens 25 gezeichnet). "
            "Krankenhauspatienten ohne dokumentierte ausgewählte Risiken, keine gesunde "
            "Referenzpopulation und keine diagnostischen Grenzwerte. "
            "Punkte = Median; Linien = P2,5–P97,5, nicht Konfidenzintervalle."
        )

    def _set_transforms(self, frame: pd.DataFrame) -> None:
        self._selected_transforms = frame.head(25).copy()
        self.transforms_plot.clear()
        self.transforms_table.setRowCount(len(self._selected_transforms))
        if frame.empty:
            self.transforms_note.setText("Kein manifestierter Transformvergleich für diesen Marker.")
            return
        self.transforms_plot.addItem(pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("#88939b", width=1),
        ))
        ticks = []
        for index, (_, row) in enumerate(self._selected_transforms.iterrows()):
            label = f"{row.get('primary_transform', '?')} → {row.get('comparison_transform', '?')}"
            bias = _finite(row.get("bias"))
            low, high = (_finite(row.get(name)) for name in (
                "limit_of_agreement_low", "limit_of_agreement_high"
            ))
            y = float(len(self._selected_transforms) - index)
            ticks.append((y, label[:30]))
            if bias is not None:
                self.transforms_plot.plot([bias], [y], pen=None, symbol="o",
                                          symbolSize=8, symbolBrush="#2a9195")
            if low is not None and high is not None and low <= high:
                self.transforms_plot.plot([low, high], [y, y],
                                          pen=pg.mkPen("#82939e", width=2))
            bias_ci = f"{_fmt(bias)} ({_fmt(row.get('bias_ci_low'))}–{_fmt(row.get('bias_ci_high'))})"
            values = (
                label, _integer(row.get("paired_count")), bias_ci,
                f"{_fmt(low)}–{_fmt(high)}", _fmt(row.get("spearman_rho")),
                str(row.get("agreement_claim_scope") or "nicht belegt"),
            )
            for column, value in enumerate(values):
                self.transforms_table.setItem(index, column, QTableWidgetItem(value))
        self.transforms_plot.getAxis("left").setTicks([ticks])
        self.transforms_plot.setYRange(.5, len(self._selected_transforms) + .5, padding=0)
        self.transforms_note.setText(
            "Aus stichprobenartig ausgewählten Trainingspartitionen. "
            "Punkt = mittlere Methodendifferenz, Linie = 95-%-Differenzbereich "
            "(Limits of Agreement), kein CI und kein Bland–Altman-Streudiagramm "
            "der Einzelpaare. Korrelation allein beweist keine Austauschbarkeit."
        )

    def _set_repeatability(self, frame: pd.DataFrame) -> None:
        self.repeat_table.setRowCount(min(25, len(frame)))
        if frame.empty:
            self.repeat_note.setText("Keine hashgeprüften Replikatkennzahlen für diesen Marker.")
            return
        for index, (_, row) in enumerate(frame.head(25).iterrows()):
            values = (
                str(row.get("status") or "unbekannt"),
                _integer(row.get("n_pairs_evaluable", row.get("n_pairs", row.get("paired_count")))),
                _fmt(row.get("icc")), _fmt(row.get("repeatability_coefficient")),
                _fmt(row.get("smallest_detectable_change")),
                str(row.get("estimand") or "Messfehler plus kurzfristige physiologische Änderung"),
            )
            for column, value in enumerate(values):
                self.repeat_table.setItem(index, column, QTableWidgetItem(value))
        self.repeat_note.setText(
            "Replikatpaare liefern eine obere Schranke für Messfehler, weil echte "
            "kurzfristige Physiologie mitwirken kann. Ein individueller klinischer "
            "Änderungsschwellenwert folgt daraus nicht automatisch."
        )

    def _metadata(self) -> dict[str, object]:
        context = self._context
        return {
            "run_id": context.run_id if context else None,
            "snapshot_hash": context.snapshot_hash if context else None,
            "marker_id": context.marker_id if context else None,
            "source_notes": context.source_notes if context else (),
            "note": "Forschungsansicht; Krankenhaus-Perzentile sind keine Normalwerte.",
        }

    def _export_ranges_plot(self, *_args) -> None:
        if not self._selected_ranges.empty:
            choose_plot_export(self, self.ranges_plot, self._metadata(),
                               proposed_name="marker_krankenhaus_perzentile.png")

    def _export_ranges_table(self, *_args) -> None:
        if not self._selected_ranges.empty:
            choose_table_export(self, self._selected_ranges, self._metadata(),
                                proposed_name="marker_krankenhaus_perzentile.csv")

    def _export_transforms(self, *_args) -> None:
        if not self._selected_transforms.empty:
            choose_table_export(self, self._selected_transforms, self._metadata(),
                                proposed_name="marker_transformvergleich.csv")

    def _export_transforms_plot(self, *_args) -> None:
        if not self._selected_transforms.empty:
            choose_plot_export(self, self.transforms_plot, self._metadata(),
                               proposed_name="marker_transformdifferenzen.png")

    def _export_repeat(self, *_args) -> None:
        if self._context is not None and not self._context.repeatability.empty:
            choose_table_export(self, self._context.repeatability, self._metadata(),
                                proposed_name="marker_wiederholbarkeit.csv")
