"""Endpoint-specific measurement availability and subgroup audit."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import duckdb
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultPage
from ...mimic.segment_availability import SEGMENT_AVAILABILITY_VERSION
from ...mimic.storage import StageManifest, verify_output_integrity
from .export import choose_plot_export, choose_table_export
from .row_scope import validation_output_assurance


@dataclass(frozen=True)
class AvailabilitySnapshot:
    endpoint_id: str
    marker_id: str
    reference_eligible: int | None
    raw_observed: int | None
    analysis_observed: int | None
    restriction_excluded: int | None
    subgroups: pd.DataFrame
    note: str
    missing_reasons: pd.DataFrame | None = None
    run_id: str = ""
    snapshot_hash: str = ""
    segment_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    segment_note: str = "Segmentbeurteilbarkeit wird geprüft."


def _count(value: object) -> int | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return int(result) if result >= 0 and result.is_integer() else None


def load_availability(
    page: ResultPage, endpoint_id: str, marker_id: str = "",
) -> AvailabilitySnapshot:
    empty = pd.DataFrame()
    if page.status not in {"ok", "empty"} or page.rows.empty or not endpoint_id:
        return AvailabilitySnapshot(endpoint_id, "", None, None, None, None, empty,
                                    "Für diesen Endpunkt liegen noch keine kompatiblen Messdaten vor.")
    scoped = page.rows.loc[page.rows["endpoint_id"].astype(str).eq(endpoint_id)]
    if marker_id and "feature_id" in scoped:
        scoped = scoped.loc[scoped["feature_id"].astype(str).eq(marker_id)]
    if scoped.empty:
        return AvailabilitySnapshot(endpoint_id, "", None, None, None, None, empty,
                                    "Für diesen Endpunkt ist kein auswahlfähiger Marker vorhanden.")
    row = scoped.iloc[0]
    marker = str(row.get("feature_id") or "")
    eligible = _count(row.get("n_reference_eligible"))
    raw = _count(row.get("n_marker_raw_observed"))
    observed = _count(row.get("n_marker_observed"))
    restricted = _count(row.get("n_analysis_restriction_excluded"))
    valid = (
        eligible is not None and raw is not None and observed is not None
        and 0 <= observed <= raw <= eligible
        and (restricted is None or restricted == raw - observed)
    )
    note = (
        "Der Zähler bezieht sich auf referenzgeeignete Index-EKGs, nicht auf alle "
        "Patienten der Datenbank. Nicht ausgewählt, fehlende Referenz und "
        "technischer Signalausfall sind nicht dieselbe Ursache."
        if valid else
        "Ein konsistenter, vollständig definierter Verfügbarkeitstrichter fehlt; "
        "deshalb werden keine Nenner erfunden."
    )
    if not valid:
        eligible = raw = observed = restricted = None
    reasons = pd.DataFrame(columns=["category", "count", "share_of_eligible_pct", "interpretation"])
    if valid:
        reasons = pd.DataFrame([
            {"category": "Analysegeeignet", "count": observed,
             "share_of_eligible_pct": 100 * observed / eligible if eligible else None,
             "interpretation": "Marker messbar und Analysebedingungen erfüllt"},
            {"category": "Analyseregel ausgeschlossen", "count": raw - observed,
             "share_of_eligible_pct": 100 * (raw - observed) / eligible if eligible else None,
             "interpretation": "Marker roh messbar; Analysebedingungen nicht erfüllt"},
            {"category": "Marker roh nicht messbar", "count": eligible - raw,
             "share_of_eligible_pct": 100 * (eligible - raw) / eligible if eligible else None,
             "interpretation": "Ursache im aggregierten Ergebnis nicht weiter aufgeschlüsselt"},
        ])
    subgroups = empty
    artifact = page.artifact_path
    if artifact is not None and marker:
        run_root = artifact.parent
        screening = run_root / "screening_results.parquet"
        try:
            manifest = StageManifest.load(run_root / "validate.manifest.json")
            assurance = validation_output_assurance(manifest, screening, page.run_id)
        except (OSError, TypeError, ValueError):
            assurance = None
        if assurance is not None:
            if assurance == "manifest_only":
                note += " Subgruppen-Datei nur im abgeschlossenen Manifest deklariert; Ausgabe-SHA-256 fehlt."
            connection = duckdb.connect(":memory:")
            try:
                connection.execute("SET memory_limit='512MB'")
                connection.execute("SET threads=1")
                schema = {column[0] for column in connection.execute(
                    "SELECT * FROM read_parquet(?) LIMIT 0", [str(screening)]
                ).description}
                required = {
                    "endpoint_id", "feature_id", "analysis_type", "evaluation_split",
                    "subgroup_variable", "subgroup_level", "roc_auc", "n_positive",
                    "n_negative", "inference_status", "subgroup_time_scope",
                }
                if required.issubset(schema):
                    ci_projection = ", ".join(
                        f'"{name}"' if name in schema else f'NULL AS "{name}"'
                        for name in ("roc_auc_ci_low", "roc_auc_ci_high")
                    )
                    subgroups = connection.execute(
                        "SELECT subgroup_variable, subgroup_level, subgroup_time_scope, "
                        f"roc_auc, n_positive, n_negative, inference_status, {ci_projection} "
                        "FROM read_parquet(?) WHERE endpoint_id = ? AND feature_id = ? "
                        "AND analysis_type = 'subgroup_stability' "
                        "AND evaluation_split = 'validation' "
                        "ORDER BY subgroup_variable, subgroup_level LIMIT 100",
                        [str(screening), endpoint_id, marker],
                    ).fetchdf()
            finally:
                connection.close()
    segment_rows, segment_note = _load_segment_availability(page, endpoint_id)
    return AvailabilitySnapshot(endpoint_id, marker, eligible, raw, observed,
                                restricted, subgroups, note, reasons, page.run_id,
                                page.snapshot_hash, segment_rows, segment_note)


def _load_segment_availability(page: ResultPage, endpoint_id: str) -> tuple[pd.DataFrame, str]:
    empty = pd.DataFrame()
    if page.artifact_path is None:
        return empty, "Kein kompatibler Segmentstand."
    root = page.artifact_path.parent / "availability"
    path = root / "segment_availability.parquet"
    try:
        manifest = StageManifest.load(root / "segment_availability.manifest.json")
    except (OSError, TypeError, ValueError):
        return empty, "Segmentbeurteilbarkeit ist noch nicht abgeschlossen."
    resolved = str(path.resolve())
    if (manifest.status != "complete" or manifest.stage != "segment-availability"
            or manifest.stage_version != SEGMENT_AVAILABILITY_VERSION
            or resolved not in {str(Path(item).resolve()) for item in manifest.output_paths}
            or not path.is_file()):
        return empty, "Segmentbeurteilbarkeit ist unvollständig oder versionsfremd."
    records = [record for record in manifest.output_integrity
               if str(Path(str(record.get("path", ""))).resolve()) == resolved]
    if manifest.output_integrity and (
            len(records) != 1 or not verify_output_integrity(records, mode="sha256")):
        return empty, "Segmentbeurteilbarkeit hat eine beschädigte oder ungeprüfte Datei."
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='256MB'")
        connection.execute("SET threads=1")
        schema = {item[0] for item in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)],
        ).description}
        needed = {"endpoint_id", "segment", "sentinel_marker", "n_positive",
                  "n_negative", "available_positive_pct", "available_negative_pct",
                  "gap_points", "differential"}
        if not needed.issubset(schema):
            return empty, "Segmenttabelle hat keinen kompatiblen Ergebnisvertrag."
        rows = connection.execute(
            "SELECT segment, sentinel_marker, n_positive, n_negative, "
            "available_positive_pct, available_negative_pct, gap_points, differential "
            "FROM read_parquet(?) WHERE endpoint_id = ? ORDER BY segment LIMIT 20",
            [str(path), endpoint_id],
        ).fetchdf()
    except duckdb.Error:
        return empty, "Segmenttabelle ist nicht lesbar."
    finally:
        connection.close()
    if rows.empty:
        return empty, "Für diesen Endpunkt liegen keine ausreichenden Fall-/Kontrollgruppen vor."
    integrity = "SHA-256-geprüft" if records else "älteres Manifest ohne Dateihash"
    return rows, (
        "Segment-Sentinelverfügbarkeit bei Fällen und Kontrollen; Development Train+Validation "
        f"gemeinsam, nicht Validation-spezifisch. {integrity}. "
        "Differenz = Fälle minus Kontrollen in Prozentpunkten; keine Marker-spezifische Ursache."
    )


class AtlasAvailabilityWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: AvailabilitySnapshot | None = None
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        marker_page = QWidget()
        layout = QVBoxLayout(marker_page)
        self.tabs.addTab(marker_page, "Marker & Untergruppen")
        self.note = QLabel("Messverfügbarkeit und Untergruppen werden geprüft.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.funnel_plot = pg.PlotWidget()
        self.funnel_plot.setMinimumHeight(180)
        self.funnel_plot.setLabel("left", "Referenzgeeignete Index-EKGs")
        layout.addWidget(self.funnel_plot)
        self.reason_table = QTableWidget(0, 4)
        self.reason_table.setHorizontalHeaderLabels((
            "Teilmenge", "EKGs", "Anteil Referenzgeeignete", "Bedeutung",
        ))
        self.reason_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.reason_table.setMinimumHeight(135)
        layout.addWidget(self.reason_table)
        self.subgroup_note = QLabel("Untergruppen: noch nicht geladen.")
        self.subgroup_note.setWordWrap(True)
        layout.addWidget(self.subgroup_note)
        self.subgroup_plot = pg.PlotWidget()
        self.subgroup_plot.setMinimumHeight(180)
        self.subgroup_plot.setLabel("bottom", "Validation-AUC")
        self.subgroup_plot.setXRange(0, 1, padding=0)
        layout.addWidget(self.subgroup_plot)
        self.subgroup_table = QTableWidget(0, 6)
        self.subgroup_table.setHorizontalHeaderLabels((
            "Untergruppe", "Zeitbezug", "AUC", "95-%-CI", "Fälle / Kontrollen", "Schätzstatus",
        ))
        self.subgroup_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.subgroup_table)
        actions = QHBoxLayout()
        plot_export = QPushButton("Verfügbarkeit als PNG/SVG")
        plot_export.setToolTip("Exportiert nur einen konsistenten Trichter mit separater Herkunftsdatei.")
        plot_export.clicked.connect(self._export_plot)
        actions.addWidget(plot_export)
        reasons_export = QPushButton("Verfügbarkeitsgruppen als CSV")
        reasons_export.setToolTip(
            "Exportiert die aus belegten Zählern abgeleiteten Gruppen; technische Einzelursachen bleiben unbekannt."
        )
        reasons_export.clicked.connect(self._export_reasons)
        actions.addWidget(reasons_export)
        subgroup_plot_export = QPushButton("Untergruppen-Forest als PNG/SVG")
        subgroup_plot_export.setToolTip(
            "Exportiert nur tatsächlich berechnete Subgruppen-AUC-Konfidenzintervalle mit Herkunftsdatei."
        )
        subgroup_plot_export.clicked.connect(self._export_subgroup_plot)
        actions.addWidget(subgroup_plot_export)
        table_export = QPushButton("Untergruppen als CSV")
        table_export.setToolTip("Exportiert die tatsächlich geladenen, deskriptiven Untergruppenwerte.")
        table_export.clicked.connect(self._export_table)
        actions.addWidget(table_export)
        actions.addStretch(1)
        layout.addLayout(actions)
        segment_page = QWidget()
        segment_layout = QVBoxLayout(segment_page)
        self.tabs.addTab(segment_page, "Segment nach Fallstatus")
        self.segment_note = QLabel("Segmentbeurteilbarkeit wird geprüft.")
        self.segment_note.setWordWrap(True)
        segment_layout.addWidget(self.segment_note)
        self.segment_plot = pg.PlotWidget()
        self.segment_plot.setMinimumHeight(220)
        self.segment_plot.setLabel("left", "Sentinel messbar", units="%")
        self.segment_plot.setYRange(0, 100, padding=0)
        segment_layout.addWidget(self.segment_plot)
        self.segment_table = QTableWidget(0, 5)
        self.segment_table.setHorizontalHeaderLabels((
            "Segment", "Fälle / Kontrollen", "bei Fällen messbar",
            "bei Kontrollen messbar", "Differenz (pp)",
        ))
        self.segment_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        segment_layout.addWidget(self.segment_table)
        segment_export = QPushButton("Segmentvergleich als CSV")
        segment_export.setToolTip(
            "Exportiert nur aggregierte Development-Sentinelverfügbarkeit, keine Marker-Ursachen."
        )
        segment_export.clicked.connect(self._export_segment)
        segment_layout.addWidget(segment_export)

    def set_snapshot(self, snapshot: AvailabilitySnapshot) -> None:
        self._snapshot = snapshot
        self._show_segments(snapshot)
        self.funnel_plot.clear()
        self.subgroup_plot.clear()
        self.subgroup_plot.hide()
        self.subgroup_table.setRowCount(0)
        reasons = snapshot.missing_reasons
        self.reason_table.setRowCount(0 if reasons is None else len(reasons))
        if reasons is not None:
            for index, (_, reason) in enumerate(reasons.iterrows()):
                share = reason.get("share_of_eligible_pct")
                values = (reason["category"], reason["count"],
                          f"{share:.1f} %" if pd.notna(share) else "nicht definiert",
                          reason["interpretation"])
                for column, value in enumerate(values):
                    self.reason_table.setItem(index, column, QTableWidgetItem(str(value)))
        self.note.setText(snapshot.note)
        counts = (snapshot.reference_eligible, snapshot.raw_observed,
                  snapshot.analysis_observed)
        if all(value is not None for value in counts):
            for x, count in enumerate(counts):
                self.funnel_plot.addItem(pg.BarGraphItem(
                    x=[x], height=[count], width=.65,
                    brush=pg.mkBrush((35, 145, 155) if x == 2 else (100, 145, 165)),
                ))
            self.funnel_plot.getAxis("bottom").setTicks([[
                (0, "Referenz geeignet"), (1, "Marker roh messbar"),
                (2, "Analysegeeignet"),
            ]])
            share = f"{100 * counts[2] / counts[0]:.1f} %" if counts[0] else "nicht definiert"
            restricted = (snapshot.restriction_excluded
                          if snapshot.restriction_excluded is not None else "unbekannt")
            self.funnel_plot.setTitle(
                f"{counts[0]} → {counts[1]} → {counts[2]} EKGs; "
                f"analysegeeignet: {share}; Restriktion: {restricted}"
            )
        else:
            self.funnel_plot.setTitle("Verfügbarkeitstrichter nicht berechenbar")
        frame = snapshot.subgroups
        if frame.empty:
            self.subgroup_note.setText(
                "Keine manifestierten Subgruppenergebnisse für diesen Marker. "
                "Ein fehlender Eintrag ist kein stabiler Null-Effekt."
            )
            return
        def finite(value: object) -> float | None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None
        intervals = 0
        ticks = []
        self.subgroup_table.setRowCount(len(frame))
        for index, (_, row) in enumerate(frame.iterrows()):
            auc = finite(row.get("roc_auc"))
            low = finite(row.get("roc_auc_ci_low"))
            high = finite(row.get("roc_auc_ci_high"))
            valid_ci = (auc is not None and low is not None and high is not None
                        and 0 <= low <= auc <= high <= 1)
            if valid_ci:
                y = float(len(frame) - index)
                self.subgroup_plot.plot([low, high], [y, y], pen=pg.mkPen("#298f9d", width=2))
                self.subgroup_plot.plot([auc], [y], pen=None, symbol="o",
                                        symbolSize=8, symbolBrush="#298f9d")
                ticks.append((y, f"{row['subgroup_variable']}: {row['subgroup_level']}"[:35]))
                intervals += 1
            values = (
                f"{row['subgroup_variable']}: {row['subgroup_level']}",
                str(row.get("subgroup_time_scope") or "unbekannt"),
                f"{auc:.3f}" if auc is not None else "nicht schätzbar",
                f"{low:.3f}–{high:.3f}" if valid_ci else "nicht berechnet",
                f"{row.get('n_positive', '?')} / {row.get('n_negative', '?')}",
                str(row.get("inference_status") or "unbekannt"),
            )
            for column, value in enumerate(values):
                self.subgroup_table.setItem(index, column, QTableWidgetItem(value))
        self.subgroup_plot.getAxis("left").setTicks([ticks])
        self.subgroup_plot.setYRange(.5, max(1.5, len(frame) + .5), padding=0)
        self.subgroup_plot.setVisible(intervals > 0)
        self.subgroup_note.setText(
            f"Deskriptive Validation-AUC nach Untergruppe; {intervals} von {len(frame)} "
            "mit echtem 95-%-CI. Ohne Interaktionstest ist ein Gruppenunterschied "
            "nicht nachgewiesen. Fehlende CI werden nicht geschätzt."
        )

    def _show_segments(self, snapshot: AvailabilitySnapshot) -> None:
        self.segment_note.setText(snapshot.segment_note)
        self.segment_plot.clear()
        rows = snapshot.segment_rows
        self.segment_table.setRowCount(len(rows))
        ticks = []
        for index, (_, row) in enumerate(rows.iterrows()):
            positive = float(row.available_positive_pct)
            negative = float(row.available_negative_pct)
            if not (math.isfinite(positive) and math.isfinite(negative)
                    and 0 <= positive <= 100 and 0 <= negative <= 100):
                continue
            x = float(index)
            self.segment_plot.addItem(pg.BarGraphItem(
                x=[x - .19], height=[positive], width=.34,
                brush=pg.mkBrush("#d95f0e"),
            ))
            self.segment_plot.addItem(pg.BarGraphItem(
                x=[x + .19], height=[negative], width=.34,
                brush=pg.mkBrush("#3182bd"),
            ))
            ticks.append((x, str(row.segment).upper()))
            values = (
                str(row.segment).upper(), f"{int(row.n_positive)} / {int(row.n_negative)}",
                f"{positive:.1f} %", f"{negative:.1f} %",
                f"{float(row.gap_points):+.1f}",
            )
            for column, value in enumerate(values):
                self.segment_table.setItem(index, column, QTableWidgetItem(value))
        self.segment_plot.getAxis("bottom").setTicks([ticks])
        if not rows.empty:
            self.segment_note.setText(
                snapshot.segment_note + " Orange = Fälle; Blau = Kontrollen. "
                "Eine große Differenz ist ein Bias-Warnsignal, kein Signifikanztest."
            )

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "snapshot_hash": snapshot.snapshot_hash if snapshot else None,
            "endpoint_id": snapshot.endpoint_id if snapshot else None,
            "feature_id": snapshot.marker_id if snapshot else None,
            "split": "validation",
            "note": "Index-EKG-Verfügbarkeit und deskriptive Untergruppen; fehlende Ursachen nicht erfunden.",
        }

    def _export_plot(self, *_args) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.reference_eligible is None:
            return
        choose_plot_export(
            self, self.funnel_plot, self._metadata(),
            proposed_name="marker_verfuegbarkeit.png",
        )

    def _export_table(self, *_args) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.subgroups.empty:
            return
        choose_table_export(
            self, snapshot.subgroups, self._metadata(),
            proposed_name="marker_untergruppen.csv",
        )

    def _export_reasons(self, *_args) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.missing_reasons is None or snapshot.missing_reasons.empty:
            return
        choose_table_export(
            self, snapshot.missing_reasons, self._metadata(),
            proposed_name="marker_verfuegbarkeitsgruppen.csv",
        )

    def _export_subgroup_plot(self, *_args) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.subgroups.empty or self.subgroup_plot.isHidden():
            return
        choose_plot_export(
            self, self.subgroup_plot, self._metadata(),
            proposed_name="marker_untergruppen_forest.png",
        )

    def _export_segment(self, *_args) -> None:
        snapshot = self._snapshot
        if snapshot is not None and not snapshot.segment_rows.empty:
            metadata = {**self._metadata(), "scope": "train_and_validation_combined",
                        "integrity_note": snapshot.segment_note}
            choose_table_export(self, snapshot.segment_rows, metadata,
                                proposed_name="segment_fall_kontroll_verfuegbarkeit.csv")
