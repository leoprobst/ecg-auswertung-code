"""Classical ECG rules and machine reports as separate quality controls."""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...mimic.result_query import ResultPage
from ...mimic.control_panel import CONTROL_PANEL_VERSION
from ...mimic.storage import StageManifest
from .export import choose_plot_export, choose_table_export
from .map_view import AtlasBaselineWidget
from .row_scope import validation_output_assurance


@dataclass(frozen=True)
class RuleAuditSnapshot:
    classical: pd.DataFrame
    machine: pd.DataFrame
    status: str
    note: str
    run_id: str = ""
    snapshot_hash: str = ""
    registry: pd.DataFrame = field(default_factory=pd.DataFrame)
    controls: pd.DataFrame = field(default_factory=pd.DataFrame)
    control_verdict: dict = field(default_factory=dict)


_COLUMNS = (
    "endpoint_id", "baseline_id", "evaluation_split", "baseline_type",
    "n_total", "n_positive", "n_negative", "sensitivity", "specificity",
    "sensitivity_ci_low", "sensitivity_ci_high", "specificity_ci_low",
    "specificity_ci_high", "inference_status", "threshold_source",
    "positive_count", "negative_count", "uncertain_count",
    "not_mentioned_count", "unavailable_count", "uncertain_excluded",
)
_REQUIRED = {"endpoint_id", "baseline_id", "evaluation_split", "sensitivity",
             "specificity", "n_positive", "n_negative", "inference_status"}
_CONTROL_NAMES = {
    "hyperkalemia_peaked_t": "Hyperkaliämie – spitze T-Welle",
    "hyperkalemia_qrs_widening": "Hyperkaliämie – QRS-Verbreiterung",
    "hypokalemia_tpeak_tend": "Hypokaliämie – T-Peak–T-Ende",
    "hypocalcemia_qt": "Hypokalzämie – QT-Verlängerung",
    "qt_prolonging_drug_qt": "QT-verlängerndes Medikament – QT",
    "digoxin_st_depression": "Digoxin – ST-Senkung",
    "hypocalcemia_p_width": "Hypokalzämie – P-Loop-Breite",
    "digoxin_p_duration": "Digoxin – P-Dauer",
    "hypomagnesemia_qrs_axis": "Hypomagnesiämie – QRS-Loop-Fläche",
    "qt_drug_p_width": "QT-verlängerndes Medikament – P-Loop-Breite",
}
_CONTROL_STATUSES = {
    "recovered": "erwartete Richtung wiedergefunden",
    "wrong_direction": "Gegenrichtung",
    "missed": "keine Richtung erkennbar",
    "direction_only": "Richtung ohne Signifikanz",
    "no_uncertainty_reported": "Unsicherheit fehlt",
    "silent_as_expected": "unauffällig wie erwartet",
    "false_positive": "auffällige Negativkontrolle",
    "not_tested": "nicht prüfbar",
}


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count(value: object) -> str:
    number = _finite(value)
    return str(int(number)) if number is not None and number >= 0 and number.is_integer() else "?"


def _metric(row: pd.Series, name: str) -> str:
    value = _finite(row.get(name))
    low = _finite(row.get(f"{name}_ci_low"))
    high = _finite(row.get(f"{name}_ci_high"))
    if value is None:
        return "nicht schätzbar"
    if low is not None and high is not None and 0 <= low <= value <= high <= 1:
        return f"{value:.3f} ({low:.3f}–{high:.3f})"
    return f"{value:.3f} (CI nicht berechnet)"


def _inference_label(value: object) -> str:
    return {
        "formal_inference": "Intervall-Inferenz vorhanden",
        "descriptive_only": "nur deskriptiv",
        "underpowered": "zu geringe Fallzahl",
        "non_estimable": "nicht schätzbar",
    }.get(str(value), "Schätzstatus unbekannt")


def _load_controls(run_root: Path, run_id: str) -> tuple[pd.DataFrame, dict]:
    root = run_root / "controls"
    table_path = root / "control_panel.parquet"
    verdict_path = root / "control_panel_verdict.json"
    try:
        manifest = StageManifest.load(root / "control_panel.manifest.json")
        if manifest.stage != "control-panel" or manifest.status != "complete" \
                or manifest.stage_version != CONTROL_PANEL_VERSION:
            return pd.DataFrame(), {}
        declared = {str(Path(path).resolve()) for path in manifest.output_paths}
        if not {str(table_path.resolve()), str(verdict_path.resolve())}.issubset(declared):
            return pd.DataFrame(), {}
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        if verdict.get("run_id") != run_id or verdict.get("verdict_split") != "validation":
            return pd.DataFrame(), {}
        table = pd.read_parquet(table_path)
        required = {"control_id", "kind", "evaluation_split", "status", "found",
                    "expected_direction", "observed_effect", "target_kind"}
        if not required.issubset(table.columns):
            return pd.DataFrame(), {}
        table = table.loc[table["evaluation_split"].astype(str).eq("validation")].copy()
        return table, verdict
    except (OSError, TypeError, ValueError, KeyError):
        return pd.DataFrame(), {}


def _read_table(path: Path) -> pd.DataFrame:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        schema = {column[0] for column in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)],
        ).description}
        if not _REQUIRED.issubset(schema):
            raise ValueError(f"{path.name}: required result columns missing")
        projected = ", ".join(
            f'"{column}"' if column in schema else f'NULL AS "{column}"'
            for column in _COLUMNS
        )
        return connection.execute(
            f"SELECT {projected} FROM read_parquet(?) "
            "WHERE evaluation_split = 'validation' "
            "ORDER BY endpoint_id, baseline_id LIMIT 5000", [str(path)],
        ).fetchdf()
    finally:
        connection.close()


def load_rule_audit(page: ResultPage) -> RuleAuditSnapshot:
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return RuleAuditSnapshot(pd.DataFrame(), pd.DataFrame(), "pending",
                                 "Noch kein kompatibler abgeschlossener Validation-Stand.")
    run_root = page.artifact_path.parent
    try:
        manifest = StageManifest.load(run_root / "validate.manifest.json")
    except (OSError, TypeError, ValueError):
        return RuleAuditSnapshot(pd.DataFrame(), pd.DataFrame(), "missing",
                                 "Validierungsmanifest fehlt oder ist unlesbar.",
                                 page.run_id, page.snapshot_hash)
    names = ("classical_baseline_results.parquet", "machine_report_baseline_results.parquet")
    frames = []
    missing = []
    for name in names:
        path = run_root / name
        if validation_output_assurance(manifest, path, page.run_id) is None:
            frames.append(pd.DataFrame())
            missing.append(name)
            continue
        try:
            frames.append(_read_table(path))
        except (duckdb.Error, OSError, ValueError):
            frames.append(pd.DataFrame())
            missing.append(name)
    registry_path = run_root / "baseline_registry.parquet"
    registry = pd.DataFrame()
    if validation_output_assurance(manifest, registry_path, page.run_id) is not None:
        try:
            registry = pd.read_parquet(registry_path).drop_duplicates("baseline_id")
        except (OSError, ValueError, KeyError):
            registry = pd.DataFrame()
    status = "partial" if missing else "ok"
    note = (
        "Klassische Regeln und Geräteberichte sind getrennte Vergleiche mit dem "
        "jeweiligen klinischen Endpunkt; sie können verschiedene auswertbare Fälle haben. "
        "Ein Literatur-Pass/Fail wird daraus nicht berechnet."
    )
    if missing:
        note += " Fehlend oder Integritätsprüfung fehlgeschlagen: " + ", ".join(missing) + "."
    if not manifest.output_integrity:
        note += " Validierungsdateien sind nur im abgeschlossenen Manifest deklariert; Ausgabe-SHA-256 fehlen."
    if registry.empty:
        note += " Lesbare Regelnamen aus dem Run-Register sind nicht verfügbar."
    controls, control_verdict = _load_controls(run_root, page.run_id)
    if not control_verdict:
        note += " Kontrolltafel ist noch nicht als passender Validation-Stand verfügbar."
    return RuleAuditSnapshot(frames[0], frames[1], status, note,
                             page.run_id, page.snapshot_hash, registry,
                             controls, control_verdict)


class AtlasRuleAuditWidget(QWidget):
    endpoint_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: RuleAuditSnapshot | None = None
        self._scoped_classical = pd.DataFrame()
        self._scoped_machine = pd.DataFrame()
        self._rule_names: dict[str, tuple[str, str]] = {}
        layout = QVBoxLayout(self)
        self.note = QLabel("Klassische Regeln und Geräteberichte werden geprüft …")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        row = QHBoxLayout()
        row.addWidget(QLabel("Klinischer Endpunkt"))
        self.endpoint = QComboBox()
        self.endpoint.currentIndexChanged.connect(self._draw)
        row.addWidget(self.endpoint, 1)
        layout.addLayout(row)
        self.tabs = QTabWidget()
        self.rules_page = QWidget()
        rules_layout = QVBoxLayout(self.rules_page)
        self.rules_note = QLabel(
            "Sensitivität und Spezifität beziehen sich auf den jeweiligen klinischen "
            "Validation-Endpunkt. Krankenhauskohorte, kein Leitlinien-Pass/Fail."
        )
        self.rules_note.setWordWrap(True)
        rules_layout.addWidget(self.rules_note)
        self.rule_search = QLineEdit()
        self.rule_search.setPlaceholderText("Regelname oder ID suchen")
        self.rule_search.setToolTip("Filtert die Regeln des gewählten klinischen Endpunkts.")
        self.rule_search.textChanged.connect(self._draw)
        rules_layout.addWidget(self.rule_search)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(250)
        self.plot.setLabel("bottom", "Spezifität")
        self.plot.setLabel("left", "Sensitivität")
        self.plot.setXRange(0, 1, padding=0)
        self.plot.setYRange(0, 1, padding=0)
        self.rules_table = QTableWidget(0, 5)
        self.rules_table.setHorizontalHeaderLabels((
            "Klassische Regel", "Sensitivität (95-%-CI)", "Spezifität (95-%-CI)",
            "Fälle / Kontrollen", "Schätzstatus",
        ))
        self.rules_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.rule_display = QTabWidget()
        self.rule_display.addTab(self.plot, "Diagramm")
        self.rule_display.addTab(self.rules_table, "Zahlen")
        rules_layout.addWidget(self.rule_display, 1)
        plot_export = QPushButton("Regelvergleich als PNG/SVG")
        plot_export.setToolTip("Exportiert die angezeigten Validation-Punkte mit Herkunftsdatei.")
        plot_export.clicked.connect(self._export_plot)
        rules_layout.addWidget(plot_export)
        self.tabs.addTab(self.rules_page, "Regeln gegen klinische Referenz")
        self.machine_page = QWidget()
        machine_layout = QVBoxLayout(self.machine_page)
        self.machine_note = QLabel(
            "Geräteberichte stammen aus demselben EKG. „Nicht erwähnt“ wird im "
            "Backend als nicht positiv gewertet; unklare Meldungen werden ausgeschlossen. "
            "Das ist kein unabhängiger Goldstandard."
        )
        self.machine_note.setWordWrap(True)
        machine_layout.addWidget(self.machine_note)
        self.machine_table = QTableWidget(0, 7)
        self.machine_table.setHorizontalHeaderLabels((
            "Gerätebefund", "Sensitivität", "Spezifität", "Fälle / Kontrollen",
            "Unklar", "Nicht erwähnt", "Nicht verfügbar",
        ))
        self.machine_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        machine_layout.addWidget(self.machine_table)
        self.tabs.addTab(self.machine_page, "Gerätebericht")
        self.control_page = QWidget()
        control_layout = QVBoxLayout(self.control_page)
        self.control_note = QLabel("Positiv- und Negativkontrollen werden für Validation geprüft.")
        self.control_note.setWordWrap(True)
        control_layout.addWidget(self.control_note)
        self.control_table = QTableWidget(0, 5)
        self.control_table.setHorizontalHeaderLabels((
            "Kontrolle", "Art", "Erwartung", "Beobachteter Effekt", "Einordnung",
        ))
        self.control_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        control_layout.addWidget(self.control_table)
        self.tabs.addTab(self.control_page, "Physiologische Kontrollen")
        self.definitions_page = QWidget()
        definitions_layout = QVBoxLayout(self.definitions_page)
        self.definitions_note = QLabel(
            "Registrierte klassische Messdefinitionen und Quellen. Ein Literatur-Pass/Fail "
            "ist nur mit vergleichbarer Population, Messdefinition und Endpunkt zulässig."
        )
        self.definitions_note.setWordWrap(True)
        definitions_layout.addWidget(self.definitions_note)
        self.definitions_table = QTableWidget(0, 5)
        self.definitions_table.setHorizontalHeaderLabels((
            "Regel", "Formel / Messdefinition", "Quelle", "Literaturmetriken", "Ausschlüsse",
        ))
        self.definitions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.definitions_table.setAlternatingRowColors(True)
        definitions_layout.addWidget(self.definitions_table)
        self.tabs.addTab(self.definitions_page, "Definitionen & Quellen")
        self.explorer = AtlasBaselineWidget()
        self.explorer.endpoint_requested.connect(self.endpoint_requested.emit)
        self.tabs.addTab(self.explorer, "Alle klassischen Ergebnisse")
        layout.addWidget(self.tabs)
        exports = QHBoxLayout()
        for title, callback, tooltip in (
            ("Regeln als CSV", self._export_rules, "Exportiert die angezeigten Regelwerte mit Herkunftsdatei."),
            ("Geräteberichte als CSV", self._export_machine,
             "Exportiert die angezeigten Geräteberichtwerte mit Herkunftsdatei."),
            ("Kontrollen als CSV", self._export_controls,
             "Exportiert die Validation-Kontrollen mit Herkunft und eingeschränkter Aussagegrenze."),
            ("Regeldefinitionen als CSV", self._export_definitions,
             "Exportiert das zum Run gehörende Regelregister; kein Literatur-Pass/Fail."),
        ):
            button = QPushButton(title)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            exports.addWidget(button)
        exports.addStretch(1)
        layout.addLayout(exports)

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.explorer.set_frame(frame)

    def set_provenance(self, run_id: str, snapshot_hash: str) -> None:
        self.explorer.set_provenance(run_id, snapshot_hash)

    def set_snapshot(self, snapshot: RuleAuditSnapshot) -> None:
        self._snapshot = snapshot
        self._rule_names = {
            str(row.baseline_id): (str(row.display_name), str(row.source_title))
            for row in snapshot.registry.itertuples()
        } if not snapshot.registry.empty else {}
        self._show_definitions(snapshot.registry)
        controls = snapshot.controls
        self.control_table.setRowCount(len(controls))
        for index, (_, row) in enumerate(controls.iterrows()):
            control_id = str(row.get("control_id") or "")
            effect = _finite(row.get("observed_effect"))
            kind = str(row.get("target_kind") or "")
            expectation = {1: "höher", -1: "niedriger", 0: "kein starker Effekt"}.get(
                int(row.get("expected_direction") or 0), "unbekannt",
            )
            values = (
                _CONTROL_NAMES.get(control_id, control_id),
                "Positivkontrolle" if row.get("kind") == "positive" else "Negativkontrolle",
                expectation,
                (f"{effect:.3f} {'AUC' if kind == 'binary' else 'Spearman-ρ'}"
                 if effect is not None else "nicht messbar"),
                _CONTROL_STATUSES.get(str(row.get("status")), "Status unbekannt"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setToolTip(f"ID: {control_id}\n{str(row.get('mechanism') or '')}")
                self.control_table.setItem(index, column, item)
        verdict = snapshot.control_verdict
        if verdict:
            self.control_note.setText(
                f"Validation: {int(verdict.get('n_positive_recovered', 0))} von "
                f"{int(verdict.get('n_positive_tested', 0))} prüfbaren Positivkontrollen "
                f"wiedergefunden; {int(verdict.get('n_negative_false_positive', 0))} von "
                f"{int(verdict.get('n_negative_tested', 0))} Negativkontrollen auffällig. "
                "Eine Negativkontrolle misst mögliche Störfaktoren, nicht die Abwesenheit "
                "einer physiologischen Wirkung. Das ältere Stage-Manifest enthält keine Dateihashes."
            )
        else:
            self.control_note.setText(
                "Keine vollständig zugeordnete Validation-Kontrolltafel für diesen Run vorhanden."
            )
        self.note.setText(snapshot.note)
        previous = self.endpoint.currentData()
        endpoints = sorted(set(snapshot.classical.get("endpoint_id", pd.Series(dtype=str)).dropna().astype(str))
                           | set(snapshot.machine.get("endpoint_id", pd.Series(dtype=str)).dropna().astype(str)))
        self.endpoint.blockSignals(True)
        self.endpoint.clear()
        for endpoint in endpoints:
            self.endpoint.addItem(endpoint, endpoint)
        if previous is not None:
            index = self.endpoint.findData(previous)
            if index >= 0:
                self.endpoint.setCurrentIndex(index)
        self.endpoint.blockSignals(False)
        self._draw()

    def _show_definitions(self, registry: pd.DataFrame) -> None:
        self.definitions_table.setRowCount(min(len(registry), 100))
        if registry.empty:
            self.definitions_note.setText(
                "Kein zum Run gehörendes Regelregister verfügbar; ein Literaturvergleich ist nicht berechenbar."
            )
            return
        self.definitions_note.setText(
            f"{len(registry)} registrierte Regeln, höchstens 100 angezeigt. "
            "Die Quelle belegt die Definition, nicht automatisch diagnostische Leistung "
            "in dieser Krankenhauskohorte. Ohne kompatible Literaturmetrik kein Pass/Fail."
        )
        for index, (_, row) in enumerate(registry.head(100).iterrows()):
            literature = row.get("literature_reference_metrics")
            has_literature = isinstance(literature, dict) and bool(literature)
            exclusions = row.get("exclusions")
            exclusions_text = (
                ", ".join(map(str, exclusions)) if isinstance(exclusions, (list, tuple)) and exclusions
                else "nicht hinterlegt"
            )
            values = (
                str(row.get("display_name") or row.get("baseline_id") or "unbekannt"),
                str(row.get("formula") or "nicht hinterlegt"),
                str(row.get("source_title") or "Quelle nicht hinterlegt"),
                "dokumentiert; Kohortenabgleich offen" if has_literature else
                "keine vergleichbare Referenzzahl hinterlegt",
                exclusions_text,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 2:
                    item.setToolTip(
                        f"Version: {row.get('source_version') or 'unbekannt'}\n"
                        f"URL: {row.get('source_url') or 'nicht hinterlegt'}"
                    )
                self.definitions_table.setItem(index, column, item)

    def _draw(self, *_args) -> None:
        snapshot = self._snapshot
        endpoint = str(self.endpoint.currentData() or "")
        self.plot.clear()
        self.rules_table.setRowCount(0)
        self.machine_table.setRowCount(0)
        if snapshot is None or not endpoint:
            return
        classical = snapshot.classical
        machine = snapshot.machine
        if not classical.empty:
            scoped = classical.loc[classical["endpoint_id"].astype(str).eq(endpoint)].copy()
            search = self.rule_search.text().strip().casefold()
            if search:
                scoped = scoped.loc[scoped["baseline_id"].astype(str).map(
                    lambda baseline_id: search in baseline_id.casefold() or search in
                    self._rule_names.get(baseline_id, ("", ""))[0].casefold()
                )]
            rule_count = len(scoped)
            self._scoped_classical = scoped.head(25).copy()
        else:
            rule_count = 0
            self._scoped_classical = pd.DataFrame()
        self._scoped_machine = machine.loc[machine["endpoint_id"].astype(str).eq(endpoint)].head(25).copy() \
            if not machine.empty else pd.DataFrame()
        self.rules_table.setRowCount(len(self._scoped_classical))
        for index, (_, row) in enumerate(self._scoped_classical.iterrows()):
            sensitivity = _finite(row.get("sensitivity"))
            specificity = _finite(row.get("specificity"))
            if sensitivity is not None and specificity is not None:
                self.plot.plot([specificity], [sensitivity], pen=None, symbol="o",
                               symbolSize=9, symbolBrush="#2a9297")
                sx = (_finite(row.get("specificity_ci_low")), _finite(row.get("specificity_ci_high")))
                sy = (_finite(row.get("sensitivity_ci_low")), _finite(row.get("sensitivity_ci_high")))
                if None not in sx and 0 <= sx[0] <= specificity <= sx[1] <= 1:
                    self.plot.plot(list(sx), [sensitivity, sensitivity], pen=pg.mkPen("#2a9297", width=2))
                if None not in sy and 0 <= sy[0] <= sensitivity <= sy[1] <= 1:
                    self.plot.plot([specificity, specificity], list(sy), pen=pg.mkPen("#2a9297", width=2))
            values = (
                self._rule_names.get(str(row.get("baseline_id")),
                                     (str(row.get("baseline_id") or "unbekannt"), ""))[0],
                _metric(row, "sensitivity"),
                _metric(row, "specificity"),
                f"{_count(row.get('n_positive'))} / {_count(row.get('n_negative'))}",
                _inference_label(row.get("inference_status")),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    baseline_id = str(row.get("baseline_id") or "")
                    source = self._rule_names.get(baseline_id, ("", ""))[1]
                    item.setToolTip(f"ID: {baseline_id}" + (f"\nQuelle: {source}" if source else ""))
                self.rules_table.setItem(index, column, item)
        self.machine_table.setRowCount(len(self._scoped_machine))
        for index, (_, row) in enumerate(self._scoped_machine.iterrows()):
            values = (
                str(row.get("baseline_id") or "unbekannt"),
                _metric(row, "sensitivity"), _metric(row, "specificity"),
                f"{_count(row.get('n_positive'))} / {_count(row.get('n_negative'))}",
                _count(row.get("uncertain_count")),
                _count(row.get("not_mentioned_count")),
                _count(row.get("unavailable_count")),
            )
            for column, value in enumerate(values):
                self.machine_table.setItem(index, column, QTableWidgetItem(value))
        self.rules_note.setText(
            f"{len(self._scoped_classical)} von {rule_count} passenden Regeln für {endpoint}; "
            "Punkte sind Sensitivität/Spezifität gegen die klinische Referenz. "
            "Intervalle erscheinen nur, wenn tatsächlich berechnet. "
            "Keine gepaarte Überlegenheitsprüfung zwischen Regeln oder Gerätebericht."
        )

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "snapshot_hash": snapshot.snapshot_hash if snapshot else None,
            "endpoint_id": self.endpoint.currentData(), "split": "validation",
            "view": "classical_and_device_quality_control",
            "note": "Deskriptiv; Krankenhauskohorte; keine Leitlinien-Pass/Fail-Bewertung.",
        }

    def _export_plot(self, *_args) -> None:
        if not self._scoped_classical.empty:
            choose_plot_export(self, self.plot, self._metadata(),
                               proposed_name="klassische_regeln_sensitivitaet_spezifitaet.png")

    def _export_rules(self, *_args) -> None:
        if not self._scoped_classical.empty:
            choose_table_export(self, self._scoped_classical, self._metadata(),
                                proposed_name="klassische_regeln_validation.csv")

    def _export_machine(self, *_args) -> None:
        if not self._scoped_machine.empty:
            choose_table_export(self, self._scoped_machine, self._metadata(),
                                proposed_name="geraetebericht_validation.csv")

    def _export_controls(self, *_args) -> None:
        if self._snapshot is not None and not self._snapshot.controls.empty:
            choose_table_export(self, self._snapshot.controls, self._metadata(),
                                proposed_name="physiologische_kontrollen_validation.csv")

    def _export_definitions(self, *_args) -> None:
        if self._snapshot is not None and not self._snapshot.registry.empty:
            choose_table_export(
                self, self._snapshot.registry,
                {**self._metadata(), "view": "run_classical_rule_definitions",
                 "claim": "definitions_and_sources_not_literature_concordance"},
                proposed_name="klassische_regeldefinitionen.csv",
            )
