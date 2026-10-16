"""A careful overview of broad ECG/VCG association discovery."""
from __future__ import annotations

from dataclasses import dataclass
import math

import duckdb
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...mimic.models import ResultQuery
from ...mimic.result_query import ResultRepository
from .export import choose_plot_export, choose_table_export
from .row_scope import compatible_validation_clauses


@dataclass(frozen=True)
class DiscoverySnapshot:
    status: str
    message: str
    rows: pd.DataFrame
    total_count: int
    domains: pd.DataFrame
    run_id: str
    snapshot_hash: str


def load_discovery_snapshot(repository: ResultRepository, run_id: str) -> DiscoverySnapshot:
    descriptor = next(
        (item for item in repository.available_runs() if item.run_id == run_id),
        None,
    )
    if descriptor is None or not descriptor.complete_compatible:
        return DiscoverySnapshot(
            "pending", "Die Validation ist noch nicht als kompatibler vollständiger Lauf verfügbar.",
            pd.DataFrame(), 0, pd.DataFrame(), run_id, "",
        )
    if descriptor.phenome_status != "complete" or not descriptor.phenome_compatible:
        return DiscoverySnapshot(
            "pending", "Der breite Scan ist für diesen Lauf noch nicht abgeschlossen oder nicht kompatibel.",
            pd.DataFrame(), 0, pd.DataFrame(), run_id, "",
        )
    page = repository.query(ResultQuery(
        perspective="discovery", run_id=run_id,
        complete_runs_only=True, validation_only=True,
        sort_by="evidence_rank", limit=75,
    ))
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return DiscoverySnapshot(
            page.status, page.message_de or "Keine manifestierten Discovery-Ergebnisse vorhanden.",
            pd.DataFrame(), 0, pd.DataFrame(), run_id, page.snapshot_hash,
        )
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        path = str(page.artifact_path)
        schema = {item[0] for item in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [path],
        ).description}
        if not {"discovery_domain", "adjustment_robust", "unadjusted_replicated"}.issubset(schema):
            domains = pd.DataFrame()
        else:
            extra = compatible_validation_clauses(schema)
            where = " AND ".join(extra)
            domains = connection.execute(
                "SELECT COALESCE(CAST(discovery_domain AS VARCHAR), 'Andere') AS domain, "
                "COUNT(*) AS hypothesis_count FROM read_parquet(?) "
                "WHERE COALESCE(TRY_CAST(adjustment_robust AS BOOLEAN), false) "
                "AND COALESCE(TRY_CAST(unadjusted_replicated AS BOOLEAN), false) "
                + (f"AND {where} " if where else "") +
                "GROUP BY domain ORDER BY hypothesis_count DESC, domain LIMIT 12",
                [path],
            ).fetchdf()
        return DiscoverySnapshot(
            page.status, page.message_de, page.rows, page.total_count,
            domains, run_id, page.snapshot_hash,
        )
    finally:
        connection.close()


def _display(value: object, fallback: str = "nicht angegeben") -> str:
    if value is None or pd.isna(value):
        return fallback
    result = str(value).strip()
    return result if result else fallback


def _q_label(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "nicht berechnet"
    return "< 0,001" if number < .001 else f"{number:.3f}"


_GROUP_LABELS = {
    "pathology_proxy": ("Pathologie-Proxy", "signierter Diskriminationseffekt 2×AUC−1"),
    "context_binary": ("Binärer Kontext", "AUC−0,5; Nullpunkt = keine Trennung"),
    "context_continuous": ("Kontinuierlicher Kontext", "Spearman-ρ"),
    "future_outcome": ("Späteres Ereignis", "log(Hazard Ratio) pro SD"),
}


def hypothesis_group(row: pd.Series) -> str:
    discovery_class = str(row.get("discovery_class", ""))
    if discovery_class == "pathology_hypothesis":
        return "pathology_proxy"
    if discovery_class == "future_outcome_hypothesis":
        return "future_outcome"
    if discovery_class == "non_pathology_association":
        kind = str(row.get("target_kind", ""))
        return "context_binary" if kind == "binary" else (
            "context_continuous" if kind == "continuous" else ""
        )
    return ""


def hypothesis_scatter_frame(rows: pd.DataFrame, group: str) -> pd.DataFrame:
    """Use only a single effect scale and disclose its q-value family."""
    points = []
    for _, row in rows.iterrows():
        if hypothesis_group(row) != group:
            continue
        effect = pd.to_numeric(row.get("association_effect"), errors="coerce")
        if pd.isna(effect) or not math.isfinite(float(effect)):
            continue
        x = float(effect) - (.5 if group == "context_binary" else 0.)
        if group in {"pathology_proxy", "context_continuous"} and not -1 <= x <= 1:
            continue
        if group == "context_binary" and not -.5 <= x <= .5:
            continue
        validation = pd.to_numeric(row.get("validation_q_value"), errors="coerce")
        if pd.isna(validation) or not 0 <= validation <= 1:
            continue
        q, source = float(validation), "Validation-q; familiengetrennt nach Zieltyp"
        points.append({
            "target_id": row.get("target_id"), "target_name": row.get("target_name"),
            "feature_id": row.get("feature_id"),
            "effect_x": x, "q_value": q,
            "minus_log10_q_capped": min(20., -math.log10(max(q, 1e-300))),
            "q_source": source,
            "adjustment_robust": str(row.get("adjustment_robust", "")).lower() == "true",
        })
    return pd.DataFrame(points)


class AtlasDiscoveryWidget(QWidget):
    """Separate association classes; no pooled volcano across effect scales."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._snapshot: DiscoverySnapshot | None = None
        layout = QVBoxLayout(self)
        title = QLabel("Breiter Scan: Welche Zusammenhänge verdienen Nachprüfung?")
        title.setStyleSheet("font-weight: 650; font-size: 16px;")
        layout.addWidget(title)
        self.note = QLabel(
            "Pathologie-, Prognose- und Kontextassoziationen bleiben getrennte Hypothesen. "
            "Ein Diagnosecode oder eine Expertenannotation ersetzt keinen unabhängigen Krankheitsnachweis."
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.domain_plot = pg.PlotWidget()
        self.domain_plot.setMinimumHeight(220)
        self.domain_plot.setLabel("left", "Adjustiert robuste, replizierte Paare")
        layout.addWidget(self.domain_plot)
        self.domain_plot.hide()
        self.chart_note = QLabel("Keine adjustiert robusten, replizierten Gruppen für diesen Lauf berechnet.")
        self.chart_note.setWordWrap(True)
        layout.addWidget(self.chart_note)
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Hypothesenkarte"))
        self.group = QComboBox()
        self.group.currentIndexChanged.connect(self._draw_hypotheses)
        group_row.addWidget(self.group, 1)
        layout.addLayout(group_row)
        self.hypothesis_note = QLabel(
            "Effektmaße mit unterschiedlichen Skalen werden nicht in einem Diagramm vermischt."
        )
        self.hypothesis_note.setWordWrap(True)
        layout.addWidget(self.hypothesis_note)
        self.hypothesis_plot = pg.PlotWidget()
        self.hypothesis_plot.setMinimumHeight(270)
        self.hypothesis_plot.setLabel("left", "−log10(q), bei 20 gedeckelt")
        layout.addWidget(self.hypothesis_plot)
        self.hypothesis_plot.hide()
        self._hypothesis_points = pd.DataFrame()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Klinisches Ziel, Marker oder Bereich suchen")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_rows)
        layout.addWidget(self.search)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels((
            "Ziel", "Marker", "Bereich", "Prüfstatus", "Adjustiertes q", "Fälle / Kontrollen",
        ))
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)
        export_row = QHBoxLayout()
        chart_export = QPushButton("Bereichsdiagramm als PNG/SVG")
        chart_export.setToolTip("Exportiert die Zählung adjustiert robuster, replizierter Hypothesen.")
        chart_export.clicked.connect(self._export_chart)
        export_row.addWidget(chart_export)
        table_export = QPushButton("Discovery-Auswahl als CSV")
        table_export.setToolTip("Exportiert höchstens 75 geladenen Hypothesen samt Herkunft, unabhängig vom Suchfilter.")
        table_export.clicked.connect(self._export_table)
        export_row.addWidget(table_export)
        scatter_export = QPushButton("Hypothesenkarte als PNG/SVG")
        scatter_export.setToolTip("Exportiert nur die ausgewählte, skalengetrennte Hypothesenkarte mit Herkunftsdatei.")
        scatter_export.clicked.connect(self._export_hypothesis_plot)
        export_row.addWidget(scatter_export)
        points_export = QPushButton("Kartendaten als CSV")
        points_export.setToolTip("Exportiert die tatsächlich gezeichneten, begrenzten Hypothesenpunkte mit Herkunftsdatei.")
        points_export.clicked.connect(self._export_hypothesis_points)
        export_row.addWidget(points_export)
        export_row.addStretch(1)
        layout.addLayout(export_row)

    def set_snapshot(self, snapshot: DiscoverySnapshot) -> None:
        self._snapshot = snapshot
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.domain_plot.clear()
        self.hypothesis_plot.clear()
        self.hypothesis_plot.hide()
        self._hypothesis_points = pd.DataFrame()
        self.group.blockSignals(True)
        self.group.clear()
        if snapshot.status not in {"ok", "empty"}:
            self.note.setText(snapshot.message)
            self.domain_plot.hide()
            self.chart_note.setText("Kein Diagramm ohne abgeschlossene, kompatible Discovery-Stufe.")
            self.group.blockSignals(False)
            self.table.setSortingEnabled(True)
            return
        self.note.setText(
            f"{snapshot.total_count:,} Hypothesen im manifestierten breiten Scan; "
            "höchstens 75 nach Evidenz sortiert geladen. Die Tabelle zeigt verschiedene "
            "Fragestellungen, nicht einheitliche Diagnosegüte."
        )
        if snapshot.domains.empty:
            self.domain_plot.hide()
            self.chart_note.setText("Keine adjustiert robusten und replizierten Paare mit Bereichsangabe vorhanden.")
        else:
            counts = snapshot.domains["hypothesis_count"].astype(float).tolist()
            labels = snapshot.domains["domain"].astype(str).tolist()
            self.domain_plot.addItem(pg.BarGraphItem(
                x=list(range(len(counts))), height=counts, width=.68,
                brush="#29a3a3",
            ))
            self.domain_plot.getAxis("bottom").setTicks([[
                (index, label.replace("_", " ")[:17])
                for index, label in enumerate(labels)
            ]])
            self.domain_plot.setXRange(-.6, len(counts) - .4, padding=0)
            self.domain_plot.show()
            self.chart_note.setText(
                "Balkenhöhe = Anzahl Marker–Ziel-Paare, nicht Patienten oder unabhängige Entdeckungen."
            )
        rows = snapshot.rows.head(75)
        for key, (title, _scale) in _GROUP_LABELS.items():
            if any(hypothesis_group(row) == key for _, row in rows.iterrows()):
                self.group.addItem(title, key)
        self.group.blockSignals(False)
        self._draw_hypotheses()
        self.table.setRowCount(len(rows))
        for index, (_, row) in enumerate(rows.iterrows()):
            positive = pd.to_numeric(row.get("n_positive"), errors="coerce")
            negative = pd.to_numeric(row.get("n_negative"), errors="coerce")
            n_text = (
                f"{int(positive)} / {int(negative)}"
                if pd.notna(positive) and pd.notna(negative)
                else "nicht angegeben"
            )
            values = (
                _display(row.get("target_name"), _display(row.get("target_id"))),
                _display(row.get("feature_name_de"), _display(row.get("feature_id"))),
                _display(row.get("scope_display_de"), _display(row.get("discovery_domain"))),
                _display(row.get("status_de"), _display(row.get("status_id"))),
                _q_label(row.get("adjusted_q_value")), n_text,
            )
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(value))
        self.table.setSortingEnabled(True)
        self._filter_rows()

    def _draw_hypotheses(self, *_args) -> None:
        self.hypothesis_plot.clear()
        snapshot = self._snapshot
        group = str(self.group.currentData() or "")
        if snapshot is None or not group:
            self._hypothesis_points = pd.DataFrame()
            self.hypothesis_plot.hide()
            self.hypothesis_note.setText(
                "Keine Hypothesen mit sicher zuordenbarem Effektmaß und q-Wert geladen."
            )
            return
        points = hypothesis_scatter_frame(snapshot.rows.head(75), group)
        self._hypothesis_points = points
        if points.empty:
            self.hypothesis_plot.hide()
            self.hypothesis_note.setText("Für diese Effektklasse fehlen darstellbare Effekt-/q-Paare.")
            return
        title, scale = _GROUP_LABELS[group]
        self.hypothesis_plot.setLabel("bottom", scale)
        self.hypothesis_plot.addItem(pg.InfiniteLine(pos=0, angle=90,
                                                   pen=pg.mkPen("#89939b", width=1)))
        for robust, color in ((False, "#8a9da5"), (True, "#198f91")):
            subset = points.loc[points["adjustment_robust"].eq(robust)]
            if not subset.empty:
                self.hypothesis_plot.plot(
                    subset["effect_x"].tolist(), subset["minus_log10_q_capped"].tolist(),
                    pen=None, symbol="o", symbolSize=7, symbolBrush=color,
                )
        self.hypothesis_plot.setVisible(True)
        self.hypothesis_note.setText(
            f"{title}: {len(points)} der höchstens 75 nach Evidenz vorgewählten Hypothesen "
            "mit darstellbaren Werten. Keine vollständige Volcano- oder QQ-Analyse aller Tests. "
            "Höhen über 20 werden bei 20 abgeschnitten; FDR-Familie steht im CSV. "
            "Türkis = adjustiert robust laut Ergebnisvertrag, grau = nicht so bestätigt."
        )

    def _filter_rows(self, *_args) -> None:
        search = self.search.text().strip().casefold()
        for row in range(self.table.rowCount()):
            haystack = " ".join(
                self.table.item(row, column).text()
                for column in range(4)
                if self.table.item(row, column) is not None
            ).casefold()
            self.table.setRowHidden(row, bool(search and search not in haystack))

    def _metadata(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "run_id": snapshot.run_id if snapshot else None,
            "snapshot_hash": snapshot.snapshot_hash if snapshot else None,
            "split": "validation", "view": "broad_discovery",
            "note": "Hypothesengenerierender breiter Scan; keine automatische Diagnose oder klinische Freigabe.",
        }

    def _export_chart(self, *_args) -> None:
        if self._snapshot is None or self._snapshot.domains.empty:
            return
        choose_plot_export(
            self, self.domain_plot, self._metadata(),
            proposed_name="discovery_bereiche.png",
        )

    def _export_table(self, *_args) -> None:
        if self._snapshot is None or self._snapshot.rows.empty:
            return
        choose_table_export(
            self, self._snapshot.rows.head(75), self._metadata(),
            proposed_name="discovery_hypothesen.csv",
        )

    def _export_hypothesis_plot(self, *_args) -> None:
        if not self._hypothesis_points.empty:
            choose_plot_export(self, self.hypothesis_plot,
                               {**self._metadata(), "effect_group": self.group.currentData()},
                               proposed_name="discovery_hypothesenkarte.png")

    def _export_hypothesis_points(self, *_args) -> None:
        if not self._hypothesis_points.empty:
            choose_table_export(self, self._hypothesis_points,
                                {**self._metadata(), "effect_group": self.group.currentData()},
                                proposed_name="discovery_hypothesenkarte.csv")
