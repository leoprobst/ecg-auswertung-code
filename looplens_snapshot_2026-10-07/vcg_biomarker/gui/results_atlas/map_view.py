"""Clinical-domain navigation map over sealed validation result artifacts."""
from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ...mimic.models import ResultQuery
from ...mimic.result_query import ResultPage, ResultRepository
from .charts import auc_matrix_rgba
from .export import choose_plot_export, choose_table_export
from .discovery_view import DiscoverySnapshot, load_discovery_snapshot
from .row_scope import compatible_validation_clauses
from .transport_view import TransportSnapshot, load_transport_snapshot


_REFERENCE_NAMES = {
    "timed_quantitative": "zeitlich zugeordnetes Echo/Labor",
    "administrative_or_text": "administrativer Code oder Text",
    "same_ecg_device": "Gerätebefund desselben ECGs",
    "unverified_or_unknown": "unklare Referenz",
}
_FAMILY_NAMES = {
    "qrs_t_coupling": "QRS–T-Kopplung",
    "spatial_ventricular_gradient": "Ventrikulärer Gradient",
    "repolarization_intervals": "Repolarisationsintervalle",
    "beat_variability": "Schlagvariabilität",
    "dipolarity_pca": "Dipolarität/PCA",
    "spectral_wavelet": "Spektrum/Wavelets",
    "topology_and_recurrence": "Topologie/Rekurrenz",
    "kinematics": "Schleifenkinematik",
    "magnitude_voltage": "Spannung/Betrag",
    "integrated_qrst_geometry": "QRST-Geometrie",
    "atrial_activation": "Vorhofaktivierung",
    "st_ischemia_geometry": "ST-Geometrie",
    "t_repolarization": "T-Repolarisation",
    "qrs_activation_geometry": "QRS-Aktivierung",
    "ecg_intervals": "ECG-Intervalle",
    "other_vcg": "Weitere VCG-Marker",
    "small_interpretable_score": "Kleine interpretierbare Scores",
}


def _display(value: object, fallback: str) -> str:
    if value is None or pd.isna(value):
        return fallback
    label = str(value).strip()
    return label if label else fallback


def load_map_frame(
    repository: ResultRepository, run_id: str,
    *, verified_page: ResultPage | None = None,
) -> pd.DataFrame:
    """One selected AUC per endpoint/family for *navigation*, not inference."""
    page = verified_page or repository.query(ResultQuery(
        perspective="pathologies", run_id=run_id,
        validation_only=True, complete_runs_only=True, limit=1,
    ))
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return pd.DataFrame()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        path = str(page.artifact_path)
        schema = {item[0] for item in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [path],
        ).description}
        needed = {
            "endpoint_id", "endpoint_name_de", "pathology_group", "conceptual_family",
            "feature_id", "roc_auc", "selection_eligible", "candidate_type",
            "reference_independence", "claim_ceiling", "evaluation_split",
        }
        if not needed.issubset(schema):
            return pd.DataFrame()
        scope = " AND ".join(compatible_validation_clauses(schema))
        return connection.execute(
            "SELECT CAST(endpoint_id AS VARCHAR) AS endpoint_id, "
            "MAX(CAST(endpoint_name_de AS VARCHAR)) AS endpoint_name_de, "
            "COALESCE(CAST(pathology_group AS VARCHAR), 'Andere') AS pathology_group, "
            "COALESCE(CAST(conceptual_family AS VARCHAR), 'Sonstige Marker') AS conceptual_family, "
            "MAX(TRY_CAST(roc_auc AS DOUBLE)) AS navigation_auc, "
            "COUNT(*) AS candidate_count "
            "FROM read_parquet(?) "
            "WHERE " + scope + " "
            "AND CAST(selection_eligible AS BOOLEAN) = true "
            "AND COALESCE(CAST(candidate_type AS VARCHAR), '') <> 'classical_rule' "
            "AND COALESCE(CAST(reference_independence AS VARCHAR), '') <> 'same_ecg_technical_comparison' "
            "AND COALESCE(CAST(claim_ceiling AS VARCHAR), '') <> 'agreement_only' "
            "AND TRY_CAST(roc_auc AS DOUBLE) IS NOT NULL "
            "AND TRY_CAST(roc_auc AS DOUBLE) BETWEEN 0 AND 1 "
            "GROUP BY endpoint_id, pathology_group, conceptual_family "
            "ORDER BY pathology_group, endpoint_id, conceptual_family",
            [path],
        ).fetchdf()
    finally:
        connection.close()


def load_baseline_frame(
    repository: ResultRepository, run_id: str,
    *, verified_page: ResultPage | None = None,
) -> pd.DataFrame:
    """A bounded view of classical ECG rules, not a guideline pass/fail test."""
    page = verified_page or repository.query(ResultQuery(
        perspective="pathologies", run_id=run_id,
        validation_only=True, complete_runs_only=True, limit=1,
    ))
    if page.status not in {"ok", "empty"} or page.artifact_path is None:
        return pd.DataFrame()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        path = str(page.artifact_path)
        schema = {item[0] for item in connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [path],
        ).description}
        if not {"candidate_type", "endpoint_id", "feature_id", "roc_auc"}.issubset(schema):
            return pd.DataFrame()
        fields = (
            "endpoint_id", "endpoint_name_de", "feature_id", "feature_name_de",
            "reference_kind", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
            "n_positive", "n_negative", "baseline_registry_id",
        )
        projected = ", ".join(
            f'"{name}"' if name in schema else f'NULL AS "{name}"'
            for name in fields
        )
        where = "CAST(candidate_type AS VARCHAR) = 'classical_rule'"
        where += " AND " + " AND ".join(compatible_validation_clauses(schema)) if compatible_validation_clauses(schema) else ""
        count = int(connection.execute(
            f"SELECT COUNT(*) FROM read_parquet(?) WHERE {where}", [path],
        ).fetchone()[0])
        result = connection.execute(
            f"SELECT {projected} FROM read_parquet(?) WHERE {where} "
            "ORDER BY endpoint_id, feature_id LIMIT 100", [path],
        ).fetchdf()
        result.attrs["total_count"] = count
        return result
    finally:
        connection.close()


class MapLoad(QThread):
    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, generation: int, repository: ResultRepository, run_id: str):
        super().__init__()
        self.generation = generation
        self.repository = repository
        self.run_id = run_id

    def run(self) -> None:
        try:
            from .rule_audit import load_rule_audit

            repository = ResultRepository(
                self.repository.config,
                expected_feature_code_hash=self.repository.expected_feature_code_hash,
                expected_validation_code_hash=self.repository.expected_validation_code_hash,
            )
            page = repository.query(ResultQuery(
                perspective="pathologies", run_id=self.run_id,
                validation_only=True, complete_runs_only=True, limit=1,
            ))
            map_frame = load_map_frame(repository, self.run_id, verified_page=page)
            baseline_frame = load_baseline_frame(repository, self.run_id, verified_page=page)
            rule_audit = load_rule_audit(page)
            try:
                discovery = load_discovery_snapshot(repository, self.run_id)
            except Exception as exc:
                discovery = DiscoverySnapshot(
                    "failed", f"Breiter Scan konnte nicht gelesen werden: {type(exc).__name__}: {exc}",
                    pd.DataFrame(), 0, pd.DataFrame(), self.run_id, "",
                )
            try:
                transport = load_transport_snapshot(repository, self.run_id)
            except Exception as exc:
                transport = TransportSnapshot(
                    "failed", f"Externe Kohorten konnten nicht gelesen werden: {type(exc).__name__}: {exc}",
                    pd.DataFrame(), pd.DataFrame(), self.run_id, "",
                )
            self.ready.emit(self.generation, (map_frame, baseline_frame, discovery, transport, rule_audit))
        except Exception as exc:
            self.failed.emit(self.generation, f"Landkarte nicht verfügbar: {type(exc).__name__}: {exc}")


class AtlasMapWidget(QWidget):
    endpoint_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._endpoints: list[str] = []
        self._families: list[str] = []
        self._grid = np.empty((0, 0), dtype=float)
        self._provenance: dict[str, object] = {}
        layout = QVBoxLayout(self)
        title = QLabel("Wo liegen die auswertbaren Forschungssignale?")
        title.setStyleSheet("font-weight: 650; font-size: 16px;")
        layout.addWidget(title)
        self.note = QLabel(
            "Farbe zeigt die höchste Validation-AUC einer Markerfamilie je Endpunkt – "
            "nur zur Navigation, kein unabhängiger Wirksamkeitsnachweis. "
            "Leere Felder sind nicht gemessen; AUC 0,5 ist ein gemessener Neutralwert."
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        row = QHBoxLayout()
        row.addWidget(QLabel("Klinischer Bereich"))
        self.domain = QComboBox()
        self.domain.currentIndexChanged.connect(self._draw)
        row.addWidget(self.domain)
        row.addStretch(1)
        layout.addLayout(row)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(360)
        self.plot.setLabel("bottom", "Klinischer Endpunkt")
        self.plot.setLabel("left", "Markerfamilie")
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        layout.addWidget(self.plot, 1)
        legend_row = QHBoxLayout()
        legend_row.addWidget(QLabel("AUC 0,5 · keine Trennung"))
        gradient = pg.colormap.get("viridis").map(np.linspace(0, 1, 180), mode="byte")
        gradient_rgba = np.tile(gradient[None, :, :], (14, 1, 1)).copy()
        gradient_image = QImage(
            gradient_rgba.data, 180, 14, 180 * 4, QImage.Format.Format_RGBA8888,
        ).copy()
        self.legend_gradient = QLabel()
        self.legend_gradient.setPixmap(QPixmap.fromImage(gradient_image))
        legend_row.addWidget(self.legend_gradient)
        legend_row.addWidget(QLabel("AUC 1,0 · perfekte Trennung"))
        legend_row.addStretch(1)
        legend_row.addWidget(QLabel("□ transparent: nicht berechnet"))
        layout.addLayout(legend_row)
        export_row = QHBoxLayout()
        plot_export = QPushButton("Landkarte als PNG/SVG")
        plot_export.setToolTip("Exportiert den gewählten klinischen Bereich mit Herkunftsdatei.")
        plot_export.clicked.connect(self._export_plot)
        export_row.addWidget(plot_export)
        csv_export = QPushButton("Kartendaten als CSV")
        csv_export.setToolTip("Exportiert die aggregierten Navigationswerte; keine klinischen Befunde.")
        csv_export.clicked.connect(self._export_table)
        export_row.addWidget(csv_export)
        export_row.addStretch(1)
        layout.addLayout(export_row)
        self.selection = QLabel("Endpunkt anklicken, um seine Belege zu sehen.")
        self.selection.setWordWrap(True)
        layout.addWidget(self.selection)

    def set_frame(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()
        previous = self.domain.currentData()
        self.domain.blockSignals(True)
        self.domain.clear()
        if not frame.empty and "pathology_group" in frame:
            for name in sorted(frame["pathology_group"].dropna().astype(str).unique()):
                self.domain.addItem(name, name)
        if previous is not None:
            index = self.domain.findData(previous)
            if index >= 0:
                self.domain.setCurrentIndex(index)
        self.domain.blockSignals(False)
        self._draw()

    def set_provenance(self, run_id: str, snapshot_hash: str) -> None:
        self._provenance = {
            "run_id": run_id, "snapshot_hash": snapshot_hash,
            "split": "validation", "view": "clinical_domain_marker_map",
            "note": "Maximum innerhalb einer Markerfamilie nur zur Navigation; kein unabhängiger Leistungsnachweis.",
        }

    def _export_plot(self, *_args) -> None:
        if self._frame.empty or not self._endpoints:
            return
        metadata = {**self._provenance, "pathology_group": self.domain.currentData()}
        choose_plot_export(self, self.plot, metadata, proposed_name="markerlandkarte.png")

    def _export_table(self, *_args) -> None:
        if self._frame.empty:
            return
        frame = self._frame.loc[
            self._frame["pathology_group"].astype(str).eq(str(self.domain.currentData()))
        ]
        choose_table_export(
            self, frame, {**self._provenance, "pathology_group": self.domain.currentData()},
            proposed_name="markerlandkarte.csv",
        )

    def _draw(self, *_args) -> None:
        self.plot.clear()
        self._endpoints = []
        self._families = []
        self._grid = np.empty((0, 0), dtype=float)
        if self._frame.empty or self.domain.currentData() is None:
            self.selection.setText("Für diesen Lauf ist noch keine auswertbare Markerlandkarte vorhanden.")
            return
        frame = self._frame.loc[
            self._frame["pathology_group"].astype(str).eq(str(self.domain.currentData()))
        ]
        # A bounded, explicit window. All other results remain in the full explorer.
        endpoints = frame[["endpoint_id", "endpoint_name_de"]].drop_duplicates().head(18)
        self._endpoints = endpoints["endpoint_id"].astype(str).tolist()
        labels = endpoints["endpoint_name_de"].fillna(endpoints["endpoint_id"]).astype(str).tolist()
        self._families = frame["conceptual_family"].fillna("Sonstige Marker").astype(str).value_counts().head(14).index.tolist()
        if not self._endpoints or not self._families:
            self.selection.setText("Für diesen klinischen Bereich fehlen auswertbare Kombinationen.")
            return
        grid = np.full((len(self._endpoints), len(self._families)), np.nan)
        for _, row in frame.iterrows():
            endpoint = str(row["endpoint_id"])
            family = str(row["conceptual_family"])
            value = row["navigation_auc"]
            if endpoint in self._endpoints and family in self._families and pd.notna(value):
                grid[self._endpoints.index(endpoint), self._families.index(family)] = float(value)
        self._grid = grid
        self.plot.addItem(pg.ImageItem(auc_matrix_rgba(grid)))
        self.plot.getAxis("bottom").setTicks([[(i + .5, name[:22]) for i, name in enumerate(labels)]])
        self.plot.getAxis("left").setTicks([[
            (i + .5, _FAMILY_NAMES.get(name, name.replace("_", " "))[:28])
            for i, name in enumerate(self._families)
        ]])
        self.plot.setXRange(0, len(self._endpoints), padding=0)
        self.plot.setYRange(0, len(self._families), padding=0)
        self.selection.setText(
            f"{len(self._endpoints)} Endpunkte · {len(self._families)} Markerfamilien. "
            "Viridis: 0,5 = neutral, 1,0 = hohe Trennung; transparent = nicht berechnet."
        )

    def _clicked(self, event) -> None:
        if not self._endpoints or not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
        index = math.floor(point.x())
        family_index = math.floor(point.y())
        if 0 <= index < len(self._endpoints) and 0 <= family_index < len(self._families):
            if not np.isfinite(self._grid[index, family_index]):
                self.selection.setText(
                    "Diese Markerfamilie wurde für den Endpunkt nicht auswertbar berechnet. "
                    "Das ist kein neutraler AUC-Wert."
                )
                return
            self.endpoint_requested.emit(self._endpoints[index])


class AtlasBaselineWidget(QWidget):
    endpoint_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._provenance: dict[str, object] = {}
        layout = QVBoxLayout(self)
        title = QLabel("Klassische ECG-Regeln als Qualitäts- und Vergleichsmaßstab")
        title.setStyleSheet("font-weight: 650; font-size: 16px;")
        layout.addWidget(title)
        self.note = QLabel(
            "Die Regelwerte beziehen sich auf die jeweilige Krankenhauskohorte und Referenz. "
            "Sie prüfen keine Leitlinie insgesamt; ein Gerätebericht ist kein unabhängiger Goldstandard. "
            "Die vollständige Liste bleibt unter „Alle Forschungsansichten“ verfügbar."
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Ziel oder klassische Regel suchen")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_rows)
        layout.addWidget(self.search)
        export_button = QPushButton("Klassische Vergleichsregeln als CSV")
        export_button.setToolTip(
            "Exportiert die geladenen maximal 100 Regelzeilen mit Herkunftsdatei; "
            "der Suchfilter ändert die CSV nicht."
        )
        export_button.clicked.connect(self._export_table)
        layout.addWidget(export_button)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels((
            "Klinisches Ziel", "Klassische ECG-Regel", "Validation-AUC", "Fälle / Kontrollen", "Referenz",
        ))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(self._open_row)
        layout.addWidget(self.table, 1)

    def set_frame(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        if frame.empty:
            self.note.setText(
                "Für diesen Lauf ist kein auswertbarer klassischer Regelvergleich verfügbar. "
                "Daraus folgt weder Übereinstimmung noch Abweichung von Leitlinien."
            )
            self.table.setSortingEnabled(True)
            return
        total = int(frame.attrs.get("total_count", len(frame)))
        self.note.setText(
            f"{total:,} Regel–Endpunkt-Zeilen im Lauf; hier höchstens 100 alphabetisch. "
            "Validation-AUC ist eine Kohortenkennzahl, kein Leitlinien-Pass/Fail. "
            "Doppelklick öffnet den Endpunkt."
        )
        self.table.setRowCount(len(frame))
        for index, (_, row) in enumerate(frame.iterrows()):
            endpoint_id = _display(row.get("endpoint_id"), "")
            auc = pd.to_numeric(row.get("roc_auc"), errors="coerce")
            low = pd.to_numeric(row.get("roc_auc_ci_low"), errors="coerce")
            high = pd.to_numeric(row.get("roc_auc_ci_high"), errors="coerce")
            auc_text = f"{auc:.2f}" if pd.notna(auc) else "nicht berechnet"
            if pd.notna(low) and pd.notna(high):
                auc_text += f" ({low:.2f}–{high:.2f})"
            positive = pd.to_numeric(row.get("n_positive"), errors="coerce")
            negative = pd.to_numeric(row.get("n_negative"), errors="coerce")
            counts = (
                f"{int(positive)} / {int(negative)}"
                if pd.notna(positive) and pd.notna(negative)
                else "nicht angegeben"
            )
            values = (
                _display(row.get("endpoint_name_de"), endpoint_id),
                _display(row.get("feature_name_de"), _display(row.get("feature_id"), "")),
                auc_text, counts, _REFERENCE_NAMES.get(
                    _display(row.get("reference_kind"), "unverified_or_unknown"),
                    _display(row.get("reference_kind"), "nicht angegeben").replace("_", " "),
                ),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, endpoint_id)
                self.table.setItem(index, column, item)
        self.table.setSortingEnabled(True)
        self._filter_rows()

    def set_provenance(self, run_id: str, snapshot_hash: str) -> None:
        self._provenance = {
            "run_id": run_id, "snapshot_hash": snapshot_hash,
            "split": "validation", "view": "classical_rule_comparison",
            "note": "Krankenhauskohorten-Vergleich, kein Leitlinien-Pass/Fail.",
        }

    def _export_table(self, *_args) -> None:
        if self._frame.empty:
            return
        choose_table_export(
            self, self._frame, self._provenance,
            proposed_name="klassische_ecg_regeln.csv",
        )

    def _filter_rows(self, *_args) -> None:
        search = self.search.text().strip().casefold()
        for row in range(self.table.rowCount()):
            haystack = " ".join(
                self.table.item(row, column).text()
                for column in (0, 1)
                if self.table.item(row, column) is not None
            ).casefold()
            self.table.setRowHidden(row, bool(search and search not in haystack))

    def _open_row(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            endpoint = item.data(Qt.ItemDataRole.UserRole)
            if endpoint:
                self.endpoint_requested.emit(str(endpoint))
