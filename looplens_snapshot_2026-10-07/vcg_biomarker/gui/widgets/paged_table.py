from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ...mimic.user_guidance import (
    biomarker_display_name,
    biomarker_tooltip,
    clinical_display_term,
)


COLUMN_LABELS_DE = {
    "clinical_target_de": "Pathologie / klinisches Ziel",
    "clinical_category_de": "Klinische Kategorie",
    "clinical_target_definition_de": "Definition",
    "pathology_group": "Pathologiegruppe",
    "endpoint_id": "Klinischer Endpunkt",
    "target_name": "Zusammenhang",
    "target_kind": "Referenztyp",
    "feature_id": "Biomarker",
    "candidate_type": "Markertyp",
    "conceptual_family": "Markerfamilie",
    "panel_id": "Panel",
    "evaluation_split": "Split",
    "inference_status": "Powerstatus",
    "n_total": "Gesamt",
    "n_positive": "Fälle",
    "n_negative": "Kontrollen",
    "missingness": "Fehlend",
    "roc_auc": "ROC-AUC",
    "roc_auc_ci_low": "AUC-CI unten",
    "roc_auc_ci_high": "AUC-CI oben",
    "average_precision": "Average Precision",
    "average_precision_ci_low": "AP-CI unten",
    "average_precision_ci_high": "AP-CI oben",
    "sensitivity_at_90_specificity": "Sens. @ 90 % Spez.",
    "sensitivity_at_95_specificity": "Sens. am Trainingsschwellenwert (95-%-Ziel)",
    "sensitivity_at_95_specificity_reported": "Beobachtete Sens. (Trainingsziel: 95 % Spez.)",
    "operating_point_validation_status": "Validierungsziel-Status",
    "operating_point_training_status": "Trainingsziel / Schwellenprovenienz",
    "operating_point_ranking_eligible": "Operating Point zur Rangierung zugelassen",
    "selection_eligible": "Zur Auswahl zugelassen",
    "specificity_at_90_sensitivity": "Spez. @ 90 % Sens.",
    "q_value": "FDR q",
    "rank_biserial": "Effekt",
    "direction_agreement": "Richtung stabil",
    "train_validation_auc_gap": "Train-Val.-Gap",
    "baseline_display_name": "Leitlinienvergleich",
    "delta_auc": "Δ AUC",
    "comparison_verdict": "Verdikt (Code)",
    "comparison_verdict_de": "Verdikt",
    "novel_vs_literature": "Vergleich mit Literatur",
    "evidence_grade": "Evidenz",
    "clinical_interpretation": "Einordnung",
    "evidence_tier": "Evidenzstufe",
    "discovery_domain": "Domäne",
    "validation_effect": "Validation-Effekt",
    "validation_q_value": "Validation q",
    "validation_auc": "Validation-AUC",
    "validation_ci_low": "AUC-CI unten",
    "validation_ci_high": "AUC-CI oben",
    "validation_stable": "Validation stabil",
    "replicated": "Repliziert",
    "historical_implementation": "Neuberechnung nötig",
    "confounding_control": "Adjustierung",
    "requires_endpoint_specific_adjustment": "Detailprüfung nötig",
    "followup_priority": "Nachverfolgung",
    "novelty_class": "Neuheitsklasse",
    "interpretation": "Einordnung",
}

COLUMN_TOOLTIPS_DE = {
    "roc_auc": (
        "Fläche unter der ROC-Kurve auf dem angezeigten Split; 0,5 entspricht "
        "Zufall. Die Markerrichtung wurde ausschließlich in Train festgelegt."
    ),
    "average_precision": (
        "Average Precision auf dem angezeigten Split; immer zusammen mit der "
        "Endpunktprävalenz interpretieren."
    ),
    "sensitivity_at_90_specificity": (
        "Sensitivität auf dem angezeigten Split bei einer ausschließlich in "
        "Train für mindestens 90 % Spezifität gewählten Schwelle."
    ),
    "sensitivity_at_95_specificity": (
        "Sensitivität auf dem angezeigten Split bei einer ausschließlich in "
        "Train für mindestens 95 % Spezifität gewählten Schwelle."
    ),
    "specificity_at_90_sensitivity": (
        "Spezifität auf dem angezeigten Split bei einer ausschließlich in "
        "Train für mindestens 90 % Sensitivität gewählten Schwelle."
    ),
    "q_value": (
        "Benjamini-Hochberg-korrigierter p-Wert innerhalb der vorab "
        "festgelegten Endpunkt- und Panelfamilie."
    ),
    "evidence_grade": (
        "Interne Entwicklungseinstufung aus Power, FDR, AUC, Stabilität und "
        "Referenzunabhängigkeit; keine klinische Evidenzklasse."
    ),
    "direction_agreement": (
        "Zeigt, ob die in Train gefundene pathologische Richtung in "
        "Validation erhalten blieb."
    ),
    "train_validation_auc_gap": (
        "Train-AUC minus Validation-AUC. Große absolute Abstände sprechen "
        "gegen stabile Generalisierung."
    ),
    "delta_auc": (
        "Gepaarte AUC-Differenz zum anwendbaren kontinuierlichen Vergleich. "
        "Binäre Leitlinienregeln werden stattdessen am Betriebspunkt verglichen."
    ),
    "validation_q_value": (
        "FDR-korrigierter Replikations-p-Wert im Validation-Split des breiten "
        "Scans."
    ),
    "replicated": (
        "Gleiche Richtung und vorab definierte Signifikanz-/Stabilitätsregeln "
        "in Train und Validation; der Locked Test wurde nicht gescannt."
    ),
    "historical_implementation": (
        "Die Assoziation stammt aus einem Lauf vor der aktuellen "
        "Biomarker-, Delineations- oder Rollenimplementierung und muss in "
        "Train/Validation neu berechnet werden."
    ),
    "followup_priority": (
        "Methodische Priorisierung für eine erneute, endpunktspezifische "
        "Prüfung; keine klinische Evidenzstufe."
    ),
    "confounding_control": (
        "Der breite Scan ist eine unadjustierte Entdeckungsanalyse. Alter, "
        "Geschlecht, klinischer Kontext und weitere Confounder werden erst "
        "in der endpunktspezifischen Detailprüfung kontrolliert."
    ),
    "comparison_verdict_de": (
        "Abgeleitet aus der gepaarten Subject-Cluster-Bootstrap-CI "
        "(AUC-Differenz bei kontinuierlichen Vergleichsmarkern, "
        "Sensitivitätsdifferenz bei fixen Leitlinienregeln). Unter "
        "formaler Inferenzpower (< 100 je Klasse) immer "
        "'nicht genügend Evidenz', nie eine mögliche Zufallsaussage."
    ),
    "novel_vs_literature": (
        "Einfacher Schlagwortabgleich mit einer kleinen, kuratierten "
        "Literaturliste. 'Nicht beschrieben' ist keine Neuheitsgarantie, "
        "nur ein Hinweis, dass diese Liste den Zusammenhang noch nicht "
        "kennt."
    ),
}


class ParquetPageModel(QAbstractTableModel):
    def __init__(self, *, page_size: int = 250, parent=None):
        super().__init__(parent)
        self.page_size = max(25, int(page_size))
        self.path: Optional[Path] = None
        self.where_sql = ""
        self.parameters: tuple = ()
        self.total_rows = 0
        self.page_index = 0
        self.frame = pd.DataFrame()
        self.sort_column = ""
        self.sort_descending = False
        self.last_error = ""
        self._source_signature: tuple[int, int] = (0, 0)

    @property
    def page_count(self) -> int:
        if self.total_rows <= 0:
            return 0
        return (self.total_rows + self.page_size - 1) // self.page_size

    def set_source(
        self,
        path: str | Path | None,
        *,
        where_sql: str = "",
        parameters: tuple = (),
    ) -> None:
        resolved = Path(path).resolve() if path else None
        signature = _file_signature(resolved)
        if (
            resolved == self.path
            and str(where_sql).strip() == self.where_sql
            and tuple(parameters) == self.parameters
            and signature == self._source_signature
        ):
            return
        self.beginResetModel()
        self.path = resolved
        self.where_sql = str(where_sql).strip()
        self.parameters = tuple(parameters)
        self._source_signature = signature
        self.page_index = 0
        self.sort_column = ""
        self.sort_descending = False
        self.total_rows = 0
        self.frame = pd.DataFrame()
        self.endResetModel()
        self.reload()

    def reload(self) -> None:
        self.beginResetModel()
        try:
            self._source_signature = _file_signature(self.path)
            self._load_page()
        finally:
            self.endResetModel()

    def set_page(self, page_index: int) -> None:
        bounded = min(max(0, int(page_index)), max(0, self.page_count - 1))
        if bounded == self.page_index:
            return
        self.page_index = bounded
        self.reload()

    def row_dict(self, row: int) -> dict:
        if row < 0 or row >= len(self.frame):
            return {}
        return self.frame.iloc[row].to_dict()

    def rowCount(self, _parent=QModelIndex()) -> int:
        return len(self.frame)

    def columnCount(self, _parent=QModelIndex()) -> int:
        return len(self.frame.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        column = str(self.frame.columns[index.column()])
        if role == Qt.ItemDataRole.DisplayRole:
            if column in {"endpoint_id", "label_id"} and not pd.isna(value):
                return clinical_display_term(value).display_name_de
            if column in {"feature_id", "vcg_feature_id"} and not pd.isna(value):
                return biomarker_display_name(value)
            return _display_value(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            if column in {"endpoint_id", "label_id"} and not pd.isna(value):
                term = clinical_display_term(value)
                return (
                    f"{term.display_name_de}\n{term.short_definition_de}\n"
                    f"Interne ID: {value}"
                )
            if column in {"feature_id", "vcg_feature_id"} and not pd.isna(value):
                return f"{biomarker_tooltip(value)}\nInterne ID: {value}"
            text = _display_value(value)
            return text if len(text) > 30 else None
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            return int(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )
        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.frame.columns):
                column = str(self.frame.columns[section])
                if role == Qt.ItemDataRole.DisplayRole:
                    return COLUMN_LABELS_DE.get(column, column)
                if role == Qt.ItemDataRole.ToolTipRole:
                    return COLUMN_TOOLTIPS_DE.get(
                        column,
                        (
                            f"Interne Ergebnisspalte: {column}. "
                            "Klicken zum Sortieren."
                        ),
                    )
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.page_index * self.page_size + section + 1
        return None

    def sort(
        self,
        column: int,
        order: Qt.SortOrder = Qt.SortOrder.AscendingOrder,
    ) -> None:
        if not 0 <= column < len(self.frame.columns):
            return
        self.sort_column = str(self.frame.columns[column])
        self.sort_descending = order == Qt.SortOrder.DescendingOrder
        self.page_index = 0
        self.reload()

    def _load_page(self) -> None:
        self.last_error = ""
        if self.path is None or not self.path.is_file():
            self.total_rows = 0
            self.frame = pd.DataFrame()
            return
        relation = "read_parquet(?)"
        where = f" WHERE {self.where_sql}" if self.where_sql else ""
        count_query = f"SELECT COUNT(*) FROM {relation}{where}"
        order = ""
        if self.sort_column:
            direction = "DESC" if self.sort_descending else "ASC"
            escaped = self.sort_column.replace('"', '""')
            order = f' ORDER BY "{escaped}" {direction} NULLS LAST'
        page_query = (
            f"SELECT * FROM {relation}{where}{order} "
            "LIMIT ? OFFSET ?"
        )
        try:
            with duckdb.connect() as connection:
                self.total_rows = int(
                    connection.execute(
                        count_query,
                        [str(self.path), *self.parameters],
                    ).fetchone()[0]
                )
                offset = self.page_index * self.page_size
                self.frame = connection.execute(
                    page_query,
                    [
                        str(self.path),
                        *self.parameters,
                        self.page_size,
                        offset,
                    ],
                ).fetchdf()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.total_rows = 1
            self.page_index = 0
            self.frame = pd.DataFrame(
                {
                    "status": ["nicht verfügbar"],
                    "diagnostic": [self.last_error],
                }
            )


class PagedParquetTable(QWidget):
    selection_changed = pyqtSignal()

    def __init__(self, *, page_size: int = 250, parent=None):
        super().__init__(parent)
        self.model = ParquetPageModel(page_size=page_size, parent=self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setSortingEnabled(True)
        self.view.setAlternatingRowColors(True)
        self.view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.view.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.view.verticalHeader().setDefaultSectionSize(24)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.selectionModel().selectionChanged.connect(
            lambda *_: self.selection_changed.emit()
        )
        layout.addWidget(self.view, 1)

        pager = QHBoxLayout()
        self.previous_button = QPushButton("Zurück")
        self.previous_button.setToolTip("Vorherige Ergebnisseite anzeigen")
        self.previous_button.clicked.connect(
            lambda: self.set_page(self.model.page_index - 1)
        )
        pager.addWidget(self.previous_button)
        self.page_label = QLabel("Keine Daten")
        pager.addWidget(self.page_label)
        self.next_button = QPushButton("Weiter")
        self.next_button.setToolTip("Nächste Ergebnisseite anzeigen")
        self.next_button.clicked.connect(
            lambda: self.set_page(self.model.page_index + 1)
        )
        pager.addWidget(self.next_button)
        pager.addStretch(1)
        layout.addLayout(pager)
        self._update_pager()

    def set_source(
        self,
        path: str | Path | None,
        *,
        where_sql: str = "",
        parameters: tuple = (),
    ) -> None:
        self.model.set_source(
            path,
            where_sql=where_sql,
            parameters=parameters,
        )
        self._update_pager()

    def reload(self) -> None:
        self.model.reload()
        self._update_pager()

    def set_page(self, page: int) -> None:
        self.model.set_page(page)
        self._update_pager()

    def selectRow(self, row: int) -> None:
        self.view.selectRow(row)

    def selected_row(self) -> dict:
        indexes = self.view.selectionModel().selectedRows()
        if not indexes:
            return {}
        return self.model.row_dict(indexes[0].row())

    def _update_pager(self) -> None:
        pages = self.model.page_count
        self.previous_button.setEnabled(self.model.page_index > 0)
        self.next_button.setEnabled(
            pages > 0 and self.model.page_index + 1 < pages
        )
        if pages:
            start = self.model.page_index * self.model.page_size + 1
            end = min(
                self.model.total_rows,
                start + len(self.model.frame) - 1,
            )
            self.page_label.setText(
                f"{start:,}–{end:,} von {self.model.total_rows:,}"
            )
        else:
            self.page_label.setText("Keine Daten")


def _display_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.5g}"
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _file_signature(path: Optional[Path]) -> tuple[int, int]:
    if path is None or not path.is_file():
        return (0, 0)
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size
