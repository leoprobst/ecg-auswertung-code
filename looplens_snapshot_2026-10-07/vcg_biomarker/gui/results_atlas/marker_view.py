"""Marker dossier: one definition, multiple non-pooled clinical questions."""
from __future__ import annotations

import math

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...mimic.models import ResultQuery
from ...mimic.result_query import ResultPage, ResultRepository
from .export import choose_plot_export, choose_table_export
from .marker_context import AtlasMarkerContextWidget, MarkerContext, load_marker_context


class MarkerLoad(QThread):
    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(
        self, generation: int, repository: ResultRepository,
        run_id: str, feature_id: str,
    ):
        super().__init__()
        self.generation = generation
        self.repository = repository
        self.run_id = run_id
        self.feature_id = feature_id

    def run(self) -> None:
        try:
            repository = ResultRepository(
                self.repository.config,
                expected_feature_code_hash=self.repository.expected_feature_code_hash,
                expected_validation_code_hash=self.repository.expected_validation_code_hash,
            )
            page = repository.query(ResultQuery(
                perspective="biomarkers", run_id=self.run_id,
                feature_ids=(self.feature_id,), validation_only=True,
                complete_runs_only=True, sort_by="evidence_rank", limit=200,
            ))
            context = load_marker_context(page, self.feature_id)
            self.ready.emit(self.generation, (page, context))
        except Exception as exc:
            self.failed.emit(
                self.generation,
                f"Marker-Dossier konnte nicht geladen werden: {type(exc).__name__}: {exc}",
            )


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _text(value: object, fallback: str = "nicht angegeben") -> str:
    if value is None or pd.isna(value):
        return fallback
    result = str(value).strip()
    return result if result else fallback


def marker_rows(page: ResultPage) -> pd.DataFrame:
    if page.status not in {"ok", "empty"} or page.rows.empty:
        return pd.DataFrame()
    frame = page.rows.copy()
    if "reference_independence" in frame:
        frame = frame.loc[~frame["reference_independence"].astype(str).eq(
            "same_ecg_technical_comparison"
        )]
    if "historical_pre_audit_result" in frame:
        frame = frame.loc[~frame["historical_pre_audit_result"].astype(str).str.casefold().eq("true")]
    elif "historical_implementation" in frame:
        frame = frame.loc[~frame["historical_implementation"].astype(str).str.casefold().eq("true")]
    if "roc_auc" in frame:
        frame = frame.loc[pd.to_numeric(frame["roc_auc"], errors="coerce").notna()]
    return frame.head(25)


class _AucTableItem(QTableWidgetItem):
    """Keep the CI in the display text while sorting by the numeric AUC."""

    def __init__(self, label: str, value: float | None):
        super().__init__(label)
        self._numeric_value = value if value is not None else float("-inf")

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, _AucTableItem):
            return self._numeric_value < other._numeric_value
        return super().__lt__(other)


class AtlasMarkerWidget(QWidget):
    """An exploratory dossier; AUCs across endpoints are not pooled."""

    idle = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._generation = 0
        self._loaded_generation = -1
        self._request: tuple[str, str] | None = None
        self._worker: MarkerLoad | None = None
        self._shutting_down = False
        self._page: ResultPage | None = None
        self._display_frame = pd.DataFrame()
        self._interval_count = 0
        layout = QVBoxLayout(self)
        self.title = QLabel("Marker-Dossier: Marker in der Endpunkt-Tabelle auswählen")
        self.title.setStyleSheet("font-weight: 650; font-size: 16px;")
        layout.addWidget(self.title)
        self.definition = QLabel(
            "Die Formel, Einheit und Messvoraussetzungen stammen aus dem versiegelten Ergebnisstand."
        )
        self.definition.setWordWrap(True)
        layout.addWidget(self.definition)
        self.note = QLabel(
            "Jede Zeile ist eine eigene klinische Frage. AUCs verschiedener Endpunkte sind "
            "kein gemeinsamer Score für Spezifität oder klinischen Nutzen."
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.tabs = QTabWidget()
        performance_page = QWidget()
        performance_layout = QVBoxLayout(performance_page)
        self.tabs.addTab(performance_page, "Klinische Endpunkte")
        self.context_view = AtlasMarkerContextWidget()
        self.tabs.addTab(self.context_view, "Messung & Verteilung")
        layout.addWidget(self.tabs, 1)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(230)
        self.plot.setLabel("bottom", "Validation-ROC-AUC je Endpunkt")
        self.plot.setXRange(0, 1, padding=0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Klinisches Ziel in der Tabelle suchen")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_rows)
        performance_layout.addWidget(self.search)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels((
            "Klinisches Ziel", "AUC (95-%-CI)", "Fälle / Kontrollen",
            "Evidenzstufe", "Referenz",
        ))
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.performance_display = QTabWidget()
        self.performance_display.addTab(self.plot, "AUC mit 95-%-Intervall")
        self.performance_display.addTab(self.table, "Alle Markerzahlen")
        performance_layout.addWidget(self.performance_display, 1)
        actions = QHBoxLayout()
        chart_export = QPushButton("AUC-Diagramm als PNG/SVG")
        chart_export.setToolTip(
            "Exportiert nur Endpunkte mit tatsächlich berechnetem 95-%-Intervall und Herkunftsdatei."
        )
        chart_export.clicked.connect(self._export_plot)
        actions.addWidget(chart_export)
        table_export = QPushButton("Angezeigte Endpunkte als CSV")
        table_export.setToolTip(
            "Exportiert höchstens 25 Marker-Endpunkt-Zeilen, auf die aktuelle Suche begrenzt."
        )
        table_export.clicked.connect(self._export_table)
        actions.addWidget(table_export)
        actions.addStretch(1)
        performance_layout.addLayout(actions)

    def select_marker(
        self, repository: ResultRepository, run_id: str, feature_id: str,
    ) -> None:
        if self._shutting_down:
            return
        request = (run_id, feature_id)
        if request == self._request:
            return
        self._request = request
        self._generation += 1
        self._page = None
        self._display_frame = pd.DataFrame()
        self._interval_count = 0
        self.plot.clear()
        self.performance_display.setTabEnabled(0, False)
        self.performance_display.setCurrentIndex(1)
        self.table.setRowCount(0)
        self.context_view.set_context(MarkerContext(
            feature_id, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            ("Mess- und Verteilungsartefakte werden geprüft …",) * 3,
        ))
        self.title.setText(f"Marker-Dossier: {feature_id}")
        if self._worker is not None:
            self.note.setText("Vorige Abfrage wird abgeschlossen; danach folgt der neue Marker.")
            return
        self._start(repository, run_id, feature_id)

    def _start(
        self, repository: ResultRepository, run_id: str, feature_id: str,
    ) -> None:
        worker = MarkerLoad(self._generation, repository, run_id, feature_id)
        self._worker = worker
        worker.ready.connect(self._loaded)
        worker.failed.connect(self._failed)
        worker.finished.connect(lambda w=worker: self._retire(w))
        worker.start()

    def _retire(self, worker: MarkerLoad) -> None:
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()
        if self._shutting_down:
            self.idle.emit()
            return
        if self._loaded_generation != self._generation and self._request:
            self._start(worker.repository, *self._request)

    def request_shutdown(self) -> bool:
        self._shutting_down = True
        if self._worker is None or not self._worker.isRunning():
            return True
        self._worker.requestInterruption()
        return False

    def _failed(self, generation: int, message: str) -> None:
        if generation == self._generation:
            self._loaded_generation = generation
            self.note.setText(message)

    def _loaded(self, generation: int, payload: tuple[ResultPage, MarkerContext]) -> None:
        if generation != self._generation:
            return
        page, context = payload
        self._page = page
        self._loaded_generation = generation
        self.context_view.set_context(context)
        self.plot.clear()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        if page.status not in {"ok", "empty"}:
            self.note.setText(page.message_de or "Kein kompatibles Marker-Ergebnis vorhanden.")
            self.table.setSortingEnabled(True)
            return
        frame = marker_rows(page)
        self._display_frame = frame.copy()
        if frame.empty:
            self.note.setText("Für diesen Marker liegen keine eigenständigen, auswertbaren Endpunkte vor.")
            self.table.setSortingEnabled(True)
            return
        first = frame.iloc[0]
        self.title.setText(_text(first.get("feature_name_de"), self._request[1] if self._request else "Marker"))
        self.definition.setText(_text(
            first.get("feature_tooltip_de"),
            "Formel oder Messvoraussetzungen sind für diesen Ergebnisstand nicht dokumentiert.",
        ))
        self.note.setText(
            f"{page.total_count:,} Endpunkt-Zeilen für diesen Marker; höchstens 25 werden gezeigt. "
            "Kein gepoolter Krankheitsscore und keine automatische Befundungsregel."
        )
        self.table.setRowCount(len(frame))
        ticks: list[tuple[float, str]] = []
        interval_count = 0
        for index, (_, row) in enumerate(frame.iterrows()):
            endpoint = _text(row.get("endpoint_name_de"), _text(row.get("endpoint_id")))
            auc = _number(row.get("roc_auc"))
            low = _number(row.get("roc_auc_ci_low"))
            high = _number(row.get("roc_auc_ci_high"))
            auc_text = "nicht berechnet" if auc is None else f"{auc:.2f}"
            if auc is not None:
                if low is not None and high is not None and low <= auc <= high:
                    y = float(interval_count + 1)
                    self.plot.plot([low, high], [y, y], pen=pg.mkPen("#29a3a3", width=2))
                    auc_text += f" ({low:.2f}–{high:.2f})"
                    self.plot.plot([auc], [y], pen=None, symbol="o", symbolSize=8,
                                   symbolBrush="#29a3a3")
                    ticks.append((y, endpoint[:32]))
                    interval_count += 1
                else:
                    auc_text += " (CI nicht berechnet)"
            positive = _number(row.get("n_positive"))
            negative = _number(row.get("n_negative"))
            counts = (
                f"{int(positive)} / {int(negative)}"
                if positive is not None and negative is not None
                else "nicht angegeben"
            )
            stage = _text(row.get("evidence_stage"), "exploratory")
            stage_text = {
                "exploratory": "Forschungshypothese",
                "replicated_association": "Replizierte Assoziation",
                "guideline_candidate": "Kriterienkandidat",
                "clinically_validated": "Klinisch validiert",
            }.get(stage, "Forschungshypothese")
            values = (endpoint, auc_text, counts, stage_text,
                      _text(row.get("reference_kind")).replace("_", " "))
            for column, value in enumerate(values):
                item = _AucTableItem(value, auc) if column == 1 else QTableWidgetItem(value)
                self.table.setItem(index, column, item)
        self.plot.getAxis("left").setTicks([ticks])
        self.plot.setXRange(0, 1, padding=0)
        self.plot.setYRange(.5, max(1, interval_count) + .5, padding=0)
        self.performance_display.setTabEnabled(0, interval_count > 0)
        self.performance_display.setCurrentIndex(0 if interval_count else 1)
        self._interval_count = interval_count
        self.note.setText(
            self.note.text() + f" {interval_count} von {len(frame)} angezeigten Endpunkten "
            "haben ein gültiges AUC-Intervall; nur diese erscheinen im Diagramm."
        )
        self.table.setSortingEnabled(True)
        self._filter_rows()

    def _filter_rows(self, *_args) -> None:
        search = self.search.text().strip().casefold()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            self.table.setRowHidden(
                row, bool(search and (item is None or search not in item.text().casefold()))
            )

    def _metadata(self) -> dict[str, object]:
        page = self._page
        return {
            "run_id": page.run_id if page else None,
            "snapshot_hash": page.snapshot_hash if page else None,
            "feature_id": self._request[1] if self._request else None,
            "split": "validation", "view": "marker_endpoint_dossier",
            "note": "AUCs verschiedener Endpunkte sind nicht gepoolt; keine klinische Befundungsregel.",
        }

    def _export_plot(self, *_args) -> None:
        if self._interval_count:
            choose_plot_export(self, self.plot, self._metadata(),
                               proposed_name="marker_endpunkte_auc.png")

    def _export_table(self, *_args) -> None:
        frame = self._display_frame
        if frame.empty:
            return
        search = self.search.text().strip().casefold()
        if search:
            names = frame.get("endpoint_name_de", frame["endpoint_id"]).fillna(
                frame["endpoint_id"]
            ).astype(str)
            frame = frame.loc[names.str.casefold().str.contains(search, regex=False)]
        if not frame.empty:
            choose_table_export(self, frame, self._metadata(),
                                proposed_name="marker_endpunkte_validation.csv")
