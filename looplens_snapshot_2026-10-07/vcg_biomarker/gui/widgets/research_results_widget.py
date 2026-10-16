"""Read-only presentation of score, causal and regional research contracts."""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyarrow.parquet as pq
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QPlainTextEdit,
    QSplitter,
    QAbstractItemView,
)

log = logging.getLogger(__name__)


def research_rows(payload):
    """Missing estimates remain missing, never converted to neutral numbers."""
    if isinstance(payload, dict) and "contract" in payload and "tasks" in payload:
        contract = payload.get("contract")
        if (
            not isinstance(contract, dict)
            or contract.get("version") != "1.0.0"
            or contract.get("evidence_stage") != "exploratory"
            or contract.get("clinical_claim_allowed") is not False
            or payload.get("clinical_claim_allowed") is not False
            or payload.get("locked_test_read") is not False
            or payload.get("status") != "complete"
            or not isinstance(payload["tasks"], list)
        ):
            raise ValueError("Erweiterungsbericht ohne gültigen Forschungsabschluss")
        rows = []
        for task in payload["tasks"]:
            if not isinstance(task, dict) or not isinstance(task.get("kind"), str):
                raise ValueError("Ungültiger Erweiterungsauftrag")
            rows.append(dict(
                name=task["kind"], model="Zusatzanalyse",
                metric="Ergebniszeilen (keine Patientenzahl)",
                value=task.get("n_rows"), ci=None, n=None,
                status="Explorativ – Aufgabenübersicht, keine klinische Freigabe",
                detail=task,
            ))
        return rows
    if isinstance(payload, dict) and "deep_combination_version" in payload:
        if (
            payload.get("deep_combination_version") != "1.2.0"
            or payload.get("measurement_role")
            != "inner_selection_not_independent_confirmation"
            or payload.get("clinical_claim_allowed") is not False
            or payload.get("locked_test_read") is not False
        ):
            raise ValueError(
                "Kombinationsbericht ohne verifizierten internen Auswahlstatus"
            )
        return [
            dict(
                name=row.get("endpoint_id", "?"),
                model=f"Tiefe {row.get('depth', '?')}",
                metric="Interne Auswahl-AUC",
                value=row.get("best_confirmation_auc"),
                ci=None,
                n=None,
                status="Explorativ – keine unabhängige Bestätigung",
                detail=row,
            )
            for row in payload.get("depth_profile_rows", [])
        ]
    if not isinstance(payload, dict) or payload.get("version") not in {
        "1.0.0",
        "1.1.0",
    }:
        raise ValueError("Unbekannter oder inkompatibler Ergebnisvertrag")
    if payload.get("clinical_claim_allowed") is not False:
        raise ValueError("Expliziter Forschungsstatus fehlt")
    rows = []

    def add(name, model, metric, value, ci, n, status, detail):
        rows.append(
            dict(
                name=name,
                model=model,
                metric=metric,
                value=value,
                ci=ci,
                n=n,
                status=status,
                detail=detail,
            )
        )

    if "task_registry" in payload:
        for result in payload.get("results", []):
            endpoint = result.get("endpoint_id", result.get("endpoint", "?"))
            for name, model in result.get("models", {}).items():
                metrics = model.get("evaluation", {}).get("metrics", {})
                metric_key, metric_title = ("auc", "AUC")
                if "rmse" in metrics:
                    metric_key, metric_title = "rmse", "RMSE (kleiner ist besser)"
                elif "harrell_c" in metrics:
                    metric_key, metric_title = "harrell_c", "Harrell-C"
                add(
                    endpoint,
                    name,
                    metric_title,
                    metrics.get(metric_key),
                    metrics.get(f"{metric_key}_ci95"),
                    metrics.get("n_patients"),
                    model.get("status", "Forschung / verschachtelte CV"),
                    model,
                )
            if not result.get("models"):
                add(
                    endpoint,
                    "—",
                    "—",
                    None,
                    None,
                    None,
                    result.get("status", "nicht berechnet"),
                    result,
                )
            for name, comparison in result.get("paired_comparisons", {}).items():
                add(
                    endpoint,
                    name,
                    "ΔAUC (gepaart)",
                    comparison.get("delta_auc_second_minus_first"),
                    comparison.get("delta_auc_ci95"),
                    comparison.get("n_patients"),
                    comparison.get("status", "nicht berechnet"),
                    comparison,
                )
    elif "effects" in payload:
        for name, effect in payload["effects"].items():
            add(
                name,
                "AIPW: Behandlung minus Vergleich",
                effect.get("scale", "Effekt"),
                effect.get("ate"),
                effect.get("ci95"),
                payload.get("n_patients"),
                payload.get("status", "?"),
                effect,
            )
    elif "paired_differences" in payload:
        for method, result in payload.get("results", {}).items():
            add(
                payload.get("task", "Region"),
                method,
                "Macro-F1",
                result.get("macro_f1"),
                None,
                result.get("n_patients"),
                result.get("evaluation_design", "?"),
                result,
            )
        for name, comparison in payload["paired_differences"].items():
            add(
                payload.get("task", "Region"),
                name,
                "ΔTrefferquote (patientengewichtet)",
                comparison.get("patient_equal_accuracy_difference"),
                comparison.get("ci95"),
                None,
                "Bedingter Patientenbootstrap",
                comparison,
            )
    elif payload.get("status") in {"non_estimable", "failed"}:
        add("Nicht auswertbar", "—", "—", None, None, None, payload["status"], payload)
    else:
        raise ValueError("Kein unterstützter Score-/Therapie-/Mappingbericht")
    return rows


class ResearchResultsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.payload: dict = {}
        self._run_signature = None
        layout = QVBoxLayout(self)
        warning = QLabel(
            "Forschung: keine Diagnosefreigabe und keine Therapieempfehlung. "
            "SVD-Invarianten sind keine anatomischen XYZ-Koordinaten."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        toolbar = QHBoxLayout()
        self.reports = QComboBox()
        self.reports.setMinimumContentsLength(20)
        self.reports.currentIndexChanged.connect(self._selection)
        toolbar.addWidget(self.reports, 1)
        button = QPushButton("Bericht öffnen…")
        button.setToolTip(
            "Einen bestehenden Score-, Behandlungs- oder Mappingbericht schreibgeschützt öffnen; startet keine Analyse."
        )
        button.clicked.connect(self._open)
        toolbar.addWidget(button)
        export = QPushButton("Tabelle als CSV…")
        export.setToolTip(
            "Nur die angezeigten Forschungskennzahlen exportieren; keine Schwellen oder Produktionsdaten verändern."
        )
        export.clicked.connect(self._export)
        toolbar.addWidget(export)
        overview = QPushButton("Provenienz & alle Gates")
        overview.setToolTip(
            "Vollständigen Ergebnisvertrag einschließlich Unsicherheiten, fehlender Gates und Provenienz anzeigen."
        )
        overview.clicked.connect(
            lambda: self.details.setPlainText(
                json.dumps(self.payload, indent=2, ensure_ascii=False)
            )
        )
        toolbar.addWidget(overview)
        layout.addLayout(toolbar)
        self.status = QLabel("Noch kein Bericht geladen.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.formula = QLabel("Keine eingefrorene Formel ausgewählt.")
        self.formula.setWordWrap(True)
        layout.addWidget(self.formula)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Fragestellung", "Modell", "Maß", "Wert", "95-%-CI", "Patienten", "Status"]
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.currentCellChanged.connect(self._detail)
        header = self.table.horizontalHeader()
        assert header is not None
        header.setStretchLastSection(True)
        splitter.addWidget(self.table)
        plots = QSplitter()
        self.effect = pg.PlotWidget()
        self.calibration = pg.PlotWidget()
        self.effect.getAxis("bottom").enableAutoSIPrefix(False)
        self.effect.hideAxis("left")
        self.calibration.getAxis("bottom").enableAutoSIPrefix(False)
        self.calibration.getAxis("left").enableAutoSIPrefix(False)
        plots.addWidget(self.effect)
        plots.addWidget(self.calibration)
        splitter.addWidget(plots)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        splitter.addWidget(self.details)
        splitter.setSizes([250, 220, 120])
        layout.addWidget(splitter, 1)

    def set_run_root(self, root):
        paths = (
            sorted(Path(root).glob("score_discovery/*/score_research_report.json"))
            if root
            else []
        )
        deep = (
            Path(root) / "deep_combinations" / "deep_combination_summary.json"
            if root
            else None
        )
        if deep and deep.is_file():
            paths.append(deep)
        if root:
            paths.extend(sorted(Path(root).glob("pre_run_extensions/*/report.json")))
        try:
            signature = (
                str(root),
                tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths),
            )
        except OSError:
            signature = None
        if signature is not None and signature == self._run_signature:
            return
        self._run_signature = signature
        self.reports.blockSignals(True)
        self.reports.clear()
        for path in paths:
            title = (
                "Kombinationssuche"
                if path.name == "deep_combination_summary.json"
                else ("Zusatzanalysen · " if path.name == "report.json"
                      else "Score-Suche · ") + path.parent.name
            )
            self.reports.addItem(title, str(path))
        self.reports.blockSignals(False)
        self.clear(
            "Keine neuen Score-Ergebnisse für diesen Run. Alte Pilotresultate enthalten diese Suche gegebenenfalls nicht."
        )
        if paths:
            self._selection(0)

    def clear(self, message):
        self.rows = []
        self.payload = {}
        self.table.setRowCount(0)
        self.effect.clear()
        self.calibration.clear()
        self.details.clear()
        self.status.setText(message)
        self.formula.setText("Keine eingefrorene Formel ausgewählt.")

    def _selection(self, index):
        path = self.reports.itemData(index)
        if path:
            self.load_report(Path(path))

    def load_report(self, path):
        self.clear("Lade Forschungsbericht…")
        try:
            if Path(path).stat().st_size > 32 * 1024**2:
                raise ValueError(
                    "Bericht zu groß für die interaktive Übersicht (32 MiB)"
                )
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if "deep_combination_version" in payload:
                research_rows(payload)  # Validate scope before reading companion data.
                profile = Path(path).parent / "depth_profile.parquet"
                if profile.is_file():
                    if profile.stat().st_size > 32 * 1024**2:
                        raise ValueError(
                            "Tiefenprofil zu groß für die interaktive Übersicht"
                        )
                    payload["depth_profile_rows"] = (
                        pq.ParquetFile(profile)
                        .read(
                            columns=[
                                "endpoint_id",
                                "depth",
                                "best_confirmation_auc",
                                "best_components",
                                "evaluations",
                                "stop_reason",
                            ]
                        )
                        .to_pylist()
                    )
            rows = research_rows(payload)
            self.payload = payload
            self.rows = rows[:2000]
            self.table.setRowCount(len(self.rows))
            for i, row in enumerate(self.rows):
                values = [
                    row["name"],
                    row["model"],
                    row["metric"],
                    row["value"],
                    row["ci"],
                    row["n"],
                    row["status"],
                ]
                for j, value in enumerate(values):
                    text = (
                        "nicht berechnet"
                        if value is None
                        else f"{value:.4g}"
                        if isinstance(value, float)
                        else str(value)
                    )
                    self.table.setItem(i, j, QTableWidgetItem(text))
            self.table.resizeColumnsToContents()
            self.details.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
            self.status.setText(
                f"{Path(path)} · {len(rows)} Einträge · Forschungsstatus; "
                "fehlende Konfidenzintervalle und fehlende Kalibration werden nicht ergänzt."
            )
            if self.rows:
                self.table.setCurrentCell(0, 0)
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            log.warning("Research report rejected: %s", path, exc_info=True)
            self.clear("Bericht nicht darstellbar: " + str(exc))

    def _detail(self, row, *_):
        try:
            self._render_detail(row)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
            log.warning("Malformed research detail", exc_info=True)
            self.effect.clear()
            self.calibration.clear()
            self.status.setText(
                "Unvollständiger oder ungültiger Ergebnisinhalt: " + str(exc)
            )

    def _render_detail(self, row):
        self.effect.clear()
        self.calibration.clear()
        if not 0 <= row < len(self.rows):
            return
        item = self.rows[row]
        spec = item["detail"].get("score_spec", {})
        if spec:
            terms = [
                f"{c:.3g} × ({name} − {mean:.3g}) / {scale:.3g}"
                for name, c, mean, scale in zip(
                    spec.get("components", []),
                    spec.get("coefficients", []),
                    spec.get("means", []),
                    spec.get("scales", []),
                )
            ]
            self.formula.setText(
                "Eingefrorene Formel: "
                + spec.get("link", "logit")
                + " = "
                + str(spec.get("intercept", 0))
                + " + "
                + " + ".join(terms)
                + " · Fehlende Komponenten: Trainingsmediane; komplett fehlend: keine Vorhersage."
            )
        else:
            self.formula.setText(
                "Keine diagnostisch freigegebene Formel. Details und Aussagegrenzen unten."
            )
        self.details.setPlainText(
            json.dumps(item["detail"], indent=2, ensure_ascii=False)
        )
        self.effect.setTitle(item["name"] + " · " + item["model"])
        self.effect.setLabel("bottom", item["metric"])
        if item["metric"] in {"AUC", "Macro-F1", "Interne Auswahl-AUC"}:
            self.effect.setXRange(0, 1, padding=0.02)
        else:
            self.effect.enableAutoRange(axis="x")
        value = item["value"]
        if isinstance(value, (int, float)) and np.isfinite(value):
            self.effect.plot(
                [value], [0], pen=None, symbol="o", symbolSize=12, symbolBrush="#0072B2"
            )
            ci = item["ci"]
            if isinstance(ci, (list, tuple)) and len(ci) == 2 and np.isfinite(ci).all():
                self.effect.plot(ci, [0, 0], pen=pg.mkPen("#0072B2", width=3))
        bins = (
            item["detail"]
            .get("evaluation", {})
            .get("metrics", {})
            .get("calibration_bins", [])
        )
        self.calibration.setTitle(
            "Kalibration"
            if bins
            else "Kalibration nicht berechnet / für diesen Bericht nicht definiert"
        )
        if bins:
            self.calibration.setLabel("bottom", "Vorhergesagte Wahrscheinlichkeit")
            self.calibration.setLabel("left", "Beobachteter Anteil")
            self.calibration.plot(
                [0, 1], [0, 1], pen=pg.mkPen("#888888", style=Qt.PenStyle.DashLine)
            )
            self.calibration.plot(
                [b["predicted"] for b in bins],
                [b["observed"] for b in bins],
                pen=pg.mkPen("#0072B2"),
                symbol="o",
            )
            for b in bins:
                ci = b.get("observed_ci95")
                if (
                    isinstance(ci, (list, tuple))
                    and len(ci) == 2
                    and np.isfinite(ci).all()
                ):
                    self.calibration.plot(
                        [b["predicted"]] * 2, ci, pen=pg.mkPen("#0072B2", width=2)
                    )

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Forschungsbericht (nur Lesen)", "", "JSON (*.json)"
        )
        if path:
            self.load_report(Path(path))

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Angezeigte Tabelle exportieren", "research.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        item.text()
                        if (item := self.table.horizontalHeaderItem(i)) is not None
                        else ""
                        for i in range(7)
                    ]
                )
                for row in range(self.table.rowCount()):
                    writer.writerow(
                        [
                            item.text()
                            if (item := self.table.item(row, col)) is not None
                            else ""
                            for col in range(7)
                        ]
                    )
        except OSError as exc:
            self.status.setText("Export fehlgeschlagen: " + str(exc))


class CaseResearchStatusWidget(QPlainTextEdit):
    """Never apply a cohort model to an unverified laboratory feature generation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.set_case_payload({})

    def set_case_payload(self, payload):
        lines = [
            "Forschung – keine Diagnose- oder Therapieempfehlung.",
            "Kohorten-AUC ist keine Wahrscheinlichkeit für diesen Patienten.",
            "Neue Scores werden nicht ohne nachgewiesene Feature-/Einheitenkompatibilität angewendet.",
            "",
        ]
        for key, label in (
            ("score_evaluation", "Eingefrorener Score"),
            ("localization_result", "Ursprungsregion"),
        ):
            value = payload.get(key)
            if not isinstance(value, dict):
                lines.append(
                    label
                    + ": nicht berechnet / kein kompatibles Modellergebnis vorhanden."
                )
            elif value.get("clinical_claim_allowed") is not False:
                lines.append(
                    label
                    + ": Ergebnisvertrag nicht verifiziert; keine Anzeige als Befund."
                )
            else:
                lines.append(
                    label + ":\n" + json.dumps(value, indent=2, ensure_ascii=False)
                )
        lines += [
            "",
            "Therapieeffekte sind Kohortenschätzungen, keine individuelle Behandlungsempfehlung.",
            "Fokuslokalisation braucht eine unabhängige Mappingreferenz; keine Millimeter-/Oktantenposition.",
        ]
        self.setPlainText("\n".join(lines))
