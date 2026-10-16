"""Clinical-question-first paired lead effects, with explicit missing intervals."""
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QAbstractItemView

from ...core.data_loader import LEAD_NAMES_12


def _finite_number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _rank_lead_results(frame):
    """Descriptive held-out AUC rank, never a new training selection."""
    result = frame.copy()
    auc = result.get("validation_auc", pd.Series(index=result.index, dtype=float)).map(_finite_number)
    auc = pd.to_numeric(auc, errors="coerce")
    valid = result.status.eq("estimable") & auc.between(0, 1)
    result["validation_auc_rank"] = auc.where(valid).rank(method="min", ascending=False)
    return result.sort_values(["validation_auc_rank", "lead"], na_position="last", kind="stable")


class LeadComparisonWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.note = QLabel("Noch keine Ableitungsvergleiche berechnet.")
        self.note.setWordWrap(True); layout.addWidget(self.note)
        bar = QHBoxLayout()
        self.endpoint = QComboBox(); self.family = QComboBox()
        self.endpoint.setAccessibleName("Klinischer Endpunkt")
        self.family.setAccessibleName("Markerfamilie für den ausgewählten Endpunkt")
        bar.addWidget(self.endpoint, 3); bar.addWidget(self.family, 1); layout.addLayout(bar)
        self.comparator = QComboBox()
        self.comparator.setAccessibleName("Vergleichsbasis der Ableitungseffekte")
        self.comparator.addItem("Gegen klinische Kovariatenbasis", "delta_auc_clinical_baseline")
        self.comparator.addItem("Gegen im Training gewählte Referenzableitung", "delta_auc_train_anchor")
        layout.addWidget(self.comparator)
        self.summary = QLabel(); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Δ Validation-AUC gegenüber klinischer Kovariatenbasis")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.showGrid(x=True, y=False, alpha=.15)
        self.plot.getAxis("left").setWidth(250)
        layout.addWidget(self.plot, 2)
        self.table = QTableWidget(); layout.addWidget(self.table, 1)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.frame = pd.DataFrame()
        self.rendered_effects = []
        self.endpoint.currentTextChanged.connect(self.refresh_families)
        self.family.currentTextChanged.connect(self.draw)
        self.comparator.currentIndexChanged.connect(self.draw)

    def set_source(self, path: Path):
        try:
            self.set_frame(pd.read_parquet(path) if path.is_file() else pd.DataFrame())
        except (OSError, ValueError, KeyError) as exc:
            logging.getLogger(__name__).exception("Cannot load lead comparison results: %s", path)
            self.set_frame(pd.DataFrame())
            self.note.setText(f"Ableitungsvergleich nicht lesbar: {exc}")

    def set_frame(self, frame: pd.DataFrame):
        required = {"endpoint_id", "family", "lead", "status", "n_validation_common", "reference_kind"}
        if not frame.empty and not required <= set(frame):
            raise ValueError("Inkompatibler Ableitungs-Ergebnisvertrag")
        if not frame.empty and frame.duplicated(["endpoint_id", "family", "lead"]).any():
            raise ValueError("Mehrdeutige Kohorten/Modelle: keine automatische Zusammenführung")
        if not frame.empty and not frame.lead.isin(LEAD_NAMES_12).all():
            raise ValueError("Unbekannte Ableitung im Ergebnisvertrag")
        self.frame = frame.copy()
        previous = self.endpoint.currentText()
        self.endpoint.blockSignals(True); self.endpoint.clear()
        if not frame.empty:
            self.endpoint.addItems(sorted(frame.endpoint_id.astype(str).unique()))
            if self.endpoint.findText(previous) >= 0:
                self.endpoint.setCurrentText(previous)
        self.endpoint.blockSignals(False)
        self.refresh_families()

    def refresh_families(self):
        previous = self.family.currentText()
        self.family.blockSignals(True); self.family.clear()
        if not self.frame.empty:
            scoped = self.frame.loc[self.frame.endpoint_id.eq(self.endpoint.currentText())]
            self.family.addItems(sorted(scoped.family.astype(str).unique()))
            if self.family.findText(previous) >= 0:
                self.family.setCurrentText(previous)
        self.family.blockSignals(False)
        self.draw()

    def draw(self):
        self.plot.clear(); self.table.setRowCount(0); self.table.setColumnCount(0)
        self.plot.getAxis("left").setTicks([])
        self.summary.clear(); self.rendered_effects = []
        if self.frame.empty:
            self.note.setText("Keine berechneten Ableitungsvergleiche. Historische Pilotmessungen werden nicht ergänzt oder erfunden.")
            return
        selected = self.frame.loc[self.frame.endpoint_id.eq(self.endpoint.currentText()) & self.frame.family.eq(self.family.currentText())]
        if selected.empty:
            self.note.setText("Für diese Auswahl liegen keine Ergebnisse vor.")
            return
        effect_column = self.comparator.currentData()
        anchor_view = effect_column == "delta_auc_train_anchor"
        # Non-estimable fits legitimately carry no anchor in the backend.
        # They must not hide valid comparisons from the remaining leads.
        estimable = selected.loc[selected.status.eq("estimable")]
        anchor_values = estimable.get("anchor_selected_in_train", pd.Series(dtype=str)).dropna().astype(str)
        anchors = anchor_values.unique()
        anchor_valid = len(anchor_values) == len(estimable) and len(anchors) == 1 and anchors[0] in LEAD_NAMES_12
        self.plot.setLabel("bottom", "Δ Validation-AUC gegenüber " + (
            f"Trainings-Referenzableitung {anchors[0]}" if anchor_view and anchor_valid else
            "Trainings-Referenzableitung (nicht eindeutig dokumentiert)" if anchor_view else
            "klinischer Kovariatenbasis"))
        self.note.setText("Explorative, patientengetrennte Modelle mit gleichen Vergleichspatienten. "
            "AUC bewertet klinische Basis + Ableitung; der Zusatznutzen steht separat als ΔAUC. "
            "AUC ist keine Trefferquote und keine Wahrscheinlichkeit einer Erkrankung. "
            "95%-Bootstrap-Intervalle sind nicht multiplikitätsadjustiert. q-Werte testen die bedingte Assoziation, nicht den AUC-Gewinn. "
            "Keine Freigabe zur Stations-Ableitungswahl.\n"
            + "Referenz: " + "; ".join(selected.reference_kind.dropna().astype(str).unique()))
        self.plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#888888", style=Qt.PenStyle.DashLine)))
        ticks = []
        unavailable = 0
        missing_intervals = 0
        for index, lead in enumerate(LEAD_NAMES_12):
            y = len(LEAD_NAMES_12)-1-index
            match = selected.loc[selected.lead.eq(lead)]
            if match.empty:
                ticks.append((y, f"{lead} — kein Ergebnis"))
                continue
            row = match.iloc[0]
            value = _finite_number(row.get(effect_column)) if not anchor_view or anchor_valid else None
            n = _finite_number(row.get("n_validation_common"))
            label = f"{lead} · n={int(n)}" if n is not None and n >= 0 else f"{lead} · n unbekannt"
            if row.status != "estimable" or value is None or not -1 <= value <= 1:
                ticks.append((y, label + " · nicht schätzbar")); unavailable += 1
                continue
            ticks.append((y, label))
            low = _finite_number(row.get(effect_column + "_ci_low"))
            high = _finite_number(row.get(effect_column + "_ci_high"))
            interval_ok = low is not None and high is not None and -1 <= low <= high <= 1
            if interval_ok:
                self.plot.plot([low, high], [y, y], pen=pg.mkPen("#0072B2", width=2))
            else:
                missing_intervals += 1
            color = "#0072B2" if interval_ok else "#D55E00"
            self.plot.plot([value], [y], pen=None, symbol="o" if interval_ok else "t", symbolBrush=color,
                           symbolPen=pg.mkPen(color))
            self.rendered_effects.append({"lead": lead, "value": value, "interval_drawn": interval_ok})
        self.summary.setText(f"{len(self.rendered_effects)}/{len(selected)} Effekte darstellbar; {unavailable} nicht schätzbar/ungültig. "
            f"Blau: vollständiges 95-%-KI. Orange/Dreieck: {missing_intervals} ohne vollständiges gültiges KI. "
            "Fehlende Ergebnisse sind keine neutralen Effekte.")
        if anchor_view:
            self.summary.setText(self.summary.text() + (
                " Positiv: besser als die Trainings-Referenz, negativ: schlechter; keine Neuauswahl in Validation."
                if anchor_valid else " Referenzableitung fehlt oder ist widersprüchlich; keine Effekte dargestellt."
            ))
        self.plot.getAxis("left").setTicks([ticks]); self.plot.setYRange(-1, 12, padding=0)
        selected = _rank_lead_results(selected)
        self.summary.setText(self.summary.text() + " Tabelle: deskriptiver Rang nach Validation-AUC innerhalb dieser Pathologie/Markerfamilie; Gleichstände teilen einen Rang. Keine bewiesene Überlegenheit bei überlappender Unsicherheit.")
        columns = [name for name in ("validation_auc_rank", "lead", "status", "n_train_common", "n_validation_common", "n_validation_positive",
            "n_validation_negative", "validation_auc", "validation_auc_ci_low", "validation_auc_ci_high", "delta_auc_clinical_baseline", "delta_auc_clinical_baseline_ci_low",
            "delta_auc_clinical_baseline_ci_high", "n_lead_measured", "anchor_selected_in_train", "abstention_reason",
            "delta_auc_train_anchor", "delta_auc_train_anchor_ci_low", "delta_auc_train_anchor_ci_high",
            "n_index", "n_excluded_reference", "n_excluded_extraction_after_reference",
            "n_excluded_covariates_after_extraction", "n_excluded_leads_after_covariates", "n_common_complete",
            "confirmation_status", "confirmation_p_value", "confirmation_q_global_bh", "confirmation_q_domain_bh",
            "confirmation_q_global_by", "confirmation_direction_preserved", "confirmation_reason") if name in selected]
        labels = {"validation_auc_rank": "Rang (Validation-AUC)", "lead": "Ableitung", "status": "Status", "n_train_common": "Training n",
            "n_index": "Indexpatienten", "n_excluded_reference": "Referenz fehlt/ungeeignet",
            "n_excluded_extraction_after_reference": "Danach: Extraktion fehlt",
            "n_excluded_covariates_after_extraction": "Danach: Kovariaten fehlen",
            "n_excluded_leads_after_covariates": "Danach: Ableitungswerte fehlen",
            "n_common_complete": "Gemeinsam vollständig",
            "n_validation_common": "Validation n", "n_validation_positive": "Fälle", "n_validation_negative": "Kontrollen",
            "validation_auc": "AUC", "delta_auc_clinical_baseline": "ΔAUC",
            "anchor_selected_in_train": "Trainings-Referenz",
            "delta_auc_train_anchor": "ΔAUC zur Referenz",
            "delta_auc_train_anchor_ci_low": "Referenzvergleich: KI unten",
            "delta_auc_train_anchor_ci_high": "Referenzvergleich: KI oben",
            "delta_auc_clinical_baseline_ci_low": "95-%-KI unten", "delta_auc_clinical_baseline_ci_high": "95-%-KI oben",
            "n_lead_measured": "Messbar n", "abstention_reason": "Abstention-Grund"}
        self.table.setColumnCount(len(columns)); self.table.setHorizontalHeaderLabels([labels.get(name, name) for name in columns])
        for column, name in enumerate(columns):
            self.table.horizontalHeaderItem(column).setToolTip(name)
            if name in {"validation_auc_ci_low", "validation_auc_ci_high"}:
                self.table.horizontalHeaderItem(column).setText(
                    "AUC: 95-%-KI " + ("unten" if name.endswith("low") else "oben"))
        self.table.setRowCount(len(selected))
        for index, (_, row) in enumerate(selected.iterrows()):
            for column, name in enumerate(columns):
                value = row[name]
                if pd.isna(value):
                    display = "nicht berechnet"
                elif isinstance(value, (float, np.floating)):
                    display = f"{value:.4g}"
                elif name == "status":
                    display = {"estimable": "schätzbar", "non_estimable": "nicht schätzbar"}.get(str(value), str(value))
                else:
                    display = str(value)
                item = QTableWidgetItem(display); item.setToolTip(str(value))
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()
