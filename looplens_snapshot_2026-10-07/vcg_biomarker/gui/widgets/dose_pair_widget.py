"""Projected dose-pair view, no full marker matrix or refitting in the GUI."""
import logging
from concurrent.futures import Future
from threading import Thread
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyqtgraph as pg
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import QEvent, QTimer, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QApplication

from ...mimic.patient_identity import known_patient_mask

LOGGER = logging.getLogger(__name__)
KEYS = ("drug_class", "dose_itemid", "dose_unit")
REQUIRED = {*KEYS, "subject_id", "dose_amount", "dose_rank", "dose_rank_status"}
DISPLAY_DB_CONFIG = {"threads": 1, "memory_limit": "256MB", "max_temp_directory_size": "0B"}


def _display_signature(path):
    """Cheap display invalidation only, never scientific provenance validation."""
    path = Path(path).resolve()
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def dose_display_sample(frame, marker, *, rank=False, limit=6000):
    """Counts precede deterministic display sampling; no cross-drug pooling."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Positive display limit required")
    x, y = "dose_rank" if rank else "dose_amount", f"delta_{marker}"
    info = dict(n_total=len(frame), n_valid=0, n_patients=0, n_unknown_patient_rows=0, n_shown=0)
    if not (REQUIRED | {y}) <= set(frame):
        raise ValueError("Incomplete dose-pair contract")
    if len(frame[list(KEYS)].drop_duplicates()) > 1:
        raise ValueError("Select exactly one drug-class/item/unit group")
    numeric = frame[[x, y]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    valid &= frame.dose_rank_status.eq("eligible").fillna(False).to_numpy(dtype=bool)
    valid &= numeric[x].gt(0).fillna(False).to_numpy(dtype=bool)
    if rank:
        valid &= numeric[x].le(1).fillna(False).to_numpy(dtype=bool)
    result = numeric.loc[valid].copy()
    # Keep identifiers attached through filtering AND display sampling. Never
    # resolve a clicked point by its floating-point coordinates or row position.
    for name in ("subject_id", "first_study_id", "second_study_id"):
        if name in frame:
            result[name] = frame.loc[valid, name].to_numpy()
    patients = frame.loc[valid, "subject_id"]
    known = known_patient_mask(patients)
    info.update(n_valid=len(result), n_patients=int(patients[known].nunique()),
        n_unknown_patient_rows=int((~known).sum()))
    if len(result) > limit:
        selection = np.sort(np.random.default_rng(0).choice(len(result), limit, replace=False))
        result = result.iloc[selection]
    info["n_shown"] = len(result)
    return result, info


def _load_dose_view(kind, path, signature, group=None, marker=None):
    """Worker owns its connection; never touches Qt objects."""
    if kind == "metadata":
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows == 0:
            return [], []
        schema = set(parquet.schema_arrow.names)
        if not REQUIRED <= schema:
            raise ValueError("Inkompatibler Dosis-Paarvertrag")
        with duckdb.connect(config=DISPLAY_DB_CONFIG) as con:
            groups = con.execute('''SELECT DISTINCT drug_class, dose_itemid, dose_unit
                FROM read_parquet(?) WHERE drug_class IS NOT NULL AND dose_itemid IS NOT NULL
                AND dose_unit IS NOT NULL ORDER BY 1, 2, 3''', [str(path)]).fetchall()
        result = (groups, [name[6:] for name in sorted(schema) if name.startswith("delta_")])
    else:
        column = '"' + ("delta_"+marker).replace('"', '""') + '"'
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        identity = ", ".join(name if name in schema else f"NULL AS {name}"
            for name in ("first_study_id", "second_study_id"))
        with duckdb.connect(config=DISPLAY_DB_CONFIG) as con:
            frame = con.execute(f'''SELECT drug_class, dose_itemid, dose_unit, subject_id,
                dose_amount, dose_rank, dose_rank_status, {column}, {identity}
                FROM read_parquet(?) WHERE drug_class=? AND dose_itemid=? AND dose_unit=?''',
                [str(path), *group]).df()
        result = {mode: dose_display_sample(frame, marker, rank=mode) for mode in (False, True)}
    if signature != _display_signature(path):
        raise ValueError("Dosisdatei beim Lesen geaendert; erneut laden")
    return result


def _run_load(future, arguments):
    try:
        future.set_result(_load_dose_view(*arguments))
    except Exception as exc:
        # Worker exceptions are delivered as explicit GUI error, never swallowed.
        LOGGER.exception("Dose-view background read failed")
        future.set_exception(exc)


class DosePairWidget(QWidget):
    open_ecg_requested = pyqtSignal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.counts = {}
        self._source_signature = None
        self._sample_key = None
        self._samples = {}
        self._generation = 0
        self._active = None
        self._pending = None
        self._load_timer = QTimer(self)
        self._load_timer.setInterval(20)
        self._load_timer.timeout.connect(self._poll_load)
        layout = QVBoxLayout(self)
        self.note = QLabel("Retrospektive Dosisassoziation, keine Kausalitaet. Eine Gruppe je Substanz-ID und Einheit. "
            "Dosisrang ist ein relativer Rang innerhalb der Substanz/Einheit, keine aequivalente Arzneimitteldosis. "
            "Counts gelten nach Paarselektion; Ausschluesse davor sind hier nicht erfasst.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        controls = QHBoxLayout()
        self.group = QComboBox()
        self.marker = QComboBox()
        self.scale = QComboBox()
        self.group.setAccessibleName("Arzneimittelklasse, Substanz-ID und Einheit")
        self.marker.setAccessibleName("Dosisantwort-Marker")
        self.scale.setAccessibleName("Dosisachse")
        self.scale.addItem("Dokumentierte Dosis", False)
        self.scale.addItem("Relativer Dosisrang", True)
        for control in (self.group, self.marker, self.scale):
            controls.addWidget(control)
        layout.addLayout(controls)
        self.status = QLabel("Keine Dosis-Paardaten vorhanden.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.selected_pair = None
        self.pair_detail = QLabel("Punkt anklicken: Kennungen des EKG-Paars fuer die Laborpruefung.")
        self.pair_detail.setWordWrap(True)
        layout.addWidget(self.pair_detail)
        self.copy_pair = QPushButton("Ausgewaehlte Paar-Kennungen kopieren")
        self.copy_pair.setToolTip("Patienten- und beide Studienkennungen des eindeutig ausgewaehlten Punkts lokal in die Zwischenablage kopieren.")
        self.copy_pair.setEnabled(False)
        self.copy_pair.clicked.connect(self._copy_pair)
        layout.addWidget(self.copy_pair)
        self.open_pair_buttons = []
        for side, label in (("first", "Erstes EKG im Labor oeffnen"), ("second", "Zweites EKG im Labor oeffnen")):
            button = QPushButton(label)
            button.setToolTip("Ausgewaehltes Development-EKG nach Patienten-, Zeit- und Pfadpruefung in die lokale Fallbibliothek importieren; keine automatische Analyse.")
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, selected_side=side: self._request_ecg(selected_side))
            layout.addWidget(button)
            self.open_pair_buttons.append(button)
        self.plot = pg.PlotWidget()
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot.showGrid(x=True, y=True, alpha=.15)
        layout.addWidget(self.plot)
        self._palette()
        for control in (self.group, self.marker, self.scale):
            control.currentIndexChanged.connect(self.draw)

    def _palette(self):
        self.plot.setBackground(self.palette().color(QPalette.ColorRole.Base))
        for axis in ("bottom", "left"):
            self.plot.getAxis(axis).setTextPen(self.palette().color(QPalette.ColorRole.Text))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "plot"):
            self._palette()

    def set_source(self, path):
        try:
            signature = _display_signature(path) if path is not None else None
        except OSError:
            signature = None
        if signature is not None and signature == self._source_signature and self.source is not None:
            return  # Periodic refresh must not reread data or reset selection.
        self._generation += 1
        self._pending = None
        self._source_signature = None
        self._sample_key = None
        self._samples = {}
        self.source = None
        self.group.blockSignals(True)
        self._clear_pair_selection()
        self.marker.blockSignals(True)
        self.group.clear()
        self.marker.clear()
        self.plot.clear()
        self.plot.setLabel("bottom", "Dosis")
        self.plot.setLabel("left", "Markeraenderung")
        self.counts = {}
        self.status.setText("Keine Dosis-Paardaten vorhanden.")
        self.group.blockSignals(False)
        self.marker.blockSignals(False)
        if signature is not None:
            self.source = Path(path)
            self._source_signature = signature
            self.status.setText("Dosisgruppen werden im Hintergrund geladen ...")
            self._submit_load(("metadata", self.source, signature))

    @property
    def loading(self):
        return self._active is not None or self._pending is not None

    def _submit_load(self, arguments):
        request = (self._generation, arguments)
        if self._active is not None:
            if self._active[0] != request:
                self._pending = request  # Latest request only, no unbounded queue.
            else:
                self._pending = None
        else:
            future = Future()
            self._active = (request, future)
            Thread(target=_run_load, args=(future, arguments), daemon=True,
                   name="dose-view-read").start()
        self._load_timer.start()

    def _poll_load(self):
        if self._active is not None and self._active[1].done():
            (generation, arguments), future = self._active
            self._active = None
            if generation == self._generation and self._pending is None:
                try:
                    result = future.result()
                    if arguments[0] == "metadata":
                        groups, markers = result
                        self.group.blockSignals(True)
                        self.marker.blockSignals(True)
                        for key in groups:
                            self.group.addItem(f"{key[0]} | Item {key[1]} | {key[2]}", key)
                        for marker in markers:
                            self.marker.addItem(marker, marker)
                        self.group.blockSignals(False)
                        self.marker.blockSignals(False)
                        if not groups or not markers:
                            self.source = None
                            self.status.setText("Keine Dosis-Paardaten vorhanden.")
                        else:
                            self.draw()
                    else:
                        self._sample_key = (arguments[2], tuple(arguments[3]), arguments[4])
                        self._samples = result
                        self.draw()
                except Exception as exc:
                    LOGGER.exception("Dose-view result could not be applied")
                    self._samples = {}
                    self._sample_key = None
                    self._source_signature = None
                    self.status.setText(f"Dosis-Paardaten nicht lesbar: {exc}")
            if self._pending is not None:
                generation, arguments = self._pending
                self._pending = None
                if generation == self._generation:
                    self._submit_load(arguments)
        if not self.loading:
            self._load_timer.stop()

    def closeEvent(self, event):
        self._generation += 1
        self._pending = None
        self._load_timer.stop()
        # Read-only daemon worker has no QWidget reference and cannot write UI.
        super().closeEvent(event)

    def draw(self, *_):
        self._clear_pair_selection()
        self.plot.clear()
        self.plot.setLabel("bottom", "Dosis")
        self.plot.setLabel("left", "Markeraenderung")
        self.counts = {}
        if self.source is None or self.group.currentData() is None or self.marker.currentData() is None:
            return
        marker = self.marker.currentData()
        group = self.group.currentData()
        rank = self.scale.currentData()
        try:
            signature = _display_signature(self.source)
            if signature != self._source_signature:
                raise ValueError("Dosisdatei geaendert; Ergebnisansicht aktualisieren")
            key = (signature, tuple(group), marker)
            if key != self._sample_key:
                self._samples = {}
                self._sample_key = None
                self.status.setText("Dosis-Paare werden im Hintergrund geladen ...")
                self._submit_load(("sample", self.source, signature, tuple(group), marker))
                return
            points, info = self._samples[bool(rank)]
        except (OSError, ValueError, duckdb.Error) as exc:
            self._samples = {}
            self._sample_key = None
            LOGGER.exception("Dose-pair plot failed")
            self.status.setText(f"Dosisgrafik nicht berechenbar: {exc}")
            return
        self.counts = info
        self.status.setText(f"{info['n_shown']} dargestellt / {info['n_valid']} auswertbar / {info['n_total']} Paare; "
            f"{info['n_patients']} bekannte Patienten, {info['n_unknown_patient_rows']} auswertbare Paare ohne Kennung. "
            "Nur Darstellung; keine Neuberechnung von Modellen auf der Grafikstichprobe.")
        self.plot.setLabel("bottom", "Relativer Dosisrang (0 bis 1)" if rank else f"Dokumentierte Dosis [{group[2]}]")
        self.plot.setLabel("left", f"Aenderung {marker}")
        if not points.empty:
            records = points.to_dict("records")
            item = self.plot.plot(points.iloc[:, 0].to_numpy(), points.iloc[:, 1].to_numpy(), pen=None,
                symbol="o", symbolSize=5, symbolBrush="#56B4E9", symbolPen=None, data=records)
            item.scatter.sigClicked.connect(
                lambda scatter, clicked, event, selected_key=key:
                    self._point_clicked(scatter, clicked, event, selected_key=selected_key))

    def _clear_pair_selection(self):
        self.selected_pair = None
        self.copy_pair.setEnabled(False)
        for button in self.open_pair_buttons:
            button.setEnabled(False)
        self.pair_detail.setText("Punkt anklicken: Kennungen des EKG-Paars fuer die Laborpruefung.")

    def _point_clicked(self, _scatter, points, _event=None, *, selected_key=None):
        self._clear_pair_selection()
        if selected_key is not None and selected_key != self._sample_key:
            return
        try:
            if self.source is None or _display_signature(self.source) != self._source_signature:
                self.pair_detail.setText("Quelldatei geaendert; Paar-Auswahl erst nach Aktualisierung.")
                return
        except OSError:
            self.pair_detail.setText("Quelldatei nicht mehr verfuegbar.")
            return
        if len(points) != 1:
            self.pair_detail.setText("Mehrere ueberlagerte Punkte: keine eindeutige Paar-Auswahl. Bitte hineinzoomen.")
            return
        record = points[0].data()
        names = ("subject_id", "first_study_id", "second_study_id")
        if not isinstance(record, dict) or not known_patient_mask(pd.Series([record.get(name) for name in names])).all():
            self.pair_detail.setText("Paar-Kennungen fehlen; keine Zuordnung aus Punktposition oder Markerwert erraten.")
            return
        self.selected_pair = {name: record[name] for name in names}
        self.pair_detail.setText("Patient: {subject_id} | Erstes EKG: {first_study_id} | Zweites EKG: {second_study_id}. "
            "Nur Identifikation, noch kein automatischer Rohsignalimport.".format(**self.selected_pair))
        self.copy_pair.setEnabled(True)
        for button in self.open_pair_buttons:
            button.setEnabled(True)

    def _request_ecg(self, side):
        if self.selected_pair is not None:
            try:
                if _display_signature(self.source) != self._source_signature:
                    self._clear_pair_selection()
                    return
            except OSError:
                self._clear_pair_selection()
                return
            self.open_ecg_requested.emit(dict(self.selected_pair), side)

    def _copy_pair(self):
        if self.selected_pair is not None:
            QApplication.clipboard().setText("\n".join(f"{key}: {value}" for key, value in self.selected_pair.items()))
