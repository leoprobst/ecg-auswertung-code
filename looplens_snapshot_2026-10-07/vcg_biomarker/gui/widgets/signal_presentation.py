"""Read-only presentation of measured ECG, derived VCG and modeled activation.

The three panes share one *signal-time* clock. No analysis or annotation is
started here; all inputs come from the already loaded ECG laboratory case.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QTabWidget, QVBoxLayout, QWidget)

from ..heart3d.widgets import AnimationClock, HeartGLView, RhythmStrip
from ..motion import current_motion_policy
from ..vcglab.widgets import Loop3DWidget, scene_for


LEAD_GRID = (("I", "aVR", "V1", "V4"),
             ("II", "aVL", "V2", "V5"),
             ("III", "aVF", "V3", "V6"))
INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


def presentation_camera_scale(width: int, height: int) -> float:
    """Keep the vertical field of view usable in unusually wide 3D panes."""
    if width <= 0 or height <= 0:
        return 1.0
    return max(1.0, (width / height) / 2.23)


def vector_bridge_values(record, seconds: float):
    """Recorded algorithm input and derived, baseline-corrected Kors vector at one sample."""
    if record.fs <= 0 or len(record.ecg) == 0:
        return None
    names = tuple(record.lead_names)
    if not all(name in names for name in INDEPENDENT_LEADS):
        return None
    sample = int(np.clip(round(seconds * record.fs), 0, len(record.ecg) - 1))
    raw = np.asarray(record.ecg[sample, [names.index(name) for name in INDEPENDENT_LEADS]], dtype=float)
    xyz = np.asarray(record.xyz("kors", corrected=True)[sample], dtype=float)
    if not np.isfinite(raw).all() or not np.isfinite(xyz).all():
        return None
    return sample, raw, xyz


def beat_constellation_points(record) -> list[dict]:
    """Descriptive per-beat QRS vector peaks; no rhythm diagnosis is inferred."""
    xyz = np.asarray(record.xyz("kors", corrected=True), dtype=float)
    points = []
    for beat in record.beats:
        start = max(0, min(len(xyz), int(beat.qrs_on)))
        end = max(start, min(len(xyz), int(beat.qrs_off)))
        segment = xyz[start:end]
        norms = np.linalg.norm(segment, axis=1) if len(segment) else np.empty(0)
        finite = norms[np.isfinite(norms)]
        points.append({"index": beat.index, "time": beat.r / record.fs,
                       "peak_mv": float(np.max(finite)) if len(finite) else None,
                       "cluster": beat.cluster, "accepted": beat.accepted,
                       "outlier": beat.outlier, "beat_type": beat.beat_type})
    return points


class BeatConstellationPanel(QWidget):
    """Clickable beat map linking actual beat windows to the shared presentation clock."""

    seekRequested = pyqtSignal(float)

    def __init__(self, record, *, dark: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.points = beat_constellation_points(record)
        self._by_index = {item["index"]: item for item in self.points}
        root = QVBoxLayout(self)
        self.caption = QLabel("Jeder Punkt ist ein tatsächlich abgegrenzter Schlag · Höhe: Maximum der Kors-QRS-Vektornorm (mV)")
        self.caption.setWordWrap(True)
        root.addWidget(self.caption)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#10161e" if dark else "#ffffff")
        self.plot.setLabel("bottom", "R-Zeit", units="s")
        self.plot.setLabel("left", "QRS-Vektornorm", units="mV")
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMinimumHeight(150)
        root.addWidget(self.plot, 1)
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#91a8b5", width=1))
        self.plot.addItem(self.cursor)
        palette = ("#4fd1c5", "#ffbc55", "#a78bfa", "#77bdfb", "#f472b6", "#a3c56c")
        spots = []
        for item in self.points:
            cluster = int(item["cluster"])
            rejected = not item["accepted"] or item["outlier"]
            spots.append({"pos": (item["time"], item["peak_mv"] or 0.0), "data": item["index"],
                          "symbol": "x" if rejected else "o", "size": 15 if rejected else 12,
                          "brush": pg.mkBrush(palette[cluster % len(palette)] if not rejected else "#667785"),
                          "pen": pg.mkPen("#e8f1f5" if dark else "#17313d", width=1.4)})
        self.scatter = pg.ScatterPlotItem(spots=spots, hoverable=True)
        self.scatter.sigClicked.connect(self._clicked)
        self.plot.addItem(self.scatter)
        maximum = max((item["peak_mv"] or 0.0 for item in self.points), default=0.0)
        self.plot.setYRange(0.0, max(0.2, maximum * 1.25), padding=0)
        self.plot.setXRange(0.0, len(record.ecg) / record.fs, padding=0)
        self.detail = QLabel("Punkt anklicken: denselben Schlag in EKG, VCG und Modell anspringen. "
                             "Farbe = technischer Morphologiecluster; × = ausgeschlossen/Ausreißer. "
                             "Cluster und Schlagtyp sind keine gesicherte Rhythmusdiagnose.")
        self.detail.setWordWrap(True)
        root.addWidget(self.detail)

    def _clicked(self, _scatter, points, _event) -> None:
        if points:
            item = self._by_index.get(points[0].data())
            if item is not None:
                self.seekRequested.emit(float(item["time"]))

    def set_time(self, seconds: float) -> None:
        self.cursor.setPos(seconds)
        if not self.points:
            self.detail.setText("Keine abgegrenzten Schläge vorhanden; keine Schlagkarte berechenbar.")
            return
        item = min(self.points, key=lambda value: abs(value["time"] - seconds))
        status = "ausgeschlossen/Ausreißer" if not item["accepted"] or item["outlier"] else "verwendbar"
        peak = "nicht messbar" if item["peak_mv"] is None else f'{item["peak_mv"]:.3f} mV'
        self.detail.setText(f'Schlag #{item["index"] + 1} · R {item["time"]:.2f} s · '
                            f'Cluster {item["cluster"]} · Typ {item["beat_type"]} · {status} · '
                            f'QRS-Vektormaximum {peak}. '
                            'Cluster/Typ sind technische Zuordnungen, keine gesicherte Rhythmusdiagnose.')


def uncertainty_lens_data(record) -> tuple[tuple[str, str, str], ...]:
    """Separate observed availability from clinical and model uncertainty."""
    signal = np.asarray(record.ecg)
    finite = int(np.count_nonzero(np.isfinite(signal)))
    total = int(signal.size)
    beats = list(record.beats)
    accepted = sum(bool(beat.accepted) and not bool(beat.outlier) for beat in beats)
    p_bounds = sum(beat.p_on is not None and beat.p_off is not None for beat in beats)
    t_bounds = sum(beat.t_on is not None and beat.t_end is not None for beat in beats)
    quality = record.payload.get("delineation", {}).get("rhythm_context", {}).get("quality_score") if isinstance(record.payload, dict) else None
    quality_text = f" · Qualitätswert {quality:.2f} (ohne klinischen Grenzwert)" if isinstance(quality, (int, float)) and np.isfinite(quality) else ""
    return (
        ("01 · SIGNAL & AUSWAHL", f"{finite}/{total} endliche Samples · {accepted}/{len(beats)} Schläge verwendbar",
         f"Pfad: {record.source} · {record.fs:g} Hz{quality_text}. Vollständigkeit ist keine Diagnosequalität."),
        ("02 · WELLENGRENZEN", f"P: {p_bounds}/{len(beats)} · T: {t_bounds}/{len(beats)} mit beidseitigen Grenzen",
         "QRS-Indizes liegen im Beat-Record vor, können aber algorithmische Ersatzgrenzen enthalten; hier keine unabhängige Fiduzialprüfung."),
        ("03 · KLINISCHE REFERENZ", "In dieser Ansicht nicht geprüft",
         "Echo, Labor, Therapie und Uhrsynchronisierung benötigen eigene fall- und zeitgeprüfte Nachweise. Fehlend heißt unbekannt, nicht negativ."),
        ("04 · HERZMODELL", "Illustration · keine Ortswahrscheinlichkeit",
         "Generische Geometrie und Vorwärtsannahmen. Die sichtbare Aktivierung ist weder invasives Mapping noch eine patientenspezifische Rekonstruktion."),
    )


class UncertaintyLensPanel(QWidget):
    """A static, non-scored evidence lens for the currently loaded presentation."""

    def __init__(self, record, *, dark: bool, parent: QWidget | None = None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        note = QLabel("Vier verschiedene Grenzen · kein zusammengezählter Vertrauenswert. "
                      "Die Karten trennen beobachtete Verfügbarkeit von unbekannter klinischer Wahrheit.")
        note.setWordWrap(True)
        root.addWidget(note)
        grid = QGridLayout()
        self.cards = []
        for index, (heading, value, detail) in enumerate(uncertainty_lens_data(record)):
            card = QFrame(self)
            card.setObjectName("uncertaintyLensCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(10, 5, 10, 5)
            title = QLabel(heading)
            title.setStyleSheet("font-size: 11px; font-weight: 700; letter-spacing: 1px;")
            summary = QLabel(value)
            summary.setWordWrap(True)
            summary.setStyleSheet("font-size: 13px; font-weight: 600;")
            explanation = QLabel(detail)
            explanation.setWordWrap(True)
            explanation.setStyleSheet("font-size: 10px;")
            layout.addWidget(title)
            layout.addWidget(summary)
            layout.addWidget(explanation)
            grid.addWidget(card, index // 2, index % 2)
            self.cards.append(card)
        root.addLayout(grid, 1)
        if dark:
            self.setStyleSheet("QFrame#uncertaintyLensCard { background: #172733; border: 1px solid #45616e; border-radius: 5px; }")
        else:
            self.setStyleSheet("QFrame#uncertaintyLensCard { background: #ffffff; border: 1px solid #abc4ce; border-radius: 5px; }")


class VectorBridgePanel(QWidget):
    """Animated explanation of the eight-lead input and the derived Kors vector."""

    def __init__(self, record, *, dark: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self.record = record
        self._last_sample = -1
        root = QVBoxLayout(self)
        self.note = QLabel("Acht algebraisch unabhängige Ableitungen → Kors-Transformation → X/Y/Z. "
                           "VCG ist keine zweite unabhängige Messung; die Vektor-Baseline wird separat korrigiert.")
        self.note.setWordWrap(True)
        root.addWidget(self.note)
        charts = QHBoxLayout()
        self.inputs = pg.PlotWidget(background="#10161e" if dark else "#ffffff")
        self.outputs = pg.PlotWidget(background="#10161e" if dark else "#ffffff")
        charts.addWidget(self.inputs, 2)
        charts.addWidget(self.outputs, 1)
        root.addLayout(charts, 1)
        for plot in (self.inputs, self.outputs):
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.showGrid(x=True, y=False, alpha=0.15)
            plot.setLabel("bottom", "mV")
            plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.inputs.setTitle("Eingang · Analyse-Sample")
        self.outputs.setTitle("Ergebnis · baselinekorrigiert")
        self.inputs.getAxis("left").setTicks([[(i, name) for i, name in enumerate(INDEPENDENT_LEADS)]])
        self.outputs.getAxis("left").setTicks([[(i, name) for i, name in enumerate(("X links", "Y inferior", "Z posterior"))]])
        self.input_bars = pg.BarGraphItem(x0=np.zeros(8), y0=np.arange(8), width=np.zeros(8), height=0.65,
                                          brush="#4fd1c5")
        self.output_bars = pg.BarGraphItem(x0=np.zeros(3), y0=np.arange(3), width=np.zeros(3), height=0.65,
                                           brush="#5aa9ff")
        self.inputs.addItem(self.input_bars)
        self.outputs.addItem(self.output_bars)
        self.inputs.setYRange(-0.5, 8, padding=0)
        self.outputs.setYRange(-0.5, 3, padding=0)
        self.sample_note = QLabel("Noch kein Sample gewählt")
        root.addWidget(self.sample_note)
        self._input_limit = self._range(record.ecg[:, [record.lead_names.index(n) for n in INDEPENDENT_LEADS]])
        self._output_limit = self._range(record.xyz("kors", corrected=True))
        self.inputs.setXRange(-self._input_limit, self._input_limit, padding=0)
        self.outputs.setXRange(-self._output_limit, self._output_limit, padding=0)

    @staticmethod
    def _range(array) -> float:
        values = np.asarray(array, dtype=float)
        finite = np.abs(values[np.isfinite(values)])
        return max(0.1, 1.1 * float(np.max(finite))) if finite.size else 0.1

    def set_time(self, seconds: float) -> None:
        values = vector_bridge_values(self.record, seconds)
        if values is None:
            self.sample_note.setText("Acht Ableitungen oder ein gültiger Vektor fehlen · nicht auswertbar")
            return
        sample, raw, xyz = values
        if sample == self._last_sample:
            return
        self._last_sample = sample
        self.input_bars.setOpts(x0=np.minimum(raw, 0.0), width=np.abs(raw))
        self.output_bars.setOpts(x0=np.minimum(xyz, 0.0), width=np.abs(xyz))
        self.sample_note.setText(f"Analyse-Sample {sample} bei {sample / self.record.fs:.3f} s · "
                                 f"X {xyz[0]:+.3f}, Y {xyz[1]:+.3f}, Z {xyz[2]:+.3f} mV · "
                                 "positive Achsen: links / inferior / posterior")


class SynchronizedLeadGrid(pg.GraphicsLayoutWidget):
    """Simultaneous native 12-lead window; only cursor movement redraws per frame."""

    def __init__(self, timeline, *, dark: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self.timeline = timeline
        self._window: tuple[float, float] | None = None
        self.setBackground("#10161e" if dark else "#ffffff")
        self._plots = []
        self._curves = []
        self._cursors = []
        names = tuple(timeline.lead_names)
        for row, leads in enumerate(LEAD_GRID):
            for col, name in enumerate(leads):
                plot = self.addPlot(row=row, col=col, title=name)
                plot.showGrid(x=True, y=True, alpha=0.12)
                plot.setMouseEnabled(x=False, y=True)
                plot.setMenuEnabled(False)
                if col:
                    plot.hideAxis("left")
                else:
                    plot.setLabel("left", "mV")
                if row < 2:
                    plot.hideAxis("bottom")
                else:
                    plot.setLabel("bottom", "s")
                curve = plot.plot(pen=pg.mkPen("#4fd1c5" if dark else "#0f6b78", width=1.2))
                cursor = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#ffd447" if dark else "#9a6700", width=1.2))
                plot.addItem(cursor)
                self._plots.append(plot)
                self._curves.append((curve, names.index(name) if name in names else None))
                self._cursors.append(cursor)

    def set_time(self, seconds: float) -> None:
        if self._window is None or not (self._window[0] <= seconds <= self._window[1]):
            self._set_window(seconds)
        for cursor in self._cursors:
            cursor.setPos(seconds)

    def _set_window(self, seconds: float) -> None:
        timeline = self.timeline
        signal = timeline.signal
        if signal is None or timeline.fs <= 0 or not len(signal):
            return
        duration = len(signal) / timeline.fs
        start = float(np.clip(seconds - 0.4, 0.0, max(0.0, duration - 2.0)))
        end = min(duration, start + 2.0)
        lo = int(np.floor(start * timeline.fs))
        hi = min(len(signal), max(lo + 1, int(np.ceil(end * timeline.fs))))
        window = np.asarray(signal[lo:hi], dtype=float)
        t = np.arange(lo, hi) / timeline.fs
        selected = [index for _, index in self._curves if index is not None and index < window.shape[1]]
        finite = window[:, selected] if selected else np.empty((0, 0))
        valid = np.abs(finite[np.isfinite(finite)])
        amplitude = float(np.max(valid)) if valid.size else 1.0
        amplitude = max(amplitude * 1.1, 0.2)
        for plot, (curve, index) in zip(self._plots, self._curves):
            if index is None or index >= window.shape[1]:
                curve.setData([], [])
            else:
                curve.setData(t, window[:, index], connect="finite")
            plot.setXRange(start, end, padding=0)
            plot.setYRange(-amplitude, amplitude, padding=0)
        self._window = (start, end)


def beat_at_time(record, seconds: float):
    """Choose a recorded beat by sample time, without mixing morphologies."""
    if not record.beats or record.fs <= 0:
        return None
    sample = int(np.clip(round(seconds * record.fs), 0, max(len(record.ecg) - 1, 0)))
    prior = [beat for beat in record.beats if beat.r <= sample]
    return prior[-1] if prior else record.beats[0]


class SignalPresentationDialog(QDialog):
    TOUR_STEPS = (
        (0, "12-KANAL-EKG", "Links unten sind zwölf simultane Ableitungen derselben Zeitbasis zu sehen."),
        (2, "ACHT EINGÄNGE", "I, II und V1–V6 werden am angezeigten Sample in den Kors-Vektor X/Y/Z überführt."),
        (0, "VCG-SCHLEIFE", "Rechts oben ist die räumliche Darstellung aus demselben EKG; sie ist keine zweite Messung."),
        (0, "HERZMODELL", "Links oben illustriert ein generisches Modell die Aktivierung, kein invasives Mapping."),
        (4, "AUSSAGEGRENZEN", "Die vier Karten trennen Messverfügbarkeit, Wellengrenzen, Referenz und Modellannahmen."),
    )

    def __init__(self, *, engine, model, record, dark: bool = True,
                 synthetic: bool = False, public_demo: bool = False,
                 parent: QWidget | None = None):
        super().__init__(parent)
        if engine.timeline is None or record is None:
            raise ValueError("Vorführung benötigt einen geladenen Laborfall und VCG-Record")
        if synthetic and public_demo:
            raise ValueError("Demofall kann nicht gleichzeitig synthetisch und gemessen sein")
        self.setWindowTitle("Signalreise · " + ("SYNTHETISCHE DEMO" if synthetic else
                                               "ÖFFENTLICHE DEMO" if public_demo else "EKG, VCG und Modell"))
        self.resize(1380, 830)
        if dark:
            self.setStyleSheet("""
                QDialog { background: #0b111a; color: #e8f1f5; }
                QLabel { color: #dce8ee; }
                QTabWidget::pane { border: 1px solid #344655; border-radius: 5px; }
                QTabBar::tab { background: #172533; color: #b9cbd4; padding: 6px 16px; }
                QTabBar::tab:selected { background: #263c4b; color: #ffffff; }
                QPushButton { background: #203644; color: #e8f1f5; border: 1px solid #4a697a;
                              border-radius: 5px; padding: 6px 13px; }
                QPushButton:hover { background: #2c5363; }
            """)
        else:
            self.setStyleSheet("""
                QDialog { background: #f4f8fa; color: #172934; }
                QLabel { color: #172934; }
                QTabWidget::pane { border: 1px solid #abc1cb; border-radius: 5px; }
                QTabBar::tab { background: #e0ebef; color: #243f4c; padding: 6px 16px; }
                QTabBar::tab:selected { background: #ffffff; color: #123846; }
                QPushButton { background: #ffffff; color: #123846; border: 1px solid #8ba9b6;
                              border-radius: 5px; padding: 6px 13px; }
                QPushButton:hover { background: #e5f3f5; }
            """)
        self.engine = engine
        self.record = record
        self.synthetic = synthetic
        self.public_demo = public_demo
        self._tour_step = None
        self._beat_index = None
        self.clock = AnimationClock(self)
        self.clock._timer.setInterval(33)  # presentation only; cap redraw near 30 fps
        self.clock.duration = min(float(engine.timeline.duration), len(record.ecg) / record.fs)

        root = QVBoxLayout(self)
        title = QLabel("SYNTHETISCHER DEMOFALL · KEIN PATIENT" if synthetic else
                       "ÖFFENTLICHER DEMOFALL · KEIN AKTUELLER PATIENT" if public_demo else
                       "EIN SCHLAG · DREI SICHTEN")
        title.setStyleSheet("font-size: 18px; font-weight: 700; letter-spacing: 2px; padding: 5px;")
        root.addWidget(title)
        source = {"core": "Hauptanalyse", "payload": "Laboranalyse", "quick": "Schnellschätzung"}.get(record.source, record.source)
        source_kind = ("EKG: simuliertes Signal" if synthetic else
                       "EKG: öffentliches gemessenes Demo-Signal (PhysioNet Challenge 2021/PTB-XL, CC BY 4.0)"
                       if public_demo else "EKG: gemessenes Signal")
        self._base_note = (source_kind
                           + "  ·  VCG: aus demselben EKG berechnet (Kors; " + source
                           + ")  ·  Herz: Modellillustration, kein invasives Mapping")
        self.note = QLabel(self._base_note)
        self.note.setWordWrap(True)
        self.note.setStyleSheet("font-size: 12px; color: #9aa7b5; padding-bottom: 5px;" if dark
                           else "font-size: 12px; color: #526773; padding-bottom: 5px;")
        root.addWidget(self.note)

        panes = QHBoxLayout()
        self.heart = HeartGLView(model, engine, self)
        self.heart.setMinimumSize(400, 300)
        self._vcg_base_distance = None
        if self.heart.gl is not None:
            # Wide presentation panes are much shorter than the standalone heart
            # window; include atria and basal vessels instead of cropping them.
            from PyQt6.QtGui import QVector3D
            self.heart.gl.opts["center"] = QVector3D(0.0, 0.0, 25.0)
            self.heart.gl.opts["fov"] = 45
            self.heart.gl.setCameraPosition(distance=500, elevation=12, azimuth=-90)
        # Both 3D panes form one dark presentation stage, including in the
        # light theme; ECG and controls continue to follow the chosen theme.
        self.vcg = Loop3DWidget(self, compact=False, dark=True)
        self.vcg.setMinimumSize(400, 300)
        self.vcg.set_auto_rotate(False)
        panes.addWidget(self.heart, 1)
        panes.addWidget(self.vcg, 1)
        root.addLayout(panes, 1)
        captions = QHBoxLayout()
        captions.addWidget(QLabel("MODELLIERTE AKTIVIERUNG · hypothetische Mechanik"), 1)
        captions.addWidget(QLabel("ABGELEITETE VCG-SCHLEIFE · Kors, mV"), 1)
        root.addLayout(captions)

        self.ecg_tabs = QTabWidget(self)
        self.leads = SynchronizedLeadGrid(engine.timeline, dark=dark, parent=self)
        self.ecg_tabs.addTab(self.leads, "12 Ableitungen · simultan")
        self.strip = RhythmStrip(self)
        self.strip.set_timeline(engine.timeline)
        self.strip.seekRequested.connect(self.clock.seek)
        self.ecg_tabs.addTab(self.strip, "Rhythmusstreifen · II")
        self.bridge = VectorBridgePanel(record, dark=dark, parent=self)
        self.ecg_tabs.addTab(self.bridge, "EKG → VCG · Messmoment")
        self.beat_map = BeatConstellationPanel(record, dark=dark, parent=self)
        self.beat_map.seekRequested.connect(self.clock.seek)
        self.ecg_tabs.addTab(self.beat_map, "Schlagkarte · Qualität")
        self.uncertainty = UncertaintyLensPanel(record, dark=dark, parent=self)
        self.ecg_tabs.addTab(self.uncertainty, "Unsicherheitslinse")
        self.ecg_tabs.currentChanged.connect(self._tab_changed)
        self.ecg_tabs.setMinimumHeight(230)
        root.addWidget(self.ecg_tabs)
        scale_note = QLabel(("EKG: synthetische Samples" if synthetic else "EKG: Originalsamples")
                            + ", mV · 2-s-Fenster mit gemeinsamer automatischer Amplitudenskala; "
                              "keine Maßstabsbefundung in dieser Vorführansicht")
        scale_note.setWordWrap(True)
        root.addWidget(scale_note)
        controls = QHBoxLayout()
        self.play_button = QPushButton("Abspielen")
        self.play_button.clicked.connect(self.clock.toggle)
        controls.addWidget(self.play_button)
        self.draw_loop = QCheckBox("VCG zeichnen")
        self.draw_loop.setChecked(True)
        self.draw_loop.setToolTip("Zeigt nur aufgezeichnete VCG-Samples bis zum Zeitcursor; außerhalb eines Schlags die vollständige Schleife.")
        self.draw_loop.toggled.connect(self.vcg.set_trace_reveal)
        self.vcg.set_trace_reveal(True)
        controls.addWidget(self.draw_loop)
        self.tour_button = QPushButton("Geführte Tour")
        self.tour_button.clicked.connect(self._toggle_tour)
        controls.addWidget(self.tour_button)
        self.tour_prev = QPushButton("Zurück")
        self.tour_prev.clicked.connect(lambda: self._set_tour_step(self._tour_step - 1))
        self.tour_prev.setVisible(False)
        controls.addWidget(self.tour_prev)
        self.tour_next = QPushButton("Weiter")
        self.tour_next.clicked.connect(self._advance_tour)
        self.tour_next.setVisible(False)
        controls.addWidget(self.tour_next)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, max(0, int(round(self.clock.duration * 1000))))
        self.seek.valueChanged.connect(lambda value: self.clock.seek(value / 1000.0))
        controls.addWidget(self.seek, 1)
        self.time_label = QLabel("0,00 s")
        controls.addWidget(self.time_label)
        close_button = QPushButton("Schließen")
        close_button.clicked.connect(self.close)
        controls.addWidget(close_button)
        root.addLayout(controls)
        self.clock.timeChanged.connect(self._on_time)
        self.clock.playingChanged.connect(lambda playing: self.play_button.setText("Pause" if playing else "Abspielen"))
        current_motion_policy().changed.connect(self._motion_changed)
        if record.beats:
            first = record.beats[record.representative_index()]
            self.clock.seek(min(self.clock.duration, first.r / record.fs))
        else:
            self._on_time(0.0)

    def _motion_changed(self, reduced: bool) -> None:
        if reduced:
            self.clock.pause()

    def _toggle_tour(self) -> None:
        if self._tour_step is not None:
            self._tour_step = None
            self.note.setText(self._base_note)
            self.tour_button.setText("Geführte Tour")
            self.tour_prev.setVisible(False)
            self.tour_next.setVisible(False)
        else:
            self.clock.pause()
            self.tour_button.setText("Tour beenden")
            self.tour_prev.setVisible(True)
            self.tour_next.setVisible(True)
            self._set_tour_step(0)

    def _advance_tour(self) -> None:
        if self._tour_step is None:
            return
        if self._tour_step >= len(self.TOUR_STEPS) - 1:
            self._toggle_tour()
        else:
            self._set_tour_step(self._tour_step + 1)

    def _set_tour_step(self, index: int) -> None:
        self._tour_step = int(np.clip(index, 0, len(self.TOUR_STEPS) - 1))
        tab, heading, explanation = self.TOUR_STEPS[self._tour_step]
        self.clock.pause()
        self.ecg_tabs.setCurrentIndex(tab)
        self.note.setText(f"{self._base_note}\n"
                          f"SCHRITT {self._tour_step + 1}/{len(self.TOUR_STEPS)} · {heading}: {explanation}")
        self.tour_prev.setEnabled(self._tour_step > 0)
        self.tour_next.setText("Abschließen" if self._tour_step == len(self.TOUR_STEPS) - 1 else "Weiter")
        if self.record.beats:
            beat = self.record.beats[self.record.representative_index()]
            position = beat.qrs_on + int(round(0.04 * self.record.fs)) if self._tour_step in (1, 2, 3) else beat.r
            self.clock.seek(min(self.clock.duration, position / self.record.fs))

    def _tab_changed(self, index: int) -> None:
        if index == 2:
            self.bridge.set_time(self.clock.time)
        elif index == 3:
            self.beat_map.set_time(self.clock.time)

    def _on_time(self, seconds: float) -> None:
        self.strip.set_time(seconds)
        self.leads.set_time(seconds)
        if self.ecg_tabs.currentIndex() == 2 and (not self.clock.playing or int(seconds * 10) != int(getattr(self, "_bridge_time", -1) * 10)):
            self._bridge_time = seconds
            self.bridge.set_time(seconds)
        elif self.ecg_tabs.currentIndex() == 3:
            self.beat_map.set_time(seconds)
        self.time_label.setText(f"{seconds:.2f} s".replace(".", ","))
        with QSignalBlocker(self.seek):
            self.seek.setValue(int(round(seconds * 1000)))
        beat = beat_at_time(self.record, seconds)
        if beat is not None:
            if beat.index != self._beat_index:
                self._beat_index = beat.index
                scene = scene_for(self.record, beat, "kors", True)
                if scene is not None and len(scene.xyz):
                    self.vcg.set_scene(scene)
                    if self.vcg.gl is not None:
                        self._vcg_base_distance = float(self.vcg.gl.opts["distance"])
                        self._fit_presentation_views()
            if self.vcg.scene_data is not None:
                start, end = self.record.beat_slice(beat)
                sample = int(round(seconds * self.record.fs)) - start
                active = 0 <= sample < end - start
                self.vcg.set_cursor_visible(active)
                if active:
                    self.vcg.set_cursor(sample)
        if self.heart.gl is not None:
            self.heart.update_frame(self.engine.frame(seconds))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_presentation_views()
        if not current_motion_policy().reduced:
            self.clock.play()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "heart"):
            self._fit_presentation_views()

    def _fit_presentation_views(self) -> None:
        """Fit visual cameras only; never alter the measured signal coordinates."""
        if self.heart.gl is not None:
            gl = self.heart.gl
            desired = 500.0 * presentation_camera_scale(gl.width(), gl.height())
            if abs(float(gl.opts["distance"]) - desired) > 1.0:
                gl.setCameraPosition(distance=desired)
        if self.vcg.gl is not None and self._vcg_base_distance is not None:
            gl = self.vcg.gl
            # A real loop can extend towards the camera as well as vertically;
            # leave headroom for perspective instead of optimizing on the small
            # synthetic demo loop alone.
            desired = self._vcg_base_distance * 1.35 * presentation_camera_scale(gl.width(), gl.height())
            if abs(float(gl.opts["distance"]) - desired) > 0.01:
                gl.setCameraPosition(distance=desired)

    def hideEvent(self, event) -> None:
        self.clock.pause()
        self.vcg.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self.clock.pause()
        self.vcg.stop()
        super().closeEvent(event)
