"""Ansichten des VCG-Labors: Ebenen, 3D-Schleife, Skalarspuren, Maßtafel, Lesung, Vorschau, Hauptfenster."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QMainWindow, QPushButton, QScrollArea, QSplitter, QTableWidget,
                             QTableWidgetItem, QTextBrowser, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..heart3d.widgets import CollapsibleSection
from ..motion import current_motion_policy
from .analysis import METHOD_LABELS, METHODS, VCGBeat, VCGRecord, analyze_signal, quick_record, record_from_payload
from .measures import (BeatBounds, BeatMeasures, INSCRIPTION_TEXT, PLANE_LABELS, PLANES, beat_variability, screen_coords,
                       serial_difference, spatial_velocity)
from .reading import ReadingResult, read_beat

# Wellen- und Vektorfarben je Thema; Grundtöne (Fläche, Rand, Text) kommen aus ``gui.theme``.
SEGMENT_COLORS = {
    True: {"p": "#5aa9ff", "qrs": "#4fd1c5", "t": "#ff9f43", "rest": "#55606c", "cursor": "#ffffff", "overlay": "#3a4654"},
    False: {"p": "#1d5bbf", "qrs": "#0f6b78", "t": "#b06a00", "rest": "#9aa7ad", "cursor": "#1d2a30", "overlay": "#c3d1d6"},
}
VECTOR_COLOR_SETS = {
    True: {"qrs_max": "#4fd1c5", "t_max": "#ff9f43", "st_j60": "#ff5f6d", "svg": "#c792ea", "p_max": "#5aa9ff",
           "qrs_initial_20": "#9be7a6", "qrs_terminal_40": "#f0e68c"},
    False: {"qrs_max": "#0f6b78", "t_max": "#b06a00", "st_j60": "#b42318", "svg": "#6b4bab", "p_max": "#1d5bbf",
            "qrs_initial_20": "#176b42", "qrs_terminal_40": "#8a6d00"},
}
STATUS_COLOR_SETS = {
    True: {"normal": "#2fa365", "grenzwertig": "#e0a53d", "hoch": "#e0605a", "niedrig": "#e0a53d", "abweichend": "#e0a53d"},
    False: {"normal": "#176b42", "grenzwertig": "#9a6700", "hoch": "#b42318", "niedrig": "#9a6700", "abweichend": "#9a6700"},
}
TIER_COLOR_SETS = {
    True: {"K": "#1f97a4", "E": "#e0a53d", "H": "#c792ea", "i": "#8fb9bd"},
    False: {"K": "#0f6b78", "E": "#9a6700", "H": "#6b4bab", "i": "#52616b"},
}
DEFAULT_DARK = True
COLORS = SEGMENT_COLORS[DEFAULT_DARK]  # Rückfallebene für Aufrufer ohne Thema
VECTOR_COLORS = VECTOR_COLOR_SETS[DEFAULT_DARK]
AREA_VECTORS = ("svg",)  # in mV·ms: nur Richtung zeigen, Länge auf die Schleifengröße normiert


def theme_tokens(dark: bool):
    from ..theme import tokens_for
    return tokens_for(bool(dark))


def plot_surface(dark: bool) -> str:
    t = theme_tokens(dark)
    return t.surface if dark else "#ffffff"
ECG_MM_PER_MS, ECG_MM_PER_MV = 0.025, 10.0
REFERENCE_PATH = Path(__file__).with_name("reference_limits.json")


def load_reference() -> Optional[dict]:
    try:
        return json.loads(REFERENCE_PATH.read_text(encoding="utf-8")) if REFERENCE_PATH.exists() else None
    except Exception:
        return None


# ----------------------------------------------------------------------------- Szene
@dataclass
class LoopScene:
    """Ein Schlagfenster für alle Ansichten: ursprungskorrigiertes XYZ, Fiducials relativ zum Fenster, Cursor."""
    xyz: np.ndarray
    fs: float
    bounds: BeatBounds
    label: str = ""
    cursor: int = 0
    overlays: list[tuple[np.ndarray, BeatBounds]] = field(default_factory=list)
    measures: Optional[BeatMeasures] = None
    origin_uncorrected_shift: Optional[np.ndarray] = None

    def segment(self, name: str) -> Optional[tuple[int, int]]:
        b, n = self.bounds, len(self.xyz)
        if name == "qrs":
            return max(0, b.qrs_on), min(n, b.qrs_off)
        if name == "t":
            if b.t_end is None:
                return None
            return max(0, b.t_on if b.t_on else b.qrs_off), min(n, b.t_end)
        if name == "p":
            if b.p_on is None or b.p_off is None:
                return None
            return max(0, b.p_on), min(n, b.p_off)
        return None

    def segment_of(self, k: int) -> str:
        for name in ("qrs", "t", "p"):
            s = self.segment(name)
            if s and s[0] <= k < s[1]:
                return name
        return "rest"


def _segments_with_rest(scene: LoopScene) -> list[tuple[str, int, int]]:
    """Alle Abschnitte in Zeitfolge inkl. Zwischenstücken (rest), damit die Kurve lückenlos ist."""
    marks = []
    for name in ("p", "qrs", "t"):
        s = scene.segment(name)
        if s and s[1] > s[0]:
            marks.append((name, s[0], s[1]))
    marks.sort(key=lambda m: m[1])
    out, pos, n = [], 0, len(scene.xyz)
    for name, a, b in marks:
        if a > pos:
            out.append(("rest", pos, a + 1))
        out.append((name, a, b))
        pos = b - 1
    if pos < n - 1:
        out.append(("rest", pos, n))
    return out


def loop_tail_points(xyz: np.ndarray, cursor: int, fs: float, duration_ms: float = 80.0) -> np.ndarray:
    """Actual samples immediately preceding the cursor, never interpolated."""
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or fs <= 0:
        return np.empty((0, 3), dtype=np.float32)
    k = int(np.clip(cursor, 0, len(xyz) - 1))
    count = max(2, int(round(duration_ms * fs / 1000.0)))
    return np.asarray(xyz[max(0, k - count + 1):k + 1], dtype=np.float32)


def loop_revealed_points(xyz: np.ndarray, cursor: int) -> np.ndarray:
    """Recorded prefix of one loop, without interpolation or invented frames."""
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        return np.empty((0, 3), dtype=np.float32)
    k = int(np.clip(cursor, 0, len(xyz) - 1))
    return np.asarray(xyz[:k + 1], dtype=np.float32)


# ----------------------------------------------------------------------------- Ebene
class LoopPlaneWidget(pg.PlotWidget):
    cursorMoved = pyqtSignal(int)

    def __init__(self, plane: str, parent: Optional[QWidget] = None, compact: bool = False, dark: bool = DEFAULT_DARK):
        super().__init__(parent, background=plot_surface(dark))
        self.plane = plane
        self.compact = compact
        self._dark = bool(dark)
        self.colors = SEGMENT_COLORS[self._dark]
        self.vector_colors = VECTOR_COLOR_SETS[self._dark]
        title, left, right, top, bottom = PLANE_LABELS[plane]
        self.setTitle(title if not compact else title.split(" (")[0], size="9pt" if not compact else "8pt")
        vb = self.getViewBox()
        vb.setAspectLocked(True, ratio=1.0)
        vb.invertY(True)  # Bildschirm-y nach unten = zweite Achse (inferior bzw. anterior)
        self.showGrid(x=True, y=True, alpha=0.18)
        for ax in ("left", "bottom"):
            self.getAxis(ax).enableAutoSIPrefix(False)
        self.setLabel("bottom", f"{left} ← mV → {right}")
        self.setLabel("left", f"{bottom} ← mV → {top}")
        if compact:
            self.hideAxis("left")
            self.hideAxis("bottom")
        self.setMenuEnabled(False)
        self.show_dashes = True
        self.show_vectors = True
        self.show_overlays = False
        self._curves: dict[str, list[pg.PlotDataItem]] = {}
        self._dash_items: list[pg.ScatterPlotItem] = []
        self._vector_items: list = []
        self._overlay_items: list = []
        self._cursor = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(self.colors["cursor"]), pen=pg.mkPen(theme_tokens(self._dark).border_strong, width=0.5))
        self._tail = pg.PlotDataItem(pen=pg.mkPen(self.colors["cursor"], width=2.2))
        self._origin = pg.ScatterPlotItem(size=7, symbol="+", pen=pg.mkPen(self.colors["cursor"], width=1))
        self.addItem(self._origin)
        self.addItem(self._tail)
        self.addItem(self._cursor)
        self._arrow = pg.ArrowItem(angle=0, tipAngle=35, headLen=9, brush=pg.mkBrush(self.colors["qrs"]), pen=None)
        self.addItem(self._arrow)
        self._arrow.hide()
        self.scene_data: Optional[LoopScene] = None
        self.scene().sigMouseClicked.connect(self._clicked)

    # ---- Aufbau
    def set_scene(self, scene: LoopScene) -> None:
        self.scene_data = scene
        for items in self._curves.values():
            for it in items:
                self.removeItem(it)
        self._curves.clear()
        for it in self._dash_items + self._vector_items + self._overlay_items:
            self.removeItem(it)
        self._dash_items, self._vector_items, self._overlay_items = [], [], []
        if self.show_overlays:
            for oxyz, ob in scene.overlays:
                sc = LoopScene(oxyz, scene.fs, ob)
                for name, a, b in _segments_with_rest(sc):
                    if name == "rest":
                        continue
                    p = screen_coords(oxyz[a:b], self.plane)
                    it = pg.PlotDataItem(p[:, 0], p[:, 1], pen=pg.mkPen(self.colors["overlay"], width=1.0))
                    self.addItem(it)
                    self._overlay_items.append(it)
        for name, a, b in _segments_with_rest(scene):
            p = screen_coords(scene.xyz[a:b], self.plane)
            width = 1.0 if name == "rest" else (1.6 if self.compact else 2.0)
            it = pg.PlotDataItem(p[:, 0], p[:, 1], pen=pg.mkPen(self.colors[name], width=width))
            self.addItem(it)
            self._curves.setdefault(name, []).append(it)
            if self.show_dashes and name != "rest" and not self.compact:
                step = max(1, int(round(0.002 * scene.fs)))  # alle 2 ms
                idx = np.arange(a, b, step)
                big = ((idx - a) % max(1, int(round(0.010 * scene.fs)))) == 0  # jede 10 ms größer
                pts = screen_coords(scene.xyz[idx], self.plane)
                col = QColor(self.colors[name]).darker(140) if self._dark else QColor(self.colors[name]).lighter(115)
                sizes = np.where(big, 5.0, 2.6)
                sc = pg.ScatterPlotItem(pts[:, 0], pts[:, 1], size=sizes, brush=pg.mkBrush(col), pen=None, pxMode=True)
                self.addItem(sc)
                self._dash_items.append(sc)
        if self.show_vectors and scene.measures is not None and not self.compact:
            radius = float(np.max(np.linalg.norm(scene.xyz, axis=1))) if len(scene.xyz) else 1.0
            for key, color in self.vector_colors.items():
                v = scene.measures.vectors.get(key)
                if v is None or np.linalg.norm(v) < 1e-9:
                    continue
                # Flächenvektoren (mV·ms) haben keinen mV-Maßstab: nur Richtung, Länge auf die Schleife normiert
                vv = v / np.linalg.norm(v) * 0.8 * radius if key in AREA_VECTORS else v
                p = screen_coords(np.vstack([np.zeros(3), vv]), self.plane)
                it = pg.PlotDataItem(p[:, 0], p[:, 1], pen=pg.mkPen(color, width=1.2, style=Qt.PenStyle.DashLine))
                self.addItem(it)
                self._vector_items.append(it)
        # Richtungspfeil bei 30 % der QRS-Schleife
        s = scene.segment("qrs")
        if s and s[1] - s[0] > 6:
            k = s[0] + int(0.3 * (s[1] - s[0]))
            p0, p1 = screen_coords(scene.xyz[k - 1:k + 1], self.plane)
            d = p1 - p0
            if np.linalg.norm(d) > 1e-9:
                ang = float(np.degrees(np.arctan2(d[1], d[0])))
                self._arrow.setStyle(angle=180 + ang)  # ArrowItem zeigt bei 0° nach links
                self._arrow.setPos(p1[0], p1[1])
                self._arrow.show()
            else:
                self._arrow.hide()
        self.set_cursor(scene.cursor)
        self.autoRange(padding=0.08)

    def apply_theme(self, dark: bool) -> None:
        from ..theme import apply_plot_theme
        self._dark = bool(dark)
        self.colors = SEGMENT_COLORS[self._dark]
        self.vector_colors = VECTOR_COLOR_SETS[self._dark]
        apply_plot_theme(self, self._dark)
        self.setBackground(plot_surface(self._dark))
        self._cursor.setBrush(pg.mkBrush(self.colors["cursor"]))
        self._tail.setPen(pg.mkPen(self.colors["cursor"], width=2.2))
        self._arrow.setBrush(pg.mkBrush(self.colors["qrs"]))
        if self.scene_data is not None:
            self.set_scene(self.scene_data)

    def set_cursor(self, k: int) -> None:
        sc = self.scene_data
        if sc is None or len(sc.xyz) == 0:
            return
        k = int(np.clip(k, 0, len(sc.xyz) - 1))
        p = screen_coords(sc.xyz[k:k + 1], self.plane)
        self._cursor.setData(p[:, 0], p[:, 1])
        tail = int(0.012 * sc.fs)
        pt = screen_coords(sc.xyz[max(0, k - tail):k + 1], self.plane)
        self._tail.setData(pt[:, 0], pt[:, 1])

    def _clicked(self, ev) -> None:
        sc = self.scene_data
        if sc is None or not self.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = self.getViewBox().mapSceneToView(ev.scenePos())
        pts = screen_coords(sc.xyz, self.plane)
        k = int(np.argmin(np.hypot(pts[:, 0] - p.x(), pts[:, 1] - p.y())))
        self.cursorMoved.emit(k)


# ----------------------------------------------------------------------------- 3D
class Loop3DWidget(QWidget):
    """Räumliche Schleife (pyqtgraph OpenGL); ohne OpenGL bleibt ein Hinweis stehen."""

    def __init__(self, parent: Optional[QWidget] = None, compact: bool = False, dark: bool = DEFAULT_DARK):
        super().__init__(parent)
        self.compact = compact
        self._dark = bool(dark)
        self.colors = SEGMENT_COLORS[self._dark]
        self.vector_colors = VECTOR_COLOR_SETS[self._dark]
        self.gl = None
        self._items: list = []
        self._cursor = None
        self._tail = None
        self._halo = None
        self._trace = None
        self.trace_reveal = False
        self._cursor_visible = True
        self.auto_rotate = compact
        self._reduced_motion = current_motion_policy().reduced
        self.scene_data: Optional[LoopScene] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        try:
            import pyqtgraph.opengl as gl
            self._gl = gl
            self.gl = gl.GLViewWidget()
            self.gl.setBackgroundColor(plot_surface(self._dark))
            self.gl.opts["distance"] = 4.5 if not compact else 4.0
            self.gl.opts["elevation"] = 18
            self.gl.opts["azimuth"] = -60
            lay.addWidget(self.gl)
            self._build_axes()
        except Exception as exc:
            self.gl = None
            lab = QLabel(f"3D nicht verfügbar ({exc})")
            lab.setWordWrap(True)
            lay.addWidget(lab)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        current_motion_policy().changed.connect(self.set_reduced_motion)
        if compact and not self._reduced_motion:
            self._timer.start()

    @staticmethod
    def to_gl(xyz: np.ndarray) -> np.ndarray:
        """Frank (X links, Y inferior, Z posterior) → GL (x links, y anterior, z kranial)."""
        return np.column_stack([xyz[:, 0], -xyz[:, 2], -xyz[:, 1]]).astype(np.float32)

    def _build_axes(self, L: float = 1.5) -> None:
        gl = self._gl
        for it in getattr(self, "_axis_items", []):
            self.gl.removeItem(it)
        self._axis_items = []
        for vec, col, name in (((L, 0, 0), (0.6, 0.8, 1.0, 0.8), "L"), ((0, 0, -L), (1.0, 0.8, 0.5, 0.8), "I"), ((0, -L, 0), (0.8, 0.6, 1.0, 0.8), "P")):
            it = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], vec], dtype=np.float32), color=col, width=1.2 if not self.compact else 0.8, antialias=True)
            self.gl.addItem(it)
            self._axis_items.append(it)
            if not self.compact:
                try:
                    t = gl.GLTextItem(pos=np.array(vec, dtype=np.float32) * 1.06, text=name, color=QColor(220, 228, 236), font=QFont("Segoe UI", 8))
                    self.gl.addItem(t)
                except Exception:
                    pass
        if not self.compact and not getattr(self, "_grids_built", False):
            self._grids_built = True
            for axis in ("x", "y", "z"):
                g = gl.GLGridItem()
                g.setSize(3, 3)
                g.setSpacing(0.5, 0.5)
                g.setColor((90, 100, 115, 40))
                if axis == "x":
                    g.rotate(90, 0, 1, 0)
                elif axis == "y":
                    g.rotate(90, 1, 0, 0)
                self.gl.addItem(g)

    def set_scene(self, scene: LoopScene) -> None:
        self.scene_data = scene
        if self.gl is None:
            return
        gl = self._gl
        for it in self._items:
            self.gl.removeItem(it)
        self._items = []
        for name, a, b in _segments_with_rest(scene):
            pts = self.to_gl(scene.xyz[a:b])
            c = QColor(self.colors[name])
            width = 1.0 if name == "rest" else 2.6
            it = gl.GLLinePlotItem(pos=pts, color=(c.redF(), c.greenF(), c.blueF(), 0.55 if name == "rest" else 1.0), width=width, antialias=True)
            self.gl.addItem(it)
            self._items.append(it)
            if name != "rest" and not self.compact:
                step = max(1, int(round(0.002 * scene.fs)))
                idx = np.arange(a, b, step)
                sc = gl.GLScatterPlotItem(pos=self.to_gl(scene.xyz[idx]), size=3.0, color=(c.redF() * 0.7, c.greenF() * 0.7, c.blueF() * 0.7, 1.0), pxMode=True)
                self.gl.addItem(sc)
                self._items.append(sc)
        if scene.measures is not None and not self.compact:
            radius = float(np.max(np.linalg.norm(scene.xyz, axis=1))) if len(scene.xyz) else 1.0
            for key, color in self.vector_colors.items():
                v = scene.measures.vectors.get(key)
                if v is None or np.linalg.norm(v) < 1e-9:
                    continue
                vv = v / np.linalg.norm(v) * 0.8 * radius if key in AREA_VECTORS else v
                c = QColor(color)
                it = gl.GLLinePlotItem(pos=self.to_gl(np.vstack([np.zeros(3), vv])), color=(c.redF(), c.greenF(), c.blueF(), 0.9), width=1.5, antialias=True)
                self.gl.addItem(it)
                self._items.append(it)
        if self._tail is None:
            self._tail = gl.GLLinePlotItem(pos=np.zeros((0, 3), dtype=np.float32),
                                           color=(1.0, 1.0, 1.0, 0.95), width=4.0, antialias=True)
            self.gl.addItem(self._tail)
        if self._trace is None:
            self._trace = gl.GLLinePlotItem(pos=np.zeros((0, 3), dtype=np.float32),
                                            color=(0.31, 0.82, 0.77, 1.0), width=3.0, antialias=True)
            self.gl.addItem(self._trace)
        if self._halo is None:
            self._halo = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=np.float32), size=18.0,
                                              color=(1.0, 1.0, 1.0, 0.22), pxMode=True)
            self.gl.addItem(self._halo)
        if self._cursor is None:
            cc = QColor(self.colors["cursor"])
            self._cursor = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32), size=9.0,
                                                color=(cc.redF(), cc.greenF(), cc.blueF(), 1.0), pxMode=True)
            self.gl.addItem(self._cursor)
        self.set_cursor(scene.cursor)
        self._apply_trace_visibility()
        pts = self.to_gl(scene.xyz)
        centre = pts.mean(0)
        radius = float(np.max(np.linalg.norm(pts - centre, axis=1))) if len(pts) else 1.0
        from PyQt6.QtGui import QVector3D
        self.gl.opts["center"] = QVector3D(float(centre[0]), float(centre[1]), float(centre[2]))
        if self.compact:
            self._build_axes(0.6 * radius)
            self.gl.opts["distance"] = max(0.8, 2.6 * radius)
        else:
            self.gl.opts["distance"] = max(1.5, 3.0 * radius)
        self.gl.update()

    def set_cursor(self, k: int) -> None:
        if self.gl is None or self._cursor is None or self.scene_data is None or not len(self.scene_data.xyz):
            return
        k = int(np.clip(k, 0, len(self.scene_data.xyz) - 1))
        self.scene_data.cursor = k
        head = self.to_gl(self.scene_data.xyz[k:k + 1])
        self._cursor.setData(pos=head)
        if self._halo is not None:
            self._halo.setData(pos=head)
        if self._tail is not None:
            points = loop_tail_points(self.scene_data.xyz, k, self.scene_data.fs)
            self._tail.setData(pos=self.to_gl(points) if len(points) else np.zeros((0, 3), dtype=np.float32))
        if self._trace is not None and self.trace_reveal:
            revealed = loop_revealed_points(self.scene_data.xyz, k)
            self._trace.setData(pos=self.to_gl(revealed) if len(revealed) else np.zeros((0, 3), dtype=np.float32))

    def set_trace_reveal(self, enabled: bool) -> None:
        """Draw only measured samples up to the cursor; retain static loop outside it."""
        self.trace_reveal = bool(enabled)
        if self.scene_data is not None:
            self.set_cursor(self.scene_data.cursor)
        self._apply_trace_visibility()

    def _apply_trace_visibility(self) -> None:
        reveal_active = self.trace_reveal and self._cursor_visible
        for item in self._items:
            item.setVisible(not reveal_active)
        if self._trace is not None:
            self._trace.setVisible(reveal_active)

    def set_cursor_visible(self, visible: bool) -> None:
        self._cursor_visible = bool(visible)
        for item in (self._cursor, self._halo, self._tail):
            if item is not None:
                item.setVisible(bool(visible))
        self._apply_trace_visibility()

    def _tick(self) -> None:
        if self.gl is not None and self.auto_rotate and not self._reduced_motion and self.isVisible():
            try:
                self.gl.orbit(0.4, 0)
            except Exception:
                self._timer.stop()

    def apply_theme(self, dark: bool) -> None:
        self._dark = bool(dark)
        self.colors = SEGMENT_COLORS[self._dark]
        self.vector_colors = VECTOR_COLOR_SETS[self._dark]
        if self.gl is None:
            return
        self.gl.setBackgroundColor(plot_surface(self._dark))
        for item in (self._cursor, self._tail, self._halo, self._trace):
            if item is not None:
                self.gl.removeItem(item)
        self._cursor = None
        self._tail = None
        self._halo = None
        self._trace = None
        if self.scene_data is not None:
            self.set_scene(self.scene_data)

    def set_auto_rotate(self, on: bool) -> None:
        self.auto_rotate = on
        if on and not self._reduced_motion and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()

    def set_reduced_motion(self, on: bool) -> None:
        self._reduced_motion = bool(on)
        self.set_auto_rotate(self.auto_rotate)

    def stop(self) -> None:
        self._timer.stop()

    def hideEvent(self, e) -> None:
        self._timer.stop()
        super().hideEvent(e)

    def showEvent(self, e) -> None:
        if self.auto_rotate and not self._reduced_motion:
            self._timer.start()
        super().showEvent(e)


# ----------------------------------------------------------------------------- Skalarspuren
class ScalarTraceWidget(pg.GraphicsLayoutWidget):
    """X, Y, Z, |V| (Standardmaßstab 25 mm/s : 10 mm/mV, Seitenverhältnis gesperrt) und räumliche
    Geschwindigkeit nebeneinander gegen die Zeit; Klick setzt den Cursor in allen Ansichten."""

    cursorMoved = pyqtSignal(int)
    NAMES = ("X (links +)", "Y (inferior +)", "Z (posterior +)", "|V|", "v (mV/ms)")

    def __init__(self, parent: Optional[QWidget] = None, dark: bool = DEFAULT_DARK):
        super().__init__(parent)
        self._dark = bool(dark)
        self.colors = SEGMENT_COLORS[self._dark]
        self.setBackground(plot_surface(self._dark))
        self.plots, self.curves, self.lines, self.regions = [], [], [], []
        self.scene_data: Optional[LoopScene] = None
        for k, name in enumerate(self.NAMES):
            p = self.addPlot(row=0, col=k)
            p.setTitle(name, size="8pt")
            p.setMenuEnabled(False)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setLabel("bottom", "mm (25 mm/s)" if k < 4 else "ms")
            p.setLabel("left", "mm (10 mm/mV)" if k < 4 else "mV/ms")
            for ax in ("left", "bottom"):
                p.getAxis(ax).enableAutoSIPrefix(False)
            if k < 4:
                p.getViewBox().setAspectLocked(True, ratio=1.0)
            c = p.plot(pen=pg.mkPen(theme_tokens(self._dark).text, width=1.2))
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen(self.colors["cursor"], width=1))
            p.addItem(line)
            self.plots.append(p)
            self.curves.append(c)
            self.lines.append(line)
            self.regions.append([])
        self.scene().sigMouseClicked.connect(self._clicked)

    def set_scene(self, scene: LoopScene) -> None:
        self.scene_data = scene
        t_ms = np.arange(len(scene.xyz)) * 1000.0 / scene.fs
        x_mm = t_ms * ECG_MM_PER_MS
        mag = np.linalg.norm(scene.xyz, axis=1)
        vel = spatial_velocity(scene.xyz, scene.fs) / 1000.0
        ys = [scene.xyz[:, 0] * ECG_MM_PER_MV, scene.xyz[:, 1] * ECG_MM_PER_MV, scene.xyz[:, 2] * ECG_MM_PER_MV, mag * ECG_MM_PER_MV, vel]
        for k, (p, c, y) in enumerate(zip(self.plots, self.curves, ys)):
            xs = x_mm if k < 4 else t_ms
            c.setData(xs, y)
            for r in self.regions[k]:
                p.removeItem(r)
            self.regions[k] = []
            for name in ("p", "qrs", "t"):
                s = scene.segment(name)
                if s is None:
                    continue
                col = QColor(self.colors[name])
                col.setAlpha(45 if self._dark else 30)
                r = pg.LinearRegionItem(values=(xs[s[0]], xs[min(len(xs) - 1, s[1] - 1)]), movable=False, brush=pg.mkBrush(col), pen=pg.mkPen(None))
                r.setZValue(-10)
                p.addItem(r)
                self.regions[k].append(r)
            if k < 4:
                self._fit_standard(p, float(xs[0]), float(xs[-1]), float(np.min(y)), float(np.max(y)))
            else:
                p.enableAutoRange()
        self.set_cursor(scene.cursor)

    def apply_theme(self, dark: bool) -> None:
        """Achsen sind hier PlotItems (kein PlotWidget): Farben direkt setzen statt ``apply_plot_theme``."""
        self._dark = bool(dark)
        self.colors = SEGMENT_COLORS[self._dark]
        t = theme_tokens(self._dark)
        self.setBackground(plot_surface(self._dark))
        for p, c, line in zip(self.plots, self.curves, self.lines):
            for axis_name in ("left", "bottom"):
                axis = p.getAxis(axis_name)
                if axis is not None:
                    axis.setPen(t.text)
                    axis.setTextPen(t.text)
            p.titleLabel.setText(p.titleLabel.text, color=t.text, size="8pt")
            c.setPen(pg.mkPen(t.text, width=1.2))
            line.setPen(pg.mkPen(self.colors["cursor"], width=1))
        if self.scene_data is not None:
            self.set_scene(self.scene_data)

    @staticmethod
    def _fit_standard(p, x0: float, x1: float, y0: float, y1: float) -> None:
        """Bei gesperrtem Seitenverhältnis beide Bereiche so setzen, dass das ganze Fenster sichtbar bleibt."""
        vb = p.getViewBox()
        rect = vb.sceneBoundingRect()
        ratio = (rect.height() / rect.width()) if rect.width() > 10 and rect.height() > 10 else 1.0
        x_span = max(1e-6, (x1 - x0) * 1.04)
        y_need = x_span * ratio
        yc, y_span = 0.5 * (y0 + y1), (y1 - y0) * 1.1
        if y_span < y_need:
            y_span = y_need
        else:
            x_span = y_span / max(ratio, 1e-6)
        xc = 0.5 * (x0 + x1)
        vb.setRange(xRange=(xc - x_span / 2, xc + x_span / 2), yRange=(yc - y_span / 2, yc + y_span / 2), padding=0)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if getattr(self, "scene_data", None) is not None:
            QTimer.singleShot(0, lambda: self.set_scene(self.scene_data) if self.scene_data is not None else None)

    def set_cursor(self, k: int) -> None:
        if self.scene_data is None:
            return
        ms = k * 1000.0 / self.scene_data.fs
        for j, line in enumerate(self.lines):
            line.setPos(ms * ECG_MM_PER_MS if j < 4 else ms)

    def _clicked(self, ev) -> None:
        if self.scene_data is None:
            return
        for j, p in enumerate(self.plots):
            vb = p.getViewBox()
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                x = vb.mapSceneToView(ev.scenePos()).x()
                ms = x / ECG_MM_PER_MS if j < 4 else x
                self.cursorMoved.emit(int(round(ms * self.scene_data.fs / 1000.0)))
                return


# ----------------------------------------------------------------------------- Maßtafel und Lesung
def status_color(status: str, dark: bool) -> str:
    t = theme_tokens(dark)
    return STATUS_COLOR_SETS[bool(dark)].get(status, t.text_muted)


class MeasureTree(QTreeWidget):
    vectorSelected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None, dark: bool = DEFAULT_DARK):
        super().__init__(parent)
        self._dark = bool(dark)
        self.setHeaderLabels(["Maß", "Wert", "Einheit", "Bewertung"])
        self.setColumnWidth(0, 250)
        self.setColumnWidth(1, 60)
        self.setColumnWidth(2, 55)
        self.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.setStyleSheet("font-size: 10px;")
        self.setRootIsDecorated(True)
        self.itemClicked.connect(self._clicked)

    def set_measures(self, m: Optional[BeatMeasures], expanded: tuple[str, ...] = ("QRS-Vektoren", "QRS-T-Beziehung", "ST-Vektor")) -> None:
        self._measures_cache = m
        self.clear()
        if m is None:
            return
        for group, items in m.by_group().items():
            top = QTreeWidgetItem([group, "", "", ""])
            top.setFont(0, QFont("Segoe UI", 9, QFont.Weight.Bold))
            self.addTopLevelItem(top)
            for x in items:
                st = x.status
                row = QTreeWidgetItem([x.label, x.text(), x.unit, st if st not in ("", "–") else ("ohne Referenz" if st == "" else "")])
                row.setForeground(3, QColor(status_color(st, self._dark)))
                row.setForeground(1, QColor(theme_tokens(self._dark).text_strong))
                tip = x.note
                if x.limit is not None:
                    lo = "–" if x.limit.low is None else f"{x.limit.low:g}"
                    hi = "–" if x.limit.high is None else f"{x.limit.high:g}"
                    tip += f"\nReferenz {lo} … {hi} {x.unit} ({x.limit.source})"
                row.setToolTip(0, tip.strip())
                row.setToolTip(3, tip.strip())
                row.setData(0, Qt.ItemDataRole.UserRole, x.key)
                top.addChild(row)
            top.setExpanded(group in expanded)
        # Drehrichtungen als eigene Gruppe
        if m.inscriptions:
            top = QTreeWidgetItem(["Drehrichtung (Anzeigeebene)", "", "", ""])
            top.setFont(0, QFont("Segoe UI", 9, QFont.Weight.Bold))
            self.addTopLevelItem(top)
            for seg, planes in m.inscriptions.items():
                for pl, (kind, minor, area) in planes.items():
                    row = QTreeWidgetItem([f"{seg.upper()} {pl}", INSCRIPTION_TEXT[kind], "", f"gegenläufig {minor * 100:.0f} %"])
                    top.addChild(row)
            top.setExpanded(True)
        if m.octants:
            top = QTreeWidgetItem(["Oktanten-Verweildauer QRS", "", "", ""])
            top.setFont(0, QFont("Segoe UI", 9, QFont.Weight.Bold))
            self.addTopLevelItem(top)
            for k, v in list(m.octants.items())[:4]:
                top.addChild(QTreeWidgetItem([k.replace("L", "links").replace("R", "rechts").replace(" I", " inferior").replace(" S", " superior").replace(" P", " posterior").replace(" A", " anterior"), f"{v * 100:.0f}", "%", ""]))

    def apply_theme(self, dark: bool) -> None:
        self._dark = bool(dark)
        self._measures_cache = getattr(self, "_measures_cache", None)
        if self._measures_cache is not None:
            self.set_measures(self._measures_cache)

    def _clicked(self, item, col) -> None:
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key:
            self.vectorSelected.emit(str(key))


class ReadingPanel(QTextBrowser):
    def __init__(self, parent: Optional[QWidget] = None, dark: bool = DEFAULT_DARK):
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setMinimumHeight(120)
        self.setMaximumHeight(260)
        self._result: Optional[ReadingResult] = None
        self._header = ""
        self.apply_theme(dark)

    def apply_theme(self, dark: bool) -> None:
        self._dark = bool(dark)
        t = theme_tokens(self._dark)
        self.setStyleSheet(f"font-size: 11px; background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; border-radius: 6px;")
        self.set_result(self._result, self._header)

    def set_result(self, r: Optional[ReadingResult], header: str = "") -> None:
        self._result, self._header = r, header
        t = theme_tokens(getattr(self, "_dark", DEFAULT_DARK))
        if r is None:
            self.setHtml(f"<span style='color:{t.text_muted}'><i>Keine Lesung.</i></span>")
            return
        tiers = TIER_COLOR_SETS[bool(getattr(self, "_dark", DEFAULT_DARK))]
        rows = "".join(f"<div style='margin:2px 0'>{x.html(tiers, t.text_muted)}</div>" for x in r.readings)
        legend = (f"<div style='color:{t.text_muted}; font-size:9px; margin-top:4px'>K klassisches Kriterium des Labors · "
                  "E VCG-Regel mit Quelle · H Hypothese/Heuristik · i Hinweis</div>")
        self.setHtml((f"<div style='color:{t.text_muted}; font-size:10px'>{header}</div>" if header else "") + rows + legend)


# ----------------------------------------------------------------------------- Vorschau
class MiniVCGWidget(QWidget):
    """Kleines Vorschaufenster: drehende räumliche Schleife des Repräsentativschlags, „Vergrößern“ öffnet das Labor."""

    openRequested = pyqtSignal()
    analysisReady = pyqtSignal(int, object)

    def __init__(self, parent: Optional[QWidget] = None, *, size: int = 200, auto_analysis: bool = True, dark: bool = DEFAULT_DARK):
        super().__init__(parent)
        self._dark = bool(dark)
        self.auto_analysis = auto_analysis
        self._run = 0
        self.record: Optional[VCGRecord] = None
        self.reference = load_reference()
        self._window: Optional[VCGLabWindow] = None
        self._closing = False
        self.analysisReady.connect(self._on_ready)
        self.view = Loop3DWidget(self, compact=True, dark=self._dark)
        self.view.setFixedSize(size, size)
        self.caption = QLabel("Kein ECG geladen")
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet(f"font-size: 10px; color: {theme_tokens(self._dark).text_muted};")
        self.open_button = QPushButton("Vergrößern")
        self.open_button.setToolTip("VCG-Labor mit Ebenen, Maßen und Schleifenlesung öffnen")
        self.open_button.clicked.connect(self.openRequested.emit)
        self.rotate_check = QCheckBox("Drehen")
        self.rotate_check.setChecked(not current_motion_policy().reduced)
        self.rotate_check.setStyleSheet("font-size: 10px;")
        self.rotate_check.toggled.connect(self.view.set_auto_rotate)
        self.view.set_auto_rotate(self.rotate_check.isChecked())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.view)
        lay.addWidget(self.caption)
        row = QHBoxLayout()
        row.addWidget(self.open_button, 1)
        row.addWidget(self.rotate_check)
        lay.addLayout(row)
        self.openRequested.connect(self.open_window)

    # ---- Daten
    def set_signal(self, signal: np.ndarray, fs: float, lead_names=None, label: str = "", age=None, sex=None) -> None:
        try:
            names = tuple(lead_names) if lead_names is not None else None
            rec = quick_record(signal, fs, names or (), label, age, sex)
            self._apply(rec)
        except Exception as exc:
            self.caption.setText(f"VCG nicht verfügbar: {exc}")
            return
        if self.auto_analysis:
            self._run += 1
            run = self._run
            sig, names_ = np.asarray(signal, dtype=float), names or ()

            def work() -> None:
                try:
                    out = analyze_signal(sig, fs, names_, label, age, sex)
                except Exception as exc:
                    out = f"{type(exc).__name__}: {exc}"
                try:
                    self.analysisReady.emit(run, out)
                except RuntimeError:
                    pass  # Vorschau inzwischen geschlossen

            threading.Thread(target=work, name="vcglab-analysis", daemon=True).start()
            self.caption.setText(self.caption.text() + " · Analyse läuft …")

    def set_analysis(self, payload: Optional[dict], signal: np.ndarray, fs: float, lead_names, label: str = "", age=None, sex=None) -> None:
        """Laborbefund übernehmen (keine eigene Delineation)."""
        self._run += 1
        try:
            rec = record_from_payload(signal, fs, lead_names, payload, label, age, sex) if payload else quick_record(signal, fs, lead_names, label, age, sex)
            self._apply(rec)
        except Exception as exc:
            self.caption.setText(f"Analysestand nicht übernommen: {exc}")

    def set_record(self, rec: VCGRecord) -> None:
        self._run += 1
        self._apply(rec)

    def _on_ready(self, run: int, out) -> None:
        if run != self._run:
            return
        if isinstance(out, VCGRecord):
            self._apply(out)
        else:
            self.caption.setText(self.caption.text().replace(" · Analyse läuft …", "") + f" · Hauptprojekt-Analyse nicht verfügbar ({out})")

    def _apply(self, rec: VCGRecord) -> None:
        self.record = rec
        scene = scene_for(rec, None, "kors", True)
        if scene is not None:
            self.view.set_scene(scene)
        m = scene.measures if scene else None
        src = {"core": "Delineation (Hauptprojekt)", "payload": "Laborbefund", "quick": "Schnellschätzung"}.get(rec.source, rec.source)
        txt = f"{rec.label or 'VCG'} · {len(rec.beats)} Schläge · {src}"
        if m is not None:
            a = m.get("qrs_t_angle_mean")
            txt += f" · QRS-T {a:.0f}°" if a is not None else ""
        self.caption.setText(txt)
        if self._window is not None:
            self._window.set_record(rec)

    def set_dark(self, dark: bool) -> None:
        """Vorschau und ein geöffnetes Laborfenster dem Thema des Hauptfensters folgen lassen."""
        self._dark = bool(dark)
        self.view.apply_theme(self._dark)
        self.caption.setStyleSheet(f"font-size: 10px; color: {theme_tokens(self._dark).text_muted};")
        if self._window is not None:
            self._window.panel.set_dark(self._dark)

    def open_window(self) -> None:
        if self._window is None:
            self._window = VCGLabWindow(parent=None, dark=self._dark)
            self._window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            if self.record is not None:
                self._window.set_record(self.record)
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()

    def stop(self) -> None:
        self._closing = True
        self.view.stop()
        if self._window is not None:
            try:
                self._window.close()
            except Exception:
                pass

    def closeEvent(self, e) -> None:
        self.stop()
        super().closeEvent(e)


def scene_for(rec: VCGRecord, beat: Optional[VCGBeat], method: str, corrected: bool, use_median: bool = False,
              overlays: bool = False, reference: Optional[dict] = None) -> Optional[LoopScene]:
    """Fensterausschnitt (P−80 ms … T+120 ms) eines Schlags oder des Medianschlags als Szene."""
    if use_median and rec.median_beat is not None and rec.median_ecg is not None:
        xyz = rec.median_xyz(method, corrected)
        beat = rec.median_beat
        label = "Medianschlag (fiducial-gewarpt)"
    else:
        xyz = rec.xyz(method, corrected)
        if beat is None:
            if not rec.beats:
                return None
            beat = rec.beats[rec.representative_index()]
        label = f"Schlag {beat.label}"
    a, b = rec.beat_slice(beat) if beat.index >= 0 else (max(0, beat.qrs_on - int(0.3 * rec.fs)), min(len(xyz), (beat.t_end or beat.qrs_off) + int(0.12 * rec.fs)))
    if beat.index < 0 and rec.median_ecg is not None:
        a = max(0, (beat.p_on if beat.p_on is not None else beat.qrs_on) - int(0.08 * rec.fs))
    seg = xyz[a:b]
    bounds = beat.bounds(a)
    q0 = max(0, bounds.qrs_on)
    m = (beat_measures_safe(seg, rec.fs, bounds, rec.sex, reference)
         if rec.source != 'payload' or beat.accepted else None)
    scene = LoopScene(seg, rec.fs, bounds, label, cursor=q0, measures=m)
    if overlays:
        for ob in rec.dominant_beats():
            if ob.index == beat.index:
                continue
            oa, obnd = rec.beat_slice(ob)
            scene.overlays.append((rec.xyz(method, corrected)[oa:obnd], ob.bounds(oa)))
    return scene


def beat_measures_safe(seg, fs, bounds, sex, reference):
    from .measures import beat_measures
    try:
        return beat_measures(seg, fs, bounds, sex, reference)
    except Exception:
        return None


# ----------------------------------------------------------------------------- Hauptfenster
class VCGLabPanel(QWidget):
    """Vollständiges Labor als Widget: eigenes Fenster (``VCGLabWindow``) oder Registerkarte im ECG-Labor."""

    catalogReady = pyqtSignal(int, object)
    analysisReady = pyqtSignal(int, object)
    recordChanged = pyqtSignal(object)  # fertiger VCGRecord (für die Vorschaukarte des ECG-Labors)

    def __init__(self, parent: Optional[QWidget] = None, dark: bool = DEFAULT_DARK):
        super().__init__(parent)
        self._dark = bool(dark)
        self.record: Optional[VCGRecord] = None
        self.compare_record: Optional[VCGRecord] = None
        self.reference = load_reference()
        self.scene_data: Optional[LoopScene] = None
        self._catalog_run = 0
        self._analysis_run = 0
        self._pending: Optional[tuple] = None  # (signal, fs, names, label, age, sex, payload)
        self.catalogReady.connect(self._on_catalog_ready)
        self.analysisReady.connect(self._on_analysis_ready)
        self._build()

    # ---- Aufbau
    def _build(self) -> None:
        central = self
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        left = QSplitter(Qt.Orientation.Vertical)
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        self.planes = {pl: LoopPlaneWidget(pl, dark=self._dark) for pl in PLANES}
        self.view3d = Loop3DWidget(dark=self._dark)
        grid.addWidget(self.planes["frontal"], 0, 0)
        grid.addWidget(self.planes["transversal"], 0, 1)
        grid.addWidget(self.planes["sagittal"], 1, 0)
        grid.addWidget(self.view3d, 1, 1)
        for pl in self.planes.values():
            pl.cursorMoved.connect(self.set_cursor)
        legend = QLabel()
        legend.setWordWrap(True)
        self.legend = legend
        grid.addWidget(legend, 2, 0, 1, 2)
        left.addWidget(grid_w)
        self.traces = ScalarTraceWidget(dark=self._dark)
        self.traces.cursorMoved.connect(self.set_cursor)
        self.traces.setMinimumHeight(220)
        left.addWidget(self.traces)
        left.setSizes([620, 300])
        root.addWidget(left, 1)
        # ---- rechte Spalte
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(4, 0, 4, 0)
        pl.addWidget(self._section("Ansicht"))
        row = QGridLayout()
        row.addWidget(QLabel("Schlag"), 0, 0)
        self.beat_combo = QComboBox()
        self.beat_combo.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.beat_combo, 0, 1, 1, 3)
        row.addWidget(QLabel("Transformation"), 1, 0)
        self.method_combo = QComboBox()
        for m in METHODS:
            self.method_combo.addItem(METHOD_LABELS[m], m)
        self.method_combo.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.method_combo, 1, 1, 1, 3)
        self.origin_check = QCheckBox("Ursprung korrigiert")
        self.origin_check.setChecked(True)
        self.origin_check.setToolTip("Physiologischer Nullpunkt (TP/PR/TQ-Fenster des Hauptprojekts) abgezogen")
        self.origin_check.toggled.connect(self.refresh)
        row.addWidget(self.origin_check, 2, 0, 1, 2)
        self.dash_check = QCheckBox("Zeitmarken 2 ms")
        self.dash_check.setChecked(True)
        self.dash_check.toggled.connect(self.refresh)
        row.addWidget(self.dash_check, 2, 2, 1, 2)
        self.vector_check = QCheckBox("Vektoren")
        self.vector_check.setChecked(True)
        self.vector_check.setToolTip("QRS-/T-Maximalvektor, ST J+60, initiale 20 / terminale 40 ms (in mV) sowie der SVG als Richtung")
        self.vector_check.toggled.connect(self.refresh)
        row.addWidget(self.vector_check, 3, 0, 1, 2)
        self.overlay_check = QCheckBox("Alle Schläge hinterlegen")
        self.overlay_check.toggled.connect(self.refresh)
        row.addWidget(self.overlay_check, 3, 2, 1, 2)
        self.rotate_check = QCheckBox("3D drehen")
        self.rotate_check.toggled.connect(self.view3d.set_auto_rotate)
        row.addWidget(self.rotate_check, 4, 0, 1, 2)
        self.motion_check = QCheckBox("Bewegung reduzieren")
        self.motion_check.setToolTip("Stoppt automatische Kamerabewegung; Messungen und Cursor bleiben unverändert.")
        self.motion_check.setChecked(current_motion_policy().reduced)
        self.motion_check.toggled.connect(current_motion_policy().set_reduced)
        current_motion_policy().changed.connect(self.motion_check.setChecked)
        row.addWidget(self.motion_check, 6, 0, 1, 4)
        row.addWidget(QLabel("Geschlecht (Grenzwerte)"), 5, 0, 1, 2)
        self.sex_combo = QComboBox()
        for lab, v in (("unbekannt", None), ("weiblich", "F"), ("männlich", "M")):
            self.sex_combo.addItem(lab, v)
        self.sex_combo.currentIndexChanged.connect(self._sex_changed)
        row.addWidget(self.sex_combo, 5, 2, 1, 2)
        self.play_button = QToolButton()
        self.play_button.setText("▶ Abspielen")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_play)
        row.addWidget(self.play_button, 4, 2)
        self.speed = QComboBox()
        for lab, v in (("0,05×", 0.05), ("0,1×", 0.1), ("0,25×", 0.25), ("0,5×", 0.5)):
            self.speed.addItem(lab, v)
        self.speed.setCurrentIndex(1)
        row.addWidget(self.speed, 4, 3)
        pl.addLayout(row)
        self.status = QLabel("Kein ECG geladen.")
        self.status.setWordWrap(True)
        self.status.setProperty("vcgMuted", True)
        pl.addWidget(self.status)
        pl.addWidget(self._section("Schleifenlesung"))
        self.reading = ReadingPanel(dark=self._dark)
        pl.addWidget(self.reading)
        head = QHBoxLayout()
        head.addWidget(self._section("Maße dieses Schlags"), 1)
        self.export_button = QToolButton()
        self.export_button.setText("CSV …")
        self.export_button.setToolTip("Maße, Bewertung und Referenzquelle als CSV speichern")
        self.export_button.clicked.connect(self._export_measures)
        head.addWidget(self.export_button)
        pl.addLayout(head)
        self.tree = MeasureTree(dark=self._dark)
        self.tree.setMinimumHeight(320)
        pl.addWidget(self.tree, 1)
        # ---- ausklappbare Zusatzbereiche
        var_section = CollapsibleSection("Schlag-zu-Schlag-Variabilität (dominante Morphologie)")
        self.variability = QLabel("–")
        self.variability.setWordWrap(True)
        self.variability.setStyleSheet("font-size: 10px;")
        var_section.body.addWidget(self.variability)
        pl.addWidget(var_section)
        fid_section = CollapsibleSection("Transformationsgüte und Ursprung")
        self.fidelity_label = QLabel("–")
        self.fidelity_label.setWordWrap(True)
        self.fidelity_label.setStyleSheet("font-size: 10px;")
        fid_section.body.addWidget(self.fidelity_label)
        pl.addWidget(fid_section)
        cmp_section = CollapsibleSection("Serieller Vergleich (ΔST, ΔVG; experimentell)")
        cmp_row = QHBoxLayout()
        self.compare_button = QPushButton("Vergleichs-ECG (WFDB) …")
        self.compare_button.clicked.connect(self._pick_compare)
        cmp_row.addWidget(self.compare_button)
        self.compare_clear = QToolButton()
        self.compare_clear.setText("×")
        self.compare_clear.clicked.connect(self._clear_compare)
        cmp_row.addWidget(self.compare_clear)
        cmp_section.body.addLayout(cmp_row)
        self.compare_label = QLabel("Zweite Aufnahme derselben Person laden; ΔST > 0,05 mV und ΔVG > 16,2 mV·ms gelten nach ter Haar 2013 als Ischämiehinweis.")
        self.compare_label.setWordWrap(True)
        self.compare_label.setStyleSheet("font-size: 10px;")
        cmp_section.body.addWidget(self.compare_label)
        pl.addWidget(cmp_section)
        cat_section = CollapsibleSection("Markerregister des Hauptprojekts (Suche, Berechnung auf Abruf; experimentell)")
        cat_row = QHBoxLayout()
        self.catalog_search = QLineEdit()
        self.catalog_search.setPlaceholderText("Name/Familie, z. B. planarity, t_loop, kinematik")
        self.catalog_search.returnPressed.connect(self._search_catalog)
        cat_row.addWidget(self.catalog_search, 1)
        self.catalog_button = QPushButton("Berechnen")
        self.catalog_button.clicked.connect(self._search_catalog)
        cat_row.addWidget(self.catalog_button)
        cat_section.body.addLayout(cat_row)
        self.catalog_table = QTableWidget(0, 4)
        self.catalog_table.setHorizontalHeaderLabels(["Marker", "Wert", "Einheit", "Familie"])
        self.catalog_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.catalog_table.setStyleSheet("font-size: 10px;")
        self.catalog_table.setMinimumHeight(180)
        cat_section.body.addWidget(self.catalog_table)
        self.catalog_status = QLabel("")
        self.catalog_status.setProperty("vcgMuted", True)
        cat_section.body.addWidget(self.catalog_status)
        pl.addWidget(cat_section)
        pl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(430)
        scroll.setMaximumWidth(520)
        root.addWidget(scroll, 0)
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._advance)
        self.tree.vectorSelected.connect(self._vector_selected)
        self.set_dark(self._dark)

    @staticmethod
    def _section(title: str) -> QLabel:
        lab = QLabel(title)
        lab.setProperty("vcgSection", True)
        lab.setStyleSheet("font-weight: 600; margin-top: 6px;")
        return lab

    # ---- Daten aus dem ECG-Labor (Analyse erst, wenn die Registerkarte gebraucht wird)
    def set_source(self, signal, fs: float, lead_names, label: str = "", age=None, sex=None, payload: Optional[dict] = None) -> None:
        """Signal und Laborbefund hinterlegen; gerechnet wird erst bei ``ensure_record`` (Anzeige der Karte)."""
        self._analysis_run += 1
        self.record = None
        self.compare_record = None
        self.scene_data = None
        if signal is None:
            self._pending = None
            self.status.setText("Kein ECG geladen.")
            return
        self._pending = (np.asarray(signal, dtype=float), float(fs), tuple(lead_names), label, age, sex, payload or None)
        self.status.setText(f"{label or 'ECG'} geladen · Auswertung startet beim Öffnen dieser Ansicht.")
        if self.isVisible():
            self.ensure_record()

    def ensure_record(self) -> None:
        """Rechnet den VCG-Datensatz einmal je geladenem Fall (Hintergrund; Laborbefund als Rückfallebene)."""
        if self._pending is None or self.record is not None:
            return
        signal, fs, names, label, age, sex, payload = self._pending
        self._analysis_run += 1
        run = self._analysis_run
        self.status.setText(f"{label or 'ECG'} · Auswertung läuft …")

        def work() -> None:
            try:
                # A saved laboratory result already contains the production
                # delineation. Reusing it avoids a second, potentially
                # inconsistent P/QRS/T interpretation in the VCG viewer.
                out = (record_from_payload(signal, fs, names, payload, label, age, sex)
                       if payload else analyze_signal(signal, fs, names, label, age, sex))
            except Exception as exc:
                try:
                    out = record_from_payload(signal, fs, names, payload, label, age, sex) if payload else quick_record(signal, fs, names, label, age, sex)
                    out.notes.append(f"eigene Delineation nicht möglich ({type(exc).__name__}: {exc})")
                except Exception as exc2:
                    out = f"{type(exc2).__name__}: {exc2}"
            try:
                self.analysisReady.emit(run, out)
            except RuntimeError:
                pass  # Fenster inzwischen geschlossen

        threading.Thread(target=work, name="vcglab-record", daemon=True).start()

    def _on_analysis_ready(self, run: int, out) -> None:
        if run != self._analysis_run:
            return
        if isinstance(out, VCGRecord):
            self.set_record(out)
        else:
            self.status.setText(f"Auswertung nicht möglich: {out}")

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.ensure_record()

    def set_dark(self, dark: bool) -> None:
        """Alle Ansichten auf das Thema des Hauptfensters umstellen (``gui.theme``)."""
        self._dark = bool(dark)
        t = theme_tokens(self._dark)
        for pl in self.planes.values():
            pl.apply_theme(self._dark)
        self.view3d.apply_theme(self._dark)
        self.traces.apply_theme(self._dark)
        self.tree.apply_theme(self._dark)
        self.reading.apply_theme(self._dark)
        colors, vectors = SEGMENT_COLORS[self._dark], VECTOR_COLOR_SETS[self._dark]
        self.legend.setText(
            f"<span style='color:{colors['p']}'>■</span> P &nbsp; "
            f"<span style='color:{colors['qrs']}'>■</span> QRS &nbsp; "
            f"<span style='color:{colors['t']}'>■</span> T &nbsp; Punkte alle 2 ms (groß: 10 ms), "
            f"Pfeil = Laufrichtung bei 30 % des QRS &nbsp;·&nbsp; gestrichelt: "
            f"<span style='color:{vectors['qrs_max']}'>QRS-max</span>, "
            f"<span style='color:{vectors['t_max']}'>T-max</span>, "
            f"<span style='color:{vectors['st_j60']}'>ST J+60</span>, "
            f"<span style='color:{vectors['svg']}'>SVG (nur Richtung)</span>, "
            f"<span style='color:{vectors['qrs_initial_20']}'>initiale 20 ms</span>, "
            f"<span style='color:{vectors['qrs_terminal_40']}'>terminale 40 ms</span>")
        self.legend.setStyleSheet(f"font-size: 9px; color: {t.text_muted};")
        muted = f"font-size: 10px; color: {t.text_muted};"
        small = f"font-size: 10px; color: {t.text};"
        for widget, style in ((self.status, muted), (self.catalog_status, muted), (self.variability, small),
                              (self.fidelity_label, small), (self.compare_label, small)):
            widget.setStyleSheet(style)
        self.tree.setStyleSheet(f"font-size: 10px; color: {t.text}; background-color: {t.surface};")
        self.catalog_table.setStyleSheet(f"font-size: 10px; color: {t.text}; background-color: {t.surface};")
        for label in self.findChildren(QLabel):
            if label.property("vcgSection"):
                label.setStyleSheet(f"font-weight: 600; margin-top: 6px; color: {t.text_strong};")

    def stop(self) -> None:
        self._timer.stop()
        self.view3d.stop()

    def set_record(self, rec: VCGRecord) -> None:
        self.record = rec
        self.recordChanged.emit(rec)
        self.beat_combo.blockSignals(True)
        self.beat_combo.clear()
        if rec.median_beat is not None:
            self.beat_combo.addItem("Medianschlag (fiducial-gewarpt, Hauptprojekt)", -1)
        rep = rec.representative_index() if rec.beats else 0
        self.beat_combo.addItem(f"Repräsentativer Schlag #{rep + 1}", rep)
        for b in rec.beats:
            self.beat_combo.addItem(b.label, b.index)
        self.beat_combo.setCurrentIndex(0)
        self.beat_combo.blockSignals(False)
        self.sex_combo.blockSignals(True)
        self.sex_combo.setCurrentIndex({None: 0, "F": 1, "M": 2}.get(rec.sex, 0))
        self.sex_combo.blockSignals(False)
        self._compare_scene = None
        self.refresh()
        self._update_variability()
        self._update_fidelity()

    def _sex_changed(self) -> None:
        if self.record is not None:
            self.record.sex = self.sex_combo.currentData()
            self.refresh()

    def _current_beat(self) -> tuple[Optional[VCGBeat], bool]:
        idx = self.beat_combo.currentData()
        if idx is None or self.record is None:
            return None, False
        if idx == -1:
            return None, True
        return self.record.beats[int(idx)], False

    def refresh(self) -> None:
        rec = self.record
        if rec is None:
            return
        beat, median = self._current_beat()
        method = self.method_combo.currentData() or "kors"
        for pl in self.planes.values():
            pl.show_dashes = self.dash_check.isChecked()
            pl.show_vectors = self.vector_check.isChecked()
            pl.show_overlays = self.overlay_check.isChecked()
        scene = scene_for(rec, beat, method, self.origin_check.isChecked(), median, self.overlay_check.isChecked(), self.reference)
        if scene is None:
            self.status.setText("Kein Schlag verfügbar.")
            return
        self.scene_data = scene
        for pl in self.planes.values():
            pl.set_scene(scene)
        self.view3d.set_scene(scene)
        self.traces.set_scene(scene)
        self.tree.set_measures(scene.measures)
        m = scene.measures
        if m is not None:
            reading = read_beat(m, qrs_ms=m.get("qrs_duration"), criteria=rec.criteria, rhythm_class=rec.rhythm_class, rhythm_flags=rec.rhythm_flags,
                                fidelity=rec.fidelity.get(method, {}), xyz=scene.xyz, fs=rec.fs, bounds=scene.bounds, sex=rec.sex)
            self.reading.set_result(reading, f"{scene.label} · {METHOD_LABELS[method]} · Ursprung {'korrigiert' if self.origin_check.isChecked() else 'roh'}")
        else:
            self.reading.set_result(None)
        src = {"core": "Delineation des Hauptprojekts", "payload": "Laborbefund", "quick": "Schnellschätzung"}.get(rec.source, rec.source)
        o = rec.origin.get(method)
        self.status.setText(f"{rec.label} · {len(rec.beats)} Schläge, {len(rec.dominant_beats())} dominante Morphologie · Rhythmus {rec.rhythm_class} · {src}"
                            + (f" · Ursprung {o.method} ({o.confidence:.2f})" if o else "")
                            + (" · " + "; ".join(rec.notes) if rec.notes else ""))

    def set_cursor(self, k: int) -> None:
        if self.scene_data is None:
            return
        self.scene_data.cursor = int(np.clip(k, 0, len(self.scene_data.xyz) - 1))
        for pl in self.planes.values():
            pl.set_cursor(self.scene_data.cursor)
        self.view3d.set_cursor(self.scene_data.cursor)
        self.traces.set_cursor(self.scene_data.cursor)

    def _toggle_play(self, on: bool) -> None:
        self.play_button.setText("⏸ Pause" if on else "▶ Abspielen")
        if on:
            self._timer.start()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        if self.scene_data is None:
            return
        step = max(1, int(round(self.speed.currentData() * self.scene_data.fs * 0.02)))
        k = self.scene_data.cursor + step
        if k >= len(self.scene_data.xyz):
            k = 0
        self.set_cursor(k)

    def _vector_selected(self, key: str) -> None:
        if self.scene_data is None or self.scene_data.measures is None:
            return
        # Cursor auf den Zeitpunkt des gewählten Maßes setzen, wo sinnvoll
        b = self.scene_data.bounds
        if key.startswith("qrs_max"):
            v = self.scene_data.measures.vectors.get("qrs_max")
            seg = self.scene_data.xyz[b.qrs_on:b.qrs_off]
            if v is not None and len(seg):
                self.set_cursor(b.qrs_on + int(np.argmin(np.linalg.norm(seg - v, axis=1))))
        elif key.startswith("t_max") and b.t_end:
            s = (b.t_on or b.qrs_off)
            seg = self.scene_data.xyz[s:b.t_end]
            if len(seg):
                self.set_cursor(s + int(np.argmax(np.linalg.norm(seg, axis=1))))
        elif key.startswith("st_j60"):
            self.set_cursor(b.qrs_off + int(0.06 * self.scene_data.fs))
        elif key.startswith("st_j"):
            self.set_cursor(b.qrs_off)
        elif key.startswith("qrs_initial"):
            self.set_cursor(b.qrs_on + int(0.02 * self.scene_data.fs))
        elif key.startswith("qrs_terminal"):
            self.set_cursor(b.qrs_off - int(0.02 * self.scene_data.fs))

    def _export_measures(self) -> None:
        if self.scene_data is None or self.scene_data.measures is None or self.record is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Maße speichern", f"{self.record.label or 'vcg'}_masse.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        m = self.scene_data.measures
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh, delimiter=";")
                w.writerow(["record", "schlag", "transformation", "ursprung", "gruppe", "schluessel", "mass", "wert", "einheit", "bewertung", "referenz_von", "referenz_bis", "quelle", "hinweis"])
                method = self.method_combo.currentData() or "kors"
                o = self.record.origin.get(method)
                for x in m.measures:
                    lim = x.limit
                    w.writerow([self.record.label, self.scene_data.label, METHOD_LABELS[method], (o.method if o else "roh"), x.group, x.key, x.label,
                                "" if x.value is None or not np.isfinite(x.value) else f"{x.value:.6g}", x.unit, x.status,
                                "" if lim is None or lim.low is None else f"{lim.low:.6g}", "" if lim is None or lim.high is None else f"{lim.high:.6g}",
                                lim.source if lim else "", x.note])
                for seg, planes in m.inscriptions.items():
                    for plane, (kind, minor, area) in planes.items():
                        w.writerow([self.record.label, self.scene_data.label, METHOD_LABELS[method], (o.method if o else "roh"), "Drehrichtung",
                                    f"{seg}_{plane}", f"{seg.upper()} {plane}", INSCRIPTION_TEXT[kind], "", f"gegenlaeufig {minor * 100:.0f} %", "", "",
                                    "Chou; Perez-Riera 2021", f"Flaeche {area:.3f} mV^2"])
        except OSError as exc:
            self.status.setText(f"Export fehlgeschlagen: {exc}")
            return
        self.status.setText(f"Maße gespeichert: {path}")

    # ---- Zusatzbereiche
    def _update_variability(self) -> None:
        rec = self.record
        if rec is None:
            return
        method = self.method_combo.currentData() or "kors"
        per = []
        for b in rec.dominant_beats():
            try:
                per.append(rec.measures_for(b, method, self.reference))
            except Exception:
                continue
        v = beat_variability(per)
        f = lambda x, u: "–" if x is None else f"{x:.1f}{u}"  # noqa: E731
        self.variability.setText(f"{v.n_beats} Schläge · QRS-Achse SD {f(v.qrs_axis_sd_deg, '°')} · |QRS| CV {f(None if v.qrs_magnitude_cv is None else v.qrs_magnitude_cv * 100, ' %')} · "
                                 f"T-Achse SD {f(v.t_axis_sd_deg, '°')} · |T| CV {f(None if v.t_magnitude_cv is None else v.t_magnitude_cv * 100, ' %')} · "
                                 f"QRS-T-Winkel SD {f(v.qrs_t_angle_sd_deg, '°')} (Hasan 2012)")

    def _update_fidelity(self) -> None:
        rec = self.record
        if rec is None:
            return
        parts = []
        for m in METHODS:
            fid = rec.fidelity.get(m, {})
            if fid:
                parts.append(f"<b>{METHOD_LABELS[m]}</b>: r̄ {np.mean(list(fid.values())):.2f} (" + ", ".join(f"{k} {v:.2f}" for k, v in fid.items()) + ")")
            o = rec.origin.get(m)
            if o is not None:
                parts.append(f"Ursprung {m}: {o.note}; Vektor ({o.vector[0]:.3f}, {o.vector[1]:.3f}, {o.vector[2]:.3f}) mV")
        self.fidelity_label.setText("<br>".join(parts) + "<br><span style='color:#7d8895'>Korrelation gemessener und aus XYZ rekonstruierter Ableitungen (Dower-Vorwärtsmatrix).</span>")

    def _pick_compare(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Vergleichs-ECG (WFDB-Header)", "", "WFDB (*.hea)")
        if not path:
            return
        try:
            from .analysis import load_wfdb
            sig, fs, names, meta = load_wfdb(path)
            try:
                rec = analyze_signal(sig, fs, names, Path(path).stem, meta.get("age"), meta.get("sex"))
            except Exception:
                rec = quick_record(sig, fs, names, Path(path).stem)
        except Exception as exc:
            self.compare_label.setText(f"Laden fehlgeschlagen: {exc}")
            return
        self.compare_record = rec
        self._update_compare()

    def _clear_compare(self) -> None:
        self.compare_record = None
        self.compare_label.setText("Kein Vergleichs-ECG.")

    def _update_compare(self) -> None:
        if self.record is None or self.compare_record is None:
            return
        method = self.method_combo.currentData() or "kors"
        a = self.record.median_measures(method, self.reference) or self.record.measures_for(self.record.beats[self.record.representative_index()], method, self.reference)
        b = self.compare_record.median_measures(method, self.reference) or self.compare_record.measures_for(self.compare_record.beats[self.compare_record.representative_index()], method, self.reference)
        d = serial_difference(a, b)
        if d is None:
            self.compare_label.setText("Differenzvektoren nicht berechenbar (ST- oder VG-Vektor fehlt).")
            return
        flag = "<b style='color:#ff5f6d'>beide Schwellen überschritten – Ischämiehinweis</b>" if d.ischemia_flag else "unter den Schwellen"
        self.compare_label.setText(f"{self.record.label} → {self.compare_record.label}<br>ΔST {d.delta_st_mv:.3f} mV (Schwelle 0,05) · ΔVG {d.delta_vg_mv_ms:.1f} mV·ms (Schwelle 16,2) · "
                                   f"Winkel ΔST/ΔVG {'–' if d.angle_deg is None else f'{d.angle_deg:.0f}°'} (Autoren: 15 ± 14°)<br>{flag} [ter Haar 2013; Treskes 2015]")

    def _search_catalog(self) -> None:
        rec = self.record
        if rec is None or rec.delineation is None:
            self.catalog_status.setText("Markerregister nur mit Delineation des Hauptprojekts (WFDB-Datei laden).")
            return
        query = self.catalog_search.text().strip().lower()
        if not query:
            return
        try:
            from vcg_biomarker.core.biomarkers import BiomarkerRegistry
            defs = [d for d in BiomarkerRegistry.definitions() if query in d.name.lower() or query in str(d.family).lower()]
        except Exception as exc:
            self.catalog_status.setText(f"Register nicht verfügbar: {exc}")
            return
        defs = defs[:60]
        if not defs:
            self.catalog_status.setText("Kein Marker gefunden.")
            return
        self._catalog_run += 1
        run = self._catalog_run
        names = [d.name for d in defs]
        fam = {d.name: (d.unit, str(d.family)) for d in defs}
        self.catalog_status.setText(f"{len(names)} Marker werden berechnet …")
        method = self.method_combo.currentData() or "kors"

        def work() -> None:
            try:
                from vcg_biomarker.core.biomarkers import BiomarkerRegistry
                ecg = rec.median_ecg if rec.median_ecg is not None else rec.ecg
                xyz = rec.median_xyz(method, False) if rec.median_ecg is not None else rec.xyz(method, False)
                res = BiomarkerRegistry.compute_all(delineation=rec.delineation, ecg_12lead=ecg, vcg_xyz=xyz, fs=int(rec.fs), names=names,
                                                    record_ecg_12lead=rec.ecg, record_vcg_xyz=rec.xyz(method, False))
                out = [(r.name, r.value, fam.get(r.name, ("", ""))) for r in res]
            except Exception as exc:
                out = f"{type(exc).__name__}: {exc}"
            try:
                self.catalogReady.emit(run, out)
            except RuntimeError:
                pass  # Fenster inzwischen geschlossen

        threading.Thread(target=work, name="vcglab-catalog", daemon=True).start()

    def _on_catalog_ready(self, run: int, out) -> None:
        if run != self._catalog_run:
            return
        if not isinstance(out, list):
            self.catalog_status.setText(f"Berechnung fehlgeschlagen: {out}")
            return
        self.catalog_table.setRowCount(len(out))
        for i, (name, value, (unit, family)) in enumerate(out):
            txt = "–" if value is None or not np.isfinite(value) else (f"{value:.3g}" if abs(value) < 1000 else f"{value:.0f}")
            for j, s in enumerate((name, txt, unit, family)):
                self.catalog_table.setItem(i, j, QTableWidgetItem(s))
        self.catalog_status.setText(f"{len(out)} Marker berechnet (Medianzyklus, {self.method_combo.currentText()}).")

    def closeEvent(self, e) -> None:
        self.stop()
        super().closeEvent(e)


class VCGLabWindow(QMainWindow):
    """Eigenständiges Fenster um ``VCGLabPanel`` (Prüf-GUI und „Vergrößern“ aus der Vorschau)."""

    def __init__(self, parent: Optional[QWidget] = None, dark: bool = DEFAULT_DARK):
        super().__init__(parent)
        self.setWindowTitle("VCG-Analyselabor")
        self.resize(1500, 920)
        self.setStyleSheet(f"QMainWindow {{ background-color: {theme_tokens(dark).bg}; }}")
        self.panel = VCGLabPanel(self, dark=dark)
        self.setCentralWidget(self.panel)

    def set_record(self, rec: VCGRecord) -> None:
        self.panel.set_record(rec)

    def __getattr__(self, name):  # Panelzugriffe (Tests, Vorschau) transparent weiterreichen
        panel = self.__dict__.get("panel")
        if panel is not None and hasattr(panel, name):
            return getattr(panel, name)
        raise AttributeError(name)

    def closeEvent(self, e) -> None:
        self.panel.stop()
        super().closeEvent(e)
