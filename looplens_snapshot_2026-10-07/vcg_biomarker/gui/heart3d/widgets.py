"""Qt-Widgets: 3D-Ansicht, Mini-Fenster, großes Steuerfenster mit 2D-Schnitt und Rhythmusstreifen."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
import threading

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QToolButton,
    QVBoxLayout, QWidget,
)

from .activation import PATTERN_LABELS, Pattern
from .engine import COLOR_MODES, SCENARIO_LABELS, AnimationEngine, FrameData, Layers, RhythmScenario
from .model import HeartModel, default_model
from .section import SectionView
from .timeline import BeatTimeline, timeline_from_payload, timeline_from_signal
from .torso import ELECTRODES, VIRTUAL_PRESETS, DipoleSimulator, surface_point, torso_mesh
from .pathology import zone_centroid

ILLUSTRATION_NOTE = "Illustration: Zeitstruktur aus dem ECG, Mechanik aus Modellvorgaben – keine Messung."
ATTRIBUTION = "Herzmodell: BodyParts3D, © The Database Center for Life Science, CC BY-SA 2.1 JP"

CAMERA_PRESETS = {
    "Anterior": dict(elevation=12, azimuth=-90),
    "Links lateral": dict(elevation=8, azimuth=0),
    "Von links-anterior": dict(elevation=15, azimuth=-45),
    "Posterior": dict(elevation=10, azimuth=90),
    "Von oben": dict(elevation=75, azimuth=-90),
    "Apex": dict(elevation=-60, azimuth=-90),
    "Thorax (weit)": dict(elevation=10, azimuth=-70, distance=640),
}
TRACE_MV_MM = 20.0  # Höhe je mV im Raum; Breite folgt dem Standardverhältnis 25 mm/s : 10 mm/mV (1 s = 2,5 mV-Höhen)
TRACE_WIDTH_MM = 60.0  # Bezugsbreite für Beschriftungen
ECG_MM_PER_MS = 0.025  # 25 mm/s
ECG_MM_PER_MV = 10.0  # 10 mm/mV


def _overlay_gl_options() -> dict:
    """Wie ``translucent``, aber ohne Tiefentest: Overlays bleiben vor der Herzwand sichtbar."""
    from OpenGL import GL
    return {GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True, GL.GL_ALPHA_TEST: False, GL.GL_CULL_FACE: False,
            "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)}
SEGMENT_COLORS = {0: (0.55, 1.0, 0.55), 1: (0.85, 0.95, 1.0), 2: (0.85, 0.6, 1.0)}  # P grün, QRS weißblau, T violett


# ----------------------------------------------------------------------------- Zeitgeber
class AnimationClock(QObject):
    timeChanged = pyqtSignal(float)
    playingChanged = pyqtSignal(bool)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._elapsed = pg.QtCore.QElapsedTimer()
        self.time = 0.0
        self.speed = 1.0
        self.duration = 10.0
        self.loop = True
        self._last_ms = 0

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def play(self) -> None:
        if not self._timer.isActive():
            self._elapsed.start()
            self._last_ms = 0
            self._timer.start()
            self.playingChanged.emit(True)

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.playingChanged.emit(False)

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def seek(self, t: float) -> None:
        self.time = float(np.clip(t, 0.0, max(self.duration, 0.0)))
        self.timeChanged.emit(self.time)

    def set_speed(self, speed: float) -> None:
        self.speed = float(np.clip(speed, 0.05, 4.0))

    def _tick(self) -> None:
        now = self._elapsed.elapsed()
        dt = (now - self._last_ms) / 1000.0
        self._last_ms = now
        dt = min(dt, 0.1)
        t = self.time + dt * self.speed
        if t >= self.duration:
            t = 0.0 if self.loop else self.duration
            if not self.loop:
                self.pause()
        self.time = t
        self.timeChanged.emit(t)


# ----------------------------------------------------------------------------- Shader
HEART_SHADER = "heart_shaded"


def ensure_heart_shader() -> str:
    """Registriert einmalig einen Beleuchtungs-Shader mit Kopflicht, Kantenlicht und Glanz.

    pyqtgraphs eingebautes ``shaded`` beleuchtet nur von einer festen Seite und lässt
    kameragewandte Flächen dunkel; hier wird beidseitig beleuchtet (robust gegen die
    Flächenorientierung des Quellmodells).
    """
    from pyqtgraph.opengl import shaders
    if HEART_SHADER in shaders.ShaderProgram.names:
        return HEART_SHADER
    shaders.ShaderProgram(HEART_SHADER, [
        shaders.VertexShader("""
            uniform mat4 u_mvp;
            uniform mat3 u_normal;
            attribute vec4 a_position;
            attribute vec3 a_normal;
            attribute vec4 a_color;
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                v_normal = normalize(u_normal * a_normal);
                v_color = a_color;
                gl_Position = u_mvp * a_position;
            }
        """),
        shaders.FragmentShader("""
            #ifdef GL_ES
            precision mediump float;
            #endif
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                vec3 n = normalize(v_normal);
                vec3 key = normalize(vec3(0.35, 0.55, 1.0));
                vec3 fill = normalize(vec3(-0.6, -0.2, 0.6));
                float d1 = abs(dot(n, key));
                float d2 = abs(dot(n, fill));
                float rim = pow(1.0 - abs(n.z), 2.5);
                float spec = pow(d1, 28.0);
                vec3 rgb = v_color.rgb * (0.28 + 0.62 * d1 + 0.18 * d2) + vec3(0.10) * rim + vec3(0.18) * spec;
                gl_FragColor = vec4(min(rgb, vec3(1.0)), v_color.a);
            }
        """),
    ])
    return HEART_SHADER


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Vektorisierte Vertexnormalen (pyqtgraphs MeshData rechnet sie in Python-Schleifen)."""
    v = vertices.astype(np.float32)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(v)
    idx = faces.ravel()
    rep = np.repeat(fn, 3, axis=0)
    for k in range(3):
        n[:, k] = np.bincount(idx, weights=rep[:, k], minlength=len(v))
    length = np.linalg.norm(n, axis=1)
    length[length == 0] = 1.0
    return (n / length[:, None]).astype(np.float32)


def fast_update_mesh(item, meshdata, vertices: np.ndarray, colors: Optional[np.ndarray], faces_for_normals: np.ndarray) -> None:
    """Aktualisiert Positionen/Farben eines GLMeshItem ohne Neuaufbau der MeshData-Strukturen."""
    meshdata._vertexes = np.ascontiguousarray(vertices, dtype=np.float32)
    meshdata._vertexNormals = vertex_normals(meshdata._vertexes, faces_for_normals)
    meshdata._vertexesIndexedByFaces = None
    meshdata._vertexNormalsIndexedByFaces = None
    meshdata._faceNormals = None
    meshdata._faceNormalsIndexedByFaces = None
    if colors is not None:
        meshdata._vertexColors = np.ascontiguousarray(colors, dtype=np.float32)
        meshdata._vertexColorsIndexedByFaces = None
    item.meshDataChanged()


def fast_update_colors(item, meshdata, colors: np.ndarray) -> None:
    meshdata._vertexColors = np.ascontiguousarray(colors, dtype=np.float32)
    meshdata._vertexColorsIndexedByFaces = None
    item.meshDataChanged()


def _ellipsoid(centre: np.ndarray, radii: np.ndarray, n: int = 28) -> tuple[np.ndarray, np.ndarray]:
    md = pg.opengl.MeshData.sphere(rows=n, cols=n)
    v = md.vertexes() * radii + centre
    return v.astype(np.float32), md.faces().astype(np.int32)


# ----------------------------------------------------------------------------- 3D-Ansicht
class HeartGLView(QWidget):
    """Kapselt den GLViewWidget; fällt bei fehlendem OpenGL auf ein Hinweislabel zurück."""

    def __init__(self, model: HeartModel, engine: AnimationEngine, parent: Optional[QWidget] = None, *, compact: bool = False):
        super().__init__(parent)
        self.model = model
        self.engine = engine
        self.compact = compact
        self._items: dict[str, object] = {}
        self._meshdata: dict[str, object] = {}
        self._cutaway_active = False
        self._wall_faces_full = model.wall.faces
        self._wall_faces_cut = self._cutaway_faces()
        self._dark = True
        self._translucent_wall = False
        self._trace_beat_index = -1
        self._wide_active = False
        self._virtual = (108.0, -10.0, 0.0)  # φ, z, Abstand
        self.simulator = DipoleSimulator(0.5 * (model.lv_frame.center + model.rv_frame.center))
        self._sim_timeline_id = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.gl = None
        try:
            import pyqtgraph.opengl as gl
            self._gl = gl
            self.gl = gl.GLViewWidget()
            self.gl.setBackgroundColor((16, 22, 30, 255))
            shader_name = ensure_heart_shader()
            self._shader = shader_name
            self.gl.opts["distance"] = 210 if not compact else 215
            self.gl.opts["fov"] = 45
            self.gl.setCameraPosition(elevation=12, azimuth=-90)
            layout.addWidget(self.gl)
            self._build_items()
            if not compact:
                self._build_overlay_items()
        except Exception as exc:  # pragma: no cover - hängt von Treibern ab
            self.gl = None
            fallback = QLabel(f"3D-Ansicht nicht verfügbar: {exc}")
            fallback.setWordWrap(True)
            layout.addWidget(fallback)

    # ---- Aufbau
    def _cutaway_faces(self) -> np.ndarray:
        v = self.model.wall.vertices
        centroid_y = v[self.model.wall.faces][:, :, 1].mean(1)
        return self.model.wall.faces[centroid_y > -2.0]

    def _build_items(self) -> None:
        gl = self._gl
        m = self.model
        for name, part in m.parts.items():
            if name in ("lv_cavity", "rv_cavity", "la_cavity", "ra_cavity"):
                continue
            if part.kind == "static" and part.faces.size == 0:
                continue
            md = gl.MeshData(vertexes=part.vertices, faces=part.faces)
            md._vertexNormals = vertex_normals(part.vertices, part.faces)
            self._meshdata[name] = md
            if name == "wall":
                md.setVertexColors(self.engine._wall_base_colors)
                item = gl.GLMeshItem(meshdata=md, smooth=True, shader=self._shader, drawEdges=False, glOptions="opaque")
            elif part.kind == "atrium_wall":
                md.setVertexColors(np.tile(part.color, (part.n_vertices, 1)))
                item = gl.GLMeshItem(meshdata=md, smooth=True, shader=self._shader, drawEdges=False, glOptions="opaque")
            else:
                item = gl.GLMeshItem(meshdata=md, smooth=True, shader=self._shader, color=tuple(part.color), drawEdges=False, glOptions="opaque")
            self.gl.addItem(item)
            self._items[name] = item
        # Leitungssystem
        self._conduction_items = {}
        for name, path in m.conduction_paths.items():
            if len(path) < 2:
                continue
            item = gl.GLLinePlotItem(pos=path.astype(np.float32), color=(0.6, 0.6, 0.3, 0.6), width=3, antialias=True, mode="line_strip")
            item.setGLOptions("translucent")
            self.gl.addItem(item)
            self._conduction_items[name] = item
        sa_av = np.vstack([m.landmarks["sa_node"], m.landmarks["av_node"]]).astype(np.float32)
        item = gl.GLLinePlotItem(pos=sa_av, color=(0.6, 0.6, 0.3, 0.35), width=2, antialias=True, mode="line_strip")
        item.setGLOptions("translucent")
        self.gl.addItem(item)
        self._conduction_items["sa_av"] = item
        self._conduction_paths3d = {name: path for name, path in m.conduction_paths.items() if len(path) >= 2}
        self._conduction_paths3d["sa_av"] = sa_av
        self._pulses = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=(1.0, 0.95, 0.4, 0.0), size=12, pxMode=True)
        self._pulses.setGLOptions(_overlay_gl_options())
        self.gl.addItem(self._pulses)
        self._nodes = gl.GLScatterPlotItem(pos=np.vstack([m.landmarks["sa_node"], m.landmarks["av_node"]]).astype(np.float32),
                                           color=(1.0, 0.9, 0.3, 0.9), size=9, pxMode=True)
        self._nodes.setGLOptions("translucent")
        self.gl.addItem(self._nodes)
        self._origin = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=(1.0, 0.95, 0.2, 0.0), size=18, pxMode=True)
        self._origin.setGLOptions("additive")
        self.gl.addItem(self._origin)
        self._origin_halo = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=(1.0, 0.8, 0.2, 0.0), size=40, pxMode=True)
        self._origin_halo.setGLOptions("additive")
        self.gl.addItem(self._origin_halo)
        self._vector = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(0.3, 1.0, 0.5, 0.9), width=4, antialias=True, mode="line_strip")
        self._vector.setGLOptions(_overlay_gl_options())
        self._vector.setVisible(False)
        self.gl.addItem(self._vector)
        self._cone = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1.0, 0.5, 0.2, 0.8), width=2, antialias=True, mode="lines")
        self._cone.setGLOptions("translucent")
        self._cone.setVisible(False)
        self.gl.addItem(self._cone)
        # Perikardhülle (Niedervoltage-Hinweis)
        centre = 0.5 * (m.lv_frame.center + m.rv_frame.center)
        ext = np.abs(m.wall.vertices - centre).max(0) * 1.25 + 8.0
        v, f = _ellipsoid(centre, ext)
        md = gl.MeshData(vertexes=v, faces=f)
        self._effusion = gl.GLMeshItem(meshdata=md, smooth=True, shader="balloon", color=(0.75, 0.8, 0.9, 0.16), drawEdges=False, glOptions="translucent")
        self._effusion.setVisible(False)
        self.gl.addItem(self._effusion)
        self._lesion = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=(0.75, 0.45, 1.0, 0.0), size=30, pxMode=True)
        self._lesion.setGLOptions(_overlay_gl_options())
        self._lesion.setVisible(False)
        self.gl.addItem(self._lesion)
        self._lesion_label = gl.GLTextItem(pos=np.zeros(3, np.float32), text="", color=(220, 170, 255, 230), font=QFont("Segoe UI", 9))
        self._lesion_label.setVisible(False)
        self.gl.addItem(self._lesion_label)

    def _build_overlay_items(self) -> None:
        """VCG-Schleife, Elektroden, Ableitungskurven, Thorax, virtuelle Elektrode (nur große Ansicht)."""
        gl = self._gl
        overlay = _overlay_gl_options()
        self._vcg = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1, 1, 1, 0.8), width=2.5, antialias=True, mode="line_strip")
        self._vcg.setGLOptions(overlay)
        self._vcg.setVisible(False)
        self.gl.addItem(self._vcg)
        self._vcg_point = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=(1.0, 1.0, 1.0, 1.0), size=12, pxMode=True)
        self._vcg_point.setGLOptions(overlay)
        self._vcg_point.setVisible(False)
        self.gl.addItem(self._vcg_point)
        self._vcg_shadows = []
        for _ in range(3):
            item = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(0.7, 0.7, 0.8, 0.35), width=1.5, antialias=True, mode="line_strip")
            item.setGLOptions(overlay)
            item.setVisible(False)
            self.gl.addItem(item)
            self._vcg_shadows.append(item)
        self._vcg_planes = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(0.5, 0.5, 0.6, 0.25), width=1, antialias=True, mode="lines")
        self._vcg_planes.setGLOptions("translucent")
        self._vcg_planes.setVisible(False)
        self.gl.addItem(self._vcg_planes)
        # Thorax
        tv, tf = torso_mesh()
        self._torso_vertices = tv
        md = gl.MeshData(vertexes=tv, faces=tf)
        md.setVertexColors(np.tile(np.array([0.85, 0.9, 1.0, 0.10], np.float32), (len(tv), 1)))
        self._torso_md = md
        self._torso = gl.GLMeshItem(meshdata=md, smooth=True, shader=self._shader, drawEdges=False, glOptions="translucent")
        from OpenGL import GL
        torso_opts = {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: True, GL.GL_ALPHA_TEST: False, GL.GL_CULL_FACE: True,
                      "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)}
        self._torso.setGLOptions(torso_opts)  # nur die Vorderseite der Hülle zeichnen: keine Ringartefakte
        self._torso.setVisible(False)
        self.gl.addItem(self._torso)
        # Elektroden
        positions = np.vstack([e.position for e in ELECTRODES]).astype(np.float32)
        self._electrode_positions = positions
        self._electrodes = gl.GLScatterPlotItem(pos=positions, color=(0.95, 0.95, 0.95, 0.95), size=11, pxMode=True)
        self._electrodes.setGLOptions("translucent")
        self._electrodes.setVisible(False)
        self.gl.addItem(self._electrodes)
        self._electrode_labels = []
        for e, p in zip(ELECTRODES, positions):
            label = gl.GLTextItem(pos=p + np.array([0, -6, 6], np.float32), text=e.name, color=(240, 240, 240, 220), font=QFont("Segoe UI", 9))
            label.setVisible(False)
            self.gl.addItem(label)
            self._electrode_labels.append(label)
        # Ableitungskurven neben den Elektroden
        self._traces: dict[str, object] = {}
        self._trace_frames: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # origin, e_h, e_v
        self._trace_scale: dict[str, float] = {}

        def frame_at(phi: float, z: float, lift: float = 14.0):
            origin = surface_point(phi, z, lift)
            outward = origin - surface_point(phi, z, 0.0)
            outward /= np.linalg.norm(outward) + 1e-9
            e_h = np.cross([0, 0, 1.0], outward)
            e_h /= np.linalg.norm(e_h) + 1e-9
            return origin, e_h, np.array([0, 0, 1.0])

        def add_trace(lead: str, frame_axes) -> None:
            item = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(0.4, 0.9, 0.85, 0.9), width=1.6, antialias=True, mode="line_strip")
            item.setGLOptions(_overlay_gl_options())  # ohne Tiefentest: die Thoraxhülle darf die Kurven nicht verdecken
            item.setVisible(False)
            self.gl.addItem(item)
            self._traces[lead] = item
            self._trace_frames[lead] = frame_axes

        for e in ELECTRODES:
            if e.name in ("RA", "LA", "LL", "RL"):
                continue  # Extremitätenableitungen stehen im Cabrera-Kreis, nicht an den Elektroden
            origin, e_h, e_v = frame_at(e.phi_deg, e.z)
            for k, lead in enumerate(e.leads):
                add_trace(lead, (origin + e_v * (k * 24.0 - 18.0), e_h, e_v))
        # Extremitätenableitungen in der Frontalebene dort, von wo sie auf das Herz schauen (Cabrera-Kreis):
        # aVL −30°, I 0°, II +60°, aVF +90°, III +120°, aVR −150° (0° = links, positiv = nach kaudal)
        centre = 0.5 * (self.model.lv_frame.center + self.model.rv_frame.center)
        radius = 190.0
        for lead, deg in (("aVL", -30.0), ("I", 0.0), ("II", 60.0), ("aVF", 90.0), ("III", 120.0), ("aVR", -150.0)):
            ang = np.radians(deg)
            direction = np.array([np.cos(ang), 0.0, -np.sin(ang)])
            origin = centre + direction * radius + np.array([0.0, -60.0, 0.0])  # vor dem Herzen, damit es nicht verdeckt
            e_h = np.array([-np.sin(ang), 0.0, -np.cos(ang)])  # tangential zum Kreis
            e_v = direction.copy()  # Ausschlag zeigt vom Herzen weg (positiv = Ableitung sieht die Front kommen)
            add_trace(lead, (origin, e_h, e_v))
            self._trace_scale[lead] = 1.6  # Extremitätenableitungen größer, sie stehen frei im Raum
        self._einthoven = None
        self._trace_dots = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=(1.0, 0.85, 0.3, 1.0), size=7, pxMode=True)
        self._trace_dots.setGLOptions(_overlay_gl_options())
        self._trace_dots.setVisible(False)
        self.gl.addItem(self._trace_dots)
        self._trace_labels = []
        for lead, (origin, e_h, _) in self._trace_frames.items():
            label = gl.GLTextItem(pos=(origin - e_h * (TRACE_WIDTH_MM / 2 + 4)).astype(np.float32), text=lead, color=(160, 230, 220, 200), font=QFont("Segoe UI", 8))
            label.setVisible(False)
            self.gl.addItem(label)
            self._trace_labels.append(label)
        # Virtuelle Elektrode
        self._virtual_marker = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=(1.0, 0.3, 0.9, 1.0), size=14, pxMode=True)
        self._virtual_marker.setGLOptions("translucent")
        self._virtual_marker.setVisible(False)
        self.gl.addItem(self._virtual_marker)
        self._virtual_trace = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1.0, 0.45, 0.95, 0.95), width=2.0, antialias=True, mode="line_strip")
        self._virtual_trace.setGLOptions(_overlay_gl_options())
        self._virtual_trace.setVisible(False)
        self.gl.addItem(self._virtual_trace)
        self._virtual_label = gl.GLTextItem(pos=np.zeros(3, np.float32), text="virtuell", color=(255, 150, 240, 230), font=QFont("Segoe UI", 9))
        self._virtual_label.setVisible(False)
        self.gl.addItem(self._virtual_label)
        self._virtual_dot = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=(1.0, 0.85, 0.3, 1.0), size=7, pxMode=True)
        self._virtual_dot.setGLOptions(_overlay_gl_options())
        self._virtual_dot.setVisible(False)
        self.gl.addItem(self._virtual_dot)
        self._virtual_frame = None
        self._virtual_signal = None
        self._set_virtual_geometry()

    def rebuild_torso(self) -> None:
        """Nach geänderten Körpermaßen: Silhouette, Elektrodenorte, Kurvenrahmen und Dipol-Leadfield neu aufbauen."""
        if not self.has_overlays:
            return
        tv, tf = torso_mesh()
        self._torso_vertices = tv
        md = self._gl.MeshData(vertexes=tv, faces=tf)
        md.setVertexColors(np.tile(np.array([0.85, 0.9, 1.0, 0.10], np.float32), (len(tv), 1)))
        self._torso_md = md
        self._torso.setMeshData(meshdata=md)
        positions = np.vstack([e.position for e in ELECTRODES]).astype(np.float32)
        self._electrode_positions = positions
        self._electrodes.setData(pos=positions)
        for label, p in zip(self._electrode_labels, positions):
            label.setData(pos=p + np.array([0, -6, 6], np.float32))
        for e in ELECTRODES:
            if e.name in ("RA", "LA", "LL", "RL"):
                continue
            origin = surface_point(e.phi_deg, e.z, 14.0)
            outward = origin - e.position
            outward /= np.linalg.norm(outward) + 1e-9
            e_h = np.cross([0, 0, 1.0], outward)
            e_h /= np.linalg.norm(e_h) + 1e-9
            for k, lead in enumerate(e.leads):
                self._trace_frames[lead] = (origin + np.array([0, 0, 1.0]) * (k * 24.0 - 18.0), e_h, np.array([0, 0, 1.0]))
        for label, (lead, (origin, e_h, _)) in zip(self._trace_labels, self._trace_frames.items()):
            label.setData(pos=(origin - e_h * (TRACE_WIDTH_MM / 2 + 4)).astype(np.float32))
        self._set_virtual_geometry()
        self.simulator = DipoleSimulator(0.5 * (self.model.lv_frame.center + self.model.rv_frame.center))
        self._sim_timeline_id = None
        self._virtual_signal = None
        self._trace_beat_index = -1

    # ---- Anzeige
    @property
    def has_overlays(self) -> bool:
        return self.gl is not None and hasattr(self, "_vcg")

    def set_dark(self, dark: bool) -> None:
        self._dark = bool(dark)
        if self.gl is not None:
            self.gl.setBackgroundColor((16, 22, 30, 255) if dark else (245, 246, 248, 255))

    def set_camera(self, preset: str) -> None:
        if self.gl is not None and preset in CAMERA_PRESETS:
            self.gl.setCameraPosition(**CAMERA_PRESETS[preset])

    def orbit(self, degrees: float) -> None:
        if self.gl is not None:
            self.gl.orbit(degrees, 0)

    def set_wall_opacity(self, opacity: float) -> None:
        self.engine.wall_opacity = float(np.clip(opacity, 0.15, 1.0))
        translucent = self.engine.wall_opacity < 0.999
        if translucent != self._translucent_wall and self.gl is not None:
            self._items["wall"].setGLOptions("translucent" if translucent else "opaque")
            self._translucent_wall = translucent

    def set_virtual_electrode(self, phi_deg: float, z: float, offset: float) -> None:
        self._virtual = (float(phi_deg), float(z), float(offset))
        if self.has_overlays:
            self._set_virtual_geometry()
            self._virtual_signal = None
            self._trace_beat_index = -1

    def virtual_position(self) -> np.ndarray:
        phi, z, off = self._virtual
        return surface_point(phi, z, off)

    def _set_virtual_geometry(self) -> None:
        phi, z, off = self._virtual
        pos = surface_point(phi, z, off)
        origin = surface_point(phi, z, off + 14.0)
        outward = origin - pos
        outward /= np.linalg.norm(outward) + 1e-9
        e_h = np.cross([0, 0, 1.0], outward)
        e_h /= np.linalg.norm(e_h) + 1e-9
        self._virtual_frame = (origin, e_h, np.array([0, 0, 1.0]))
        self._virtual_marker.setData(pos=pos[None, :].astype(np.float32))
        self._virtual_label.setData(pos=(pos + np.array([0, -6, 6])).astype(np.float32))

    def apply_layers(self, layers: Layers) -> None:
        if self.gl is None:
            return
        for name, item in self._items.items():
            part = self.model.parts[name]
            if part.kind == "atrium_wall":
                item.setVisible(layers.atria)
            elif name in ("aorta", "aortic_arch", "pulmonary_trunk", "svc", "ivc"):
                item.setVisible(layers.vessels)
            elif name in ("lad", "rca", "lcx"):
                item.setVisible(layers.coronaries)
        for item in self._conduction_items.values():
            item.setVisible(layers.conduction_system)
        self._nodes.setVisible(layers.conduction_system)
        self._pulses.setVisible(layers.conduction_system)
        self._vector.setVisible(layers.heart_vector)
        self._cone.setVisible(layers.stall_cone)
        self._origin.setVisible(layers.origin_markers)
        self._origin_halo.setVisible(layers.origin_markers)
        self._effusion.setVisible(layers.pathology and self.engine.pathology.effusion)
        est = self.engine.pathology
        centre = zone_centroid(self.model, np.maximum(est.ischemia, est.scar)) if layers.pathology else None
        self._lesion.setVisible(centre is not None)
        self._lesion_label.setVisible(centre is not None)
        if centre is not None:
            kinds = []
            if est.ischemia.max() > 0.3:
                kinds.append("Ischämie (transmural)" if est.ischemia_kind == "transmural" else "Ischämie (subendokardial)")
            if est.scar.max() > 0.3:
                kinds.append("Narbe")
            kind = " + ".join(kinds) or "Zone"
            self._lesion.setData(pos=centre[None, :].astype(np.float32), color=(0.75, 0.45, 1.0, 0.35), size=34)
            self._lesion_label.setData(pos=(centre + np.array([0, -8, 10])).astype(np.float32), text=f"{kind}: {est.territory or 'diffus'}")
        # Mutmaßliches Gefäß (AHA-Zuordnung der Läsionszone) hervorheben
        culprit = self.engine.pathology.culprit if layers.pathology else ""
        for name in ("lad", "rca", "lcx"):
            if name in self._items:
                highlight = culprit.lower() == name
                self._items[name].setColor((1.0, 0.55, 0.15, 1.0) if highlight else tuple(self.model.parts[name].color))
        if not self.has_overlays:
            return
        self._vcg.setVisible(layers.vcg_loop)
        self._vcg_point.setVisible(layers.vcg_loop)
        for item in self._vcg_shadows:
            item.setVisible(layers.vcg_loop and layers.vcg_projections)
        self._vcg_planes.setVisible(layers.vcg_loop and layers.vcg_projections)
        self._torso.setVisible(layers.torso or layers.potential_map)
        self._electrodes.setVisible(layers.electrodes)
        for label in self._electrode_labels:
            label.setVisible(layers.electrodes)
        for item in self._traces.values():
            item.setVisible(layers.lead_traces)
        for label in self._trace_labels:
            label.setVisible(layers.lead_traces)
        if self._einthoven is not None:
            self._einthoven.setVisible(layers.lead_traces or layers.electrodes)
        self._trace_dots.setVisible(layers.lead_traces)
        for item in (self._virtual_marker, self._virtual_trace, self._virtual_label, self._virtual_dot):
            item.setVisible(layers.virtual_electrode)
        wide = layers.torso or layers.potential_map or layers.electrodes or layers.lead_traces or layers.virtual_electrode
        if wide and not self._wide_active and self.gl.opts["distance"] < 400:
            self.gl.setCameraPosition(distance=640)  # einmalig beim Einschalten der Thorax-Ebenen
        self._wide_active = wide
        self._trace_beat_index = -1

    def update_frame(self, frame: FrameData) -> None:
        if self.gl is None:
            return
        layers = self.engine.layers
        wall = self._items["wall"]
        wall_md = self._meshdata["wall"]
        if layers.cutaway != self._cutaway_active:
            wall_md._faces = self._wall_faces_cut if layers.cutaway else self._wall_faces_full
            wall_md._vertexFaces = None
            wall_md._edges = None
            self._cutaway_active = layers.cutaway
        fast_update_mesh(wall, wall_md, frame.positions["wall"], frame.colors["wall"], self._wall_faces_full)
        for name in ("ra_wall", "la_wall"):
            if name in self._items and layers.atria:
                fast_update_mesh(self._items[name], self._meshdata[name], frame.positions[name], frame.colors[name],
                                 self.model.parts[name].faces)
        for name in ("aorta", "aortic_arch"):
            if name in self._items and name in frame.positions and layers.vessels:
                fast_update_mesh(self._items[name], self._meshdata[name], frame.positions[name], None,
                                 self.model.parts[name].faces)
        # Leitungssystem: Zustand je Segment (aktiv hell, blockiert rot, sonst blass) + laufende Impulse
        state = frame.conduction or {}
        blocked = set(frame.blocked or ())
        sa_av_on = "internodal" in state or "retro_internodal" in state
        self._conduction_items["sa_av"].setData(color=(1.0, 0.95, 0.3, 0.9) if sa_av_on else (0.6, 0.6, 0.3, 0.35))
        pulses = []
        for name, item in self._conduction_items.items():
            if name == "sa_av":
                if sa_av_on:
                    f = state.get("internodal", state.get("retro_internodal", 0.0))
                    pulses.append(self._along(self._conduction_paths3d[name], f))
                continue
            f = state.get(name, state.get("retro_" + name))
            to_block = state.get(name + "_to_block")
            if name in blocked:
                item.setData(color=(1.0, 0.45, 0.4, 0.7))
                if to_block is not None:
                    pulses.append(self._along(self._conduction_paths3d[name], to_block * 0.35))
            else:
                item.setData(color=(1.0, 0.95, 0.3, 1.0) if f is not None else (0.6, 0.6, 0.3, 0.55))
                if f is not None:
                    pulses.append(self._along(self._conduction_paths3d[name], f))
        if pulses and layers.conduction_system:
            self._pulses.setData(pos=np.asarray(pulses, np.float32), color=(1.0, 0.95, 0.4, 1.0))
        else:
            self._pulses.setData(pos=np.zeros((1, 3), np.float32), color=(1.0, 0.95, 0.4, 0.0))
        # Ursprungsmarker (nur während des QRS sichtbar)
        plan = frame.plan
        if plan is not None and layers.origin_markers:
            beat = plan.beat
            active = beat.qrs_on - 0.02 <= frame.time <= beat.qrs_off + 0.02
            idx = plan.ventricular.origin_vertices
            focal_idx = plan.ventricular.focal_vertices
            if idx.size and active:
                pulse = 0.5 + 0.5 * np.sin((frame.time - beat.qrs_on) * 2 * np.pi * 8.0)
                if focal_idx.size:
                    pos = self._site_centroids(frame.positions["wall"], focal_idx)
                    self._origin.setData(pos=pos, color=(1.0, 0.95, 0.2, 1.0), size=16)
                    self._origin_halo.setData(pos=pos, color=(1.0, 0.7, 0.2, 0.25 + 0.25 * pulse), size=44)
                else:
                    pos = self._site_centroids(frame.positions["wall"], idx)
                    self._origin.setData(pos=pos, color=(1.0, 0.95, 0.3, 0.7), size=7)
                    self._origin_halo.setData(pos=pos, color=(1.0, 0.8, 0.2, 0.12), size=18)
            else:
                self._origin.setData(color=(1.0, 0.95, 0.2, 0.0))
                self._origin_halo.setData(color=(1.0, 0.75, 0.2, 0.0))
        if layers.heart_vector and frame.heart_vector is not None:
            centre = 0.5 * (self.model.lv_frame.center + self.model.rv_frame.center)
            v = np.asarray(frame.heart_vector, dtype=np.float64)
            tip = centre + v * self.engine.vcg_scale
            self._vector.setData(pos=np.vstack([centre, tip]).astype(np.float32))
        if layers.stall_cone and plan is not None:
            self._update_cone(plan)
        if self.has_overlays:
            self._update_overlays(frame)

    # ---- Overlays der großen Ansicht
    def _ensure_simulator(self) -> None:
        tl = self.engine.timeline
        if tl is not None and id(tl) != self._sim_timeline_id:
            self.simulator.set_timeline(tl)
            self._sim_timeline_id = id(tl)
            self._virtual_signal = None
            self._trace_beat_index = -1

    def _update_overlays(self, frame: FrameData) -> None:
        layers = self.engine.layers
        plan = frame.plan
        tl = self.engine.timeline
        if plan is None or tl is None:
            return
        beat = plan.beat
        if layers.vcg_loop:
            loop = self.engine.vcg_loop(beat)
            if loop is not None:
                pts = loop["points"]
                seg = loop["segment"]
                done = loop["first_sample"] + np.arange(len(pts)) <= frame.sample_index
                colors = np.empty((len(pts), 4), np.float32)
                for s_id, rgb in SEGMENT_COLORS.items():
                    colors[seg == s_id, :3] = rgb
                colors[:, 3] = np.where(done, 0.95, 0.22)
                self._vcg.setData(pos=pts, color=colors)
                k = int(np.clip(frame.sample_index - loop["first_sample"], 0, len(pts) - 1))
                self._vcg_point.setData(pos=pts[k:k + 1])
                if layers.vcg_projections:
                    c = loop["centre"]
                    # Schatten auf drei Ebenen: sagittal (x), frontal (y, posterior), horizontal (z, kaudal)
                    plane_pos = {0: c[0] - 70.0, 1: c[1] + 70.0, 2: c[2] - 80.0}
                    pairs = []
                    for axis, item in zip((0, 1, 2), self._vcg_shadows):
                        shadow = pts.astype(np.float64).copy()
                        shadow[:, axis] = plane_pos[axis]
                        item.setData(pos=shadow.astype(np.float32), color=np.column_stack([colors[:, :3] * 0.7, colors[:, 3] * 0.5]))
                        u, w = [i for i in range(3) if i != axis]
                        corners = []
                        for su, sw in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
                            q = c.astype(np.float64).copy()
                            q[axis] = plane_pos[axis]
                            q[u] += su * 55.0
                            q[w] += sw * 55.0
                            corners.append(q)
                        for j in range(4):
                            pairs.append(corners[j])
                            pairs.append(corners[(j + 1) % 4])
                    self._vcg_planes.setData(pos=np.asarray(pairs, np.float32))
        if layers.lead_traces or layers.virtual_electrode or layers.potential_map:
            self._ensure_simulator()
        if layers.lead_traces:
            if beat.index != self._trace_beat_index:
                self._rebuild_traces(beat)
            self._update_trace_dots(frame)
        if layers.virtual_electrode:
            if beat.index != self._trace_beat_index or self._virtual_signal is None:
                self._rebuild_virtual_trace(beat)
            self._update_virtual_dot(frame)
        if layers.lead_traces or layers.virtual_electrode:
            self._trace_beat_index = beat.index
        if layers.potential_map:
            self._update_potential_map(frame)
        elif layers.torso:
            colors = self._torso_md.vertexColors()
            if colors is None or colors[0, 3] != np.float32(0.10):
                fast_update_colors(self._torso, self._torso_md, np.tile(np.array([0.85, 0.9, 1.0, 0.10], np.float32), (len(self._torso_vertices), 1)))

    def _beat_window(self, beat) -> tuple[int, int]:
        tl = self.engine.timeline
        start, end = tl.cycle_bounds(beat)
        return max(0, int(start * tl.fs)), min(len(tl.signal), int(end * tl.fs))

    def _trace_width(self, a: int, b: int, scale: float = 1.0) -> float:
        """Breite der Kurve im Raum nach Standardverhältnis (1 s = 2,5 × Höhe je mV)."""
        fs = self.engine.timeline.fs
        return 2.5 * TRACE_MV_MM * scale * (b - a) / fs

    def _trace_points(self, y: np.ndarray, a: int, b: int, frame_axes, scale: float = 1.0) -> np.ndarray:
        origin, e_h, e_v = frame_axes
        n = b - a
        step = max(1, n // 220)
        ys = y[a:b:step]
        w = self._trace_width(a, b, scale)
        x = np.linspace(-w / 2, w / 2, len(ys))
        return (origin[None, :] + np.outer(x, e_h) + np.outer(ys * TRACE_MV_MM * scale, e_v)).astype(np.float32)

    def _rebuild_traces(self, beat) -> None:
        tl = self.engine.timeline
        if tl.signal is None:
            return
        a, b = self._beat_window(beat)
        names = list(tl.lead_names)
        for lead, item in self._traces.items():
            if lead not in names or b - a < 3:
                item.setData(pos=np.zeros((2, 3), np.float32))
                continue
            y = tl.signal[:, names.index(lead)]
            y = y - np.median(y[a:b])
            item.setData(pos=self._trace_points(y, a, b, self._trace_frames[lead], self._trace_scale.get(lead, 1.0)))

    def _update_trace_dots(self, frame: FrameData) -> None:
        tl = self.engine.timeline
        a, b = self._beat_window(frame.plan.beat)
        k = int(np.clip(frame.sample_index, a, b - 1))
        names = list(tl.lead_names)
        dots = []
        for lead, (origin, e_h, e_v) in self._trace_frames.items():
            if lead not in names or b - a < 3:
                continue
            y = tl.signal[:, names.index(lead)]
            yv = y[k] - np.median(y[a:b])
            sc = self._trace_scale.get(lead, 1.0)
            w = self._trace_width(a, b, sc)
            x = -w / 2 + w * (k - a) / max(b - a - 1, 1)
            dots.append(origin + e_h * x + e_v * yv * TRACE_MV_MM * sc)
        if dots:
            self._trace_dots.setData(pos=np.asarray(dots, np.float32))

    def _rebuild_virtual_trace(self, beat) -> None:
        if not self.simulator.available:
            self._virtual_trace.setData(pos=np.zeros((2, 3), np.float32))
            return
        if self._virtual_signal is None:
            self._virtual_signal = self.simulator.unipolar(self.virtual_position())
        a, b = self._beat_window(beat)
        if b - a < 3:
            return
        y = self._virtual_signal - np.median(self._virtual_signal[a:b])
        self._virtual_trace.setData(pos=self._trace_points(y, a, b, self._virtual_frame))

    def _update_virtual_dot(self, frame: FrameData) -> None:
        if self._virtual_signal is None:
            return
        a, b = self._beat_window(frame.plan.beat)
        k = int(np.clip(frame.sample_index, a, b - 1))
        origin, e_h, e_v = self._virtual_frame
        yv = self._virtual_signal[k] - np.median(self._virtual_signal[a:b])
        w = self._trace_width(a, b)
        x = -w / 2 + w * (k - a) / max(b - a - 1, 1)
        self._virtual_dot.setData(pos=(origin + e_h * x + e_v * yv * TRACE_MV_MM)[None, :].astype(np.float32))

    def _update_potential_map(self, frame: FrameData) -> None:
        if not self.simulator.available:
            return
        phi = self.simulator.surface_potentials(self._torso_vertices, frame.sample_index)
        v = np.clip(phi / 0.6, -1.0, 1.0)  # ±0,6 mV Vollausschlag
        colors = np.empty((len(v), 4), np.float32)
        pos = np.clip(v, 0, 1)
        neg = np.clip(-v, 0, 1)
        colors[:, 0] = 0.85 * (1 - neg) + 0.15
        colors[:, 1] = 0.85 * (1 - np.abs(v)) + 0.1
        colors[:, 2] = 0.85 * (1 - pos) + 0.15
        colors[:, 3] = 0.42
        fast_update_colors(self._torso, self._torso_md, colors)

    def virtual_signal(self) -> Optional[np.ndarray]:
        if not self.has_overlays:
            return None
        self._ensure_simulator()
        if self._virtual_signal is None and self.simulator.available:
            self._virtual_signal = self.simulator.unipolar(self.virtual_position())
        return self._virtual_signal

    @staticmethod
    def _along(path: np.ndarray, f: float) -> np.ndarray:
        k = float(np.clip(f, 0.0, 1.0)) * (len(path) - 1)
        i = int(np.clip(np.floor(k), 0, len(path) - 2))
        return path[i] + (path[i + 1] - path[i]) * (k - i)

    def _site_centroids(self, positions: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Fasst die Vertexgruppen der Ursprungsorte zu je einem Punkt zusammen (Cluster nach Nähe)."""
        pts = positions[idx].astype(np.float32)
        if len(pts) <= 1:
            return pts
        remaining = np.ones(len(pts), dtype=bool)
        centroids = []
        while remaining.any():
            k = int(np.argmax(remaining))
            d = np.linalg.norm(pts - pts[k], axis=1)
            group = remaining & (d < 9.0)
            centroids.append(pts[group].mean(0))
            remaining &= ~group
        return np.asarray(centroids, dtype=np.float32)

    def _update_cone(self, plan) -> None:
        tl = self.engine.timeline
        hints = tl.conduction_hints if tl else {}
        comps = [hints.get("qrs_stall_direction_leftward"), hints.get("qrs_stall_direction_inferior"), hints.get("qrs_stall_direction_posterior")]
        if any(c is None or not np.isfinite(c) for c in comps):
            self._cone.setVisible(False)
            return
        from .activation import vcg_to_model
        d = vcg_to_model(np.array([comps[0], comps[1], comps[2]]))
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            return
        d = d / norm
        centre = 0.5 * (self.model.lv_frame.center + self.model.rv_frame.center)
        # Kegel als Linienbündel (Öffnung 30°)
        a = np.cross(d, [0, 0, 1.0])
        if np.linalg.norm(a) < 1e-3:
            a = np.cross(d, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        lines = []
        for k in range(12):
            ang = 2 * np.pi * k / 12
            ray = d * np.cos(np.radians(15)) + (a * np.cos(ang) + b * np.sin(ang)) * np.sin(np.radians(15))
            lines.append(centre)
            lines.append(centre + ray * 60.0)
        self._cone.setData(pos=np.asarray(lines, dtype=np.float32))
        self._cone.setVisible(True)

    def grab_image(self):
        return self.gl.grabFramebuffer() if self.gl is not None else None


# ----------------------------------------------------------------------------- Mini-Ansicht
class MiniHeartWidget(QWidget):
    """Kleines Fenster für die Ecke des ECG-Labors; beginnt automatisch zu schlagen und dreht langsam."""

    openRequested = pyqtSignal()
    analysisReady = pyqtSignal(int, object)  # Laufnummer, Nutzlast oder Fehlertext

    def __init__(self, parent: Optional[QWidget] = None, *, model: Optional[HeartModel] = None, size: int = 250,
                 auto_analysis: bool = True):
        """auto_analysis: ohne Laborbefund die Analysekette des Hauptprojekts im Hintergrund rechnen (mit Fallback)."""
        super().__init__(parent)
        self.auto_analysis = auto_analysis
        self._analysis_run = 0
        self._external_payload = False
        self.analysisReady.connect(self._on_analysis_ready)
        self.model = model or default_model()
        self.engine = AnimationEngine(self.model)
        self.clock = AnimationClock(self)
        self.view = HeartGLView(self.model, self.engine, self, compact=True)
        self.view.setFixedSize(size, size)
        self.caption = QLabel("Kein ECG geladen")
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("font-size: 10px; color: #9aa7b5;")
        self.open_button = QPushButton("Vergrößern")
        self.open_button.setToolTip("Animation im eigenen Fenster mit Steuerung, 2D-Schnitt und Rhythmusstreifen öffnen")
        self.open_button.clicked.connect(self.openRequested.emit)
        self.rotate_check = QCheckBox("Drehen")
        from ..motion import current_motion_policy
        self.rotate_check.setChecked(not current_motion_policy().reduced)
        self.rotate_check.setStyleSheet("font-size: 10px;")
        self.rotate_check.setToolTip("Langsame automatische Drehung der Vorschau")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.view)
        layout.addWidget(self.caption)
        row = QHBoxLayout()
        row.addWidget(self.open_button, 1)
        row.addWidget(self.rotate_check)
        layout.addLayout(row)
        self.clock.timeChanged.connect(self._on_time)
        self._signal = None
        self._fs = None
        self._lead_names = None
        self._payload = None
        self._orbit_timer = QTimer(self)
        self._orbit_timer.setInterval(40)
        self._orbit_timer.timeout.connect(self._orbit_tick)
        current_motion_policy().changed.connect(self._motion_changed)
        self._window: Optional[HeartAnimationWindow] = None
        self._closing = False
        self.openRequested.connect(self.open_window)

    def _orbit_tick(self) -> None:
        from ..motion import current_motion_policy
        if self._closing or not self.isVisible() or not self.rotate_check.isChecked() or current_motion_policy().reduced:
            return
        try:
            self.view.orbit(0.25)
        except Exception:
            self._orbit_timer.stop()

    def _motion_changed(self, reduced: bool) -> None:
        if reduced:
            self.clock.pause()
            self._orbit_timer.stop()
        elif self.isVisible() and not self._closing and self.engine.timeline is not None:
            self.clock.play()
            if self.rotate_check.isChecked():
                self._orbit_timer.start()

    def stop(self) -> None:
        """Zeitgeber anhalten (vor dem Schließen/Löschen, damit kein GL-Aufruf mehr auf ein zerstörtes Fenster trifft)."""
        self._closing = True
        self._orbit_timer.stop()
        self.clock.pause()
        if self._window is not None:
            try:
                self._window.close()
            except Exception:
                pass

    def hideEvent(self, event) -> None:
        self._orbit_timer.stop()
        self.clock.pause()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        from ..motion import current_motion_policy
        if not self._closing and self.engine.timeline is not None and not current_motion_policy().reduced:
            self.clock.play()
            if self.rotate_check.isChecked():
                self._orbit_timer.start()
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        self.stop()
        super().closeEvent(event)

    # ---- Daten
    def set_signal(self, signal: np.ndarray, fs: float, lead_names=None, label: str = "") -> None:
        self._external_payload = False
        try:
            self._signal, self._fs = np.asarray(signal, dtype=np.float64), float(fs)
            self._lead_names = tuple(lead_names) if lead_names is not None else None
            timeline = timeline_from_signal(self._signal, self._fs, self._lead_names or ())
            timeline.label = label
            self._apply_timeline(timeline)
        except Exception as exc:
            self.caption.setText(f"Zeitleiste nicht verfügbar: {exc}")

    def set_analysis(self, payload: Optional[dict], analysis_fs: Optional[float] = None) -> None:
        try:
            if self._signal is None:
                return
            self._external_payload = bool(payload)
            self._analysis_run += 1  # laufende Hintergrundanalyse verwerfen
            self._payload = payload or None
            if payload:
                timeline = timeline_from_payload(payload, self._signal, self._fs, self._lead_names or ())
            else:
                timeline = timeline_from_signal(self._signal, self._fs, self._lead_names or ())
            self._apply_timeline(timeline, payload or None)
        except Exception as exc:
            self.caption.setText(f"Analysestand nicht übernommen: {exc}")

    def set_timeline(self, timeline: BeatTimeline, payload: Optional[dict] = None) -> None:
        self._external_payload = bool(payload)
        self._analysis_run += 1
        self._signal, self._fs = timeline.signal, timeline.fs
        self._lead_names = timeline.lead_names
        self._apply_timeline(timeline, payload)

    def _start_auto_analysis(self) -> None:
        """Analysekette des Hauptprojekts im Hintergrund (Delineation, Kriterien, Thesenmarker)."""
        if not self.auto_analysis or self._external_payload or self._signal is None or self._signal.shape[1] < 12:
            return
        self._analysis_run += 1
        run = self._analysis_run
        signal, fs, names = self._signal, self._fs, self._lead_names or ()
        self.caption.setText(self.caption.text() + " · Analyse des Hauptprojekts läuft …")

        def work() -> None:
            try:
                from .analysis import core_payload_from_signal
                payload = core_payload_from_signal(signal, fs, names)
            except Exception as exc:
                payload = f"{type(exc).__name__}: {exc}"
            self.analysisReady.emit(run, payload)

        threading.Thread(target=work, name="heart3d-analysis", daemon=True).start()

    def _on_analysis_ready(self, run: int, payload) -> None:
        if run != self._analysis_run or self._external_payload:
            return  # veraltet oder inzwischen vom Labor versorgt
        if not isinstance(payload, dict):
            self._refresh_caption()
            self.caption.setText(self.caption.text() + f" · Hauptprojekt-Analyse nicht verfügbar ({payload}), Schnellschätzung bleibt")
            return
        try:
            timeline = timeline_from_payload(payload, self._signal, self._fs, self._lead_names or ())
            timeline.label = self.engine.timeline.label if self.engine.timeline else ""
            t_keep = self.clock.time
            self._payload = payload
            self._apply_timeline(timeline, payload, restart=False)
            self.clock.seek(min(t_keep, self.clock.duration))
        except Exception as exc:
            self.caption.setText(f"Analysestand nicht übernommen: {exc}")

    def _apply_timeline(self, timeline: BeatTimeline, payload: Optional[dict] = None, restart: bool = True) -> None:
        self.engine.set_timeline(timeline, payload)
        self.clock.duration = max(timeline.duration, 1.0)
        if restart:
            self.clock.seek(0.0)
        self._refresh_caption()
        self.view.apply_layers(self.engine.layers)
        from ..motion import current_motion_policy
        if self.isVisible() and not current_motion_policy().reduced:
            self.clock.play()
            if self.rotate_check.isChecked():
                self._orbit_timer.start()
        if self._window is not None:
            self._window.timeline_changed()
        if restart and timeline.source != "payload":
            self._start_auto_analysis()

    def _refresh_caption(self) -> None:
        tl = self.engine.timeline
        if tl is None or not tl.beats:
            self.caption.setText("Kein Schlag erkannt")
            return
        src = ("Delineation (Hauptprojekt)" if (self.engine.payload or {}).get("lab_processing", {}).get("source") == "heart3d_core_pipeline"
               else "Delineation") if tl.source == "payload" else "Schätzung"
        estimated = sum(not beat.qrs_measured or not beat.t_measured or not beat.p_measured for beat in tl.beats) if tl.source == 'payload' else 0
        qualifier = f' · {estimated} Schläge mit illustrativ ergänzten oder unbestätigten Grenzen' if estimated else ''
        self.caption.setText(f"{len(tl.beats)} Schläge · {tl.median_hr:.0f}/min · Zeiten aus {src}{qualifier} · Illustration, keine anatomische Diagnose")

    def _on_time(self, t: float) -> None:
        if self._closing or self.engine.timeline is None or not self.isVisible() or self.view.gl is None:
            return
        # Ist das große Fenster offen, genügt der Vorschau die halbe Bildrate
        self._tick = getattr(self, "_tick", 0) + 1
        if self._window is not None and self._window.isVisible() and self._tick % 2:
            return
        try:
            frame = self.engine.frame(t)
            self.view.update_frame(frame)
        except Exception as exc:  # Anzeige darf das Labor nie stören
            self.caption.setText(f"Animation angehalten: {exc}")
            self.clock.pause()

    def set_dark(self, dark: bool) -> None:
        self.view.set_dark(dark)
        if self._window is not None:
            self._window.set_dark(dark)

    def open_window(self) -> None:
        if self._window is None:
            self._window = HeartAnimationWindow(self.model, self.engine, self.clock, parent=self.window())
            self._window.destroyed.connect(lambda: setattr(self, "_window", None))
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()


# ----------------------------------------------------------------------------- Rhythmusstreifen
class RhythmStrip(pg.PlotWidget):
    seekRequested = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setBackground((16, 22, 30))
        self.setLabel("bottom", "Zeit", units="s")
        self.setLabel("left", "II", units="mV")
        self.setMouseEnabled(x=True, y=False)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.curve = self.plot(pen=pg.mkPen("#4fd1c5", width=1.2))
        self.cursor = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#ffd447", width=2))
        self.addItem(self.cursor)
        self._beat_items: list = []
        self.scene().sigMouseClicked.connect(self._clicked)

    def set_timeline(self, timeline: Optional[BeatTimeline]) -> None:
        for item in self._beat_items:
            self.removeItem(item)
        self._beat_items.clear()
        if timeline is None or timeline.signal is None:
            self.curve.setData([], [])
            return
        names = list(timeline.lead_names)
        lead = names.index("II") if "II" in names else 0
        y = timeline.signal[:, lead]
        t = np.arange(len(y)) / timeline.fs
        self.curve.setData(t, y)
        self.setXRange(0, min(timeline.duration, 10.0), padding=0.01)
        for beat in timeline.beats:
            color = "#ff7b7b" if beat.beat_type == "V" else ("#ffd447" if beat.beat_type == "P" else "#8fb3ff")
            marker = pg.InfiniteLine(pos=beat.qrs_on, angle=90, pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine))
            self.addItem(marker)
            self._beat_items.append(marker)
            label = pg.TextItem(beat.beat_type, color=color, anchor=(0.5, 1.0))
            label.setPos(beat.r_time, float(np.nanmax(y)) if len(y) else 1.0)
            self.addItem(label)
            self._beat_items.append(label)
        if timeline.atrial_events:
            for p_on, _ in timeline.atrial_events:
                marker = pg.InfiniteLine(pos=p_on, angle=90, pen=pg.mkPen("#b8ff8f", width=1, style=Qt.PenStyle.DashLine))
                self.addItem(marker)
                self._beat_items.append(marker)

    def set_time(self, t: float) -> None:
        self.cursor.setPos(t)
        view = self.getViewBox().viewRange()[0]
        if t > view[1] or t < view[0]:
            width = view[1] - view[0]
            self.setXRange(t - 0.1 * width, t + 0.9 * width, padding=0)

    def _clicked(self, event) -> None:
        pos = event.scenePos()
        if not self.sceneBoundingRect().contains(pos):
            return
        point = self.getViewBox().mapSceneToView(pos)
        self.seekRequested.emit(float(point.x()))


class LeadGridWidget(pg.GraphicsLayoutWidget):
    """12 Ableitungen als Raster: gemessen (türkis) gegen simuliert (orange), Titel mit r je Ableitung."""

    NAMES = ("I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setBackground((16, 22, 30))
        self.setMinimumHeight(300)
        self.plots: dict[str, tuple] = {}
        for k, name in enumerate(self.NAMES):
            plot = self.addPlot(row=k // 4, col=k % 4)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.setTitle(name, size="8pt")
            plot.getViewBox().setAspectLocked(True, ratio=1.0)  # mm-Einheiten: 25 mm/s, 10 mm/mV
            meas = plot.plot(pen=pg.mkPen("#4fd1c5", width=1.3), connect="finite")
            sim = plot.plot(pen=pg.mkPen("#ff9f43", width=1.3), connect="finite")
            self.plots[name] = (plot, meas, sim)

    def show(self, t: np.ndarray, measured: np.ndarray, simulated: Optional[np.ndarray], r_leads: Optional[dict] = None,
             lead_order=("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")) -> None:
        x_mm = np.asarray(t, dtype=float) * ECG_MM_PER_MS
        for j, name in enumerate(lead_order):
            plot, meas, sim = self.plots[name]
            meas.setData(x_mm, measured[:, j] * ECG_MM_PER_MV)
            if simulated is not None and len(simulated) >= len(t):
                sim.setData(x_mm, simulated[:len(t), j] * ECG_MM_PER_MV)
            else:
                sim.setData([], [])
            r = (r_leads or {}).get(name)
            if r is None:
                plot.setTitle(name, size="8pt")
            else:
                colour = "#7ee787" if r >= 0.7 else ("#ffd447" if r >= 0.4 else "#ff7b7b")
                plot.setTitle(f'{name} <span style="color:{colour}">r {r:+.2f}</span>', size="8pt")
            plot.enableAutoRange()


class CollapsibleSection(QWidget):
    """Ausklappbarer Bereich (Kopfzeile mit Pfeil, Inhalt standardmäßig eingeklappt)."""

    def __init__(self, title: str, parent: Optional[QWidget] = None, expanded: bool = False):
        super().__init__(parent)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: 600; margin-top: 6px; }")
        self.toggle.toggled.connect(self._toggled)
        self.content = QWidget()
        self.content.setVisible(expanded)
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(6, 0, 0, 0)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.toggle)
        outer.addWidget(self.content)

    def _toggled(self, on: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.content.setVisible(on)


# ----------------------------------------------------------------------------- Großes Fenster
class HeartAnimationWindow(QMainWindow):
    sweepProgress = pyqtSignal(int, int, int, object)  # Lauf, k, n, Zwischenergebnis
    sweepFinished = pyqtSignal(int, object)  # Lauf, Ergebnisliste oder Fehlertext

    def __init__(self, model: HeartModel, engine: AnimationEngine, clock: AnimationClock, parent: Optional[QWidget] = None,
                 *, research_mode: bool = False):
        super().__init__(parent)
        self.research_mode = research_mode
        self._sweep_run = 0
        self._sweep_results = []
        self.sweepProgress.connect(self._on_sweep_progress)
        self.sweepFinished.connect(self._on_sweep_finished)
        self.setWindowTitle("Elektromechanische Herzillustration")
        self.resize(1480, 900)
        self.model, self.engine, self.clock = model, engine, clock
        self.view = HeartGLView(model, engine, self)
        self.view.setMinimumSize(560, 480)
        self.section = SectionView(model, self)
        self.section.setMinimumWidth(320)
        self.strip = RhythmStrip(self)
        self.strip.setMinimumHeight(150)
        self.strip.seekRequested.connect(self.clock.seek)
        central = QWidget()
        outer = QVBoxLayout(central)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.section)
        self.splitter.addWidget(self._side_panel())
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setStretchFactor(2, 2)
        outer.addWidget(self.splitter, 1)
        outer.addLayout(self._transport())
        outer.addWidget(self.strip)
        note = QLabel(f"{ILLUSTRATION_NOTE}  ·  {ATTRIBUTION}")
        note.setStyleSheet("color: #9aa7b5; font-size: 10px;")
        note.setWordWrap(True)
        outer.addWidget(note)
        self.setCentralWidget(central)
        self.clock.timeChanged.connect(self._on_time)
        self.clock.playingChanged.connect(lambda playing: self.play_button.setText("Pause" if playing else "Abspielen"))
        self.play_button.setText("Pause" if self.clock.playing else "Abspielen")
        self.timeline_changed()
        self.view.apply_layers(self.engine.layers)

    # ---- Aufbau
    def _transport(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.play_button = QPushButton("Pause")
        self.play_button.clicked.connect(self.clock.toggle)
        row.addWidget(self.play_button)
        prev_button = QPushButton("← Schlag")
        prev_button.clicked.connect(lambda: self._step_beat(-1))
        row.addWidget(prev_button)
        next_button = QPushButton("Schlag →")
        next_button.clicked.connect(lambda: self._step_beat(1))
        row.addWidget(next_button)
        row.addWidget(QLabel("Geschwindigkeit"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(5, 200)
        self.speed_slider.setValue(int(self.clock.speed * 100))
        self.speed_slider.setFixedWidth(180)
        self.speed_slider.valueChanged.connect(self._speed_changed)
        row.addWidget(self.speed_slider)
        self.speed_label = QLabel(f"{self.clock.speed:.2f}×")
        self.speed_label.setFixedWidth(50)
        row.addWidget(self.speed_label)
        for text, value in (("0,1×", 0.1), ("0,25×", 0.25), ("1×", 1.0)):
            button = QToolButton()
            button.setText(text)
            button.clicked.connect(lambda _=False, v=value: self.speed_slider.setValue(int(v * 100)))
            row.addWidget(button)
        snapshot = QToolButton()
        snapshot.setText("Bild speichern")
        snapshot.setToolTip("3D-Ansicht als PNG speichern")
        snapshot.clicked.connect(self._save_snapshot)
        row.addWidget(snapshot)
        row.addStretch()
        self.time_label = QLabel("0,00 s")
        self.time_label.setFont(QFont("Consolas", 11))
        row.addWidget(self.time_label)
        self.phase_label = QLabel("")
        self.phase_label.setStyleSheet("color: #ffd447; font-weight: 600;")
        row.addWidget(self.phase_label)
        return row

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        panel_root = QVBoxLayout(panel)
        panel_root.setContentsMargins(4, 0, 4, 0)
        # Blöcke werden einzeln gebaut und am Ende in der Reihenfolge Ansicht → Muster → Hinweis → Hypothesen (ausklappbar) angeordnet
        blocks: dict[str, QVBoxLayout] = {}

        def block(name: str) -> QVBoxLayout:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            blocks[name] = lay
            lay._widget = w  # type: ignore[attr-defined]
            return lay

        root_layout = block("model")
        # ---- kompakter Hinweis, was gerade gezeigt wird (der ausführliche Befund gehört ins ECG-Labor)
        self.findings = QLabel("Kein ECG geladen.")
        self.findings.setWordWrap(True)
        self.findings.setTextFormat(Qt.TextFormat.RichText)
        self.findings.setStyleSheet("font-size: 10px; color: #c9d1d9; background: #131a24; border: 1px solid #2a3542; padding: 4px;")
        block("hint").addWidget(self.findings)
        # ---- Hypothesenvergleich (ausklappbar, unten)
        head = QHBoxLayout()
        self.rerun_button = QToolButton()
        self.rerun_button.setText("Neu berechnen (mittlerer Schlag)")
        self.rerun_button.clicked.connect(lambda: self._start_sweep(representative=True))
        head.addWidget(self.rerun_button)
        head.addStretch(1)
        root_layout.addLayout(head)
        self.sweep_status = QLabel("Noch keine Simulation.")
        self.sweep_status.setWordWrap(True)
        self.sweep_status.setStyleSheet("font-size: 10px; color: #9aa7b5;")
        root_layout.addWidget(self.sweep_status)
        self.sweep_table = QTableWidget(0, 6)
        self.sweep_table.setHorizontalHeaderLabels(["#", "Hypothese", "Passung", "r QRS 12-Kanal", "r QRS VCG", "r T-Welle"])
        header = self.sweep_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col, width in ((0, 26), (2, 58), (3, 74), (4, 66), (5, 62)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.sweep_table.setColumnWidth(col, width)
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sweep_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sweep_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sweep_table.setWordWrap(False)
        self.sweep_table.setMinimumHeight(150)
        self.sweep_table.setMaximumHeight(210)
        self.sweep_table.setStyleSheet("font-size: 10px;")
        self.sweep_table.itemSelectionChanged.connect(self._sweep_selection_changed)
        root_layout.addWidget(self.sweep_table)
        apply_row = QHBoxLayout()
        self.apply_button = QPushButton("Auswahl in der Animation zeigen")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply_selected_hypothesis)
        apply_row.addWidget(self.apply_button, 1)
        self.duel_button = QPushButton("Zwei Hypothesen vergleichen")
        self.duel_button.setEnabled(False)
        self.duel_button.setToolTip("Gleicher gemessener Schlag, zwei vorhandene Simulationen; keine neue Analyse")
        self.duel_button.clicked.connect(self._open_hypothesis_duel)
        apply_row.addWidget(self.duel_button)
        self.reset_button = QPushButton("Automatik")
        self.reset_button.setToolTip("Zurück zur automatischen Musterwahl aus der Analyse")
        self.reset_button.clicked.connect(self._reset_hypothesis)
        apply_row.addWidget(self.reset_button)
        root_layout.addLayout(apply_row)
        self.lead_grid = LeadGridWidget()
        root_layout.addWidget(self.lead_grid)
        self.sweep_detail = QLabel("")
        self.sweep_detail.setWordWrap(True)
        self.sweep_detail.setStyleSheet("font-size: 10px; color: #9aa7b5;")
        root_layout.addWidget(self.sweep_detail)
        # ---- Ansicht, Darstellung, Ebenen (ganz oben, immer sichtbar)
        layout = block("view")
        layout.addWidget(self._section("Ansicht"))
        camera_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(list(CAMERA_PRESETS))
        self.camera_combo.currentTextChanged.connect(self.view.set_camera)
        camera_row.addWidget(self.camera_combo)
        self.orbit_check = QCheckBox("Drehen")
        from ..motion import current_motion_policy
        self.orbit_check.setChecked(not current_motion_policy().reduced)
        camera_row.addWidget(self.orbit_check)
        self.section_check = QCheckBox("2D-Schnitt")
        self.section_check.setChecked(True)
        self.section_check.toggled.connect(self.section.setVisible)
        camera_row.addWidget(self.section_check)
        layout.addLayout(camera_row)
        self._orbit_timer = QTimer(self)
        self._orbit_timer.setInterval(40)
        self._orbit_timer.timeout.connect(
            lambda: self.view.orbit(0.2) if self.orbit_check.isChecked() and not current_motion_policy().reduced else None
        )
        current_motion_policy().changed.connect(lambda reduced: self._orbit_timer.stop() if reduced else self._orbit_timer.start())
        if not current_motion_policy().reduced:
            self._orbit_timer.start()

        layout.addWidget(self._section("Darstellung"))
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Farben"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(list(COLOR_MODES))
        self.color_combo.currentTextChanged.connect(self._color_mode_changed)
        style_row.addWidget(self.color_combo, 1)
        style_row.addWidget(QLabel("Wand"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(15, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setToolTip("Durchsichtigkeit der Kammerwand")
        self.opacity_slider.valueChanged.connect(lambda v: (self.view.set_wall_opacity(v / 100.0), self._on_time(self.clock.time)))
        style_row.addWidget(self.opacity_slider, 1)
        layout.addLayout(style_row)

        self.layer_checks: dict[str, QCheckBox] = {}
        groups = (
            ("Herz", (("excitation", "Erregung (gelb)"), ("repolarization", "Repolarisation (blau)"),
                      ("mechanics", "Mechanik"), ("isochrones", "Isochronen"),
                      ("conduction_system", "Leitungssystem"), ("origin_markers", "Ursprungsmarker"),
                      ("pathology", "Pathologie"), ("cutaway", "Schnitt (anteriore Wand aus)"))),
            ("Anatomie", (("atria", "Vorhöfe"), ("vessels", "Große Gefäße"), ("coronaries", "Koronarien"))),
            ("Vektor", (("heart_vector", "Herzvektor (Kors)"), ("vcg_loop", "VCG-Schleife"),
                        ("vcg_projections", "VCG-Projektionen"), ("stall_cone", "Stall-Kegel (Hypothese)"))),
            ("Thorax", (("torso", "Silhouette"), ("electrodes", "Klebeelektroden"),
                        ("lead_traces", "Ableitungskurven im Raum"), ("potential_map", "Potentialkarte (Dipol)"),
                        ("virtual_electrode", "Virtuelle Elektrode"))),
        )
        for title, entries in groups:
            layout.addWidget(self._section(f"Ebenen · {title}"))
            grid = QGridLayout()
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(1)
            for k, (attr, text) in enumerate(entries):
                check = QCheckBox(text)
                check.setChecked(getattr(self.engine.layers, attr))
                check.toggled.connect(lambda state, a=attr: self._layer_toggled(a, state))
                self.layer_checks[attr] = check
                grid.addWidget(check, k // 2, k % 2)
            layout.addLayout(grid)
            if title == "Vektor":
                vcg_row = QHBoxLayout()
                vcg_row.addWidget(QLabel("VCG-Maßstab"))
                self.vcg_scale_slider = QSlider(Qt.Orientation.Horizontal)
                self.vcg_scale_slider.setRange(20, 300)
                self.vcg_scale_slider.setValue(100)
                self.vcg_scale_slider.setToolTip("Größe der VCG-Schleife und des Herzvektors (100 % = Spitzenvektor 45 mm)")
                self.vcg_scale_slider.valueChanged.connect(self._vcg_scale_changed)
                vcg_row.addWidget(self.vcg_scale_slider, 1)
                self.vcg_scale_label = QLabel("100 %")
                self.vcg_scale_label.setFixedWidth(44)
                vcg_row.addWidget(self.vcg_scale_label)
                layout.addLayout(vcg_row)

        block("misc")
        ve_section = CollapsibleSection("Virtuelle Elektrode (Dipolnäherung, experimentell)")
        blocks["virtual"] = QVBoxLayout()
        blocks["virtual"].addWidget(ve_section)
        layout = ve_section.body
        ve_row = QHBoxLayout()
        self.virtual_preset = QComboBox()
        self.virtual_preset.addItems(list(VIRTUAL_PRESETS) + ["Frei"])
        self.virtual_preset.currentTextChanged.connect(self._virtual_preset_changed)
        ve_row.addWidget(self.virtual_preset, 1)
        layout.addLayout(ve_row)
        ve_grid = QGridLayout()
        self.virtual_sliders: dict[str, QSlider] = {}
        for r, (key, text, lo, hi, val) in enumerate((("phi", "Winkel (° um den Thorax, 0 = vorn, + links)", -180, 180, 108),
                                                     ("z", "Höhe (mm)", -300, 200, -10), ("off", "Abstand von der Haut (mm)", 0, 80, 0))):
            ve_grid.addWidget(QLabel(text), r, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(val)
            slider.valueChanged.connect(self._virtual_slider_changed)
            self.virtual_sliders[key] = slider
            ve_grid.addWidget(slider, r, 1)
        layout.addLayout(ve_grid)
        self.virtual_plot = pg.PlotWidget()
        self.virtual_plot.setBackground((16, 22, 30))
        self.virtual_plot.setFixedHeight(150)
        self.virtual_plot.showGrid(x=True, y=True, alpha=0.15)
        self.virtual_plot.setLabel("left", "mV")
        self.virtual_plot.addLegend(offset=(4, 4))
        self.virtual_curve = self.virtual_plot.plot(pen=pg.mkPen("#ff66f0", width=1.6), name="virtuell (simuliert)")
        self.reference_meas = self.virtual_plot.plot(pen=pg.mkPen("#4fd1c5", width=1.2), name="Referenz gemessen")
        self.reference_sim = self.virtual_plot.plot(pen=pg.mkPen("#4fd1c5", width=1.0, style=Qt.PenStyle.DashLine), name="Referenz simuliert")
        self.virtual_cursor = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#ffd447", width=1))
        self.virtual_plot.addItem(self.virtual_cursor)
        layout.addWidget(self.virtual_plot)
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Referenz"))
        self.reference_combo = QComboBox()
        self.reference_combo.addItems(["V1", "V2", "V3", "V4", "V5", "V6"])
        self.reference_combo.setCurrentText("V4")
        self.reference_combo.currentTextChanged.connect(lambda _: self._refresh_virtual_plot(force=True))
        ref_row.addWidget(self.reference_combo)
        self.fit_label = QLabel("")
        self.fit_label.setStyleSheet("font-size: 10px; color: #9aa7b5;")
        self.fit_label.setWordWrap(True)
        ref_row.addWidget(self.fit_label, 1)
        layout.addLayout(ref_row)
        layout = block("pattern")

        layout.addWidget(self._section("Erregungsmuster und Rhythmus"))
        self.pattern_combo = QComboBox()
        for pattern in Pattern:
            self.pattern_combo.addItem(PATTERN_LABELS[pattern], pattern)
        self.pattern_combo.currentIndexChanged.connect(self._pattern_changed)
        layout.addWidget(self.pattern_combo)
        self.scenario_combo = QComboBox()
        for scenario in RhythmScenario:
            self.scenario_combo.addItem(SCENARIO_LABELS[scenario], scenario)
        self.scenario_combo.setToolTip("Sonderrhythmen: Ersatzrhythmen, AV-Knoten-Reentry, AVRT über Bahn, Kammertachykardie um Narbe")
        self.scenario_combo.currentIndexChanged.connect(self._scenario_changed)
        layout.addWidget(self.scenario_combo)
        self.thesis_check = QCheckBox("Thesen-Modulation aus der Analyse (Hypothese)")
        self.thesis_check.setChecked(self.engine.use_thesis_modulation)
        self.thesis_check.setToolTip("Zeitverzerrung aus Bogenlängenanteil und Durchbruchzahl aus früher Aufhebung; "
                                     "verändert nur die Zeitachse, nie die Reihenfolge.")
        self.thesis_check.toggled.connect(self._thesis_toggled)
        layout.addWidget(self.thesis_check)

        pd_section = CollapsibleSection("Patientendaten (optional: Thoraxmaße, Herzlage-Vorannahme)")
        blocks["patient"] = QVBoxLayout()
        blocks["patient"].addWidget(pd_section)
        layout = pd_section.body
        pd_row = QGridLayout()
        self.patient_fields = {}
        for k, (key, text, lo, hi, default) in enumerate((("age", "Alter (J)", 0, 110, 0), ("height", "Größe (cm)", 0, 220, 0), ("weight", "Gewicht (kg)", 0, 250, 0))):
            pd_row.addWidget(QLabel(text), 0, 2 * k)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(default)
            spin.setSpecialValueText("–")
            spin.valueChanged.connect(self._patient_changed)
            self.patient_fields[key] = spin
            pd_row.addWidget(spin, 0, 2 * k + 1)
        pd_row.addWidget(QLabel("Geschlecht"), 1, 0)
        self.patient_sex = QComboBox()
        self.patient_sex.addItems(["–", "männlich", "weiblich"])
        self.patient_sex.currentTextChanged.connect(self._patient_changed)
        pd_row.addWidget(self.patient_sex, 1, 1)
        self.patient_note = QLabel("Referenzthorax (175 cm, BMI 24)")
        self.patient_note.setStyleSheet("font-size: 10px; color: #9aa7b5;")
        self.patient_note.setWordWrap(True)
        pd_row.addWidget(self.patient_note, 1, 2, 1, 4)
        layout.addLayout(pd_row)

        inv_section = CollapsibleSection("Experimentell: Was-wäre-wenn-Editor, Stimulationsvorhersage, Einstellungen")
        blocks["experimental"] = QVBoxLayout()
        blocks["experimental"].addWidget(inv_section)
        layout = inv_section.body
        inv_row = QHBoxLayout()
        self.sweep_button = QPushButton("Hypothesen durchspielen (aktueller Schlag)")
        self.sweep_button.setToolTip("Erregungshypothesen (Leitungssystem, Blockhöhe, Fokusorte, Bahnen, Stimulation) simulieren, "
                                     "als 12 Ableitungen vorwärts rechnen und mit dem gemessenen QRS vergleichen")
        self.sweep_button.clicked.connect(lambda: self._start_sweep(representative=False))
        inv_row.addWidget(self.sweep_button, 1)
        layout.addLayout(inv_row)
        self.personalize_check = QCheckBox("Herzlage je Aufzeichnung mitoptimieren (±15°, Hoekema 1999; in der Validierung ohne Gewinn)")
        self.personalize_check.setChecked(False)
        self.personalize_check.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.personalize_check)
        ed_row = QHBoxLayout()
        ed_row.addWidget(QLabel("Was-wäre-wenn-Editor (Klick im 2D-Schnitt)"))
        self.editor_mode = QComboBox()
        self.editor_mode.addItems(["aus", "Fokus setzen", "Block/Verzögerung an Bahn", "Akzessorische Bahn setzen",
                                   "Kreis-Zentrum (Narbe) setzen", "Stimulationsort setzen"])
        self.editor_mode.setToolTip("Im 2D-Schnitt klicken: Fokus/Bahn/Elektrode auf die Wand setzen, oder eine Leitungsbahn "
                                    "anklicken (aus → 40 ms verzögert → blockiert → aus); die 12 Ableitungen werden sofort simuliert")
        ed_row.addWidget(self.editor_mode, 1)
        self.editor_clear = QToolButton()
        self.editor_clear.setText("Editor leeren")
        self.editor_clear.clicked.connect(self._editor_clear)
        ed_row.addWidget(self.editor_clear)
        layout.addLayout(ed_row)
        pace_row = QHBoxLayout()
        self.pace_button = QPushButton("Stimulation vorhersagen (His / LBBAP / CRT)")
        self.pace_button.setToolTip("QRS-Dauer und Erregung unter His-Bündel-, Linksschenkel-Areal- und biventrikulärer Stimulation "
                                    "für die beste Leitungshypothese simulieren (Upadhyay 2019, Strocchi 2023)")
        self.pace_button.clicked.connect(self._predict_pacing)
        pace_row.addWidget(self.pace_button, 1)
        self.pace_apply = QComboBox()
        self.pace_apply.addItems(["Vorhersage nur anzeigen", "His-Stimulation zeigen", "LBBAP zeigen", "CRT zeigen"])
        self.pace_apply.currentIndexChanged.connect(self._apply_pacing_choice)
        pace_row.addWidget(self.pace_apply)
        layout.addLayout(pace_row)
        self.pace_status = QLabel("")
        self.pace_status.setWordWrap(True)
        self.pace_status.setStyleSheet("font-size: 10px; color: #9aa7b5;")
        layout.addWidget(self.pace_status)
        self.sweep_plot = pg.PlotWidget()
        self.sweep_plot.setBackground((16, 22, 30))
        self.sweep_plot.setFixedHeight(150)
        self.sweep_plot.showGrid(x=True, y=True, alpha=0.15)
        self.sweep_plot.setLabel("left", "mm (10 mm/mV)")
        self.sweep_plot.setLabel("bottom", "mm (25 mm/s), 0 = QRS-Beginn")
        self.sweep_plot.getViewBox().setAspectLocked(True, ratio=1.0)
        self.sweep_plot.addLegend(offset=(4, 4))
        self.sweep_meas = self.sweep_plot.plot(pen=pg.mkPen("#4fd1c5", width=1.4), name="gemessen", connect="finite")
        self.sweep_sim = self.sweep_plot.plot(pen=pg.mkPen("#ff9f43", width=1.4), name="simuliert", connect="finite")
        layout.addWidget(self.sweep_plot)
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel("Ableitung"))
        self.sweep_lead = QComboBox()
        self.sweep_lead.addItems(["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"])
        self.sweep_lead.setCurrentText("V1")
        self.sweep_lead.currentTextChanged.connect(lambda _: self._sweep_selection_changed())
        sl_row.addWidget(self.sweep_lead)
        sl_row.addStretch(1)
        layout.addLayout(sl_row)
        detail_section = CollapsibleSection("Dieser Schlag im Detail (Zeiten, Marker, Hinweise)")
        blocks["detail"] = QVBoxLayout()
        blocks["detail"].addWidget(detail_section)
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(220)
        self.info.setStyleSheet("font-size: 11px;")
        detail_section.body.addWidget(self.info)
        # ---- Zusammenbau: Ansicht → Muster → Hinweis → Hypothesen (ausklappbar) → Experimentelles → Rest
        model_section = CollapsibleSection("Erregungsmodell im Vergleich zum ECG (Hypothesen, Stufe H)")
        model_section.body.addWidget(blocks["model"]._widget)
        for name in ("view", "pattern", "hint"):
            panel_root.addWidget(blocks[name]._widget)
        panel_root.addWidget(model_section)
        for name in ("experimental", "virtual", "patient", "detail"):
            panel_root.addLayout(blocks[name])
        if not self.research_mode:
            # The patient illustration no longer doubles as an inverse simulator.
            model_section.hide()
            blocks["pattern"]._widget.hide()
            for name in ("experimental", "virtual", "patient"):
                for index in range(blocks[name].count()):
                    widget = blocks[name].itemAt(index).widget()
                    if widget is not None:
                        widget.hide()
            simulation_button = QPushButton("Eigenständige Simulationswerkbank öffnen")
            simulation_button.clicked.connect(self._open_simulation_workspace)
            panel_root.addWidget(simulation_button)
        panel_root.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._virtual_plot_beat = -1
        self._info_beat = -1
        self._info_dirty = True
        self._editor = {"focus": None, "delays": {}, "accessory": None, "centre": None, "pacing": None}
        self._fitter = None
        self._pacing_results = {}
        self._auto_sweep_timeline = None
        self._atrial_results = []
        self.section.clicked.connect(self._section_clicked)
        return scroll

    @staticmethod
    def _section(title: str) -> QLabel:
        label = QLabel(title)
        label.setStyleSheet("font-weight: 600; margin-top: 6px;")
        return label

    # ---- Steuerung
    def set_dark(self, dark: bool) -> None:
        self.view.set_dark(dark)
        self.section.set_dark(dark)

    def _speed_changed(self, value: int) -> None:
        self.clock.set_speed(value / 100.0)
        self.speed_label.setText(f"{value / 100.0:.2f}×")

    def _layer_toggled(self, attr: str, state: bool) -> None:
        self._info_dirty = True
        if attr == "pathology":
            self.engine.set_pathology_visible(bool(state))
        else:
            setattr(self.engine.layers, attr, bool(state))
        if attr == "vcg_loop" and state and self.opacity_slider.value() > 60:
            self.opacity_slider.setValue(45)  # Schleife liegt im Herzen: Wand durchscheinend machen
        self.view.apply_layers(self.engine.layers)
        self._on_time(self.clock.time)

    def _vcg_scale_changed(self, value: int) -> None:
        self.engine.set_vcg_scale_factor(value / 100.0)
        self.vcg_scale_label.setText(f"{value} %")
        self._on_time(self.clock.time)

    def _color_mode_changed(self, mode: str) -> None:
        self.engine.set_color_mode(mode)
        self._on_time(self.clock.time)

    def _pattern_changed(self, index: int) -> None:
        pattern = self.pattern_combo.itemData(index)
        if isinstance(pattern, Pattern):
            self._info_dirty = True
            if pattern != Pattern.AUTO:
                self.engine.set_hypothesis(None)
            self.engine.set_pattern_override(pattern)
            self._on_time(self.clock.time)

    def _scenario_changed(self, index: int) -> None:
        scenario = self.scenario_combo.itemData(index)
        if isinstance(scenario, RhythmScenario):
            self._info_dirty = True
            self.engine.set_rhythm_scenario(scenario)
            self._on_time(self.clock.time)

    def _thesis_toggled(self, state: bool) -> None:
        self._info_dirty = True
        self.engine.set_thesis_modulation(bool(state))
        self._on_time(self.clock.time)

    def _virtual_preset_changed(self, name: str) -> None:
        if name in VIRTUAL_PRESETS:
            phi, z = VIRTUAL_PRESETS[name]
            for key, value in (("phi", phi), ("z", z), ("off", 0)):
                self.virtual_sliders[key].blockSignals(True)
                self.virtual_sliders[key].setValue(int(value))
                self.virtual_sliders[key].blockSignals(False)
            self._apply_virtual()

    def _virtual_slider_changed(self, _value: int) -> None:
        if self.virtual_preset.currentText() != "Frei":
            self.virtual_preset.blockSignals(True)
            self.virtual_preset.setCurrentText("Frei")
            self.virtual_preset.blockSignals(False)
        self._apply_virtual()

    def _apply_virtual(self) -> None:
        s = self.virtual_sliders
        self.view.set_virtual_electrode(s["phi"].value(), s["z"].value(), s["off"].value())
        if not self.engine.layers.virtual_electrode:
            self.layer_checks["virtual_electrode"].setChecked(True)
        self._refresh_virtual_plot(force=True)
        self._on_time(self.clock.time)

    def _patient_changed(self, *_args) -> None:
        from .torso import configure_torso
        f = self.patient_fields
        sex = {"männlich": "m", "weiblich": "f"}.get(self.patient_sex.currentText())
        note = configure_torso(f["height"].value() or None, f["weight"].value() or None, sex)
        age = f["age"].value() or None
        self.engine.patient = {"age": age, "sex": sex, "height_cm": f["height"].value() or None, "weight_kg": f["weight"].value() or None}
        self.patient_note.setText(note + (f", {age} J" if age else ""))
        self.view.rebuild_torso()
        self._refresh_virtual_plot(force=True)
        self._on_time(self.clock.time)

    # ---- Was-wäre-wenn-Editor
    def _ensure_fitter(self):
        tl = self.engine.timeline
        if tl is None or tl.signal is None or not self.view.has_overlays:
            return None
        self.view._ensure_simulator()
        if not self.view.simulator.available:
            return None
        if self._fitter is None or self._fitter.tl is not tl:
            from .inverse import InverseFitter
            self._fitter = InverseFitter(self.model, self.engine.solver, self.view.simulator, tl)
            patient = dict(self.engine.patient)
            if any(patient.get(k) for k in ("age", "sex", "height_cm", "weight_kg")):
                self._fitter.set_patient(patient.get("age"), patient.get("sex"), patient.get("height_cm"), patient.get("weight_kg"))
        return self._fitter

    def _section_clicked(self, info: dict) -> None:
        mode = self.editor_mode.currentText()
        if mode == "aus":
            return
        ed = self._editor
        if mode == "Block/Verzögerung an Bahn":
            path = info.get("path")
            if path in (None, "his"):
                return
            current = ed["delays"].get(path)
            nxt = 40.0 if current is None else (float("inf") if np.isfinite(current) else None)
            if nxt is None:
                ed["delays"].pop(path, None)
            else:
                ed["delays"][path] = nxt
        else:
            v = info.get("vertex")
            if v is None:
                return
            key = {"Fokus setzen": "focus", "Akzessorische Bahn setzen": "accessory", "Kreis-Zentrum (Narbe) setzen": "centre",
                   "Stimulationsort setzen": "pacing"}[mode]
            ed[key] = int(v)
            if key in ("focus", "pacing", "accessory"):
                for other in ("focus", "pacing", "accessory"):
                    if other != key:
                        ed[other] = None
        self._apply_editor()

    def _editor_hypothesis(self):
        from .inverse import Hypothesis, reentry_hypothesis
        ed = self._editor
        if ed["focus"] is not None:
            if ed["centre"] is not None:
                return reentry_hypothesis(self.model, ed["focus"], ed["centre"])
            return Hypothesis(f"edit_focus_{ed['focus']}", "Editor: Fokus", "focus", "V", origin_vertex=ed["focus"])
        if ed["pacing"] is not None:
            return Hypothesis(f"edit_pace_{ed['pacing']}", "Editor: Stimulationsort", "pacing", "P", origin_vertex=ed["pacing"])
        if ed["accessory"] is not None:
            return Hypothesis(f"edit_acc_{ed['accessory']}", "Editor: akzessorische Bahn", "accessory", "N", accessory_vertex=ed["accessory"])
        if ed["delays"]:
            delays = tuple(sorted(ed["delays"].items()))
            label = ", ".join(f"{k} {'Block' if np.isinf(v) else f'{v:.0f} ms'}" for k, v in delays)
            return Hypothesis("edit_cond_" + "_".join(k for k, _ in delays), f"Editor: {label}", "conduction", "N", extra_delays=delays)
        return None

    def _apply_editor(self) -> None:
        h = self._editor_hypothesis()
        if h is None:
            self.engine.set_hypothesis(None)
            self.sweep_detail.setText("Editor leer – Automatik.")
            self._on_time(self.clock.time)
            return
        tl = self.engine.timeline
        beat = tl.beat_at(self.clock.time) if tl is not None else None
        # Hypothese für den aktuellen Schlagtyp anwenden (Editor gilt unabhängig vom Schlagtyp)
        if beat is not None:
            h = h.__class__(**{**h.__dict__, "beat_type": beat.beat_type})
        self.engine.set_hypothesis(h)
        self._info_dirty = True
        fitter = self._ensure_fitter()
        if fitter is not None and beat is not None:
            try:
                res = fitter.evaluate(h, beat)
                self._sweep_results = [res]
                self._sweep_beat = beat
                self.sweep_table.setRowCount(1)
                for col, text in enumerate(("E", h.label, f"{res.score:.2f}", f"{res.r_mean:.2f}", f"{res.r_vcg:.2f}",
                                            "–" if res.r_t_wave is None else f"{res.r_t_wave:.2f}")):
                    item = QTableWidgetItem(text)
                    self.sweep_table.setItem(0, col, item)
                self.sweep_table.selectRow(0)
                self.sweep_status.setText(f"{h.label} · Passung {res.score:.2f} (r QRS {res.r_mean:.2f}, r VCG {res.r_vcg:.2f})")
                self.duel_button.setEnabled(False)
            except Exception as exc:
                self.sweep_status.setText(f"Editor-Simulation fehlgeschlagen: {exc}")
        self._on_time(self.clock.time)

    def _editor_clear(self) -> None:
        self._editor = {"focus": None, "delays": {}, "accessory": None, "centre": None, "pacing": None}
        self._apply_editor()

    # ---- Stimulationsvorhersage
    def _predict_pacing(self) -> None:
        from .inverse import activation_from_hypothesis, conduction_hypotheses, csp_hypotheses
        tl = self.engine.timeline
        if tl is None or not tl.beats:
            return
        beat = tl.beat_at(self.clock.time)
        base = self.engine.hypothesis
        if base is None or base.family != "conduction":
            # beste Leitungshypothese aus der letzten Rangliste, sonst aus dem erkannten Muster
            cands = [r.hypothesis for r in self._sweep_results if r.hypothesis.family == "conduction"]
            if cands:
                base = cands[0]
            else:
                plan = self.engine.plan_for(beat)
                key = {"lbbb": "lbbb", "rbbb": "rbbb", "lafb": "lafb", "lpfb": "lpfb"}.get(plan.ventricular.pattern.value, "normal")
                base = {h.key: h for h in conduction_hypotheses()}[key]
        solver = self.engine.solver
        raw_base = float(np.percentile(activation_from_hypothesis(solver, base, 100.0, exact_scale=False, scale_limits=(1, 1)).times_ms, 97))
        scale = beat.qrs_duration_ms / max(raw_base, 1.0)
        lines = [f"Ausgang: {base.label} (Modell {raw_base:.0f} ms → gemessen {beat.qrs_duration_ms:.0f} ms, Faktor {scale:.2f})"]
        self._pacing_results = {}
        for h in csp_hypotheses(base):
            raw = float(np.percentile(activation_from_hypothesis(solver, h, 100.0, exact_scale=False, scale_limits=(1, 1)).times_ms, 97))
            self._pacing_results[h.key] = h
            delta = (raw - raw_base) * scale
            verdict = "korrigiert" if raw < raw_base - 15 else ("verkürzt" if raw < raw_base - 5 else ("unverändert" if abs(raw - raw_base) <= 5 else "verlängert"))
            lines.append(f"{h.label}: ≈ {raw * scale:.0f} ms ({delta:+.0f} ms, {verdict})")
        if np.isfinite(base.lv_delay_ms) and base.lv_delay_ms > 0:
            lines.append("Hinweis: verzögerter, nicht blockierter linker Schenkel – His-Stimulation korrigiert nach Upadhyay 2019 nicht.")
        self.pace_status.setText(" · ".join(lines))

    def _apply_pacing_choice(self, index: int) -> None:
        key = {1: "csp_his", 2: "csp_lbbap", 3: "csp_biv"}.get(index)
        if key is None:
            self.engine.set_hypothesis(None)
        else:
            if key not in self._pacing_results:
                self._predict_pacing()
            h = self._pacing_results.get(key)
            if h is not None:
                tl = self.engine.timeline
                beat = tl.beat_at(self.clock.time) if tl is not None else None
                if beat is not None:
                    h = h.__class__(**{**h.__dict__, "beat_type": beat.beat_type})
                self.engine.set_hypothesis(h)
        self._info_dirty = True
        self._on_time(self.clock.time)

    # ---- Inverse Simulation
    def _representative_beat(self):
        """Mittlerer akzeptierter Schlag des dominanten Typs (für den automatischen Lauf)."""
        tl = self.engine.timeline
        beats = [b for b in tl.beats if b.beat_type in ("N", "S") and b.accepted] or list(tl.beats)
        return beats[len(beats) // 2]

    def _start_sweep(self, representative: bool = False) -> None:
        tl = self.engine.timeline
        if tl is None or tl.signal is None or not tl.beats or not self.view.has_overlays:
            self.sweep_status.setText("Kein Signal für die Simulation.")
            return
        beat = self._representative_beat() if representative is True else tl.beat_at(self.clock.time)
        self.view._ensure_simulator()
        if not self.view.simulator.available:
            self.sweep_status.setText("Dipol-Leadfield nicht kalibrierbar (zwölf Standardableitungen nötig).")
            return
        self._sweep_run += 1
        run = self._sweep_run
        self.sweep_button.setEnabled(False)
        self.sweep_table.setRowCount(0)
        self.duel_button.setEnabled(False)
        self._sweep_results = []
        self.sweep_status.setText(f"Simuliere Hypothesen für Schlag {beat.index + 1} (Typ {beat.beat_type}) …")
        model, solver, simulator, hints = self.model, self.engine.solver, self.view.simulator, dict(tl.conduction_hints)
        personalize = self.personalize_check.isChecked()
        patient = dict(self.engine.patient)
        from .activation import AtrialPattern
        try:
            atrial_disturbed = self.engine.plan_for(beat).atrial.pattern in (AtrialPattern.FLUTTER, AtrialPattern.FIBRILLATION)
        except Exception:
            atrial_disturbed = False

        def work() -> None:
            try:
                from .inverse import ATRIAL_LABELS, InverseFitter, atrial_sweep, hypotheses_for_beat
                fitter = InverseFitter(model, solver, simulator, tl)
                hypotheses = hypotheses_for_beat(model, beat, hints)
                localization_qrs_only = any(h.family in {"focus", "accessory", "pacing"} for h in hypotheses)
                if atrial_disturbed or localization_qrs_only:
                    # The rough secondary T-wave model is not an independent
                    # localization signal; origin ranking must use activation.
                    fitter.use_t_wave = False
                if any(patient.get(k) for k in ("age", "sex", "height_cm", "weight_kg")):
                    fitter.set_patient(patient.get("age"), patient.get("sex"), patient.get("height_cm"), patient.get("weight_kg"))
                results = fitter.sweep(beat, hypotheses, personalize=personalize,
                                       progress=lambda k, n, r: self.sweepProgress.emit(run, k, n, r))
                atrial = atrial_sweep(fitter, beat)
                self._pending_atrial = atrial
                notes = [n for n in (fitter.prior_note, fitter.orientation_note) if n]
                if not beat.qrs_measured:
                    notes.append("QRS-Grenzen nur geschätzt; keine verifizierte Lokalisation")
                if beat.beat_type == "?" or beat.source != "payload":
                    notes.append("Schlagtyp nicht unabhängig bestätigt")
                if beat.has_p and not beat.p_measured:
                    notes.append("Vorhoffit ausgesetzt: P-Grenzen nicht verifiziert")
                if localization_qrs_only:
                    notes.append("Ursprungs-/Bahnvergleich nur QRS; T-Welle nicht als Ortsnachweis gewertet")
                elif not fitter.use_t_wave:
                    notes.append("T-Welle wegen Vorhofflattern/-flimmern nicht gewertet")
                if atrial:
                    notes.append("Vorhof: " + ", ".join(f"{ATRIAL_LABELS[k]} r={r:.2f}" for k, r, _ in atrial[:2]))
                self.sweepFinished.emit(run, (beat, results, " · ".join(notes)))
            except Exception as exc:
                self.sweepFinished.emit(run, f"{type(exc).__name__}: {exc}")

        threading.Thread(target=work, name="heart3d-inverse", daemon=True).start()

    def _on_sweep_progress(self, run: int, k: int, n: int, result) -> None:
        if run != self._sweep_run:
            return
        self.sweep_status.setText(f"{k}/{n} simuliert … zuletzt: {result.hypothesis.label} (Passung {result.score:.2f})")

    def _on_sweep_finished(self, run: int, payload) -> None:
        if run != self._sweep_run:
            return
        self.sweep_button.setEnabled(True)
        if not isinstance(payload, tuple):
            self.sweep_status.setText(f"Simulation fehlgeschlagen: {payload}")
            return
        beat, results, orientation_note = payload
        self._atrial_results = getattr(self, "_pending_atrial", [])
        self._sweep_beat = beat
        self._sweep_results = results
        self.sweep_table.setRowCount(len(results))
        for row, r in enumerate(results):
            for col, text in enumerate((str(row + 1), r.hypothesis.label, f"{r.score:.2f}", f"{r.r_mean:.2f}", f"{r.r_vcg:.2f}",
                                        "–" if r.r_t_wave is None else f"{r.r_t_wave:.2f}")):
                item = QTableWidgetItem(text)
                if col != 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.sweep_table.setItem(row, col, item)
        best = results[0] if results else None
        if best is not None:
            gap = (best.score - results[1].score) if len(results) > 1 else 0.0
            from .inverse import HEART_ROTATION_EULER_DEG
            self.sweep_status.setText(
                f"{len(results)} Hypothesen · beste: {best.label_short if hasattr(best, 'label_short') else best.hypothesis.label} "
                f"(Passung {best.score:.2f}, Abstand zur zweiten {gap:.2f}, QRS-Zeitskalierung ×{best.time_scale_factor:.2f}"
                f"{' am Modelllimit' if best.time_scale_limited else ''}) · Dipol-Leadfield R² V1–V6: {self.view.simulator.fit_summary} · "
                f"Herzlage-Kalibrierung {HEART_ROTATION_EULER_DEG[0]:.0f}/{HEART_ROTATION_EULER_DEG[1]:.0f}/{HEART_ROTATION_EULER_DEG[2]:.0f}° · {orientation_note} · "
                "Stufe H: relative Modellpassung, keine validierte anatomische Lokalisation"
                + (" · LSB/Blockhöhe: nur explorative Modellhypothese; keine unabhängige Mapping-Validierung"
                   if best.hypothesis.family == "conduction" and (np.isinf(best.hypothesis.lv_delay_ms) or best.hypothesis.lv_delay_ms > 0) else ""))
            self.sweep_table.selectRow(0)
        self.apply_button.setEnabled(bool(results))
        self.duel_button.setEnabled(len(results) >= 2)
        self._update_findings_card()

    def _open_hypothesis_duel(self) -> None:
        if len(self._sweep_results) < 2 or self._sweep_beat is None:
            return
        selected = self.sweep_table.selectionModel().selectedRows() if self.sweep_table.selectionModel() else []
        index_a = selected[0].row() if selected else 0
        index_a = int(np.clip(index_a, 0, len(self._sweep_results) - 1))
        index_b = next((index for index in range(len(self._sweep_results)) if index != index_a), None)
        if index_b is None:
            return
        result_a, result_b = self._sweep_results[index_a], self._sweep_results[index_b]
        time_a, measured_a, simulated_a = self._beat_curves(self._sweep_beat, result_a)
        time_b, measured_b, simulated_b = self._beat_curves(self._sweep_beat, result_b)
        if not np.array_equal(time_a, time_b) or not np.allclose(measured_a, measured_b, equal_nan=True):
            self.sweep_status.setText("Vergleich ausgesetzt: Simulationen verwenden nicht denselben gemessenen Schlag.")
            return
        from .hypothesis_duel import HypothesisDuelDialog
        self._duel_dialog = HypothesisDuelDialog(time_ms=time_a, measured=measured_a,
                                                 simulated_a=simulated_a, simulated_b=simulated_b,
                                                 result_a=result_a, result_b=result_b, parent=self)
        self._duel_dialog.show()
        self._duel_dialog.raise_()

    def _sweep_selection_changed(self) -> None:
        rows = self.sweep_table.selectionModel().selectedRows() if self.sweep_table.selectionModel() else []
        if not rows or not self._sweep_results:
            return
        r = self._sweep_results[rows[0].row()]
        beat = self._sweep_beat
        lead = self.sweep_lead.currentText()
        names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        t, meas, sim = self._beat_curves(beat, r)
        j = names.index(lead)
        self.sweep_meas.setData(t * ECG_MM_PER_MS, meas[:, j] * ECG_MM_PER_MV)
        self.sweep_sim.setData(t * ECG_MM_PER_MS, sim[:, j] * ECG_MM_PER_MV)
        self.lead_grid.show(t, meas, sim, r.r_leads)
        worst = sorted(r.r_leads.items(), key=lambda kv: kv[1])[:3]
        self.sweep_detail.setText(f"{r.hypothesis.label}: r {lead} {r.r_leads.get(lead, 0):+.2f} · Verschiebung {r.shift_ms:+.0f} ms · "
                                  f"RMSE {r.rmse_mv:.2f} mV · schwächste Ableitungen " + ", ".join(f"{k} {v:+.2f}" for k, v in worst))

    def _beat_curves(self, beat, result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ganzer Schlag (P bis T-Ende) gemessen und simuliert; Zeit in ms relativ zum QRS-Beginn, Lücken als NaN."""
        tl = self.engine.timeline
        names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        cols = [list(tl.lead_names).index(n) for n in names]
        fs = tl.fs
        q0 = int(beat.qrs_on * fs)
        start = (beat.p_on - 0.06) if beat.has_p else beat.qrs_on - 0.25
        end = (beat.t_end + 0.12) if beat.t_end else beat.qrs_off + 0.4
        if beat.index + 1 < len(tl.beats):
            nxt = tl.beats[beat.index + 1]
            end = min(end, (nxt.p_on - 0.02) if nxt.has_p else nxt.qrs_on - 0.08)
        a, b = max(0, int(start * fs)), min(len(tl.signal), int(end * fs))
        base = tl.signal[max(0, q0 - int(0.06 * fs)):max(1, q0 - int(0.01 * fs)), :][:, cols].mean(0)
        meas = tl.signal[a:b, :][:, cols] - base
        sim = np.full_like(meas, np.nan)
        qa, qb = q0, min(b, q0 + len(result.simulated_leads))
        if qb > qa:
            sim[qa - a:qb - a] = result.simulated_leads[:qb - qa]
        if result.simulated_t_leads is not None and result.t_window is not None:
            ta, tb = result.t_window
            ta2, tb2 = max(ta, a), min(tb, b, ta + len(result.simulated_t_leads))
            if tb2 > ta2:
                sim[ta2 - a:tb2 - a] = result.simulated_t_leads[ta2 - ta:tb2 - ta]
        if beat.has_p and self._atrial_results:
            _, _, p_sim = self._atrial_results[0]
            pa, pb = int(beat.p_on * fs), int(beat.p_off * fs)
            if p_sim is not None and pb > pa and pa >= a and pb <= b and len(p_sim) >= pb - pa:
                sim[pa - a:pb - a] = p_sim[:pb - pa]
        t = (np.arange(a, b) - q0) / fs * 1000.0
        return t, meas, sim

    def _apply_selected_hypothesis(self) -> None:
        rows = self.sweep_table.selectionModel().selectedRows() if self.sweep_table.selectionModel() else []
        if not rows or not self._sweep_results:
            return
        r = self._sweep_results[rows[0].row()]
        self.engine.set_hypothesis(r.hypothesis)
        self.pattern_combo.blockSignals(True)
        self.pattern_combo.setCurrentIndex(0)
        self.pattern_combo.blockSignals(False)
        self._info_dirty = True
        self._on_time(self.clock.time)

    def _reset_hypothesis(self) -> None:
        self.engine.set_hypothesis(None)
        self._info_dirty = True
        self._on_time(self.clock.time)

    def _step_beat(self, direction: int) -> None:
        tl = self.engine.timeline
        if tl is None or not tl.beats:
            return
        beat = tl.beat_at(self.clock.time)
        k = int(np.clip(beat.index + direction, 0, len(tl.beats) - 1))
        target = tl.beats[k]
        self.clock.seek(target.p_on - 0.02 if target.has_p else target.qrs_on - 0.05)

    def _save_snapshot(self) -> None:
        image = self.view.grab_image()
        if image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Bild speichern", "herzillustration.png", "PNG (*.png)")
        if path:
            image.save(path)

    def _open_simulation_workspace(self):
        ancestor = self.parentWidget()
        while ancestor is not None:
            if hasattr(ancestor, "simulation_workspace") and hasattr(ancestor, "workspace_tabs"):
                ancestor.workspace_tabs.setCurrentWidget(ancestor.simulation_workspace)
                return
            ancestor = ancestor.parentWidget()
        from ..simulation_workspace import SimulationWorkspace
        from PyQt6.QtWidgets import QApplication

        def target():
            tl = self.engine.timeline
            if tl is None or tl.signal is None:
                raise ValueError("Kein EKG geladen")
            return tl.signal, tl.fs, tuple(tl.lead_names), {"source": "heart_viewer_timeline"}

        if not hasattr(self, "_simulation_window"):
            window = QMainWindow(self)
            window.setWindowTitle("Elektrophysiologie · eigenständige Forschungswerkbank")
            bench = SimulationWorkspace(window, target_provider=target)
            window.setCentralWidget(bench)
            window.resize(1280, 850)
            QApplication.instance().aboutToQuit.connect(bench.shutdown)
            self._simulation_window = window
        self._simulation_window.show()
        self._simulation_window.raise_()

    def timeline_changed(self) -> None:
        self.strip.set_timeline(self.engine.timeline)
        self._virtual_plot_beat = -1
        self._info_dirty = True
        self._sweep_results = []
        self._atrial_results = []
        self.sweep_table.setRowCount(0)
        self._update_findings_card()
        tl = self.engine.timeline
        if self.research_mode and tl is not None and tl.signal is not None and tl.beats and self._auto_sweep_timeline is not tl:
            self._auto_sweep_timeline = tl
            QTimer.singleShot(400, lambda: self._start_sweep(representative=True))
        self.view.apply_layers(self.engine.layers)
        self._on_time(self.clock.time)

    def _on_time(self, t: float) -> None:
        if self.engine.timeline is None:
            return
        try:
            frame = self.engine.frame(t)
        except Exception as exc:
            self.info.setPlainText(f"Animation angehalten: {exc}")
            self.clock.pause()
            return
        if self.isVisible():
            self.view.update_frame(frame)
            if self.section.isVisible():
                self.section.set_frame(frame, self.engine.timeline, self.engine.pathology)
        self.strip.set_time(t)
        self.time_label.setText(f"{t:6.2f} s".replace(".", ","))
        self.phase_label.setText(frame.phase_label)
        if frame.plan is None or frame.plan.beat.index != self._info_beat or self._info_dirty:
            self._update_info(frame)
            self._info_beat = frame.plan.beat.index if frame.plan is not None else -1
            self._info_dirty = False
        if self.engine.layers.virtual_electrode and frame.plan is not None:
            self._refresh_virtual_plot(beat_index=frame.plan.beat.index)
            self.virtual_cursor.setPos(t)

    def _refresh_virtual_plot(self, beat_index: Optional[int] = None, force: bool = False) -> None:
        tl = self.engine.timeline
        if tl is None or tl.signal is None or not self.view.has_overlays:
            return
        if beat_index is None:
            beat_index = tl.beat_at(self.clock.time).index
        if not force and beat_index == self._virtual_plot_beat:
            return
        self._virtual_plot_beat = beat_index
        beat = tl.beats[beat_index]
        start, end = tl.cycle_bounds(beat)
        a, b = max(0, int(start * tl.fs)), min(len(tl.signal), int(end * tl.fs))
        if b - a < 3:
            return
        t = np.arange(a, b) / tl.fs
        virt = self.view.virtual_signal()
        if virt is not None and len(virt) >= b:
            self.virtual_curve.setData(t, virt[a:b] - np.median(virt[a:b]))
        ref = self.reference_combo.currentText()
        names = list(tl.lead_names)
        if ref in names:
            y = tl.signal[a:b, names.index(ref)]
            self.reference_meas.setData(t, y - np.median(y))
            sim = self.view.simulator
            if sim.available:
                from .torso import ELECTRODE_BY_NAME
                ys = sim.unipolar(ELECTRODE_BY_NAME[ref].position)[a:b]
                self.reference_sim.setData(t, ys - np.median(ys))
        self.virtual_plot.setXRange(start, end, padding=0.01)
        self.fit_label.setText(f"Übereinstimmung R² V1–V6: {self.view.simulator.fit_summary} · Dipol im homogenen Leiter, "
                               f"Zentrum/Maßstab an V1–V6 angepasst – Näherung, keine Messung")

    # ---- Befundkarte
    def _update_findings_card(self) -> None:
        """Kurzer Hinweis, was die Ansichten gerade zeigen (der Befund selbst steht im ECG-Labor)."""
        tl = self.engine.timeline
        if tl is None or not tl.beats:
            self.findings.setText("Kein ECG geladen.")
            return
        est = self.engine.pathology
        beat = tl.beat_at(self.clock.time) if self.clock.time > 0 else tl.beats[min(2, len(tl.beats) - 1)]
        try:
            plan = self.engine.plan_for(beat)
        except Exception:
            plan = None
        parts = []
        if plan is not None:
            from .engine import SCENARIO_LABELS, RhythmScenario
            parts.append(f"<b>Gezeigt</b>: {plan.ventricular.label} [{plan.pattern_tier}]" +
                         (f", {SCENARIO_LABELS[plan.scenario].split(' (')[0]}" if plan.scenario != RhythmScenario.AUTO else "") +
                         f"; Vorhof: {plan.atrial.label.split(':')[0]}")
        if est.findings:
            shown = [t for tier, t, _ in est.findings if any(w in t for w in ("Ischämie", "Narbe", "Hypertrophie", "Vorhof", "Niedervoltage"))]
            if shown:
                parts.append("<b>Illustrative Musterhypothesen</b>: " + "; ".join(shown[:3])
                             + (f" ({est.territory})" if est.territory else "")
                             + "; kein unabhängiger Krankheitsnachweis")
        if self._sweep_results:
            best = self._sweep_results[0]
            second = self._sweep_results[1].score if len(self._sweep_results) > 1 else best.score
            gap = best.score - second
            strength = "größerer Abstand" if gap >= 0.08 else ("kleiner Abstand" if gap >= 0.03 else "nahezu gleich")
            parts.append(f"<b>Unvalidierte Modellhypothese</b> [H]: {best.hypothesis.label} (relative Passung {best.score:.2f}, {strength})")
        self.findings.setText("<br>".join(parts) if parts else "–")

    def _update_info(self, frame: FrameData) -> None:
        plan = frame.plan
        tl = self.engine.timeline
        if plan is None or tl is None:
            self.info.setPlainText("Kein Schlag am Zeitcursor.")
            return
        beat = plan.beat
        ev = plan.events
        tier = {"P": "Physiologie (Musterbibliothek)", "K": "Kriterium/Klasse aus der Analyse", "H": "Hypothese (Projektthese)"}.get(plan.pattern_tier, plan.pattern_tier)
        timing_source = ('gespeicherte Delineation' if beat.source == 'payload' and beat.qrs_measured
                         else 'illustrative Zeitannahme')
        qrs_display = (f'{beat.qrs_duration_ms:.0f} ms' if beat.qrs_measured else
                       f'{beat.qrs_duration_ms:.0f} ms illustrativ, nicht gemessen')
        p_display = ('messbare P-Grenzen' if beat.p_measured else 'P-Grenzen nicht gesichert')
        lines = [
            f"Schlag {beat.index + 1} von {len(tl.beats)} · Typ {beat.beat_type} · QRS-Zeitquelle: {timing_source}",
            f"Frequenz {beat.hr:.0f}/min · QRS {qrs_display} · {p_display}",
            "",
            f"Kammermuster: {plan.ventricular.label}",
            f"Stufe: {tier}",
            f"Grund: {plan.pattern_reason}",
            f"Vorhof: {plan.atrial.label}",
        ]
        if plan.origin_label:
            lines.append(f"Illustrierte Modellregion: {plan.origin_label}; keine klinisch validierte Fokuslokalisation")
        if plan.scenario != RhythmScenario.AUTO:
            lines.append(f"Rhythmusszenario: {SCENARIO_LABELS[plan.scenario]}")
        lines += [
            "",
            f"Weissler (HF {beat.hr:.0f}): PEP {ev.pep_ms:.0f} ms · LVET {ev.lvet_ms:.0f} ms · QS2 {ev.qs2_ms:.0f} ms · IVRT {ev.ivrt_ms:.0f} ms",
            f"AV-Klappen zu {ev.av_close - beat.qrs_on:+.3f} s · Aorta auf {ev.ao_open - beat.qrs_on:+.3f} s · zu {ev.ao_close - beat.qrs_on:+.3f} s · AV auf {ev.av_open - beat.qrs_on:+.3f} s (relativ zum QRS-Beginn)",
        ]
        if plan.notes:
            lines.append("")
            lines.extend(f"• {note}" for note in plan.notes)
        est = self.engine.pathology
        lines.append("")
        if est.findings:
            lines.append("Pathologie-Schätzung (K Kriterium der Analyse · E Schnellschätzung aus dem Signal · H Hypothese):")
            for tier_id, title, detail in est.findings:
                lines.append(f"[{tier_id}] {title}: {detail}")
        else:
            lines.append("Pathologie-Schätzung: keine ST-/Q-/Voltage-Auffälligkeit erkannt")
        if est.ischemia_kind == "transmural" and est.measures is not None and est.measures.st_60:
            try:
                from .inverse import HEART_ROTATION, ischemia_forward_check
                chk = ischemia_forward_check(self.model, self.engine.solver, HEART_ROTATION, est.ischemia, est.ischemia_kind, est.measures.st_60)
                if chk:
                    lines.append(f"[H] Rückrechnung des Verletzungsstroms aus der Zone: ST-Vektor-Winkel zur Messung {chk['angle_deg']:.0f}°, "
                                 f"r über 12 Ableitungen {chk['r_leads']:.2f} ({'stimmig' if chk['angle_deg'] < 35 and chk['r_leads'] > 0.5 else 'abweichend – Zone mit Vorsicht'})")
            except Exception as exc:
                lines.append(f"Rückrechnung nicht möglich: {exc}")
        elif est.ischemia_kind == "subendocardial":
            lines.append("[H] Rückrechnung: diffuse subendokardiale Zone hat keinen definierten Verletzungsstrom-Vektor – nicht prüfbar")
        m = est.measures
        if m is not None and m.st_j:
            st_text = " ".join(f"{k} {v * 10:+.1f}" for k, v in m.st_j.items() if abs(v) >= 0.05)
            if st_text:
                lines.append(f"J-Punkt-Abweichungen ≥ 0,5 mm: {st_text} (mm)")
        if tl.rhythm_class != "unknown":
            lines.append("")
            lines.append(f"Rhythmusklasse (technisch): {tl.rhythm_class}")
        hints = {k: v for k, v in tl.conduction_hints.items() if v is not None and k in ("lbbb", "rbbb", "wpw", "paced", "left_axis", "right_axis", "first_degree_av_block")}
        if hints:
            lines.append("Leitungshinweise: " + ", ".join(f"{k}={v}" for k, v in hints.items()))
        thesis = {k: v for k, v in tl.conduction_hints.items() if isinstance(v, float) and np.isfinite(v) and k.startswith("qrs_")}
        if thesis:
            lines.append("Thesenmarker (Analyse): " + " · ".join(f"{k.replace('qrs_', '')}={v:.2f}" for k, v in thesis.items())
                         + (" – fließen in die Zeitachse/Durchbruchzahl ein" if self.engine.use_thesis_modulation else " – Modulation aus"))
        if frame.conduction_notes:
            lines.extend(f"• {n}" for n in frame.conduction_notes)
        src = (self.engine.payload or {}).get("lab_processing", {}).get("source")
        if src == "heart3d_core_pipeline":
            lines.append("Quelle: Delineation, Rhythmus, Kriterien und Marker aus der Analysekette des Hauptprojekts (ohne Laborbefund).")
            for note in (self.engine.payload or {}).get("notes", ()):
                lines.append(f"• {note}")
        lines.append("")
        lines.append(ILLUSTRATION_NOTE)
        self.info.setPlainText("\n".join(lines))

    def closeEvent(self, event) -> None:
        self._orbit_timer.stop()
        super().closeEvent(event)
