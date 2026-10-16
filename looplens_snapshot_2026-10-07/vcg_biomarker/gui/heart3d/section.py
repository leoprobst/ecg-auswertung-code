"""2D-Schnitt (Vierkammer-Ebene) mit Reizleitungsschema, läuft parallel zur 3D-Ansicht.

Die Schnittebene geht durch LV-Apex, RV-Zentrum und LA-Zentrum (Vierkammerblick).
Alle Wandvertices in einer 11 mm dicken Scheibe um diese Ebene werden mit den
Farben und Positionen des laufenden Bildes gezeichnet – Erregung, Repolarisation,
Ischämie-/Narbenzonen und Kontraktion erscheinen damit ohne zweite Rechnung.
Darüber liegt das Reizleitungssystem als Projektion (Sinusknoten, Bachmann-Bündel,
AV-Knoten, His, Schenkel, Faszikel) mit laufendem Impuls sowie Marker für Blöcke,
akzessorische Bahn, Fokus und Stimulationselektrode.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from .activation import AtrialPattern, Pattern
from .model import HeartModel, smooth_path

SLAB_HALF_MM = 5.5


class SectionGeometry:
    def __init__(self, model: HeartModel):
        self.model = model
        apex = model.landmarks["lv_apex"]
        la = model.landmarks["la_center"]
        rv = model.landmarks["rv_center"]
        lv = model.landmarks["lv_center"]
        n = np.cross(la - apex, rv - apex)
        n = n / np.linalg.norm(n)
        if n[1] < 0:  # Normale nach posterior, Betrachter blickt von anterior
            n = -n
        self.normal = n
        self.origin = 0.5 * (lv + rv)
        up = la - apex
        up = up - n * np.dot(up, n)
        self.e_up = up / np.linalg.norm(up)  # Bildschirm nach oben = Richtung Basis
        self.e_right = np.cross(self.e_up, n)  # Bildschirm nach rechts
        if self.e_right[0] < 0:  # Patient links soll im Bild rechts liegen (wie Echo-4CH gespiegelt, hier anatomisch)
            self.e_right = -self.e_right
        self.slab: dict[str, np.ndarray] = {}
        for name in ("wall", "ra_wall", "la_wall", "aorta", "pulmonary_trunk"):
            part = model.parts.get(name)
            if part is None:
                continue
            d = (part.vertices.astype(np.float64) - self.origin) @ n
            half = SLAB_HALF_MM if name in ("wall",) else SLAB_HALF_MM * 1.6
            self.slab[name] = np.where(np.abs(d) < half)[0]
        pts = model.wall.vertices[self.slab["wall"]]
        xy = self.project(pts)
        self.bounds = (xy[:, 0].min() - 15, xy[:, 0].max() + 15, xy[:, 1].min() - 15, xy[:, 1].max() + 55)

    def project(self, points: np.ndarray) -> np.ndarray:
        rel = np.atleast_2d(points).astype(np.float64) - self.origin
        return np.column_stack([rel @ self.e_right, rel @ self.e_up])


class SectionView(QWidget):
    """Zeichnet Schnitt + Reizleitungsschema mit QPainter."""

    clicked = pyqtSignal(object)  # {"vertex": int|None, "path": str|None, "x": float, "y": float}

    def __init__(self, model: HeartModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model = model
        self.geometry_ = SectionGeometry(model)
        self.setMinimumSize(300, 340)
        self._frame = None
        self._timeline = None
        self._pathology = None
        self._dark = True
        self._paths2d = {name: self.geometry_.project(path) for name, path in model.conduction_paths.items() if len(path) >= 2}
        self._sa = self.geometry_.project(model.landmarks["sa_node"])[0]
        self._av = self.geometry_.project(model.landmarks["av_node"])[0]
        self._his = self.geometry_.project(model.landmarks["his"])[0]
        la_wall = model.parts["la_wall"].vertices
        self._bachmann = self.geometry_.project(la_wall[model.atrial_sources["la_wall"]["bachmann"]])[0]
        self.show_labels = True
        # Vorhofbahnen als leichte Bögen (Bachmann-Bündel über das Vorhofdach, internodale Bahn)
        self._atrial_paths = {
            "bachmann": self._arc(self._sa, self._bachmann, 0.25),
            "internodal": self._arc(self._sa, self._av, -0.12),
        }
        # Purkinje-Zweige: von den Bahnenden fächerförmig zu den Durchbruchsorten der Kammer
        wall = model.wall.vertices
        self._twigs: dict[str, list[np.ndarray]] = {}
        for path_name, sites in (("lbb_anterior", ("lv_anterior_paraseptal", "lv_apical")),
                                 ("lbb_posterior", ("lv_posterior_paraseptal", "lv_apical")),
                                 ("rbb", ("rv_anterior_papillary", "rv_apical", "rv_outflow")),
                                 ("lbb", ("lv_mid_septal",))):
            path = self._paths2d.get(path_name)
            if path is None:
                continue
            end = path[-1]
            twigs = []
            for site in sites:
                idx = model.purkinje_sites.get(site)
                if idx is None or len(idx) == 0:
                    continue
                target = self.geometry_.project(wall[idx].mean(0))[0]
                twigs.append(self._arc(end, target, 0.18 if len(twigs) % 2 else -0.18))
            self._twigs[path_name] = twigs

    @staticmethod
    def _arc(a: np.ndarray, b: np.ndarray, bulge: float) -> np.ndarray:
        """Bogen von a nach b, seitlich ausgelenkt um bulge·|b−a|."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        n = np.array([-d[1], d[0]])
        mid = 0.5 * (a + b) + n * bulge
        return smooth_path(np.vstack([a, mid, b]), 4)

    @staticmethod
    def _path(pts: np.ndarray) -> QPainterPath:
        path = QPainterPath(QPointF(pts[0][0], pts[0][1]))
        for x, y in pts[1:]:
            path.lineTo(x, y)
        return path

    def set_dark(self, dark: bool) -> None:
        self._dark = bool(dark)
        self.update()

    def mousePressEvent(self, event) -> None:
        if self._frame is None:
            return
        pos = event.position()
        sx, sy = float(pos.x()), float(pos.y())
        to_screen, s = self._mapper()
        geo = self.geometry_
        idx = geo.slab["wall"]
        pts = to_screen(geo.project(self._frame.positions["wall"][idx]))
        d = np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)
        k = int(np.argmin(d))
        vertex = int(idx[k]) if d[k] < 14.0 else None
        best_path, best_d = None, 1e9
        for name, path in self._paths2d.items():
            spts = to_screen(path)
            dd = np.min(np.hypot(spts[:, 0] - sx, spts[:, 1] - sy))
            if dd < best_d:
                best_path, best_d = name, dd
        self.clicked.emit({"vertex": vertex, "path": best_path if best_d < 12.0 else None, "x": sx, "y": sy})
        super().mousePressEvent(event)

    def set_frame(self, frame, timeline, pathology=None) -> None:
        self._frame, self._timeline, self._pathology = frame, timeline, pathology
        self.update()

    # ---- Abbildung
    def _mapper(self):
        x0, x1, y0, y1 = self.geometry_.bounds
        w, h = self.width() - 16, self.height() - 16
        s = min(w / (x1 - x0), h / (y1 - y0))
        cx = 8 + (w - s * (x1 - x0)) / 2
        cy = 8 + (h - s * (y1 - y0)) / 2

        def to_screen(xy: np.ndarray) -> np.ndarray:
            xy = np.atleast_2d(xy)
            return np.column_stack([cx + (xy[:, 0] - x0) * s, cy + (y1 - xy[:, 1]) * s])
        return to_screen, s

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor(16, 22, 30) if self._dark else QColor(245, 246, 248)
        painter.fillRect(self.rect(), bg)
        frame = self._frame
        if frame is None:
            painter.setPen(QColor(150, 160, 170))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Kein Schlag")
            return
        to_screen, s = self._mapper()
        painter.drawImage(0, 0, self._raster(frame, to_screen, s, bg))
        self._draw_conduction(painter, to_screen, s, frame)
        self._draw_markers(painter, to_screen, s, frame)
        if self.show_labels:
            self._draw_labels(painter, to_screen, s)
        painter.end()

    def _raster(self, frame, to_screen, s: float, bg: QColor) -> QImage:
        """Wandpunkte der Scheibe als Scheibchen in ein halb aufgelöstes Bild splatten (gepackte uint32-Farben)."""
        geo = self.geometry_
        w, h = max(1, self.width()), max(1, self.height())
        half = 2
        hw, hh = (w + half - 1) // half, (h + half - 1) // half
        r = int(max(1, round(1.7 * s / half)))
        pad = r + 1
        W = hw + 2 * pad
        bg_packed = np.uint32(0xFF000000 | (bg.red() << 16) | (bg.green() << 8) | bg.blue())
        buf = np.full(((hh + 2 * pad) * W,), bg_packed, np.uint32)
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) <= r * r + 0.5
        ox, oy = xx[disk].ravel(), yy[disk].ravel()
        bg_rgb = np.array([bg.red(), bg.green(), bg.blue()], np.float32) / 255.0
        layers = []
        for name in ("aorta", "pulmonary_trunk"):
            idx = geo.slab.get(name)
            if idx is None or idx.size == 0:
                continue
            c = self.model.parts[name].color[:3] * 0.35 + bg_rgb * 0.65
            layers.append((frame.positions.get(name, self.model.parts[name].vertices)[idx], np.tile(c, (idx.size, 1))))
        for name in ("ra_wall", "la_wall", "wall"):
            idx = geo.slab.get(name)
            if idx is None or idx.size == 0 or name not in frame.colors:
                continue
            layers.append((frame.positions.get(name, self.model.parts[name].vertices)[idx], frame.colors[name][idx, :3]))
        for pos, rgb in layers:
            pts = to_screen(geo.project(pos)) / half
            px = np.clip(np.round(pts[:, 0]).astype(np.int64), -1, hw) + pad
            py = np.clip(np.round(pts[:, 1]).astype(np.int64), -1, hh) + pad
            c = (np.clip(rgb, 0, 1) * 255).astype(np.uint32)
            packed = np.uint32(0xFF000000) | (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
            lin = (py[:, None] + oy[None, :]) * W + (px[:, None] + ox[None, :])
            buf[lin] = packed[:, None]
        img = buf.reshape(hh + 2 * pad, W)[pad:pad + hh, pad:pad + hw]
        self._raster_buffer = np.ascontiguousarray(img)  # Referenz halten, solange das QImage lebt
        image = QImage(self._raster_buffer.data, hw, hh, hw * 4, QImage.Format.Format_ARGB32)
        return image.scaled(hw * half, hh * half, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

    @staticmethod
    def _point_on(pts: np.ndarray, f: float) -> np.ndarray:
        k = float(np.clip(f, 0.0, 1.0)) * (len(pts) - 1)
        i = int(np.clip(np.floor(k), 0, len(pts) - 2))
        return pts[i] + (pts[i + 1] - pts[i]) * (k - i)

    def _draw_conduction(self, painter: QPainter, to_screen, s: float, frame) -> None:
        state = frame.conduction or {}
        blocked = set(frame.blocked or ())
        dim = QColor(205, 195, 120, 150)
        lit = QColor(255, 240, 90, 255)
        glow = QColor(255, 220, 60, 70)
        red = QColor(255, 80, 80, 240)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        def stroke(pts: np.ndarray, on: bool, width: float, dashed: bool = False, color: Optional[QColor] = None) -> None:
            path = self._path(to_screen(pts))
            if on:
                painter.setPen(QPen(glow, width + 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.drawPath(path)
            pen = QPen(color or (lit if on else dim), width if on else width * 0.7,
                       Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)

        def pulse(pts: np.ndarray, f: float, r: float = 4.0) -> None:
            q = self._point_on(to_screen(pts), f)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 240, 120, 90))
            painter.drawEllipse(QPointF(*q), r * 2, r * 2)
            painter.setBrush(QBrush(lit))
            painter.drawEllipse(QPointF(*q), r, r)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        def cross(q: np.ndarray, text: str = "") -> None:
            x, y = q
            painter.setPen(QPen(red, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(x - 6, y - 6), QPointF(x + 6, y + 6))
            painter.drawLine(QPointF(x - 6, y + 6), QPointF(x + 6, y - 6))
            if text:
                painter.setPen(red)
                painter.drawText(QPointF(x + 9, y + 4), text)

        BLOCK_AT = 0.35
        # ---- Vorhofbahnen
        for name in ("internodal", "bachmann"):
            pts = self._atrial_paths[name]
            f = state.get(name, state.get("retro_" + name))
            stroke(pts, f is not None, 2.4, dashed=(name == "bachmann"))
            if f is not None:
                pulse(pts, f)
        # ---- Kammerbahnen
        widths = {"his": 3.2, "lbb": 3.0, "rbb": 2.6, "lbb_anterior": 2.2, "lbb_posterior": 2.2}
        labels = {"lbb": "LSB", "rbb": "RSB", "lbb_anterior": "LAFB", "lbb_posterior": "LPFB"}
        for name, pts in self._paths2d.items():
            f = state.get(name)
            retro = state.get("retro_" + name)
            to_block = state.get(name + "_to_block")
            if name in blocked:
                # bis zum Block hell, dahinter blass-rot gestrichelt
                spts = to_screen(pts)
                cut = int(round(BLOCK_AT * (len(spts) - 1)))
                head = pts[:cut + 1] if cut >= 1 else pts[:2]
                stroke(head, to_block is not None, widths.get(name, 2.2))
                stroke(pts[cut:], False, widths.get(name, 2.2), dashed=True, color=QColor(255, 120, 120, 120))
                if to_block is not None:
                    pulse(pts, to_block * BLOCK_AT)
            else:
                on = f is not None or retro is not None
                stroke(pts, on, widths.get(name, 2.2))
                if f is not None:
                    pulse(pts, f)
                if retro is not None:
                    pulse(pts, retro)
        twig_on = any(k in state for k in ("lbb_anterior", "lbb_posterior", "rbb", "retro_rbb", "retro_lbb_anterior"))
        for name, twigs in self._twigs.items():
            base = name if name != "lbb" else "lbb_anterior"
            on = twig_on and (base not in blocked and "lbb" not in (blocked if name != "rbb" else set()))
            for twig in twigs:
                stroke(twig, on, 1.3)
        # ---- Blockmarker
        for name in blocked:
            if name in self._paths2d:
                cross(self._point_on(to_screen(self._paths2d[name]), BLOCK_AT), labels.get(name, ""))
            elif name == "av":
                x, y = to_screen(self._av)[0]
                painter.setPen(QPen(red, 3.0))
                painter.drawLine(QPointF(x - 10, y + 9), QPointF(x + 10, y + 9))
                painter.drawLine(QPointF(x - 10, y + 13), QPointF(x + 10, y + 13))
                painter.setPen(red)
                painter.drawText(QPointF(x + 13, y + 14), "AV-Block III")
        # ---- Knoten (AV-Knoten leuchtet während der Verzögerung)
        painter.setPen(Qt.PenStyle.NoPen)
        av_f = state.get("av", state.get("retro_av"))
        for pt, r, active in ((self._sa, 5.5, "internodal" in state), (self._av, 5.5, av_f is not None), (self._his, 3.5, "his" in state or "retro_his" in state)):
            x, y = to_screen(pt)[0]
            painter.setBrush(QColor(255, 230, 90, 170 if active else 70))
            painter.drawEllipse(QPointF(x, y), r + (7 if active else 4), r + (7 if active else 4))
            painter.setBrush(QColor(255, 235, 110, 240))
            painter.drawEllipse(QPointF(x, y), r, r)
        if av_f is not None:
            x, y = to_screen(self._av)[0]
            painter.setPen(QPen(lit, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(QRectF(x - 9, y - 9, 18, 18), 90 * 16, -int(360 * 16 * av_f))
        # ---- Kreiserregungen (AVNRT im Knoten, Kammertachykardie um Narbe)
        if "avn_loop" in state:
            x, y = to_screen(self._av)[0]
            painter.setPen(QPen(QColor(255, 200, 80, 220), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(x, y), 13, 13)
            ang = 2 * np.pi * state["avn_loop"]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(lit))
            painter.drawEllipse(QPointF(x + 13 * np.cos(ang), y + 13 * np.sin(ang)), 4.0, 4.0)
            painter.setPen(QColor(255, 200, 80))
            painter.drawText(QPointF(x + 17, y - 10), "Kreis im AV-Knoten")
        if "vt_loop" in state and frame.plan is not None and frame.plan.ventricular.focal_vertices.size:
            fpos = frame.positions["wall"][frame.plan.ventricular.focal_vertices].mean(0)
            x, y = to_screen(self.geometry_.project(fpos))[0]
            painter.setPen(QPen(QColor(255, 120, 120, 220), 2.0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(x, y), 22, 22)
            ang = 2 * np.pi * state["vt_loop"]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(lit))
            painter.drawEllipse(QPointF(x + 22 * np.cos(ang), y + 22 * np.sin(ang)), 4.5, 4.5)
            painter.setPen(QColor(255, 140, 140))
            painter.drawText(QPointF(x + 26, y + 20), "Kreis um Narbe/Isthmus")
        # ---- Vorhofflattern / -flimmern
        ra = to_screen(self.geometry_.project(self.model.landmarks["ra_center"]))[0]
        if "flutter" in state:
            painter.setPen(QPen(QColor(255, 200, 80, 200), 2.0))
            painter.drawEllipse(QPointF(*ra), 16, 16)
            ang = 2 * np.pi * state["flutter"]
            q = ra + np.array([16 * np.cos(ang), 16 * np.sin(ang)])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(lit))
            painter.drawEllipse(QPointF(*q), 4.5, 4.5)
            painter.setPen(QColor(255, 200, 80))
            painter.drawText(QPointF(ra[0] + 20, ra[1]), f"Kreiserregung {state.get('flutter_cycle_ms', 200):.0f} ms")
        elif "af" in state:
            rng = np.random.default_rng(int(frame.time * 30))
            la = to_screen(self.geometry_.project(self.model.landmarks["la_center"]))[0]
            painter.setPen(Qt.PenStyle.NoPen)
            for centre in (ra, la):
                for _ in range(6):
                    q = centre + rng.normal(0, 14, 2)
                    painter.setBrush(QColor(255, 220, 80, int(rng.integers(60, 200))))
                    painter.drawEllipse(QPointF(*q), 2.5, 2.5)
            painter.setPen(QColor(255, 200, 80))
            painter.drawText(QPointF(ra[0] - 20, ra[1] + 26), "Flimmern")

    def _draw_markers(self, painter: QPainter, to_screen, s: float, frame) -> None:
        plan = frame.plan
        if plan is None:
            return
        pattern = plan.ventricular.pattern
        font = QFont("Segoe UI", 8)
        painter.setFont(font)
        tl = self._timeline
        if tl is not None and tl.conduction_hints.get("first_degree_av_block"):
            x, y = to_screen(self._av)[0]
            painter.setPen(QColor(255, 180, 80))
            painter.drawText(QPointF(x + 10, y + 14), "AV verzögert")
        # Fokus / Elektrode / Bahn
        focal = plan.ventricular.focal_vertices
        beat = plan.beat
        active = beat.qrs_on - 0.02 <= frame.time <= beat.qrs_off + 0.03
        focal_f = (frame.conduction or {}).get("focal")
        if focal.size:
            pos = frame.positions["wall"][focal].mean(0)
            x, y = to_screen(self.geometry_.project(pos))[0]
            pulse = (1.0 - focal_f) if focal_f is not None else (0.3 if active else 0.0)
            if pattern == Pattern.RV_PACING:
                # Elektrode von der oberen Hohlvene zum RV-Apex
                top = to_screen(self.geometry_.project(self.model.landmarks["sa_node"] + np.array([0, 0, 25.0])))[0]
                painter.setPen(QPen(QColor(200, 210, 230), 2.0))
                painter.drawLine(QPointF(*top), QPointF(x, y))
                painter.setPen(QColor(200, 210, 230))
                painter.drawText(QPointF(x + 8, y - 6), "Elektrode")
            elif pattern == Pattern.WPW:
                la_pt = self.geometry_.project(self.model.landmarks["la_center"])[0]
                arc = self._arc(la_pt, self.geometry_.project(pos)[0], 0.2)
                acc = (frame.conduction or {}).get("accessory")
                painter.setPen(QPen(QColor(255, 160, 60, 255 if acc is not None else 150), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(self._path(to_screen(arc)))
                if acc is not None:
                    q = self._point_on(to_screen(arc), acc)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(255, 230, 90, 240))
                    painter.drawEllipse(QPointF(*q), 4.5, 4.5)
                painter.setPen(QColor(255, 160, 60))
                painter.drawText(QPointF(x + 8, y - 6), "akzessorische Bahn")
            else:
                painter.setPen(QColor(255, 220, 60))
                painter.drawText(QPointF(x + 8, y - 6), "ektoper Fokus")
            star = self._star(x, y, 6 + 4 * pulse)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 230, 60, int(140 + 100 * pulse)))
            painter.drawPolygon(star)
        retro_acc = (frame.conduction or {}).get("retro_accessory")
        if retro_acc is not None or (frame.conduction or {}).get("circuit_avrt"):
            site = self.model.purkinje_sites.get("lv_lateral_base")
            if site is not None and len(site):
                pos = frame.positions["wall"][site].mean(0)
                la_pt = self.geometry_.project(self.model.landmarks["la_center"])[0]
                arc = self._arc(la_pt, self.geometry_.project(pos)[0], 0.2)
                painter.setPen(QPen(QColor(255, 160, 60, 255 if retro_acc is not None else 150), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(self._path(to_screen(arc)))
                if retro_acc is not None:
                    q = self._point_on(to_screen(arc), retro_acc)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(255, 230, 90, 240))
                    painter.drawEllipse(QPointF(*q), 4.5, 4.5)
                painter.setPen(QColor(255, 160, 60))
                x, y = to_screen(self.geometry_.project(pos))[0]
                painter.drawText(QPointF(x + 8, y - 6), "akzessorische Bahn (retrograd)" if retro_acc is not None else "akzessorische Bahn")
        # Pathologie-Beschriftung
        if self._pathology is not None and self._pathology.findings:
            painter.setPen(QColor(220, 140, 255) if self._dark else QColor(120, 40, 160))
            y = self.height() - 8 - 12 * min(3, len(self._pathology.findings))
            for tier, title, _ in self._pathology.findings[:3]:
                painter.drawText(QPointF(8, y), f"[{tier}] {title}")
                y += 12

    @staticmethod
    def _star(x: float, y: float, r: float) -> QPolygonF:
        pts = []
        for k in range(10):
            ang = np.pi / 2 + k * np.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            pts.append(QPointF(x + rr * np.cos(ang), y - rr * np.sin(ang)))
        return QPolygonF(pts)

    def _draw_labels(self, painter: QPainter, to_screen, s: float) -> None:
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(200, 210, 220) if self._dark else QColor(60, 70, 80))
        m = self.model
        for key, text, dx in (("ra_center", "RA", 0), ("la_center", "LA", 0), ("rv_center", "RV", 0), ("lv_center", "LV", 0)):
            x, y = to_screen(self.geometry_.project(m.landmarks[key]))[0]
            painter.drawText(QPointF(x - 8 + dx, y + 4), text)
        for pt, text, dx, dy in ((self._sa, "SA", 7, -6), (self._av, "AV", 7, -4), (self._his, "His", 7, 8)):
            x, y = to_screen(pt)[0]
            painter.drawText(QPointF(x + dx, y + dy), text)
        painter.setPen(QColor(140, 150, 160))
        painter.drawText(QPointF(8, 14), "Vierkammer-Schnitt · Reizleitung (Projektion)")
