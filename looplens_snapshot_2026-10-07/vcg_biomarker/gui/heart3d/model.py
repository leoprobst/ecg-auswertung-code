"""Geometrisches Herzmodell: Laden, Kammerzuordnung, Koordinaten, Leitungssystem.

Grundlage ist das BodyParts3D-Modell (CC BY-SA 2.1 JP, siehe
``gui/resources/heart3d/ATTRIBUTION.txt``). Koordinaten in Millimetern, auf den
Schwerpunkt der Kammerwand zentriert. Achsen: +x = links (Patient), -y = anterior,
+z = kranial.

Alle abgeleiteten Größen (Kammerzugehörigkeit, Apex-Basis-Koordinate, AHA-Segment,
Endokard/Epikard, Landmarken des Leitungssystems, Purkinje-Durchbruchsorte) werden
aus der Geometrie berechnet und sind Näherungen für die Illustration, keine
anatomischen Messungen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

RESOURCE_PATH = Path(__file__).resolve().parents[1] / "resources" / "heart3d" / "heart_model.npz"

CHAMBER_LV = 0
CHAMBER_RV = 1
CHAMBER_LA = 2
CHAMBER_RA = 3

# Farben (RGBA 0..1): Ventrikel leicht verschieden, Vorhöfe dunkler, Gefäße neutral.
COLOR_LV = np.array([0.80, 0.30, 0.30, 1.0], dtype=np.float32)
COLOR_RV = np.array([0.72, 0.36, 0.44, 1.0], dtype=np.float32)
COLOR_LA = np.array([0.62, 0.26, 0.30, 1.0], dtype=np.float32)
COLOR_RA = np.array([0.58, 0.30, 0.38, 1.0], dtype=np.float32)
COLOR_ARTERY = np.array([0.85, 0.42, 0.40, 1.0], dtype=np.float32)
COLOR_VEIN = np.array([0.45, 0.42, 0.62, 1.0], dtype=np.float32)
COLOR_CORONARY = np.array([0.90, 0.55, 0.45, 1.0], dtype=np.float32)


@dataclass
class Part:
    name: str
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int32
    color: np.ndarray  # (4,) float32
    kind: str = "static"  # static | ventricle_wall | atrium_wall

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])


@dataclass
class ChamberFrame:
    center: np.ndarray
    axis: np.ndarray  # Einheitsvektor Apex -> Basis
    e_septal: np.ndarray  # Einheitsvektor senkrecht zur Achse, zeigt zum Septum/Nachbarkammer
    e_anterior: np.ndarray  # Einheitsvektor senkrecht zur Achse, zeigt nach anterior
    apex: np.ndarray
    base: np.ndarray
    proj_apex: float
    proj_base: float

    def coordinates(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Liefert (u, angle_deg, radius): u 0 = Apex, 1 = Basis; Winkel 0 = septal, +90 = anterior."""
        rel = points - self.center
        proj = rel @ self.axis
        u = (proj - self.proj_apex) / max(self.proj_base - self.proj_apex, 1e-6)
        radial = rel - np.outer(proj, self.axis)
        a = radial @ self.e_septal
        b = radial @ self.e_anterior
        angle = np.degrees(np.arctan2(b, a))
        radius = np.linalg.norm(radial, axis=1)
        return np.clip(u, -0.2, 1.2), angle, radius


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _chamber_frame(cavity: np.ndarray, toward_base: np.ndarray, toward_septum: np.ndarray) -> ChamberFrame:
    center = cavity.mean(0)
    rel = cavity - center
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    axis = _unit(vt[0])
    if np.dot(axis, toward_base - center) < 0:
        axis = -axis
    proj = rel @ axis
    proj_apex = float(np.percentile(proj, 1.0))
    proj_base = float(np.percentile(proj, 97.0))
    apex = center + axis * proj_apex
    base = center + axis * proj_base
    septal = toward_septum - center
    septal = _unit(septal - axis * np.dot(septal, axis))
    anterior = np.array([0.0, -1.0, 0.0])
    anterior = anterior - axis * np.dot(anterior, axis)
    anterior = anterior - septal * np.dot(anterior, septal)
    anterior = _unit(anterior)
    return ChamberFrame(center, axis, septal, anterior, apex, base, proj_apex, proj_base)


def _aha_segment(u: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """AHA-17-Segmente aus Apex-Basis-Koordinate und Winkel (0 = Septummitte, + = anterior)."""
    seg = np.zeros(len(u), dtype=np.int16)
    a = np.mod(angle + 180.0, 360.0) - 180.0
    basal = u >= 0.7
    mid = (u >= 0.4) & ~basal
    apical = (u >= 0.15) & ~basal & ~mid
    apex = u < 0.15
    six = np.where(a < -120, 5, np.where(a < -60, 4, np.where(a < 0, 3, np.where(a < 60, 2, np.where(a < 120, 1, 6)))))
    # 1 anterior, 2 anteroseptal, 3 inferoseptal, 4 inferior, 5 inferolateral, 6 anterolateral
    seg[basal] = six[basal]
    seg[mid] = six[mid] + 6
    four = np.where(np.abs(a) <= 45, 14, np.where((a > 45) & (a <= 135), 13, np.where((a < -45) & (a >= -135), 15, 16)))
    seg[apical] = four[apical]
    seg[apex] = 17
    return seg


def _clip_part(part: Optional[Part], z_min: float = -1e9, z_max: float = 1e9) -> None:
    """Entfernt Dreiecke außerhalb eines z-Bereichs (Gefäßstümpfe kürzen)."""
    if part is None:
        return
    z = part.vertices[:, 2]
    keep = ((z >= z_min) & (z <= z_max))[part.faces].all(axis=1)
    part.faces = part.faces[keep]


def smooth_path(path: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Chaikin-Glättung einer Polylinie (Endpunkte bleiben erhalten)."""
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            new.append(0.75 * a + 0.25 * b)
            new.append(0.25 * a + 0.75 * b)
        new.append(pts[-1])
        pts = np.asarray(new)
    return pts


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = vertices.astype(np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(v)
    for k in range(3):
        n[:, k] = np.bincount(faces.ravel(), weights=np.repeat(fn[:, k], 3), minlength=len(v))
    length = np.linalg.norm(n, axis=1)
    length[length == 0] = 1.0
    return n / length[:, None]


def _mesh_graph(vertices: np.ndarray, faces: np.ndarray) -> csr_matrix:
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    length = np.linalg.norm(vertices[i] - vertices[j], axis=1).astype(np.float64)
    length = np.maximum(length, 1e-3)
    graph = coo_matrix((length, (i, j)), shape=(len(vertices), len(vertices))).tocsr()
    graph.sum_duplicates()
    # Duplikate wurden addiert -> auf Mittel zurückführen (jede Kante kommt genau zweimal je Richtung vor)
    counts = coo_matrix((np.ones_like(length), (i, j)), shape=graph.shape).tocsr()
    counts.sum_duplicates()
    graph.data = graph.data / np.maximum(counts.data, 1.0)
    return graph


@dataclass
class HeartModel:
    parts: dict[str, Part]
    wall_chamber: np.ndarray  # 0 LV, 1 RV je Kammerwand-Vertex (Septum nach Cavum-Nähe)
    wall_septal: np.ndarray  # bool: Vertex liegt im Septum (beiden Cava nahe)
    wall_endocardial: np.ndarray  # bool (< 2,5 mm zum Cavum)
    wall_inner: np.ndarray  # bool: innere Oberfläche der Wand (Endokard einschließlich trabekulärer Bereiche)
    wall_normal_out: np.ndarray  # (N, 3) Vertexnormale, vom Cavum weg orientiert (transmurale Richtung)
    wall_near_cavity: np.ndarray  # bool, innere Wandhälfte (Trabekel-/Papillarbereich)
    wall_u: np.ndarray  # Apex-Basis-Koordinate der jeweiligen Kammer
    wall_angle: np.ndarray  # Winkel um die Kammerachse (deg)
    wall_segment: np.ndarray  # AHA-Segment (LV 1..17, RV 101..105)
    wall_graph: csr_matrix  # Kantenlängen in mm
    lv_frame: ChamberFrame
    rv_frame: ChamberFrame
    la_frame: ChamberFrame
    ra_frame: ChamberFrame
    landmarks: dict[str, np.ndarray]
    conduction_paths: dict[str, np.ndarray]  # Polylinien (K, 3)
    purkinje_sites: dict[str, np.ndarray]  # Vertex-Indizes der Kammerwand
    atrial_graphs: dict[str, csr_matrix]
    atrial_sources: dict[str, dict[str, int]]  # je Vorhof: name -> Vertex-Index
    center_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Transmurale Kanten (Endokard -> äußere Wand), damit die Erregung die Wand durchquert statt um die Ränder zu laufen
    transmural_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    transmural_length: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    # ------------------------------------------------------------------ Laden
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HeartModel":
        path = Path(path) if path is not None else RESOURCE_PATH
        data = np.load(path)
        names = sorted({key[:-2] for key in data.files if key.endswith("_v")})
        colors = {
            "wall": COLOR_LV, "lv_cavity": COLOR_LV, "rv_cavity": COLOR_RV,
            "la_wall": COLOR_LA, "ra_wall": COLOR_RA, "la_cavity": COLOR_LA, "ra_cavity": COLOR_RA,
            "aorta": COLOR_ARTERY, "aortic_arch": COLOR_ARTERY, "pulmonary_trunk": COLOR_VEIN,
            "svc": COLOR_VEIN, "ivc": COLOR_VEIN, "lad": COLOR_CORONARY, "rca": COLOR_CORONARY, "lcx": COLOR_CORONARY,
        }
        kinds = {"wall": "ventricle_wall", "la_wall": "atrium_wall", "ra_wall": "atrium_wall"}
        parts = {
            name: Part(name, np.asarray(data[name + "_v"], dtype=np.float32), np.asarray(data[name + "_f"], dtype=np.int32),
                       colors.get(name, COLOR_ARTERY).copy(), kinds.get(name, "static"))
            for name in names
        }
        _clip_part(parts.get("ivc"), z_min=-38.0)
        _clip_part(parts.get("svc"), z_max=75.0)
        return cls.build(parts, np.asarray(data["center_mm"], dtype=np.float32) if "center_mm" in data.files else np.zeros(3, np.float32))

    @classmethod
    def build(cls, parts: dict[str, Part], center_mm: np.ndarray) -> "HeartModel":
        wall = parts["wall"].vertices.astype(np.float64)
        lv_cav = parts["lv_cavity"].vertices.astype(np.float64)
        rv_cav = parts["rv_cavity"].vertices.astype(np.float64)
        la_cav = parts["la_cavity"].vertices.astype(np.float64)
        ra_cav = parts["ra_cavity"].vertices.astype(np.float64)
        la_center, ra_center = la_cav.mean(0), ra_cav.mean(0)
        lv_frame = _chamber_frame(lv_cav, la_center, rv_cav.mean(0))
        rv_frame = _chamber_frame(rv_cav, ra_center, lv_cav.mean(0))
        la_frame = _chamber_frame(la_cav, la_center + np.array([0, 0, 40.0]), ra_center)
        ra_frame = _chamber_frame(ra_cav, ra_center + np.array([0, 0, 40.0]), la_center)

        d_lv = cKDTree(lv_cav).query(wall)[0]
        d_rv = cKDTree(rv_cav).query(wall)[0]
        chamber = np.where(d_lv <= d_rv, CHAMBER_LV, CHAMBER_RV).astype(np.int8)
        septal = (d_lv < 13.0) & (d_rv < 13.0)
        endocardial = np.minimum(d_lv, d_rv) < 2.5
        near_cavity = np.minimum(d_lv, d_rv) < 7.0
        # Innere Oberfläche: nahe am Cavum oder (bis 8 mm) mit Normale zum Cavum hin (trabekuläre Bereiche)
        d_cav = np.minimum(d_lv, d_rv)
        target = np.where((d_lv <= d_rv)[:, None], lv_cav[cKDTree(lv_cav).query(wall)[1]], rv_cav[cKDTree(rv_cav).query(wall)[1]])
        to_cav = target - wall
        to_cav /= np.linalg.norm(to_cav, axis=1)[:, None] + 1e-9
        normals = _vertex_normals(wall, parts["wall"].faces)
        inner = endocardial | ((d_cav < 8.0) & ((normals * to_cav).sum(1) > 0.2))
        normal_out = np.where(((normals * to_cav).sum(1) > 0)[:, None], -normals, normals)

        u_lv, ang_lv, _ = lv_frame.coordinates(wall)
        u_rv, ang_rv, _ = rv_frame.coordinates(wall)
        u = np.where(chamber == CHAMBER_LV, u_lv, u_rv)
        angle = np.where(chamber == CHAMBER_LV, ang_lv, ang_rv)
        segment = _aha_segment(np.clip(u_lv, 0, 1), ang_lv)
        rv_seg = np.where(u_rv < 0.2, 105, np.where((u_rv > 0.75) & (ang_rv > 30), 104, np.where(ang_rv > 60, 101, np.where(ang_rv > -60, 102, 103))))
        # 101 anterior, 102 lateral(septumnah frei), 103 inferior, 104 Ausflusstrakt, 105 Apex
        segment = np.where(chamber == CHAMBER_LV, segment, rv_seg).astype(np.int16)

        graph = _mesh_graph(parts["wall"].vertices, parts["wall"].faces)
        # Transmurale Verbindungen: jeder äußere Wandvertex zum nächsten endokardialen Vertex derselben Kammer
        tm_edges, tm_len = [], []
        for ch in (CHAMBER_LV, CHAMBER_RV):
            endo_idx = np.where((chamber == ch) & inner)[0]
            outer_idx = np.where((chamber == ch) & ~inner)[0]
            if endo_idx.size == 0 or outer_idx.size == 0:
                continue
            dist, nearest = cKDTree(wall[endo_idx]).query(wall[outer_idx])
            ok = dist < 18.0
            tm_edges.append(np.column_stack([endo_idx[nearest[ok]], outer_idx[ok]]))
            tm_len.append(np.maximum(dist[ok], 1.0))
        transmural_edges = np.vstack(tm_edges) if tm_edges else np.zeros((0, 2), np.int64)
        transmural_length = np.concatenate(tm_len) if tm_len else np.zeros(0)

        landmarks: dict[str, np.ndarray] = {
            "lv_center": lv_frame.center, "rv_center": rv_frame.center,
            "la_center": la_center, "ra_center": ra_center,
            "lv_apex": lv_frame.apex, "rv_apex": rv_frame.apex,
            "lv_base": lv_frame.base, "rv_base": rv_frame.base,
        }
        # Sinusknoten: RA-Wandpunkt nahe der Einmündung der oberen Hohlvene
        ra_wall = parts["ra_wall"].vertices.astype(np.float64)
        la_wall = parts["la_wall"].vertices.astype(np.float64)
        svc = parts["svc"].vertices.astype(np.float64) if "svc" in parts else ra_wall
        svc_bottom = svc[np.argmin(svc[:, 2])]
        sa_index = int(np.argmin(np.linalg.norm(ra_wall - svc_bottom, axis=1)))
        landmarks["sa_node"] = ra_wall[sa_index]
        # AV-Knoten: septaler Wandpunkt nahe der Basis, dem rechten Vorhof am nächsten
        septal_basal = np.where(septal & (u_lv > 0.75))[0]
        if septal_basal.size == 0:
            septal_basal = np.where(septal)[0]
        d_ra = cKDTree(ra_cav).query(wall[septal_basal])[0]
        av_index = int(septal_basal[np.argmin(d_ra)])
        landmarks["av_node"] = wall[av_index]
        landmarks["his"] = wall[av_index] - lv_frame.axis * 8.0
        av_ra_index = int(np.argmin(np.linalg.norm(ra_wall - wall[av_index], axis=1)))
        av_la_index = int(np.argmin(np.linalg.norm(la_wall - wall[av_index], axis=1)))
        bachmann_index = int(np.argmin(np.linalg.norm(la_wall - ra_wall[sa_index], axis=1)))

        def _bin_path(mask: np.ndarray, coordinate: np.ndarray, start: float, stop: float, bins: int) -> np.ndarray:
            idx = np.where(mask)[0]
            if idx.size == 0:
                return np.zeros((0, 3))
            edges = np.linspace(start, stop, bins + 1)
            pts = []
            for k in range(bins):
                sel = idx[(coordinate[idx] <= edges[k]) & (coordinate[idx] > edges[k + 1])]
                if sel.size:
                    pts.append(wall[sel].mean(0))
            return np.asarray(pts) if pts else np.zeros((0, 3))

        lv_endo_septal = (chamber == CHAMBER_LV) & endocardial & septal
        rv_endo_septal = (chamber == CHAMBER_RV) & endocardial & septal
        lbb = _bin_path(lv_endo_septal & (np.abs(ang_lv) < 35), u_lv, 0.9, 0.45, 6)
        lbb_ant = _bin_path((chamber == CHAMBER_LV) & endocardial & (ang_lv > 25) & (ang_lv < 80), u_lv, 0.75, 0.3, 5)
        lbb_post = _bin_path((chamber == CHAMBER_LV) & endocardial & (ang_lv < -25) & (ang_lv > -80), u_lv, 0.75, 0.3, 5)
        rbb = _bin_path(rv_endo_septal, u_rv, 0.9, 0.25, 7)
        conduction_paths = {
            "his": np.vstack([landmarks["av_node"], landmarks["his"]]),
            "lbb": np.vstack([landmarks["his"], lbb]) if lbb.size else np.vstack([landmarks["his"]]),
            "lbb_anterior": np.vstack([lbb[len(lbb) // 2] if lbb.size else landmarks["his"], lbb_ant]) if lbb_ant.size else np.zeros((0, 3)),
            "lbb_posterior": np.vstack([lbb[len(lbb) // 2] if lbb.size else landmarks["his"], lbb_post]) if lbb_post.size else np.zeros((0, 3)),
            "rbb": np.vstack([landmarks["his"], rbb]) if rbb.size else np.vstack([landmarks["his"]]),
        }
        conduction_paths = {name: smooth_path(path) if len(path) >= 3 else path for name, path in conduction_paths.items()}

        def _site(mask: np.ndarray, count: int = 12) -> np.ndarray:
            idx = np.where(mask)[0]
            if idx.size == 0:
                return np.zeros(0, dtype=np.int64)
            centre = wall[idx].mean(0)
            order = np.argsort(np.linalg.norm(wall[idx] - centre, axis=1))
            return idx[order[:count]]

        lv_endo = (chamber == CHAMBER_LV) & endocardial
        rv_endo = (chamber == CHAMBER_RV) & endocardial
        purkinje_sites = {
            "lv_anterior_paraseptal": _site(lv_endo & (u_lv > 0.55) & (u_lv < 0.8) & (ang_lv > 30) & (ang_lv < 75)),
            "lv_mid_septal": _site(lv_endo & (u_lv > 0.35) & (u_lv < 0.6) & (np.abs(ang_lv) < 25)),
            "lv_posterior_paraseptal": _site(lv_endo & (u_lv > 0.45) & (u_lv < 0.75) & (ang_lv < -30) & (ang_lv > -75)),
            "lv_apical": _site(lv_endo & (u_lv < 0.35) & (u_lv > 0.1)),
            "rv_anterior_papillary": _site(rv_endo & (u_rv > 0.25) & (u_rv < 0.55) & (ang_rv > 40)),
            "rv_apical": _site(rv_endo & (u_rv < 0.25)),
            "rv_outflow": _site(rv_endo & (u_rv > 0.8)),
            "lv_lateral_base": _site((chamber == CHAMBER_LV) & (u_lv > 0.75) & (np.abs(ang_lv) > 140)),
            "rv_free_wall_base": _site((chamber == CHAMBER_RV) & (u_rv > 0.7) & (ang_rv > 60) & (ang_rv < 150)),
        }
        atrial_graphs = {
            "la_wall": _mesh_graph(parts["la_wall"].vertices, parts["la_wall"].faces),
            "ra_wall": _mesh_graph(parts["ra_wall"].vertices, parts["ra_wall"].faces),
        }
        atrial_sources = {
            "ra_wall": {"sa_node": sa_index, "av_node": av_ra_index},
            "la_wall": {"bachmann": bachmann_index, "av_node": av_la_index},
        }
        return cls(parts, chamber, septal, endocardial, inner, normal_out.astype(np.float32), near_cavity, u.astype(np.float32), angle.astype(np.float32), segment, graph,
                   lv_frame, rv_frame, la_frame, ra_frame, landmarks, conduction_paths, purkinje_sites,
                   atrial_graphs, atrial_sources, np.asarray(center_mm, dtype=np.float32),
                   transmural_edges.astype(np.int64), transmural_length.astype(np.float64))

    # ------------------------------------------------------------ Hilfen
    @property
    def wall(self) -> Part:
        return self.parts["wall"]

    def wall_vertex_near(self, point: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        vertices = self.wall.vertices
        idx = np.arange(len(vertices)) if mask is None else np.where(mask)[0]
        return int(idx[np.argmin(np.linalg.norm(vertices[idx] - point, axis=1))])

    def wall_vertex_in_direction(self, direction: np.ndarray, chamber: Optional[int] = None) -> int:
        """Wandvertex in Richtung ``direction`` vom Herzzentrum (für ektope Ursprünge)."""
        d = _unit(np.asarray(direction, dtype=np.float64))
        vertices = self.wall.vertices.astype(np.float64)
        centre = 0.5 * (self.lv_frame.center + self.rv_frame.center)
        rel = vertices - centre
        norm = np.linalg.norm(rel, axis=1) + 1e-6
        cosine = (rel @ d) / norm
        mask = cosine > 0.85
        if chamber is not None:
            mask &= self.wall_chamber == chamber
        if not mask.any():
            mask = cosine > 0.6
            if not mask.any():
                return int(np.argmax(cosine))
        idx = np.where(mask)[0]
        # Epikardnah bevorzugen (äußere Hälfte der Radien), damit der Fokus an der Wand sitzt
        radii = norm[idx]
        return int(idx[np.argmin(np.abs(radii - np.percentile(radii, 60)))])

    def extent(self) -> float:
        return float(np.linalg.norm(self.wall.vertices, axis=1).max())


@lru_cache(maxsize=1)
def default_model() -> HeartModel:
    return HeartModel.load()
