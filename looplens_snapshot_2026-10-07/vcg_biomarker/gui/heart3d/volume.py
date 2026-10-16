"""Volumetrisches Wandgitter für Erregungsrechnung und Vorwärtsmodell.

Die BodyParts3D-Kammerwand ist ein dünnes Oberflächennetz; für die Ausbreitung durch die
Wand (endokardial → epikardial) und für das Doppelschicht-Quellmodell braucht es aber ein
Volumen. Deshalb wird je Kammer aus dem glatten Cavum-Netz ein Schalengitter aufgebaut:
das Cavum-Netz ist die Endokardschicht, weitere Schichten liegen entlang der Cavum-
Normalen nach außen bis zur anatomischen Wanddicke (LV/Septum ≈ 10 mm, RV ≈ 4 mm,
Apex ≈ 6 mm; Kawel 2012, Ho 2006). Kanten verlaufen innerhalb der Schichten (Netz-
kanten) und zwischen den Schichten (transmural); am Septum werden LV- und RV-Schale
gekoppelt. Die Kammerwand-Vertices des Anzeigenetzes werden ihrem nächsten Gitter-
knoten zugeordnet, sodass Farben und Mechanik unverändert auf dem Anzeigenetz laufen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from .model import CHAMBER_LV, CHAMBER_RV, HeartModel, _aha_segment

LV_THICKNESS_MM = 10.0
RV_THICKNESS_MM = 4.0
APEX_THICKNESS_MM = 6.0
SEPTAL_DISTANCE_MM = 13.0


def subdivide(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mittelpunkt-Unterteilung (jedes Dreieck -> 4)."""
    v = vertices.astype(np.float64)
    edges = {}
    new_v = [v]
    counter = [len(v)]

    def mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in edges:
            edges[key] = counter[0]
            counter[0] += 1
            new_v.append(0.5 * (v[a] + v[b])[None, :])
        return edges[key]

    out = []
    for a, b, c in faces:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    return np.vstack(new_v), np.asarray(out, dtype=np.int64)


def _vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(v)
    for k in range(3):
        n[:, k] = np.bincount(f.ravel(), weights=np.repeat(fn[:, k], 3), minlength=len(v))
    length = np.linalg.norm(n, axis=1)
    length[length == 0] = 1.0
    return n / length[:, None]


def _vertex_area(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return np.bincount(f.ravel(), weights=np.repeat(area / 3.0, 3), minlength=len(v))


@dataclass
class WallGrid:
    positions: np.ndarray  # (N, 3)
    chamber: np.ndarray  # int8
    layer: np.ndarray  # 0 = Endokard .. K = Epikard
    n_layers: np.ndarray  # K je Knoten
    septal: np.ndarray  # bool
    endo: np.ndarray  # bool (Schicht 0)
    epi: np.ndarray  # bool (äußerste Schicht)
    u: np.ndarray  # Apex-Basis-Koordinate
    angle: np.ndarray
    segment: np.ndarray
    normal_out: np.ndarray  # (N, 3) transmurale Richtung (vom Cavum weg)
    volume: np.ndarray  # mm³
    edges_i: np.ndarray
    edges_j: np.ndarray
    edges_len: np.ndarray
    edge_transmural: np.ndarray  # bool
    node_of_vertex: np.ndarray  # Anzeigenetz-Vertex -> Gitterknoten
    purkinje_sites: dict[str, np.ndarray]  # name -> Knotenindizes (Endokard)

    @property
    def n_nodes(self) -> int:
        return int(len(self.positions))

    def node_near(self, point: np.ndarray, endo_only: bool = False) -> int:
        idx = np.where(self.endo)[0] if endo_only else np.arange(self.n_nodes)
        return int(idx[np.argmin(np.linalg.norm(self.positions[idx] - point, axis=1))])


def build_wall_grid(model: HeartModel, layers_lv: int = 3, layers_rv: int = 2) -> WallGrid:
    lv_cav = model.parts["lv_cavity"]
    rv_cav = model.parts["rv_cavity"]
    nodes, chamber, layer, n_layers, septal, u_all, ang_all, seg_all, normals, volume = [], [], [], [], [], [], [], [], [], []
    ei, ej, elen, etm = [], [], [], []
    offset = 0
    other_tree = {CHAMBER_LV: cKDTree(rv_cav.vertices.astype(np.float64)), CHAMBER_RV: cKDTree(lv_cav.vertices.astype(np.float64))}
    for ch, part, frame, K, thick in ((CHAMBER_LV, lv_cav, model.lv_frame, layers_lv, LV_THICKNESS_MM),
                                      (CHAMBER_RV, rv_cav, model.rv_frame, layers_rv, RV_THICKNESS_MM)):
        v0, f = subdivide(part.vertices, part.faces)
        n0 = _vertex_normals(v0, f)
        centre = v0.mean(0)
        flip = ((v0 - centre) * n0).sum(1) < 0
        n0[flip] *= -1.0  # nach außen (vom Cavum weg)
        u, ang, _ = frame.coordinates(v0)
        u = np.clip(u, 0.0, 1.0)
        d_other = other_tree[ch].query(v0)[0]
        is_septal = d_other < SEPTAL_DISTANCE_MM
        t_node = np.full(len(v0), thick)
        t_node[u < 0.15] = APEX_THICKNESS_MM
        if ch == CHAMBER_RV:
            t_node[is_septal] = 3.0  # RV-seitige Septumschicht; der Rest des Septums gehört zur LV-Schale
        else:
            t_node[is_septal] = LV_THICKNESS_MM
        seg = _aha_segment(u, ang) if ch == CHAMBER_LV else np.where(
            u < 0.2, 105, np.where((u > 0.75) & (ang > 30), 104, np.where(ang > 60, 101, np.where(ang > -60, 102, 103)))).astype(np.int16)
        area0 = _vertex_area(v0, f)
        e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        base = len(v0)
        for k in range(K + 1):
            pos = v0 + n0 * (t_node * k / K)[:, None]
            nodes.append(pos)
            chamber.append(np.full(base, ch, np.int8))
            layer.append(np.full(base, k, np.int16))
            n_layers.append(np.full(base, K, np.int16))
            septal.append(is_septal)
            u_all.append(u)
            ang_all.append(ang)
            seg_all.append(seg)
            normals.append(n0)
            # Schichtdicke: innere/äußere Schicht halb
            dz = t_node / K * (0.5 if k in (0, K) else 1.0)
            volume.append(area0 * dz)
            # Kanten innerhalb der Schicht
            a, b = offset + k * base + e[:, 0], offset + k * base + e[:, 1]
            ei.append(a)
            ej.append(b)
            elen.append(np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1))
            etm.append(np.zeros(len(e), bool))
            if k < K:
                # transmurale Kanten zur nächsten Schicht
                idx = np.arange(base)
                ei.append(offset + k * base + idx)
                ej.append(offset + (k + 1) * base + idx)
                elen.append(np.maximum(t_node / K, 0.5))
                etm.append(np.ones(base, bool))
        offset += (K + 1) * base
    positions = np.vstack(nodes)
    chamber = np.concatenate(chamber)
    layer = np.concatenate(layer)
    n_layers = np.concatenate(n_layers)
    septal = np.concatenate(septal)
    u_all = np.concatenate(u_all)
    ang_all = np.concatenate(ang_all)
    seg_all = np.concatenate(seg_all)
    normals = np.vstack(normals)
    volume = np.concatenate(volume)
    ei, ej, elen, etm = np.concatenate(ei), np.concatenate(ej), np.concatenate(elen), np.concatenate(etm)
    # Septale Kopplung: äußere LV-Schale <-> RV-Knoten (Septum ist ein gemeinsames Gewebe)
    lv_outer = np.where((chamber == CHAMBER_LV) & (layer == n_layers) & septal)[0]
    rv_sept = np.where((chamber == CHAMBER_RV) & septal)[0]
    if lv_outer.size and rv_sept.size:
        # jeder septale RV-Knoten an seinen nächsten äußeren LV-Septumknoten (das Septum ist ein Gewebe;
        # Restabstand wird mit Myokardgeschwindigkeit überbrückt) und umgekehrt
        d1, j1 = cKDTree(positions[lv_outer]).query(positions[rv_sept])
        d2, j2 = cKDTree(positions[rv_sept]).query(positions[lv_outer])
        ok1, ok2 = d1 < 14.0, d2 < 14.0
        ei = np.concatenate([ei, rv_sept[ok1], lv_outer[ok2]])
        ej = np.concatenate([ej, lv_outer[j1[ok1]], rv_sept[j2[ok2]]])
        elen = np.concatenate([elen, np.maximum(d1[ok1], 0.5), np.maximum(d2[ok2], 0.5)])
        etm = np.concatenate([etm, np.ones(int(ok1.sum() + ok2.sum()), bool)])
    endo = layer == 0
    epi = layer == n_layers
    # Zuordnung Anzeigenetz -> Gitter
    node_of_vertex = cKDTree(positions).query(model.wall.vertices.astype(np.float64))[1]
    # Purkinje-Orte: Endokardknoten nahe den Zentren der Wand-basierten Orte
    endo_idx = np.where(endo)[0]
    endo_tree = cKDTree(positions[endo_idx])
    purkinje_sites = {}
    for name, verts in model.purkinje_sites.items():
        if len(verts) == 0:
            continue
        centre = model.wall.vertices[verts].mean(0)
        _, near = endo_tree.query(centre, k=min(6, len(endo_idx)))
        purkinje_sites[name] = endo_idx[np.atleast_1d(near)]
    return WallGrid(positions, chamber, layer, n_layers, septal, endo, epi, u_all.astype(np.float32), ang_all.astype(np.float32),
                    seg_all.astype(np.int16), normals, volume, ei.astype(np.int64), ej.astype(np.int64), elen.astype(np.float64), etm,
                    node_of_vertex.astype(np.int64), purkinje_sites)
