"""Erregungsausbreitung auf dem Herzmodell.

Die Aktivierungszeit jedes Wandvertex entsteht aus Ursprungsquellen (Purkinje-
Durchbrüche, ektoper Fokus, Schrittmacherelektrode, akzessorische Bahn) und einer
kürzesten-Wege-Rechnung auf dem Mesh (Dijkstra) mit zwei Leitungsgeschwindigkeiten:
schnell entlang des endokardialen Purkinje-Netzes, langsam im Arbeitsmyokard. Die
rohen Zeiten werden auf die gemessene QRS-Dauer des jeweiligen Schlags skaliert;
das Muster (Reihenfolge) kommt aus dem Modell, die Dauer aus dem ECG.

Musterbibliothek (Stufe P, Physiologie): Durrer 1970, Auricchio 2004 (LSB),
Fantoni 2005 (RSB). Stufe H (Hypothese) sind die Modulationen aus den
Projektmarkern; sie verzerren nur die Zeitachse, nie die Reihenfolge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .model import CHAMBER_LV, CHAMBER_RV, HeartModel
from .volume import WallGrid, build_wall_grid

# Leitungsparameter (kalibriert 2026-09-22 an PTB-XL, s. INVERSE_SIMULATION_RECHERCHE): rohe Aktivierungsdauer
# normal ≈ 84 ms, LSB ≈ 137 ms, RSB ≈ 130 ms, LAFB ≈ 91 ms
V_PURKINJE = 2.0  # mm/ms, endokardiales Netz
V_MUSCLE = 0.55  # mm/ms, Arbeitsmyokard entlang der Wand
V_TRANSMURAL = 0.3  # mm/ms, quer zur Faser (endokardial -> epikardial; kalibriert)
V_BLOCKED_ENDO = 0.5  # mm/ms, Endokard der blockierten Kammer (retrograd erreichtes Netz, langsame Zonen)
V_SEPTAL_SLOW = 0.9  # mm/ms, septale Endokardschicht ohne Faszikelversorgung
V_ATRIUM = 0.9  # mm/ms


class Pattern(str, Enum):
    AUTO = "auto"
    NORMAL = "normal"
    LBBB = "lbbb"
    RBBB = "rbbb"
    LAFB = "lafb"
    LPFB = "lpfb"
    RV_PACING = "rv_pacing"
    WPW = "wpw"
    ECTOPIC = "ectopic"


PATTERN_LABELS = {
    Pattern.AUTO: "Automatisch aus der Analyse",
    Pattern.NORMAL: "Normale Erregung (Durrer)",
    Pattern.LBBB: "Linksschenkelblock",
    Pattern.RBBB: "Rechtsschenkelblock",
    Pattern.LAFB: "Linksanteriorer Faszikelblock",
    Pattern.LPFB: "Linksposteriorer Faszikelblock",
    Pattern.RV_PACING: "Rechtsventrikuläre Stimulation",
    Pattern.WPW: "Präexzitation (akzessorische Bahn)",
    Pattern.ECTOPIC: "Ektoper Fokus",
}


class AtrialPattern(str, Enum):
    SINUS = "sinus"
    RETROGRADE = "retrograde"
    ECTOPIC_LEFT = "ectopic_left"
    FLUTTER = "flutter"
    FIBRILLATION = "fibrillation"
    NONE = "none"


@dataclass
class ActivationResult:
    times_ms: np.ndarray  # je Kammerwand-Vertex, relativ zum QRS-Beginn
    pattern: Pattern
    tier: str  # P Physiologie | K Kriterium | H Hypothese
    label: str
    origin_vertices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    origin_label: str = ""
    notes: tuple[str, ...] = ()
    focal_vertices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # betont dargestellte Ursprünge
    node_times_ms: Optional[np.ndarray] = None  # Zeiten auf dem volumetrischen Wandgitter (für das Vorwärtsmodell)
    blocked: tuple[str, ...] = ()


@dataclass
class AtrialResult:
    ra_times_ms: np.ndarray
    la_times_ms: np.ndarray
    pattern: AtrialPattern
    label: str


class ActivationSolver:
    """Berechnet und cached Aktivierungskarten je Muster."""

    def __init__(self, model: HeartModel, grid: Optional[WallGrid] = None):
        self.model = model
        self.grid = grid or wall_grid_for(model)
        self._cache: dict[tuple, np.ndarray] = {}
        g = self.grid
        # Kanten in beide Richtungen; transmural gilt die langsamere der beiden Knotengeschwindigkeiten
        self._edges_i = np.concatenate([g.edges_i, g.edges_j])
        self._edges_j = np.concatenate([g.edges_j, g.edges_i])
        self._edges_len = np.concatenate([g.edges_len, g.edges_len])
        self._edge_transmural = np.concatenate([g.edge_transmural, g.edge_transmural])
        self._n = g.n_nodes
        # Leitungsparameter (kalibrierbar): Purkinje-, Myokard-, septale Endokardgeschwindigkeit (mm/ms),
        # Endokardgeschwindigkeit der blockierten Kammer, transmurale Geschwindigkeit
        self.params: dict[str, float] = dict(v_purkinje=V_PURKINJE, v_muscle=V_MUSCLE, v_septal_slow=V_SEPTAL_SLOW,
                                             v_blocked_endo=V_BLOCKED_ENDO, v_transmural=V_TRANSMURAL,
                                             purkinje_lv_u=0.7, purkinje_rv_u=0.65, purkinje_posterobasal_u=0.55,
                                             rv_entry_offset_ms=12.0, size_factor=1.0)  # RV-Eintritt ~12 ms nach LV (Durrer 1970; kalibriert)

    def set_params(self, **kwargs: float) -> None:
        checked = {}
        for key, value in kwargs.items():
            if key not in self.params:
                raise ValueError(f"Unbekannter Simulationsparameter: {key}")
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"Nicht-endlicher Simulationsparameter: {key}")
            if key.startswith("v_") or key == "size_factor":
                if number <= 0:
                    raise ValueError(f"Leitgeschwindigkeit/Größenfaktor muss positiv sein: {key}")
            elif key.startswith("purkinje_") and not 0 <= number <= 1:
                raise ValueError(f"Purkinje-Flächenanteil außerhalb [0, 1]: {key}")
            elif key == "rv_entry_offset_ms" and number < 0:
                raise ValueError("RV-Eintrittsverzögerung darf nicht negativ sein")
            checked[key] = number
        self.params.update(checked)
        self._cache.clear()

    def node_of(self, wall_vertex: Optional[int]) -> Optional[int]:
        """Anzeigenetz-Vertex -> Gitterknoten."""
        return None if wall_vertex is None else int(self.grid.node_of_vertex[int(wall_vertex)])

    # ------------------------------------------------------------ Kern
    def _solve(self, sources: dict[int, float], velocity: np.ndarray) -> np.ndarray:
        """Dijkstra mit virtuellem Quellknoten; Kantengewichte in ms."""
        v_edge = 0.5 * (velocity[self._edges_i] + velocity[self._edges_j])
        v_tm = np.minimum(np.minimum(velocity[self._edges_i], velocity[self._edges_j]), self.params["v_transmural"])
        v_edge = np.where(self._edge_transmural, v_tm, v_edge)
        weights = self._edges_len * self.params.get("size_factor", 1.0) / np.maximum(v_edge, 1e-3)
        n = self._n
        offset = -min(min(sources.values()), 0.0)  # negative Startzeiten (Präexzitation) verschieben
        rows = np.concatenate([self._edges_i, np.full(len(sources), n)])
        cols = np.concatenate([self._edges_j, np.fromiter(sources.keys(), dtype=np.int64, count=len(sources))])
        data = np.concatenate([weights, np.fromiter((v + offset for v in sources.values()), dtype=np.float64, count=len(sources))])
        graph = coo_matrix((data, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
        times = dijkstra(graph, directed=True, indices=[n], min_only=True)
        times = np.asarray(times).ravel()[:n] - offset
        if not np.isfinite(times).all():
            missing = int((~np.isfinite(times)).sum())
            raise ValueError(f"Aktivierungsnetz nicht vollständig erreichbar: {missing} Knoten")
        return times

    def _velocity_field(self, lv_purkinje: bool, rv_purkinje: bool, lv_anterior_block: bool = False,
                        lv_posterior_block: bool = False, all_muscle: bool = False) -> np.ndarray:
        g = self.grid
        p = self.params
        v = np.full(self._n, p["v_muscle"], dtype=np.float64)
        if all_muscle:
            return v
        # Purkinje-Netz auf dem Endokard bis nahe an die Basis (Durrer 1970); ausgespart bleiben der
        # posterobasale LV-Bereich und der RV-Ausflusstrakt, die zuletzt über das Arbeitsmyokard erreicht werden.
        p_lv, p_rv, p_pb = p.get("purkinje_lv_u", 0.88), p.get("purkinje_rv_u", 0.8), p.get("purkinje_posterobasal_u", 0.72)
        lv_posterobasal = (g.u > p_pb) & (g.angle < -50) & (g.angle > -160)
        lv_endo = (g.chamber == CHAMBER_LV) & g.endo & (((g.u < p_lv) & ~lv_posterobasal) | (g.u < 0.12))
        rv_endo = (g.chamber == CHAMBER_RV) & g.endo & (((g.u < p_rv) & (g.segment != 104)) | (g.u < 0.12))
        if lv_purkinje:
            v[lv_endo] = p["v_purkinje"]
            if lv_anterior_block:
                v[lv_endo & (g.angle > 20) & (g.angle < 120)] = p["v_septal_slow"]
            if lv_posterior_block:
                v[lv_endo & (g.angle < -20) & (g.angle > -120)] = p["v_septal_slow"]
        else:
            v[lv_endo] = p["v_blocked_endo"]
            v[lv_endo & g.septal] = p["v_septal_slow"]
        if rv_purkinje:
            v[rv_endo] = p["v_purkinje"]
        else:
            v[rv_endo] = p["v_blocked_endo"]
            v[rv_endo & g.septal] = p["v_septal_slow"]
        return v

    def _sources_for(self, pattern: Pattern, origin_vertex: Optional[int], breakthrough_sites: int,
                     accessory_vertex: Optional[int]) -> dict[int, float]:
        """Quellen als Gitterknoten; origin_vertex/accessory_vertex sind Anzeigenetz-Vertices."""
        sites = self.grid.purkinje_sites
        sources: dict[int, float] = {}
        rv_off = self.params.get("rv_entry_offset_ms", 0.0)
        origin_vertex = self.node_of(origin_vertex)
        accessory_vertex = self.node_of(accessory_vertex)

        def add(name: str, t0: float) -> None:
            for idx in sites.get(name, ()):
                sources[int(idx)] = min(t0, sources.get(int(idx), np.inf))

        if pattern in (Pattern.NORMAL, Pattern.LAFB, Pattern.LPFB, Pattern.RBBB, Pattern.WPW):
            lv_order = ["lv_mid_septal", "lv_anterior_paraseptal", "lv_posterior_paraseptal", "lv_apical"]
            if pattern == Pattern.LAFB:
                lv_order.remove("lv_anterior_paraseptal")
            if pattern == Pattern.LPFB:
                lv_order.remove("lv_posterior_paraseptal")
            for k, name in enumerate(lv_order[:max(1, breakthrough_sites)]):
                add(name, 2.0 * k)
            if pattern != Pattern.RBBB:
                add("rv_anterior_papillary", 6.0 + rv_off)
                add("rv_apical", 12.0 + rv_off)
            if pattern == Pattern.WPW and accessory_vertex is not None:
                sources[int(accessory_vertex)] = -30.0
        elif pattern == Pattern.LBBB:
            add("rv_anterior_papillary", 0.0)
            add("rv_apical", 6.0)
        elif pattern == Pattern.RV_PACING:
            for idx in sites.get("rv_apical", ()):
                sources[int(idx)] = 0.0
        elif pattern == Pattern.ECTOPIC:
            if origin_vertex is None:
                origin_vertex = int(sites["lv_lateral_base"][0]) if len(sites.get("lv_lateral_base", ())) else 0
            sources[int(origin_vertex)] = 0.0
        if not sources:
            sources[int(sites["lv_mid_septal"][0])] = 0.0
        return sources

    def ventricular(self, pattern: Pattern, qrs_duration_ms: float, *, origin_vertex: Optional[int] = None,
                    breakthrough_sites: int = 4, accessory_vertex: Optional[int] = None,
                    arc_fraction_half_time: Optional[float] = None) -> ActivationResult:
        key = (pattern.value, origin_vertex, breakthrough_sites, accessory_vertex)
        raw = self._cache.get(key)
        if raw is None:
            if pattern == Pattern.ECTOPIC or pattern == Pattern.RV_PACING:
                velocity = self._velocity_field(False, False, all_muscle=True)
                # Nach dem ersten Drittel wird das Purkinje-Netz retrograd erreicht: Endokard etwas schneller
                velocity[self.grid.endo] = 1.1
            else:
                velocity = self._velocity_field(
                    lv_purkinje=pattern != Pattern.LBBB, rv_purkinje=pattern != Pattern.RBBB,
                    lv_anterior_block=pattern == Pattern.LAFB, lv_posterior_block=pattern == Pattern.LPFB)
            sources = self._sources_for(pattern, origin_vertex, breakthrough_sites, accessory_vertex)
            raw = self._solve(sources, velocity)
            self._cache[key] = raw
        times = raw - raw.min()
        span = float(np.percentile(times[self.grid.node_of_vertex], 97.0))  # Dauer auf dem Anzeigenetz = QRS-Dauer
        if span <= 0:
            span = 1.0
        times = times / span * float(qrs_duration_ms)
        notes: list[str] = []
        tier = "P"
        if arc_fraction_half_time is not None and np.isfinite(arc_fraction_half_time) and 0.2 < arc_fraction_half_time < 0.8:
            gamma = float(np.log(arc_fraction_half_time) / np.log(0.5))
            gamma = float(np.clip(gamma, 0.6, 1.6))
            tau = np.clip(times / max(float(qrs_duration_ms), 1e-3), 0.0, 1.0)
            times = np.power(tau, gamma) * float(qrs_duration_ms)
            tier = "H"
            notes.append(f"Zeitverzerrung aus Bogenlängenanteil {arc_fraction_half_time:.2f} (Hypothese, Reihenfolge unverändert)")
        sources = self._sources_for(pattern, origin_vertex, breakthrough_sites, accessory_vertex)
        origin = self.vertices_of_nodes(np.fromiter(sources.keys(), dtype=np.int64, count=len(sources)))
        if pattern in (Pattern.ECTOPIC, Pattern.RV_PACING):
            focal = np.array([int(origin_vertex)], dtype=np.int64) if origin_vertex is not None else origin
        elif pattern == Pattern.WPW and accessory_vertex is not None:
            focal = np.array([int(accessory_vertex)], dtype=np.int64)
        else:
            focal = np.zeros(0, dtype=np.int64)
        return ActivationResult(times[self.grid.node_of_vertex].astype(np.float32), pattern, tier, PATTERN_LABELS[pattern],
                                origin, "", tuple(notes), focal, node_times_ms=times.astype(np.float32))

    def vertices_of_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Gitterknoten -> je ein Anzeigenetz-Vertex (nächster)."""
        if nodes.size == 0:
            return nodes
        wall = self.model.wall.vertices
        pos = self.grid.positions[nodes]
        from scipy.spatial import cKDTree
        return cKDTree(wall).query(pos)[1].astype(np.int64)

    # ------------------------------------------------------------ Vorhöfe
    def _atrial_solve(self, part: str, sources: dict[int, float]) -> np.ndarray:
        graph = self.model.atrial_graphs[part].tocoo()
        n = graph.shape[0]
        weights = graph.data / V_ATRIUM
        rows = np.concatenate([graph.row, np.full(len(sources), n)])
        cols = np.concatenate([graph.col, np.fromiter(sources.keys(), dtype=np.int64, count=len(sources))])
        data = np.concatenate([weights, np.fromiter(sources.values(), dtype=np.float64, count=len(sources))])
        g = coo_matrix((data, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
        times = np.asarray(dijkstra(g, directed=True, indices=[n], min_only=True)).ravel()[:n]
        finite = np.isfinite(times)
        times[~finite] = times[finite].max() if finite.any() else 0.0
        return times

    def atrial(self, pattern: AtrialPattern, p_duration_ms: float) -> AtrialResult:
        key = ("atrial", pattern.value)
        cached = self._cache.get(key)
        src = self.model.atrial_sources
        if cached is None:
            if pattern == AtrialPattern.SINUS:
                ra = self._atrial_solve("ra_wall", {src["ra_wall"]["sa_node"]: 0.0})
                la = self._atrial_solve("la_wall", {src["la_wall"]["bachmann"]: 25.0})
            elif pattern == AtrialPattern.RETROGRADE:
                ra = self._atrial_solve("ra_wall", {src["ra_wall"]["av_node"]: 0.0})
                la = self._atrial_solve("la_wall", {src["la_wall"]["av_node"]: 0.0})
            elif pattern == AtrialPattern.ECTOPIC_LEFT:
                la_vertices = self.model.parts["la_wall"].vertices
                far = int(np.argmax(la_vertices[:, 0]))
                la = self._atrial_solve("la_wall", {far: 0.0})
                ra = self._atrial_solve("ra_wall", {src["ra_wall"]["av_node"]: 40.0, src["ra_wall"]["sa_node"]: 45.0})
            elif pattern == AtrialPattern.FLUTTER:
                # Kreisen um den Trikuspidalring: Winkel um die RA-Achse -> Zeit
                ra_v = self.model.parts["ra_wall"].vertices.astype(np.float64)
                _, angle, _ = self.model.ra_frame.coordinates(ra_v)
                ra = np.mod(angle + 180.0, 360.0) / 360.0 * 200.0
                la = self._atrial_solve("la_wall", {src["la_wall"]["bachmann"]: 60.0})
                la = np.mod(la, 200.0)
            else:
                ra = np.zeros(self.model.parts["ra_wall"].n_vertices)
                la = np.zeros(self.model.parts["la_wall"].n_vertices)
            cached = np.concatenate([ra, la])
            self._cache[key] = cached
        n_ra = self.model.parts["ra_wall"].n_vertices
        ra, la = cached[:n_ra].copy(), cached[n_ra:].copy()
        if pattern in (AtrialPattern.SINUS, AtrialPattern.RETROGRADE, AtrialPattern.ECTOPIC_LEFT):
            both = np.concatenate([ra, la])
            span = max(float(np.percentile(both, 97.0)), 1.0)
            ra = ra / span * p_duration_ms
            la = la / span * p_duration_ms
        labels = {
            AtrialPattern.SINUS: "Sinusknoten → rechter Vorhof → Bachmann-Bündel → linker Vorhof",
            AtrialPattern.RETROGRADE: "Retrograde Vorhoferregung vom AV-Knoten",
            AtrialPattern.ECTOPIC_LEFT: "Ektop-atrialer Ursprung links",
            AtrialPattern.FLUTTER: "Vorhofflattern: Kreiserregung um den Trikuspidalring",
            AtrialPattern.FIBRILLATION: "Vorhofflimmern: keine geordnete Erregung",
            AtrialPattern.NONE: "Keine Vorhoferregung dargestellt",
        }
        return AtrialResult(ra.astype(np.float32), la.astype(np.float32), pattern, labels[pattern])


_GRIDS: dict[int, WallGrid] = {}


def wall_grid_for(model: HeartModel) -> WallGrid:
    grid = _GRIDS.get(id(model))
    if grid is None:
        grid = build_wall_grid(model)
        _GRIDS[id(model)] = grid
    return grid


# ------------------------------------------------------------------ Auswahl
def vcg_to_model(vector: np.ndarray) -> np.ndarray:
    """VCG (X links, Y inferior, Z posterior) -> Modell (x links, y posterior, z kranial)."""
    x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
    return np.array([x, z, -y], dtype=np.float64)


def octant_to_direction(octant: int) -> np.ndarray:
    """Oktant-Bits (x>0, y>0, z>0) des VCG-Flächenvektors -> grobe Richtung im VCG-Raum."""
    bits = int(octant)
    sx = 1.0 if bits & 1 else -1.0
    sy = 1.0 if bits & 2 else -1.0
    sz = 1.0 if bits & 4 else -1.0
    return np.array([sx, sy, sz]) / np.sqrt(3.0)


def choose_pattern(hints: dict, beat_type: str, override: Pattern = Pattern.AUTO) -> tuple[Pattern, str, str]:
    """Liefert (Muster, Vertrauensstufe, Begründung)."""
    if override != Pattern.AUTO:
        return override, "K", "Manuell gewählt"
    if beat_type == "V":
        return Pattern.ECTOPIC, "K", "Sekundäre QRS-Morphologie / vorzeitiger breiter Schlag"
    if hints.get("paced"):
        return Pattern.RV_PACING, "K", "Stimulationsartefakt erkannt (Marker pacing_spike_slope_ratio)"
    if hints.get("wpw"):
        return Pattern.WPW, "K", "Delta-Welle erkannt (delta_wave_present_flag)"
    if hints.get("lbbb"):
        return Pattern.LBBB, "K", "Klassisches Kriterium: Linksschenkelblock-Muster"
    if hints.get("rbbb"):
        if hints.get("left_axis"):
            return Pattern.RBBB, "K", "Rechtsschenkelblock-Muster mit Linksachse (bifaszikulär möglich)"
        return Pattern.RBBB, "K", "Klassisches Kriterium: Rechtsschenkelblock-Muster"
    if hints.get("left_axis") and hints.get("lbbb") is False and hints.get("rbbb") is False:
        return Pattern.LAFB, "K", "Linksachse ohne Schenkelblock: Faszikelblock als Muster angenommen"
    return Pattern.NORMAL, "P", "Kein Leitungsmuster erkannt: normale Sequenz nach Durrer"


def choose_atrial_pattern(timeline_rhythm: str, flags: tuple[str, ...], beat_has_p: bool, beat_type: str,
                          p_axis_left: bool = False, atrial_frequency_hz: Optional[float] = None,
                          ventricular_hz: Optional[float] = None) -> AtrialPattern:
    if timeline_rhythm == "af_compatible_pattern":
        return AtrialPattern.FIBRILLATION
    if timeline_rhythm == "flutter_compatible_pattern":
        return AtrialPattern.FLUTTER
    if "no_consistent_discrete_p" in flags and atrial_frequency_hz is not None and 3.5 <= atrial_frequency_hz <= 6.5 and not beat_has_p:
        # regelmäßige Vorhofaktivität 210–390/min ohne diskrete P; nur wenn die Kammerfrequenz ein ganzzahliger
        # Teiler der Vorhoffrequenz ist (2:1 … 4:1), sonst ist die Spektralspitze eher ein Artefakt
        if ventricular_hz:
            ratio = atrial_frequency_hz / ventricular_hz
            if abs(ratio - round(ratio)) <= 0.08 * round(ratio) and 2 <= round(ratio) <= 3:
                return AtrialPattern.FLUTTER
        else:
            return AtrialPattern.FLUTTER
    if "no_consistent_discrete_p" in flags and timeline_rhythm == "irregular" and not beat_has_p:
        return AtrialPattern.FIBRILLATION
    if beat_type == "V" and not beat_has_p:
        return AtrialPattern.RETROGRADE
    if not beat_has_p:
        return AtrialPattern.NONE
    if p_axis_left:
        return AtrialPattern.ECTOPIC_LEFT
    return AtrialPattern.SINUS


def accessory_vertex_for(model: HeartModel, delta_axis_deg: Optional[float]) -> int:
    """Grobe Bahnlokalisation: links-lateral (Standard) oder rechts-seitig bei Delta-Achse < 0°."""
    if delta_axis_deg is not None and np.isfinite(delta_axis_deg) and delta_axis_deg < 0:
        idx = model.purkinje_sites.get("rv_free_wall_base")
    else:
        idx = model.purkinje_sites.get("lv_lateral_base")
    return int(idx[0]) if idx is not None and len(idx) else 0


def ectopic_origin_vertex(model: HeartModel, beat_vector: Optional[np.ndarray], octant: Optional[float]) -> tuple[int, str]:
    """Ursprung eines ektopen Schlags aus dem mittleren QRS-Vektor (Kors) oder dem Oktanten.

    Der mittlere Vektor zeigt vom Fokus weg; der Fokus liegt an der Wand in Gegenrichtung.
    Genauigkeit in der Literatur 70–74 % je Region (Nachbarregion 91–93 %).
    """
    direction = None
    if beat_vector is not None and np.all(np.isfinite(beat_vector)) and np.linalg.norm(beat_vector) > 1e-4:
        direction = -vcg_to_model(beat_vector)
        label = "aus dem QRS-Vektor des Schlags (Kors)"
    elif octant is not None and np.isfinite(octant):
        direction = -vcg_to_model(octant_to_direction(int(octant)))
        label = "aus dem Oktanten des QRS-Flächenvektors"
    if direction is None:
        return int(model.purkinje_sites["lv_lateral_base"][0]), "Standardort (kein Vektor verfügbar)"
    return model.wall_vertex_in_direction(direction), label
