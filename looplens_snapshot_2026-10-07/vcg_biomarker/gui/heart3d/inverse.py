"""Inverse Simulation: Erregungshypothesen durchspielen und an den gemessenen Ableitungen messen.

Vorgehen (nach der Literatur zur 12-Kanal-ECGI mit Aktivierungsmodellen):
* Vorwärtsmodell: Aktivierungszeiten je Hypothese aus dem Kürzeste-Wege-Modell
  (Fastest-Route-Algorithmus, van Dam 2009; Potyagaylo 2016), Quellmodell die
  uniforme Doppelschicht (Scher 1994; van Oosterom, ECGSIM 2004): der Summendipol
  zu jeder Zeit ist das Flächenintegral der Wellenfront mit ihrer Normalen, hier
  diskret als Summe über Vertices (lokale Fläche × Ausbreitungsrichtung aus dem
  Gradienten der Aktivierungszeit, Aufstrichkern 3 ms).
* Ableitungen aus dem Dipol über das an V1–V6 kalibrierte Leadfield des Thorax-
  modells (Einzeldipol gegen Wilson-Terminal); zusätzlich Vergleich im VCG-Raum
  (Kors) als leadfield-unabhängige Kontrolle (FRA-V, Potyagaylo 2019).
* Vergleich mit der gemessenen QRS-Morphologie (Pearson je Ableitung nach globalem
  Maßstab und ±8 ms Verschiebung; Pezzuto 2021 r ≈ 0,92 mit patientenspezifischer
  Anatomie), Rangfolge der Hypothesen. Die QRS-Dauer wird für jede Hypothese auf
  den Messwert skaliert, verglichen wird die Form.
* Hypothesenfamilien: Leitungssystem (normal, kompletter Schenkelblock, verzögerte
  Schenkel 20/40/60 ms – Upadhyay 2019 fand bei LSB-Muster in 36 % erhaltene
  Purkinje-Aktivierung –, Faszikelblöcke, bifaszikulär, unspezifische Verzögerung),
  ektope Fokusorte auf einem AHA-Raster mit Verfeinerung, akzessorische Bahnen am
  AV-Ring, Stimulationsorte.

Grenzen (Stufe H): generische Geometrie, vereinfachter Volumenleiter und keine
patientenspezifische Faseranisotropie. Publizierte Millimeterfehler mit bekannter
Geometrie und zusätzlichen Referenzen gelten NICHT für diesen Viewer. Benachbarte
Regionen (z. B. RV-Freiwand ↔ Septum) können verwechselt werden. Auch eine hohe
Passung ist ohne unabhängiges elektrophysiologisches Mapping kein Ortsnachweis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .activation import ActivationResult, ActivationSolver, Pattern
from .model import CHAMBER_LV, CHAMBER_RV, HeartModel
from .pathology import SEGMENT_REGION
from .timeline import DOWER, KORS, Beat, BeatTimeline

UPSTROKE_SIGMA_MS = 3.0
MAX_SHIFT_MS = 12.0
T_WAVE_WEIGHT = 0.2  # Anteil der T-Wellen-Passung an der Gesamtpassung (sekundäre Repolarisation folgt der Erregung)
# Drehung des Modellherzens in die Thoraxlage (Modellkoordinaten -> Modellkoordinaten), kalibriert an
# normalen PTB-XL-Aufzeichnungen (Rangfolge-Optimum der normalen Erregung); Einheitsmatrix = unkalibriert.
# Kalibriert am 2026-09-22 (gemeinsam mit den Leitungsparametern) an 104 Aufzeichnungen (PTB-XL normal/LSB/RSB/LAFB/LPFB,
# MIMIC-IV-ECG ventrikulär stimuliert; Trainingshälfte), Zielgröße ausgeglichene Trefferquote der Hypothesenfamilie:
# Euler z/x/y = +15°/+15°/0° (Dower-Leadfield). Testhälfte s. INVERSE_SIMULATION_RECHERCHE.
HEART_ROTATION = np.array([[ 0.965926, -0.258819,  0.      ],
 [ 0.25    ,  0.933013, -0.258819],
 [ 0.066987,  0.25    ,  0.965926]])
HEART_ROTATION_EULER_DEG = (15.0, 15.0, 0.0)
AXIS_ROTATION_SIGN = 1.0  # Vorzeichen so, dass ein positiver Winkel die frontale QRS-Achse nach inferior dreht (geprüft in den Tests)


# ------------------------------------------------------------------ Hypothesen
@dataclass(frozen=True)
class Hypothesis:
    key: str
    label: str
    family: str  # conduction | focus | accessory | pacing
    beat_type: str = "N"  # N (Leitungssystem) oder V/P (Fokus, Stimulation)
    lv_delay_ms: float = 0.0  # Verzögerung des linken Schenkels; inf = kompletter Block
    rv_delay_ms: float = 0.0
    lafb: bool = False
    lpfb: bool = False
    ivcd: bool = False  # unspezifische intraventrikuläre Verzögerung: kein Purkinje-Netz, gleichmäßig langsam
    origin_vertex: Optional[int] = None
    accessory_vertex: Optional[int] = None
    breakthrough_sites: int = 4
    line_of_block: str = ""  # "" | anterior | lateral | inferior: funktionelle transmurale Blocklinie im LV (Auricchio 2004)
    csp: str = ""  # Leitungssystem-Stimulation: "" | his | lbbap | biv (Strocchi 2023, Upadhyay 2019)
    reentry_centre: Optional[int] = None  # Kreiserregung: Wandvertex im Zentrum des Kreises (Narbe/Isthmus), Austritt = origin_vertex
    extra_delays: tuple = ()  # freie Verzögerungen ((Segment, ms), ...) aus dem Was-wäre-wenn-Editor

    @property
    def blocked(self) -> tuple[str, ...]:
        out = [seg for seg, delay in self.extra_delays if np.isinf(delay)]
        if np.isinf(self.lv_delay_ms):
            out += ["lbb", "lbb_anterior", "lbb_posterior"]
        else:
            if self.lafb:
                out.append("lbb_anterior")
            if self.lpfb:
                out.append("lbb_posterior")
        if np.isinf(self.rv_delay_ms):
            out.append("rbb")
        return tuple(out)

    @property
    def pattern(self) -> Pattern:
        if self.family == "focus" or self.family == "reentry":
            return Pattern.ECTOPIC
        if self.family == "pacing":
            return Pattern.NORMAL if self.csp == "his" else Pattern.RV_PACING
        if self.family == "accessory":
            return Pattern.WPW
        if np.isinf(self.lv_delay_ms):
            return Pattern.LBBB
        if np.isinf(self.rv_delay_ms):
            return Pattern.RBBB
        if self.lafb:
            return Pattern.LAFB
        if self.lpfb:
            return Pattern.LPFB
        return Pattern.NORMAL

    @property
    def notes(self) -> tuple[str, ...]:
        out = []
        if 0 < self.lv_delay_ms < np.inf:
            out.append(f"linker Schenkel um {self.lv_delay_ms:.0f} ms verzögert (erhaltene Purkinje-Aktivierung)")
        if 0 < self.rv_delay_ms < np.inf:
            out.append(f"rechter Schenkel um {self.rv_delay_ms:.0f} ms verzögert")
        if self.ivcd:
            out.append("unspezifische intraventrikuläre Leitungsverzögerung (ohne Purkinje-Netz)")
        if self.line_of_block:
            out.append(f"U-förmige LV-Erregung um eine funktionelle Blocklinie ({self.line_of_block}; Auricchio 2004)")
        if self.csp == "his":
            out.append("His-Bündel-Stimulation: Erregung über das eigene Leitungssystem (Korrektur nur bei komplettem proximalem Block, Upadhyay 2019)")
        elif self.csp == "lbbap":
            out.append("Linksschenkel-Areal-Stimulation: LV über Purkinje, RV verzögert über das Septum (Strocchi 2023)")
        elif self.csp == "biv":
            out.append("Biventrikuläre Stimulation (RV-Apex + LV lateral, Koronarsinus)")
        if self.reentry_centre is not None:
            out.append("Kreiserregung um Narbe/Isthmus: langsame Zone im Zentrum, Austritt am Fokus (scar-related reentry)")
        for seg_name, delay in self.extra_delays:
            out.append(f"Editor: {seg_name} {'blockiert' if np.isinf(delay) else f'um {delay:.0f} ms verzögert'}")
        return tuple(out)


@dataclass
class HypothesisResult:
    hypothesis: Hypothesis
    r_leads: dict[str, float]
    r_mean: float
    r_vcg: float
    rmse_mv: float
    shift_ms: float
    scale: float
    simulated_leads: np.ndarray  # (n, 12) mV, auf das Messfenster ausgerichtet
    activation: ActivationResult
    r_t_wave: Optional[float] = None  # Passung der simulierten T-Welle (sekundäre Repolarisation), falls T-Fenster vorhanden
    simulated_t_leads: Optional[np.ndarray] = None
    t_window: Optional[tuple] = None
    time_scale_factor: float = 1.0  # per-hypothesis QRS-duration adjustment, not anatomy
    time_scale_limited: bool = False

    @property
    def score(self) -> float:
        base = 0.7 * self.r_mean + 0.3 * self.r_vcg
        if self.r_t_wave is None:
            return base
        return (1.0 - T_WAVE_WEIGHT) * base + T_WAVE_WEIGHT * self.r_t_wave


# ------------------------------------------------------------------ Vorwärtsmodell
class ForwardModel:
    """Aktivierungszeiten (Gitterknoten) → Summendipol → Ableitungen."""

    def __init__(self, model: HeartModel, solver: ActivationSolver):
        self.model = model
        self.solver = solver
        g = solver.grid
        self.grid = g
        self.vertex_volume = g.volume.astype(np.float64)
        self.rv_volume_factor = 1.5  # kalibriert (RV-Freiwand inkl. Trabekel wiegt mehr als 4 mm glatte Wand)
        v = g.positions.astype(np.float64)
        self._ei = np.concatenate([g.edges_i, g.edges_j])
        self._ej = np.concatenate([g.edges_j, g.edges_i])
        d = v[self._ej] - v[self._ei]
        self._d = d
        A = np.zeros((len(v), 3, 3))
        np.add.at(A, self._ei, d[:, :, None] * d[:, None, :])
        self._A_inv = np.linalg.pinv(A + 1e-6 * np.eye(3)[None])
        # Herzlage-Kalibrierung: Drehung des elektrischen Bezugssystems (Modell -> Thorax), s. HEART_ROTATION
        self.rotation = np.asarray(HEART_ROTATION, dtype=np.float64)

    def activation_gradient(self, times_ms: np.ndarray, max_slowness_ms_per_mm: float = 1.0 / 0.3) -> np.ndarray:
        """∇τ je Gitterknoten (ms/mm) aus kleinsten Quadraten über alle Nachbarn (Schicht + transmural)."""
        dt = times_ms[self._ej] - times_ms[self._ei]
        b = np.zeros((len(times_ms), 3))
        np.add.at(b, self._ei, self._d * dt[:, None])
        g = np.einsum("nij,nj->ni", self._A_inv, b)
        n = np.linalg.norm(g, axis=1)
        too_slow = n > max_slowness_ms_per_mm
        g[too_slow] *= (max_slowness_ms_per_mm / n[too_slow])[:, None]
        return g

    def propagation_direction(self, times_ms: np.ndarray) -> np.ndarray:
        g = self.activation_gradient(times_ms)
        n = np.linalg.norm(g, axis=1)
        g[n > 1e-9] /= n[n > 1e-9, None]
        g[n <= 1e-9] = 0.0
        return g

    def dipole(self, times_ms: np.ndarray, t_grid_ms: np.ndarray) -> np.ndarray:
        """(len(t_grid), 3) Summendipol in Modellkoordinaten (willkürliche Einheit).

        Uniforme Doppelschicht: das über die Zeit integrierte Dipolmoment eines Volumenelements ist
        Volumen × ∇τ (die Front verweilt umso länger, je langsamer sie läuft). Schnelle Purkinje-
        getriebene Endokardfronten wiegen daher wenig, langsame transmurale Myokardfronten viel.
        """
        vol = self.vertex_volume
        if self.rv_volume_factor != 1.0:
            vol = np.where((self.grid.chamber == CHAMBER_RV) & ~self.grid.septal, vol * self.rv_volume_factor, vol)
        w = vol[:, None] * self.activation_gradient(times_ms)
        # Faltung mit dem Aufstrichkern: Beiträge in Zeitklassen sammeln, dann glätten
        step = float(t_grid_ms[1] - t_grid_ms[0]) if len(t_grid_ms) > 1 else 1.0
        idx = np.clip(np.round((times_ms - t_grid_ms[0]) / step).astype(int), 0, len(t_grid_ms) - 1)
        hist = np.zeros((len(t_grid_ms), 3))
        np.add.at(hist, idx, w)
        k = int(np.ceil(3 * UPSTROKE_SIGMA_MS / step))
        kernel = np.exp(-0.5 * (np.arange(-k, k + 1) * step / UPSTROKE_SIGMA_MS) ** 2)
        kernel /= kernel.sum()
        d = np.column_stack([np.convolve(hist[:, j], kernel, mode="same") for j in range(3)])
        return d @ self.rotation.T

    def repolarization_dipole(self, node_times_ms: np.ndarray, t_grid_ms: np.ndarray, apd_ms: float,
                              transmural_ms: float = 25.0, apex_base_ms: float = 12.0, sigma_ms: float = 22.0) -> np.ndarray:
        """T-Wellen-Dipol: Repolarisationszeit = Aktivierung + APD mit transmuralem (Epikard kürzer) und
        apikobasalem (Apex kürzer) Gradienten; Front läuft rückwärts zum Potentialsprung, daher −∇τ_r,
        breiter Kern (van Oosterom 2001; Qian 2025: APD-Gradienten als Modellparameter)."""
        g = self.grid
        layer_frac = g.layer.astype(np.float64) / np.maximum(g.n_layers.astype(np.float64), 1.0)
        apd = apd_ms - transmural_ms * layer_frac - apex_base_ms * (1.0 - g.u.astype(np.float64))
        t_rep = node_times_ms.astype(np.float64) + apd
        w = self.vertex_volume[:, None] * self.activation_gradient(t_rep, max_slowness_ms_per_mm=4.0)
        step = float(t_grid_ms[1] - t_grid_ms[0]) if len(t_grid_ms) > 1 else 1.0
        idx = np.clip(np.round((t_rep - t_grid_ms[0]) / step).astype(int), 0, len(t_grid_ms) - 1)
        hist = np.zeros((len(t_grid_ms), 3))
        np.add.at(hist, idx, -w)
        k = int(np.ceil(3 * sigma_ms / step))
        kernel = np.exp(-0.5 * (np.arange(-k, k + 1) * step / sigma_ms) ** 2)
        kernel /= kernel.sum()
        d = np.column_stack([np.convolve(hist[:, j], kernel, mode="same") for j in range(3)])
        return d @ self.rotation.T

    def activation_for(self, h: Hypothesis, qrs_ms: float) -> ActivationResult:
        return activation_from_hypothesis(self.solver, h, qrs_ms, exact_scale=False)


def activation_from_hypothesis(solver: ActivationSolver, h: Hypothesis, qrs_ms: float, exact_scale: bool = True,
                               scale_limits: tuple[float, float] = (0.8, 1.35)) -> ActivationResult:
    """Aktivierungskarte einer Hypothese (Erweiterung der Musterbibliothek um Verzögerungen).

    exact_scale=True: Dauer auf die gemessene QRS-Dauer skaliert (Anzeige). False: physiologische Zeiten, nur eine
    begrenzte globale Anpassung (Patientenfaktor) – für den Vergleich, damit z. B. bei Schenkelblock die gesunde
    Kammer schnell bleibt und nur die blockierte Kammer die Verlängerung trägt.
    """
    grid = solver.grid
    sites = grid.purkinje_sites
    sources: dict[int, float] = {}
    rv_off = solver.params.get("rv_entry_offset_ms", 0.0)
    origin_node = solver.node_of(h.origin_vertex)
    accessory_node = solver.node_of(h.accessory_vertex)

    def add(name: str, t0: float) -> None:
        for idx in sites.get(name, ()):
            sources[int(idx)] = min(t0, sources.get(int(idx), np.inf))

    lv_order_all = ["lv_mid_septal", "lv_anterior_paraseptal", "lv_posterior_paraseptal", "lv_apical"]
    if h.family == "pacing" and h.csp == "his":
        # His-Stimulation: wie normale Erregung; ein bestehender Block (lv/rv_delay) bleibt, wenn er distal liegt
        velocity = solver._velocity_field(lv_purkinje=not np.isinf(h.lv_delay_ms), rv_purkinje=not np.isinf(h.rv_delay_ms))
        if not np.isinf(h.lv_delay_ms):
            for k, name in enumerate(lv_order_all):
                add(name, h.lv_delay_ms + 2.0 * k)
        if not np.isinf(h.rv_delay_ms):
            add("rv_anterior_papillary", h.rv_delay_ms + 6.0 + rv_off)
            add("rv_apical", h.rv_delay_ms + 12.0 + rv_off)
        if not sources:
            sources[int(sites["lv_mid_septal"][0])] = 0.0
    elif h.family == "pacing" and h.csp == "lbbap":
        # Linksschenkel-Areal-Stimulation: LV-Purkinje ab dem linken Schenkel, RV über das Septum (Myokard)
        velocity = solver._velocity_field(lv_purkinje=True, rv_purkinje=True)
        for k, name in enumerate(lv_order_all):
            add(name, 2.0 * k)
        # RV erst über das Septum und retrograd in den rechten Schenkel (klinisch QRS ≈ 110–130 ms)
        add("rv_anterior_papillary", 35.0)
        add("rv_apical", 40.0)
    elif h.family == "pacing" and h.csp == "biv":
        velocity = solver._velocity_field(False, False, all_muscle=True)
        velocity[grid.endo] = 1.1
        add("rv_apical", 0.0)
        add("lv_lateral_base", 0.0)
    elif h.family == "reentry" and origin_node is not None:
        velocity = solver._velocity_field(False, False, all_muscle=True)
        velocity[grid.endo] = 1.1
        if h.reentry_centre is not None:
            centre = solver.model.wall.vertices[int(h.reentry_centre)]
            dist = np.linalg.norm(grid.positions - centre, axis=1)
            velocity[dist < 18.0] = 0.15  # Isthmus/Narbenzone: sehr langsam
        sources[origin_node] = 0.0
    elif h.family in ("focus", "pacing") and origin_node is not None:
        velocity = solver._velocity_field(False, False, all_muscle=True)
        velocity[grid.endo] = 1.1
        sources[origin_node] = 0.0
    else:
        if h.ivcd:
            velocity = solver._velocity_field(False, False, all_muscle=True)
            velocity[:] = solver.params["v_muscle"] * 0.8
        else:
            velocity = solver._velocity_field(lv_purkinje=not np.isinf(h.lv_delay_ms), rv_purkinje=not np.isinf(h.rv_delay_ms),
                                              lv_anterior_block=h.lafb, lv_posterior_block=h.lpfb)
        lv_order = ["lv_mid_septal", "lv_anterior_paraseptal", "lv_posterior_paraseptal", "lv_apical"]
        if h.lafb:
            lv_order.remove("lv_anterior_paraseptal")
        if h.lpfb:
            lv_order.remove("lv_posterior_paraseptal")
        if not np.isinf(h.lv_delay_ms):
            for k, name in enumerate(lv_order[:max(1, h.breakthrough_sites)]):
                add(name, h.lv_delay_ms + 2.0 * k)
        if not np.isinf(h.rv_delay_ms):
            add("rv_anterior_papillary", h.rv_delay_ms + 6.0 + rv_off)
            add("rv_apical", h.rv_delay_ms + 12.0 + rv_off)
        if h.family == "accessory" and accessory_node is not None:
            sources[accessory_node] = -30.0
        for seg_name, delay in h.extra_delays:
            # Editor: Faszikel/Schenkel verzögern oder blockieren (Quellen des Segments verschieben/entfernen)
            names = {"lbb": lv_order_all, "lbb_anterior": ["lv_anterior_paraseptal"], "lbb_posterior": ["lv_posterior_paraseptal"],
                     "rbb": ["rv_anterior_papillary", "rv_apical"]}.get(seg_name, [])
            for name in names:
                for idx in sites.get(name, ()):
                    if int(idx) in sources:
                        sources[int(idx)] = (sources[int(idx)] + float(delay)) if np.isfinite(delay) else np.inf
            sources = {k: v for k, v in sources.items() if np.isfinite(v)}
            if seg_name == "lbb" and np.isinf(delay):
                velocity[(grid.chamber == CHAMBER_LV) & grid.endo & grid.septal] = solver.params["v_septal_slow"]
            if seg_name == "rbb" and np.isinf(delay):
                velocity[(grid.chamber == CHAMBER_RV) & grid.endo & grid.septal] = solver.params["v_septal_slow"]
        if h.line_of_block:
            centre_angle = {"anterior": 60.0, "lateral": 180.0, "inferior": -60.0}[h.line_of_block]
            dang = np.abs(((grid.angle - centre_angle + 180.0) % 360.0) - 180.0)
            line = (grid.chamber == CHAMBER_LV) & (dang < 12.0) & (grid.u > 0.2) & (grid.u < 0.92)
            velocity[line] = 0.12  # funktionelle Blocklinie: nahezu leitungsunfähig
        if not sources:  # beidseitiger Block: Ersatzrhythmus aus dem Septum
            sources[int(sites["lv_mid_septal"][0])] = 0.0
    raw = solver._solve(sources, velocity)
    times = raw - raw.min()
    span = float(np.percentile(times[grid.node_of_vertex], 97.0)) or 1.0
    factor = float(qrs_ms) / span
    if not exact_scale:
        factor = float(np.clip(factor, scale_limits[0], scale_limits[1]))
    times = times * factor
    origin = solver.vertices_of_nodes(np.fromiter(sources.keys(), dtype=np.int64, count=len(sources)))
    if h.family in ("focus", "pacing", "reentry") and h.origin_vertex is not None:
        focal = np.array([int(h.origin_vertex)], np.int64)
    elif h.family == "accessory" and h.accessory_vertex is not None:
        focal = np.array([int(h.accessory_vertex)], np.int64)
    else:
        focal = np.zeros(0, np.int64)
    return ActivationResult(times[grid.node_of_vertex].astype(np.float32), h.pattern, "H", f"Hypothese: {h.label}", origin, "",
                            h.notes, focal, node_times_ms=times.astype(np.float32), blocked=h.blocked)


# ------------------------------------------------------------------ Hypothesenraum
def _site_vertex(model: HeartModel, segment: int, epicardial: bool) -> Optional[int]:
    mask = (model.wall_segment == segment) & (model.wall_endocardial if not epicardial else ~model.wall_near_cavity)
    idx = np.where(mask)[0]
    if idx.size == 0:
        mask = model.wall_segment == segment
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
    centre = model.wall.vertices[idx].mean(0)
    return int(idx[np.argmin(np.linalg.norm(model.wall.vertices[idx] - centre, axis=1))])


def conduction_hypotheses() -> list[Hypothesis]:
    inf = float("inf")
    out = [
        Hypothesis("normal", "Normale Erregung (Durrer)", "conduction"),
        Hypothesis("lbbb", "Kompletter Linksschenkelblock (proximal / links-intrahisär)", "conduction", lv_delay_ms=inf),
        Hypothesis("lbb_d20", "Linker Schenkel 20 ms verzögert (inkomplett)", "conduction", lv_delay_ms=20),
        Hypothesis("lbb_d40", "Linker Schenkel 40 ms verzögert", "conduction", lv_delay_ms=40),
        Hypothesis("lbb_d60", "Linker Schenkel 60 ms verzögert (LSB-Muster mit erhaltenem Purkinje)", "conduction", lv_delay_ms=60),
        Hypothesis("lafb", "Linksanteriorer Faszikelblock", "conduction", lafb=True),
        Hypothesis("lpfb", "Linksposteriorer Faszikelblock", "conduction", lpfb=True),
        Hypothesis("rbbb", "Kompletter Rechtsschenkelblock (proximal)", "conduction", rv_delay_ms=inf),
        Hypothesis("rbb_d20", "Rechter Schenkel 20 ms verzögert (inkomplett)", "conduction", rv_delay_ms=20),
        Hypothesis("rbb_d40", "Rechter Schenkel 40 ms verzögert", "conduction", rv_delay_ms=40),
        Hypothesis("rbb_d60", "Rechter Schenkel 60 ms verzögert (distal)", "conduction", rv_delay_ms=60),
        Hypothesis("rbbb_lafb", "RSB + LAFB (bifaszikulär)", "conduction", rv_delay_ms=inf, lafb=True),
        Hypothesis("rbbb_lpfb", "RSB + LPFB (bifaszikulär)", "conduction", rv_delay_ms=inf, lpfb=True),
        Hypothesis("lbbb_rvd", "LSB mit zusätzlich verzögertem rechten Schenkel", "conduction", lv_delay_ms=inf, rv_delay_ms=30),
        Hypothesis("ivcd", "Unspezifische intraventrikuläre Verzögerung", "conduction", ivcd=True),
        Hypothesis("lbbb_line_ant", "LSB mit anteriorer Blocklinie (U-förmige LV-Erregung)", "conduction", lv_delay_ms=inf, line_of_block="anterior"),
        Hypothesis("lbbb_line_lat", "LSB mit lateraler Blocklinie", "conduction", lv_delay_ms=inf, line_of_block="lateral"),
        Hypothesis("lbbb_line_inf", "LSB mit inferiorer Blocklinie", "conduction", lv_delay_ms=inf, line_of_block="inferior"),
        Hypothesis("normal_b2", "Normal mit 2 Durchbruchsorten (These frühe Aufhebung)", "conduction", breakthrough_sites=2),
    ]
    return out


def focus_hypotheses(model: HeartModel, beat_type: str = "V") -> list[Hypothesis]:
    out = []
    for seg in list(range(1, 18)) + [101, 102, 103, 104, 105]:
        for epi in (False, True):
            v = _site_vertex(model, seg, epi)
            if v is None:
                continue
            name = SEGMENT_REGION.get(seg, str(seg))
            out.append(Hypothesis(f"focus_{seg}_{'epi' if epi else 'endo'}",
                                  f"Fokus {name} ({'epikardial' if epi else 'endokardial'})", "focus", beat_type, origin_vertex=v))
    return out


def refine_focus(model: HeartModel, best: Hypothesis, radius_mm: float = 18.0, count: int = 6) -> list[Hypothesis]:
    if best.origin_vertex is None:
        return []
    v = model.wall.vertices
    d = np.linalg.norm(v - v[best.origin_vertex], axis=1)
    idx = np.where((d < radius_mm) & (d > 3.0))[0]
    if idx.size == 0:
        return []
    rng = np.random.default_rng(int(best.origin_vertex))
    pick = rng.choice(idx, size=min(count, idx.size), replace=False)
    return [Hypothesis(f"focus_ref_{int(k)}", f"Fokus verfeinert nahe {best.label.split(' (')[0].replace('Fokus ', '')} (+{d[k]:.0f} mm)",
                       "focus", best.beat_type, origin_vertex=int(k)) for k in pick]


def accessory_hypotheses(model: HeartModel) -> list[Hypothesis]:
    out = []
    for chamber, frame, tag in ((CHAMBER_LV, model.lv_frame, "mitral"), (CHAMBER_RV, model.rv_frame, "trikuspidal")):
        base = (model.wall_chamber == chamber) & (model.wall_u > 0.8)
        for ang in range(-150, 181, 60):
            sel = base & (np.abs(((model.wall_angle - ang + 180) % 360) - 180) < 30)
            idx = np.where(sel)[0]
            if idx.size == 0:
                continue
            centre = model.wall.vertices[idx].mean(0)
            v = int(idx[np.argmin(np.linalg.norm(model.wall.vertices[idx] - centre, axis=1))])
            out.append(Hypothesis(f"acc_{tag}_{ang}", f"Akzessorische Bahn {tag} {ang:+d}°", "accessory", "N", accessory_vertex=v))
    return out


def pacing_hypotheses(model: HeartModel) -> list[Hypothesis]:
    sites = model.purkinje_sites
    out = []
    for name, label in (("rv_apical", "RV-Apex"), ("rv_outflow", "RV-Ausflusstrakt"), ("lv_mid_septal", "septal (His-/LSB-Bereich)"),
                        ("lv_lateral_base", "LV lateral (Koronarsinus)"), ("rv_anterior_papillary", "RV-Septum mittig")):
        idx = sites.get(name)
        if idx is not None and len(idx):
            out.append(Hypothesis(f"pace_{name}", f"Stimulation {label}", "pacing", "P", origin_vertex=int(idx[0])))
    return out


def csp_hypotheses(base: Optional[Hypothesis] = None) -> list[Hypothesis]:
    """Leitungssystem-Stimulation zur Vorhersage der QRS-Korrektur (His, Linksschenkel-Areal, biventrikulär)."""
    lv = base.lv_delay_ms if base is not None else 0.0
    rv = base.rv_delay_ms if base is not None else 0.0
    # Bei komplettem proximalem Block korrigiert His-Stimulation (Rekrutierung distaler Purkinje-Fasern, Upadhyay 2019: 85 %);
    # bei erhaltener, verzögerter Purkinje-Aktivierung bleibt die Verzögerung bestehen.
    his_lv = 0.0 if np.isinf(lv) else lv
    his_rv = 0.0 if np.isinf(rv) else rv
    return [
        Hypothesis("csp_his", "His-Bündel-Stimulation", "pacing", "P", lv_delay_ms=his_lv, rv_delay_ms=his_rv, csp="his"),
        Hypothesis("csp_lbbap", "Linksschenkel-Areal-Stimulation (LBBAP)", "pacing", "P", csp="lbbap"),
        Hypothesis("csp_biv", "Biventrikuläre Stimulation (CRT)", "pacing", "P", csp="biv"),
    ]


def reentry_hypothesis(model: HeartModel, exit_vertex: int, centre_vertex: Optional[int] = None) -> Hypothesis:
    if centre_vertex is None:
        centre_vertex = exit_vertex
    return Hypothesis(f"reentry_{int(exit_vertex)}", "Kreiserregung um Narbe (Austritt am Fokus)", "reentry", "V",
                      origin_vertex=int(exit_vertex), reentry_centre=int(centre_vertex))


def ischemia_forward_check(model: HeartModel, solver: ActivationSolver, rotation: np.ndarray, ischemia: np.ndarray,
                           kind: str, measured_st60: dict) -> dict:
    """Verletzungsstrom der geschätzten Zone vorwärts rechnen und mit den gemessenen ST-Abweichungen vergleichen.

    Uniforme Doppelschicht während der ST-Strecke: die Zone hat ein verändertes Plateau; der Dipol zeigt bei
    transmuraler Ischämie nach außen (zur Läsion), bei subendokardialer zum Cavum. Rückgabe: ST-Vektor (VCG),
    Winkel zum gemessenen ST-Vektor, Korrelation über die 12 Ableitungen (Maßstab frei).
    """
    from scipy.spatial import cKDTree
    grid = solver.grid
    w_nodes = np.asarray(ischemia, dtype=np.float64)[cKDTree(model.wall.vertices).query(grid.positions)[1]]
    sign = 1.0 if kind == "transmural" else -1.0
    d = sign * (grid.volume[:, None] * w_nodes[:, None] * grid.normal_out).sum(0)
    d = np.asarray(rotation) @ d
    vcg = np.array([d[0], -d[2], d[1]])
    n = np.linalg.norm(vcg)
    if n < 1e-9:
        return {}
    vcg_u = vcg / n
    eight = DOWER @ vcg_u
    lead_i, lead_ii = eight[0], eight[1]
    sim = np.array([lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2, lead_i - lead_ii / 2, lead_ii - lead_i / 2, *eight[2:]])
    names = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    meas = np.array([float(measured_st60.get(k, 0.0)) for k in names])
    r = float(np.corrcoef(sim, meas)[0, 1]) if np.std(sim) > 1e-9 and np.std(meas) > 1e-9 else 0.0
    cols8 = [0, 1, 6, 7, 8, 9, 10, 11]
    meas_vcg = KORS @ meas[cols8]
    angle = float(np.degrees(np.arccos(np.clip(np.dot(vcg_u, meas_vcg) / (np.linalg.norm(meas_vcg) + 1e-9), -1, 1))))
    return {"st_vector_sim": vcg_u, "angle_deg": angle, "r_leads": r, "simulated_leads": sim, "measured_leads": meas}


# ------------------------------------------------------------------ Vergleich
class InverseFitter:
    def __init__(self, model: HeartModel, solver: ActivationSolver, simulator, timeline: BeatTimeline):
        self.model = model
        # Patient-specific research fits must never mutate the live animation
        # solver, which can be used concurrently by the GUI and other cases.
        local_solver = ActivationSolver(model, grid=solver.grid)
        local_solver.set_params(**solver.params)
        self.forward = ForwardModel(model, local_solver)
        self._base_size_factor = float(local_solver.params["size_factor"])
        self.simulator = simulator  # DipoleSimulator (kalibriert)
        self.tl = timeline
        self.orientation_note = ""
        self.prior_note = ""
        self.use_t_wave = True
        from .torso import ELECTRODE_BY_NAME, dipole_lead_field
        self._lead_field = np.vstack([dipole_lead_field(ELECTRODE_BY_NAME[n].position[None, :], simulator.centre)[0] - simulator._wct
                                      for n in ("V1", "V2", "V3", "V4", "V5", "V6")])  # (6, 3)
        ra, la, ll = (dipole_lead_field(ELECTRODE_BY_NAME[n].position[None, :], simulator.centre)[0] for n in ("RA", "LA", "LL"))
        self._limb = {"I": la - ra, "II": ll - ra}

    def measured_window(self, beat: Beat, extra_ms: float = 0.0) -> tuple[np.ndarray, np.ndarray, int, int]:
        tl = self.tl
        a, b = int(beat.qrs_on * tl.fs), int((beat.qrs_off + extra_ms / 1000.0) * tl.fs)
        b = min(b, len(tl.signal))
        names = list(tl.lead_names)
        cols = [names.index(n) for n in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")]
        base = tl.signal[max(0, a - int(0.06 * tl.fs)):max(1, a - int(0.01 * tl.fs)), :][:, cols].mean(0)
        seg = tl.signal[a:b, :][:, cols] - base
        t = (np.arange(a, b) - a) / tl.fs * 1000.0
        return t, seg, a, b

    def simulate_leads(self, activation: ActivationResult, t_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(12 Ableitungen (n, 12), Dipol im VCG-Raum (n, 3)) auf dem Messraster."""
        step = max(float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else 2.0, 0.5)
        grid = np.arange(-15.0, t_ms[-1] + 15.0 + step, step)
        d = self.forward.dipole(activation.node_times_ms.astype(np.float64), grid)
        # Modell (x links, y posterior, z kranial) -> VCG (X links, Y inferior, Z posterior)
        vcg = np.column_stack([d[:, 0], -d[:, 2], d[:, 1]])
        return self._resample(self.leads_from_vcg(vcg), grid, t_ms), self._resample(vcg, grid, t_ms)

    @staticmethod
    def leads_from_vcg(vcg: np.ndarray) -> np.ndarray:
        """12 Ableitungen aus dem Herzvektor über die Dower-Transformation (empirischer Thorax)."""
        eight = vcg @ DOWER.T  # I, II, V1..V6
        lead_i, lead_ii = eight[:, 0], eight[:, 1]
        return np.column_stack([lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2, lead_i - lead_ii / 2,
                                lead_ii - lead_i / 2, eight[:, 2:]])

    @staticmethod
    def _resample(y: np.ndarray, grid: np.ndarray, t_ms: np.ndarray, shift: float = 0.0) -> np.ndarray:
        return np.column_stack([np.interp(t_ms + shift, grid, y[:, j]) for j in range(y.shape[1])])

    # Zwischenspeicher: rohe (unskalierte, ungedrehte) Aktivierung und Dipol je Hypothese und Parametersatz –
    # unabhängig von der Aufzeichnung, daher über Aufzeichnungen hinweg wiederverwendbar (Kalibrierung, Sweeps)
    _raw_cache: dict = {}

    def _raw_for(self, h: Hypothesis):
        key = (h.key, h.origin_vertex, h.accessory_vertex, tuple(sorted(self.forward.solver.params.items())),
               float(self.forward.rv_volume_factor), id(self.model))
        item = self._raw_cache.get(key)
        if item is None:
            act = activation_from_hypothesis(self.forward.solver, h, 100.0, exact_scale=False, scale_limits=(1.0, 1.0))
            span = float(np.percentile(act.times_ms, 97.0)) or 1.0
            grid = np.arange(-15.0, float(act.node_times_ms.max()) + 20.0, 1.0)
            rot = self.forward.rotation
            self.forward.rotation = np.eye(3)
            try:
                d = self.forward.dipole(act.node_times_ms.astype(np.float64), grid)
            finally:
                self.forward.rotation = rot
            item = (act, span, grid, d)
            if len(self._raw_cache) > 4000:
                self._raw_cache.clear()
            self._raw_cache[key] = item
        return item

    def evaluate(self, h: Hypothesis, beat: Beat, scale_limits: tuple[float, float] = (0.8, 1.35)) -> HypothesisResult:
        act_raw, span, grid_raw, d_raw = self._raw_for(h)
        desired_factor = beat.qrs_duration_ms / span
        factor = float(np.clip(desired_factor, scale_limits[0], scale_limits[1]))
        # Zeitskalierung: t -> factor·t, Dipol ∝ ∇τ skaliert mit 1/factor (Front verweilt länger, Amplitude je ms kleiner)
        grid = grid_raw * factor
        d = (d_raw @ self.forward.rotation.T) / factor
        times = act_raw.times_ms * factor
        activation = ActivationResult(times.astype(np.float32), act_raw.pattern, act_raw.tier, act_raw.label, act_raw.origin_vertices,
                                      act_raw.origin_label, act_raw.notes, act_raw.focal_vertices,
                                      node_times_ms=(act_raw.node_times_ms * factor).astype(np.float32), blocked=act_raw.blocked)
        sim_end = float(np.percentile(times, 97.0))
        t_ms, meas, a, b = self.measured_window(beat, extra_ms=max(0.0, sim_end - beat.qrs_duration_ms))
        vcg_sim = np.column_stack([d[:, 0], -d[:, 2], d[:, 1]])
        twelve = self.leads_from_vcg(vcg_sim)
        names = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
        cols8 = [0, 1, 6, 7, 8, 9, 10, 11]
        vcg_meas = meas[:, cols8] @ KORS.T
        meas_c = meas - meas.mean(0)
        best = None
        for shift in np.arange(-MAX_SHIFT_MS, MAX_SHIFT_MS + 0.1, 2.0):
            sim = self._resample(twelve, grid, t_ms, shift)
            num = float(np.sum(sim * meas))
            den = float(np.sum(sim * sim)) + 1e-12
            scale = max(num / den, 1e-6)  # nur positive Maßstäbe: eine Vorzeichenumkehr darf nie als Passung zählen
            sim_s = sim * scale
            sim_c = sim_s - sim_s.mean(0)
            sx, sy = np.sqrt((sim_c ** 2).sum(0)), np.sqrt((meas_c ** 2).sum(0))
            r_all = np.where((sx > 1e-9) & (sy > 1e-9), (sim_c * meas_c).sum(0) / (sx * sy + 1e-12), 0.0)
            r_leads = {n: float(r_all[j]) for j, n in enumerate(names)}
            r_mean = float(r_all.mean())
            vs = self._resample(vcg_sim, grid, t_ms, shift)
            vs_c, vm_c = vs - vs.mean(0), vcg_meas - vcg_meas.mean(0)
            sx, sy = np.sqrt((vs_c ** 2).sum(0)), np.sqrt((vm_c ** 2).sum(0))
            r_vcg = float(np.where((sx > 1e-9) & (sy > 1e-9), (vs_c * vm_c).sum(0) / (sx * sy + 1e-12), 0.0).mean())
            rmse = float(np.sqrt(np.mean((sim_s - meas) ** 2)))
            cand = HypothesisResult(h, r_leads, r_mean, r_vcg, rmse, float(shift), scale, sim_s, activation)
            cand.time_scale_factor = factor
            cand.time_scale_limited = bool(desired_factor < scale_limits[0] or desired_factor > scale_limits[1])
            if best is None or cand.score > best.score:
                best = cand
        if self.use_t_wave and beat.t_on and beat.t_end and beat.t_end > beat.t_on + 0.05:
            self._evaluate_t_wave(best, beat, activation, factor)
        return best

    def _evaluate_t_wave(self, result: HypothesisResult, beat: Beat, activation: ActivationResult, factor: float) -> None:
        """T-Welle aus der Aktivierung der Hypothese simulieren und mit der gemessenen T-Welle vergleichen."""
        tl = self.tl
        names = list(tl.lead_names)
        cols = [names.index(n) for n in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")]
        a = int(beat.qrs_on * tl.fs)
        ta, tb = int(beat.t_on * tl.fs), min(int(beat.t_end * tl.fs), len(tl.signal))
        if tb - ta < 5:
            return
        base = tl.signal[max(0, a - int(0.06 * tl.fs)):max(1, a - int(0.01 * tl.fs)), :][:, cols].mean(0)
        meas = tl.signal[ta:tb, :][:, cols] - base
        t_ms = (np.arange(ta, tb) - a) / tl.fs * 1000.0
        node_times = activation.node_times_ms.astype(np.float64)
        # APD so wählen, dass das Ende der Repolarisation dem gemessenen T-Ende entspricht
        t_end_ms = (beat.t_end - beat.qrs_on) * 1000.0
        apd = float(np.clip(t_end_ms - float(np.percentile(node_times, 97.0)) - 10.0, 150.0, 450.0))
        grid = np.arange(0.0, t_end_ms + 60.0, 2.0)
        d = self.forward.repolarization_dipole(node_times, grid, apd)
        vcg = np.column_stack([d[:, 0], -d[:, 2], d[:, 1]])
        sim = self._resample(self.leads_from_vcg(vcg), grid, t_ms)
        num = float(np.sum(sim * meas))
        den = float(np.sum(sim * sim)) + 1e-12
        sim_s = sim * max(num / den, 1e-6)
        sc, mc = sim_s - sim_s.mean(0), meas - meas.mean(0)
        sx, sy = np.sqrt((sc ** 2).sum(0)), np.sqrt((mc ** 2).sum(0))
        r = np.where((sx > 1e-9) & (sy > 1e-9), (sc * mc).sum(0) / (sx * sy + 1e-12), 0.0)
        result.r_t_wave = float(r.mean())
        result.simulated_t_leads = sim_s
        result.t_window = (ta, tb)

    def set_patient(self, age=None, sex=None, height_cm=None, weight_kg=None) -> str:
        """Patientenvorannahmen anwenden: Achsen-Vorannahme als Drehung um die anterior-posteriore Achse,
        beobachtete QRS-Dauer als begrenzten Größenfaktor (priors.py).

        A CV association is *not* applied separately: the observed QRS-duration
        regression already combines anatomy and conduction, so dividing by an
        additional CV factor would double-count an unidentifiable component.
        """
        from scipy.spatial.transform import Rotation as Rot
        from .priors import patient_prior
        prior = patient_prior(age, sex, height_cm, weight_kg)
        base = np.asarray(HEART_ROTATION, dtype=np.float64)
        # Drehung um die Modell-y-Achse (anterior-posterior): positiver Winkel verschiebt die frontale Achse nach inferior
        extra = Rot.from_euler("y", AXIS_ROTATION_SIGN * prior.axis_shift_deg, degrees=True).as_matrix()
        self.forward.rotation = extra @ base
        self.forward.solver.set_params(size_factor=self._base_size_factor * prior.duration_factor)
        self.prior_note = prior.note
        return prior.note

    def personalize_orientation(self, beat: Beat, candidates: list[Hypothesis], span_deg: float = 15.0) -> tuple[np.ndarray, str]:
        """Herzlage je Aufzeichnung: kleine Drehungen um die kalibrierte Lage (Hoekema 1999: Längsachse ±9° quer, ±5° frontal),
        gewählt nach der mittleren Passung der besten Hypothesen. Liefert (Rotation, Beschreibung)."""
        from scipy.spatial.transform import Rotation as Rot
        base = self.forward.rotation.copy()
        best = (None, -np.inf, (0.0, 0.0, 0.0))
        for psi in (-span_deg, 0.0, span_deg):
            for theta in (-span_deg, 0.0, span_deg):
                for phi in (-span_deg, 0.0, span_deg):
                    extra = Rot.from_euler("zxy", [psi, theta, phi], degrees=True).as_matrix()
                    self.forward.rotation = extra @ base
                    score = float(np.mean([self.evaluate(h, beat).score for h in candidates]))
                    if score > best[1]:
                        best = (self.forward.rotation.copy(), score, (psi, theta, phi))
        self.forward.rotation = best[0]
        psi, theta, phi = best[2]
        return best[0], f"Herzlage angepasst: z {psi:+.0f}°, x {theta:+.0f}°, y {phi:+.0f}° gegenüber der Kalibrierung"

    def sweep(self, beat: Beat, hypotheses: list[Hypothesis], progress: Optional[Callable[[int, int, HypothesisResult], None]] = None,
              refine: bool = True, personalize: bool = False) -> list[HypothesisResult]:
        self.orientation_note = ""
        results = []
        for k, h in enumerate(hypotheses):
            res = self.evaluate(h, beat)
            results.append(res)
            if progress:
                progress(k + 1, len(hypotheses), res)
        if personalize and len(results) >= 3:
            results.sort(key=lambda r: r.score, reverse=True)
            _, self.orientation_note = self.personalize_orientation(beat, [r.hypothesis for r in results[:3]])
            results = [self.evaluate(h, beat) for h in hypotheses]
        if refine:
            best_focus = max((r for r in results if r.hypothesis.family == "focus"), key=lambda r: r.score, default=None)
            if best_focus is not None:
                extra = refine_focus(self.model, best_focus.hypothesis)
                for k, h in enumerate(extra):
                    res = self.evaluate(h, beat)
                    results.append(res)
                    if progress:
                        progress(len(hypotheses) + k + 1, len(hypotheses) + len(extra), res)
        results.sort(key=lambda r: r.score, reverse=True)
        return results


# ------------------------------------------------------------------ Vorhöfe (P-Welle)
ATRIAL_LABELS = {
    "sinus": "Sinusknoten (RA hoch, Bachmann → LA)",
    "ectopic_left": "Linksatrialer Fokus (Pulmonalvenen-nah)",
    "retrograde": "Retrograd vom AV-Knoten (tiefe Vorhoferregung)",
    "low_right": "Tief rechtsatrial (Koronarsinus-nah)",
}


class AtrialForward:
    """P-Wellen-Dipol aus der Vorhofaktivierung (dünne Wand: Fläche × 2,5 mm × ∇τ entlang der Oberfläche)."""

    def __init__(self, model: HeartModel, solver: ActivationSolver, thickness_mm: float = 2.5):
        self.model, self.solver = model, solver
        self.parts = {}
        for name in ("ra_wall", "la_wall"):
            part = model.parts[name]
            v = part.vertices.astype(np.float64)
            f = part.faces
            a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            area = np.bincount(f.ravel(), weights=np.repeat(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) / 3.0, 3), minlength=len(v))
            g = model.atrial_graphs[name].tocoo()
            d = v[g.col] - v[g.row]
            A = np.zeros((len(v), 3, 3))
            np.add.at(A, g.row, d[:, :, None] * d[:, None, :])
            self.parts[name] = dict(ei=g.row, ej=g.col, d=d, A_inv=np.linalg.pinv(A + 1e-6 * np.eye(3)[None]), volume=area * thickness_mm)

    def times_for(self, key: str) -> dict[str, np.ndarray]:
        from .activation import AtrialPattern
        src = self.model.atrial_sources
        if key == "sinus":
            res = self.solver.atrial(AtrialPattern.SINUS, 100.0)
            return {"ra_wall": res.ra_times_ms, "la_wall": res.la_times_ms}
        if key == "ectopic_left":
            res = self.solver.atrial(AtrialPattern.ECTOPIC_LEFT, 100.0)
            return {"ra_wall": res.ra_times_ms, "la_wall": res.la_times_ms}
        if key == "retrograde":
            res = self.solver.atrial(AtrialPattern.RETROGRADE, 100.0)
            return {"ra_wall": res.ra_times_ms, "la_wall": res.la_times_ms}
        # tief rechtsatrial: Start am AV-nahen RA-Punkt, LA über den Koronarsinus-nahen Bereich
        ra = self.solver._atrial_solve("ra_wall", {src["ra_wall"]["av_node"]: 0.0})
        la = self.solver._atrial_solve("la_wall", {src["la_wall"]["av_node"]: 15.0})
        return {"ra_wall": ra, "la_wall": la}

    def dipole(self, times: dict[str, np.ndarray], grid: np.ndarray, sigma_ms: float = 6.0) -> np.ndarray:
        hist = np.zeros((len(grid), 3))
        step = float(grid[1] - grid[0])
        for name, tm in times.items():
            p = self.parts[name]
            dt = tm[p["ej"]] - tm[p["ei"]]
            b = np.zeros((len(tm), 3))
            np.add.at(b, p["ei"], p["d"] * dt[:, None])
            g = np.einsum("nij,nj->ni", p["A_inv"], b)
            n = np.linalg.norm(g, axis=1)
            too = n > 3.0
            g[too] *= (3.0 / n[too])[:, None]
            w = p["volume"][:, None] * g
            idx = np.clip(np.round((tm - grid[0]) / step).astype(int), 0, len(grid) - 1)
            np.add.at(hist, idx, w)
        k = int(np.ceil(3 * sigma_ms / step))
        kernel = np.exp(-0.5 * (np.arange(-k, k + 1) * step / sigma_ms) ** 2)
        kernel /= kernel.sum()
        return np.column_stack([np.convolve(hist[:, j], kernel, mode="same") for j in range(3)])


def atrial_sweep(fitter: "InverseFitter", beat: Beat) -> list[tuple[str, float, np.ndarray]]:
    """Rangfolge der Vorhofhypothesen an der gemessenen P-Welle: (Schlüssel, r, simulierte 12 Ableitungen)."""
    if not beat.has_p or not beat.p_measured:
        return []
    tl = fitter.tl
    names = list(tl.lead_names)
    cols = [names.index(n) for n in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")]
    a, b = int(beat.p_on * tl.fs), int(beat.p_off * tl.fs)
    if b - a < 4:
        return []
    base = tl.signal[max(0, a - int(0.04 * tl.fs)):max(1, a - int(0.005 * tl.fs)), :][:, cols].mean(0)
    meas = tl.signal[a:b, :][:, cols] - base
    t_ms = (np.arange(a, b) - a) / tl.fs * 1000.0
    af = AtrialForward(fitter.model, fitter.forward.solver)
    out = []
    p_dur = (beat.p_off - beat.p_on) * 1000.0
    for key in ATRIAL_LABELS:
        times = af.times_for(key)
        span = max(float(np.percentile(np.concatenate(list(times.values())), 97.0)), 1.0)
        factor = p_dur / span
        scaled = {k: v * factor for k, v in times.items()}
        grid = np.arange(-10.0, p_dur + 30.0, 1.0)
        d = af.dipole(scaled, grid) @ fitter.forward.rotation.T
        vcg = np.column_stack([d[:, 0], -d[:, 2], d[:, 1]])
        best = (-2.0, None)
        for shift in np.arange(-10.0, 10.1, 2.0):
            sim = fitter._resample(fitter.leads_from_vcg(vcg), grid, t_ms, shift)
            num = float(np.sum(sim * meas))
            den = float(np.sum(sim * sim)) + 1e-12
            sim_s = sim * max(num / den, 1e-6)
            sc, mc = sim_s - sim_s.mean(0), meas - meas.mean(0)
            sx, sy = np.sqrt((sc ** 2).sum(0)), np.sqrt((mc ** 2).sum(0))
            r = float(np.where((sx > 1e-9) & (sy > 1e-9), (sc * mc).sum(0) / (sx * sy + 1e-12), 0.0).mean())
            if r > best[0]:
                best = (r, sim_s)
        out.append((key, best[0], best[1]))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def hypotheses_for_beat(model: HeartModel, beat: Beat, hints: dict) -> list[Hypothesis]:
    """Hypothesenraum je Schlagtyp: Leitungssystem (+ Bahnen) für N, Fokusraster (+ Stimulation) für V/P."""
    if beat.beat_type in ("V", "P"):
        out = focus_hypotheses(model, beat.beat_type) + pacing_hypotheses(model)
        return out
    out = conduction_hypotheses()
    if hints.get("wpw") or (beat.has_p and (beat.qrs_on - beat.p_on) < 0.13):
        out += accessory_hypotheses(model)
    return out
