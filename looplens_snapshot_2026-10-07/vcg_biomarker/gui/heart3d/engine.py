"""Animationsmotor: aus Zeitleiste + Aktivierungskarten je Zeitpunkt Vertexpositionen und -farben.

Mechanik (Vorlagen, keine Messung):
* Kontraktion je Vertex beginnt EMD nach der lokalen elektrischen Aktivierung,
  erreicht das Plateau nach ~120 ms, hält bis zum Ejektionsende (Weissler) und
  relaxiert über IVRT + frühe Füllung.
* Radiale Verkürzung um die Kammerachse (Endokard stärker als Epikard, damit die
  Wand dicker wird), Längsverkürzung zur Spitze hin (Basis bewegt sich zum Apex),
  leichte Torsion.
* Vorhöfe füllen sich während der Kammersystole und entleeren sich mit der
  A-Welle (nur bei vorhandener P-Welle).

Erregung (Farbe): gelbe Wellenfront an der Aktivierungsfront, leichter Gelbton
im depolarisierten Gewebe bis zur Repolarisation, hellblaue Repolarisationswelle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .activation import (ActivationSolver, ActivationResult, AtrialPattern, AtrialResult, Pattern,
                         accessory_vertex_for, choose_atrial_pattern, choose_pattern, ectopic_origin_vertex,
                         vcg_to_model)
from .model import CHAMBER_LV, CHAMBER_RV, HeartModel
from .pathology import PathologyEstimate, empty_estimate, estimate_pathology
from .timeline import KORS, Beat, BeatTimeline, MechanicalEvents, mechanical_events

YELLOW = np.array([1.0, 0.92, 0.25], dtype=np.float32)
DEPOLARIZED_TINT = np.array([1.0, 0.75, 0.35], dtype=np.float32)
REPOL_BLUE = np.array([0.55, 0.80, 1.0], dtype=np.float32)
ISCHEMIA_COLOR = np.array([0.30, 0.12, 0.72], dtype=np.float32)  # kräftig violett-blau
SCAR_COLOR = np.array([0.93, 0.93, 0.90], dtype=np.float32)  # fibrös, hell
VCG_SCALE_MM = 45.0  # Spitzenvektor der Aufzeichnung -> 45 mm
COLOR_MODES = ("Erregung", "Isochronen", "AHA-Segmente", "Nur Pathologie", "Einfarbig")
EMD_EARLY_MS = 35.0
EMD_LATE_MS = 50.0


class RhythmScenario(str, Enum):
    AUTO = "auto"
    JUNCTIONAL_ESCAPE = "junctional_escape"
    VENTRICULAR_ESCAPE = "ventricular_escape"
    AVNRT = "avnrt"
    AVRT_ORTHO = "avrt_ortho"
    AVRT_ANTI = "avrt_anti"
    VT_REENTRY = "vt_reentry"


SCENARIO_LABELS = {
    RhythmScenario.AUTO: "Automatisch (Ersatzrhythmen werden erkannt)",
    RhythmScenario.JUNCTIONAL_ESCAPE: "Junktionaler Ersatzrhythmus (His, retrograde Vorhoferregung)",
    RhythmScenario.VENTRICULAR_ESCAPE: "Ventrikulärer Ersatzrhythmus (langsamer Kammerfokus)",
    RhythmScenario.AVNRT: "AV-Knoten-Reentry (AVNRT, Kreis im Knoten, P im QRS)",
    RhythmScenario.AVRT_ORTHO: "AVRT orthodrom (antegrad AV-Knoten, retrograd akzessorische Bahn)",
    RhythmScenario.AVRT_ANTI: "AVRT antidrom (antegrad Bahn, retrograd His/AV-Knoten)",
    RhythmScenario.VT_REENTRY: "Kammertachykardie um Narbe (Kreiserregung, Austritt am Fokus)",
}


@dataclass
class Layers:
    excitation: bool = True
    mechanics: bool = True
    repolarization: bool = True
    isochrones: bool = False
    conduction_system: bool = True
    atria: bool = True
    vessels: bool = True
    coronaries: bool = True
    cutaway: bool = False
    heart_vector: bool = False
    stall_cone: bool = False
    origin_markers: bool = True
    pathology: bool = True
    vcg_loop: bool = False
    vcg_projections: bool = False
    electrodes: bool = False
    lead_traces: bool = False
    torso: bool = False
    potential_map: bool = False
    virtual_electrode: bool = False


@dataclass
class BeatPlan:
    """Alles, was der Motor je Schlag braucht (einmal berechnet)."""
    beat: Beat
    events: MechanicalEvents
    ventricular: ActivationResult
    atrial: AtrialResult
    pattern_reason: str
    pattern_tier: str
    origin_label: str
    notes: tuple[str, ...]
    mech_onset: np.ndarray  # absolute Sekunden je Wandvertex
    scenario: "RhythmScenario" = RhythmScenario.AUTO


@dataclass
class FrameData:
    time: float
    positions: dict[str, np.ndarray]
    colors: dict[str, np.ndarray]
    conduction_progress: float  # 0..1 entlang SA -> AV -> His -> Schenkel (grob, für die 3D-Linien)
    heart_vector: Optional[np.ndarray]
    plan: Optional[BeatPlan]
    phase_label: str
    sample_index: int = -1  # Zeitindex im Signal (für VCG-Punkt und Potentialkarte)
    conduction: dict = field(default_factory=dict)  # Segment -> Impulsposition 0..1 (entlang der Zeichenrichtung)
    blocked: tuple = ()  # Segmente mit Leitungsblock
    conduction_notes: tuple = ()


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class AnimationEngine:
    def __init__(self, model: HeartModel):
        self.model = model
        self.solver = ActivationSolver(model)
        self.timeline: Optional[BeatTimeline] = None
        self.pattern_override: Pattern = Pattern.AUTO
        self.use_thesis_modulation: bool = True  # Thesenmarker der Analyse fließen standardmäßig ein
        self.hypothesis = None  # aus der inversen Simulation übernommene Hypothese (inverse.Hypothesis)
        self.rhythm_override: RhythmScenario = RhythmScenario.AUTO
        self.patient: dict = {}  # optionale Patientendaten (Alter, Geschlecht, Größe, Gewicht)
        self.layers = Layers()
        self.color_mode: str = COLOR_MODES[0]
        self.wall_opacity: float = 1.0
        self._plans: dict[int, BeatPlan] = {}
        self._plain_wall_colors = self._base_wall_colors()
        self._wall_base_colors = self._plain_wall_colors
        self._static_positions = {name: part.vertices for name, part in model.parts.items()}
        self._wall_rest = model.wall.vertices
        self._lv_frame = model.lv_frame
        self._rv_frame = model.rv_frame
        self.pathology: PathologyEstimate = empty_estimate(model)
        self.payload: Optional[dict] = None
        self._vcg_cache: dict[int, dict] = {}
        self._vcg_scale: float = 1.0
        self.vcg_scale_factor: float = 1.0  # Anwenderfaktor (Schieberegler)
        self._aha_colors = self._segment_colors()

    # --------------------------------------------------------------- Setup
    def _base_wall_colors(self) -> np.ndarray:
        m = self.model
        colors = np.empty((m.wall.n_vertices, 4), dtype=np.float32)
        colors[m.wall_chamber == CHAMBER_LV] = m.parts["wall"].color
        from .model import COLOR_RV
        colors[m.wall_chamber == CHAMBER_RV] = COLOR_RV
        # Endokard etwas dunkler, Epikard etwas heller -> Tiefe im Schnittbild
        shade = np.where(m.wall_endocardial, 0.85, 1.0).astype(np.float32)
        colors[:, :3] *= shade[:, None]
        return colors

    def _segment_colors(self) -> np.ndarray:
        seg = self.model.wall_segment.astype(np.int64)
        rng = np.random.default_rng(11)
        palette = {s: 0.35 + 0.6 * rng.random(3) for s in np.unique(seg)}
        out = np.ones((len(seg), 4), dtype=np.float32)
        for s_id, c in palette.items():
            out[seg == s_id, :3] = c
        return out

    def set_timeline(self, timeline: Optional[BeatTimeline], payload: Optional[dict] = None) -> None:
        self.timeline = timeline
        self.payload = payload
        self._plans.clear()
        self._vcg_cache.clear()
        self.pathology = empty_estimate(self.model)
        if timeline is not None:
            try:
                self.pathology = estimate_pathology(self.model, timeline, payload)
            except Exception as exc:  # Schätzung darf die Animation nie verhindern
                self.pathology = empty_estimate(self.model)
                self.pathology.findings.append(("E", "Pathologie-Schätzung nicht verfügbar", str(exc)))
        self._apply_pathology_geometry()
        self._vcg_scale = self._compute_vcg_scale()

    def _apply_pathology_geometry(self) -> None:
        est = self.pathology
        if self.layers.pathology and np.abs(est.wall_offset).max() > 0.01:
            self._wall_rest = (self.model.wall.vertices + est.wall_offset).astype(np.float32)
        else:
            self._wall_rest = self.model.wall.vertices
        self._static_positions["wall"] = self._wall_rest
        base = self._plain_wall_colors.copy()
        if self.layers.pathology:
            isch = est.ischemia[:, None]
            base[:, :3] = base[:, :3] * (1.0 - 0.9 * isch) + ISCHEMIA_COLOR * (0.9 * isch)
            scar = est.scar[:, None]
            base[:, :3] = base[:, :3] * (1.0 - 0.9 * scar) + SCAR_COLOR * (0.9 * scar)
        self._wall_base_colors = base

    def set_pathology_visible(self, visible: bool) -> None:
        self.layers.pathology = bool(visible)
        self._apply_pathology_geometry()

    def set_color_mode(self, mode: str) -> None:
        if mode in COLOR_MODES:
            self.color_mode = mode

    def set_vcg_scale_factor(self, factor: float) -> None:
        self.vcg_scale_factor = float(np.clip(factor, 0.2, 4.0))
        self._vcg_cache.clear()

    @property
    def vcg_scale(self) -> float:
        return self._vcg_scale * self.vcg_scale_factor

    def _compute_vcg_scale(self) -> float:
        tl = self.timeline
        if tl is None or tl.signal is None:
            return 1.0
        names = list(tl.lead_names)
        try:
            cols = [names.index(n) for n in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]
        except ValueError:
            return 1.0
        peaks = []
        for b in tl.beats[:30]:
            a, c = int(b.qrs_on * tl.fs), int(b.qrs_off * tl.fs)
            if c - a < 3 or a < 0 or c > len(tl.signal):
                continue
            xyz = (KORS @ tl.signal[a:c, :][:, cols].T).T
            peaks.append(float(np.linalg.norm(xyz, axis=1).max()))
        peak = float(np.percentile(peaks, 95)) if peaks else 1.0
        return VCG_SCALE_MM / max(peak, 0.2)

    def vcg_loop(self, beat: Beat) -> Optional[dict]:
        """Schleife des Schlags in Modellkoordinaten (mm, um das Herzzentrum) mit Abschnittslabels."""
        cached = self._vcg_cache.get(beat.index)
        if cached is not None:
            return cached
        tl = self.timeline
        if tl is None or tl.signal is None:
            return None
        names = list(tl.lead_names)
        try:
            cols = [names.index(n) for n in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]
        except ValueError:
            return None
        start = (beat.p_on - 0.01) if beat.has_p else beat.qrs_on - 0.04
        end = (beat.t_end + 0.02) if beat.t_end else beat.qrs_off + 0.3
        a, b = max(0, int(start * tl.fs)), min(len(tl.signal), int(end * tl.fs))
        if b - a < 5:
            return None
        q0 = int(beat.qrs_on * tl.fs)
        base = tl.signal[max(0, q0 - int(0.06 * tl.fs)):max(1, q0 - int(0.01 * tl.fs)), :][:, cols].mean(0)
        xyz = (KORS @ (tl.signal[a:b, :][:, cols] - base).T).T
        pts = np.column_stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]]) * self.vcg_scale
        centre = 0.5 * (self.model.lv_frame.center + self.model.rv_frame.center)
        t = np.arange(a, b) / tl.fs
        seg = np.full(len(t), 2, dtype=np.int8)  # 0 P, 1 QRS, 2 T/sonst
        if beat.has_p:
            seg[(t >= beat.p_on) & (t < beat.p_off)] = 0
        seg[(t >= beat.qrs_on) & (t <= beat.qrs_off)] = 1
        cached = {"points": (centre + pts).astype(np.float32), "times": t, "segment": seg, "first_sample": a, "centre": centre}
        self._vcg_cache[beat.index] = cached
        return cached

    def set_pattern_override(self, pattern: Pattern) -> None:
        if pattern != self.pattern_override:
            self.pattern_override = pattern
            self._plans.clear()

    def set_rhythm_scenario(self, scenario: RhythmScenario) -> None:
        if scenario != self.rhythm_override:
            self.rhythm_override = scenario
            self._plans.clear()

    def _auto_scenario(self, beat: Beat) -> RhythmScenario:
        """Ersatzrhythmen erkennen: keine P-Welle, langsam; schmal → junktional, breit → ventrikulär."""
        tl = self.timeline
        if tl is None or beat.has_p or (tl.atrial_events):
            return RhythmScenario.AUTO
        if tl.rhythm_class in ("af_compatible_pattern", "flutter_compatible_pattern"):
            return RhythmScenario.AUTO
        hints = tl.conduction_hints
        af_hz = hints.get("atrial_frequency_hz")
        if af_hz is not None and 3.5 <= float(af_hz) <= 6.5:
            return RhythmScenario.AUTO
        if beat.hr <= 60 and beat.qrs_duration_ms < 120 and beat.beat_type in ("N", "S"):
            return RhythmScenario.JUNCTIONAL_ESCAPE
        if beat.hr <= 50 and beat.qrs_duration_ms >= 120:
            return RhythmScenario.VENTRICULAR_ESCAPE
        # Tachykardien ohne P: regelmäßig, breit und überwiegend "V" -> Kammertachykardie (Kreis); schmal -> AVNRT als häufigste SVT
        regular = tl.rhythm_class in ("regular", "wide_qrs_pattern", "mixed_or_ectopic_pattern")
        n_v = sum(1 for b in tl.beats if b.beat_type == "V")
        if beat.hr >= 100 and regular and beat.qrs_duration_ms >= 120 and n_v >= 0.6 * len(tl.beats):
            return RhythmScenario.VT_REENTRY
        if beat.hr >= 150 and regular and beat.qrs_duration_ms < 120:
            return RhythmScenario.AVNRT
        return RhythmScenario.AUTO

    def set_hypothesis(self, hypothesis) -> None:
        """Beste Hypothese der inversen Simulation für Schläge ihres Typs übernehmen (None = zurück zur Automatik)."""
        self.hypothesis = hypothesis
        self._plans.clear()

    def set_thesis_modulation(self, enabled: bool) -> None:
        if enabled != self.use_thesis_modulation:
            self.use_thesis_modulation = bool(enabled)
            self._plans.clear()

    # --------------------------------------------------------------- Planung
    def plan_for(self, beat: Beat) -> BeatPlan:
        plan = self._plans.get(beat.index)
        if plan is not None:
            return plan
        tl = self.timeline
        hints = dict(tl.conduction_hints) if tl else {}
        scenario = self.rhythm_override if self.rhythm_override != RhythmScenario.AUTO else self._auto_scenario(beat)
        pattern, tier, reason = choose_pattern(hints, beat.beat_type, self.pattern_override)
        origin_vertex = None
        origin_label = ""
        accessory = None
        notes: list[str] = []
        if scenario == RhythmScenario.JUNCTIONAL_ESCAPE:
            pattern, tier, reason = Pattern.NORMAL, "K", "Junktionaler Ersatzrhythmus: Erregung ab His-Bündel, Vorhöfe retrograd"
        elif scenario == RhythmScenario.VENTRICULAR_ESCAPE:
            pattern, tier, reason = Pattern.ECTOPIC, "K", "Ventrikulärer Ersatzrhythmus: langsamer Kammerfokus"
        elif scenario == RhythmScenario.AVNRT:
            pattern, tier, reason = Pattern.NORMAL, "K", "AVNRT: Kreiserregung im AV-Knoten, Kammer über His, Vorhöfe retrograd (P im QRS)"
        elif scenario == RhythmScenario.AVRT_ORTHO:
            pattern, tier, reason = Pattern.NORMAL, "K", "AVRT orthodrom: antegrad über AV-Knoten/His, retrograd über die akzessorische Bahn"
            accessory = accessory_vertex_for(self.model, hints.get("delta_axis_deg"))
        elif scenario == RhythmScenario.AVRT_ANTI:
            pattern, tier, reason = Pattern.WPW, "K", "AVRT antidrom: antegrad über die akzessorische Bahn (voll präexzitiert), retrograd über His/AV-Knoten"
        elif scenario == RhythmScenario.VT_REENTRY:
            pattern, tier, reason = Pattern.ECTOPIC, "K", "Kammertachykardie: Kreiserregung um eine Narbe/Isthmus, Austritt am Fokus"
        if pattern == Pattern.ECTOPIC:
            origin_vertex, origin_label = ectopic_origin_vertex(self.model, beat.qrs_vector, hints.get("secondary_morphology_qrs_octant_id"))
            mdi = hints.get("secondary_morphology_maximum_deflection_index")
            if mdi is not None and np.isfinite(mdi) and mdi >= 0.55:
                notes.append(f"MDI {mdi:.2f} ≥ 0,55: epikardialer Ursprung möglich (Daniels 2006)")
        if pattern == Pattern.WPW:
            accessory = accessory_vertex_for(self.model, hints.get("delta_axis_deg"))
        breakthrough = 4
        arc_fraction = None
        if self.use_thesis_modulation:
            cancel = hints.get("qrs_early_cancellation_fraction")
            if cancel is not None and np.isfinite(cancel):
                breakthrough = int(2 + round(2 * float(np.clip((cancel - 0.2) / 0.4, 0.0, 1.0))))
                notes.append(f"Durchbruchzahl {breakthrough} aus früher Aufhebung {cancel:.2f} (Hypothese, Katritsis-Vorhersage)")
            arc_fraction = hints.get("qrs_arc_length_fraction_at_half_time")
        hyp = self.hypothesis
        if scenario == RhythmScenario.VT_REENTRY:
            from .inverse import activation_from_hypothesis, reentry_hypothesis
            from .pathology import zone_centroid
            centre = zone_centroid(self.model, self.pathology.scar) if self.pathology.scar.max() > 0.3 else None
            centre_vertex = self.model.wall_vertex_near(centre) if centre is not None else None
            if origin_vertex is None:
                origin_vertex, origin_label = ectopic_origin_vertex(self.model, beat.qrs_vector, hints.get("secondary_morphology_qrs_octant_id"))
            ventricular = activation_from_hypothesis(self.solver, reentry_hypothesis(self.model, int(origin_vertex), centre_vertex), beat.qrs_duration_ms)
            notes.append("Kreis um " + ("die geschätzte Narbe" if centre is not None else "den Fokus") + " (Isthmus langsam)")
        elif hyp is not None and (hyp.beat_type == beat.beat_type or (hyp.beat_type == "N" and beat.beat_type == "S")):
            from .inverse import activation_from_hypothesis
            ventricular = activation_from_hypothesis(self.solver, hyp, beat.qrs_duration_ms)
            pattern, tier, reason = ventricular.pattern, "H", f"Inverse Simulation: {hyp.label}"
            origin_label = "aus der inversen Simulation" if hyp.family in ("focus", "pacing", "accessory") else origin_label
        else:
            ventricular = self.solver.ventricular(pattern, beat.qrs_duration_ms, origin_vertex=origin_vertex,
                                                  breakthrough_sites=breakthrough, accessory_vertex=accessory,
                                                  arc_fraction_half_time=arc_fraction)
        atrial_pattern = choose_atrial_pattern(tl.rhythm_class if tl else "unknown", tl.rhythm_flags if tl else (),
                                               beat.has_p, beat.beat_type,
                                               atrial_frequency_hz=hints.get("atrial_frequency_hz"), ventricular_hz=beat.hr / 60.0)
        if tl is not None and tl.atrial_events:
            atrial_pattern = AtrialPattern.SINUS
        if scenario in (RhythmScenario.JUNCTIONAL_ESCAPE, RhythmScenario.AVNRT, RhythmScenario.AVRT_ORTHO, RhythmScenario.AVRT_ANTI):
            atrial_pattern = AtrialPattern.RETROGRADE
        elif scenario in (RhythmScenario.VENTRICULAR_ESCAPE, RhythmScenario.VT_REENTRY) and not beat.has_p:
            atrial_pattern = AtrialPattern.NONE
        atrial = self.solver.atrial(atrial_pattern, beat.p_duration_ms)
        if tl is not None and tl.atrial_events:
            atrial.label = "Unabhängiger Vorhofrhythmus (AV-Dissoziation): Sinusknoten → Vorhöfe"
        if scenario == RhythmScenario.AVRT_ORTHO:
            atrial.label = "Retrograd über die akzessorische Bahn (Vorhof nach dem QRS)"
        elif scenario in (RhythmScenario.AVNRT, RhythmScenario.JUNCTIONAL_ESCAPE):
            atrial.label = "Retrograd über den AV-Knoten (P im oder kurz nach dem QRS)"
        events = mechanical_events(beat, tl.female if tl else False)
        t_act = ventricular.times_ms.astype(np.float64)
        rank = np.clip(t_act / max(float(t_act.max()), 1.0), 0.0, 1.0)
        emd = EMD_EARLY_MS + (EMD_LATE_MS - EMD_EARLY_MS) * rank
        mech_onset = beat.qrs_on + (t_act + emd) / 1000.0
        tier_final = "H" if ventricular.tier == "H" else tier
        plan = BeatPlan(beat, events, ventricular, atrial, reason, tier_final, origin_label,
                        tuple(notes) + ventricular.notes, mech_onset, scenario)
        self._plans[beat.index] = plan
        return plan

    # --------------------------------------------------------------- Frame
    def frame(self, t: float) -> FrameData:
        positions = dict(self._static_positions)
        colors: dict[str, np.ndarray] = {}
        tl = self.timeline
        beat = tl.beat_at(t) if tl else None
        if beat is None:
            colors["wall"] = self._wall_base_colors
            return FrameData(t, positions, colors, 0.0, None, None, "Kein Schlag")
        plan = self.plan_for(beat)
        mode = self.color_mode
        if mode == "AHA-Segmente":
            wall_colors = self._aha_colors.copy()
        elif mode == "Einfarbig":
            wall_colors = self._plain_wall_colors.copy()
        else:
            wall_colors = self._wall_base_colors.copy()
        t_ms = (t - beat.qrs_on) * 1000.0
        t_act = plan.ventricular.times_ms

        # ---- Erregung: Wellenfront gelb, depolarisiert leicht getönt, Repolarisation blau
        if self.layers.excitation and mode in ("Erregung", "AHA-Segmente"):
            dt = t_ms - t_act
            front = np.exp(-0.5 * (dt / 9.0) ** 2) * (dt > -18.0)
            front = np.clip(front, 0.0, 1.0)
            depolarized = (dt > 0).astype(np.float32)
            repol_start = (beat.t_on if beat.t_on else beat.qrs_off + 0.06)
            repol_end = beat.t_end if beat.t_end else repol_start + 0.2
            if self.layers.repolarization and repol_end > repol_start:
                order = 1.0 - np.clip(t_act / max(float(t_act.max()), 1.0), 0.0, 1.0)  # spät aktiviert repolarisiert früh
                t_rep = (repol_start + (repol_end - repol_start) * order - beat.qrs_on) * 1000.0
                drep = t_ms - t_rep
                repol_front = np.exp(-0.5 * (drep / 14.0) ** 2) * (drep > -25.0)
                repolarized = (drep > 0).astype(np.float32)
                depolarized = depolarized * (1.0 - repolarized)
            else:
                repol_front = np.zeros_like(front)
            base = wall_colors[:, :3]
            tinted = base * (1.0 - 0.35 * depolarized[:, None]) + DEPOLARIZED_TINT * (0.35 * depolarized[:, None])
            tinted = tinted * (1.0 - front[:, None]) + YELLOW * front[:, None]
            tinted = tinted * (1.0 - 0.5 * repol_front[:, None]) + REPOL_BLUE * (0.5 * repol_front[:, None])
            wall_colors[:, :3] = tinted
        if self.layers.pathology and mode != "Einfarbig":
            est = self.pathology
            isch = est.ischemia[:, None]
            scar = est.scar[:, None]
            if mode == "Nur Pathologie":
                wall_colors[:, :3] = self._plain_wall_colors[:, :3]
                strength_i, strength_s = 0.95, 0.95
            else:
                strength_i, strength_s = 0.55, 0.6
            wall_colors[:, :3] = wall_colors[:, :3] * (1.0 - strength_i * isch) + ISCHEMIA_COLOR * (strength_i * isch)
            wall_colors[:, :3] = wall_colors[:, :3] * (1.0 - strength_s * scar) + SCAR_COLOR * (strength_s * scar)
        if self.layers.isochrones or mode == "Isochronen":
            q = np.clip(t_act / max(float(t_act.max()), 1.0), 0.0, 1.0)
            iso = np.stack([q, 1.0 - np.abs(q - 0.5) * 2.0, 1.0 - q], axis=1)
            wall_colors[:, :3] = 0.35 * wall_colors[:, :3] + 0.65 * iso
        if self.wall_opacity < 0.999:
            wall_colors[:, 3] = self.wall_opacity
        colors["wall"] = wall_colors

        # ---- Vorhöfe
        colors["ra_wall"], colors["la_wall"] = self._atrial_colors(plan, beat, t)

        # ---- Mechanik
        if self.layers.mechanics:
            positions["wall"] = self._deform_wall(plan, t)
            positions["ra_wall"], positions["la_wall"] = self._deform_atria(plan, beat, t)
            for name in ("lv_cavity", "rv_cavity"):
                positions[name] = self._static_positions[name]
            positions["aorta"] = self._pulse(self._static_positions["aorta"], plan, t, 0.03)
            positions["aortic_arch"] = self._pulse(self._static_positions["aortic_arch"], plan, t, 0.02)

        # ---- Leitungssystem-Fortschritt und Herzvektor
        progress = self._conduction_progress(beat, t)
        vector = None
        if (self.layers.heart_vector or self.layers.vcg_loop) and tl is not None and tl.signal is not None:
            vector = self._instant_vector(tl, t)
        sample = int(t * tl.fs) if tl is not None and tl.signal is not None else -1
        state, blocked, cnotes = self.conduction_state(plan, t)
        return FrameData(t, positions, colors, progress, vector, plan, self._phase_label(plan, t), sample,
                         state, blocked, cnotes)

    # --------------------------------------------------------------- Teile
    def _contraction(self, plan: BeatPlan, t: float) -> np.ndarray:
        """Kontraktionsgrad 0..1 je Wandvertex."""
        onset = plan.mech_onset
        rise = _smoothstep((t - onset) / 0.12)
        ev = plan.events
        release_start = ev.ao_close
        release = 1.0 - _smoothstep((t - release_start) / (ev.ivrt_ms / 1000.0 + 0.16))
        c = rise * np.where(t > release_start, release, 1.0)
        if self.layers.pathology:
            # Hypo-/Akinesie: Narbe kontrahiert kaum, ischämisches Gewebe abgeschwächt
            c = c * (1.0 - 0.85 * self.pathology.scar) * (1.0 - 0.55 * self.pathology.ischemia)
        # Vor dem nächsten Schlag vollständig relaxiert
        return np.clip(c, 0.0, 1.0)

    def _deform_wall(self, plan: BeatPlan, t: float) -> np.ndarray:
        m = self.model
        c = self._contraction(plan, t)
        v = self._wall_rest.astype(np.float64)
        out = v.copy()
        for chamber, frame, k_radial_endo, k_radial_epi, k_long in (
            (CHAMBER_LV, self._lv_frame, 0.28, 0.11, 0.14),
            (CHAMBER_RV, self._rv_frame, 0.22, 0.09, 0.12),
        ):
            mask = m.wall_chamber == chamber
            rel = v[mask] - frame.center
            proj = rel @ frame.axis
            radial = rel - np.outer(proj, frame.axis)
            u = np.clip(m.wall_u[mask], 0.0, 1.0)
            cc = c[mask]
            k_r = np.where(m.wall_endocardial[mask], k_radial_endo, k_radial_epi)
            radial_scaled = radial * (1.0 - k_r * cc)[:, None]
            # Längsverkürzung: Basis wandert Richtung Apex (Apex bleibt)
            proj_new = proj - k_long * cc * u * (frame.proj_base - frame.proj_apex)
            # Torsion: Apex und Basis gegensinnig (≈ 8°)
            twist = np.radians(8.0) * cc * (u - 0.5) * 2.0
            cos_t, sin_t = np.cos(twist), np.sin(twist)
            a = radial_scaled @ frame.e_septal
            b = radial_scaled @ frame.e_anterior
            a2 = a * cos_t - b * sin_t
            b2 = a * sin_t + b * cos_t
            radial_rot = np.outer(a2, frame.e_septal) + np.outer(b2, frame.e_anterior)
            out[mask] = frame.center + np.outer(proj_new, frame.axis) + radial_rot
        return out.astype(np.float32)

    def _atrial_scale(self, plan: BeatPlan, beat: Beat, t: float) -> float:
        ev = plan.events
        scale = 1.0
        # Reservoirphase: Füllung während der Kammersystole
        fill = _smoothstep((t - ev.av_close) / max(ev.ao_close - ev.av_close, 0.05)) * (1.0 - _smoothstep((t - ev.av_open) / 0.12))
        scale += 0.06 * float(fill)
        a_on = a_off = None
        if ev.atrial_contraction_on is not None:
            a_on, a_off = ev.atrial_contraction_on, ev.atrial_contraction_off
        elif self.timeline is not None and self.timeline.atrial_events:
            event = self.timeline.atrial_event_at(t)
            if event is not None:
                a_on, a_off = event[0] + 0.08, event[1] + 0.12
        if a_on is not None:
            kick = _smoothstep((t - a_on) / 0.08) * (1.0 - _smoothstep((t - a_off) / 0.10))
            scale -= 0.09 * float(kick)
        return scale

    def _deform_atria(self, plan: BeatPlan, beat: Beat, t: float) -> tuple[np.ndarray, np.ndarray]:
        s = self._atrial_scale(plan, beat, t)
        out = []
        for name, frame, enlarge in (("ra_wall", self.model.ra_frame, self.pathology.ra_scale),
                                     ("la_wall", self.model.la_frame, self.pathology.la_scale)):
            v = self._static_positions[name]
            centre = frame.center.astype(np.float32)
            grow = np.float32(enlarge if self.layers.pathology else 1.0)
            if plan.atrial.pattern == AtrialPattern.FIBRILLATION:
                jitter = 0.4 * np.sin(t * 2 * np.pi * 7.0 + v[:, 0] * 0.3 + v[:, 2] * 0.2).astype(np.float32)
                out.append(centre + (v - centre) * (grow + 0.01 * jitter[:, None]))
            else:
                out.append(centre + (v - centre) * (np.float32(s) * grow))
        return out[0], out[1]

    def _atrial_colors(self, plan: BeatPlan, beat: Beat, t: float) -> tuple[np.ndarray, np.ndarray]:
        m = self.model
        result = []
        for name, times in (("ra_wall", plan.atrial.ra_times_ms), ("la_wall", plan.atrial.la_times_ms)):
            part = m.parts[name]
            colors = np.tile(part.color, (part.n_vertices, 1)).astype(np.float32)
            if not self.layers.excitation:
                result.append(colors)
                continue
            pattern = plan.atrial.pattern
            if pattern == AtrialPattern.FIBRILLATION:
                v = part.vertices
                phase = t * 2 * np.pi
                flicker = (np.sin(phase * 6.0 + v[:, 0] * 0.9 + v[:, 1] * 0.7 + v[:, 2] * 0.5)
                           + np.sin(phase * 4.3 - v[:, 0] * 0.5 + v[:, 2] * 1.1)
                           + np.sin(phase * 7.7 + v[:, 1] * 1.3 - v[:, 2] * 0.6))
                flicker = np.clip(0.5 + flicker / 3.0, 0.0, 1.0) ** 2
                colors[:, :3] = colors[:, :3] * (1.0 - 0.35 * flicker[:, None]) + YELLOW * (0.35 * flicker[:, None])
            elif pattern == AtrialPattern.FLUTTER:
                cycle_ms = 200.0
                phase = np.mod((t - beat.qrs_on) * 1000.0 - times, cycle_ms)
                front = np.exp(-0.5 * (phase / 12.0) ** 2) + np.exp(-0.5 * ((phase - cycle_ms) / 12.0) ** 2)
                colors[:, :3] = colors[:, :3] * (1.0 - front[:, None]) + YELLOW * front[:, None]
            elif pattern != AtrialPattern.NONE or (self.timeline and self.timeline.atrial_events):
                event = self.timeline.atrial_event_at(t) if self.timeline else None
                if event is not None:
                    start = event[0]
                    times = plan.atrial.ra_times_ms if name == "ra_wall" else plan.atrial.la_times_ms
                    if pattern in (AtrialPattern.NONE, AtrialPattern.RETROGRADE):
                        times = self.solver.atrial(AtrialPattern.SINUS, 100.0).ra_times_ms if name == "ra_wall" else self.solver.atrial(AtrialPattern.SINUS, 100.0).la_times_ms
                elif pattern == AtrialPattern.RETROGRADE:
                    start = beat.qrs_on + 0.06
                else:
                    start = beat.p_on if beat.p_on is not None else beat.qrs_on - 0.16
                dt = (t - start) * 1000.0 - times
                front = np.clip(np.exp(-0.5 * (dt / 10.0) ** 2) * (dt > -20.0), 0, 1)
                depol = ((dt > 0) & (dt < 180.0)).astype(np.float32)
                colors[:, :3] = colors[:, :3] * (1.0 - 0.25 * depol[:, None]) + DEPOLARIZED_TINT * (0.25 * depol[:, None])
                colors[:, :3] = colors[:, :3] * (1.0 - front[:, None]) + YELLOW * front[:, None]
            result.append(colors)
        return result[0], result[1]

    def _pulse(self, vertices: np.ndarray, plan: BeatPlan, t: float, amount: float) -> np.ndarray:
        ev = plan.events
        p = _smoothstep((t - ev.ao_open) / 0.08) * (1.0 - _smoothstep((t - ev.ao_close) / 0.15))
        if p <= 0:
            return vertices
        centre = vertices.mean(0)
        return (centre + (vertices - centre) * np.float32(1.0 + amount * p)).astype(np.float32)

    # --------------------------------------------------------------- Leitungszustand je Muster
    BLOCKED_BY_PATTERN = {
        Pattern.LBBB: ("lbb", "lbb_anterior", "lbb_posterior"), Pattern.RBBB: ("rbb",),
        Pattern.LAFB: ("lbb_anterior",), Pattern.LPFB: ("lbb_posterior",),
    }

    def conduction_state(self, plan: BeatPlan, t: float) -> tuple[dict, tuple, tuple]:
        """Impulsposition je Leitungssegment (0..1 in Zeichenrichtung, >1 = abgelaufen), Blöcke, Hinweise.

        Segmente: internodal (SA→AV), bachmann (SA→LA), av (AV-Knoten-Verzögerung), his, rbb, lbb,
        lbb_anterior, lbb_posterior, accessory (Vorhof→Kammer), focal (Fokus/Elektrode),
        retro_* (rückwärts, Fraktion läuft 1→0), flutter (Kreisphase), af (Flimmern).
        """
        beat = plan.beat
        pattern = plan.ventricular.pattern
        atrial = plan.atrial.pattern
        tl = self.timeline
        state: dict[str, float] = {}
        notes: list[str] = []
        blocked = tuple(plan.ventricular.blocked) if plan.ventricular.blocked else tuple(self.BLOCKED_BY_PATTERN.get(pattern, ()))
        if plan.ventricular.notes and plan.pattern_tier == "H":
            notes.extend(n for n in plan.ventricular.notes if "verzögert" in n or "Verzögerung" in n)
        av_blocked = bool(tl is not None and tl.atrial_events)

        def run(name: str, start: float, end: float, reverse: bool = False) -> None:
            if end <= start:
                return
            f = (t - start) / (end - start)
            if 0.0 <= f <= 1.0:
                state[name] = (1.0 - f) if reverse else f

        scenario = plan.scenario
        if scenario != RhythmScenario.AUTO:
            self._scenario_conduction(scenario, beat, t, state, run)
            return state, blocked, tuple(notes)
        # ---- Vorhöfe
        if atrial == AtrialPattern.FLUTTER:
            cycle = 1000.0 / float(np.clip(self.timeline.conduction_hints.get("atrial_frequency_hz", 5.0) or 5.0, 3.0, 7.0)) if tl else 200.0
            state["flutter"] = float(np.mod(t * 1000.0, cycle) / cycle)
            state["flutter_cycle_ms"] = cycle
            run("av", beat.qrs_on - 0.09, beat.qrs_on - 0.02)
        elif atrial == AtrialPattern.FIBRILLATION:
            state["af"] = 1.0
            run("av", beat.qrs_on - 0.07, beat.qrs_on - 0.02)
        elif av_blocked:
            event = tl.atrial_event_at(t)
            if event is not None:
                p_on, p_off = event
                run("internodal", p_on, p_off)
                run("bachmann", p_on + 0.01, p_off + 0.01)
                run("av", p_off, p_off + 0.12)  # läuft in den Knoten und erlischt dort
            blocked = blocked + ("av",)
        elif atrial == AtrialPattern.RETROGRADE:
            pass  # unten, nach der Kammer
        elif beat.has_p:
            run("internodal", beat.p_on, beat.p_off)
            run("bachmann", beat.p_on + 0.01, beat.p_off + 0.01)
            av_delay_end = beat.qrs_on - 0.03 if pattern != Pattern.WPW else beat.qrs_on - 0.005
            run("av", beat.p_off - 0.01, av_delay_end)
            if tl is not None and tl.conduction_hints.get("first_degree_av_block"):
                notes.append("AV-Überleitung verzögert (AV-Block I)")
        # ---- Kammer
        antegrade = pattern in (Pattern.NORMAL, Pattern.LBBB, Pattern.RBBB, Pattern.LAFB, Pattern.LPFB, Pattern.WPW)
        if antegrade:
            run("his", beat.qrs_on - 0.03, beat.qrs_on)
            for name in ("rbb", "lbb"):
                if name in blocked:
                    run(name + "_to_block", beat.qrs_on - 0.005, beat.qrs_on + 0.012)
                else:
                    run(name, beat.qrs_on - 0.005, beat.qrs_on + 0.03)
            for name in ("lbb_anterior", "lbb_posterior"):
                if "lbb" in blocked:
                    continue
                if name in blocked:
                    run(name + "_to_block", beat.qrs_on + 0.012, beat.qrs_on + 0.02)
                else:
                    run(name, beat.qrs_on + 0.012, beat.qrs_on + 0.04)
            if pattern == Pattern.WPW:
                start = beat.p_off - 0.02 if beat.has_p else beat.qrs_on - 0.06
                run("accessory", start, beat.qrs_on + 0.005)
        else:
            # Fokus / Elektrode: Ausbreitung aus dem Myokard, Leitungssystem wird rückwärts erreicht
            run("focal", beat.qrs_on, beat.qrs_on + 0.04)
            for name in ("lbb_anterior", "lbb_posterior", "rbb"):
                run("retro_" + name, beat.qrs_on + 0.02, beat.qrs_on + 0.06, reverse=True)
            run("retro_lbb", beat.qrs_on + 0.05, beat.qrs_on + 0.08, reverse=True)
            run("retro_his", beat.qrs_on + 0.08, beat.qrs_on + 0.10, reverse=True)
            if atrial == AtrialPattern.RETROGRADE and not av_blocked:
                run("retro_av", beat.qrs_on + 0.10, beat.qrs_on + 0.16, reverse=True)
                run("retro_internodal", beat.qrs_on + 0.16, beat.qrs_on + 0.24, reverse=True)
                run("retro_bachmann", beat.qrs_on + 0.16, beat.qrs_on + 0.24)
            elif av_blocked:
                run("retro_av", beat.qrs_on + 0.10, beat.qrs_on + 0.13, reverse=True)
        return state, blocked, tuple(notes)

    def _scenario_conduction(self, scenario: RhythmScenario, beat: Beat, t: float, state: dict, run) -> None:
        """Impulsverlauf der Sonderrhythmen (Ersatzrhythmen, Reentry-Tachykardien)."""
        q = beat.qrs_on
        if scenario == RhythmScenario.JUNCTIONAL_ESCAPE:
            run("his", q - 0.02, q + 0.005)
            for name in ("rbb", "lbb"):
                run(name, q - 0.005, q + 0.03)
            for name in ("lbb_anterior", "lbb_posterior"):
                run(name, q + 0.012, q + 0.04)
            run("retro_av", q - 0.02, q + 0.04, reverse=True)
            run("retro_internodal", q + 0.04, q + 0.12, reverse=True)
            run("retro_bachmann", q + 0.04, q + 0.12)
            state["origin_his"] = 1.0
        elif scenario == RhythmScenario.VENTRICULAR_ESCAPE:
            run("focal", q, q + 0.05)
            for name in ("lbb_anterior", "lbb_posterior", "rbb"):
                run("retro_" + name, q + 0.03, q + 0.08, reverse=True)
            run("retro_lbb", q + 0.07, q + 0.10, reverse=True)
            run("retro_his", q + 0.10, q + 0.13, reverse=True)
        elif scenario == RhythmScenario.AVNRT:
            cycle = max(beat.rr_prev or 0.4, 0.25)
            state["avn_loop"] = float(np.mod(t - q, cycle) / cycle)
            run("his", q - 0.02, q + 0.005)
            for name in ("rbb", "lbb"):
                run(name, q - 0.005, q + 0.03)
            for name in ("lbb_anterior", "lbb_posterior"):
                run(name, q + 0.012, q + 0.04)
            run("retro_internodal", q - 0.01, q + 0.07, reverse=True)
            run("retro_bachmann", q - 0.01, q + 0.07)
        elif scenario == RhythmScenario.AVRT_ORTHO:
            run("av", q - 0.08, q - 0.02)
            run("his", q - 0.02, q + 0.005)
            for name in ("rbb", "lbb"):
                run(name, q - 0.005, q + 0.03)
            for name in ("lbb_anterior", "lbb_posterior"):
                run(name, q + 0.012, q + 0.04)
            run("retro_accessory", q + 0.05, q + 0.11, reverse=True)
            run("retro_bachmann", q + 0.11, q + 0.19, reverse=True)
            run("retro_internodal", q + 0.11, q + 0.19, reverse=True)
            state["circuit_avrt"] = 1.0
        elif scenario == RhythmScenario.AVRT_ANTI:
            run("accessory", q - 0.06, q + 0.005)
            run("focal", q, q + 0.03)
            for name in ("lbb_anterior", "lbb_posterior", "rbb"):
                run("retro_" + name, q + 0.03, q + 0.07, reverse=True)
            run("retro_lbb", q + 0.06, q + 0.09, reverse=True)
            run("retro_his", q + 0.09, q + 0.11, reverse=True)
            run("retro_av", q + 0.11, q + 0.16, reverse=True)
            run("retro_internodal", q + 0.16, q + 0.24, reverse=True)
            run("retro_bachmann", q + 0.16, q + 0.24)
            state["circuit_avrt"] = 1.0
        elif scenario == RhythmScenario.VT_REENTRY:
            cycle = max(beat.rr_prev or 0.4, 0.25)
            state["vt_loop"] = float(np.mod(t - q, cycle) / cycle)
            run("focal", q, q + 0.04)
            for name in ("lbb_anterior", "lbb_posterior", "rbb"):
                run("retro_" + name, q + 0.03, q + 0.08, reverse=True)
            run("retro_lbb", q + 0.07, q + 0.10, reverse=True)
            run("retro_his", q + 0.10, q + 0.13, reverse=True)

    def _conduction_progress(self, beat: Beat, t: float) -> float:
        """0..1: 0 SA-Knoten, 0.5 AV-Knoten erreicht, 0.7 His, 1.0 Schenkel/Purkinje."""
        if beat.has_p:
            if t < beat.p_on:
                return 0.0
            if t < beat.p_off:
                return 0.5 * (t - beat.p_on) / max(beat.p_off - beat.p_on, 1e-3)
            if t < beat.qrs_on:
                return 0.5 + 0.5 * (t - beat.p_off) / max(beat.qrs_on - beat.p_off, 1e-3)
        else:
            if t < beat.qrs_on - 0.03:
                return 0.0
            if t < beat.qrs_on:
                return 0.6 + 0.4 * (t - (beat.qrs_on - 0.03)) / 0.03
        if t < beat.qrs_on + 0.03:
            return 1.0
        return 0.0

    def _instant_vector(self, tl: BeatTimeline, t: float) -> Optional[np.ndarray]:
        from .timeline import KORS
        names = list(tl.lead_names)
        try:
            cols = [names.index(n) for n in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]
        except ValueError:
            return None
        k = int(t * tl.fs)
        if k < 0 or k >= len(tl.signal):
            return None
        a = max(0, k - 2)
        sample = tl.signal[a:k + 3, :][:, cols].mean(0)
        xyz = KORS @ sample
        return vcg_to_model(xyz)

    def _phase_label(self, plan: BeatPlan, t: float) -> str:
        ev = plan.events
        b = plan.beat
        if b.has_p and ev.atrial_contraction_on is not None and ev.atrial_contraction_on <= t < ev.atrial_contraction_off:
            return "Vorhofkontraktion"
        if b.qrs_on <= t < ev.av_close:
            return "Kammererregung"
        if ev.av_close <= t < ev.ao_open:
            return "Isovolumetrische Kontraktion"
        if ev.ao_open <= t < ev.ao_close:
            return "Ejektion"
        if ev.ao_close <= t < ev.av_open:
            return "Isovolumetrische Relaxation"
        if ev.av_open <= t < ev.av_open + 0.25:
            return "Frühe Füllung"
        if b.has_p and b.p_on <= t < b.qrs_on:
            return "Vorhoferregung"
        return "Diastase"
