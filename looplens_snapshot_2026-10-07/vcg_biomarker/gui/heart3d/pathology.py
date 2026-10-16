"""Pathologie-Schätzung für die Illustration: Ischämie-/Narbenregionen, Hypertrophie, Vorhofvergrößerung.

Grundlage ist der Verletzungsstrom: bei transmuraler Ischämie zeigt der ST-Vektor
(hier aus den ST-Abweichungen der acht unabhängigen Ableitungen per Kors-Matrix)
zur Läsion; bei subendokardialer Ischämie zeigt er vom Endokard weg (zum Cavum),
weshalb dort nur eine diffuse endokardiale Zone gezeigt wird. Verlorene Initialkräfte
(pathologische Q-Zacken) zeigen von der Narbe weg. Beides sind Näherungen mit
Literaturgenauigkeit im Bereich der Hauptterritorien, keine Messung.

Stufen: K = Kriterium aus der Laboranalyse (``classical_criteria``), E = eigene
Schnellschätzung aus dem Signal (ohne Analyse), H = Hypothese.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .activation import vcg_to_model
from .model import CHAMBER_LV, CHAMBER_RV, HeartModel
from .timeline import KORS, Beat, BeatTimeline

# AHA-Segment -> Versorgungsgebiet (Cerqueira 2002, vereinfachte Standardzuordnung)
SEGMENT_ARTERY = {
    1: "LAD", 2: "LAD", 7: "LAD", 8: "LAD", 13: "LAD", 14: "LAD", 17: "LAD",
    3: "RCA", 4: "RCA", 9: "RCA", 10: "RCA", 15: "RCA",
    5: "LCx", 6: "LCx", 11: "LCx", 12: "LCx", 16: "LCx",
    101: "RCA", 102: "RCA", 103: "RCA", 104: "RCA", 105: "RCA",
}
SEGMENT_REGION = {
    1: "anterior", 2: "anteroseptal", 3: "inferoseptal", 4: "inferior", 5: "inferolateral", 6: "anterolateral",
    7: "anterior", 8: "anteroseptal", 9: "inferoseptal", 10: "inferior", 11: "inferolateral", 12: "anterolateral",
    13: "apikal-anterior", 14: "apikal-septal", 15: "apikal-inferior", 16: "apikal-lateral", 17: "Apex",
    101: "RV anterior", 102: "RV lateral", 103: "RV inferior", 104: "RV-Ausflusstrakt", 105: "RV-Apex",
}
# Ableitungsgruppen -> Territorium (klassische Zuordnung, Wagner 2009 / ESC STEMI 2017)
LEAD_TERRITORIES = {
    "inferior (II, III, aVF)": ("II", "III", "aVF"),
    "anterior/septal (V1–V4)": ("V1", "V2", "V3", "V4"),
    "lateral (I, aVL, V5, V6)": ("I", "aVL", "V5", "V6"),
}
TERRITORY_ARTERY = {"inferior (II, III, aVF)": "RCA (oder LCx)", "anterior/septal (V1–V4)": "LAD", "lateral (I, aVL, V5, V6)": "LCx / Diagonalast"}


@dataclass
class LeadMeasures:
    """Mediane Ableitungsmaße über die dominanten Schläge (Schnellschätzung)."""
    st_j: dict[str, float] = field(default_factory=dict)  # mV, J-Punkt
    st_60: dict[str, float] = field(default_factory=dict)  # mV, J + 60 ms
    t_peak: dict[str, float] = field(default_factory=dict)  # mV, vorzeichenbehaftet (größter Betrag)
    r_height: dict[str, float] = field(default_factory=dict)
    qrs_ptp: dict[str, float] = field(default_factory=dict)  # Spitze-Spitze im QRS
    q_depth: dict[str, float] = field(default_factory=dict)  # positiv = Tiefe
    q_duration_ms: dict[str, float] = field(default_factory=dict)
    st_vector: Optional[np.ndarray] = None  # VCG-Raum (X links, Y inferior, Z posterior), mV
    t_vector: Optional[np.ndarray] = None
    initial_vector: Optional[np.ndarray] = None  # erste 40 ms des QRS
    qtc_ms: Optional[float] = None
    n_beats: int = 0


@dataclass
class PathologyEstimate:
    ischemia: np.ndarray  # 0..1 je Wandvertex (akute Ischämie/Verletzung)
    scar: np.ndarray  # 0..1 je Wandvertex (Narbe / verlorene Kräfte)
    wall_offset: np.ndarray  # (N, 3) mm, statische Formänderung (Hypertrophie)
    la_scale: float = 1.0
    ra_scale: float = 1.0
    effusion: bool = False
    findings: list[tuple[str, str, str]] = field(default_factory=list)  # (Stufe, Titel, Erläuterung)
    territory: str = ""
    culprit: str = ""
    lead_territories: list[str] = field(default_factory=list)
    measures: Optional[LeadMeasures] = None
    ischemia_kind: str = ""  # transmural | subendocardial | ""

    @property
    def any_visible(self) -> bool:
        return bool(self.ischemia.max() > 0.05 or self.scar.max() > 0.05 or np.abs(self.wall_offset).max() > 0.1
                    or self.la_scale != 1.0 or self.ra_scale != 1.0 or self.effusion)


def empty_estimate(model: HeartModel) -> PathologyEstimate:
    n = model.wall.n_vertices
    return PathologyEstimate(np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros((n, 3), np.float32))


# ------------------------------------------------------------------ Ableitungsmaße
def lead_measures_from_timeline(tl: BeatTimeline) -> LeadMeasures:
    out = LeadMeasures()
    if tl.signal is None or not tl.beats:
        return out
    names = list(tl.lead_names)
    fs = tl.fs
    sig = tl.signal
    beats = [b for b in tl.beats if b.beat_type in ("N", "S") and b.accepted]
    if len(beats) < 1:
        beats = list(tl.beats)
    per_lead: dict[str, dict[str, list[float]]] = {n: {"j": [], "st60": [], "t": [], "r": [], "ptp": [], "q": [], "qd": []} for n in names}
    st8, t8, i8 = [], [], []
    qt_list = []
    for b in beats[:40]:
        a, q_off = int(b.qrs_on * fs), int(b.qrs_off * fs)
        if b.source == "estimated":
            # Ohne Delineation liegt das QRS-Ende unsicher: J-Punkt konservativ 70 ms nach der R-Zacke
            q_off = int((b.r_time + 0.07) * fs)
        base_a, base_b = max(0, a - int(0.07 * fs)), max(1, a - int(0.02 * fs))
        if base_b <= base_a or q_off + int(0.08 * fs) >= len(sig) or a < 0:
            continue
        base = sig[base_a:base_b].mean(0)
        x = sig - base
        j = x[q_off]
        k60 = min(len(x) - 1, q_off + int(0.06 * fs))
        st60 = x[q_off:k60 + 1].mean(0) if k60 > q_off else x[q_off]
        t_a, t_b = int((b.t_on or (b.qrs_off + 0.06)) * fs), int((b.t_end or (b.qrs_off + 0.3)) * fs)
        t_b = min(t_b, len(x) - 1)
        if t_b <= t_a:
            continue
        t_seg = x[t_a:t_b]
        t_peak = np.where(np.abs(t_seg.max(0)) >= np.abs(t_seg.min(0)), t_seg.max(0), t_seg.min(0))
        qrs = x[a:q_off]
        r = qrs.max(0)
        ptp = qrs.max(0) - qrs.min(0)
        init = x[a:min(q_off, a + int(0.04 * fs))]
        if len(init) == 0:
            continue
        for li, n in enumerate(names):
            per_lead[n]["j"].append(float(j[li]))
            per_lead[n]["st60"].append(float(st60[li]))
            per_lead[n]["t"].append(float(t_peak[li]))
            per_lead[n]["r"].append(float(r[li]))
            per_lead[n]["ptp"].append(float(ptp[li]))
            # Q-Zacke: negative Auslenkung vor dem ersten positiven Ausschlag > 0,1 mV
            lead = qrs[:, li]
            pos = np.where(lead > 0.1)[0]
            stop = int(pos[0]) if pos.size else len(lead)
            q = lead[:stop]
            depth = float(-q.min()) if q.size and q.min() < 0 else 0.0
            per_lead[n]["q"].append(depth)
            per_lead[n]["qd"].append(float((q < -0.02).sum() / fs * 1000.0) if q.size else 0.0)
        try:
            cols = [names.index(n) for n in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]
            st8.append(KORS @ st60[cols])
            t8.append(KORS @ t_seg[:, cols].mean(0))
            i8.append(KORS @ init[:, cols].mean(0))
        except ValueError:
            pass
        if b.t_end is not None and b.rr_prev and b.rr_prev > 0.3:
            qt_list.append((b.t_end - b.qrs_on) / np.sqrt(b.rr_prev) * 1000.0)
    for n in names:
        if per_lead[n]["j"]:
            out.st_j[n] = float(np.median(per_lead[n]["j"]))
            out.st_60[n] = float(np.median(per_lead[n]["st60"]))
            out.t_peak[n] = float(np.median(per_lead[n]["t"]))
            out.r_height[n] = float(np.median(per_lead[n]["r"]))
            out.qrs_ptp[n] = float(np.median(per_lead[n]["ptp"]))
            out.q_depth[n] = float(np.median(per_lead[n]["q"]))
            out.q_duration_ms[n] = float(np.median(per_lead[n]["qd"]))
    if st8:
        out.st_vector = np.median(np.asarray(st8), axis=0)
        out.t_vector = np.median(np.asarray(t8), axis=0)
        out.initial_vector = np.median(np.asarray(i8), axis=0)
    if qt_list:
        out.qtc_ms = float(np.median(qt_list))
    out.n_beats = len(beats)
    return out


def _st_elevation_threshold(lead: str) -> float:
    return 0.15 if lead in ("V2", "V3") else 0.1


def lead_based_territories(m: LeadMeasures) -> tuple[list[str], list[str]]:
    """(Territorien mit ST-Hebung in ≥ 2 Ableitungen, Territorien mit ST-Senkung)."""
    elevated, depressed = [], []
    for name, leads in LEAD_TERRITORIES.items():
        up = sum(1 for l in leads if m.st_j.get(l, 0.0) >= _st_elevation_threshold(l))
        down = sum(1 for l in leads if m.st_60.get(l, 0.0) <= -0.05)
        if up >= 2:
            elevated.append(name)
        if down >= 2:
            depressed.append(name)
    return elevated, depressed


# ------------------------------------------------------------------ Regionen
def _cone_weights(model: HeartModel, direction: np.ndarray, half_angle_deg: float, centre: Optional[np.ndarray] = None) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.zeros(model.wall.n_vertices, np.float32)
    d = d / n
    if centre is None:
        centre = 0.5 * (model.lv_frame.center + model.rv_frame.center)
    rel = model.wall.vertices.astype(np.float64) - centre
    cos = (rel @ d) / (np.linalg.norm(rel, axis=1) + 1e-6)
    c0 = np.cos(np.radians(half_angle_deg))
    w = np.clip((cos - c0) / (1.0 - c0), 0.0, 1.0)
    return (w * w * (3 - 2 * w)).astype(np.float32)


def _lesion_weights(model: HeartModel, direction: np.ndarray, half_angle_deg: float) -> np.ndarray:
    """Läsionszone um die Vektorrichtung: LV-zentriert auf der LV-Wand; RV nur bei deutlich rechts gerichtetem Vektor."""
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.zeros(model.wall.n_vertices, np.float32)
    d = d / n
    lv = (model.wall_chamber == CHAMBER_LV).astype(np.float32)
    w = _cone_weights(model, d, half_angle_deg, centre=model.lv_frame.center) * lv
    if d[0] < -0.35:  # nach rechts: rechtsventrikuläre Beteiligung
        rv = (model.wall_chamber == CHAMBER_RV).astype(np.float32)
        w = np.maximum(w, _cone_weights(model, d, half_angle_deg, centre=model.rv_frame.center) * rv * float(np.clip((-d[0] - 0.35) / 0.4, 0.0, 1.0)))
    return w


def region_summary(model: HeartModel, weights: np.ndarray, threshold: float = 0.45) -> tuple[str, str]:
    """(Regionsbeschreibung, Gefäß) aus den betroffenen AHA-Segmenten."""
    sel = weights >= threshold
    if not sel.any():
        return "", ""
    segs, counts = np.unique(model.wall_segment[sel], return_counts=True)
    order = np.argsort(counts)[::-1]
    names = []
    votes: dict[str, float] = {}
    for k in order[:4]:
        seg = int(segs[k])
        names.append(SEGMENT_REGION.get(seg, str(seg)))
        votes[SEGMENT_ARTERY.get(seg, "?")] = votes.get(SEGMENT_ARTERY.get(seg, "?"), 0.0) + float(counts[k])
    culprit = max(votes, key=votes.get) if votes else ""
    return ", ".join(dict.fromkeys(names)), culprit


# ------------------------------------------------------------------ Hauptschätzung
def _crit(payload_criteria: dict, key: str) -> Optional[bool]:
    item = payload_criteria.get(key)
    if isinstance(item, dict) and item.get("positive") is not None:
        return bool(item["positive"])
    return None


def estimate_pathology(model: HeartModel, tl: BeatTimeline, payload: Optional[dict] = None) -> PathologyEstimate:
    est = empty_estimate(model)
    measures = lead_measures_from_timeline(tl)
    est.measures = measures
    criteria = (payload or {}).get("classical_criteria", {}) or {}
    has_analysis = bool(criteria)
    tier_signal = "E"
    findings = est.findings
    if measures.n_beats == 0:
        return est
    elevated, depressed = lead_based_territories(measures)
    est.lead_territories = elevated

    # ---- Ischämie: transmural (ST-Hebung) -------------------------------------------
    st_vec = measures.st_vector
    st_mag = float(np.linalg.norm(st_vec)) if st_vec is not None else 0.0
    crit_elev = _crit(criteria, "st_elevation_contiguous")
    crit_dep = _crit(criteria, "st_depression_contiguous")
    transmural = (crit_elev is True) if has_analysis and crit_elev is not None else bool(elevated)
    if transmural and st_vec is not None and st_mag >= 0.05:
        # Ausdehnung nach dem Aldrich-Score (Aldrich 1988): Anteil des LV im Risikogebiet aus ST-Hebungen
        area_pct, aldrich_note = aldrich_area_at_risk(measures, elevated)
        w, radius = lesion_by_extent(model, vcg_to_model(st_vec), area_pct)
        # Verletzungsstrom ist transmural, epikardial betont
        w *= np.where(model.wall_endocardial, 0.8, 1.0).astype(np.float32)
        est.ischemia = np.maximum(est.ischemia, w * float(np.clip(0.4 + st_mag / 0.25, 0.0, 1.0)))
        est.ischemia_kind = "transmural"
        region, culprit = region_summary(model, w)
        est.territory, est.culprit = region, culprit
        tier = "K" if crit_elev is True else tier_signal
        findings.append((tier, "ST-Hebung: transmurale Ischämie/Verletzung möglich",
                         f"ST-Vektor {st_mag * 10:.1f} mm (Kors, J+60 ms) zeigt zur Läsion → {region or 'Region unbestimmt'}"
                         f"{' · Gefäß nach AHA-Zuordnung: ' + culprit if culprit else ''}"
                         f" · Ausdehnung {aldrich_note} → Zonenradius {radius:.0f} mm um den Vektor-Zielpunkt"
                         f"{' · Zone liegt überwiegend septal: im 2D-Schnitt, mit Schnittansicht oder Wanddurchsichtigkeit sichtbar' if _mostly_septal(model, w) else ''}"))
        if _crit(criteria, "rbbb_pattern") is True or _crit(criteria, "lbbb_pattern") is True:
            findings.append(("K", "Schenkelblock erkannt", "ST-Abweichungen können sekundär (diskordant) sein; Sgarbossa-Regeln werden hier nicht angewendet – Zone mit Vorbehalt"))
        if elevated:
            lead_art = ", ".join(f"{t} → {TERRITORY_ARTERY[t]}" for t in elevated)
            findings.append((tier_signal, "Ableitungsgruppen mit ST-Hebung", lead_art))
            agree = any(TERRITORY_ARTERY[t].startswith(culprit) for t in elevated) if culprit else False
            findings.append(("H", "Abgleich Vektor ↔ Ableitungen", "stimmig" if agree else "abweichend – Region mit Vorsicht lesen"))
        if depressed and st_vec is not None:
            findings.append((tier_signal, "Spiegelbildliche ST-Senkung", ", ".join(depressed)))
    elif ((crit_dep is True) if has_analysis and crit_dep is not None else bool(depressed)) and st_vec is not None:
        # Subendokardial: diffuse endokardiale LV-Zone (der ST-Vektor zeigt zum Cavum, nicht zur Region)
        w = np.where(model.wall_endocardial, 1.0, np.where(model.wall_near_cavity, 0.6, 0.1)).astype(np.float32)
        w *= (model.wall_chamber == CHAMBER_LV).astype(np.float32)
        strain = lvh_voltage(measures, criteria, has_analysis)
        strength = float(np.clip(0.35 + st_mag / 0.2, 0.0, 0.85)) * (0.5 if strain else 1.0)
        est.ischemia = np.maximum(est.ischemia, w * strength)
        est.ischemia_kind = "subendocardial"
        tier = "K" if crit_dep is True else tier_signal
        findings.append((tier, "ST-Senkung: subendokardiale Ischämie möglich",
                         f"ST-Vektor {st_mag * 10:.1f} mm zeigt zum Cavum; Zone diffus endokardial (LV), Ort nicht bestimmbar"
                         f"{' · Senkung in ' + ', '.join(depressed) if depressed else ''}"
                         f"{' · bei LVH-Voltage als Schädigungsmuster (Strain) nicht unterscheidbar' if strain else ''}"))
    # Wellens / de Winter aus der Analyse
    if _crit(criteria, "wellens_ecg_pattern") is True:
        w = _lesion_weights(model, np.array([0.3, -1.0, -0.2]), 40.0)  # anterior (LAD)
        est.ischemia = np.maximum(est.ischemia, 0.5 * w)
        findings.append(("K", "Wellens-Muster (V2–V4)", "kritische proximale LAD-Stenose möglich; anteriore Zone markiert"))
    if _crit(criteria, "de_winter_ecg_pattern") is True:
        w = _lesion_weights(model, np.array([0.3, -1.0, -0.2]), 40.0)
        est.ischemia = np.maximum(est.ischemia, 0.6 * w)
        findings.append(("K", "de-Winter-Muster", "LAD-Verschluss-Äquivalent; anteriore Zone markiert"))
    if _crit(criteria, "posterior_mi_pattern") is True:
        w = _lesion_weights(model, np.array([0.2, 1.0, -0.3]), 40.0)
        est.ischemia = np.maximum(est.ischemia, 0.6 * w)
        findings.append(("K", "Posteriores Infarktmuster (V1–V3)", "inferolaterale/posteriore Zone markiert (LCx/RCA)"))
    # T-Inversion ohne ST-Hebung: leichte Markierung in Richtung des invertierten T-Vektors
    crit_tinv = _crit(criteria, "t_inversion_contiguous")
    if crit_tinv is True and measures.t_vector is not None and est.ischemia.max() < 0.1:
        w = _lesion_weights(model, -vcg_to_model(measures.t_vector), 40.0)
        est.ischemia = np.maximum(est.ischemia, 0.35 * w)
        findings.append(("K", "T-Inversion in zusammenhängenden Ableitungen", f"unspezifisch; Zone {region_summary(model, w)[0] or '–'} schwach markiert"))

    # ---- Narbe: verlorene Initialkräfte ---------------------------------------------
    crit_q = _crit(criteria, "pathological_q_waves")
    q_leads = [l for l in measures.q_depth if measures.q_depth[l] >= 0.1 and measures.q_duration_ms.get(l, 0) >= 30
               and l not in ("aVR", "III")]
    if "V1" in q_leads and not (measures.q_depth.get("V2", 0.0) >= 0.1 or measures.q_depth.get("V3", 0.0) >= 0.1):
        q_leads.remove("V1")  # isoliertes Q in V1 ist normal
    q_groups = [name for name, leads in LEAD_TERRITORIES.items() if sum(1 for l in leads if l in q_leads) >= 2]
    if crit_q is True or (q_groups and not (has_analysis and crit_q is False)):
        direction = None
        if measures.initial_vector is not None and np.linalg.norm(measures.initial_vector) > 0.02:
            direction = -vcg_to_model(measures.initial_vector)
        elif q_groups:
            direction = {"inferior (II, III, aVF)": np.array([0.0, 0.2, -1.0]), "anterior/septal (V1–V4)": np.array([0.2, -1.0, 0.0]),
                         "lateral (I, aVL, V5, V6)": np.array([1.0, 0.0, 0.2])}[q_groups[0]]
        if direction is not None:
            w = _lesion_weights(model, direction, 34.0)
            est.scar = np.maximum(est.scar, w)
            region, culprit = region_summary(model, w)
            findings.append(("K" if crit_q is True else tier_signal, "Pathologische Q-Zacken: Narbe möglich",
                             f"Verlorene Initialkräfte (erste 40 ms) zeigen von der Narbe weg → {region or 'unbestimmt'}"
                             f"{' · Q in ' + ', '.join(q_leads) if q_leads else ''} (grobe Näherung)"))
    if _crit(criteria, "qs_complexes") is True and est.scar.max() < 0.1:
        w = _lesion_weights(model, np.array([0.3, -1.0, -0.3]), 34.0)
        est.scar = np.maximum(est.scar, 0.7 * w)
        findings.append(("K", "QS-Komplexe präkordial", "anteriore Narbe möglich"))
    if _crit(criteria, "poor_r_wave_progression") is True and est.scar.max() < 0.1:
        findings.append(("K", "Schlechte R-Progression", "anteriorer Infarkt, LVH oder Lagevariante – nicht dargestellt"))

    # ---- Hypertrophie / Vorhöfe -------------------------------------------------------
    lvh = lvh_voltage(measures, criteria, has_analysis)
    if lvh:
        est.wall_offset += _thickening(model, CHAMBER_LV, 3.0, 1.2)
        findings.append(("K" if has_analysis else tier_signal, "Voltage-Kriterium für LV-Hypertrophie",
                         "LV-Wand verdickt dargestellt (Endokard nach innen, Epikard leicht nach außen); Voltage allein ist wenig spezifisch"))
    rvh = _crit(criteria, "right_ventricular_hypertrophy") is True or (
        _crit(criteria, "right_axis_deviation") is True and measures.r_height.get("V1", 0.0) >= 0.7)
    if rvh:
        est.wall_offset += _thickening(model, CHAMBER_RV, 2.5, 1.0)
        findings.append(("K", "Hinweise auf RV-Hypertrophie", "RV-Wand verdickt dargestellt"))
    if _crit(criteria, "bifid_p_wave") is True or (not has_analysis and measures.t_peak and _p_terminal_negative(tl)):
        est.la_scale = 1.15
        findings.append(("K" if has_analysis else tier_signal, "P mitrale / negatives terminales P in V1", "linker Vorhof vergrößert dargestellt"))
    p_ii = _p_amplitude(tl, "II")
    if p_ii is not None and p_ii >= 0.25:
        est.ra_scale = 1.15
        findings.append((tier_signal, "P pulmonale (P in II ≥ 0,25 mV)", "rechter Vorhof vergrößert dargestellt"))
    if _crit(criteria, "low_voltage_qrs") is True or (not has_analysis and _low_voltage(measures)):
        est.effusion = True
        findings.append(("K" if has_analysis else tier_signal, "Niedervoltage",
                         "unspezifisch (Erguss, Adipositas, Emphysem, Amyloid); Perikardhülle angedeutet"))
    if _crit(criteria, "hyperkalemia_ecg_heuristic") is True or _crit(criteria, "peaked_t_wave_heuristic") is True:
        findings.append(("K", "Spitze T-Wellen / Hyperkaliämie-Heuristik", "keine Ortsdarstellung; Repolarisation folgt dem T-Fenster"))
    if measures.qtc_ms is not None and measures.qtc_ms >= 470:
        findings.append((tier_signal, f"QTc (Bazett) {measures.qtc_ms:.0f} ms verlängert", "Repolarisationswelle entsprechend spät"))
    return est


def lvh_voltage(measures: LeadMeasures, criteria: dict, has_analysis: bool) -> bool:
    if any(_crit(criteria, k) is True for k in ("sokolow_lyon", "cornell_voltage", "cornell_product", "peguero_lo_presti")):
        return True
    if has_analysis:
        return False
    # Schnellschätzung ohne S-Tiefen: nur R-Höhen (Sokolow-Anteil, aVL-Regel)
    return max(measures.r_height.get("V5", 0.0), measures.r_height.get("V6", 0.0)) >= 2.6 or measures.r_height.get("aVL", 0.0) >= 1.1


def _mostly_septal(model: HeartModel, weights: np.ndarray) -> bool:
    sel = weights >= 0.45
    return bool(sel.any() and model.wall_septal[sel].mean() > 0.5)


def zone_centroid(model: HeartModel, weights: np.ndarray, threshold: float = 0.45) -> Optional[np.ndarray]:
    sel = weights >= threshold
    if not sel.any():
        return None
    return model.wall.vertices[sel].mean(0)


def aldrich_area_at_risk(m: LeadMeasures, elevated: list[str]) -> tuple[float, str]:
    """Prozent des LV im Risikogebiet nach Aldrich 1988 (anterior: Zahl der Ableitungen mit ST-Hebung; inferior: Summe der ST-Hebung).

    Anterior: %LV = 3·[1,5·(n Ableitungen mit ST-Hebung) − 0,4]; inferior: %LV = 3·[0,6·ΣST(II, III, aVF in mm) + 2,0].
    """
    n_up = sum(1 for l, v in m.st_j.items() if v >= _st_elevation_threshold(l) and l != "aVR")
    inferior_sum_mm = sum(max(0.0, m.st_j.get(l, 0.0)) * 10.0 for l in ("II", "III", "aVF"))
    if any(t.startswith("inferior") for t in elevated) and not any(t.startswith("anterior") for t in elevated):
        pct = 3.0 * (0.6 * inferior_sum_mm + 2.0)
        note = f"{pct:.0f} % des LV (Aldrich inferior, ΣST II/III/aVF {inferior_sum_mm:.1f} mm)"
    else:
        pct = 3.0 * (1.5 * n_up - 0.4)
        note = f"{pct:.0f} % des LV (Aldrich anterior, {n_up} Ableitungen mit ST-Hebung)"
    return float(np.clip(pct, 5.0, 60.0)), note


def fit_half_angle(model: HeartModel, direction: np.ndarray, target_pct: float) -> float:
    """Öffnungswinkel des Läsionskegels so wählen, dass der LV-Anteil der Zone dem Zielprozentsatz entspricht."""
    lv_n = max(int((model.wall_chamber == CHAMBER_LV).sum()), 1)
    lo, hi = 12.0, 85.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        w = _lesion_weights(model, direction, mid)
        frac = 100.0 * float(((w >= 0.5) & (model.wall_chamber == CHAMBER_LV)).sum()) / lv_n
        if frac < target_pct:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lesion_by_extent(model: HeartModel, direction: np.ndarray, target_pct: float) -> tuple[np.ndarray, float]:
    """Zone um den Wandpunkt in Vektorrichtung, entlang der Wand gewachsen, bis der LV-Anteil dem Zielprozentsatz entspricht.

    Das Wachstum entlang der Wand (statt eines Kegels vom Zentrum) lässt das Gebiet wie ein Versorgungsgebiet um den
    Apex herumreichen; der Rand ist über 8 mm weich.
    """
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-9
    v = model.wall.vertices.astype(np.float64)
    lv = model.wall_chamber == CHAMBER_LV
    rel = v - model.lv_frame.center
    cos = (rel @ d) / (np.linalg.norm(rel, axis=1) + 1e-6)
    candidates = np.where(lv & ~model.wall_endocardial)[0]
    if candidates.size == 0:
        candidates = np.where(lv)[0]
    seed = int(candidates[np.argmax(cos[candidates])])
    dist = np.linalg.norm(v - v[seed], axis=1)
    # RV-Beteiligung nur bei deutlich rechts gerichtetem Vektor
    allowed = lv | ((model.wall_chamber == CHAMBER_RV) & (d[0] < -0.35))
    lv_n = max(int(lv.sum()), 1)
    lo, hi = 5.0, 120.0
    for _ in range(20):
        r = 0.5 * (lo + hi)
        frac = 100.0 * float((lv & (dist <= r)).sum()) / lv_n
        if frac < target_pct:
            lo = r
        else:
            hi = r
    r = 0.5 * (lo + hi)
    w = np.clip((r + 4.0 - dist) / 8.0, 0.0, 1.0) * allowed
    return (w * w * (3 - 2 * w)).astype(np.float32), r


def zone_direction_error_deg(model: HeartModel, weights: np.ndarray, st_vec_vcg: np.ndarray) -> float:
    """Winkel zwischen ST-Vektor und Richtung zum Schwerpunkt der Zone (Konsistenzprüfung)."""
    c = zone_centroid(model, weights)
    if c is None:
        return float("nan")
    d = c - model.lv_frame.center
    v = vcg_to_model(st_vec_vcg)
    cos = float(np.dot(d, v) / (np.linalg.norm(d) * np.linalg.norm(v) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _thickening(model: HeartModel, chamber: int, endo_mm: float, epi_mm: float) -> np.ndarray:
    frame = model.lv_frame if chamber == CHAMBER_LV else model.rv_frame
    v = model.wall.vertices.astype(np.float64)
    rel = v - frame.center
    proj = rel @ frame.axis
    radial = rel - np.outer(proj, frame.axis)
    r = np.linalg.norm(radial, axis=1) + 1e-6
    unit = radial / r[:, None]
    amount = np.where(model.wall_endocardial, -endo_mm, np.where(model.wall_near_cavity, -endo_mm * 0.4, epi_mm))
    amount = amount * (model.wall_chamber == chamber)
    return (unit * amount[:, None]).astype(np.float32)


def _p_amplitude(tl: BeatTimeline, lead: str) -> Optional[float]:
    if tl.signal is None or lead not in tl.lead_names:
        return None
    li = list(tl.lead_names).index(lead)
    vals = []
    for b in tl.beats[:40]:
        if not b.has_p:
            continue
        a, c = int(b.p_on * tl.fs), int(b.p_off * tl.fs)
        if a < 0 or c > len(tl.signal):
            continue
        base = tl.signal[max(0, a - int(0.03 * tl.fs)):max(1, a), li].mean() if a > 0 else 0.0
        if c > a:
            vals.append(float(tl.signal[a:c, li].max() - base))
    return float(np.median(vals)) if len(vals) >= 3 else None


def _p_terminal_negative(tl: BeatTimeline) -> bool:
    if tl.signal is None or "V1" not in tl.lead_names:
        return False
    li = list(tl.lead_names).index("V1")
    areas = []
    for b in tl.beats[:40]:
        if not b.has_p:
            continue
        a, c = int(b.p_on * tl.fs), int(b.p_off * tl.fs)
        if c - a < 4 or a < 0 or c > len(tl.signal):
            continue
        seg = tl.signal[a:c, li] - tl.signal[max(0, a - int(0.03 * tl.fs)):max(1, a), li].mean()
        half = seg[len(seg) // 2:]
        neg = half[half < 0]
        areas.append(float(-neg.sum() / tl.fs * 1000.0))  # mV·ms
    return bool(areas) and float(np.median(areas)) >= 40.0  # ≈ 1 mm·40 ms (Morris)


def _low_voltage(m: LeadMeasures) -> bool:
    limb = [m.qrs_ptp.get(l, 1.0) for l in ("I", "II", "III", "aVR", "aVL", "aVF")]
    prec = [m.qrs_ptp.get(l, 1.0) for l in ("V1", "V2", "V3", "V4", "V5", "V6")]
    return (max(limb) < 0.5) or (max(prec) < 1.0)
