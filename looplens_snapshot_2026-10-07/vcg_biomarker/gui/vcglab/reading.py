"""Musterlesung der Schleifen: regelbasierte Deutung der Maße mit Stufe und Quelle.

Stufen: K = klassisches Kriterium des Hauptprojekts (``ClassicalECGCriteria``), E = vektorkardio-
graphische Regel mit benannter Literatur, H = Hypothese/Heuristik (Schule der visuellen VCG-Lesung,
ohne validierte Schwelle). Die Lesung ergänzt den Laborbefund, sie ersetzt ihn nicht.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .measures import BeatMeasures, INSCRIPTION_TEXT, NORMAL_INSCRIPTION, azimuth_deg, jt_distance, BeatBounds


@dataclass
class Reading:
    tier: str  # K | E | H | i (Hinweis)
    title: str
    detail: str
    refs: str = ""
    weight: float = 1.0  # 0..1 Stärke der Evidenz für die Sortierung

    DEFAULT_TIER_COLORS = {"K": "#1f97a4", "E": "#e0a53d", "H": "#c792ea", "i": "#8fb9bd"}

    def html(self, tier_colors: Optional[dict] = None, muted: str = "#8fb9bd") -> str:
        color = (tier_colors or self.DEFAULT_TIER_COLORS).get(self.tier, muted)
        ref = f" <span style='color:{muted}'>[{self.refs}]</span>" if self.refs else ""
        return f"<span style='color:{color}; font-weight:700'>[{self.tier}]</span> <b>{self.title}</b> – {self.detail}{ref}"


@dataclass
class ReadingResult:
    readings: list[Reading] = field(default_factory=list)
    territory: Optional[str] = None
    summary: str = ""


def _crit(criteria: dict, key: str) -> Optional[bool]:
    """positive-Flag eines klassischen Kriteriums (``to_dict``: criterion_id, value, positive, evaluable)."""
    c = criteria.get(key)
    if not isinstance(c, dict) or not c.get("evaluable", True):
        return None
    pos = c.get("positive")
    return None if pos is None else bool(pos)


def _vec_dir(v: Optional[np.ndarray]) -> str:
    if v is None or np.linalg.norm(v) < 1e-9:
        return "–"
    parts = []
    x, y, z = v / np.linalg.norm(v)
    if abs(x) > 0.3:
        parts.append("links" if x > 0 else "rechts")
    if abs(y) > 0.3:
        parts.append("inferior" if y > 0 else "superior")
    if abs(z) > 0.3:
        parts.append("posterior" if z > 0 else "anterior")
    return "-".join(parts) if parts else "zentral"


def st_territory(v: Optional[np.ndarray]) -> Optional[str]:
    """Legacy sector label for vector orientation, not a verified lesion territory."""
    if v is None or np.linalg.norm(v) < 1e-9:
        return None
    x, y, z = v / np.linalg.norm(v)
    if y > 0.6:
        return "inferior"
    if z < -0.55:
        return "anterior/septal" if x < 0.3 else "anterolateral"
    if z > 0.6:
        return "posterior"
    if x > 0.6:
        return "lateral"
    if x < -0.6 and y < 0:
        return "rechtsventrikulär/superior?"
    return "unbestimmt"


def read_beat(m: BeatMeasures, *, qrs_ms: Optional[float], criteria: dict, rhythm_class: str, rhythm_flags: tuple[str, ...],
              fidelity: dict[str, float], xyz: Optional[np.ndarray] = None, fs: float = 500.0,
              bounds: Optional[BeatBounds] = None, sex: Optional[str] = None) -> ReadingResult:
    R: list[Reading] = []
    out = ReadingResult(R)
    g = m.get
    v = m.vectors
    qrs_ms = qrs_ms or g("qrs_duration") or 0.0
    wide = qrs_ms >= 120
    qv = v.get("qrs_max")
    term40, init20, init40 = v.get("qrs_terminal_40"), v.get("qrs_initial_20"), v.get("qrs_initial_40")
    ins = m.inscriptions.get("qrs", {})
    frontal_axis = g("qrs_frontal_axis")

    # ---- Transformationsgüte und Rhythmus (Hinweise zuerst, sie relativieren alles Weitere)
    if fidelity:
        r_mean = float(np.mean(list(fidelity.values())))
        worst = min(fidelity, key=fidelity.get)
        if r_mean < 0.85:
            R.append(Reading("i", "Transformationsgüte niedrig", f"mittlere Korrelation rekonstruierter Ableitungen r̄ = {r_mean:.2f} (schwächste {worst} {fidelity[worst]:.2f}); "
                             "die Standardmatrix beschreibt diesen Thorax schlecht, Richtungsmaße mit Vorsicht", "Kors 1990; Vondrak 2022", 0.9))
    if rhythm_class in ("atrial_fibrillation", "af", "irregular") or any("fibrillation" in f for f in rhythm_flags):
        R.append(Reading("i", "Vorhofflimmern/irregulär", "keine P-Schleife; Schlag-zu-Schlag-Streuung der T-Schleife erwartbar, Mittelung über Schläge unsicher", "", 0.8))

    # ---- Leitung
    lbbb_k = _crit(criteria, "lbbb_pattern")
    rbbb_k = _crit(criteria, "rbbb_pattern")
    slow_ms, slow_where = g("qrs_slow_ms") or 0.0, ""
    for mm in m.measures:
        if mm.key == "qrs_slow_ms":
            slow_where = mm.note.replace("Schwerpunkt ", "")
    if wide and qv is not None:
        az = azimuth_deg(qv)
        term_right_ant = term40 is not None and term40[0] < 0 and term40[2] < 0
        left_post = az is not None and 10 <= az <= 110
        if lbbb_k or (left_post and not term_right_ant and slow_ms >= 25 and slow_where in ("mittel", "terminal")):
            R.append(Reading("K" if lbbb_k else "E", "Linksschenkelblock-Muster",
                             f"QRS {qrs_ms:.0f} ms, Maximalvektor links-posterior (Azimut {az:.0f}°), langsame mittlere/terminale Einschreibung {slow_ms:.0f} ms"
                             + (", T diskordant" if (g("qrs_t_angle_mean") or 0) > 120 else ""),
                             "Pérez-Riera 2019 Ann Noninvasive Electrocardiol 24:e12644", 0.9))
        elif rbbb_k or term_right_ant:
            R.append(Reading("K" if rbbb_k else "E", "Rechtsschenkelblock-Muster",
                             f"QRS {qrs_ms:.0f} ms, terminale 40 ms rechts-anterior ({_vec_dir(term40)}), "
                             f"transversal {INSCRIPTION_TEXT.get(ins.get('transversal', ('–',))[0])}"
                             + (f", langsame terminale Einschreibung {slow_ms:.0f} ms" if slow_where == "terminal" else ""),
                             "Pérez-Riera 2021 Curr Cardiol Rev 17:50", 0.85))
        else:
            R.append(Reading("H", "Unspezifische intraventrikuläre Leitungsverzögerung", f"QRS {qrs_ms:.0f} ms ohne typisches Schenkelblock-Bild der Schleife", "", 0.5))
    if not wide and frontal_axis is not None:
        f_ins = ins.get("frontal", ("–",))[0]
        if -90 <= frontal_axis <= -30 and init20 is not None and init20[1] > 0:
            R.append(Reading("E" if f_ins == "ccw" else "H", "Linksanteriorer Faszikelblock-Muster",
                             f"Frontalachse {frontal_axis:.0f}°, initiale 20 ms inferior, frontale Schleife {INSCRIPTION_TEXT[f_ins]}"
                             + ("" if f_ins == "ccw" else " (erwartet: gegen den Uhrzeigersinn)"), "Pérez-Riera 2018; Chou", 0.7))
        elif 100 <= frontal_axis <= 180 and init20 is not None and init20[1] < 0:
            R.append(Reading("H", "Linksposteriorer Faszikelblock-Muster (Differenzial: RVH, Astheniker)",
                             f"Frontalachse {frontal_axis:.0f}°, initiale 20 ms superior, frontale Schleife {INSCRIPTION_TEXT[f_ins]}",
                             "Pérez-Riera 2018 Indian Pacing Electrophysiol J 18:217", 0.5))
    if not wide and qv is not None:
        az = azimuth_deg(qv)
        if az is not None and -100 <= az <= -25 and qv[2] < -0.5:
            R.append(Reading("H", "Prominente anteriore QRS-Kräfte", f"Maximalvektor anterior (Azimut {az:.0f}°): linksseptaler Faszikelblock, posteriorer Infarkt oder RV-Hypertrophie abgrenzen",
                             "Pérez-Riera 2023 Heart Rhythm 20:1607", 0.5))
    if slow_where == "initial" and slow_ms >= 20 and not wide:
        R.append(Reading("H", "Langsame initiale Einschreibung", f"{slow_ms:.0f} ms unter 20 % der Maximalgeschwindigkeit am QRS-Beginn – Präexzitation (Delta) oder initiale Narbe prüfen", "", 0.4))
    # Drehrichtung gegen die Erwartung
    for pl, exp in NORMAL_INSCRIPTION.items():
        got = ins.get(pl, ("–",))[0]
        if got not in ("–", exp) and not wide:
            R.append(Reading("H", f"Drehrichtung {pl} atypisch", f"{INSCRIPTION_TEXT[got]} (üblich: {INSCRIPTION_TEXT[exp]})", "Chou; Pérez-Riera 2021", 0.3))

    # ---- Hypertrophie
    lvh_k = any(_crit(criteria, k) for k in ("sokolow_lyon", "cornell_voltage", "cornell_product", "peguero_lo_presti") if _crit(criteria, k) is not None)
    qmag = g("qrs_max_magnitude") or 0.0
    qmag_status = next((mm.status for mm in m.measures if mm.key == "qrs_max_magnitude"), "")
    if qv is not None and (lvh_k or qmag_status == "hoch"):
        strain = (g("qrs_t_angle_mean") or 0) > 105
        R.append(Reading("K" if lvh_k else "E", "Linksventrikuläre Hypertrophie – Vektorbild",
                         f"QRS-Maximalvektor {qmag:.2f} mV nach {_vec_dir(qv)}" + (f", QRS-T-Winkel {g('qrs_t_angle_mean'):.0f}° (Strain-Muster)" if strain else ""),
                         "Man 2012 J Electrocardiol 45:154; Maanja 2022", 0.7 if strain else 0.5))
    if qv is not None and qv[0] < 0 and qv[2] < 0 and not wide:
        R.append(Reading("H", "Rechtsventrikuläre Dominanz", f"QRS-Maximalvektor rechts-anterior ({_vec_dir(qv)}), Frontalachse {frontal_axis:.0f}°" if frontal_axis is not None else "Maximalvektor rechts-anterior",
                         "Pérez-Riera 2021", 0.5))

    # ---- Ischämie / Infarkt
    st = v.get("st_j60")
    st_mag = g("st_j60_magnitude") or 0.0
    ste_k = _crit(criteria, "st_elevation_contiguous")
    std_k = _crit(criteria, "st_depression_contiguous")
    if st is not None and (st_mag >= 0.1 or ste_k or std_k):
        terr = st_territory(st)
        out.territory = terr
        R.append(Reading(
            "K" if (ste_k or std_k) else "H", "ST-J+60-Vektor",
            f"|ST| {st_mag:.2f} mV nach {_vec_dir(st)}; grober Richtungssektor {terr}. "
            "Weder Infarktareal noch schuldige Arterie sind daraus verifiziert. "
            + ("Zusätzlich kontiguierliche ST-Hebung im Laborregelstand. " if ste_k else
               "Zusätzlich kontiguierliche ST-Senkung im Laborregelstand. " if std_k else
               "Nur Vektorheuristik. ")
            + "Klinik, Vergleichs-EKG und unabhängige Diagnostik erforderlich.",
            "ter Haar 2013; Treskes 2015 J Electrocardiol 48:498",
            0.8 if st_mag >= 0.15 else 0.6,
        ))
    q_k = _crit(criteria, "pathological_q_waves") or _crit(criteria, "qs_complexes")
    if q_k:
        R.append(Reading("K", "Pathologische Q-Zacken/QS (Labor)", f"initiale Kräfte {_vec_dir(init40)} (40 ms)", "", 0.7))
    if init20 is not None and init40 is not None and not wide:
        # inferiorer Infarkt: initiale Kräfte superior verschoben (klassisch ≥ 25 ms superior im Frontalbild)
        if init20[1] < 0 and init40[1] < 0 and np.linalg.norm(init40) > 0.15:
            R.append(Reading("H", "Initiale Kräfte superior (Narbe inferior?)", f"initiale 20/40 ms superior ({_vec_dir(init40)}), Frontalachse {frontal_axis:.0f}°" if frontal_axis is not None else "initiale Kräfte superior",
                             "Chou; Pérez-Riera 2021 (VCG-Infarktkriterien)", 0.45))
        if init20[2] > 0 and init40[2] > 0 and np.linalg.norm(init40) > 0.15:
            R.append(Reading("H", "Initiale Kräfte posterior (Narbe anterior/septal?)", f"initiale 20/40 ms posterior ({_vec_dir(init40)}) statt anterior",
                             "Chou; Kijonka 2024 Comput Biol Med 169:107781", 0.45))
    for key, label, ref in (("qrs_planarity", "QRS-Schleife verliert Ebenheit", "Ray 2017 J Electrocardiol 50:652; Correa 2013"),
                            ("t_plane_rms", "T-Schleife nicht eben (Tavplan-Analogon)", "Rubulis 2010 J Electrocardiol 43:360")):
        mm = next((x for x in m.measures if x.key == key), None)
        if mm is not None and mm.status == "hoch":
            R.append(Reading("E", label, f"{mm.label} {mm.text()} {mm.unit} über der Referenz ({mm.limit.source})", ref, 0.5))

    # ---- Repolarisation
    ang = g("qrs_t_angle_mean")
    if ang is not None:
        if ang > 135:
            R.append(Reading("E", "Räumlicher QRS-T-Winkel abnorm", f"{ang:.0f}° (Flächenvektoren, Kors) – abnorm > 135°; bei Schenkelblock/Hypertrophie sekundär", "Kardys 2003; Kück 2018", 0.8))
        elif ang > 105:
            R.append(Reading("E", "Räumlicher QRS-T-Winkel grenzwertig", f"{ang:.0f}° (105–135°)", "Kardys 2003 Eur Heart J 24:1357", 0.5))
    tc = g("tcrt")
    if tc is not None and tc < -0.3 and not wide:
        R.append(Reading("E", "TCRT negativ", f"TCRT {tc:.2f}: Depolarisation und Repolarisation diskordant", "Acar 1999; Tse 2018", 0.5))
    svg = next((x for x in m.measures if x.key == "svg_magnitude"), None)
    if svg is not None and svg.status in ("hoch", "niedrig"):
        R.append(Reading("E", f"SVG-Betrag {svg.status}", f"{svg.text()} mV·ms ({svg.limit.source})", "Waks & Tereshchenko 2016", 0.4))

    # ---- Vorhof
    p_dur, pt, pv = g("p_duration"), v.get("p_terminal"), v.get("p_max")
    if p_dur is not None and pt is not None:
        if p_dur >= 120 and pt[2] > 0 and pt[1] < 0:
            R.append(Reading("E", "Interatrialer Block (fortgeschritten?)", f"P {p_dur:.0f} ms, terminale P-Kräfte posterior-superior ({_vec_dir(pt)})", "Bayés de Luna; Pérez-Riera 2021", 0.6))
        elif p_dur >= 120:
            R.append(Reading("H", "P verbreitert", f"P {p_dur:.0f} ms, terminale Kräfte {_vec_dir(pt)} – interatriale Leitungsstörung/LA-Vergrößerung", "", 0.4))
    if pv is not None and np.linalg.norm(pv) >= 0.25 and pv[1] > 0 and pv[2] < 0:
        R.append(Reading("H", "P-Vektor groß anterior-inferior", f"|P| {np.linalg.norm(pv):.2f} mV – rechtsatriale Belastung prüfen", "", 0.4))

    # ---- Brugada-Hinweis (JT-Abstand nach Madaloso, nur bei terminal rechts-anteriorem Vektor)
    brs_k = None  # kein Brugada-Kriterium im Hauptprojekt; nur VCG-Hinweis
    if xyz is not None and bounds is not None and term40 is not None and term40[0] < 0 and term40[2] < 0:
        jt_t = jt_distance(xyz, fs, bounds, "transversal")
        jt_s = jt_distance(xyz, fs, bounds, "sagittal")
        if brs_k or (jt_t is not None and jt_t >= 0.15):
            R.append(Reading("K" if brs_k else "H", "J-Wellen-Muster: Brugada-Abgrenzung",
                             f"JT-Abstand transversal {jt_t if jt_t is None else round(jt_t * 10, 1)} mm, sagittal {jt_s if jt_s is None else round(jt_s * 10, 1)} mm (≥ 1,5 / > 1,25 mm sprechen für Typ 1)",
                             "Madaloso 2022 J Electrocardiol 72:44", 0.5))

    R.sort(key=lambda r: ({"K": 0, "E": 1, "H": 2, "i": 3}[r.tier], -r.weight))
    if not [r for r in R if r.tier in ("K", "E", "H")]:
        R.append(Reading("i", "Keine Auffälligkeit der Schleifenlesung", "Vektoren, Drehrichtungen und QRS-T-Beziehung ohne Regeltreffer", "", 0.1))
    out.summary = "; ".join(r.title for r in R if r.tier in ("K", "E"))[:160]
    return out
