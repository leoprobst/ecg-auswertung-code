"""Vektorkardiographische Maße auf einem ursprungskorrigierten Schlag (reine NumPy-Funktionen).

Koordinaten wie im gesamten Projekt (Frank-kompatibel): +X links, +Y inferior, +Z posterior, mV.
Azimut = atan2(Z, X) in Grad (0° links, +90° posterior, −90° anterior); Elevation = Winkel zu +Y
(0° inferior, 90° horizontal, 180° superior) – identisch zu ``core.vcg_geometry``.

Anzeigeebenen (Bildschirm: x nach rechts, y nach unten) und die daraus folgende Drehrichtung:
  frontal      (Blick von vorn):   sx = +X, sy = +Y   – normale QRS-Schleife im Uhrzeigersinn
  transversal  (Blick von kaudal): sx = +X, sy = −Z   – normale QRS-Schleife gegen den Uhrzeigersinn
  sagittal     (rechts-sagittal):  sx = −Z, sy = +Y   – normale QRS-Schleife im Uhrzeigersinn
(Chou; Pérez-Riera 2021, Curr Cardiol Rev 17:50). Die Drehrichtung wird als vorzeichenbehaftete
Fläche in genau diesem Bildschirmrahmen bestimmt, damit Anzeige und Zahl nie auseinanderlaufen.

Grenzwerte: nur mit Quelle; wo die Literatur keine liefert, bleibt das Feld leer und die
datenbankeigene Referenz (``reference_limits.json``, MIMIC-IV normal, Kors) füllt es.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

EPS = 1e-12

PLANES = {
    # name: (Spaltenwahl, Vorzeichen) für Bildschirm-x und Bildschirm-y
    "frontal": ((0, 1.0), (1, 1.0)),
    "transversal": ((0, 1.0), (2, -1.0)),
    "sagittal": ((2, -1.0), (1, 1.0)),
}
PLANE_LABELS = {
    "frontal": ("Frontal (Blick von vorn)", "rechts", "links", "superior", "inferior"),
    "transversal": ("Transversal (Blick von kaudal)", "rechts", "links", "posterior", "anterior"),
    "sagittal": ("Rechts-sagittal (Blick von rechts)", "posterior", "anterior", "superior", "inferior"),
}
NORMAL_INSCRIPTION = {"frontal": "cw", "transversal": "ccw", "sagittal": "cw"}


@dataclass
class Limit:
    low: Optional[float]
    high: Optional[float]
    source: str
    borderline_high: Optional[float] = None  # Grauzone bis hier (z. B. QRS-T-Winkel 105–135°)
    sex: Optional[str] = None  # "F"/"M" wenn geschlechtsspezifisch
    circular: bool = False  # Winkel: low/high = Zentrum ± Halbbreite, Vergleich modulo 360°

    def classify(self, value: float) -> str:
        if value is None or not np.isfinite(value):
            return "–"
        if self.circular and self.low is not None and self.high is not None:
            centre, half = 0.5 * (self.low + self.high), 0.5 * (self.high - self.low)
            d = ((value - centre + 180.0) % 360.0) - 180.0
            return "abweichend" if abs(d) > half else "normal"
        if self.low is not None and value < self.low:
            return "niedrig"
        if self.high is not None and value > self.high:
            if self.borderline_high is not None and value <= self.borderline_high:
                return "grenzwertig"
            return "hoch"
        return "normal"


@dataclass
class Measure:
    key: str
    label: str
    value: Optional[float]
    unit: str
    group: str
    limit: Optional[Limit] = None
    note: str = ""
    vector: Optional[np.ndarray] = None  # zugehöriger Raumvektor (mV oder mV·ms)

    @property
    def status(self) -> str:
        if self.value is None or not np.isfinite(self.value):
            return "–"
        if self.limit is None:
            return ""
        return self.limit.classify(self.value)

    def text(self) -> str:
        if self.value is None or not np.isfinite(self.value):
            return "–"
        if abs(self.value) >= 100:
            return f"{self.value:.0f}"
        if abs(self.value) >= 10:
            return f"{self.value:.1f}"
        return f"{self.value:.2f}"


# ------------------------------------------------------------------ Literaturgrenzen (Quelle je Wert)
LIMITS: dict[str, Limit] = {
    # Kardys 2003 (Rotterdam, Kors-Matrix): normal < 105°, grenzwertig 105–135°, abnorm > 135°
    "qrs_t_angle_mean": Limit(None, 105.0, "Kardys 2003 Eur Heart J 24:1357 (Kors; >135° abnorm)", borderline_high=135.0),
    "qrs_t_angle_peak": Limit(None, 105.0, "Kardys 2003; Bergfeldt 2024 J Electrocardiol (95. Perzentil 124°)", borderline_high=124.0),
    # Scherptong 2008 (inverse Dower, 18–29 J.): SVG-Betrag f 39–143, m 59–187 mV·ms; QRS-T mean f 20–116, m 30–130
    "svg_magnitude_F": Limit(39.0, 143.0, "Scherptong 2008 J Electrocardiol 41:648 (inverse Dower, 18–29 J.)", sex="F"),
    "svg_magnitude_M": Limit(59.0, 187.0, "Scherptong 2008 J Electrocardiol 41:648 (inverse Dower, 18–29 J.)", sex="M"),
    # Tokavanich 2022 (Meta-Analyse CRT): QRS-Fläche 102–116 µVs trennt großen von kleinem Areal; Andersen 2021 LSB-Quartile Q4 ≥ 146 µVs
    "qrs_area": Limit(None, 102.0, "Tokavanich 2022 PACE 45:393 (CRT-Schwelle 102–116 µVs); Andersen 2021 (LSB Q4 ≥ 146)", borderline_high=116.0),
    # ter Haar 2013 / Treskes 2015: serielle Differenzvektoren ΔST > 0,05 mV, ΔVG > 16,2 mV·ms
    "delta_st": Limit(None, 0.05, "ter Haar 2013 J Electrocardiol 46:302 (Ballonokklusion, Kors)"),
    "delta_vg": Limit(None, 16.2, "ter Haar 2013 J Electrocardiol 46:302"),
    # TCRT: Tse 2018 Meta-Analyse – niedrigere Werte bei Ereignissen nach Infarkt (Differenz −0,36); keine harte Schwelle
    "tcrt": Limit(None, None, "Acar 1999; Tse 2018 Ann Noninvasive Electrocardiol 23:e12495 (niedrig = ungünstig)"),
    # Madaloso 2022: JT-Abstand ≥ 1,5 mm transversal (0,15 mV) für Brugada-Typ 1 vs. andere J-Wellen-Muster
    "jt_distance_transversal": Limit(None, 0.15, "Madaloso 2022 J Electrocardiol 72:44 (≥ 1,5 mm transversal, Frank)"),
    "jt_distance_sagittal": Limit(None, 0.125, "Madaloso 2022 J Electrocardiol 72:44 (> 1,25 mm sagittal, Frank)"),
    # ST-Vektor: klassische 0,1 mV-Schwelle (Kors/van Herpen-Serien; STEMI-Kriterien) – nur Orientierung
    "st_j60_magnitude": Limit(None, 0.10, "klassische ST-Schwelle 0,1 mV (Orientierung; Frank-Betrag)"),
}


# ------------------------------------------------------------------ Grundfunktionen
def azimuth_deg(v: np.ndarray) -> Optional[float]:
    if v is None or np.hypot(v[0], v[2]) < EPS:
        return None
    return float(np.degrees(np.arctan2(v[2], v[0])))


def elevation_deg(v: np.ndarray) -> Optional[float]:
    n = np.linalg.norm(v)
    if v is None or n < EPS:
        return None
    return float(np.degrees(np.arccos(np.clip(v[1] / n, -1.0, 1.0))))


def frontal_axis_deg(v: np.ndarray) -> Optional[float]:
    """Frontalachse wie im ECG (0° links, +90° inferior) aus X und Y."""
    if v is None or np.hypot(v[0], v[1]) < EPS:
        return None
    return float(np.degrees(np.arctan2(v[1], v[0])))


def angle_deg(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def peak_vector(seg: np.ndarray) -> tuple[Optional[np.ndarray], int]:
    if seg is None or len(seg) == 0:
        return None, -1
    mag = np.linalg.norm(seg, axis=1)
    k = int(np.argmax(mag))
    return seg[k].copy(), k


def area_vector_mv_ms(seg: np.ndarray, fs: float) -> Optional[np.ndarray]:
    """Zeitintegral (mV·ms); Trapezregel wie ``core.vcg_geometry.area_vector_uv_ms`` (dort in µV·ms)."""
    if seg is None or len(seg) < 2:
        return None
    dt = 1000.0 / fs
    return np.trapezoid(seg, dx=dt, axis=0) if hasattr(np, "trapezoid") else np.trapz(seg, dx=dt, axis=0)


def sum_absolute_integral(seg: np.ndarray, fs: float) -> Optional[float]:
    """SAI (mV·ms): Summe der Beträge der Zeitintegrale von |X|, |Y|, |Z| (Tereshchenko GEH)."""
    if seg is None or len(seg) < 2:
        return None
    dt = 1000.0 / fs
    a = np.abs(seg)
    integ = np.trapezoid(a, dx=dt, axis=0) if hasattr(np, "trapezoid") else np.trapz(a, dx=dt, axis=0)
    return float(np.sum(integ))


def spatial_velocity(seg: np.ndarray, fs: float, sigma_ms: float = 2.0) -> np.ndarray:
    """Räumliche Geschwindigkeit |dv/dt| in mV/s, gaußgeglättet (σ 2 ms wie die Kinematik-Marker)."""
    if seg is None or len(seg) < 3:
        return np.zeros(max(0, len(seg) if seg is not None else 0))
    d = np.gradient(seg, 1.0 / fs, axis=0)
    v = np.linalg.norm(d, axis=1)
    s = sigma_ms * fs / 1000.0
    if s > 0.5:
        r = int(np.ceil(3 * s))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
        k /= k.sum()
        v = np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")
    return v


def window_mean(seg: np.ndarray, fs: float, ms: float, terminal: bool = False) -> Optional[np.ndarray]:
    if seg is None or len(seg) == 0:
        return None
    n = max(1, int(round(ms * fs / 1000.0)))
    n = min(n, len(seg))
    return seg[-n:].mean(0) if terminal else seg[:n].mean(0)


# ------------------------------------------------------------------ Schleifengeometrie
@dataclass
class LoopGeometry:
    eigenvalues: np.ndarray  # λ1 ≥ λ2 ≥ λ3 (mV²)
    normal: np.ndarray  # Normale der Ausgleichsebene (Einheitsvektor, Z-Komponente ≥ 0 für Eindeutigkeit)
    planarity_ratio: float  # λ3/λ2 (0 = eben)
    plane_rms_uv: float  # mittlerer Abstand zur Ebene (µV)
    roundness: float  # λ2/λ1 (1 = kreisrund)
    perimeter_mv: float
    planar_area_mv2: float  # Fläche der auf die Ebene projizierten Schleife
    hull_volume_mv3: Optional[float]
    centroid: np.ndarray
    max_centroid_distance_mv: float


def loop_geometry(seg: np.ndarray) -> Optional[LoopGeometry]:
    if seg is None or len(seg) < 4:
        return None
    c = seg.mean(0)
    d = seg - c
    cov = d.T @ d / len(d)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, v = np.clip(w[order], 0.0, None), v[:, order]
    normal = v[:, 2]
    if normal[2] < 0:
        normal = -normal
    proj = d @ v[:, :2]
    x, y = proj[:, 0], proj[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    per = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)) + np.linalg.norm(seg[0] - seg[-1]))
    dist = d @ normal
    volume = None
    try:
        from scipy.spatial import ConvexHull
        if len(seg) >= 5 and w[2] > 1e-10:
            volume = float(ConvexHull(seg).volume)
    except Exception:
        volume = None
    return LoopGeometry(
        eigenvalues=w, normal=normal,
        planarity_ratio=float(w[2] / w[1]) if w[1] > EPS else 0.0,
        plane_rms_uv=float(np.sqrt(np.mean(dist ** 2)) * 1000.0),
        roundness=float(w[1] / w[0]) if w[0] > EPS else 0.0,
        perimeter_mv=per, planar_area_mv2=area, hull_volume_mv3=volume,
        centroid=c, max_centroid_distance_mv=float(np.max(np.linalg.norm(d, axis=1))),
    )


def screen_coords(seg: np.ndarray, plane: str) -> np.ndarray:
    (cx, sx), (cy, sy) = PLANES[plane]
    return np.column_stack([seg[:, cx] * sx, seg[:, cy] * sy])


def inscription(seg: np.ndarray, plane: str) -> tuple[str, float, float]:
    """Drehrichtung in der Anzeigeebene: ('cw'|'ccw'|'figure8'|'–', Anteil gegenläufiger Fläche, Fläche mV²).

    Vorzeichenbehaftete Fläche im Bildschirmrahmen (y nach unten): positiv = im Uhrzeigersinn.
    Eine Achterschleife liegt vor, wenn beide Umlaufsinne je mindestens 15 % der Gesamtfläche tragen.
    """
    if seg is None or len(seg) < 4:
        return "–", 0.0, 0.0
    p = screen_coords(seg, plane)
    p = p - p[0]
    x, y = p[:, 0], p[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]  # Dreiecksflächen um den Startpunkt
    pos, neg = float(np.sum(cross[cross > 0])), float(-np.sum(cross[cross < 0]))
    total = pos + neg
    if total < 1e-6:
        return "–", 0.0, 0.0
    minor = min(pos, neg) / total
    if minor >= 0.15:
        return "figure8", minor, 0.5 * abs(pos - neg)
    return ("cw" if pos >= neg else "ccw"), minor, 0.5 * abs(pos - neg)


INSCRIPTION_TEXT = {"cw": "im Uhrzeigersinn", "ccw": "gegen den Uhrzeigersinn", "figure8": "Achterschleife", "–": "–"}


def octant_dwell(seg: np.ndarray) -> dict[str, float]:
    """Zeitanteil je Oktant (Vorzeichen X, Y, Z); Schlüssel z. B. 'L I P' = links inferior posterior."""
    if seg is None or len(seg) == 0:
        return {}
    out: dict[str, float] = {}
    sx = np.where(seg[:, 0] >= 0, "L", "R")
    sy = np.where(seg[:, 1] >= 0, "I", "S")
    sz = np.where(seg[:, 2] >= 0, "P", "A")
    keys = [f"{a} {b} {c}" for a, b, c in zip(sx, sy, sz)]
    for k in keys:
        out[k] = out.get(k, 0.0) + 1.0 / len(keys)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def slow_inscription(seg: np.ndarray, fs: float, share: float = 0.2) -> tuple[float, str]:
    """Dauer (ms) mit Geschwindigkeit < share·Maximum und ihr Schwerpunkt (initial/mittel/terminal)."""
    v = spatial_velocity(seg, fs)
    if len(v) < 3:
        return 0.0, "–"
    slow = v < share * v.max()
    ms = float(slow.sum() * 1000.0 / fs)
    if slow.sum() == 0:
        return 0.0, "–"
    pos = float(np.mean(np.nonzero(slow)[0]) / max(1, len(v) - 1))
    return ms, ("initial" if pos < 1 / 3 else "mittel" if pos < 2 / 3 else "terminal")


def tcrt(qrs: np.ndarray, t: np.ndarray, share: float = 0.7) -> Optional[float]:
    """Total cosine R-to-T (Acar 1999, Näherung): mittlerer Kosinus zwischen T-Maximalvektor und den
    QRS-Vektoren mit Betrag ≥ share·Maximum. Werte nahe 1 = konkordant, negativ = diskordant."""
    if qrs is None or t is None or len(qrs) < 3 or len(t) < 3:
        return None
    tv, _ = peak_vector(t)
    if tv is None or np.linalg.norm(tv) < EPS:
        return None
    mag = np.linalg.norm(qrs, axis=1)
    sel = qrs[mag >= share * mag.max()]
    if len(sel) == 0:
        return None
    cos = (sel @ tv) / (np.linalg.norm(sel, axis=1) * np.linalg.norm(tv) + EPS)
    return float(np.mean(cos))


# ------------------------------------------------------------------ Schlagmaße
@dataclass
class BeatBounds:
    qrs_on: int
    qrs_off: int
    t_end: Optional[int] = None
    t_on: Optional[int] = None
    p_on: Optional[int] = None
    p_off: Optional[int] = None
    t_peak: Optional[int] = None


@dataclass
class BeatMeasures:
    measures: list[Measure] = field(default_factory=list)
    qrs_geometry: Optional[LoopGeometry] = None
    t_geometry: Optional[LoopGeometry] = None
    p_geometry: Optional[LoopGeometry] = None
    inscriptions: dict[str, dict[str, tuple[str, float, float]]] = field(default_factory=dict)  # segment → plane → (…)
    octants: dict[str, float] = field(default_factory=dict)
    vectors: dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, key: str) -> Optional[float]:
        for m in self.measures:
            if m.key == key:
                return m.value
        return None

    def by_group(self) -> dict[str, list[Measure]]:
        out: dict[str, list[Measure]] = {}
        for m in self.measures:
            out.setdefault(m.group, []).append(m)
        return out


def _lim(key: str, sex: Optional[str]) -> Optional[Limit]:
    if sex and f"{key}_{sex}" in LIMITS:
        return LIMITS[f"{key}_{sex}"]
    return LIMITS.get(key)


def beat_measures(xyz: np.ndarray, fs: float, b: BeatBounds, sex: Optional[str] = None,
                  reference: Optional[dict] = None) -> BeatMeasures:
    """Alle Kernmaße eines ursprungskorrigierten Schlags; ``reference`` (aus reference_limits.json) füllt
    fehlende Grenzwerte mit datenbankeigenen Perzentilen (Schlüssel wie Measure.key)."""
    out = BeatMeasures()
    M = out.measures
    xyz = np.asarray(xyz, dtype=float)
    n = len(xyz)
    q0, q1 = max(0, b.qrs_on), min(n, b.qrs_off)
    qrs = xyz[q0:q1]
    t_end = min(n, b.t_end) if b.t_end else None
    t_on = b.t_on if b.t_on else None
    jt = xyz[q1:t_end] if t_end and t_end > q1 else None
    t = xyz[t_on:t_end] if (t_on and t_end and t_end > t_on) else jt
    p = xyz[b.p_on:b.p_off] if (b.p_on is not None and b.p_off is not None and b.p_off > b.p_on) else None

    def add(key, label, value, unit, group, note="", vector=None):
        lim = _lim(key, sex)
        if lim is None and reference and key in reference:
            r = reference[key]
            if sex and isinstance(r.get(sex), dict):
                r = dict(r, **r[sex])
            src = f"eigene Referenz (MIMIC-IV normal, Kors, n={r.get('n', '?')}, 2,5–97,5 %{', ' + ('weiblich' if sex == 'F' else 'männlich') if sex and sex in reference[key] else ''})"
            if r.get("circular") and r.get("centre") is not None:
                lim = Limit(r["centre"] - r["half_width_97_5"], r["centre"] + r["half_width_97_5"], src.replace("2,5–97,5 %", "Zentrum ± 97,5 % der Abweichung"), circular=True)
            else:
                lim = Limit(r.get("p2_5"), r.get("p97_5"), src)
        M.append(Measure(key, label, None if value is None else float(value), unit, group, lim, note, vector))

    # ---- Hauptvektoren
    qv, qk = peak_vector(qrs)
    out.vectors["qrs_max"] = qv
    add("qrs_max_magnitude", "QRS-Maximalvektor Betrag", None if qv is None else np.linalg.norm(qv), "mV", "QRS-Vektoren", vector=qv)
    add("qrs_max_azimuth", "QRS-Maximalvektor Azimut", azimuth_deg(qv) if qv is not None else None, "°", "QRS-Vektoren")
    add("qrs_max_elevation", "QRS-Maximalvektor Elevation", elevation_deg(qv) if qv is not None else None, "°", "QRS-Vektoren")
    add("qrs_max_time", "Zeit bis QRS-Maximalvektor", None if qk < 0 else qk * 1000.0 / fs, "ms", "QRS-Vektoren")
    qa = area_vector_mv_ms(qrs, fs)
    out.vectors["qrs_area"] = qa
    add("qrs_area", "QRS-Fläche (Areal-Vektor, |∫QRS|)", None if qa is None else np.linalg.norm(qa) , "µVs", "QRS-Vektoren",
        note="1 mV·ms = 1 µVs", vector=qa)
    add("qrs_area_azimuth", "QRS-Flächenvektor Azimut", azimuth_deg(qa) if qa is not None else None, "°", "QRS-Vektoren")
    add("qrs_area_elevation", "QRS-Flächenvektor Elevation", elevation_deg(qa) if qa is not None else None, "°", "QRS-Vektoren")
    add("qrs_frontal_axis", "QRS-Achse frontal (Flächenvektor)", frontal_axis_deg(qa) if qa is not None else None, "°", "QRS-Vektoren")
    add("qrs_duration", "QRS-Dauer (Delineation)", (q1 - q0) * 1000.0 / fs, "ms", "QRS-Vektoren")
    for ms in (20, 40):
        iv = window_mean(qrs, fs, ms)
        out.vectors[f"qrs_initial_{ms}"] = iv
        add(f"qrs_initial_{ms}_azimuth", f"Initiale {ms} ms: Azimut", azimuth_deg(iv) if iv is not None else None, "°", "QRS-Verlauf", vector=iv)
        add(f"qrs_initial_{ms}_elevation", f"Initiale {ms} ms: Elevation", elevation_deg(iv) if iv is not None else None, "°", "QRS-Verlauf")
        tv_ = window_mean(qrs, fs, ms, terminal=True)
        out.vectors[f"qrs_terminal_{ms}"] = tv_
        add(f"qrs_terminal_{ms}_azimuth", f"Terminale {ms} ms: Azimut", azimuth_deg(tv_) if tv_ is not None else None, "°", "QRS-Verlauf", vector=tv_)
        add(f"qrs_terminal_{ms}_elevation", f"Terminale {ms} ms: Elevation", elevation_deg(tv_) if tv_ is not None else None, "°", "QRS-Verlauf")
    iv20, tv40 = out.vectors.get("qrs_initial_20"), out.vectors.get("qrs_terminal_40")
    add("qrs_initial_terminal_angle", "Winkel initial 20 ms ↔ terminal 40 ms", angle_deg(iv20, tv40), "°", "QRS-Verlauf")
    add("qrs_terminal_main_angle", "Winkel terminal 40 ms ↔ Maximalvektor", angle_deg(tv40, qv), "°", "QRS-Verlauf")
    sv = spatial_velocity(qrs, fs)
    add("qrs_velocity_max", "Maximale räumliche Geschwindigkeit", None if len(sv) == 0 else sv.max() / 1000.0, "mV/ms", "QRS-Verlauf")
    add("qrs_velocity_max_time", "Zeit des Geschwindigkeitsmaximums", None if len(sv) == 0 else int(np.argmax(sv)) * 1000.0 / fs, "ms", "QRS-Verlauf")
    slow_ms, slow_where = slow_inscription(qrs, fs)
    add("qrs_slow_ms", "Langsame Einschreibung (< 20 % v_max)", slow_ms, "ms", "QRS-Verlauf", note=f"Schwerpunkt {slow_where}")
    # ---- T und Repolarisation
    tv, tk = peak_vector(t) if t is not None else (None, -1)
    out.vectors["t_max"] = tv
    add("t_max_magnitude", "T-Maximalvektor Betrag", None if tv is None else np.linalg.norm(tv), "mV", "T-Vektoren", vector=tv)
    add("t_max_azimuth", "T-Maximalvektor Azimut", azimuth_deg(tv) if tv is not None else None, "°", "T-Vektoren")
    add("t_max_elevation", "T-Maximalvektor Elevation", elevation_deg(tv) if tv is not None else None, "°", "T-Vektoren")
    ta = area_vector_mv_ms(jt, fs) if jt is not None else None
    out.vectors["t_area"] = ta
    add("t_area", "JT-Fläche (|∫JT|)", None if ta is None else np.linalg.norm(ta), "µVs", "T-Vektoren", vector=ta)
    add("t_frontal_axis", "T-Achse frontal (Flächenvektor)", frontal_axis_deg(ta) if ta is not None else None, "°", "T-Vektoren")
    add("qrs_t_angle_peak", "Räumlicher QRS-T-Winkel (Spitzenvektoren)", angle_deg(qv, tv), "°", "QRS-T-Beziehung")
    add("qrs_t_angle_mean", "Räumlicher QRS-T-Winkel (Flächenvektoren)", angle_deg(qa, ta), "°", "QRS-T-Beziehung",
        note="Kors + Flächenvektoren: beste Mortalitätsvorhersage (Kück 2018)")
    add("tcrt", "TCRT (Acar-Näherung)", tcrt(qrs, t) if t is not None else None, "", "QRS-T-Beziehung")
    qrst = xyz[q0:t_end] if t_end else None
    svg = area_vector_mv_ms(qrst, fs) if qrst is not None else None
    out.vectors["svg"] = svg
    add("svg_magnitude", "Räumlicher ventrikulärer Gradient |∫QRST|", None if svg is None else np.linalg.norm(svg), "mV·ms", "QRS-T-Beziehung", vector=svg)
    add("svg_azimuth", "SVG Azimut", azimuth_deg(svg) if svg is not None else None, "°", "QRS-T-Beziehung")
    add("svg_elevation", "SVG Elevation", elevation_deg(svg) if svg is not None else None, "°", "QRS-T-Beziehung")
    add("sai_qrst", "SAI QRST (Σ|∫X|+|∫Y|+|∫Z|)", sum_absolute_integral(qrst, fs) if qrst is not None else None, "mV·ms", "QRS-T-Beziehung")
    # ---- ST-Vektor (Verletzungsstrom) am J-Punkt und J+60 ms
    for name, off in (("st_j", 0), ("st_j60", 60)):
        k = q1 + int(round(off * fs / 1000.0))
        if 0 <= k < n:
            w = xyz[max(0, k - 1):min(n, k + 2)].mean(0)
            out.vectors[name] = w
            add(f"{name}_magnitude", f"ST-Vektor {'J' if off == 0 else 'J+60 ms'} Betrag", np.linalg.norm(w), "mV", "ST-Vektor", vector=w)
            add(f"{name}_azimuth", f"ST-Vektor {'J' if off == 0 else 'J+60 ms'} Azimut", azimuth_deg(w), "°", "ST-Vektor")
            add(f"{name}_elevation", f"ST-Vektor {'J' if off == 0 else 'J+60 ms'} Elevation", elevation_deg(w), "°", "ST-Vektor")
    # ---- Schleifengeometrie
    out.qrs_geometry = loop_geometry(qrs)
    g = out.qrs_geometry
    add("qrs_planarity", "QRS-Schleife Nichtplanarität λ3/λ2", None if g is None else g.planarity_ratio, "", "Schleifenform",
        note="0 = eben; Ebenheitsverlust bei STEMI (Ray 2017)")
    add("qrs_plane_rms", "QRS-Schleife Abstand zur Ausgleichsebene", None if g is None else g.plane_rms_uv, "µV", "Schleifenform")
    add("qrs_roundness", "QRS-Schleife Rundheit λ2/λ1", None if g is None else g.roundness, "", "Schleifenform")
    add("qrs_perimeter", "QRS-Schleife Umfang", None if g is None else g.perimeter_mv, "mV", "Schleifenform", note="Correa 2013")
    add("qrs_planar_area", "QRS-Schleife Fläche in der Ebene", None if g is None else g.planar_area_mv2, "mV²", "Schleifenform")
    add("qrs_hull_volume", "QRS-Schleife Hüllvolumen", None if g is None or g.hull_volume_mv3 is None else g.hull_volume_mv3 * 1000.0, "×10⁻³ mV³", "Schleifenform", note="Correa 2013: bestes Einzelmaß für Ischämie")
    add("qrs_plane_normal_azimuth", "QRS-Ebene Normale Azimut", None if g is None else azimuth_deg(g.normal), "°", "Schleifenform")
    add("qrs_plane_normal_elevation", "QRS-Ebene Normale Elevation", None if g is None else elevation_deg(g.normal), "°", "Schleifenform")
    out.t_geometry = loop_geometry(t) if t is not None else None
    g = out.t_geometry
    add("t_planarity", "T-Schleife Nichtplanarität λ3/λ2", None if g is None else g.planarity_ratio, "", "Schleifenform")
    add("t_plane_rms", "T-Schleife Abstand zur Ebene (Tavplan-Analogon)", None if g is None else g.plane_rms_uv, "µV", "Schleifenform",
        note="erhöht → künftiger Infarkt (Rubulis 2010)")
    add("t_roundness", "T-Schleife Rundheit λ2/λ1", None if g is None else g.roundness, "", "Schleifenform")
    out.p_geometry = loop_geometry(p) if p is not None else None
    pv, _ = peak_vector(p) if p is not None else (None, -1)
    out.vectors["p_max"] = pv
    add("p_max_magnitude", "P-Maximalvektor Betrag", None if pv is None else np.linalg.norm(pv), "mV", "P-Schleife", vector=pv)
    add("p_max_azimuth", "P-Maximalvektor Azimut", azimuth_deg(pv) if pv is not None else None, "°", "P-Schleife")
    add("p_max_elevation", "P-Maximalvektor Elevation", elevation_deg(pv) if pv is not None else None, "°", "P-Schleife")
    pt = window_mean(p, fs, 30, terminal=True) if p is not None else None
    out.vectors["p_terminal"] = pt
    add("p_terminal_azimuth", "P terminale 30 ms: Azimut", azimuth_deg(pt) if pt is not None else None, "°", "P-Schleife",
        note="posterior-superior terminal → interatrialer Block (Pérez-Riera 2021)")
    add("p_terminal_elevation", "P terminale 30 ms: Elevation", elevation_deg(pt) if pt is not None else None, "°", "P-Schleife")
    add("p_duration", "P-Dauer (Delineation)", None if p is None else len(p) * 1000.0 / fs, "ms", "P-Schleife")
    # ---- Drehrichtungen und Oktanten
    for seg_name, seg in (("qrs", qrs), ("t", t), ("p", p)):
        if seg is None or len(seg) < 4:
            continue
        out.inscriptions[seg_name] = {pl: inscription(seg, pl) for pl in PLANES}
    out.octants = octant_dwell(qrs)
    return out


# ------------------------------------------------------------------ Brugada-JT-Abstand (Madaloso 2022)
def jt_distance(xyz: np.ndarray, fs: float, b: BeatBounds, plane: str) -> Optional[float]:
    """Abstand (mV) zwischen dem letzten QRS-Drittel (um J) und dem T-Schleifenbeginn in der Ebene.

    Madaloso 2022 misst den senkrechten Abstand zweier Parallelen: durch das J-nahe QRS-Ende und durch
    den Anfang der T-Schleife. Hier: Abstand des Schwerpunkts der ersten 20 ms der T-Schleife von der
    Geraden durch die letzten 20 ms des QRS (Projektion in die Ebene). 1 mm ≙ 0,1 mV.
    """
    if b.t_on is None or b.t_end is None or b.t_on <= b.qrs_off:
        return None
    n = len(xyz)
    k = max(2, int(round(0.02 * fs)))
    q = screen_coords(xyz[max(0, b.qrs_off - k):b.qrs_off], plane)
    t = screen_coords(xyz[b.t_on:min(n, b.t_on + k)], plane)
    if len(q) < 2 or len(t) < 1:
        return None
    d = q[-1] - q[0]
    if np.linalg.norm(d) < EPS:
        return float(np.linalg.norm(t.mean(0) - q[-1]))
    d = d / np.linalg.norm(d)
    normal = np.array([-d[1], d[0]])
    return float(abs(np.dot(t.mean(0) - q[-1], normal)))


# ------------------------------------------------------------------ Schlag-zu-Schlag und seriell
@dataclass
class VariabilityResult:
    n_beats: int
    qrs_axis_sd_deg: Optional[float]
    qrs_magnitude_cv: Optional[float]
    t_axis_sd_deg: Optional[float]
    t_magnitude_cv: Optional[float]
    qrs_t_angle_sd_deg: Optional[float]


def _angular_sd(vectors: list[np.ndarray]) -> Optional[float]:
    vs = [v / np.linalg.norm(v) for v in vectors if v is not None and np.linalg.norm(v) > EPS]
    if len(vs) < 3:
        return None
    m = np.mean(vs, axis=0)
    if np.linalg.norm(m) < EPS:
        return None
    m /= np.linalg.norm(m)
    ang = [np.degrees(np.arccos(np.clip(np.dot(v, m), -1, 1))) for v in vs]
    return float(np.sqrt(np.mean(np.square(ang))))


def _cv(values: list[float]) -> Optional[float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 3 or v.mean() < EPS:
        return None
    return float(v.std() / v.mean())


def beat_variability(per_beat: list[BeatMeasures]) -> VariabilityResult:
    """Schlag-zu-Schlag-Variabilität der Hauptvektoren (Hasan 2012, PLoS One 7:e49489)."""
    qv = [m.vectors.get("qrs_max") for m in per_beat]
    tv = [m.vectors.get("t_max") for m in per_beat]
    return VariabilityResult(
        n_beats=len(per_beat),
        qrs_axis_sd_deg=_angular_sd(qv),
        qrs_magnitude_cv=_cv([None if v is None else float(np.linalg.norm(v)) for v in qv]),
        t_axis_sd_deg=_angular_sd(tv),
        t_magnitude_cv=_cv([None if v is None else float(np.linalg.norm(v)) for v in tv]),
        qrs_t_angle_sd_deg=(lambda a: None if len(a) < 3 else float(np.std(a)))(
            [m.get("qrs_t_angle_peak") for m in per_beat if m.get("qrs_t_angle_peak") is not None]),
    )


@dataclass
class SerialDifference:
    delta_st_mv: float
    delta_st_vector: np.ndarray
    delta_vg_mv_ms: float
    delta_vg_vector: np.ndarray
    angle_deg: Optional[float]
    ischemia_flag: bool


def serial_difference(before: BeatMeasures, after: BeatMeasures) -> Optional[SerialDifference]:
    """Differenzvektoren ΔST (J-Punkt) und ΔVG zweier Aufnahmen (ter Haar 2013; Treskes 2015).

    Schwellen ΔST > 0,05 mV, ΔVG > 16,2 mV·ms; die Autoren fanden ΔVG ≈ 324·ΔST (r 0,85)."""
    a, b = before.vectors.get("st_j"), after.vectors.get("st_j")
    va, vb = before.vectors.get("svg"), after.vectors.get("svg")
    if a is None or b is None or va is None or vb is None:
        return None
    dst, dvg = b - a, vb - va
    return SerialDifference(float(np.linalg.norm(dst)), dst, float(np.linalg.norm(dvg)), dvg, angle_deg(dst, dvg),
                            bool(np.linalg.norm(dst) > 0.05 and np.linalg.norm(dvg) > 16.2))


# ------------------------------------------------------------------ Transformationsgüte
DOWER_FORWARD = np.array([
    [0.632, -0.235, 0.059],   # I
    [0.235, 1.066, -0.132],   # II
    [-0.515, 0.157, -0.917],  # V1
    [0.044, 0.164, -1.387],   # V2
    [0.882, 0.098, -1.277],   # V3
    [1.213, 0.127, -0.601],   # V4
    [1.125, 0.127, -0.086],   # V5
    [0.831, 0.076, 0.230],    # V6
])
DOWER_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


def transform_fidelity(ecg12: np.ndarray, xyz: np.ndarray, lead_names: tuple[str, ...]) -> dict[str, float]:
    """Wie gut erklärt das abgeleitete VCG die acht unabhängigen Ableitungen (Dower-Vorwärtsmatrix)?

    Pro Ableitung Korrelation zwischen gemessen und aus XYZ rekonstruiert; niedrige Werte zeigen,
    dass die Standardmatrix diesen Thorax schlecht beschreibt (Vorsicht bei Richtungsmaßen)."""
    out: dict[str, float] = {}
    rec = xyz @ DOWER_FORWARD.T
    names = list(lead_names)
    for j, lead in enumerate(DOWER_LEADS):
        if lead not in names:
            continue
        m = ecg12[:, names.index(lead)]
        r = rec[:, j]
        if m.std() < EPS or r.std() < EPS:
            continue
        out[lead] = float(np.corrcoef(m, r)[0, 1])
    return out
