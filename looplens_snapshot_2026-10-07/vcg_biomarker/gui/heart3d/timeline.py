"""Schlagzeitleiste: Ereigniszeiten je Schlag aus der Laboranalyse oder aus einer Schnellschätzung.

Elektrische Zeiten (P, QRS, T) kommen aus der Delineation der Laboranalyse, wenn
vorhanden; sonst aus einer leichten R-Zacken-Erkennung mit Vorlagenabständen
(als ``estimated`` gekennzeichnet). Mechanische Zeiten (Klappen, Ejektion) folgen
den systolischen Zeitintervallen nach Weissler 1968 als Funktion der Herzfrequenz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# Segmentnamen der Laborausgabe (``_wave_segments_for_beat``) auf Kurzformen abbilden
SEGMENT_ALIASES = {"p_wave": "p", "qrs_complex": "qrs", "t_wave": "t", "pr_segment": "pr", "st_segment": "st", "u_wave": "u"}

# Dower-Transformation: I, II, V1..V6 aus dem Frank-VCG (X, Y, Z) (Dower 1980; Edenbrandt & Pahlm 1988).
# Empirische Übertragung im realen, begrenzten Thorax – damit werden simulierte Herzvektoren in Ableitungen
# umgerechnet, ohne die Grenzflächeneffekte des Thorax selbst modellieren zu müssen.
DOWER = np.array([
    [0.632, -0.235, 0.059],   # I
    [0.235, 1.066, -0.132],   # II
    [-0.515, 0.157, -0.917],  # V1
    [0.044, 0.164, -1.387],   # V2
    [0.882, 0.098, -1.277],   # V3
    [1.213, 0.127, -0.601],   # V4
    [1.125, 0.127, -0.086],   # V5
    [0.831, 0.076, 0.230],    # V6
])

# Kors-Regression (X, Y, Z) aus I, II, V1..V6 (Kors et al. 1990)
KORS = np.array([
    [0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],
    [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],
    [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31],
])


@dataclass
class Beat:
    index: int
    r_time: float
    qrs_on: float
    qrs_off: float
    p_on: Optional[float]
    p_off: Optional[float]
    t_on: Optional[float]
    t_end: Optional[float]
    rr_prev: Optional[float]
    beat_type: str = "N"  # N normal/dominant, V ventrikulär/sekundäre Morphologie, S supraventrikulär, P paced, ? unbekannt
    cluster_id: int = 0
    source: str = "estimated"  # payload | estimated
    accepted: bool = True
    qrs_vector: Optional[np.ndarray] = None  # mittlerer QRS-Vektor (X links, Y inferior, Z posterior) in mV
    qrs_measured: bool = False  # explicit, assessable segment from the saved laboratory result
    t_measured: bool = False
    p_measured: bool = False

    @property
    def has_p(self) -> bool:
        return self.p_on is not None and self.p_off is not None

    @property
    def hr(self) -> float:
        if self.rr_prev and self.rr_prev > 0.2:
            return 60.0 / self.rr_prev
        return 70.0

    @property
    def qrs_duration_ms(self) -> float:
        return max(40.0, (self.qrs_off - self.qrs_on) * 1000.0)

    @property
    def p_duration_ms(self) -> float:
        if self.has_p:
            return max(50.0, (self.p_off - self.p_on) * 1000.0)
        return 100.0

    @property
    def is_ectopic(self) -> bool:
        return self.beat_type in ("V", "S")


@dataclass
class MechanicalEvents:
    """Zeiten in Sekunden (absolut) für einen Schlag."""
    atrial_contraction_on: Optional[float]
    atrial_contraction_off: Optional[float]
    av_close: float  # isovolumetrische Kontraktion beginnt
    ao_open: float
    ao_close: float
    av_open: float
    pep_ms: float
    lvet_ms: float
    qs2_ms: float
    ivrt_ms: float


def weissler(hr: float, female: bool = False) -> tuple[float, float, float]:
    """(PEP, LVET, QS2) in ms nach Weissler 1968; Frauen um wenige ms länger."""
    hr = float(np.clip(hr, 35.0, 180.0))
    pep = 131.0 - 0.4 * hr + (2.0 if female else 0.0)
    lvet = 413.0 - 1.7 * hr + (5.0 if female else 0.0)
    qs2 = 546.0 - 2.1 * hr + (3.0 if female else 0.0)
    return pep, lvet, qs2


def mechanical_events(beat: Beat, female: bool = False) -> MechanicalEvents:
    pep, lvet, qs2 = weissler(beat.hr, female)
    ivrt = float(np.clip(90.0 - 0.2 * beat.hr, 40.0, 100.0))
    # Ejektionsende nicht später als kurz nach dem T-Ende, wenn dieses vorliegt
    ao_close = beat.qrs_on + (pep + lvet) / 1000.0
    if beat.t_end is not None:
        ao_close = min(ao_close, beat.t_end + 0.04)
        ao_close = max(ao_close, beat.qrs_on + pep / 1000.0 + 0.12)
    return MechanicalEvents(
        atrial_contraction_on=(beat.p_on + 0.08) if beat.has_p else None,
        atrial_contraction_off=(beat.p_off + 0.12) if beat.has_p else None,
        av_close=beat.qrs_on + 0.045,
        ao_open=beat.qrs_on + pep / 1000.0,
        ao_close=ao_close,
        av_open=ao_close + ivrt / 1000.0,
        pep_ms=pep, lvet_ms=lvet, qs2_ms=qs2, ivrt_ms=ivrt,
    )


@dataclass
class BeatTimeline:
    beats: list[Beat]
    duration: float
    fs: float
    rhythm_class: str = "unknown"
    rhythm_flags: tuple[str, ...] = ()
    conduction_hints: dict[str, Any] = field(default_factory=dict)  # lbbb, rbbb, paced, wpw, lafb, lpfb, av_block
    feature_values: dict[str, Any] = field(default_factory=dict)
    source: str = "estimated"
    female: bool = False
    signal: Optional[np.ndarray] = None  # (n, leads) nativ, für den Rhythmusstreifen
    lead_names: tuple[str, ...] = LEADS_12
    atrial_events: Optional[list[tuple[float, float]]] = None  # unabhängige P-Wellen (AV-Dissoziation)
    label: str = ""

    def atrial_event_at(self, t: float) -> Optional[tuple[float, float]]:
        """Letzte unabhängige Vorhoferregung, die vor t begann (nur bei AV-Dissoziation gesetzt)."""
        if not self.atrial_events:
            return None
        best = None
        for p_on, p_off in self.atrial_events:
            if p_on - 0.02 <= t:
                best = (p_on, p_off)
            else:
                break
        return best

    def beat_at(self, t: float) -> Optional[Beat]:
        """Der Schlag, dessen Zyklus t enthält (vom P-Beginn bzw. QRS-Beginn bis zum nächsten)."""
        if not self.beats:
            return None
        starts = np.array([self._cycle_start(b) for b in self.beats])
        k = int(np.searchsorted(starts, t, side="right") - 1)
        if k < 0:
            return self.beats[0]
        return self.beats[k]

    def cycle_bounds(self, beat: Beat) -> tuple[float, float]:
        k = beat.index
        start = self._cycle_start(beat)
        if k + 1 < len(self.beats):
            end = self._cycle_start(self.beats[k + 1])
        else:
            end = min(self.duration, beat.qrs_on + max(1.0, beat.rr_prev or 1.0))
        return start, max(end, start + 0.2)

    @staticmethod
    def _cycle_start(beat: Beat) -> float:
        return (beat.p_on - 0.02) if beat.has_p else (beat.qrs_on - 0.12)

    @property
    def median_hr(self) -> float:
        rr = [b.rr_prev for b in self.beats if b.rr_prev]
        return 60.0 / float(np.median(rr)) if rr else 70.0

    def is_af_like(self) -> bool:
        return self.rhythm_class in ("af_compatible_pattern",) or "no_consistent_discrete_p" in self.rhythm_flags and self.rhythm_class == "irregular"

    def is_flutter_like(self) -> bool:
        return self.rhythm_class == "flutter_compatible_pattern"


# ----------------------------------------------------------------------- Aufbau
def _bandpass(x: np.ndarray, fs: float, lo: float = 5.0, hi: float = 20.0) -> np.ndarray:
    nyq = 0.5 * fs
    hi = min(hi, 0.45 * fs)
    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def detect_r_peaks(signal: np.ndarray, fs: float, lead_names: Iterable[str] = LEADS_12) -> np.ndarray:
    """Leichte R-Zacken-Erkennung (Energiehülle), nur für die Illustration."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    names = list(lead_names)
    lead = names.index("II") if "II" in names and signal.shape[1] > names.index("II") else 0
    x = signal[:, lead]
    x = x - np.nanmedian(x)
    x = np.nan_to_num(x)
    if len(x) < int(fs):
        return np.zeros(0, dtype=int)
    f = _bandpass(x, fs)
    energy = f ** 2
    win = max(1, int(0.12 * fs))
    env = np.convolve(energy, np.ones(win) / win, mode="same")
    threshold = 0.25 * np.percentile(env, 99)
    peaks, _ = find_peaks(env, height=threshold, distance=int(0.28 * fs))
    # Auf das lokale Maximum des Betrags im Rohsignal schieben
    refined = []
    half = int(0.05 * fs)
    for p in peaks:
        a, b = max(0, p - half), min(len(x), p + half)
        seg = np.abs(x[a:b])
        refined.append(a + int(np.argmax(seg)))
    return np.asarray(sorted(set(refined)), dtype=int)


def qrs_vector_from_signal(signal: np.ndarray, fs: float, qrs_on: float, qrs_off: float,
                           lead_names: Iterable[str] = LEADS_12) -> Optional[np.ndarray]:
    """Mittlerer QRS-Vektor (Kors) eines Schlags; Referenz ist das PR-Segment davor."""
    names = list(lead_names)
    try:
        cols = [names.index(n) for n in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]
    except ValueError:
        return None
    a, b = int(qrs_on * fs), int(qrs_off * fs)
    if b - a < 3 or a < 0 or b > len(signal):
        return None
    base_a = max(0, a - int(0.06 * fs))
    base = signal[base_a:max(base_a + 1, a - int(0.01 * fs)), :][:, cols].mean(0)
    window = signal[a:b, :][:, cols] - base
    xyz = KORS @ window.mean(0)
    if not np.all(np.isfinite(xyz)):
        return None
    return xyz


def timeline_from_signal(signal: np.ndarray, fs: float, lead_names: Iterable[str] = LEADS_12) -> BeatTimeline:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    names = tuple(lead_names)
    r = detect_r_peaks(signal, fs, names) / fs
    beats: list[Beat] = []
    for k, rt in enumerate(r):
        rr = (rt - r[k - 1]) if k > 0 else (r[1] - r[0] if len(r) > 1 else None)
        qt = 0.40 * np.sqrt(np.clip(rr if rr else 0.85, 0.3, 2.0))
        qrs_on, qrs_off = rt - 0.040, rt + 0.050
        premature = bool(rr and k > 1 and rr < 0.8 * (r[k - 1] - r[k - 2]))
        beat = Beat(index=k, r_time=float(rt), qrs_on=float(qrs_on), qrs_off=float(qrs_off),
                    p_on=None if premature else float(qrs_on - 0.16), p_off=None if premature else float(qrs_on - 0.05),
                    t_on=float(qrs_off + 0.06), t_end=float(qrs_on + qt), rr_prev=float(rr) if rr else None,
                    beat_type="V" if premature else "N", source="estimated")
        beat.qrs_vector = qrs_vector_from_signal(signal, fs, qrs_on, qrs_off, names)
        beats.append(beat)
    _flag_secondary_morphology(beats)
    duration = len(signal) / fs
    rr = np.diff(r) if len(r) > 2 else np.zeros(0)
    rhythm = "unknown"
    if rr.size >= 3:
        cv = float(np.std(rr) / max(np.mean(rr), 1e-6))
        rhythm = "irregular" if cv >= 0.12 else "regular"
    return BeatTimeline(beats, duration, fs, rhythm_class=rhythm, source="estimated", signal=signal, lead_names=names)


def _flag_secondary_morphology(beats: list[Beat]) -> None:
    """Schläge, deren QRS-Vektor stark vom dominanten abweicht, als sekundäre Morphologie kennzeichnen."""
    vectors = [b.qrs_vector for b in beats if b.qrs_vector is not None]
    if len(vectors) < 3:
        return
    reference = np.median(np.asarray(vectors), axis=0)
    ref_norm = np.linalg.norm(reference)
    if ref_norm < 1e-6:
        return
    for beat in beats:
        if beat.qrs_vector is None:
            continue
        cos = float(np.dot(beat.qrs_vector, reference) / (np.linalg.norm(beat.qrs_vector) * ref_norm + 1e-9))
        if cos < 0.5 and beat.source == "estimated":
            beat.beat_type = "V"
            beat.p_on = beat.p_off = None


def timeline_from_payload(payload: dict, signal: np.ndarray, fs: float,
                          lead_names: Iterable[str] = LEADS_12) -> BeatTimeline:
    """Zeitleiste aus der Laboranalyse; display estimates remain distinguishable."""
    signal = np.asarray(signal, dtype=np.float64)
    names = tuple(lead_names)
    lab = payload.get("lab_processing", {})
    analysis_fs = float(lab.get("analysis_sampling_rate_hz", fs) or fs)
    inventory = list(payload.get("beat_inventory", ()))
    delineation = payload.get("delineation", {})
    segments_by_beat: dict[int, dict[str, dict]] = {}
    for seg in delineation.get("accepted_beat_wave_segments", ()) or ():
        try:
            bi = int(seg.get("beat_index"))
        except (TypeError, ValueError):
            continue
        name = SEGMENT_ALIASES.get(str(seg.get("segment", "")), str(seg.get("segment", "")))
        if seg.get("start_sample") is None or seg.get("end_sample") is None:
            continue
        segments_by_beat.setdefault(bi, {})[name] = seg
    beats: list[Beat] = []
    prev_r: Optional[float] = None
    for k, item in enumerate(inventory):
        r_sample = item.get("r_peak_sample")
        if r_sample is None:
            continue
        rt = float(r_sample) / analysis_fs
        segs = segments_by_beat.get(int(item.get("beat_index", k)), {})

        def bounds(name: str) -> tuple[Optional[float], Optional[float]]:
            s = segs.get(name)
            if not s:
                return None, None
            return float(s["start_sample"]) / analysis_fs, float(s["end_sample"]) / analysis_fs

        def verified(name: str) -> bool:
            s = segs.get(name)
            return bool(s and s.get('status') == 'measurable'
                        and s.get('start_sample') is not None and s.get('end_sample') is not None
                        and int(s['end_sample']) > int(s['start_sample']))

        p_on, p_off = bounds("p")
        q_on, q_off = bounds("qrs")
        t_on, t_end = bounds("t")
        if q_on is None or q_off is None:
            q_on, q_off = rt - 0.040, rt + 0.050
        rr = (rt - prev_r) if prev_r is not None else None
        cluster = int(item.get("morphology_cluster_id", 0) or 0)
        beat_type = _normalize_beat_type(item.get("beat_type"))
        if beat_type is None:
            # Morphology cluster membership does not establish an atrial or
            # ventricular focus. Preserve the unknown beat diagnosis.
            beat_type = "?"
        if t_end is None:
            qt = 0.40 * np.sqrt(np.clip(rr if rr else 0.85, 0.3, 2.0))
            t_on, t_end = q_off + 0.06, q_on + qt
        beat = Beat(index=k, r_time=rt, qrs_on=q_on, qrs_off=q_off, p_on=p_on, p_off=p_off, t_on=t_on, t_end=t_end,
                    rr_prev=rr, beat_type=beat_type, cluster_id=cluster, source="payload",
                    accepted=bool(item.get("accepted", True)), qrs_measured=verified('qrs'),
                    t_measured=verified('t'), p_measured=verified('p'))
        beat.qrs_vector = qrs_vector_from_signal(signal, fs, q_on, q_off, names)
        beats.append(beat)
        prev_r = rt
    if not beats:
        return timeline_from_signal(signal, fs, names)
    for idx, beat in enumerate(beats):
        beat.index = idx
    context = delineation.get("rhythm_context", {}) or {}
    hints = _conduction_hints(payload)
    for key in ("atrial_frequency_hz", "independent_atrial_rate_hz", "p_wave_coverage"):
        value = context.get(key)
        if value is not None:
            try:
                hints[key] = float(value)
            except (TypeError, ValueError):
                pass
    return BeatTimeline(beats, len(signal) / fs, fs, rhythm_class=str(context.get("class", "unknown")),
                        rhythm_flags=tuple(context.get("flags", ()) or ()), conduction_hints=hints,
                        feature_values=dict(payload.get("feature_values", {}) or {}), source="payload",
                        signal=signal, lead_names=names)


def _normalize_beat_type(value: object) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text or text in ("unknown", "?", "none", "nicht dokumentiert"):
        return None
    if text in ("v", "ventricular", "pvc", "ves", "ectopic_ventricular", "vt"):
        return "V"
    if text in ("s", "supraventricular", "pac", "sves", "atrial", "ectopic_atrial"):
        return "S"
    if text in ("p", "paced", "pacing", "stimulated"):
        return "P"
    if text in ("n", "normal", "dominant", "sinus", "primary"):
        return "N"
    return None


def _positive(criteria: dict, key: str) -> Optional[bool]:
    item = criteria.get(key)
    if not isinstance(item, dict):
        return None
    value = item.get("positive")
    return bool(value) if value is not None else None


def _conduction_hints(payload: dict) -> dict[str, Any]:
    criteria = payload.get("classical_criteria", {}) or {}
    features = payload.get("feature_values", {}) or {}
    hints: dict[str, Any] = {
        "lbbb": _positive(criteria, "lbbb_pattern"),
        "rbbb": _positive(criteria, "rbbb_pattern"),
        "first_degree_av_block": _positive(criteria, "first_degree_av_block"),
        "left_axis": _positive(criteria, "left_axis_deviation"),
        "right_axis": _positive(criteria, "right_axis_deviation"),
    }
    delta = features.get("delta_wave_present_flag")
    hints["wpw"] = bool(delta) if delta is not None and np.isfinite(float(delta or 0)) else None
    hints["delta_axis_deg"] = features.get("delta_wave_frontal_axis_deg")
    spike = features.get("pacing_spike_slope_ratio")
    hints["paced"] = (float(spike) > 4.0) if spike is not None and np.isfinite(float(spike)) else None
    for key in ("qrs_arc_length_fraction_at_half_time", "qrs_early_cancellation_fraction", "qrs_cancellation_fraction",
                "qrs_stall_direction_leftward", "qrs_stall_direction_posterior", "qrs_stall_direction_inferior",
                "qrs_spatial_velocity_initial_burst_ms", "qrs_spatial_velocity_first_to_last_peak_ms",
                "primary_morphology_qrs_octant_id", "secondary_morphology_qrs_octant_id",
                "secondary_morphology_maximum_deflection_index"):
        value = features.get(key)
        if value is not None:
            try:
                hints[key] = float(value)
            except (TypeError, ValueError):
                pass
    return hints
