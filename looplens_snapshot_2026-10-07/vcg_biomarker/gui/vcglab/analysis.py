"""Datenmodell des VCG-Labors und lesende Kopplung an die Analysekette des Hauptprojekts.

``analyze_signal`` rechnet – wie das ECG-Labor – Signalpfade, Mehrkanal-R-Erkennung, Fusions-
Delineation, physiologischen VCG-Ursprung, den fiducial-gewarpten Medianschlag und die klassischen
Kriterien; Ergebnis ist ein ``VCGRecord`` mit beiden Transformationen (Kors, inverse Dower).
Schlägt ``core`` fehl, liefert ``quick_record`` eine Schnellschätzung aus ``heart3d.timeline``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .measures import BeatBounds, BeatMeasures, beat_measures, transform_fidelity

log = logging.getLogger(__name__)

LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
ANALYSIS_FS = 500
METHODS = ("kors", "inverse_dower")
METHOD_LABELS = {"kors": "Kors (Regression, 1990)", "inverse_dower": "Inverse Dower (1988)"}

@dataclass
class VCGBeat:
    index: int
    r: int
    qrs_on: int
    qrs_off: int
    p_on: Optional[int] = None
    p_off: Optional[int] = None
    t_on: Optional[int] = None
    t_peak: Optional[int] = None
    t_end: Optional[int] = None
    beat_type: str = "N"
    cluster: int = 0
    accepted: bool = True
    outlier: bool = False

    def bounds(self, offset: int = 0) -> BeatBounds:
        sh = lambda v: None if v is None else v - offset  # noqa: E731
        return BeatBounds(self.qrs_on - offset, self.qrs_off - offset, sh(self.t_end), sh(self.t_on), sh(self.p_on), sh(self.p_off), sh(self.t_peak))

    @property
    def label(self) -> str:
        flag = "" if self.beat_type == "N" else f" {self.beat_type}"
        return f"#{self.index + 1}{flag}" + ("" if self.accepted else " (verworfen)")


@dataclass
class OriginInfo:
    vector: np.ndarray
    method: str
    confidence: float
    note: str


@dataclass
class VCGRecord:
    ecg: np.ndarray  # (n, 12) mV bei ANALYSIS_FS, diagnostischer Pfad
    fs: float
    lead_names: tuple[str, ...]
    label: str = ""
    age: Optional[float] = None
    sex: Optional[str] = None
    beats: list[VCGBeat] = field(default_factory=list)
    vcg_raw: dict[str, np.ndarray] = field(default_factory=dict)
    origin: dict[str, OriginInfo] = field(default_factory=dict)
    median_ecg: Optional[np.ndarray] = None  # Record mit fiducial-gewarptem Medianzyklus (core)
    median_beat: Optional[VCGBeat] = None  # Fiducials des Repräsentativschlags im Median-Record
    rhythm_class: str = "unknown"
    rhythm_flags: tuple[str, ...] = ()
    criteria: dict = field(default_factory=dict)  # klassische Kriterien (to_dict)
    payload: dict = field(default_factory=dict)
    delineation: object = None
    source: str = "quick"  # core | quick | payload
    notes: list[str] = field(default_factory=list)
    fidelity: dict[str, dict[str, float]] = field(default_factory=dict)

    # ---- Zugriff
    def xyz(self, method: str = "kors", corrected: bool = True) -> np.ndarray:
        raw = self.vcg_raw[method]
        if corrected and method in self.origin:
            return raw - self.origin[method].vector[None, :]
        return raw

    def median_xyz(self, method: str = "kors", corrected: bool = True) -> Optional[np.ndarray]:
        if self.median_ecg is None:
            return None
        raw = derive_vcg(self.median_ecg, self.lead_names, method)
        if corrected and method in self.origin:
            return raw - self.origin[method].vector[None, :]
        return raw

    @property
    def dominant_cluster(self) -> int:
        counts: dict[int, int] = {}
        for b in self.beats:
            if b.accepted and not b.outlier:
                counts[b.cluster] = counts.get(b.cluster, 0) + 1
        return max(counts, key=counts.get) if counts else 0

    def representative_index(self) -> int:
        """Schlag, dessen QRS-Maximalvektor dem Median der dominanten Population am nächsten liegt."""
        cand = [b for b in self.beats if b.accepted and not b.outlier and b.cluster == self.dominant_cluster]
        if not cand:
            cand = [b for b in self.beats if b.accepted] or self.beats
        if not cand:
            return 0
        xyz = self.xyz()
        vecs = []
        for b in cand:
            seg = xyz[b.qrs_on:b.qrs_off]
            vecs.append(seg[np.argmax(np.linalg.norm(seg, axis=1))] if len(seg) else np.zeros(3))
        med = np.median(np.array(vecs), axis=0)
        k = int(np.argmin([np.linalg.norm(v - med) for v in vecs]))
        return cand[k].index

    def beat_slice(self, beat: VCGBeat, pre_ms: float = 80.0, post_ms: float = 120.0) -> tuple[int, int]:
        n = len(self.ecg)
        start = beat.p_on if beat.p_on is not None else beat.qrs_on
        end = beat.t_end if beat.t_end is not None else beat.qrs_off + int(0.4 * self.fs)
        a = max(0, start - int(pre_ms * self.fs / 1000.0))
        b = min(n, end + int(post_ms * self.fs / 1000.0))
        if beat.index + 1 < len(self.beats):
            nb = self.beats[beat.index + 1]
            b = min(b, (nb.p_on if nb.p_on is not None else nb.qrs_on) - 1)
        return a, max(a + 2, b)

    def measures_for(self, beat: VCGBeat, method: str = "kors", reference: Optional[dict] = None) -> BeatMeasures:
        return beat_measures(self.xyz(method), self.fs, beat.bounds(), self.sex, reference)

    def median_measures(self, method: str = "kors", reference: Optional[dict] = None) -> Optional[BeatMeasures]:
        xyz = self.median_xyz(method)
        if xyz is None or self.median_beat is None:
            return None
        return beat_measures(xyz, self.fs, self.median_beat.bounds(), self.sex, reference)

    def dominant_beats(self) -> list[VCGBeat]:
        dc = self.dominant_cluster
        return [b for b in self.beats if b.accepted and not b.outlier and b.cluster == dc]


# ------------------------------------------------------------------ Transformationen
def reorder(signal: np.ndarray, names: Iterable[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    names = [str(n) for n in names]
    if all(n in names for n in LEADS_12):
        return signal[:, [names.index(n) for n in LEADS_12]], LEADS_12
    raise ValueError("Zwölf Standardableitungen erforderlich (I, II, III, aVR, aVL, aVF, V1–V6)")


def derive_vcg(ecg12: np.ndarray, names: tuple[str, ...], method: str = "kors") -> np.ndarray:
    from vcg_biomarker.core.vcg_transform import VCGTransformRegistry

    ordered, _ = reorder(np.asarray(ecg12, dtype=float), names)
    return VCGTransformRegistry.transform(ordered, method).xyz


def quick_origin(xyz: np.ndarray, beats: list[VCGBeat], fs: float) -> OriginInfo:
    """Ursprung ohne core: Median des 40-ms-Fensters, das 20 ms vor dem QRS-Beginn endet (PR-Nähe)."""
    w, gap = int(0.04 * fs), int(0.02 * fs)
    rows = [xyz[b.qrs_on - gap - w:b.qrs_on - gap] for b in beats if b.qrs_on - gap - w >= 0]
    rows = [r for r in rows if len(r)]
    if not rows:
        return OriginInfo(np.zeros(3), "none", 0.0, "kein Fenster vor QRS")
    stack = np.vstack(rows)
    return OriginInfo(np.median(stack, axis=0), "pr_quick", 0.3, f"PR-Fenster (40 ms) über {len(rows)} Schläge, Schnellschätzung")


# ------------------------------------------------------------------ Schnellpfad (heart3d-Zeitleiste)
def quick_record(signal: np.ndarray, fs: float, names: Iterable[str] = LEADS_12, label: str = "",
                 age: Optional[float] = None, sex: Optional[str] = None) -> VCGRecord:
    from ..heart3d.timeline import timeline_from_signal
    sig, lead_names = reorder(np.asarray(signal, dtype=float), names)
    if not np.isclose(fs, ANALYSIS_FS):
        from scipy.signal import resample_poly
        from fractions import Fraction
        fr = Fraction(ANALYSIS_FS / fs).limit_denominator(1000)
        sig = resample_poly(sig, fr.numerator, fr.denominator, axis=0)
        fs = float(ANALYSIS_FS)
    sig = sig - np.median(sig, axis=0)
    tl = timeline_from_signal(sig, fs, lead_names)
    beats = []
    for b in tl.beats:
        s = lambda v: None if v is None else int(round(v * fs))  # noqa: E731
        beats.append(VCGBeat(b.index, s(b.r_time), s(b.qrs_on), s(b.qrs_off), s(b.p_on), s(b.p_off), s(b.t_on), None, s(b.t_end),
                             b.beat_type, b.cluster_id, b.accepted, b.beat_type == "V"))
    rec = VCGRecord(sig, fs, lead_names, label, age, sex, beats, rhythm_class=tl.rhythm_class, rhythm_flags=tuple(tl.rhythm_flags), source="quick")
    for m in METHODS:
        rec.vcg_raw[m] = derive_vcg(sig, lead_names, m)
        rec.origin[m] = quick_origin(rec.vcg_raw[m], beats, fs)
        rec.fidelity[m] = transform_fidelity(sig, rec.vcg_raw[m], lead_names)
    rec.notes.append("Schnellschätzung (ohne Analysekette des Hauptprojekts)")
    return rec


# ------------------------------------------------------------------ Hauptpfad (core)
def _marker(mp) -> Optional[int]:
    return None if mp is None else int(mp.sample)


def beats_from_delineation(delineation, fs: float) -> list[VCGBeat]:
    accepted = {id(b) for b in delineation.accepted_beats}
    out = []
    for k, b in enumerate(delineation.beats):
        q_on = _marker(b.r_onset) if b.r_onset is not None else _marker(b.q_peak)
        q_off = _marker(b.j_point) if b.j_point is not None else _marker(b.r_offset)
        r = int(b.r_peak.sample)
        if q_on is None:
            q_on = r - int(0.04 * fs)
        if q_off is None:
            q_off = r + int(0.05 * fs)
        bt = str(getattr(b, "beat_type", "unknown") or "unknown")
        bt = {"normal": "N", "N": "N", "V": "V", "pvc": "V", "ventricular": "V", "S": "S", "pac": "S", "supraventricular": "S",
              "paced": "P", "P": "P"}.get(bt, "V" if b.morphology_outlier else "N")
        out.append(VCGBeat(k, r, q_on, q_off, _marker(b.p_onset), _marker(b.p_offset),
                           _marker(b.t_onset), _marker(b.t_peak), _marker(b.t_offset), bt,
                           int(b.morphology_cluster_id or 0), id(b) in accepted, bool(b.morphology_outlier)))
    return out


def analyze_signal(signal: np.ndarray, fs: float, names: Iterable[str] = LEADS_12, label: str = "",
                   age: Optional[float] = None, sex: Optional[str] = None, powerline_hz: float = 50.0) -> VCGRecord:
    """Analysekette des Hauptprojekts (identisch zum ECG-Labor) → VCGRecord mit beiden Transformationen."""
    from vcg_biomarker.core.delineation.fusion_delineator import FusionDelineator
    from vcg_biomarker.core.preprocessing import (construct_robust_morphology_record, detect_r_peaks_multilead,
                                                  prepare_signal_paths, resample_signal)
    from vcg_biomarker.core.vcg_geometry import estimate_physiological_vcg_origin
    from vcg_biomarker.core.vcg_transform import VCGTransformRegistry

    sig, lead_names = reorder(np.asarray(signal, dtype=float), names)
    original_fs = float(fs)
    if not np.isclose(original_fs, ANALYSIS_FS):
        sig = resample_signal(sig, original_fs, ANALYSIS_FS)
    paths = prepare_signal_paths(sig, ANALYSIS_FS, powerline=powerline_hz)
    r_peaks = detect_r_peaks_multilead(paths["delineation"], ANALYSIS_FS)
    if len(r_peaks) < 2:
        raise ValueError("weniger als zwei Schläge erkannt")
    vcg_del = VCGTransformRegistry.transform(paths["delineation"], "kors")
    delineator = FusionDelineator(weights={"neurokit": 0.4, "multilead": 0.35, "vcg": 0.25},
                                  conflict_threshold=20.0, capture_candidates=False)
    delineation = delineator.delineate_record(ecg_12lead=paths["delineation"], r_peaks=r_peaks, fs=ANALYSIS_FS,
                                              ecg_id=0, vcg_xyz=vcg_del.xyz)
    diag = paths["diagnostic"]
    rec = VCGRecord(diag, float(ANALYSIS_FS), lead_names, label, age, sex, beats_from_delineation(delineation, ANALYSIS_FS),
                    rhythm_class=delineation.rhythm.rhythm_class.value, rhythm_flags=tuple(delineation.rhythm.flags),
                    delineation=delineation, source="core")
    for m in METHODS:
        rec.vcg_raw[m] = derive_vcg(diag, lead_names, m)
        try:
            assessment = estimate_physiological_vcg_origin(rec.vcg_raw[m], delineation, ANALYSIS_FS)
            if assessment.primary is not None:
                c = assessment.primary
                rec.origin[m] = OriginInfo(np.asarray(c.vector_millivolts, dtype=float), c.method, float(c.confidence),
                                           f"{c.method}, Fenster {c.start_sample}–{c.end_sample}, Konfidenz {c.confidence:.2f}")
            else:
                rec.origin[m] = quick_origin(rec.vcg_raw[m], rec.beats, ANALYSIS_FS)
                rec.notes.append(f"physiologischer Ursprung ({m}) nicht bestimmbar: {assessment.reason}; PR-Schnellschätzung")
        except Exception as exc:
            rec.origin[m] = quick_origin(rec.vcg_raw[m], rec.beats, ANALYSIS_FS)
            rec.notes.append(f"Ursprungsschätzung ({m}) fehlgeschlagen: {exc}")
        rec.fidelity[m] = transform_fidelity(diag, rec.vcg_raw[m], lead_names)
    # Medianschlag (fiducial-gewarpt, wie die Morphologiemarker des Hauptprojekts)
    try:
        robust, diagnostics = construct_robust_morphology_record(paths["morphology"], delineation)
        rep = delineation.representative_beat
        if rep is not None and diagnostics.get("status") == "used":
            rec.median_ecg = robust
            rec.median_beat = beats_from_delineation(type("D", (), {"beats": [rep], "accepted_beats": [rep]})(), ANALYSIS_FS)[0]
            rec.median_beat.index = -1
        else:
            rec.notes.append(f"Medianschlag nicht verfügbar ({diagnostics.get('status')})")
    except Exception as exc:
        rec.notes.append(f"Medianschlag fehlgeschlagen: {exc}")
    try:
        from vcg_biomarker.mimic.classical import ClassicalECGCriteria
        classical = ClassicalECGCriteria().compute(diag, delineation, age=age, sex=sex)
        rec.criteria = {key: value.to_dict() for key, value in classical.items()}
    except Exception as exc:
        rec.notes.append(f"klassische Kriterien nicht berechnet: {exc}")
    rec.payload = {
        "lab_processing": {"analysis_sampling_rate_hz": float(ANALYSIS_FS), "source": "vcglab_core_pipeline"},
        "delineation": {"rhythm_context": {"class": rec.rhythm_class, "flags": list(rec.rhythm_flags),
                                           "quality_score": float(delineation.quality_score)}},
        "classical_criteria": rec.criteria,
    }
    return rec


def record_from_payload(signal: np.ndarray, fs: float, names: Iterable[str], payload: dict, label: str = "",
                        age: Optional[float] = None, sex: Optional[str] = None) -> VCGRecord:
    """Laborbefund (Nutzlast des ECG-Labors) → VCGRecord ohne erneute Delineation."""
    from ..heart3d.timeline import timeline_from_payload
    sig, lead_names = reorder(np.asarray(signal, dtype=float), names)
    afs = float(payload.get("lab_processing", {}).get("analysis_sampling_rate_hz", fs) or fs)
    if not np.isclose(fs, afs):
        from vcg_biomarker.core.preprocessing import resample_signal
        sig = resample_signal(sig, float(fs), int(afs))
    tl = timeline_from_payload(payload, sig, afs, lead_names)
    beats = []
    for b in tl.beats:
        s = lambda v: None if v is None else int(round(v * afs))  # noqa: E731
        beats.append(VCGBeat(b.index, s(b.r_time), s(b.qrs_on), s(b.qrs_off),
                             s(b.p_on) if b.p_measured else None, s(b.p_off) if b.p_measured else None,
                             s(b.t_on) if b.t_measured else None, None,
                             s(b.t_end) if b.t_measured else None,
                             b.beat_type, b.cluster_id, b.accepted and b.qrs_measured, False))
    rec = VCGRecord(sig, afs, lead_names, label, age, sex, beats, rhythm_class=tl.rhythm_class, rhythm_flags=tuple(tl.rhythm_flags),
                    criteria=dict(payload.get("classical_criteria", {})), payload=payload, source="payload")
    unverified = sum(not b.qrs_measured or not b.t_measured or not b.p_measured for b in tl.beats)
    if unverified:
        rec.notes.append(f"{unverified}/{len(tl.beats)} Schläge mit mindestens einer nicht verifizierten P/QRS/T-Grenze; illustrative 3D-Zeiten sind keine VCG-Messgrenzen")
    for m in METHODS:
        rec.vcg_raw[m] = derive_vcg(sig, lead_names, m)
        rec.origin[m] = quick_origin(rec.vcg_raw[m], beats, afs)
        rec.fidelity[m] = transform_fidelity(sig, rec.vcg_raw[m], lead_names)
    return rec


def load_wfdb(path: str) -> tuple[np.ndarray, float, tuple[str, ...], dict]:
    import wfdb
    record = wfdb.rdrecord(path[:-4] if path.lower().endswith(".hea") else path)
    signal = np.asarray(record.p_signal, dtype=np.float64)
    names = tuple(str(n).upper().replace("AVR", "aVR").replace("AVL", "aVL").replace("AVF", "aVF") for n in record.sig_name)
    meta: dict = {}
    for c in record.comments or []:
        low = c.lower()
        if low.startswith("age:"):
            try:
                meta["age"] = float(c.split(":", 1)[1])
            except ValueError:
                pass
        if low.startswith("sex:"):
            v = c.split(":", 1)[1].strip().upper()
            meta["sex"] = "F" if v.startswith("F") else "M" if v.startswith("M") else None
    return signal, float(record.fs), names, meta
