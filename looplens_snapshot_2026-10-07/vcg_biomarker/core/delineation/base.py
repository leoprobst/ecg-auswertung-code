from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

import numpy as np


_FIDUCIAL_ATTRIBUTES = (
    "p_onset",
    "p_peak",
    "p_offset",
    "q_peak",
    "r_onset",
    "r_offset",
    "s_peak",
    "j_point",
    "t_onset",
    "t_peak",
    "t_offset",
    "u_onset",
    "u_peak",
    "u_offset",
)

# Per-wave floor on a segment's measurement confidence.
#
# ``segment_confidence`` is the *minimum* of the segment's two boundary
# markers, so its scale depends on which wave those boundaries belong to. A
# QRS is bounded by two sharp, high-amplitude flanks; a P wave by two shallow,
# low-amplitude ones. Measured on all 200 LUDB records, the same score means
# very different things per wave:
#
#     P    p05 0.000   median 0.417   p95 0.579
#     T    p05 0.323   median 0.505   p95 0.709
#     QRS  p05 0.613   median 0.809   p95 0.881
#
# A single threshold of 0.35 therefore sat on the dense middle of the P
# distribution while never touching QRS at all. It discarded 61 % of P waves
# and 44 % of T waves whose measurements were, when kept, accurate to a median
# of 2-4 ms with a positive predictive value of 0.99. Across the whole range
# 0.00-0.35 the PPV moved by 1.2 points for P and 0.3 for T -- the score barely
# separates right from wrong here, it separates measured from not measured.
#
# The values below restore the distinction the rest of this module already
# makes: _FIDUCIAL_UNCERTAINTY_LIMIT_MS has always been per wave, because
# different waves have different achievable precision. This is the same idea
# applied to the confidence gate.
#
# Composite segments take the floor of the weakest wave they span, because
# their confidence is a minimum over boundaries from both. Gating ``pr`` at the
# QRS level would discard PR intervals for exactly the reason P waves were
# being discarded.
#
# Rationale, measurements and the literature comparison:
# DELINEATION_ABSTENTION_2026-08-18.md
_MEASUREMENT_CONFIDENCE_THRESHOLD = {
    "p": 0.15,
    "pr": 0.15,
    "qrs": 0.35,
    "st": 0.15,
    "t": 0.15,
    "u": 0.15,
    "tu": 0.15,
    "qu": 0.15,
    "qrst": 0.15,
    "st_t": 0.15,
    "beat": 0.15,
}
DEFAULT_MEASUREMENT_CONFIDENCE_THRESHOLD = 0.35


def measurement_confidence_threshold(segment: str) -> float:
    """Lowest confidence at which a segment's measurement is still used.

    Unknown segment names fall back to the strictest value rather than the
    most permissive one, so a new wave has to be calibrated deliberately
    instead of inheriting a loose gate by accident.
    """

    name = str(segment).casefold()
    base = (
        name.split("_", 1)[0]
        if name.startswith(("qrs_", "t_", "u_"))
        else name
    )
    return _MEASUREMENT_CONFIDENCE_THRESHOLD.get(
        base, DEFAULT_MEASUREMENT_CONFIDENCE_THRESHOLD
    )


# Which wave a single fiducial point belongs to.
#
# ``measurement_confidence_threshold`` answers for *segments* ("p", "qrs").
# Consumers usually hold a single point instead -- ``p_onset``, ``t_peak`` --
# and asking the segment function for those names silently returns the strict
# fallback, because "p_onset" is not a segment key.  That is how the per-wave
# calibration came to apply to segment gates while every point gate stayed at
# the old uniform 0.35: the segment gate admitted a beat and the point gate
# then discarded it, so the segment gate never bound.  Measured on 60 LUDB
# records, 200 of 457 admitted beats (43.8 %) were lost that way.
_FIDUCIAL_WAVE = {
    "p_onset": "p",
    "p_peak": "p",
    "p_offset": "p",
    "retrograde_p_onset": "p",
    "retrograde_p_offset": "p",
    "q_peak": "qrs",
    "r_onset": "qrs",
    "r_peak": "qrs",
    "s_peak": "qrs",
    "r_offset": "qrs",
    # The J point is the end of depolarization, not the middle of the ST
    # segment; it is read at QRS strictness so a fast upstroke cannot be
    # mistaken for repolarization.
    "j_point": "qrs",
    "t_onset": "t",
    "t_peak": "t",
    "t_offset": "t",
    "u_onset": "u",
    "u_peak": "u",
    "u_offset": "u",
}


def fiducial_confidence_threshold(attribute: str) -> float:
    """Lowest confidence at which a single fiducial point is still used.

    Same scale and same calibration as ``measurement_confidence_threshold``,
    keyed by point name instead of segment name.  Unknown names fall back to
    the strictest value for the same reason: a new landmark has to be
    calibrated on purpose rather than inherit a loose gate.
    """

    wave = _FIDUCIAL_WAVE.get(str(attribute).casefold())
    if wave is None:
        return DEFAULT_MEASUREMENT_CONFIDENCE_THRESHOLD
    return measurement_confidence_threshold(wave)


_FIDUCIAL_UNCERTAINTY_LIMIT_MS = {
    "p_onset": 40.0,
    "p_offset": 40.0,
    "r_onset": 30.0,
    "r_offset": 30.0,
    "t_onset": 60.0,
    "t_offset": 60.0,
    "u_onset": 80.0,
    "u_offset": 80.0,
}


class WaveType(Enum):
    P_ONSET = "p_onset"
    P_PEAK = "p_peak"
    P_OFFSET = "p_offset"
    Q_PEAK = "q_peak"
    R_ONSET = "r_onset"
    R_PEAK = "r_peak"
    R_OFFSET = "r_offset"
    S_PEAK = "s_peak"
    J_POINT = "j_point"
    T_ONSET = "t_onset"
    T_PEAK = "t_peak"
    T_OFFSET = "t_offset"
    U_ONSET = "u_onset"
    U_PEAK = "u_peak"
    U_OFFSET = "u_offset"


class WaveStatus(str, Enum):
    """Measurement state of a physiological waveform.

    ``ABSENT`` is reserved for a rhythm-compatible absence. A detector that
    simply cannot establish a boundary must use ``INDETERMINATE`` or
    ``NOT_ASSESSABLE`` so missing reference information is never converted
    into a normal/negative measurement.
    """

    PRESENT = "present"
    ABSENT = "absent"
    INDETERMINATE = "indeterminate"
    NOT_ASSESSABLE = "not_assessable"
    FUSED = "fused"
    REPLACED_BY_ATRIAL_ACTIVITY = "replaced_by_atrial_activity"


class RhythmClass(str, Enum):
    UNKNOWN = "unknown"
    REGULAR = "regular"
    IRREGULAR = "irregular"
    AF_COMPATIBLE = "af_compatible_pattern"
    FLUTTER_COMPATIBLE = "flutter_compatible_pattern"
    MULTIPLE_ATRIAL_ACTIVITY = "multiple_atrial_activity"
    WIDE_QRS = "wide_qrs_pattern"
    MIXED = "mixed_or_ectopic_pattern"


@dataclass(frozen=True)
class RhythmAssessment:
    """Technical rhythm context used to guard waveform measurements.

    The class deliberately describes signal patterns, not diagnoses.
    """

    rhythm_class: RhythmClass = RhythmClass.UNKNOWN
    confidence: float = 0.0
    rr_median_ms: Optional[float] = None
    rr_cv: Optional[float] = None
    rr_successive_irregularity: Optional[float] = None
    rr_lag1_correlation: Optional[float] = None
    rr_turning_point_fraction: Optional[float] = None
    p_wave_coverage: Optional[float] = None
    p_timing_mad_ms: Optional[float] = None
    p_candidate_coverage: Optional[float] = None
    p_candidate_timing_mad_ms: Optional[float] = None
    p_candidate_timing_p10_p90_ms: Optional[float] = None
    atrial_frequency_hz: Optional[float] = None
    atrial_spectral_concentration: Optional[float] = None
    atrial_bandpower_ratio: Optional[float] = None
    independent_atrial_rate_hz: Optional[float] = None
    independent_atrial_rate_concentration: Optional[float] = None
    flags: tuple[str, ...] = ()

    @property
    def discrete_p_not_evaluable(self) -> bool:
        return self.rhythm_class in {
            RhythmClass.AF_COMPATIBLE,
            RhythmClass.FLUTTER_COMPATIBLE,
            RhythmClass.MULTIPLE_ATRIAL_ACTIVITY,
        }

    @property
    def low_discrete_p_coverage(self) -> bool:
        return (
            "no_consistent_discrete_p" in self.flags
            and "stable_discrete_p_coupling" not in self.flags
            and self.p_wave_coverage is not None
            and self.p_wave_coverage < 0.50
        )

    @property
    def requires_beatwise_rate_correction(self) -> bool:
        return self.rhythm_class in {
            RhythmClass.IRREGULAR,
            RhythmClass.AF_COMPATIBLE,
            RhythmClass.FLUTTER_COMPATIBLE,
            RhythmClass.MULTIPLE_ATRIAL_ACTIVITY,
            RhythmClass.MIXED,
        }


@dataclass
class MarkerPoint:
    sample: int
    confidence: float
    source: str
    uncertainty_ms: float = 0.0
    support_count: int = 1
    lead_names: tuple[str, ...] = ()
    evidence_sources: tuple[str, ...] = ()
    conflict_ms: float = 0.0

    def __post_init__(self):
        self.sample = int(self.sample)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.uncertainty_ms = max(0.0, float(self.uncertainty_ms))
        self.support_count = max(1, int(self.support_count))
        self.lead_names = tuple(dict.fromkeys(str(item) for item in self.lead_names))
        self.evidence_sources = tuple(
            dict.fromkeys(str(item) for item in self.evidence_sources)
        )
        self.conflict_ms = max(0.0, float(self.conflict_ms))

    @property
    def manually_reviewed(self) -> bool:
        return self.source.startswith("manual")


@dataclass(frozen=True)
class BeatIntervalMeasurement:
    beat_index: int
    preceding_rr_seconds: float
    qt_ms: float
    qrs_ms: Optional[float]
    jt_ms: Optional[float]
    confidence: float


@dataclass
class BeatDelineation:
    r_peak: MarkerPoint
    p_onset: Optional[MarkerPoint] = None
    p_peak: Optional[MarkerPoint] = None
    p_offset: Optional[MarkerPoint] = None
    q_peak: Optional[MarkerPoint] = None
    r_onset: Optional[MarkerPoint] = None
    r_offset: Optional[MarkerPoint] = None
    s_peak: Optional[MarkerPoint] = None
    j_point: Optional[MarkerPoint] = None
    t_onset: Optional[MarkerPoint] = None
    t_peak: Optional[MarkerPoint] = None
    t_offset: Optional[MarkerPoint] = None
    u_onset: Optional[MarkerPoint] = None
    u_peak: Optional[MarkerPoint] = None
    u_offset: Optional[MarkerPoint] = None
    # Retrograde P candidate: a P-like deflection found AFTER this beat's
    # own QRS (post-QRS/ST-region search), separate from p_onset/p_peak/
    # p_offset (the ordinary pre-QRS P) so existing consumers of the
    # primary P fields are unaffected. Exploratory/lower-confidence by
    # construction -- see MultileadDelineator._search_retrograde_p_candidates.
    retrograde_p_onset: Optional[MarkerPoint] = None
    retrograde_p_peak: Optional[MarkerPoint] = None
    retrograde_p_offset: Optional[MarkerPoint] = None
    retrograde_p_status: WaveStatus = WaveStatus.INDETERMINATE

    p_wave_absent: bool = False
    t_wave_absent: bool = False
    u_wave_absent: bool = False
    p_t_fusion_risk: bool = False
    wide_qrs: bool = False
    bundle_branch_block: bool = False
    p_status: WaveStatus = WaveStatus.INDETERMINATE
    qrs_status: WaveStatus = WaveStatus.INDETERMINATE
    t_status: WaveStatus = WaveStatus.INDETERMINATE
    u_status: WaveStatus = WaveStatus.INDETERMINATE
    atrial_activity_count: int = 0
    morphology_outlier: bool = False
    boundary_conflict: bool = False
    review_reasons: tuple[str, ...] = ()
    rr_interval_to_previous_ms: Optional[float] = None
    coupling_ratio: Optional[float] = None
    morphology_cluster_id: Optional[int] = None
    # Optional external beat annotation, never inferred from an outlier or wide QRS.
    # Temporal PVC/sinus research requires both the label and its source.
    beat_type: str = "unknown"
    beat_type_source: str = ""

    def __post_init__(self) -> None:
        if self.p_wave_absent and self.p_status == WaveStatus.INDETERMINATE:
            self.p_status = WaveStatus.ABSENT
        if self.t_wave_absent and self.t_status == WaveStatus.INDETERMINATE:
            self.t_status = WaveStatus.ABSENT
        if self.u_wave_absent and self.u_status == WaveStatus.INDETERMINATE:
            self.u_status = WaveStatus.ABSENT

    def resolved_wave_status(self, wave: str) -> WaveStatus:
        if wave == "p":
            if self.p_status not in {
                WaveStatus.INDETERMINATE,
                WaveStatus.PRESENT,
            }:
                return self.p_status
            if self.p_wave_absent:
                return (
                    self.p_status
                    if self.p_status != WaveStatus.INDETERMINATE
                    else WaveStatus.ABSENT
                )
            if any(
                getattr(self, name) is not None
                for name in ("p_onset", "p_peak", "p_offset")
            ):
                return WaveStatus.PRESENT
            return self.p_status
        if wave == "qrs":
            if self.qrs_status not in {
                WaveStatus.INDETERMINATE,
                WaveStatus.PRESENT,
            }:
                return self.qrs_status
            if self.r_onset is not None and self.r_offset is not None:
                return WaveStatus.PRESENT
            return self.qrs_status
        if wave == "t":
            if self.t_status not in {
                WaveStatus.INDETERMINATE,
                WaveStatus.PRESENT,
            }:
                return self.t_status
            if self.t_wave_absent:
                return (
                    self.t_status
                    if self.t_status != WaveStatus.INDETERMINATE
                    else WaveStatus.ABSENT
                )
            if any(
                getattr(self, name) is not None
                for name in ("t_onset", "t_peak", "t_offset")
            ):
                return WaveStatus.PRESENT
            return self.t_status
        if wave == "u":
            if self.u_status not in {
                WaveStatus.INDETERMINATE,
                WaveStatus.PRESENT,
            }:
                return self.u_status
            if self.u_wave_absent:
                return (
                    self.u_status
                    if self.u_status != WaveStatus.INDETERMINATE
                    else WaveStatus.ABSENT
                )
            if any(
                getattr(self, name) is not None
                for name in ("u_onset", "u_peak", "u_offset")
            ):
                return WaveStatus.PRESENT
            return self.u_status
        raise ValueError(f"Unknown wave: {wave}")

    def marker_confidence(self, *attributes: str) -> float:
        markers = [getattr(self, name, None) for name in attributes]
        if not markers or any(marker is None for marker in markers):
            return 0.0
        return float(min(marker.confidence for marker in markers))

    def segment_confidence(self, segment: str) -> float:
        requirements = {
            "p": ("p_onset", "p_offset"),
            "pr": ("p_offset", "r_onset"),
            "qrs": ("r_onset", "r_offset"),
            "st": ("r_offset", "t_onset"),
            "t": ("t_onset", "t_offset"),
            "u": ("u_onset", "u_offset"),
            "tu": ("t_offset", "u_onset"),
            "qu": ("r_onset", "u_offset"),
            "qrst": ("r_onset", "t_offset"),
            "st_t": ("r_offset", "t_offset"),
            "beat": ("r_onset", "t_offset"),
            # The atrial window, anchored on the ventricles: from the end of
            # repolarisation to the next depolarisation. It is defined whether
            # or not a P wave exists, which is the whole reason it is here.
            # Measured over 11.798 ECGs the P segment is unusable in 61,0 % of
            # recordings, and the loss is not random -- 70,7 % of the patients
            # who went on to develop atrial fibrillation had no measurable P
            # wave, against 52,6 % of those who did not. Every atrial finding
            # in this project is currently conditioned on not having the
            # pathology it is looking for.
            "tq": ("t_offset", "r_onset"),
        }
        base = (
            segment.split("_", 1)[0]
            if segment.startswith(("qrs_", "t_", "u_"))
            else segment
        )
        required = requirements.get(base)
        if required is None:
            return self.r_peak.confidence
        if base in {"p", "pr"} and self.resolved_wave_status("p") != WaveStatus.PRESENT:
            return 0.0
        if base in {"pr", "qrs", "st", "qrst", "qu", "beat", "tq"}:
            if self.resolved_wave_status("qrs") != WaveStatus.PRESENT:
                return 0.0
        if base in {"t", "tu", "st", "st_t", "qrst", "beat", "tq"}:
            if self.resolved_wave_status("t") != WaveStatus.PRESENT:
                return 0.0
        if base in {"u", "tu", "qu"} and self.resolved_wave_status("u") != WaveStatus.PRESENT:
            return 0.0
        markers = [
            getattr(self, attribute, None) for attribute in required
        ]
        if any(marker is None for marker in markers):
            return 0.0
        if any(
            marker.uncertainty_ms
            > _FIDUCIAL_UNCERTAINTY_LIMIT_MS.get(attribute, np.inf)
            for attribute, marker in zip(required, markers)
        ):
            return 0.0
        return self.marker_confidence(*required)

    def segment_boundary_uncertainty_ms(
        self,
        segment: str,
    ) -> Optional[float]:
        requirements = {
            "p": ("p_onset", "p_offset"),
            "pr": ("p_offset", "r_onset"),
            "qrs": ("r_onset", "r_offset"),
            "st": ("r_offset", "t_onset"),
            "t": ("t_onset", "t_offset"),
            "u": ("u_onset", "u_offset"),
            "tu": ("t_offset", "u_onset"),
            "qu": ("r_onset", "u_offset"),
            "qrst": ("r_onset", "t_offset"),
            "st_t": ("r_offset", "t_offset"),
            "beat": ("r_onset", "t_offset"),
            # The atrial window, anchored on the ventricles: from the end of
            # repolarisation to the next depolarisation. It is defined whether
            # or not a P wave exists, which is the whole reason it is here.
            # Measured over 11.798 ECGs the P segment is unusable in 61,0 % of
            # recordings, and the loss is not random -- 70,7 % of the patients
            # who went on to develop atrial fibrillation had no measurable P
            # wave, against 52,6 % of those who did not. Every atrial finding
            # in this project is currently conditioned on not having the
            # pathology it is looking for.
            "tq": ("t_offset", "r_onset"),
        }
        base = (
            segment.split("_", 1)[0]
            if segment.startswith(("qrs_", "t_", "u_"))
            else segment
        )
        required = requirements.get(base)
        if required is None:
            return 0.0
        markers = [getattr(self, attribute, None) for attribute in required]
        if any(marker is None for marker in markers):
            return None
        return float(
            np.hypot(*[marker.uncertainty_ms for marker in markers])
        )

    def has_excessive_segment_uncertainty(self, segment: str) -> bool:
        requirements = {
            "p": ("p_onset", "p_offset"),
            "pr": ("p_offset", "r_onset"),
            "qrs": ("r_onset", "r_offset"),
            "st": ("r_offset", "t_onset"),
            "t": ("t_onset", "t_offset"),
            "u": ("u_onset", "u_offset"),
            "tu": ("t_offset", "u_onset"),
            "qu": ("r_onset", "u_offset"),
            "qrst": ("r_onset", "t_offset"),
            "st_t": ("r_offset", "t_offset"),
            "beat": ("r_onset", "t_offset"),
            # The atrial window, anchored on the ventricles: from the end of
            # repolarisation to the next depolarisation. It is defined whether
            # or not a P wave exists, which is the whole reason it is here.
            # Measured over 11.798 ECGs the P segment is unusable in 61,0 % of
            # recordings, and the loss is not random -- 70,7 % of the patients
            # who went on to develop atrial fibrillation had no measurable P
            # wave, against 52,6 % of those who did not. Every atrial finding
            # in this project is currently conditioned on not having the
            # pathology it is looking for.
            "tq": ("t_offset", "r_onset"),
        }
        base = (
            segment.split("_", 1)[0]
            if segment.startswith(("qrs_", "t_", "u_"))
            else segment
        )
        return any(
            marker is not None
            and marker.uncertainty_ms
            > _FIDUCIAL_UNCERTAINTY_LIMIT_MS.get(attribute, np.inf)
            for attribute in requirements.get(base, ())
            for marker in (getattr(self, attribute, None),)
        )

    @property
    def pr_interval_samples(self) -> Optional[float]:
        if (
            self.resolved_wave_status("p") != WaveStatus.PRESENT
            or self.resolved_wave_status("qrs") != WaveStatus.PRESENT
            or self.p_onset is None
            or self.r_onset is None
        ):
            return None
        return (self.r_onset.sample - self.p_onset.sample)

    @property
    def qrs_duration_samples(self) -> Optional[float]:
        if (
            self.resolved_wave_status("qrs") == WaveStatus.PRESENT
            and self.r_onset is not None
            and self.r_offset is not None
        ):
            return (self.r_offset.sample - self.r_onset.sample)
        return None

    @property
    def qt_interval_samples(self) -> Optional[float]:
        if (
            self.resolved_wave_status("qrs") == WaveStatus.PRESENT
            and self.resolved_wave_status("t") == WaveStatus.PRESENT
            and self.r_onset is not None
            and self.t_offset is not None
        ):
            return (self.t_offset.sample - self.r_onset.sample)
        return None

    def pr_interval_ms(self, fs: int) -> Optional[float]:
        s = self.pr_interval_samples
        return s / fs * 1000.0 if s is not None else None

    def qrs_duration_ms(self, fs: int) -> Optional[float]:
        s = self.qrs_duration_samples
        return s / fs * 1000.0 if s is not None else None

    def qrs_duration_bounds_ms(
        self,
        fs: int,
    ) -> Optional[tuple[float, float]]:
        duration = self.qrs_duration_ms(fs)
        if (
            duration is None
            or self.r_onset is None
            or self.r_offset is None
        ):
            return None
        # Boundary uncertainties are conservative half-widths. Adding them
        # avoids claiming a threshold crossing when either edge is ambiguous.
        uncertainty = (
            self.r_onset.uncertainty_ms
            + self.r_offset.uncertainty_ms
        )
        return (
            max(0.0, duration - uncertainty),
            duration + uncertainty,
        )

    def qrs_definitely_exceeds_ms(
        self,
        fs: int,
        threshold_ms: float,
    ) -> bool:
        bounds = self.qrs_duration_bounds_ms(fs)
        return bool(
            bounds is not None
            and bounds[0] > float(threshold_ms)
        )

    def qt_interval_ms(self, fs: int) -> Optional[float]:
        s = self.qt_interval_samples
        return s / fs * 1000.0 if s is not None else None

    def get_marker(self, wave_type: WaveType) -> Optional[MarkerPoint]:
        mapping = {
            WaveType.P_ONSET: self.p_onset,
            WaveType.P_PEAK: self.p_peak,
            WaveType.P_OFFSET: self.p_offset,
            WaveType.Q_PEAK: self.q_peak,
            WaveType.R_ONSET: self.r_onset,
            WaveType.R_PEAK: self.r_peak,
            WaveType.R_OFFSET: self.r_offset,
            WaveType.S_PEAK: self.s_peak,
            WaveType.J_POINT: self.j_point,
            WaveType.T_ONSET: self.t_onset,
            WaveType.T_PEAK: self.t_peak,
            WaveType.T_OFFSET: self.t_offset,
            WaveType.U_ONSET: self.u_onset,
            WaveType.U_PEAK: self.u_peak,
            WaveType.U_OFFSET: self.u_offset,
        }
        return mapping.get(wave_type)


def ventricular_rr_seconds(r_samples, fs: float) -> Optional[float]:
    """Mean RR interval (seconds) of a beat sequence: the ventricular rate.

    The median RR used before is the coupling interval in a bigeminy and
    flips between the short and long interval with the number of beats; on
    development ECGs labelled bigeminy by the recorder the median-based rate
    ran +5 bpm (up to +26) above the recorder's ventricular rate, and it
    enters every model as the rate covariate. The mean RR is the number of
    beats per unit time, which is what the recorder reports as rate.
    """

    samples = np.sort(np.asarray(list(r_samples), dtype=float))
    if samples.size < 2 or not fs or fs <= 0:
        return None
    value = float(samples[-1] - samples[0]) / float(samples.size - 1) / float(fs)
    return value if np.isfinite(value) and value > 0 else None


@dataclass
class RecordDelineation:
    beats: list[BeatDelineation]
    fs: int
    n_samples: int
    ecg_id: int
    quality_score: float = 0.0
    source_method: str = "fusion"
    rhythm: RhythmAssessment = field(default_factory=RhythmAssessment)
    diagnostics: dict[str, object] = field(default_factory=dict)
    review_required: bool = False
    review_reasons: tuple[str, ...] = ()
    _beat_cache_signature: Optional[tuple] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _cached_plausible_beats: tuple[BeatDelineation, ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )
    _cached_accepted_beats: tuple[BeatDelineation, ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )
    _cached_median_beat: Optional[BeatDelineation] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _cached_representative_beat: Optional[BeatDelineation] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _beat_cache_read_depth: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def plausible_beats(self) -> list[BeatDelineation]:
        self._ensure_beat_cache()
        return list(self._cached_plausible_beats)

    @property
    def accepted_beats(self) -> list[BeatDelineation]:
        self._ensure_beat_cache()
        return list(self._cached_accepted_beats)

    @property
    def median_beat(self) -> Optional[BeatDelineation]:
        self._ensure_beat_cache()
        return self._cached_median_beat

    @property
    def representative_beat(self) -> Optional[BeatDelineation]:
        """Robust record-level fiducial consensus aligned to the medoid beat."""

        self._ensure_beat_cache()
        return self._cached_representative_beat

    def suppress_discrete_p(self, reason: str) -> int:
        """Treat every beat's P wave as replaced by atrial activity.

        For rhythm evidence from outside the signal analysis. Atrial flutter
        with fixed 2:1 or 4:1 conduction places one flutter wave at a
        constant distance before each QRS; it is delineated as a P wave with
        a stable PR, and the flutter search, which requires low P coverage,
        never runs. On development ECGs with a flutter statement and no
        device P, 9 of 54 received a P wave, and no spectral feature told
        conducted flutter with such a "P" from sinus tachycardia (AUC
        0.42-0.60, audit_tools/flutter_harmonic_audit.py). Manually reviewed
        P markers are kept. Returns the number of beats changed.
        """

        changed = 0
        for beat in self.beats:
            if any(
                marker is not None and marker.manually_reviewed
                for marker in (beat.p_onset, beat.p_peak, beat.p_offset)
            ):
                continue
            beat.p_onset = None
            beat.p_peak = None
            beat.p_offset = None
            beat.p_wave_absent = True
            beat.p_status = WaveStatus.REPLACED_BY_ATRIAL_ACTIVITY
            if reason not in beat.review_reasons:
                beat.review_reasons += (reason,)
            changed += 1
        if changed:
            if reason not in self.rhythm.flags:
                self.rhythm = replace(
                    self.rhythm,
                    flags=self.rhythm.flags + (reason,),
                )
            self.diagnostics[reason] = changed
            self.invalidate_beat_cache()
        return changed

    def invalidate_beat_cache(self) -> None:
        self._beat_cache_signature = None
        self._cached_plausible_beats = ()
        self._cached_accepted_beats = ()
        self._cached_median_beat = None
        self._cached_representative_beat = None

    @contextmanager
    def stable_beat_cache(self):
        """Pin the mutation-checked consensus during one read-only panel.

        Public delineations remain mutable for manual review, so an ordinary
        property access verifies the complete beat signature. A marker panel
        makes thousands of read-only accesses. Verify once at its boundary,
        skip only redundant checks while it runs, and check again on the next
        external access after the panel exits.
        """

        self._ensure_beat_cache()
        self._beat_cache_read_depth += 1
        try:
            yield self
        finally:
            self._beat_cache_read_depth = max(
                0,
                self._beat_cache_read_depth - 1,
            )

    def _ensure_beat_cache(self) -> None:
        if (
            self._beat_cache_read_depth > 0
            and self._beat_cache_signature is not None
        ):
            return
        signature = self._current_beat_signature()
        if signature == self._beat_cache_signature:
            return
        plausible = tuple(
            beat
            for beat in self.beats
            if self._is_physiologically_plausible(beat)
        )
        accepted = self._select_accepted_beats(plausible)
        median = self._select_median_beat(accepted)
        representative = self._build_representative_beat(accepted, median, self.fs)
        self._cached_plausible_beats = plausible
        self._cached_accepted_beats = accepted
        self._cached_median_beat = median
        self._cached_representative_beat = representative
        self._beat_cache_signature = signature

    def _current_beat_signature(self) -> tuple:
        marker_attributes = ("r_peak",) + _FIDUCIAL_ATTRIBUTES
        return (
            int(self.fs),
            int(self.n_samples),
            tuple(
                (
                    tuple(
                        (
                            None
                            if (marker := getattr(beat, attribute)) is None
                            else (
                                int(marker.sample),
                                float(marker.confidence),
                                float(marker.uncertainty_ms),
                                int(marker.support_count),
                                marker.source,
                            )
                        )
                        for attribute in marker_attributes
                    ),
                    bool(beat.p_wave_absent),
                    bool(beat.t_wave_absent),
                    bool(beat.u_wave_absent),
                    bool(beat.p_t_fusion_risk),
                    bool(beat.wide_qrs),
                    bool(beat.bundle_branch_block),
                    beat.p_status.value,
                    beat.qrs_status.value,
                    beat.t_status.value,
                    beat.u_status.value,
                    int(beat.atrial_activity_count),
                    bool(beat.morphology_outlier),
                    bool(beat.boundary_conflict),
                    str(beat.beat_type),
                    str(beat.beat_type_source),
                )
                for beat in self.beats
            ),
        )

    @staticmethod
    def _select_accepted_beats(
        plausible: tuple[BeatDelineation, ...],
    ) -> tuple[BeatDelineation, ...]:
        if len(plausible) < 3:
            return plausible

        matrix = np.asarray(
            [
                [
                    (
                        float(getattr(beat, attribute).sample - beat.r_peak.sample)
                        if getattr(beat, attribute) is not None
                        else np.nan
                    )
                    for attribute in _FIDUCIAL_ATTRIBUTES
                ]
                for beat in plausible
            ],
            dtype=float,
        )
        informative = np.sum(np.isfinite(matrix), axis=0) >= max(
            2,
            int(np.ceil(len(plausible) / 2)),
        )
        if not np.any(informative):
            return plausible
        values = matrix[:, informative]
        median = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - median), axis=0)
        scale = np.maximum(1.4826 * mad, 1.0)
        standardized = np.minimum(
            np.abs(values - median) / scale,
            30.0,
        )
        squared = standardized**2
        finite_counts = np.sum(np.isfinite(squared), axis=1)
        mean_squared = np.full(len(squared), np.inf, dtype=float)
        has_comparable_markers = finite_counts > 0
        mean_squared[has_comparable_markers] = (
            np.nansum(squared[has_comparable_markers], axis=1)
            / finite_counts[has_comparable_markers]
        )
        distances = np.sqrt(mean_squared)
        accepted = [
            beat
            for beat, distance in zip(plausible, distances)
            if np.isfinite(distance) and distance <= 8.0
        ]
        return tuple(accepted) if len(accepted) >= 2 else plausible

    @staticmethod
    def _select_median_beat(
        beats: tuple[BeatDelineation, ...],
    ) -> Optional[BeatDelineation]:
        if not beats:
            return None
        if len(beats) == 1:
            return beats[0]
        marker_attributes = ("r_peak",) + _FIDUCIAL_ATTRIBUTES
        manually_reviewed = [
            beat
            for beat in beats
            if any(
                marker is not None and marker.manually_reviewed
                for attribute in marker_attributes
                for marker in (getattr(beat, attribute),)
            )
        ]
        if manually_reviewed:
            return max(
                manually_reviewed,
                key=lambda beat: (
                    sum(
                        marker is not None and marker.manually_reviewed
                        for attribute in marker_attributes
                        for marker in (getattr(beat, attribute),)
                    ),
                    sum(
                        marker.confidence
                        for attribute in marker_attributes
                        for marker in (getattr(beat, attribute),)
                        if marker is not None and marker.manually_reviewed
                    ),
                    beat.r_peak.confidence,
                ),
            )

        vectors = []
        for beat in beats:
            r_sample = beat.r_peak.sample
            vectors.append([
                (
                    float(getattr(beat, attribute).sample - r_sample)
                    if getattr(beat, attribute) is not None
                    else np.nan
                )
                for attribute in _FIDUCIAL_ATTRIBUTES
            ])
        matrix = np.asarray(vectors, dtype=float)
        informative = np.any(np.isfinite(matrix), axis=0)
        if not np.any(informative):
            return max(beats, key=lambda beat: beat.r_peak.confidence)
        matrix = matrix[:, informative]
        median = np.nanmedian(matrix, axis=0)
        distances = []
        for index, vector in enumerate(matrix):
            available = np.isfinite(vector) & np.isfinite(median)
            if not np.any(available):
                distance = float("inf")
            else:
                distance = float(np.mean(np.abs(vector[available] - median[available])))
            confidence_penalty = 1.0 - beats[index].r_peak.confidence
            distances.append(distance + confidence_penalty)
        return beats[int(np.argmin(distances))]

    @staticmethod
    def _build_representative_beat(
        beats: tuple[BeatDelineation, ...],
        medoid: Optional[BeatDelineation],
        fs: int,
    ) -> Optional[BeatDelineation]:
        if medoid is None:
            return None
        if len(beats) == 1:
            return medoid

        representative = deepcopy(medoid)
        target_r = medoid.r_peak.sample
        minimum_support = max(2, int(np.ceil(len(beats) * 0.5)))
        marker_tolerances_ms = {
            "p_onset": 30.0,
            "p_peak": 25.0,
            "p_offset": 30.0,
            "q_peak": 14.0,
            "r_onset": 16.0,
            "r_offset": 18.0,
            "s_peak": 14.0,
            "j_point": 18.0,
            "t_onset": 35.0,
            "t_peak": 35.0,
            "t_offset": 45.0,
            "u_onset": 60.0,
            "u_peak": 50.0,
            "u_offset": 80.0,
        }
        for attribute in _FIDUCIAL_ATTRIBUTES:
            medoid_marker = getattr(medoid, attribute)
            if medoid_marker is not None and medoid_marker.manually_reviewed:
                continue
            observations = [
                (
                    marker.sample - beat.r_peak.sample,
                    marker,
                )
                for beat in beats
                for marker in (getattr(beat, attribute),)
                if marker is not None and marker.confidence >= 0.20
            ]
            if len(observations) < minimum_support:
                setattr(representative, attribute, None)
                continue
            offsets = np.asarray([item[0] for item in observations], dtype=float)
            confidences = np.asarray(
                [item[1].confidence for item in observations],
                dtype=float,
            )
            order = np.argsort(offsets)
            cumulative = np.cumsum(confidences[order])
            midpoint = cumulative[-1] / 2.0
            selected = order[int(np.searchsorted(cumulative, midpoint))]
            offset = int(round(offsets[selected]))
            median_offset = float(np.median(offsets))
            mad_samples = float(np.median(np.abs(offsets - median_offset)))
            dispersion_ms = 1.4826 * mad_samples / fs * 1000.0
            tolerance = marker_tolerances_ms.get(attribute, 25.0)
            coverage = len(observations) / len(beats)
            stability = float(np.exp(-dispersion_ms / max(tolerance, 1.0)))
            base_confidence = float(np.median(confidences))
            repeated_evidence = 1.0 - np.exp(-len(observations) / 3.0)
            confidence = float(
                np.clip(
                    (
                        base_confidence
                        + (1.0 - base_confidence)
                        * 0.25
                        * repeated_evidence
                        * stability
                    )
                    * (0.70 + 0.30 * coverage)
                    * (0.75 + 0.25 * stability),
                    0.0,
                    1.0,
                )
            )
            sources = tuple(
                dict.fromkeys(
                    source
                    for _, marker in observations
                    for source in (
                        marker.evidence_sources
                        if marker.evidence_sources
                        else (marker.source,)
                    )
                )
            )
            leads = tuple(
                dict.fromkeys(
                    lead
                    for _, marker in observations
                    for lead in marker.lead_names
                )
            )
            setattr(
                representative,
                attribute,
                MarkerPoint(
                    sample=target_r + offset,
                    confidence=confidence,
                    source="record_consensus",
                    uncertainty_ms=max(
                        dispersion_ms,
                        float(np.median([
                            marker.uncertainty_ms
                            for _, marker in observations
                        ])),
                    ),
                    support_count=len(observations),
                    lead_names=leads,
                    evidence_sources=sources,
                    conflict_ms=max(
                        marker.conflict_ms for _, marker in observations
                    ),
                ),
            )

        representative.p_wave_absent = (
            sum(beat.p_wave_absent for beat in beats) > len(beats) / 2
        )
        representative.t_wave_absent = (
            sum(beat.t_wave_absent for beat in beats) > len(beats) / 2
        )
        representative.u_wave_absent = (
            sum(beat.u_wave_absent for beat in beats) > len(beats) / 2
        )
        representative.p_t_fusion_risk = any(
            beat.p_t_fusion_risk for beat in beats
        )
        representative.wide_qrs = (
            sum(beat.wide_qrs for beat in beats) >= len(beats) / 2
        )
        representative.bundle_branch_block = (
            sum(beat.bundle_branch_block for beat in beats) >= len(beats) / 2
        )
        representative.boundary_conflict = any(
            beat.boundary_conflict for beat in beats
        )
        # Wave status by majority of the accepted beats, like the fiducials
        # above. The representative is a copy of the medoid and used to keep
        # the medoid's own status: when that single beat had rejected P
        # boundaries, the whole record lost its P measurement although the
        # consensus boundaries -- taken from all beats -- were present. LUDB,
        # 2026-09-26: 23 of 177 records with P in most beats lost this way,
        # consensus P as accurate as in the retained records (onset/offset
        # median absolute error 14/16 against 12.5/14 ms). Only an upgrade to
        # PRESENT, only with consensus boundaries, never over manual review.
        for wave, bounds in (("p", ("p_onset", "p_offset")), ("t", ("t_onset", "t_offset"))):
            if any(
                marker is not None and marker.manually_reviewed
                for marker in (getattr(medoid, name) for name in bounds)
            ):
                continue
            present = sum(
                beat.resolved_wave_status(wave) == WaveStatus.PRESENT
                for beat in beats
            )
            if present > len(beats) / 2 and all(
                getattr(representative, name) is not None for name in bounds
            ):
                setattr(representative, f"{wave}_status", WaveStatus.PRESENT)
        return representative

    @property
    def n_beats(self) -> int:
        return len(self.beats)

    @property
    def n_accepted_beats(self) -> int:
        return len(self.accepted_beats)

    @property
    def n_plausible_beats(self) -> int:
        return len(self.plausible_beats)

    def segment_confidence(self, segment: str) -> float:
        beat = self.representative_beat
        if beat is None:
            return 0.0
        base = (
            segment.split("_", 1)[0]
            if segment.startswith(("qrs_", "t_", "u_"))
            else segment
        )
        manually_reviewed_p = any(
            marker is not None and marker.manually_reviewed
            for marker in (beat.p_onset, beat.p_peak, beat.p_offset)
        )
        if (
            base in {"p", "pr"}
            and self.rhythm.low_discrete_p_coverage
            and not manually_reviewed_p
        ):
            return 0.0
        return beat.segment_confidence(segment)

    def is_segment_evaluable(
        self,
        segment: str,
        minimum_confidence: Optional[float] = None,
    ) -> bool:
        base = segment.casefold().split("_", 1)[0]
        if (
            bool(self.diagnostics.get("severe_signal_incoherence", False))
            and base in {
                "p", "pr", "qrs", "st", "t", "u", "qt", "jt", "qrst", "tq"
            }
        ):
            return False
        if minimum_confidence is None:
            minimum_confidence = measurement_confidence_threshold(segment)
        return self.segment_confidence(segment) >= minimum_confidence

    def qt_rr_measurements(
        self,
        minimum_confidence: float = 0.35,
    ) -> tuple[BeatIntervalMeasurement, ...]:
        """Pair each evaluable QT interval with its immediately preceding RR."""

        if not np.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be finite and within [0, 1]")
        if len(self.beats) < 2 or not np.isfinite(self.fs) or self.fs <= 0:
            return ()
        peaks = [getattr(beat, "r_peak", None) for beat in self.beats]
        if any(peak is None or not np.isfinite(peak.sample) or peak.sample < 0 for peak in peaks):
            return ()  # Unknown timing cannot be removed and bridged safely.
        if len({peak.sample for peak in peaks}) != len(peaks):
            return ()
        accepted_ids = {id(beat) for beat in self.accepted_beats}
        ordered = sorted(
            enumerate(self.beats),
            key=lambda item: item[1].r_peak.sample,
        )
        measurements: list[BeatIntervalMeasurement] = []
        for position in range(1, len(ordered)):
            beat_index, beat = ordered[position]
            if id(beat) not in accepted_ids:
                continue
            previous = ordered[position - 1][1]
            rr_seconds = (
                beat.r_peak.sample - previous.r_peak.sample
            ) / float(self.fs)
            if not 0.25 <= rr_seconds <= 3.0:
                continue
            confidences = (beat.r_peak.confidence, previous.r_peak.confidence,
                           beat.segment_confidence("qrst"))
            if any(not np.isfinite(value) or not minimum_confidence <= value <= 1
                   for value in confidences):
                continue
            qt_ms = beat.qt_interval_ms(self.fs)
            if qt_ms is None or not 140.0 <= qt_ms <= 1000.0:
                continue
            qrs_ms = beat.qrs_duration_ms(self.fs)
            jt_ms = (
                (beat.t_offset.sample - beat.r_offset.sample)
                / self.fs
                * 1000.0
                if beat.t_offset is not None and beat.r_offset is not None
                else None
            )
            confidence = min(confidences)
            measurements.append(
                BeatIntervalMeasurement(
                    beat_index=beat_index,
                    preceding_rr_seconds=float(rr_seconds),
                    qt_ms=float(qt_ms),
                    qrs_ms=None if qrs_ms is None else float(qrs_ms),
                    jt_ms=None if jt_ms is None else float(jt_ms),
                    confidence=float(confidence),
                )
            )
        return tuple(measurements)

    def qt_rr_fast_beat_exclusion_count(self) -> int:
        """Accepted beat pairs skipped by qt_rr_measurements() solely for
        having a preceding RR below the 0.25s (240bpm) floor -- unlike the
        p_t_fusion_risk path, that exclusion previously left no trace.
        Recomputed independently rather than folded into
        qt_rr_measurements() itself, so that read-only query's existing
        callers and return type are unaffected.
        """
        if len(self.beats) < 2 or not np.isfinite(self.fs) or self.fs <= 0:
            return 0
        peaks = [getattr(beat, "r_peak", None) for beat in self.beats]
        if any(peak is None or not np.isfinite(peak.sample) or peak.sample < 0 for peak in peaks):
            return 0
        if len({peak.sample for peak in peaks}) != len(peaks):
            return 0
        accepted_ids = {id(beat) for beat in self.accepted_beats}
        ordered = sorted(
            enumerate(self.beats),
            key=lambda item: item[1].r_peak.sample,
        )
        count = 0
        for position in range(1, len(ordered)):
            _, beat = ordered[position]
            if id(beat) not in accepted_ids:
                continue
            previous = ordered[position - 1][1]
            rr_seconds = (
                beat.r_peak.sample - previous.r_peak.sample
            ) / float(self.fs)
            if rr_seconds < 0.25:
                count += 1
        return count

    def _is_physiologically_plausible(self, beat: BeatDelineation) -> bool:
        markers = (
            beat.r_peak,
            beat.p_onset,
            beat.p_peak,
            beat.p_offset,
            beat.q_peak,
            beat.r_onset,
            beat.r_offset,
            beat.s_peak,
            beat.j_point,
            beat.t_onset,
            beat.t_peak,
            beat.t_offset,
            beat.u_onset,
            beat.u_peak,
            beat.u_offset,
        )
        if beat.r_peak.confidence < 0.2:
            return False
        if any(
            marker is not None
            and not 0 <= marker.sample < self.n_samples
            for marker in markers
        ):
            return False
        if beat.r_onset is not None and beat.r_offset is not None:
            if not beat.r_onset.sample < beat.r_peak.sample < beat.r_offset.sample:
                return False
            qrs_ms = beat.qrs_duration_ms(self.fs)
            if qrs_ms is None or not 20.0 <= qrs_ms <= 360.0:
                return False
        if beat.q_peak is not None:
            lower = beat.r_onset.sample if beat.r_onset is not None else 0
            if not lower <= beat.q_peak.sample <= beat.r_peak.sample:
                return False
        if beat.s_peak is not None:
            upper = (
                beat.r_offset.sample
                if beat.r_offset is not None
                else self.n_samples - 1
            )
            if not beat.r_peak.sample <= beat.s_peak.sample <= upper:
                return False
        if beat.p_onset is not None and beat.p_offset is not None:
            if not beat.p_onset.sample < beat.p_offset.sample <= beat.r_peak.sample:
                return False
            if (
                beat.p_peak is not None
                and not beat.p_onset.sample
                <= beat.p_peak.sample
                <= beat.p_offset.sample
            ):
                return False
            p_duration_ms = (
                beat.p_offset.sample - beat.p_onset.sample
            ) / self.fs * 1000.0
            if not 20.0 <= p_duration_ms <= 300.0:
                return False
        if beat.t_onset is not None and beat.t_offset is not None:
            if not beat.r_peak.sample < beat.t_onset.sample < beat.t_offset.sample:
                return False
            if (
                beat.t_peak is not None
                and not beat.t_onset.sample
                <= beat.t_peak.sample
                <= beat.t_offset.sample
            ):
                return False
            t_duration_ms = (
                beat.t_offset.sample - beat.t_onset.sample
            ) / self.fs * 1000.0
            if not 30.0 <= t_duration_ms <= 650.0:
                return False
        if beat.u_onset is not None and beat.u_offset is not None:
            if beat.t_offset is None or not (
                beat.t_offset.sample
                <= beat.u_onset.sample
                < beat.u_offset.sample
            ):
                return False
            if (
                beat.u_peak is not None
                and not beat.u_onset.sample
                <= beat.u_peak.sample
                <= beat.u_offset.sample
            ):
                return False
            u_duration_ms = (
                beat.u_offset.sample - beat.u_onset.sample
            ) / self.fs * 1000.0
            if not 20.0 <= u_duration_ms <= 300.0:
                return False
        if beat.j_point is not None and beat.r_offset is not None:
            if abs(beat.j_point.sample - beat.r_offset.sample) > 0.08 * self.fs:
                return False
        qt_ms = beat.qt_interval_ms(self.fs)
        if qt_ms is not None and not 140.0 <= qt_ms <= 1000.0:
            return False
        if self.fs > 0 and len(self.beats) > 1:
            # RR < 200ms (>300bpm) exceeds even the fastest reported
            # SVT/VT rates; treat it as a likely duplicate/spurious R-peak
            # detection rather than a real beat, instead of silently
            # accepting an implausible pair.
            minimum_rr_samples = 0.200 * self.fs
            nearest_gap = min(
                (
                    abs(other.r_peak.sample - beat.r_peak.sample)
                    for other in self.beats
                    if other is not beat
                    # A beat at the exact same R-peak sample is the same
                    # beat slot (e.g. a manual-revision candidate checked
                    # before it replaces the prior version at that slot),
                    # not a separate too-close detection.
                    and other.r_peak.sample != beat.r_peak.sample
                ),
                default=None,
            )
            if nearest_gap is not None and nearest_gap < minimum_rr_samples:
                return False
        return True


class DelineatorBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def delineate(
        self,
        ecg_12lead: np.ndarray,
        r_peaks: np.ndarray,
        fs: int,
        vcg_xyz: Optional[np.ndarray] = None,
    ) -> list[BeatDelineation]: ...
