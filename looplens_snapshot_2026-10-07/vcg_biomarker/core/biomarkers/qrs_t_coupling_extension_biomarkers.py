"""Four markers around the QRS-T shape coupling, the strongest family of A117.

In the A117 10k pilot the shape distances between the QRS and T loops --
``qrs_t_normalized_discrete_frechet``, ``qrs_t_normalized_dtw_distance``,
``qrs_t_loop_mirror_symmetry``, ``qrs_t_morphological_discordance`` -- were
the only family above 0,75 validation AUC for LV systolic dysfunction, heart
failure and cardiomyopathy, on 88 % of recordings, and on the same rows they
beat both the mean spatial QRS-T angle (0,738) and the Direct-8-lead angle
(0,805 against 0,828). Every one of them is a single number from the
representative beat. This module asks the three questions the family left
open, each as a post-outcome hypothesis in the discovery reserve:

``qrs_t_frechet_beat_median`` / ``qrs_t_frechet_beat_mad``
    The same normalized discrete Fréchet distance, beat by beat, on the
    record's own primary-morphology beats: the median, and the median
    absolute deviation as the beat-to-beat stability of the coupling. A
    coupling that wanders from beat to beat is a different observation from
    one that is merely large. Hypothesis: the scatter carries prognosis
    beyond the level, as beat-to-beat repolarisation variability does.

``qrs_t_frechet_eligible_beat_count``
    How many beats entered the two summaries. A count, so that the summaries
    can be read against their denominator and the minimum can be varied as
    a declared sensitivity rather than a hidden constant.

``qrs_t_mirror_symmetry_direct_8lead``
    The mirror distance between the centred, RMS-normalised QRS and
    sign-inverted T trajectories, computed on the eight native leads (I, II,
    V1-V6) instead of the Kors XYZ. A transform-robustness check of the
    best marker: if the coupling is a property of the heart rather than of
    the transform, it should be visible in the untransformed leads too.
    Formally a candidate with the same intended reading; its correlation
    with the XYZ version is measured, not assumed.

None of these claims a mechanism. The QRS-T coupling is a global summary of
depolarisation and repolarisation sequence; its clinical reading stays at
association until an independent cohort has seen it.

**Measured on 495 unselected training recordings the day the module was
written (2026-09-18), before any endpoint was read:**

* The beat median exists on 99,8 % of recordings against 88,3 % for the
  representative-beat Fréchet distance, with a median of 11 eligible beats;
  where both exist, R² = 0,97 (Spearman 0,98). Above the 0,95 line the
  project draws for a new quantity -- so the median is declared here as an
  *availability variant* of the representative marker, not as new
  information, and the redundancy stage is expected to group the two.
* The MAD is the new quantity: R² = 0,10 against the beat median and 0,11
  against the representative distance.
* The native-lead mirror symmetry follows the Kors version at R² = 0,89,
  which is what a transform-robustness check should show, and it exists on
  89,7 % of recordings.
* The four together cost 15 ms per recording, about 3 core-hours on the
  full cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial import distance

from .base import (
    BiomarkerBase,
    BiomarkerDefinition,
    BiomarkerRegistry,
    BiomarkerResult,
)
from .catalog_biomarkers import (
    EPS,
    FeatureContext,
    FeatureValue,
    _context,
    _resample_curve,
    _shape_normalize,
)
from ..data_loader import LEAD_MAP
from ..delineation.base import RecordDelineation


QRS_T_COUPLING_EXTENSION_VERSION = "1.0.1"
QRS_T_COUPLING_EXTENSION_SOURCE = "qrs_t_coupling_extension_v1"
FAMILY = "QRS-T-Kopplung erweitert"

# The same 64 arc-length points the representative Fréchet distance uses,
# so that the beat median is the same quantity and not a coarser cousin.
RESAMPLE_POINTS = 64
# A median needs a beat on either side; three is where the Direct-8 summary
# stopped losing recordings to its count rather than to its baseline.
MINIMUM_BEATS = 3
MINIMUM_SEGMENT_SAMPLES = 8
DIRECT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
DIRECT_INDICES = tuple(LEAD_MAP[lead] for lead in DIRECT_LEADS)

REF_FRECHET = "https://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf"
REF_DIRECT_8LEAD = "https://doi.org/10.1038/s41598-026-37361-w"
REF_A117 = (
    "A117-2026-09-16-736108 10k pilot, 2026-09-18: QRS-T shape coupling family "
    "0,77-0,83 validation AUC for LVEF < 40 % on the same rows; the beat-level "
    "and native-lead questions were not in that run."
)


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_qrs_t_coupling_extension_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_qrs_t_coupling_extension_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


# ------------------------------------------------------------ Fréchet, batched


def discrete_frechet_batch(cost: np.ndarray) -> np.ndarray:
    """Discrete Fréchet distance for a stack of cost matrices, shape (B, n, m).

    The same recurrence as the representative-beat calculation, run over all
    beats of a recording at once: the 4.096 cells of a 64 x 64 table are
    visited once per recording instead of once per beat.
    """

    values = np.asarray(cost, dtype=float)
    if values.ndim != 3 or not values.size:
        return np.full(values.shape[0] if values.ndim == 3 else 0, np.nan)
    batch, rows, columns = values.shape
    table = np.empty_like(values)
    table[:, 0, 0] = values[:, 0, 0]
    for row in range(1, rows):
        table[:, row, 0] = np.maximum(table[:, row - 1, 0], values[:, row, 0])
    for column in range(1, columns):
        table[:, 0, column] = np.maximum(table[:, 0, column - 1], values[:, 0, column])
    for row in range(1, rows):
        for column in range(1, columns):
            table[:, row, column] = np.maximum(
                values[:, row, column],
                np.minimum(
                    np.minimum(table[:, row - 1, column], table[:, row - 1, column - 1]),
                    table[:, row, column - 1],
                ),
            )
    result = table[:, -1, -1]
    finite = np.all(np.isfinite(values.reshape(batch, -1)), axis=1)
    return np.where(finite, result, np.nan)


def _beat_sample(beat, attr: str, wave: str) -> Optional[int]:
    # One calibrated fiducial reader for every beat-level module.
    from .thesis_derived_biomarkers import _beat_sample as read

    return read(beat, attr, wave)


def _beat_loop_pairs(ctx: FeatureContext) -> list[tuple[np.ndarray, np.ndarray]]:
    """(QRS, T) arc-length curves per primary-morphology beat of the record."""

    def build() -> list[tuple[np.ndarray, np.ndarray]]:
        record = ctx._record_vcg_xyz
        if record is None:
            return []
        record = np.asarray(record, dtype=float)
        if record.ndim != 2 or record.shape[1] != 3:
            return []
        beats = getattr(ctx.delineation, "beats", None) or ()
        limit = len(record) - 1
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for beat in beats:
            if getattr(beat, "morphology_cluster_id", None) != 0:
                continue
            q0 = _beat_sample(beat, "r_onset", "qrs")
            q1 = _beat_sample(beat, "r_offset", "qrs")
            t0 = _beat_sample(beat, "t_onset", "t")
            t1 = _beat_sample(beat, "t_offset", "t")
            if None in (q0, q1, t0, t1):
                continue
            if not (0 <= q0 < q1 <= t0 < t1 <= limit):
                continue
            if q1 - q0 < MINIMUM_SEGMENT_SAMPLES or t1 - t0 < MINIMUM_SEGMENT_SAMPLES:
                continue
            qrs = record[q0 : q1 + 1]
            t_wave = record[t0 : t1 + 1]
            if not (np.all(np.isfinite(qrs)) and np.all(np.isfinite(t_wave))):
                continue
            q_curve = _resample_curve(qrs, RESAMPLE_POINTS)
            t_curve = _resample_curve(t_wave, RESAMPLE_POINTS)
            if q_curve is None or t_curve is None:
                continue
            q_normalized = _shape_normalize(q_curve)
            t_normalized = _shape_normalize(t_curve)
            if q_normalized is None or t_normalized is None:
                continue
            pairs.append((q_normalized, t_normalized))
        return pairs

    return _memo(ctx, ("beat_loop_pairs",), build)


def _beat_frechet_values(ctx: FeatureContext) -> np.ndarray:
    def build() -> np.ndarray:
        pairs = _beat_loop_pairs(ctx)
        if not pairs:
            return np.empty(0)
        cost = np.stack([distance.cdist(q, t) for q, t in pairs])
        values = discrete_frechet_batch(cost)
        return values[np.isfinite(values)]

    return _memo(ctx, ("beat_frechet_values",), build)


def _frechet_notes(count: int) -> str:
    return (
        f"contributing_beats={count}; minimum={MINIMUM_BEATS}; primary_morphology_cluster_only; "
        "same_normalized_frechet_as_representative_marker; beat_class_homogeneity=not_verified"
    )


def _frechet_beat_median(ctx: FeatureContext) -> FeatureValue:
    values = _beat_frechet_values(ctx)
    if len(values) < MINIMUM_BEATS:
        return _abstain(f"fewer than {MINIMUM_BEATS} beats with a measurable QRS and T loop; found {len(values)}")
    return FeatureValue(float(np.median(values)), notes=_frechet_notes(len(values)))


def _frechet_beat_mad(ctx: FeatureContext) -> FeatureValue:
    values = _beat_frechet_values(ctx)
    if len(values) < MINIMUM_BEATS:
        return _abstain(f"fewer than {MINIMUM_BEATS} beats with a measurable QRS and T loop; found {len(values)}")
    median = float(np.median(values))
    return FeatureValue(
        float(np.median(np.abs(values - median))),
        notes=_frechet_notes(len(values)) + "; MAD is not a confidence interval",
    )


def _frechet_beat_count(ctx: FeatureContext) -> FeatureValue:
    if ctx._record_vcg_xyz is None:
        return _abstain("no record VCG")
    values = _beat_frechet_values(ctx)
    if not values.size and not (ctx.is_segment_evaluable("qrs") and ctx.is_segment_evaluable("t")):
        # A representative beat can fail while record-level beats are usable.
        # Never hide their actual contribution to a reported median/MAD.
        return _abstain("QRS or T segment not evaluable")
    return FeatureValue(float(len(values)), notes="beats entering the Fréchet beat summaries")


# ------------------------------------------------ mirror symmetry, native leads


def _mirror_symmetry_direct_8lead(ctx: FeatureContext) -> FeatureValue:
    signal = ctx.ecg_12lead
    if signal is None:
        return _abstain("no representative 12-lead signal")
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or signal.shape[1] < len(LEAD_MAP):
        return _abstain("representative signal is not a 12-lead array")
    if not (ctx.is_segment_evaluable("qrs") and ctx.is_segment_evaluable("t")):
        return _abstain("QRS or T segment not evaluable")
    qrs = ctx.bounds("qrs")
    t_wave = ctx.bounds("t")
    if qrs is None or t_wave is None:
        return _abstain("QRS or T bounds missing")
    q0, q1 = qrs
    t0, t1 = t_wave
    if not (0 <= q0 < q1 <= t0 < t1 < len(signal)):
        return _abstain("segment boundaries out of order")
    q = signal[q0 : q1 + 1][:, list(DIRECT_INDICES)]
    t = signal[t0 : t1 + 1][:, list(DIRECT_INDICES)]
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(t))):
        return _abstain("non-finite samples in the native leads")
    q_curve = _resample_curve(q, RESAMPLE_POINTS)
    t_curve = _resample_curve(t, RESAMPLE_POINTS)
    if q_curve is None or t_curve is None:
        return _abstain("degenerate native-lead trajectory")
    q_normalized = _shape_normalize(q_curve)
    t_normalized = _shape_normalize(t_curve)
    if q_normalized is None or t_normalized is None:
        return _abstain("zero-scale native-lead trajectory")
    # Centering makes the value baseline-free: no isoelectric window enters.
    mirror = float(np.sqrt(np.mean((q_normalized + t_normalized) ** 2)))
    if not np.isfinite(mirror):
        return _abstain("non-finite mirror distance")
    return FeatureValue(
        float(mirror / (1.0 + mirror)),
        notes=(
            "native leads I, II, V1-V6; centred and RMS-normalised, baseline-free; "
            "same bounded form as qrs_t_loop_mirror_symmetry on Kors XYZ"
        ),
    )


# ------------------------------------------------------------------ registry


@dataclass(frozen=True)
class QrsTCouplingExtensionSpec:
    name: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    references: tuple[str, ...] = ()


QRS_T_COUPLING_EXTENSION_SPECS: tuple[QrsTCouplingExtensionSpec, ...] = (
    QrsTCouplingExtensionSpec(
        "qrs_t_frechet_beat_median",
        "Median over the record's primary-morphology beats of the normalized "
        "discrete Fréchet distance between each beat's own QRS and T loops; "
        "an availability variant of qrs_t_normalized_discrete_frechet (R² 0,97 "
        "where both exist, 99,8 % against 88,3 % available).",
        "median_b d_F(norm(Q_b), norm(T_b)); 64 arc-length points, centred and "
        "RMS-scaled per curve; >= 3 beats",
        "dimensionless",
        _frechet_beat_median,
        (REF_FRECHET, REF_A117),
    ),
    QrsTCouplingExtensionSpec(
        "qrs_t_frechet_beat_mad",
        "Median absolute deviation across beats of the normalized discrete "
        "Fréchet distance: beat-to-beat stability of the QRS-T shape coupling.",
        "median_b |d_F,b - median_b d_F,b|; >= 3 beats",
        "dimensionless",
        _frechet_beat_mad,
        (REF_FRECHET, REF_A117),
    ),
    QrsTCouplingExtensionSpec(
        "qrs_t_frechet_eligible_beat_count",
        "Number of primary-morphology beats with confident QRS and T bounds "
        "that entered the Fréchet beat summaries.",
        "count of beats with d_F,b",
        "count",
        _frechet_beat_count,
        (REF_A117,),
    ),
    QrsTCouplingExtensionSpec(
        "qrs_t_mirror_symmetry_direct_8lead",
        "Bounded mirror distance between the centred, RMS-normalised QRS and "
        "sign-inverted T trajectories in the eight native leads I, II, V1-V6.",
        "d / (1 + d), d = rms(norm(Q) + norm(T)) over 64 arc-length points in "
        "native-lead space; representative beat",
        "dimensionless",
        _mirror_symmetry_direct_8lead,
        (REF_DIRECT_8LEAD, REF_A117),
    ),
)


class QrsTCouplingExtensionBiomarker(BiomarkerBase):
    def __init__(self, spec: QrsTCouplingExtensionSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def unit(self) -> str:
        return self._spec.unit

    @property
    def category(self) -> str:
        return "vcg"

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def references(self) -> list[str]:
        return list(self._spec.references)

    def compute(
        self,
        delineation: RecordDelineation,
        ecg_12lead: Optional[np.ndarray] = None,
        vcg_xyz: Optional[np.ndarray] = None,
        fs: int = 500,
    ) -> BiomarkerResult:
        ctx = _context(delineation, ecg_12lead, vcg_xyz, fs)
        try:
            value = self._spec.calculator(ctx)
        except Exception as error:  # noqa: BLE001 - one marker must not stop a panel
            value = _abstain(f"{type(error).__name__}: {error}")
        notes = [f"qrs_t_coupling_extension_version={QRS_T_COUPLING_EXTENSION_VERSION}"]
        if value.notes:
            notes.insert(0, value.notes)
        return BiomarkerResult(
            name=self._spec.name,
            value=value.value,
            unit=self._spec.unit,
            category="vcg",
            confidence=float(value.confidence),
            description=self._spec.description,
            references=list(self._spec.references),
            computation_notes="; ".join(notes),
            family=FAMILY,
            formula=self._spec.formula,
        )


def qrs_t_coupling_extension_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in QRS_T_COUPLING_EXTENSION_SPECS)


def register_qrs_t_coupling_extension_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in QRS_T_COUPLING_EXTENSION_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), QrsTCouplingExtensionBiomarker):
                continue
            raise RuntimeError(
                f"Coupling-extension marker collides with an existing biomarker: {spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=FAMILY,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=QRS_T_COUPLING_EXTENSION_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Post-outcome hypothesis formed from the A117 pilot's QRS-T "
                    "coupling result; discovery reserve only, no cutoff, no "
                    "mechanism claimed. Its correlation with the representative-"
                    "beat marker it extends is measured before any reading."
                ),
            )
        )
        BiomarkerRegistry.register(QrsTCouplingExtensionBiomarker(spec))


__all__ = [
    "FAMILY",
    "MINIMUM_BEATS",
    "QRS_T_COUPLING_EXTENSION_SOURCE",
    "QRS_T_COUPLING_EXTENSION_SPECS",
    "QRS_T_COUPLING_EXTENSION_VERSION",
    "QrsTCouplingExtensionBiomarker",
    "discrete_frechet_batch",
    "qrs_t_coupling_extension_marker_ids",
    "register_qrs_t_coupling_extension_biomarkers",
]
