"""Markers for the physiology/pathology theses of 23.09.2026.

The theses are in ``THESEN_2026-09-23_PHYSIOLOGIE_UND_PATHOLOGIE.md`` at the
project root; their fixed versions in ``THESEN_FASSUNGEN_2026-09-24.md``.
Section 10 of the theses names the only extraction additions worth making
before the restart: the T-area and gradient difference of the secondary
morphology cluster (T10), optionally a PR-segment vector (T15). J-Tpeak (T6)
is derivable from intervals but is written here from one beat's own
fiducials so that the numerator and the correction share one RR.

``secondary_morphology_*`` (T10)
    Wilson/Geselowitz: the QRST integral does not depend on the activation
    sequence to first order, so a change of sequence (an extrasystole) moves
    the T area by about minus the QRS-area change. Per record the mean QRS
    and JT area vectors of the secondary cluster (id 1) minus those of the
    primary cluster (id 0), origin-corrected like every SVG marker, beats
    with a complete T wave only (T offset before the next beat's P or QRS).
    The ratio |delta SVG| / |delta QRS area| is the thesis' R; the cosine
    between delta T and delta QRS is expected near -1.

``pr_segment_to_st_vector_angle_deg``, ``pr_segment_to_p_vector_ratio`` (T15)
    Pericarditis injures atria and ventricles, so the PR-segment vector
    (the registry's Ta vector: PR mean minus TP reference) should oppose the
    ST vector, and be large relative to the P vector.

``j_tpeak_ms``, ``j_tpeak_c_ms`` (T6)
    J point to T peak of the representative beat; corrected as
    J-Tpeak / (RR / 1000 ms)^0.58 (Vicente et al., Clin Pharmacol Ther 2019,
    PMID 30447156; method from Johannesen et al. 2014, PMID 25054430).

All are post-outcome hypotheses: written after the project's results were
seen, never core or sentinel markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .base import (
    BiomarkerBase,
    BiomarkerDefinition,
    BiomarkerRegistry,
    BiomarkerResult,
)
from .catalog_biomarkers import (
    FeatureContext,
    FeatureValue,
    _context,
    _trapz,
    _vector_array,
)
from ..delineation.base import RecordDelineation, fiducial_confidence_threshold, ventricular_rr_seconds

PHYSIOLOGY_THESIS_VERSION = "1.0.0"
PHYSIOLOGY_THESIS_SOURCE = "physiology_thesis_v1"
EPS = 1e-12
MINIMUM_PRIMARY_BEATS = 3
MINIMUM_SECONDARY_BEATS = 1
BOUNDARY_CONFIDENCE = 0.35
J_TPEAK_RR_EXPONENT = 0.58


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _point(beat, attribute: str, minimum: float = BOUNDARY_CONFIDENCE) -> Optional[int]:
    point = getattr(beat, attribute, None)
    if point is None or getattr(point, "confidence", 1.0) < minimum:
        return None
    return int(point.sample)


def _cluster_area_pairs(ctx: FeatureContext, cluster_id: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Origin-corrected (QRS area, JT area) per beat of one cluster, mV*s."""

    record = ctx._record_vcg_xyz
    assessment = ctx.vcg_origin_assessment
    if record is None or assessment.primary is None:
        return []
    origin = assessment.primary.vector_millivolts[np.newaxis, :]
    beats = ctx.delineation.beats
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index, beat in enumerate(beats):
        if getattr(beat, "morphology_cluster_id", None) != cluster_id:
            continue
        onset, offset = _point(beat, "r_onset"), _point(beat, "r_offset")
        t_offset = _point(beat, "t_offset")
        if onset is None or offset is None or t_offset is None:
            continue
        if not 0 <= onset < offset < t_offset < len(record):
            continue
        if beat.resolved_wave_status("t").value != "present":
            continue
        if index + 1 < len(beats):
            following = beats[index + 1]
            limits = [
                sample
                for sample in (
                    _point(following, "p_onset", 0.0),
                    _point(following, "r_onset", 0.0),
                )
                if sample is not None
            ]
            if limits and t_offset >= min(limits):
                continue  # T wave truncated by the next beat
        qrs = np.asarray(_trapz(record[onset:offset + 1] - origin, 1.0 / ctx.fs, axis=0), dtype=float)
        jt = np.asarray(_trapz(record[offset:t_offset + 1] - origin, 1.0 / ctx.fs, axis=0), dtype=float)
        if np.all(np.isfinite(qrs)) and np.all(np.isfinite(jt)):
            pairs.append((qrs, jt))
    return pairs


def _cluster_deltas(ctx: FeatureContext) -> Optional[tuple[np.ndarray, np.ndarray]]:
    cache = getattr(ctx, "_physiology_thesis_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_physiology_thesis_cache", cache)
    if "deltas" not in cache:
        primary = _cluster_area_pairs(ctx, 0)
        secondary = _cluster_area_pairs(ctx, 1)
        if len(primary) < MINIMUM_PRIMARY_BEATS or len(secondary) < MINIMUM_SECONDARY_BEATS:
            cache["deltas"] = None
        else:
            delta_qrs = np.mean([q for q, _ in secondary], axis=0) - np.mean([q for q, _ in primary], axis=0)
            delta_t = np.mean([t for _, t in secondary], axis=0) - np.mean([t for _, t in primary], axis=0)
            cache["deltas"] = (delta_qrs, delta_t)
    return cache["deltas"]


def _t_area_delta(ctx: FeatureContext) -> FeatureValue:
    deltas = _cluster_deltas(ctx)
    if deltas is None:
        return _abstain("fewer than 3 primary or 1 secondary beats with a complete T wave")
    return FeatureValue(float(np.linalg.norm(deltas[1])))


def _svg_delta(ctx: FeatureContext) -> FeatureValue:
    deltas = _cluster_deltas(ctx)
    if deltas is None:
        return _abstain("fewer than 3 primary or 1 secondary beats with a complete T wave")
    return FeatureValue(float(np.linalg.norm(deltas[0] + deltas[1])))


def _svg_to_qrs_ratio(ctx: FeatureContext) -> FeatureValue:
    deltas = _cluster_deltas(ctx)
    if deltas is None:
        return _abstain("fewer than 3 primary or 1 secondary beats with a complete T wave")
    qrs_change = float(np.linalg.norm(deltas[0]))
    if qrs_change <= EPS:
        return _abstain("no QRS-area change between the clusters")
    return FeatureValue(float(np.linalg.norm(deltas[0] + deltas[1]) / qrs_change))


def _t_qrs_cosine(ctx: FeatureContext) -> FeatureValue:
    deltas = _cluster_deltas(ctx)
    if deltas is None:
        return _abstain("fewer than 3 primary or 1 secondary beats with a complete T wave")
    a, b = deltas[1], deltas[0]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return _abstain("degenerate area change")
    return FeatureValue(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


def _pr_vectors(ctx: FeatureContext) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from .electrophysiology_markers import _mean_vector, _ta_vector

    ta = _ta_vector(ctx)
    p_vector = _mean_vector(ctx.vcg_segment("p"))
    st = _vector_array(ctx, "st", "mean")
    if ta is None or p_vector is None or st is None:
        return None
    st = np.asarray(st, dtype=float)
    if st.shape != (3,) or not np.all(np.isfinite(st)):
        return None
    return np.asarray(ta, dtype=float), np.asarray(p_vector, dtype=float), st


def _pr_to_st_angle(ctx: FeatureContext) -> FeatureValue:
    vectors = _pr_vectors(ctx)
    if vectors is None:
        return _abstain("PR, P or ST vector unavailable")
    ta, _, st = vectors
    na, ns = float(np.linalg.norm(ta)), float(np.linalg.norm(st))
    if na <= EPS or ns <= EPS:
        return _abstain("degenerate PR or ST vector")
    cosine = float(np.clip(np.dot(ta, st) / (na * ns), -1.0, 1.0))
    return FeatureValue(float(np.degrees(np.arccos(cosine))))


def _pr_to_p_ratio(ctx: FeatureContext) -> FeatureValue:
    vectors = _pr_vectors(ctx)
    if vectors is None:
        return _abstain("PR, P or ST vector unavailable")
    ta, p_vector, _ = vectors
    p_size = float(np.linalg.norm(p_vector))
    if p_size <= EPS:
        return _abstain("no P vector")
    return FeatureValue(float(np.linalg.norm(ta) / p_size))


def _j_tpeak_ms(ctx: FeatureContext) -> Optional[float]:
    beat = ctx.delineation.representative_beat
    if beat is None:
        return None
    j = _point(beat, "j_point")
    if j is None:
        j = _point(beat, "r_offset")
    t_peak = _point(beat, "t_peak", fiducial_confidence_threshold("t_peak"))
    if j is None or t_peak is None or t_peak <= j:
        return None
    if beat.resolved_wave_status("t").value != "present":
        return None
    return (t_peak - j) / ctx.fs * 1000.0


def _j_tpeak(ctx: FeatureContext) -> FeatureValue:
    value = _j_tpeak_ms(ctx)
    return _abstain("J point or T peak not evaluable") if value is None else FeatureValue(value)


def _j_tpeak_c(ctx: FeatureContext) -> FeatureValue:
    value = _j_tpeak_ms(ctx)
    if value is None:
        return _abstain("J point or T peak not evaluable")
    peaks = sorted(beat.r_peak.sample for beat in ctx.delineation.beats)
    if len(peaks) < 2:
        return _abstain("fewer than two beats for RR")
    rr_seconds = ventricular_rr_seconds(peaks, ctx.fs)
    if rr_seconds is None:
        return _abstain("non-positive RR")
    return FeatureValue(float(value / rr_seconds ** J_TPEAK_RR_EXPONENT))


@dataclass(frozen=True)
class PhysiologyThesisSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    thesis: str
    references: tuple[str, ...] = ()


REF_THESES = (
    "THESEN_2026-09-23_PHYSIOLOGIE_UND_PATHOLOGIE.md, project root: own theses "
    "with their refuting observations; fixed versions in "
    "THESEN_FASSUNGEN_2026-09-24.md."
)
REF_GESELOWITZ = "Geselowitz DB. IEEE Trans Biomed Eng 1983;30:76-77, PMID 6826192 (ventricular gradient)."
REF_VICENTE = (
    "Vicente J et al. Clin Pharmacol Ther 2019, PMID 30447156 "
    "(J-Tpeak_c = J-Tpeak / (RR/1000)^0.58)."
)

PHYSIOLOGY_THESIS_SPECS: tuple[PhysiologyThesisSpec, ...] = (
    PhysiologyThesisSpec(
        "secondary_morphology_t_area_delta_magnitude",
        "Thesenmarker Gradient-Invarianz",
        "Magnitude of the JT-area vector difference, secondary minus primary "
        "morphology cluster.",
        "||mean JT area[secondary] - mean JT area[primary]||, origin-corrected, "
        "complete T waves only.",
        "mV*s",
        _t_area_delta,
        "T10: the T area compensates the QRS-area change of a changed sequence.",
        (REF_THESES, REF_GESELOWITZ),
    ),
    PhysiologyThesisSpec(
        "secondary_morphology_svg_delta_magnitude",
        "Thesenmarker Gradient-Invarianz",
        "Magnitude of the ventricular-gradient difference, secondary minus "
        "primary morphology cluster.",
        "||delta QRS area + delta JT area||.",
        "mV*s",
        _svg_delta,
        "T10: the gradient survives the change of activation sequence.",
        (REF_THESES, REF_GESELOWITZ),
    ),
    PhysiologyThesisSpec(
        "secondary_morphology_svg_to_qrs_area_delta_ratio",
        "Thesenmarker Gradient-Invarianz",
        "R of thesis T10: gradient change relative to QRS-area change between "
        "the morphology clusters.",
        "||delta SVG|| / ||delta QRS area||.",
        "ratio",
        _svg_to_qrs_ratio,
        "T10: median R well below 1; refuted if the median R is at least 1.",
        (REF_THESES, REF_GESELOWITZ),
    ),
    PhysiologyThesisSpec(
        "secondary_morphology_t_qrs_area_delta_cosine",
        "Thesenmarker Gradient-Invarianz",
        "Cosine between the JT-area change and the QRS-area change of the "
        "secondary cluster.",
        "cos(delta JT area, delta QRS area), -1 to 1.",
        "cosine",
        _t_qrs_cosine,
        "T10: the T-area change opposes the QRS-area change (cosine near -1).",
        (REF_THESES, REF_GESELOWITZ),
    ),
    PhysiologyThesisSpec(
        "pr_segment_to_st_vector_angle_deg",
        "Thesenmarker Perikard",
        "Angle between the PR-segment (Ta) vector and the mean ST vector.",
        "angle(PR mean - TP reference, mean ST vector), 0-180 degrees.",
        "degrees",
        _pr_to_st_angle,
        "T15: near-antiparallel in pericarditis, not in STEMI.",
        (REF_THESES,),
    ),
    PhysiologyThesisSpec(
        "pr_segment_to_p_vector_ratio",
        "Thesenmarker Perikard",
        "Size of the PR-segment (Ta) vector relative to the mean P vector.",
        "||Ta vector|| / ||mean P vector||.",
        "ratio",
        _pr_to_p_ratio,
        "T15: larger in pericarditis than atrial repolarisation alone explains.",
        (REF_THESES,),
    ),
    PhysiologyThesisSpec(
        "j_tpeak_ms",
        "Thesenmarker Repolarisation",
        "J point to T peak of the representative beat.",
        "T peak - J point (J point, else QRS offset), ms.",
        "ms",
        _j_tpeak,
        "T6: early repolarisation, prolonged by pure IKr block.",
        (REF_THESES, REF_VICENTE),
    ),
    PhysiologyThesisSpec(
        "j_tpeak_c_ms",
        "Thesenmarker Repolarisation",
        "Rate-corrected J-Tpeak.",
        "J-Tpeak / (mean RR / 1000 ms)^0.58.",
        "ms",
        _j_tpeak_c,
        "T6/T7/T8: early repolarisation independent of rate.",
        (REF_THESES, REF_VICENTE),
    ),
)


class PhysiologyThesisBiomarker(BiomarkerBase):
    def __init__(self, spec: PhysiologyThesisSpec) -> None:
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
        notes = [f"physiology_thesis_version={PHYSIOLOGY_THESIS_VERSION}"]
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
            family=self._spec.family,
            formula=self._spec.formula,
        )


def physiology_thesis_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in PHYSIOLOGY_THESIS_SPECS)


def register_physiology_thesis_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in PHYSIOLOGY_THESIS_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), PhysiologyThesisBiomarker):
                continue
            raise RuntimeError(
                f"Physiology thesis marker collides with an existing biomarker: {spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=PHYSIOLOGY_THESIS_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Exploratory post-outcome hypothesis with a refuting "
                    "observation. " + spec.thesis + " No diagnostic cutoff is claimed."
                ),
            )
        )
        BiomarkerRegistry.register(PhysiologyThesisBiomarker(spec))
