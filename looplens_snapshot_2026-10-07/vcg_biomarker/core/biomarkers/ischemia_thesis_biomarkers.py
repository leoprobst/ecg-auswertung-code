"""Three markers derived from the acute-ischaemia theses, after a fourth was measured and withdrawn.

The theses are in ``THESEN_AKUTE_ISCHAEMIE_2026-09-12.md`` at the project
root. Their motivation is spatially heterogeneous injury currents, not an
invertible lesion map. The measured ST vector depends on baseline, tissue
sources and volume conduction. The global vector at a speed minimum does
not localize a conduction obstacle. Comparing these vectors is a descriptive
hypothesis, not two independent regional proofs. The historical hypotheses
below are not validated mechanisms; see the revised per-marker specs and
research_pathology_round_20260913/01_ISCHEMIA.md for the forward-model limits.

The six principles of expert reading that Helseth, McLaren, de Alencar and
Smith set out (Eur Heart J Acute Cardiovasc Care, September 2026) --
dynamicity, acuteness, reciprocity, proportionality, totality, surrogacy --
are the frame; the contribution is their translation into vector quantities.

``qrs_stall_to_st_vector_angle_deg``
    Thesis O2, the centre of the paper. In acute occlusion the front stalls
    in the ischaemic zone and the injury current points at it, so the two
    vectors align. In secondary ST change -- strain, bundle branch block --
    the ST vector opposes the QRS and the front stalls where it always does,
    posterobasally. Predicted small in acute infarction and large in
    hypertrophy without infarction, and the second half is the test.

``terminal_qrs_to_st_vector_angle_deg``
    Thesis O3. Terminal QRS distortion read as alignment rather than loss:
    the ischaemic zone is activated last and repolarises at once, so the
    terminal QRS vector runs into the ST vector without returning to the
    baseline. Terminal QRS vectors and their angles to the main QRS and to
    the late T exist in the registry; the angle to the ST vector did not.

``st_vector_beatwise_direction_dispersion``
    Thesis O4. Occlusion is unstable on the scale of seconds -- collateral
    flow, spasm, thrombus -- and a zone whose border moves turns the vector.
    The registry has beat-to-beat variability of the ST *level*; direction is
    the more specific quantity, and is computed as the circular dispersion
    of the per-beat unit ST vectors, seam-free.

All three are angles or fractions. None carries an amplitude, none an ECG
interval.

**A fourth was built, measured and withdrawn.** ``lead8_st_nondipolar_fraction``
was to carry thesis O1 -- reciprocity as the projection of one dipole, and
departure from dipolarity as the information. Measured on 2.964 recordings it
had no dynamic range: the eight-lead ST deviation is 99,6-99,8 % explained by
the three dipolar singular values in everyone, so the residual is noise at
the fourth decimal (median 0,0019 against 0,0040), and its direction across
demand injury was the reverse of the prediction. The thesis may still be
true; this quantity cannot test it, and a marker without range is not kept
for the sake of the idea behind it.

**Tested on 2.964 recordings of the training split before this text was
finished.** The acute endpoints are nearly empty in the pilot cohort --
``acute_mi_high_specificity`` five positives, ``omi_revascularization_proxy``
seven -- so the acute end of every thesis waits for the full run. What could
be tested is the *other* end, the cross-prediction, and it held:

* ``qrs_stall_to_st_vector_angle_deg`` -- median **123 degrees in strict echo
  hypertrophy against 89 in its controls (+34)**, 96 in old infarction (+14),
  and **62 in probable acute infarction against 82 (-20)**. The ordering runs
  hypertrophy > old infarction > controls > acute -- the primary/secondary
  axis the thesis names. The prediction for demand injury (large angle,
  diffuse stall) was wrong: 81 against 97.
* ``terminal_qrs_to_st_vector_angle_deg`` -- 82 in hypertrophy against 63
  (+19), as a secondary change should read; old infarction 61 against 66,
  five degrees the other way from the prediction.
* ``st_vector_beatwise_direction_dispersion`` -- AUC 0,47-0,48 at every
  testable endpoint and 0,38 in hypertrophy, i.e. *more* stable there, which
  is what a secondary change should do. The acute end, where the thesis
  lives, is untested.

Both angles abstain on about half of recordings because the ST vector is
below five per cent of the QRS maximum. That is correct rather than a
defect: a normal complex has no injury current, and an angle to a vector
that is not there would be a number without a referent.
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
    _vector_array,
    _window_vector,
)
from ..delineation.base import RecordDelineation, fiducial_confidence_threshold


ISCHEMIA_THESIS_VERSION = "1.1.0"
ISCHEMIA_THESIS_SOURCE = "ischemia_thesis_v1"

EPS = 1e-12

# The last 40 ms of the QRS, matching the registry's terminal-40-ms vectors.
TERMINAL_QRS_MS = 40

# A vector below this share of the reference has no direction worth an angle.
MINIMUM_VECTOR_SHARE = 0.05

MINIMUM_BEATS = 3


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_ischemia_thesis_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_ischemia_thesis_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return None
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _st_vector(ctx: FeatureContext) -> Optional[np.ndarray]:
    """The mean origin-corrected ST vector the registry's ST markers read."""

    def build() -> Optional[np.ndarray]:
        vector = _vector_array(ctx, "st", "mean")
        if vector is None:
            return None
        values = np.asarray(vector, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            return None
        return values

    return _memo(ctx, ("st_vector",), build)


# ------------------------------------------------------- the two alignments


def _stall_to_st_angle(ctx: FeatureContext) -> FeatureValue:
    """Angle between where the front stalls and where the injury points.

    The stall vector is the one ``thesis_derived_biomarkers`` computes; it is
    read through that module so the two markers cannot disagree about the
    stall.
    """

    from .thesis_derived_biomarkers import _stall_vector

    stall = _stall_vector(ctx)
    if stall is None:
        return _abstain("no interior stall with a defined direction")
    st = _st_vector(ctx)
    if st is None:
        return _abstain("ST vector unavailable")
    reference = _vector_array(ctx, "qrs", "max")
    scale = float(np.linalg.norm(reference)) if reference is not None else 0.0
    if scale > EPS and float(np.linalg.norm(st)) < MINIMUM_VECTOR_SHARE * scale:
        return _abstain("ST vector too small to carry a direction")
    angle = _angle_deg(stall, st)
    if angle is None:
        return _abstain("degenerate vector")
    return FeatureValue(angle)


def _terminal_qrs_to_st_angle(ctx: FeatureContext) -> FeatureValue:
    """Angle between the last 40 ms of the QRS and the ST vector."""

    terminal = _window_vector(ctx, "qrs", length_ms=TERMINAL_QRS_MS, end_aligned=True)
    if terminal is None:
        return _abstain("terminal QRS window unavailable")
    terminal = np.asarray(terminal, dtype=float)
    st = _st_vector(ctx)
    if st is None:
        return _abstain("ST vector unavailable")
    reference = _vector_array(ctx, "qrs", "max")
    scale = float(np.linalg.norm(reference)) if reference is not None else 0.0
    if scale > EPS:
        if float(np.linalg.norm(st)) < MINIMUM_VECTOR_SHARE * scale:
            return _abstain("ST vector too small to carry a direction")
        if float(np.linalg.norm(terminal)) < MINIMUM_VECTOR_SHARE * scale:
            return _abstain("terminal QRS vector too small to carry a direction")
    angle = _angle_deg(terminal, st)
    if angle is None:
        return _abstain("degenerate vector")
    return FeatureValue(angle)


# ---------------------------------------------- beat-to-beat ST direction


def _beat_sample(beat, attr: str, wave: str) -> Optional[int]:
    point = getattr(beat, attr, None)
    if point is None:
        return None
    try:
        confidence = float(point.confidence)
        sample = float(point.sample)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if (
        not np.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
        or confidence < fiducial_confidence_threshold(attr)
        or not np.isfinite(sample)
        or sample < 0
        or not sample.is_integer()
    ):
        return None
    status = getattr(beat, "resolved_wave_status", None)
    if status is not None:
        try:
            if status(wave).value != "present":
                return None
        except Exception:  # noqa: BLE001 - unreadable quality status must abstain
            return None
    return int(sample)


def _beat_st_directions(ctx: FeatureContext) -> list[np.ndarray]:
    """Unit ST vector of each beat, from the record's own beats.

    Each beat's ST window runs from its J point to its T onset, and the
    vector is the window mean relative to that beat's own PR baseline, so a
    slow drift of the whole record cannot masquerade as a turning injury.
    """

    def build() -> list[np.ndarray]:
        record = ctx._record_vcg_xyz
        if record is None:
            return []
        beats = getattr(ctx.delineation, "beats", None) or ()
        limit = len(record) - 1
        reference = _vector_array(ctx, "qrs", "max")
        scale = float(np.linalg.norm(reference)) if reference is not None else 0.0
        directions: list[np.ndarray] = []
        for beat in beats:
            j = _beat_sample(beat, "j_point", "qrs")
            t_on = _beat_sample(beat, "t_onset", "t")
            p_off = _beat_sample(beat, "p_offset", "p")
            r_on = _beat_sample(beat, "r_onset", "qrs")
            if j is None or t_on is None or r_on is None:
                continue
            if not 0 <= r_on <= j < t_on <= limit:
                continue
            if t_on - j < 3:
                continue
            # baseline: the PR stretch if the P was found, else the 20 ms
            # before QRS onset
            if p_off is not None and int(p_off) < r_on:
                base_lo = max(0, min(int(p_off), limit))
            else:
                base_lo = max(0, r_on - int(0.020 * float(ctx.fs)))
            if r_on - base_lo < 2:
                continue
            baseline_samples = np.asarray(record[base_lo:r_on], dtype=float)
            if not np.all(np.isfinite(baseline_samples)):
                continue
            baseline = baseline_samples.mean(axis=0)
            window = np.asarray(record[j : t_on + 1], dtype=float)
            if window.ndim != 2 or window.shape[1] != 3 or not np.all(np.isfinite(window)):
                continue
            vector = window.mean(axis=0) - baseline
            magnitude = float(np.linalg.norm(vector))
            if not np.isfinite(magnitude) or magnitude <= EPS or (scale > EPS and magnitude < MINIMUM_VECTOR_SHARE * scale):
                continue
            directions.append(vector / magnitude)
        return directions

    return _memo(ctx, ("beat_st_directions",), build)


def _st_direction_dispersion(ctx: FeatureContext) -> FeatureValue:
    """One minus the mean resultant length of the per-beat unit ST vectors.

    Zero: the injury points the same way every beat. One: it cancels
    completely across beats. Circular, so no seam, and independent of the
    ST magnitude by construction -- the prediction is that a small, unstable
    vector is more suspicious than a large, stable one.
    """

    directions = _beat_st_directions(ctx)
    if len(directions) < MINIMUM_BEATS:
        return _abstain("fewer than three beats with a measurable ST vector")
    stack = np.stack(directions)
    resultant = float(np.linalg.norm(stack.mean(axis=0)))
    if not np.isfinite(resultant):
        return _abstain("non-finite resultant")
    return FeatureValue(
        float(np.clip(1.0 - resultant, 0.0, 1.0)),
        notes=(f"contributing_beats={len(directions)}; "
               "beat_class_homogeneity=not_verified; absolute_snr_gate=not_validated; "
               "interpretation=descriptive_direction_variability_not_ischemia_diagnosis"),
    )


# -------------------------------------------------------------------- specs


@dataclass(frozen=True)
class IschemiaThesisSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    thesis: str
    references: tuple[str, ...] = ()


REF_THESES = (
    "THESEN_AKUTE_ISCHAEMIE_2026-09-12.md, project root: own theses on acute "
    "ischaemia, each with its refuting observation."
)
REF_HELSETH = (
    "Helseth HC, Mansur P, El-Baba M, McLaren JTT, de Alencar JN, Smith SW. "
    "Electrocardiographic principles for the diagnosis of occlusion "
    "myocardial infarction. Eur Heart J Acute Cardiovasc Care 2026."
)
REF_GORGELS = (
    "Gorgels APM. The electrophysiologic continuum of occlusion myocardial "
    "infarction. J Electrocardiol 2026;96:154271."
)


ISCHEMIA_THESIS_SPECS: tuple[IschemiaThesisSpec, ...] = (
    IschemiaThesisSpec(
        "qrs_stall_to_st_vector_angle_deg",
        "Thesenmarker Ischaemie",
        "Angle between the heart vector at its slowest inscription and the "
        "mean ST vector.",
        "angle(stall vector, mean ST vector) in degrees, 0-180.",
        "degrees",
        _stall_to_st_angle,
        "Thesis O2, revised: global stall/ST alignment may differ between "
        "independently verified acute coronary injury and competing conditions. "
        "Neither vector localizes the lesion or separates primary from secondary "
        "mechanisms. Refuted if adequately precise independent estimates exclude "
        "the prespecified clinically relevant association.",
        references=(REF_THESES, REF_GORGELS),
    ),
    IschemiaThesisSpec(
        "terminal_qrs_to_st_vector_angle_deg",
        "Thesenmarker Ischaemie",
        "Angle between the last 40 ms of the QRS and the mean ST vector.",
        "angle(mean vector of the terminal 40 ms of QRS, mean ST vector) in "
        "degrees, 0-180.",
        "degrees",
        _terminal_qrs_to_st_angle,
        "Thesis O3, revised: terminal QRS/ST geometry may add acute-context "
        "information; it does not establish immediate regional repolarization "
        "or identify infarct age. Refuted if adequate independent precision "
        "excludes the prespecified incremental value over classical comparators.",
        references=(REF_THESES, REF_HELSETH),
    ),
    IschemiaThesisSpec(
        "st_vector_beatwise_direction_dispersion",
        "Thesenmarker Ischaemie",
        "Circular dispersion of the ST vector direction across beats; 0 is "
        "the same direction every beat, 1 is complete cancellation.",
        "1 - |mean over beats of the unit ST vector|, each beat's ST window "
        "from its J point to its T onset against its own PR baseline.",
        "fraction",
        _st_direction_dispersion,
        "Thesis O4, revised: short-record ST direction dispersion is a "
        "morphology descriptor, not proof of a moving ischemic border. Rhythm, "
        "ectopy, baseline and absolute SNR can cause dispersion. Refuted if "
        "adequate independent precision excludes a prespecified association "
        "after those factors and ST magnitude are addressed.",
        references=(REF_THESES, REF_HELSETH),
    ),
)


class IschemiaThesisBiomarker(BiomarkerBase):
    def __init__(self, spec: IschemiaThesisSpec) -> None:
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
        notes = [f"ischemia_thesis_version={ISCHEMIA_THESIS_VERSION}"]
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


def ischemia_thesis_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in ISCHEMIA_THESIS_SPECS)


def register_ischemia_thesis_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in ISCHEMIA_THESIS_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), IschemiaThesisBiomarker):
                continue
            raise RuntimeError(
                f"Ischaemia thesis marker collides with an existing biomarker: {spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=ISCHEMIA_THESIS_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Exploratory post-outcome hypothesis with a refuting observation; "
                    "wording revised after technical and scientific review. "
                    + spec.thesis + " No diagnostic cutoff is claimed."
                ),
            )
        )
        BiomarkerRegistry.register(IschemiaThesisBiomarker(spec))
