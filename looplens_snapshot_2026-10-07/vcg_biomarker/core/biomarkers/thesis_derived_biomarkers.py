"""Five markers derived from the propagation theses, not from the literature.

The theses are written out in ``THESEN_ERREGUNGSAUSBREITUNG_2026-09-12.md``
at the project root. This module builds the markers those theses call for
and the registry lacked; each calculator's docstring names the thesis it
serves and the observation that would refute it. Nothing here is taken from
a published marker definition -- that is the point, and also the reason
every one is registered as a post-outcome hypothesis with no cutoff.

The measured vector is a weighted global resultant. Simplified uniform
double-layer models motivate hypotheses but do not uniquely identify local
front geometry, tissue conduction velocity or an obstacle from this signal.

``qrs_stall_direction_leftward / _inferior / _posterior``
    Theses 1.1, 2.1 and 5.1: the direction of the heart vector at its
    slowest interior inscription is a global direction, not a localization
    of impaired conduction. Three direction cosines, so no angular seam. Hypothesized to spread
    more across recordings with regional wall-motion abnormality than across
    explicitly normal ones, and to turn rightward-anterior under right-heart
    load.

``qrs_arc_length_fraction_beatwise_variability``
    Theses 6.2 and 12.1: a stable substrate stalls at the same place every
    beat; a dynamic one -- ischaemia, inflammation, an electrolyte in flux --
    does not. The beat-to-beat scatter of the loop's time structure is the
    morphology variability, not a direct measurement of reentry substrate.
    Predicted higher before cardiac arrest and thirty-day death, independent
    of the mean it scatters around.

``t_arc_length_fraction_at_half_time``
    Thesis 9.1: repolarisation is a sequence, not a wave, ordered by local
    action-potential duration. A homogeneous prolongation stretches that
    sequence evenly; a heterogeneous one -- hypokalaemia striking the M cells
    hardest -- loads the late part. Predicted to separate drug-induced QT
    prolongation from hypokalaemia where the QTc itself cannot.

Constant scalar gain invariance must not be confused with invariance to
general volume-conductor changes. Pericardial effusion and COPD are
robustness strata, not guaranteed negative controls of the marker family.

**Tested on 2.500 recordings of the training split the day they were written.
The full scorecard, refutations included, is section VI of the theses paper.**
What concerns this module directly:

* The stall direction first returned a value on **11,6 %** of recordings,
  because the slowest interior moment is ordinarily the loop's approach to
  the origin, where direction is noise. The search is now restricted to
  samples whose magnitude clears the gate; availability is **97,7 %**.
* Across 396 unselected recordings the mean stall direction points
  **posterior** (unit-vector resultant +Z 0,37, X and Y near zero). That is
  the posterobasal region Durrer described as the last to activate. The
  correspondence is descriptive and does not establish that the global
  resultant localizes a wavefront or excludes measurement noise.
* The beat-to-beat scatter is available on 100 % of recordings but showed
  AUC 0,52 against thirty-day mortality and 0,51 after adjusting for the
  mean -- thesis 12.1 is **not supported** at this sample size.
* The T-loop arc fraction separated hypokalaemia from QT-prolonging drug
  exposure (+0,026 against -0,000) **but with the sign reversed** from the
  prediction. Hypokalaemia raises the fraction: the delayed M-cell
  repolarisation is flat, and a flat wave moves the vector little, so the
  first half gains. That reading is post hoc and is recorded as such.
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
from .catalog_biomarkers import FeatureContext, FeatureValue, _context
from ..delineation.base import RecordDelineation, fiducial_confidence_threshold


THESIS_DERIVED_VERSION = "1.2.0"
THESIS_DERIVED_SOURCE = "thesis_derived_v1"

EPS = 1e-12

MINIMUM_SEGMENT_SAMPLES = 12

# The stall is sought in the interior of the QRS. In the first and last tenth
# the vector is near the origin and its direction is noise, and the speed
# minimum there is the onset or the end, not a stall.
STALL_INTERIOR_MARGIN = 0.10

# A stall counts only if the vector at that moment is large enough to have a
# direction. **An interpretation, not a published threshold.** Below a fifth
# of the QRS maximum magnitude the unit vector rotates with the noise floor.
MINIMUM_STALL_MAGNITUDE_SHARE = 0.20

# Legacy reference sigma at 500 Hz; calculation uses shared physical 2 ms.
SPEED_SMOOTHING_SIGMA = 1.0

# A beat-to-beat scatter over fewer than three beats is the separation of a
# pair expressed as a spread.
MINIMUM_BEATS = 3


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_thesis_derived_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_thesis_derived_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _segment_points(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        points = ctx.vcg_segment_origin_corrected(segment)
        if points is None:
            return None
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            return None
        # Preserve sample positions: a missing instant is not a shorter beat.
        if not np.all(np.isfinite(values)):
            return None
        return values if len(values) >= MINIMUM_SEGMENT_SAMPLES else None

    return _memo(ctx, ("segment", segment), build)


def _arc_fraction_at_half(points: np.ndarray) -> Optional[float]:
    """s(T/2)/s(T) for one trajectory; the arithmetic shared by two markers."""

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(steps))
    if not np.isfinite(total) or total <= EPS:
        return None
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    half = 0.5 * (len(points) - 1)
    value = float(np.interp(half, np.arange(len(points), dtype=float), cumulative)) / total
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else None


# --------------------------------------------------------- stall direction


def _stall_vector(ctx: FeatureContext) -> Optional[np.ndarray]:
    """Unit heart vector at the slowest interior moment of the QRS.

    Theses 1.1, 2.1 and 5.1. The speed is the same smoothed spatial speed the
    stall-fraction markers read. The minimum is sought only in the interior
    tenths, and the vector there must be large enough to carry a direction.
    """

    def build() -> Optional[np.ndarray]:
        from .discovery_v3_biomarkers import _smooth

        points = _segment_points(ctx, "qrs")
        if points is None:
            return None
        from ..temporal_sampling import speed_smoothing_sigma_samples

        fs = float(getattr(ctx, "fs", 0) or 0)
        if not np.isfinite(fs) or fs <= 0:
            return None
        smoothed = _smooth(points, speed_smoothing_sigma_samples(fs))
        speed = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
        n = len(speed)
        margin = max(1, int(round(STALL_INTERIOR_MARGIN * n)))
        if n - 2 * margin < 3:
            return None
        # The first version took the interior speed minimum and then asked
        # whether the vector there was large enough to have a direction. It
        # almost never was: measured on 2.500 recordings the marker returned a
        # value on 11,6 %, because the slowest interior moment is ordinarily
        # the loop's approach to the origin, where direction is noise. The
        # question the thesis asks is where the vector stalls *while it is
        # large* -- so the search is now restricted to the samples where the
        # magnitude clears the gate, and the minimum is taken among those.
        magnitudes = np.linalg.norm(points, axis=1)
        reference = float(np.max(magnitudes))
        if reference <= EPS:
            return None
        # ``speed[i]`` sits between trajectory samples i and i + 1; the vector
        # at a step is read from the sample the step arrives at.
        arrival = magnitudes[1:]
        eligible = np.zeros(n, dtype=bool)
        eligible[margin : n - margin] = True
        eligible &= arrival >= MINIMUM_STALL_MAGNITUDE_SHARE * reference
        if int(eligible.sum()) < 3:
            return None
        candidate = np.where(eligible, speed, np.inf)
        stall = int(np.argmin(candidate))
        vector = points[stall + 1]
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= EPS:
            return None
        return vector / magnitude

    return _memo(ctx, ("stall_vector",), build)


def _stall_component(axis: int) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        vector = _stall_vector(ctx)
        if vector is None:
            return _abstain("no interior stall with a defined direction")
        value = float(vector[axis])
        if not np.isfinite(value):
            return _abstain("non-finite stall direction")
        return FeatureValue(float(np.clip(value, -1.0, 1.0)))

    return calculate


# ----------------------------------------- beat-to-beat time-structure scatter


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


def _beatwise_arc_fractions(ctx: FeatureContext) -> list[float]:
    """s(T/2)/s(T) of every beat's own QRS, from the record's beats.

    The consensus complex has one QRS; the question here is how much the time
    structure moves between beats, so each beat's own window is read.
    """

    def build() -> list[float]:
        record = ctx._record_vcg_xyz
        if record is None:
            return []
        beats = getattr(ctx.delineation, "beats", None) or ()
        limit = len(record) - 1
        values: list[float] = []
        for beat in beats:
            start = _beat_sample(beat, "r_onset", "qrs")
            end = _beat_sample(beat, "r_offset", "qrs")
            if start is None or end is None:
                continue
            if not 0 <= start < end <= limit:
                continue
            if end - start < MINIMUM_SEGMENT_SAMPLES:
                continue
            window = np.asarray(record[start : end + 1], dtype=float)
            if window.ndim != 2 or window.shape[1] != 3 or not np.all(np.isfinite(window)):
                continue
            fraction = _arc_fraction_at_half(window - window[0])
            if fraction is not None:
                values.append(fraction)
        return values

    return _memo(ctx, ("beatwise_arc_fractions",), build)


def _arc_fraction_beatwise_variability(ctx: FeatureContext) -> FeatureValue:
    """Standard deviation of the arc-length half-time fraction across beats.

    Temporal morphology scatter, not direct measurement of reentry substrate.
    Rhythm, ectopy, RR history, delineation and noise can all contribute.
    Independence from the mean and clinical specificity must be tested.
    """

    values = _beatwise_arc_fractions(ctx)
    if len(values) < MINIMUM_BEATS:
        return _abstain("fewer than three beats with a measurable QRS")
    array = np.asarray(values, dtype=float)
    scatter = float(np.std(array, ddof=1))
    if not np.isfinite(scatter):
        return _abstain("non-finite scatter")
    return FeatureValue(
        scatter,
        notes=(f"contributing_beats={len(values)}; "
               "beat_class_homogeneity=not_verified; absolute_snr_gate=not_validated; "
               "interpretation=descriptive_morphology_variability_not_reentry_substrate"),
    )


# -------------------------------------------------- T-loop arc length at half


def _t_arc_length_fraction_at_half_time(ctx: FeatureContext) -> FeatureValue:
    """s(T/2)/s(T) for the T loop.

    Thesis 9.1. Repolarisation is a sequence ordered by local action
    potential duration; a homogeneous prolongation stretches it evenly and
    leaves this fraction where it was, a heterogeneous one loads the late
    part and lowers it. The registry's ``t_terminal_path_fraction`` reads the
    final fifth of the time; this reads the midpoint, and is measured
    against it.
    """

    points = _segment_points(ctx, "t")
    if points is None:
        return _abstain("T trajectory unavailable or too short")
    value = _arc_fraction_at_half(points)
    if value is None:
        return _abstain("degenerate T trajectory")
    return FeatureValue(value)


# -------------------------------------------------------------------- specs


@dataclass(frozen=True)
class ThesisDerivedSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    thesis: str
    references: tuple[str, ...] = ()


REF_THESES = (
    "THESEN_ERREGUNGSAUSBREITUNG_2026-09-12.md, project root: own theses on "
    "excitation propagation, each with its refuting observation."
)
REF_PLONSEY = (
    "Plonsey R, Barr RC. Bioelectricity: A Quantitative Approach. The "
    "uniform double-layer model: the heart vector as the vector area of the "
    "activation front."
)

_STALL_AXES = (
    ("leftward", 0, "+X"),
    ("inferior", 1, "+Y"),
    ("posterior", 2, "+Z"),
)

THESIS_DERIVED_SPECS: tuple[ThesisDerivedSpec, ...] = tuple(
    ThesisDerivedSpec(
        f"qrs_stall_direction_{label}",
        "Thesenmarker Ausbreitung",
        f"{axis} component of the unit heart vector at the slowest interior "
        "moment of the QRS.",
        f"r(t*)[{index}] / ||r(t*)||, t* = argmin of the smoothed spatial "
        "speed over the interior 80 % of the QRS, requiring ||r(t*)|| >= "
        "20 % of the QRS maximum.",
        "cosine",
        _stall_component(index),
        "Theses 1.1, 2.1, 5.1, revised: exploratory association between global stall direction and regional "
        "wall-motion abnormalities. Direction does not directly localize "
        "slow myocardial conduction; account for anatomy, transformation "
        "and signal quality with independent imaging references. Refuted if "
        "adequately precise independent estimates exclude the prespecified "
        "minimum association; a nonsignificant p-value alone is insufficient.",
        references=(REF_THESES, REF_PLONSEY),
    )
    for label, index, axis in _STALL_AXES
) + (
    ThesisDerivedSpec(
        "qrs_arc_length_fraction_beatwise_variability",
        "Thesenmarker Ausbreitung",
        "Standard deviation across beats of the QRS arc-length fraction at "
        "half time.",
        "std over beats of s_beat(T/2)/s_beat(T), each beat's own QRS window "
        "from the record.",
        "fraction",
        _arc_fraction_beatwise_variability,
        "Theses 6.2, 12.1, revised: exploratory temporal morphology variability. It may reflect rhythm, "
        "ectopy, RR history or measurement noise as well as pathology. "
        "An adjusted mortality association is not a demonstration of reentry. "
        "Refuted if an adequately powered independent study excludes the "
        "prespecified prognostic effect; this does not refute local reentry physiology.",
        references=(REF_THESES,),
    ),
    ThesisDerivedSpec(
        "t_arc_length_fraction_at_half_time",
        "Thesenmarker Ausbreitung",
        "Share of the T loop's arc length drawn by the middle of the T "
        "duration.",
        "s(T/2) / s(T) over the origin-corrected T trajectory.",
        "fraction",
        _t_arc_length_fraction_at_half_time,
        "Thesis 9.1: a homogeneous QT prolongation leaves this where it was, "
        "a heterogeneous one lowers it. Refuted if hypokalaemia and "
        "QT-prolonging drug exposure show the same value.",
        references=(REF_THESES,),
    ),
)


class ThesisDerivedBiomarker(BiomarkerBase):
    def __init__(self, spec: ThesisDerivedSpec) -> None:
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
        notes = [f"thesis_derived_version={THESIS_DERIVED_VERSION}"]
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


def thesis_derived_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in THESIS_DERIVED_SPECS)


def register_thesis_derived_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in THESIS_DERIVED_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), ThesisDerivedBiomarker):
                continue
            raise RuntimeError(
                f"Thesis marker collides with an existing biomarker: {spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=THESIS_DERIVED_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Derived from an own thesis on excitation propagation, "
                    "written down with its refuting observation before the "
                    "marker was built. " + spec.thesis +
                    " No diagnostic cutoff is claimed."
                ),
            )
        )
        BiomarkerRegistry.register(ThesisDerivedBiomarker(spec))
