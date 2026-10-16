"""The time the loop discards, and four ways to read it back.

A vectorcardiographic loop is a parametrised space curve r(t). Its *shape* is
invariant under reparametrisation: two hearts can trace the same curve at
entirely different tempos and produce the same picture. What the picture
throws away is contained, completely, in one function -- the arc length
against time, s(t) -- and classical vectorcardiography restored it with time
dashes every two milliseconds: where the dashes crowd, the resultant VCG
vector changes slowly. This is not a measurement of tissue conduction speed.

**This information was formalised, studied systematically, and then
forgotten.** Spatial velocity electrocardiography -- the speed ||dr/dt||
plotted against time -- was a method in its own right from 1969 to 1996, with
normal values (Chorao de Aguiar 1979, n = 229), findings in infarction, bundle
branch block and hypertrophy (Sano, Mori, Sakamoto), and a randomised drug
trial. Since 2015 the literature holds **no** work on time-normalised or
dynamic loop descriptors; a revival in 2020 and 2024 (Bhattacharyya, Ghosal)
found spatial velocity reduced by 12-25 % in acute infarction while spatial
magnitude did not change -- time information the amplitude does not carry.

**The result this module rests on** is Fareh et al. (Cardiology 1996, Lyon):
seven landmarks A-G on the velocity curve and six intervals between them.
Quinidine lengthened the **B-C interval from 9,3 to 18,8 ms** (p < 0,05).
All six intervals lengthened; only BC was statistically significant. QRS
duration changed from 100.3 to 102.4 ms (reported p=0.05), and velocity values
did not significantly change. The automated landmarks below have not been
validated against the original manually identified A-G landmarks.

**The constraint every marker here satisfies.** The ECG keeps time as
intervals between fiducials -- onset, peak, end -- and a loop marker that
reduced to those would add nothing. All four below are built on the
three-dimensional path: arc length, spatial speed, three-dimensional change
of direction. No fiducial enters any of them except as the window's edges.
Redundancy is measured rather than asserted, in two separate numbers per
marker: against a pure-ECG time set, and against the nearest loop
neighbours.

**What is deliberately not here.** A dwell time at the origin after QRS onset
is anchored on an ECG fiducial and is a relative of the delta-wave flag. A
Gini coefficient of speed is permutation-invariant -- it discards the order,
which is the very thing this module is about. T-loop versions are covered by
the efferent/afferent markers that exist.

**Measured on 900 recordings before the module was finished; see the note
on each marker and the tests for the numbers that decided its fate.**
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
from ..delineation.base import RecordDelineation


LOOP_PARAMETRISATION_VERSION = "1.2.0"
LOOP_PARAMETRISATION_SOURCE = "loop_parametrisation_v1"

EPS = 1e-12

# The same floor the activation-dynamics markers apply.
MINIMUM_QRS_SAMPLES = 12

# Legacy reference value at 500 Hz; calculation uses the shared 2-ms sigma
# from core.temporal_sampling, also used by the velocity view.
SPEED_SMOOTHING_SIGMA = 1.0

# A velocity peak counts only if it is a genuine burst, not a ripple. **Both
# numbers are interpretations, not published thresholds.** A peak must reach
# a quarter of the maximum spatial speed, and must rise above the higher of
# its two contour bases by a tenth of that maximum. Without the first
# rule every noise maximum in a slow tail counts; without the second a
# shoulder on a single burst counts twice.
MINIMUM_PEAK_SHARE = 0.25
MINIMUM_PROMINENCE_SHARE = 0.10

# Direction is undefined for a velocity vector this small, so turning is
# accumulated only across steps that actually move.
MINIMUM_STEP_SPEED_SHARE = 0.02


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_loop_parametrisation_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_loop_parametrisation_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _qrs_points(ctx: FeatureContext) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        points = ctx.vcg_segment_origin_corrected("qrs")
        if points is None:
            return None
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            return None
        # Removing rows would bridge gaps and compress the physical time axis.
        # No gap reconstruction is defined for these temporal markers.
        if not np.all(np.isfinite(values)):
            return None
        return values if len(values) >= MINIMUM_QRS_SAMPLES else None

    return _memo(ctx, ("qrs_points",), build)


def _speed(ctx: FeatureContext) -> Optional[np.ndarray]:
    """Spatial speed ||dr/dt|| along the smoothed QRS trajectory, in mV/s.

    One sample shorter than the trajectory, as a difference must be.
    """

    def build() -> Optional[np.ndarray]:
        from .discovery_v3_biomarkers import _smooth

        points = _qrs_points(ctx)
        if points is None:
            return None
        fs = float(getattr(ctx, "fs", 0) or 0)
        if not np.isfinite(fs) or fs <= 0:
            return None
        from ..temporal_sampling import speed_smoothing_sigma_samples

        smoothed = _smooth(points, speed_smoothing_sigma_samples(fs))
        speed = np.linalg.norm(np.diff(smoothed, axis=0), axis=1) * fs
        if not np.all(np.isfinite(speed)) or float(np.max(speed)) <= EPS:
            return None
        return speed

    return _memo(ctx, ("speed",), build)


# ----------------------------------------------------- arc length at half time


def _arc_length_fraction_at_half_time(ctx: FeatureContext) -> FeatureValue:
    """s(T/2) / s(T): how much of the loop is drawn by the middle of the QRS.

    One half means equal arc length in the two halves, not necessarily
    uniform speed within either half. Above one half, more arc length is
    drawn early; this is not by itself proof of terminal conduction delay. It
    is the one number that makes the time dashes readable without the
    picture.

    **The honest reservation, stated before measurement.** The registry has
    ``activation_energy_q50_fraction`` -- the same question asked of the
    cumulative *squared* speed -- and ``activation_energy_timing_skew``, its
    displacement from the midpoint. Energy weights fast passages
    quadratically; arc length weights them linearly. Whether that difference
    carries anything was to be decided by R^2 against exactly those two, with
    the line drawn at 0,95 before the number was seen.
    """

    points = _qrs_points(ctx)
    if points is None:
        return _abstain("QRS trajectory unavailable or too short")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(steps))
    if not np.isfinite(total) or total <= EPS:
        return _abstain("degenerate QRS trajectory")
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    half = 0.5 * (len(points) - 1)
    value = float(np.interp(half, np.arange(len(points), dtype=float), cumulative)) / total
    if not np.isfinite(value):
        return _abstain("non-finite arc length fraction")
    return FeatureValue(float(np.clip(value, 0.0, 1.0)))


# ------------------------------------------------ spatial velocity landmarks


def landmarks_from_speed(speed: np.ndarray) -> Optional[dict[str, list[int]]]:
    """Prominent peaks of a spatial velocity curve and the troughs after them.

    Fareh's seven points A-G were placed by hand on a 250 mm/s strip. This is
    the automated reading: local maxima of ||dr/dt|| that pass the two gates
    above, and for each the lowest sample before the next accepted peak (or
    before the window's end, for the last one). Pure function of the speed
    series so that the velocity view can draw exactly what the markers read.
    """

    speed = np.asarray(speed, dtype=float)
    if speed.ndim != 1 or len(speed) < 5 or not np.all(np.isfinite(speed)):
        return None
    maximum = float(np.max(speed))
    if maximum <= EPS:
        return None
    floor = MINIMUM_PEAK_SHARE * maximum
    prominence = MINIMUM_PROMINENCE_SHARE * maximum
    interior = speed[1:-1]
    candidates = np.flatnonzero(
        (interior > speed[:-2]) & (interior >= speed[2:]) & (interior >= floor)
    ) + 1
    # Stop contour searches at higher peaks: global minima promote shoulders.
    from scipy.signal import peak_prominences

    prominences = peak_prominences(speed, candidates)[0]
    peaks = candidates[prominences >= prominence].astype(int).tolist()
    if not peaks:
        return None
    troughs: list[int] = []
    for position, peak in enumerate(peaks):
        end = peaks[position + 1] if position + 1 < len(peaks) else len(speed)
        segment = speed[peak:end]
        troughs.append(peak + int(np.argmin(segment)))
    return {"peaks": peaks, "troughs": troughs}


def _velocity_landmarks(ctx: FeatureContext) -> Optional[dict[str, list[int]]]:
    """The landmark rule applied to this recording's smoothed QRS speed."""

    def build() -> Optional[dict[str, list[int]]]:
        speed = _speed(ctx)
        if speed is None:
            return None
        return landmarks_from_speed(speed)

    return _memo(ctx, ("velocity_landmarks",), build)


def _first_to_last_peak(ctx: FeatureContext) -> FeatureValue:
    """Time between the first and the last prominent velocity peak.

    Fareh's span B-F: over how much of the complex the propagation bursts
    are spread. No fiducial enters; both ends are features of the
    three-dimensional path's speed.
    """

    landmarks = _velocity_landmarks(ctx)
    if landmarks is None:
        return _abstain("no prominent velocity peak")
    peaks = landmarks["peaks"]
    if len(peaks) < 2:
        return _abstain("only one prominent velocity peak")
    fs = float(getattr(ctx, "fs", 0) or 0)
    if fs <= 0:
        return _abstain("no sampling rate")
    return FeatureValue(float(peaks[-1] - peaks[0]) / fs * 1000.0)


def _initial_burst_duration(ctx: FeatureContext) -> FeatureValue:
    """From the first prominent velocity peak to the trough that follows it.

    Inspired by Fareh's manual B-C interval, not validated as its equivalent
    or as a specifically septal phase. All six intervals lengthened under
    quinidine; only BC was significant. Similar group medians cannot validate
    this automated landmark rule.
    """

    landmarks = _velocity_landmarks(ctx)
    if landmarks is None:
        return _abstain("no prominent velocity peak")
    fs = float(getattr(ctx, "fs", 0) or 0)
    if fs <= 0:
        return _abstain("no sampling rate")
    first_peak = landmarks["peaks"][0]
    first_trough = landmarks["troughs"][0]
    if first_trough <= first_peak:
        return _abstain("no trough after the first velocity peak")
    return FeatureValue(float(first_trough - first_peak) / fs * 1000.0)


# ------------------------------------------------- turning fraction at half time


def _turning_fraction_at_half_time(ctx: FeatureContext) -> FeatureValue:
    """Share of the loop's total turning completed by the middle of the QRS.

    The direction-domain twin of the arc-length fraction, and not the same
    quantity: a loop can travel far without turning, or turn sharply while
    travelling little. Turning is the angle between successive velocity
    vectors, accumulated across steps that actually move.

    The registry has the total (``qrs_total_turning_angle_deg``) and the
    distribution (``rotation_entropy``); it has nothing about *when* the
    turning happens. The direction of a vector is not a quantity any single
    lead provides, which is what keeps this outside the ECG.
    """

    points = _qrs_points(ctx)
    if points is None:
        return _abstain("QRS trajectory unavailable or too short")
    velocity = np.diff(points, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    if float(np.max(speed)) <= EPS:
        return _abstain("degenerate QRS trajectory")
    moving = speed >= MINIMUM_STEP_SPEED_SHARE * float(np.max(speed))
    unit = np.zeros_like(velocity)
    unit[moving] = velocity[moving] / speed[moving, None]
    turning = np.zeros(len(velocity) - 1, dtype=float)
    both = moving[:-1] & moving[1:]
    cosine = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
    turning[both] = np.arccos(cosine[both])
    total = float(np.sum(turning))
    if not np.isfinite(total) or total <= EPS:
        return _abstain("no measurable turning")
    cumulative = np.concatenate([[0.0], np.cumsum(turning)])
    # ``turning[i]`` is the angle at trajectory sample i + 1; the cumulative
    # array therefore runs over samples 1 .. n - 1 of the trajectory.
    positions = np.arange(1, len(points), dtype=float)
    half = 0.5 * (len(points) - 1)
    value = float(np.interp(half, positions, cumulative)) / total
    if not np.isfinite(value):
        return _abstain("non-finite turning fraction")
    return FeatureValue(float(np.clip(value, 0.0, 1.0)))


# -------------------------------------------------------------------- specs


@dataclass(frozen=True)
class LoopParametrisationSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    evidence_note: str
    references: tuple[str, ...] = ()


REF_FAREH = (
    "Fareh S et al. Heterogeneous effect of quinidine on the ventricular "
    "depolarization process assessed by the spatial velocity "
    "electrocardiogram of the QRS complex. Cardiology 1996;87(2):129-133."
)
REF_AGUIAR = (
    "Chorao de Aguiar AJ et al. The normal values of spatial velocity in "
    "the second part of QRS. J Electrocardiol 1979;12(4):381-386."
)
REF_GHOSAL = (
    "Ghosal T et al. Exploring cardiac vector propagation in acute "
    "myocardial infarction: a spatial velocity perspective. Future Cardiol "
    "2024;20(15-16):827-836."
)


LOOP_PARAMETRISATION_SPECS: tuple[LoopParametrisationSpec, ...] = (
    LoopParametrisationSpec(
        "qrs_arc_length_fraction_at_half_time",
        "Loop-Parametrisierung",
        "Share of the QRS loop's arc length drawn by the middle of the QRS "
        "duration.",
        "s(T/2) / s(T), with s the cumulative arc length of the "
        "origin-corrected QRS trajectory.",
        "fraction",
        _arc_length_fraction_at_half_time,
        "The reparametrisation the loop picture discards, read at one point. "
        "One half means equal arc length in the two halves, not uniform speed. "
        "Above it more arc length is drawn early. Kept only if not reconstructible from the "
        "energy-quantile markers that already exist.",
        references=(REF_GHOSAL, REF_AGUIAR),
    ),
    LoopParametrisationSpec(
        "qrs_spatial_velocity_first_to_last_peak_ms",
        "Loop-Parametrisierung",
        "Time between the first and the last prominent peak of the spatial "
        "velocity within the QRS.",
        "t(last prominent peak of ||dr/dt||) - t(first prominent peak), in "
        "ms; a peak counts at >= 25 % of the maximum speed and >= 10 % "
        "prominence over its neighbouring troughs.",
        "ms",
        _first_to_last_peak,
        "The span of the propagation bursts, Fareh's B-F. Both ends are "
        "features of the three-dimensional path's speed; no fiducial enters.",
        references=(REF_FAREH, REF_AGUIAR),
    ),
    LoopParametrisationSpec(
        "qrs_spatial_velocity_initial_burst_ms",
        "Loop-Parametrisierung",
        "Duration of the first propagation burst: first prominent velocity "
        "peak to the trough that follows it.",
        "t(first trough after the first prominent peak of ||dr/dt||) - "
        "t(first prominent peak), in ms.",
        "ms",
        _initial_burst_duration,
        "Fareh's B-C interval, which quinidine lengthened from 9,3 to 18,8 ms "
        "while neither QRS duration nor any velocity value changed. The "
        "placebo value of 9,3 ms in healthy volunteers is the plausibility "
        "check for the landmark rule.",
        references=(REF_FAREH,),
    ),
    LoopParametrisationSpec(
        "qrs_turning_fraction_at_half_time",
        "Loop-Parametrisierung",
        "Share of the loop's total turning completed by the middle of the "
        "QRS duration.",
        "sum of angles between successive velocity vectors up to T/2, "
        "divided by the same sum over the whole QRS.",
        "fraction",
        _turning_fraction_at_half_time,
        "The direction-domain twin of the arc-length fraction: a loop can "
        "travel far without turning or turn sharply while travelling little. "
        "The registry holds the total turning and its distribution, nothing "
        "about when it happens.",
        references=(REF_AGUIAR, REF_GHOSAL),
    ),
)


class LoopParametrisationBiomarker(BiomarkerBase):
    def __init__(self, spec: LoopParametrisationSpec) -> None:
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
        notes = [f"loop_parametrisation_version={LOOP_PARAMETRISATION_VERSION}"]
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


def loop_parametrisation_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in LOOP_PARAMETRISATION_SPECS)


def register_loop_parametrisation_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in LOOP_PARAMETRISATION_SPECS:
        if spec.name in existing:
            if isinstance(
                BiomarkerRegistry.get(spec.name), LoopParametrisationBiomarker
            ):
                continue
            raise RuntimeError(
                "Parametrisation marker collides with an existing biomarker: "
                f"{spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=LOOP_PARAMETRISATION_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Reads the time structure the loop picture discards, "
                    "following the spatial-velocity electrocardiography of "
                    "1969-1996 and its 2020-2024 revival. Built on the "
                    "three-dimensional path so that it is not an ECG interval "
                    "under another name. " + spec.evidence_note +
                    " No diagnostic cutoff is claimed."
                ),
            )
        )
        BiomarkerRegistry.register(LoopParametrisationBiomarker(spec))
