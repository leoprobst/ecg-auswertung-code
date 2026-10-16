"""Six markers about the *order* in which the heart is activated.

The registry holds 2.016 markers, and its coverage of propagation geometry is
close to exhaustive: 425 kinematic markers over seventeen segments, 291
projection-plane markers at exactly 97 per plane, curvature and torsion at
every scale, stall and burst fractions, cusps, winding numbers, planarity,
anisotropy. What it almost entirely lacks is **time structure** -- when the
vector arrives where, rather than where it goes and how fast.

That is the half of propagation the literature has moved toward.

**Why this matters now.** Bacharova, Varma, Schocken and de Alencar (Heart
2026), in a framework the International Society of Electrocardiology and the
ISHNE both endorse, discard the Voltage Paradigm outright: most anatomic
hypertrophy has normal QRS amplitude and many high amplitudes have normal
mass. The determinants they name instead are **conduction velocity, fibrosis,
cell-to-cell coupling and thoracic conductivity**. Three of those four corrupt
amplitude specifically -- so a marker that survives amplitude scaling is worth
more than one that does not, and two of the six below are dimensionless by
construction for that reason.

The Maastricht review of vectorcardiography in resynchronisation (Dural,
Vernooy, van Stipdonk; J Cardiovasc Dev Dis 2025) states the mechanism the
first marker rests on: synchronous activation lets opposing wavefronts cancel,
dyssynchrony leaves forces **unopposed**, and the QRS area counts what is left
over. The area alone cannot separate "much activity, well cancelled" from
"little activity, poorly cancelled"; the ratio can.

Sales-Belles et al. (Sci Rep 2025, n = 176) showed the other half is readable
from a surface recording at all: a precordial *activation delay*, built from
band-limited envelopes across V1-V6, separated left bundle branch area pacing
from right ventricular pacing (-10 against +26 ms at narrow baseline QRS) --
neither a duration nor an amplitude, but a sequence.

**What is deliberately not here.** Conduction velocity in cm/s, anisotropic
ratio, deceleration zones and the intramural conduction corridors of the
TRAMPOLINE study (Heart Rhythm 2026, mid-myocardial 18 against 39 cm/s) are
catheter findings. They cannot be reconstructed from a body-surface recording,
and a marker carrying one of those names would claim something it does not
measure. Rashid et al. (JACC Clin Electrophysiol 2026) add the sharper
limitation: substrate that is invisible in sinus rhythm appears only under an
extrastimulus. A resting ten-second strip cannot provoke; that is a boundary
of this whole project, not a gap in this module.

Nothing is proposed for the two-dimensional projections. They are saturated --
97 markers per plane, three planes, every classical reading covered -- and an
addition there would displace a curated marker for nothing.

**Measured on 896 recordings of the training split before this text was
finished.** One of the six was named for a prediction the data refuted and is
now named for what it measures; the note on
``qrs_leftward_rightward_peak_order_ms`` carries that in full. The others:

  qrs_cancellation_fraction                 P50 0,302   R^2 0,86
  qrs_xz_cross_correlation_peak_lag_ms      P50 -14 ms  R^2 0,13
  qrs_yz_cross_correlation_peak_lag_ms      P50 -14 ms  R^2 0,17
  qrs_leftward_rightward_peak_order_ms      P50 -26 ms  R^2 0,18
  qrs_posterior_anterior_peak_order_ms      P50 +26 ms  R^2 0,16
  qrs_directional_arrival_spread_fraction   P50 0,237   R^2 0,39

R^2 is against the nearest existing neighbours, including both halves of the
cancellation fraction, the X-Y lag, the septal-to-basal interval and QRS
duration. The cancellation fraction is the least independent of the six at
0,86; it is kept because the threshold for dropping was set at 0,95 *before*
the number was seen, and moving a line after reading the result is how a
decision stops meaning anything. The apicobasal gradient was kept at 0,62 and
Tpe/QT dropped at 0,977, so 0,86 sits inside the range this project has
already accepted.
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


PROPAGATION_SEQUENCE_VERSION = "1.0.1"
PROPAGATION_SEQUENCE_SOURCE = "propagation_sequence_v1"

EPS = 1e-12

# Below this many samples the QRS cannot carry a sequence: the existing
# activation-dynamics markers use the same floor, and matching it keeps the
# whole-QRS cancellation comparable with its two half-window siblings.
MINIMUM_QRS_SAMPLES = 12

# The cross-correlation helpers require a longer window than the cancellation
# does, and this is the floor the existing X-Y lag already applies.
MINIMUM_CORRELATION_SAMPLES = 20

# A direction counts as visited only if the vector's projection onto it
# reaches this share of the QRS maximum magnitude. **This number is an
# interpretation, not a published threshold.** Without it, "the time of the
# largest projection onto rightward" returns a time even when the vector never
# points right -- a number with no referent. Fifteen per cent is well above
# the noise floor of an origin-corrected loop and well below any genuine
# septal or basal force.
MINIMUM_DIRECTION_SHARE = 0.15

# The six signed axis directions, in the project's Frank-compatible
# convention: +X left, +Y inferior, +Z posterior.
AXIS_DIRECTIONS: tuple[tuple[str, int, float], ...] = (
    ("leftward", 0, 1.0),
    ("rightward", 0, -1.0),
    ("inferior", 1, 1.0),
    ("superior", 1, -1.0),
    ("posterior", 2, 1.0),
    ("anterior", 2, -1.0),
)


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_propagation_sequence_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_propagation_sequence_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _qrs_points(ctx: FeatureContext) -> Optional[np.ndarray]:
    """The origin-corrected QRS trajectory, or nothing.

    The same source ``qrs_early_cancellation_fraction`` reads, so the
    whole-window value below sits on the identical footing as its halves.
    """

    def build() -> Optional[np.ndarray]:
        points = ctx.vcg_segment_origin_corrected("qrs")
        if points is None:
            return None
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            return None
        # Integrals and peak ordering require an intact physical time axis.
        if not np.all(np.isfinite(values)):
            return None
        return values if len(values) >= MINIMUM_QRS_SAMPLES else None

    return _memo(ctx, ("qrs_points",), build)


# ---------------------------------------------------- cancellation over all


def _qrs_cancellation_fraction(ctx: FeatureContext) -> FeatureValue:
    """Temporal opposition of the resultant QRS vectors over the complex.

    ``1 - ||integral r dt|| / integral ||r|| dt``. Zero means every force
    pointed the same way and nothing cancelled; one means the net vector
    integral vanished despite nonzero integrated vector magnitude. This is
    not cancellation among simultaneous local myocardial sources, which are
    not separately observed by this formula.

    **This is not obtainable from the two halves that already exist.** The
    norm of a sum is not the sum of norms, so a complex whose halves each
    cancel little can still cancel almost completely when they oppose each
    other. No universal direction of association with dyssynchrony follows.

    The quantity the Maastricht review calls unopposed force is the numerator
    alone, which the registry carries as ``qrs_area_vector_magnitude_mv_ms``.
    Dividing by the magnitude-time integral removes a constant scalar gain,
    not arbitrary anisotropic transfer, body habitus or lead-placement effects.
    """

    points = _qrs_points(ctx)
    if points is None:
        return _abstain("QRS trajectory unavailable or too short")
    fs = float(getattr(ctx, "fs", 0) or 0)
    if fs <= 0:
        return _abstain("no sampling rate")
    absolute_area = float(
        np.trapezoid(np.linalg.norm(points, axis=1), dx=1.0 / fs)
    )
    if not np.isfinite(absolute_area) or absolute_area <= EPS:
        return _abstain("degenerate QRS trajectory")
    net_area = np.trapezoid(points, dx=1.0 / fs, axis=0)
    value = 1.0 - float(np.linalg.norm(net_area)) / absolute_area
    if not np.isfinite(value):
        return _abstain("non-finite cancellation fraction")
    return FeatureValue(float(np.clip(value, 0.0, 1.0)))


# ------------------------------------------------- the two missing axis lags


def _axis_lag(ctx: FeatureContext, first: int, second: int) -> FeatureValue:
    """Lag in milliseconds at which two VCG axes correlate best.

    The helper, the window and the lag ceiling are the ones
    ``qrs_xy_cross_correlation_peak_lag_ms`` already uses, imported rather
    than reimplemented so all three axis pairs are computed identically. Only
    the pair of axes differs.
    """

    from .advanced_vcg_biomarkers import _cross_correlation_peak

    points = ctx.vcg_segment("qrs")
    if points is None or len(points) < MINIMUM_CORRELATION_SAMPLES:
        return _abstain("QRS window shorter than the correlation floor")
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        return _abstain("QRS window is not a trajectory")
    fs = float(getattr(ctx, "fs", 0) or 0)
    if fs <= 0:
        return _abstain("no sampling rate")
    result = _cross_correlation_peak(
        values[:, first],
        values[:, second],
        max_lag=max(1, min(20, len(values) // 4)),
    )
    if result is None:
        return _abstain("one axis is constant within the QRS")
    lag, _ = result
    return FeatureValue(float(lag) / fs * 1000.0)


def _qrs_xz_lag(ctx: FeatureContext) -> FeatureValue:
    """X against Z: the left-right sweep against the front-back sweep.

    The transverse pair, and the one with the clearest anatomy behind it --
    septal forces run rightward and anterior, free-wall forces leftward and
    posterior, so the offset between the two axes is a statement about that
    sequence.
    """

    return _axis_lag(ctx, 0, 2)


def _qrs_yz_lag(ctx: FeatureContext) -> FeatureValue:
    """Y against Z: the cranio-caudal sweep against the front-back sweep."""

    return _axis_lag(ctx, 1, 2)


# ------------------------------------------------------- directional arrival


def _direction_arrivals(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """When, within the QRS, each of the six axis directions peaks.

    In milliseconds from QRS onset, and only for directions the vector
    actually visits -- see ``MINIMUM_DIRECTION_SHARE`` for why that gate is
    not optional.

    **This is the dual of what the registry already holds, not a copy of it.**
    ``qrs_initial_20ms_force_leftward`` and its siblings fix the *time* and
    report the projection; ``septal_to_basal_interval_ms`` fixes two *windows*
    -- the first and last thirds -- and reports the gap between their
    magnitude peaks. Here the *direction* is fixed and the time is the answer,
    searched across the whole complex, so a late leftward peak is found
    wherever it falls rather than only inside a preselected window.
    """

    def build() -> Optional[dict[str, float]]:
        points = _qrs_points(ctx)
        if points is None:
            return None
        fs = float(getattr(ctx, "fs", 0) or 0)
        if fs <= 0:
            return None
        reference = float(np.max(np.linalg.norm(points, axis=1)))
        if not np.isfinite(reference) or reference <= EPS:
            return None
        floor = MINIMUM_DIRECTION_SHARE * reference
        arrivals: dict[str, float] = {}
        for name, axis, sign in AXIS_DIRECTIONS:
            projection = sign * points[:, axis]
            peak = float(np.max(projection))
            if not np.isfinite(peak) or peak < floor:
                continue
            arrivals[name] = float(np.argmax(projection)) / fs * 1000.0
        return arrivals

    return _memo(ctx, ("direction_arrivals",), build)


def _opposed_delay(ctx: FeatureContext, later: str, earlier: str) -> FeatureValue:
    arrivals = _direction_arrivals(ctx)
    if arrivals is None:
        return _abstain("QRS trajectory unavailable")
    missing = [name for name in (later, earlier) if name not in arrivals]
    if missing:
        return _abstain(f"vector never points {' or '.join(missing)}")
    return FeatureValue(float(arrivals[later] - arrivals[earlier]))


def _leftward_rightward_peak_order(ctx: FeatureContext) -> FeatureValue:
    """Time of the dominant leftward force minus that of the dominant rightward one.

    **This marker was named for a prediction that measurement refuted, and it
    is named for the measurement now.** The prediction was that the septum
    depolarises first pointing rightward and the free wall follows pointing
    leftward, so the difference should be positive. Over 313 recordings the
    median came out at **-26 ms with only 6,7 % positive** -- almost the exact
    opposite.

    The cause is not an error in the arithmetic but in the anatomy assumed. A
    normal complex has *two* rightward phases: the small initial septal vector
    and the terminal basal one. Taking the largest rightward excursion finds
    the second, not the first: measured across the QRS window, the dominant
    leftward force sits at **43 % of the QRS duration and the dominant
    rightward force at 67 %**, and the rightward peak falls in the second half
    in **93 %** of recordings.

    So what this measures is the order of the two *dominant* forces -- free
    wall against terminal basal -- and its ordinary value is negative. A
    positive value is the departure worth looking at, not the norm.
    """

    return _opposed_delay(ctx, "leftward", "rightward")


def _posterior_anterior_peak_order(ctx: FeatureContext) -> FeatureValue:
    """Time of the dominant posterior force minus that of the dominant anterior one.

    Here the prediction held. Early septal forces run anterior and later
    free-wall forces run posterior, so the difference should be positive:
    measured median **+26 ms, positive in 90,2 %** of 386 recordings.

    The asymmetry with the left-right pair above is itself informative. The
    anterior force has no late counterpart the way the rightward force does,
    so the largest anterior excursion really is the early septal one.
    """

    return _opposed_delay(ctx, "posterior", "anterior")


def _directional_arrival_spread(ctx: FeatureContext) -> FeatureValue:
    """How spread out in time the directional peaks are, as a share of QRS.

    The range between the earliest and latest of the visited directions,
    divided by the QRS duration. A synchronous complex reaches its directional
    extremes close together; a complex that activates one region long after
    another spreads them apart.

    Dimensionless on purpose. Dividing by the duration keeps it from becoming
    a second reading of QRS width, which the registry measures already and
    which the review above criticises as uninformative about mechanism.
    """

    arrivals = _direction_arrivals(ctx)
    if arrivals is None:
        return _abstain("QRS trajectory unavailable")
    if len(arrivals) < 2:
        return _abstain("fewer than two directions visited")
    points = _qrs_points(ctx)
    fs = float(getattr(ctx, "fs", 0) or 0)
    if points is None or fs <= 0:
        return _abstain("QRS trajectory unavailable")
    duration_ms = len(points) / fs * 1000.0
    if duration_ms <= EPS:
        return _abstain("degenerate QRS duration")
    times = np.asarray(list(arrivals.values()), dtype=float)
    spread = float(np.max(times) - np.min(times)) / duration_ms
    if not np.isfinite(spread):
        return _abstain("non-finite spread")
    return FeatureValue(float(np.clip(spread, 0.0, 1.0)))


# -------------------------------------------------------------------- specs


@dataclass(frozen=True)
class PropagationSequenceSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    evidence_note: str
    references: tuple[str, ...] = ()


REF_LVER = (
    "Bacharova L, Varma N, Schocken DD, Felicioni SP, de Alencar JN. Left "
    "ventricular electrical remodelling: a review. Heart 2026."
)
REF_CRT_VCG = (
    "Dural M, Eerenberg F, Smits KC, Nguyen UC, Vernooy K, van Stipdonk AMW. "
    "Vectorcardiography in CRT: what we know and what there is to learn. "
    "J Cardiovasc Dev Dis 2025;12(5):177."
)
REF_SEQUENCE = (
    "Sales-Belles C et al. ECG-based evaluation of ventricular synchrony in "
    "left bundle branch area pacing through characterization of the "
    "activation sequence. Sci Rep 2025;15:15420."
)


PROPAGATION_SEQUENCE_SPECS: tuple[PropagationSequenceSpec, ...] = (
    PropagationSequenceSpec(
        "qrs_cancellation_fraction",
        "Ausbreitungsfolge",
        "Share of the QRS that cancelled itself out, over the whole complex.",
        "1 - ||integral_QRS r dt|| / integral_QRS ||r|| dt.",
        "fraction",
        _qrs_cancellation_fraction,
        "Synchronous activation lets opposing wavefronts cancel and "
        "dyssynchrony leaves forces unopposed, which is what the QRS area "
        "counts. The ratio makes that dimensionless, and amplitude is the "
        "quantity thoracic conductivity and body habitus corrupt. Half-window "
        "versions exist; the whole is not a function of the halves.",
        references=(REF_CRT_VCG, REF_LVER),
    ),
    PropagationSequenceSpec(
        "qrs_xz_cross_correlation_peak_lag_ms",
        "Ausbreitungsfolge",
        "Lag at which the X and Z axes of the QRS correlate best.",
        "argmax_lag(cross_correlation(x(t), z(t))) / fs, in ms.",
        "ms",
        _qrs_xz_lag,
        "The transverse axis pair. Septal forces run rightward and anterior, "
        "free-wall forces leftward and posterior, so the offset between the "
        "two axes states that sequence. The X-Y pair was already registered; "
        "this completes the set.",
        references=(REF_SEQUENCE,),
    ),
    PropagationSequenceSpec(
        "qrs_yz_cross_correlation_peak_lag_ms",
        "Ausbreitungsfolge",
        "Lag at which the Y and Z axes of the QRS correlate best.",
        "argmax_lag(cross_correlation(y(t), z(t))) / fs, in ms.",
        "ms",
        _qrs_yz_lag,
        "The third axis pair, completing a set of which only X-Y existed.",
        references=(REF_SEQUENCE,),
    ),
    PropagationSequenceSpec(
        "qrs_leftward_rightward_peak_order_ms",
        "Ausbreitungsfolge",
        "Time of the dominant leftward force minus that of the dominant "
        "rightward force.",
        "t(max of +X projection) - t(max of -X projection), in ms, counting "
        "only directions whose projection reaches 15 % of the QRS maximum "
        "magnitude.",
        "ms",
        _leftward_rightward_peak_order,
        "The direction is fixed and the time is the answer, which is the dual "
        "of the registry's time-local force vectors and of the fixed-window "
        "septal-to-basal interval. Named for the measurement rather than for "
        "the anatomy first assumed: the dominant rightward force is the "
        "terminal basal one, not the initial septal one, so the ordinary "
        "value is negative (median -26 ms, positive in 6,7 % of 313 "
        "recordings).",
        references=(REF_SEQUENCE, REF_LVER),
    ),
    PropagationSequenceSpec(
        "qrs_posterior_anterior_peak_order_ms",
        "Ausbreitungsfolge",
        "Time of the dominant posterior force minus that of the dominant "
        "anterior force.",
        "t(max of +Z projection) - t(max of -Z projection), in ms, under the "
        "same 15 % visitation gate.",
        "ms",
        _posterior_anterior_peak_order,
        "The ordering statement in the other transverse axis, where the "
        "predicted direction did hold: early septal forces run anterior and "
        "later free-wall forces posterior, measured median +26 ms and "
        "positive in 90,2 % of 386 recordings.",
        references=(REF_SEQUENCE, REF_LVER),
    ),
    PropagationSequenceSpec(
        "qrs_directional_arrival_spread_fraction",
        "Ausbreitungsfolge",
        "Range of the six directional arrival times as a share of QRS "
        "duration.",
        "(max - min of the visited directional arrival times) / QRS duration.",
        "fraction",
        _directional_arrival_spread,
        "Dimensionless by construction, so that it does not become a second "
        "reading of QRS width -- a measure the same review criticises as "
        "uninformative about mechanism.",
        references=(REF_SEQUENCE, REF_LVER),
    ),
)


class PropagationSequenceBiomarker(BiomarkerBase):
    def __init__(self, spec: PropagationSequenceSpec) -> None:
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
        notes = [f"propagation_sequence_version={PROPAGATION_SEQUENCE_VERSION}"]
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


def propagation_sequence_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in PROPAGATION_SEQUENCE_SPECS)


def register_propagation_sequence_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in PROPAGATION_SEQUENCE_SPECS:
        if spec.name in existing:
            if isinstance(
                BiomarkerRegistry.get(spec.name), PropagationSequenceBiomarker
            ):
                continue
            raise RuntimeError(
                "Propagation marker collides with an existing biomarker: "
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
                source=PROPAGATION_SEQUENCE_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Activation-sequence hypothesis taken from the 2025-2026 "
                    "propagation literature and registered so its claim can "
                    "be tested rather than assumed. " + spec.evidence_note +
                    " No diagnostic cutoff is claimed."
                ),
            )
        )
        BiomarkerRegistry.register(PropagationSequenceBiomarker(spec))
