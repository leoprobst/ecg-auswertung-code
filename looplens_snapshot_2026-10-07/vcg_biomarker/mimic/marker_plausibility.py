"""What a marker value cannot be, declared from the unit it is measured in.

The screen is rank based, so an impossible value survives it unnoticed:
``classical_dominant_r_v1`` reaches **1.260.002.535,94** in the current store,
because the R/S ratio in V1 divides by an S wave that can be arbitrarily small,
and it is still the second-best marker for right bundle branch block at
AUC 0,889. Spearman and the Mann-Whitney AUC do not care. Everything that is
not rank based does: linear and logistic adjustment, the gap models, the
combination search and every z-score are shifted by one such row.

**This declares impossibility, not abnormality.** A fraction outside [0, 1] is
not a rare patient, it is an arithmetic accident; a negative count is not a
finding. Nothing here says what a *healthy* value is -- that is the reference
range stage's job, and conflating the two would let a plausibility check quietly
become a normal range with no cohort behind it.

The bounds come from the unit the catalog already declares, so they cannot
drift away from the marker definitions: a marker in ``fraction`` is bounded by
construction whoever wrote it. Units that carry no bound -- ``mV/s^3``,
``1/mV``, ``dimensionless`` -- get none, and are counted as unbounded rather
than pretended to be checked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


MARKER_PLAUSIBILITY_VERSION = "1.5.0"

# The cap applied to amplitude ratios at the point they are formed. R/S in V1
# is the reason: with an S wave near zero the quotient is limited only by the
# floating-point guard, and 1,26e9 is what that produces. 50 is far outside
# anything reported clinically -- an R/S of 10 is already a dominant R wave --
# so the cap can only bind on values that were never measurements.
RATIO_LIMIT = 50.0

# An ECG lead deflection of 100 mV is not a large QRS, it is a broken record;
# the largest physiological deflection is a few millivolts.
MAXIMUM_ABS_MILLIVOLT = 100.0

# A ten-second strip cannot contain an interval longer than itself.
MAXIMUM_INTERVAL_MS = 10_000.0

# Sinus tachycardia tops out near 220; ventricular flutter is reported to about
# 400. 500 leaves room above anything real.
MAXIMUM_RATE_BPM = 500.0


@dataclass(frozen=True)
class PlausibilityRange:
    """The values a marker may take at all, and why."""

    lower: Optional[float]
    upper: Optional[float]
    reason: str
    numerical_tolerance: float = 0.0

    def violations(self, values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        bad = np.zeros(values.shape, dtype=bool)
        if self.lower is not None:
            bad |= finite & (values < self.lower - self.numerical_tolerance)
        if self.upper is not None:
            bad |= finite & (values > self.upper + self.numerical_tolerance)
        return bad


# Keyed by the unit string the biomarker catalog declares.
_UNIT_RANGES: Mapping[str, PlausibilityRange] = {
    "fraction": PlausibilityRange(
        0.0, 1.0, "a fraction lies in [0, 1] by construction"
    ),
    "binary": PlausibilityRange(0.0, 1.0, "a binary flag is 0 or 1"),
    "probability": PlausibilityRange(0.0, 1.0, "a probability lies in [0, 1]"),
    "count": PlausibilityRange(0.0, None, "a count cannot be negative"),
    "samples": PlausibilityRange(0.0, None, "a sample index cannot be negative"),
    "sample": PlausibilityRange(0.0, None, "a sample index cannot be negative"),
    "count/s": PlausibilityRange(0.0, None, "an event rate cannot be negative"),
    "ms": PlausibilityRange(
        0.0,
        MAXIMUM_INTERVAL_MS,
        "an interval measured inside a ten-second strip cannot exceed it",
    ),
    "ms^2": PlausibilityRange(
        0.0, MAXIMUM_INTERVAL_MS**2, "a squared interval is non-negative"
    ),
    "degrees": PlausibilityRange(
        -360.0, 360.0, "an angle in degrees is reported within one turn"
    ),
    "steradian": PlausibilityRange(
        0.0, 4.0 * math.pi, "a solid angle cannot exceed the full sphere"
    ),
    "bpm": PlausibilityRange(
        0.0, MAXIMUM_RATE_BPM, "a heart rate above 500/min is not a rate"
    ),
    "ratio": PlausibilityRange(
        -RATIO_LIMIT,
        RATIO_LIMIT,
        "amplitude ratios are capped where they are formed",
    ),
    "mV": PlausibilityRange(
        -MAXIMUM_ABS_MILLIVOLT,
        MAXIMUM_ABS_MILLIVOLT,
        "an electrocardiographic deflection of 100 mV is not a deflection",
    ),
    "uV": PlausibilityRange(
        -MAXIMUM_ABS_MILLIVOLT * 1000.0,
        MAXIMUM_ABS_MILLIVOLT * 1000.0,
        "as for mV, expressed in microvolts",
    ),
}

# A signed balance is bounded, but at -1 rather than 0: it reports which of two
# halves dominates, and the catalog calls that a "fraction".
_SIGNED_BALANCE = PlausibilityRange(
    -1.0, 1.0, "a signed balance runs from -1 to +1, not from 0"
)

_BALANCE_MARKERS: tuple[str, ...] = tuple(
    f"{segment}_{plane}_orientation_balance"
    for segment in ("qrs", "t", "qrst", "p")
    for plane in ("frontal_xy", "horizontal_xz", "sagittal_yz")
) + tuple(
    f"{segment}_activation_energy_early_late_balance_fraction"
    for segment in ("qrs", "t", "qrst", "p")
)

# Markers the unit rule would judge wrongly, because the quantity is
# accumulated along a path or is a signed difference rather than the bounded
# thing its unit name suggests. Each entry is a decision with a reason, not a
# name pattern, so a marker added later is checked rather than silently
# exempted by resemblance.
UNBOUNDED_BY_NATURE: Mapping[str, str] = {
    # Angles accumulated along the loop. One turn is not the limit; the total
    # turning of a fragmented QRS legitimately passes 360 degrees, and 2.644
    # was measured.
    "qrs_total_turning_angle_deg": "accumulated turning, not an orientation",
    "loop_plane_twist_integral": "an integral along the loop",
    "geodesic_direction_path_length": "a path length on the sphere",
    "qrs_integrated_sliding_plane_twist_deg": "an integral along the loop",
    "t_integrated_sliding_plane_twist_deg": "an integral along the loop",
    "qrs_holonomy_planar_departure_deg": "accumulated holonomy, signed",
    "t_holonomy_planar_departure_deg": "accumulated holonomy, signed",
    "qrst_holonomy_planar_departure_deg": "accumulated holonomy, signed",
    # Solid angles that are signed, or accumulated over a self-intersecting
    # spherical path, so the full sphere is not their ceiling.
    "qrs_direction_signed_solid_angle_sr": "signed solid angle",
    "t_direction_signed_solid_angle_sr": "signed solid angle",
    "qrs_direction_absolute_solid_angle_sr": "accumulated over a self-crossing path",
    "t_direction_absolute_solid_angle_sr": "accumulated over a self-crossing path",
    "qrs_holonomy_planar_departure_sr": "signed holonomy",
    "t_holonomy_planar_departure_sr": "signed holonomy",
    "qrst_holonomy_planar_departure_sr": "signed holonomy",
    "qrs_holonomy_absolute_planar_departure_sr": "accumulated holonomy",
    "t_holonomy_absolute_planar_departure_sr": "accumulated holonomy",
    "qrst_holonomy_absolute_planar_departure_sr": "accumulated holonomy",
    # Differences and lags declared in milliseconds. Negative is meaningful:
    # a QTc that shrinks within the record, a lag before rather than after.
    "qrs_xy_cross_correlation_peak_lag_ms": "a lag, which may be negative",
    "within_record_qtc_drift_ms": "a drift, which may be negative",
    "secondary_morphology_qrs_duration_delta_ms": "a difference between two beat populations",
    "secondary_morphology_qt_delta_ms": "a difference between two beat populations",
    "electric_diastole_ms": (
        "RR minus QT; negative when repolarisation outlasts the cycle"
    ),
    # Declared in mV but accumulated over time, so the deflection ceiling does
    # not apply.
    "global_vcg_velocity_integral": "an integral of velocity over the strip",
    # Declared as ratios but formed as indices, not as amplitude quotients.
    "qrs_early_burst_index": "an index, not an amplitude quotient",
    "qrs_late_burst_index": "an index, not an amplitude quotient",
}

# Markers whose declared unit is too loose for the unit rule to apply. Each
# names its reason rather than only its numbers.
_MARKER_RANGES: Mapping[str, PlausibilityRange] = {
    "strip_path_length_per_beat": PlausibilityRange(
        0.0, None, "accumulated path length per beat, not a single voltage deflection",
    ),
    **{
        f"{segment}_{plane}_median_strip_slenderness": PlausibilityRange(
            0.0, None, "strip width / PCA major-axis span; no upper bound of one",
        )
        for segment in ("qrs", "t")
        for plane in ("frontal_xy", "horizontal_xz", "sagittal_yz")
    },
    "t_energy_temporal_concentration": PlausibilityRange(
        1.0, None, "Cauchy-Schwarz: E[energy^2] >= E[energy]^2; no sample-independent upper bound",
        numerical_tolerance=1e-12,
    ),
    # These units are dimensionless, but the formulas have mathematical
    # bounds. These are not population reference intervals or clinical cuts.
    "qrs_st_directional_operator_commutator": PlausibilityRange(
        0.0, math.sqrt(0.5),
        "trace-one PSD operators: convexity reduces to rank-one projectors; "
        "sqrt(2)*abs(cos(theta)*sin(theta)) <= 1/sqrt(2)",
        numerical_tolerance=1e-12,
    ),
    **{
        marker_id: PlausibilityRange(0.0, 1.0, reason, numerical_tolerance=1e-12)
        for marker_id, reason in {
            "qrs_st_directional_operator_overlap": "trace product of trace-one PSD operators lies in [0,1]",
            "qrs_early_late_directional_operator_overlap": "trace product of trace-one PSD operators lies in [0,1]",
            "p_early_late_directional_operator_overlap": "trace product of trace-one PSD operators lies in [0,1]",
            "qrs_t_opposed_energy_fraction": "opposed projected energy is a subset of total T energy",
            "qrs_local24ms_cancellation_fraction": "triangle inequality bounds integrated temporal cancellation",
            "t_direction_time_linear_coupling": "Cauchy-Schwarz: E[p^2]=1 and E[||n||^2]=1",
            "t_direction_time_quadratic_coupling": "Cauchy-Schwarz: E[p^2]=1 and E[||n||^2]=1",
        }.items()
    },
    **{
        marker_id: PlausibilityRange(
            -MAXIMUM_INTERVAL_MS,
            MAXIMUM_INTERVAL_MS,
            "signed lag or peak ordering within a ten-second strip",
        )
        for marker_id in (
            "qrs_xy_cross_correlation_peak_lag_ms",
            "qrs_xz_cross_correlation_peak_lag_ms",
            "qrs_yz_cross_correlation_peak_lag_ms",
            "qrs_leftward_rightward_peak_order_ms",
            "qrs_posterior_anterior_peak_order_ms",
            "v6_v1_interpeak_interval_ms",
        )
    },
    "strip_amplitude_dispersion": PlausibilityRange(
        0.0,
        RATIO_LIMIT,
        "a coefficient of variation is non-negative but may exceed one",
    ),
    # Fréchet distances between curves are non-negative; after per-curve
    # centring and RMS scaling the leash cannot exceed the sum of the two
    # curves' radii, which is not a sample-independent constant. A median
    # absolute deviation of such values is non-negative as well.
    "qrs_t_frechet_beat_median": PlausibilityRange(
        0.0, None, "a Fréchet distance is a maximum of Euclidean distances",
    ),
    "qrs_t_frechet_beat_mad": PlausibilityRange(
        0.0, None, "a median absolute deviation is non-negative",
    ),
    "qrs_t_mirror_symmetry_direct_8lead": PlausibilityRange(
        0.0, 1.0, "d/(1+d) of a non-negative distance lies in [0, 1)",
        numerical_tolerance=1e-12,
    ),
    "classical_dominant_r_v1": PlausibilityRange(
        -RATIO_LIMIT,
        RATIO_LIMIT,
        "R/S in V1 divides by an S wave that may vanish; capped at formation",
    ),
    "classical_r_to_s_ratio": PlausibilityRange(
        -RATIO_LIMIT, RATIO_LIMIT, "as for the R/S ratio in V1"
    ),
    "classical_q_to_r_ratio": PlausibilityRange(
        -RATIO_LIMIT, RATIO_LIMIT, "as for the R/S ratio in V1"
    ),
    "classical_t_to_r_ratio": PlausibilityRange(
        -RATIO_LIMIT, RATIO_LIMIT, "as for the R/S ratio in V1"
    ),
}


def declared_range(
    marker_id: str, unit: Optional[str]
) -> Optional[PlausibilityRange]:
    """The range a marker may take, or ``None`` when it carries none."""

    marker_id = str(marker_id)
    if marker_id.startswith("direct8_") or marker_id.startswith("postectopic_direct8_") or (
        "direct_8lead" in marker_id and any(token in marker_id for token in ("_tp_", "_beat_", "stt_"))
    ):
        if "_slope" in marker_id or marker_id.startswith("postectopic_direct8_stt"):
            return PlausibilityRange(None, None, "signed research response/slope; context-dependent, no universal bound")
        if "delta" in marker_id or "tp_minus_pr" in marker_id:
            return PlausibilityRange(-180., 180., "difference of two angles in [0,180]")
        if unit == "deg":
            return PlausibilityRange(0., 180., "unsigned angle, angle MAD or RMS successive difference")
        return PlausibilityRange(0., None, "nonnegative integrated magnitude or norm ratio; no unit-based amplitude ceiling")
    specific = _MARKER_RANGES.get(marker_id)
    if specific is not None:
        return specific
    if marker_id in UNBOUNDED_BY_NATURE:
        return None
    if marker_id in _BALANCE_MARKERS:
        return _SIGNED_BALANCE
    if unit is None:
        return None
    return _UNIT_RANGES.get(str(unit))


def declared_ranges(
    units_by_marker: Mapping[str, Optional[str]],
) -> dict[str, PlausibilityRange]:
    ranges: dict[str, PlausibilityRange] = {}
    for marker_id, unit in units_by_marker.items():
        found = declared_range(marker_id, unit)
        if found is not None:
            ranges[str(marker_id)] = found
    return ranges


def plausibility_contract(marker_id: str, unit: Optional[str]) -> dict:
    """Expose audit coverage without confusing absent bounds with a pass."""
    bounds = declared_range(marker_id, unit)
    if bounds is not None:
        return {
            "coverage": "range_defined",
            "measurement_kind": (
                "accumulated_path_per_beat" if marker_id == "strip_path_length_per_beat"
                else "geometric_width_span_ratio" if marker_id.endswith("_median_strip_slenderness")
                else "marker_specific" if marker_id in _MARKER_RANGES
                else "unit_based_screen"
            ),
            "reason": bounds.reason,
            "observed_values_checked": False,
        }
    if marker_id in UNBOUNDED_BY_NATURE:
        return {"coverage": "unbounded_by_definition", "measurement_kind": "explicit_exception",
                "reason": UNBOUNDED_BY_NATURE[marker_id], "observed_values_checked": False}
    return {"coverage": "not_defined", "measurement_kind": "unspecified",
            "reason": "No mathematical range registered; not a passed numerical audit",
            "observed_values_checked": False}


def marker_units() -> dict[str, Optional[str]]:
    """The unit each registered marker declares, read from the catalog."""

    from ..core.biomarkers import BiomarkerRegistry

    units: dict[str, Optional[str]] = {}
    for name in BiomarkerRegistry.available():
        unit = None
        definition = getattr(BiomarkerRegistry, "definition", None)
        if callable(definition):
            try:
                unit = getattr(definition(name), "unit", None)
            except Exception:  # noqa: BLE001 - a missing definition is not fatal
                unit = None
        if unit is None:
            try:
                unit = BiomarkerRegistry.get(name).unit
            except Exception:  # noqa: BLE001
                unit = None
        units[str(name)] = None if unit is None else str(unit)
    return units


_REPORT_COLUMNS = (
    "marker_id",
    "checked_values",
    "violating_values",
    "lower_bound",
    "upper_bound",
    "observed_minimum",
    "observed_maximum",
    "extreme_violation",
    "reason",
    "plausibility_version",
)


def check_frame(
    frame: pd.DataFrame,
    ranges: Mapping[str, PlausibilityRange],
) -> pd.DataFrame:
    """Count, per marker, the values its declared range forbids.

    One row per checked marker, including those with no violation: a table that
    lists only failures cannot distinguish "checked and clean" from "never
    checked", and that distinction is the whole value of the check.
    """

    rows = []
    for marker_id, allowed in sorted(ranges.items()):
        if marker_id not in frame.columns:
            continue
        values = pd.to_numeric(frame[marker_id], errors="coerce").to_numpy(
            dtype=float, copy=False
        )
        finite = np.isfinite(values)
        bad = allowed.violations(values)
        observed = values[finite]
        rows.append(
            {
                "marker_id": marker_id,
                "checked_values": int(finite.sum()),
                "violating_values": int(bad.sum()),
                "lower_bound": allowed.lower,
                "upper_bound": allowed.upper,
                "observed_minimum": (
                    float(observed.min()) if observed.size else float("nan")
                ),
                "observed_maximum": (
                    float(observed.max()) if observed.size else float("nan")
                ),
                "extreme_violation": (
                    float(np.max(np.abs(values[bad])))
                    if bad.any()
                    else float("nan")
                ),
                "reason": allowed.reason,
                "plausibility_version": MARKER_PLAUSIBILITY_VERSION,
            }
        )
    return pd.DataFrame(rows, columns=list(_REPORT_COLUMNS))


def merge_reports(reports: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-partition reports into one, without re-reading anything."""

    usable = [
        report
        for report in reports
        if report is not None and not report.empty
    ]
    if not usable:
        return pd.DataFrame(columns=list(_REPORT_COLUMNS))
    combined = pd.concat(usable, ignore_index=True)
    grouped = combined.groupby("marker_id", as_index=False).agg(
        checked_values=("checked_values", "sum"),
        violating_values=("violating_values", "sum"),
        lower_bound=("lower_bound", "first"),
        upper_bound=("upper_bound", "first"),
        observed_minimum=("observed_minimum", "min"),
        observed_maximum=("observed_maximum", "max"),
        extreme_violation=("extreme_violation", "max"),
        reason=("reason", "first"),
        plausibility_version=("plausibility_version", "first"),
    )
    return grouped.sort_values(
        ["violating_values", "marker_id"], ascending=[False, True]
    ).reset_index(drop=True)


__all__ = [
    "MARKER_PLAUSIBILITY_VERSION",
    "UNBOUNDED_BY_NATURE",
    "MAXIMUM_ABS_MILLIVOLT",
    "MAXIMUM_INTERVAL_MS",
    "MAXIMUM_RATE_BPM",
    "PlausibilityRange",
    "RATIO_LIMIT",
    "check_frame",
    "declared_range",
    "declared_ranges",
    "marker_units",
    "merge_reports",
]
