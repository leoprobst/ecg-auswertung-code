"""Loop shape that does not change when the coordinate frame turns.

Three markers in this catalog took the standard deviation across the three
spatial components of a single vector and called it a dispersion. Rotate the
patient's coordinate frame and the number moves; nothing physiological may do
that. Two of the three were repaired against their own catalog definition,
which asked for a beat-to-beat scatter. The third,
``vcg_t_area_dispersion_xyz``, is coordinate-wise **by declaration** and was
left alone rather than silently redefined -- and measured under a 40 degree
rotation it shifts by 4,2 %, which is what the whole class was doing.

The same weakness runs deeper than those three. ``loop_sector_entropy`` and
``qrs_octant_dominant_fraction`` discretise direction into octants, which are
the axis-aligned sectors of whichever frame the transform happened to produce;
turn the frame and points cross sector walls. An octant histogram is a
measurement of the coordinate system as much as of the heart.

So this module is the invariant half of the catalog, and each family answers a
question the frame-dependent version could only approximate:

``shape``
    The second-moment tensor of the loop has three eigenvalues, and every
    symmetric function of them is invariant. Fractional anisotropy -- taken
    from diffusion MRI, where the same problem was solved for the same reason
    -- says how far the loop is from a sphere on a scale of 0 to 1, with no
    reference to any axis.

``bending``
    Curvature and torsion are the two Frenet-Serret invariants: between them
    they determine a space curve up to a rigid motion, which is exactly the
    ambiguity a vectorcardiogram has. Total absolute curvature counts how much
    the loop turns; torsion counts how far it leaves its own plane, and where
    that happens in the beat.

``topology``
    The Gauss linking integral asks how much the depolarisation and
    repolarisation paths wind through each other in space -- the full form of
    what the spatial QRS-T angle reports as a single number between two mean
    vectors. Writhe asks the same of one curve against itself.

``direction``
    Twenty congruent spherical sectors from an icosahedron instead of eight
    axis-aligned octants. **This helps less than it sounds, and the measurement
    says where.** Turning the frame through sixty random rotations on one
    record and reading the spread of each marker:

    ======================================  =========
    marker                                  sd/|mean|
    ======================================  =========
    ``t_octant_occupancy_entropy``              0,100
    ``t_icosahedral_sector_entropy``            0,028
    ``qrs_octant_occupancy_entropy``            0,066
    ``qrs_icosahedral_sector_entropy``          0,068
    ``qrs_octant_dominant_fraction``            0,099
    ``qrs_icosahedral_dominant_fraction``       0,123
    ======================================  =========

    So the T loop improves by a factor of 3,5, the QRS loop does not improve at
    all, and the dominant-sector version is *worse* with twenty sectors than
    with eight -- more sectors means smaller shares and a larger relative
    swing. The dominant-fraction variant was therefore not kept. Only the two
    entropies are here, and the QRS one is declared as a draw rather than an
    improvement; the redundancy stage will say whether it carries anything the
    octant version does not.

``signature``
    The level-2 term of a path signature is the Levy area, which is the area
    vector the ventricular gradient is built from -- the project already uses
    it under another name. Level 3 is unused, and it is the interesting one:
    signature terms are invariant under reparametrisation, so a level-3 norm
    describes the *shape* of the loop independently of how fast it was
    traversed or how densely it was sampled. That is the heart-rate and
    sampling-rate independence the rest of the analysis has to recover
    statistically afterwards.

Two further families are here because their reference standard exists in this
dataset, not because they are invariants:

``respiration``
    The heart axis swings with the breath. Measured beat to beat over ten
    seconds this is a weak estimate -- a strip holds ten to thirteen beats, so
    the beat series is sampled at about 1 Hz against a respiratory rate of
    0,2-0,4 Hz, two to three points per cycle. It is declared as a surrogate
    for that reason. Its value is that the reference is a *number* rather than
    a code: the ventilator's set and measured respiratory rate is recorded in
    the intensive-care tables.

``tremor``
    Body tremor enters the electrocardiogram as 3-8 Hz noise. The obvious
    place to look for it is the diastolic window, and that is the wrong place:
    fibrillatory waves live at 3-12 Hz in exactly that window, so a diastolic
    tremor marker measures atrial fibrillation. The discriminator is where the
    band sits rather than how strong it is -- tremor is a body-surface signal
    present through the whole strip, fibrillatory activity exists only between
    the ventricular complexes. ``tremor_band_systole_to_diastole_ratio`` is
    that comparison and is the marker of the pair worth reading.

    Both are measured on the strip with the median beat subtracted, and the
    first version was not. Its own test found the reason: adding a 5 Hz tone
    to the whole record and adding it only in diastole produced the same ratio
    to three digits (1,768 against 1,786), because band power inside a QRST
    window is dominated by the complex's own spectrum rather than by anything
    added to it. Subtracting what repeats leaves what does not, which is what
    a tremor is.

Nothing here is claimed to replace a marker that already exists. The redundancy
stage compares them against the whole catalog after extraction and says which
of them turn out to be an old quantity under a new name; that answer is
measured rather than asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .base import (
    BiomarkerBase,
    BiomarkerDefinition,
    BiomarkerRegistry,
    BiomarkerResult,
)
from .catalog_biomarkers import FeatureContext, FeatureValue, _context
from ..delineation.base import RecordDelineation


INVARIANT_VERSION = "1.1.1"
INVARIANT_SOURCE = "rotation_invariant_v1"

EPS = 1e-12

# A loop needs enough points for a second derivative at both ends before
# curvature and torsion mean anything.
MINIMUM_CURVE_POINTS = 8

# Resampling length for the shape families. Fixed so that a marker cannot
# change with the sampling rate, and odd enough not to alias against a
# 500 Hz grid.
CURVE_RESAMPLE_POINTS = 101

# Respiration in adults, including mechanical ventilation at both extremes.
RESPIRATORY_BAND_HZ = (0.1, 0.6)

# Physiological tremor and its pathological forms: parkinsonian rest tremor is
# 4-6 Hz, essential tremor 4-12 Hz, and the shivering and rigor band overlaps
# both. 3-8 Hz is the part that all of them share.
TREMOR_BAND_HZ = (3.0, 8.0)


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory):
    cache = getattr(ctx, "_invariant_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_invariant_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _segment_curve(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    """The complete origin-corrected segment; never bridge missing samples."""

    def build() -> Optional[np.ndarray]:
        centered = ctx.origin_corrected_vcg()
        bounds = ctx.bounds(segment)
        if centered is None or bounds is None:
            return None
        if not ctx.is_segment_evaluable(segment):
            return None
        start, end = bounds
        if end + 1 > len(centered) or end <= start:
            return None
        window = np.asarray(centered[start : end + 1, :], dtype=float)
        if not np.isfinite(window).all():
            return None
        return window if window.shape[0] >= 3 else None

    return _memo(ctx, ("segment_curve", segment), build)


def _resampled(points: np.ndarray, count: int = CURVE_RESAMPLE_POINTS) -> np.ndarray:
    """Arc-length resampling, so the shape families do not measure timing."""

    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(arc[-1])
    if not np.isfinite(total) or total <= EPS:
        return points
    grid = np.linspace(0.0, total, count)
    return np.column_stack(
        [np.interp(grid, arc, points[:, axis]) for axis in range(points.shape[1])]
    )


# --------------------------------------------------------------------------
# shape: the second-moment tensor
# --------------------------------------------------------------------------


def _eigenvalues(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        points = _segment_curve(ctx, segment)
        if points is None or points.shape[0] < 3:
            return None
        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / float(points.shape[0])
        values = np.linalg.eigvalsh(covariance)
        values = np.clip(values[::-1], 0.0, None)
        return values if np.all(np.isfinite(values)) else None

    return _memo(ctx, ("eigenvalues", segment), build)


def _fractional_anisotropy(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        values = _eigenvalues(ctx, segment)
        if values is None:
            return _abstain(f"{segment} loop unavailable")
        total = float(np.sum(values**2))
        if total <= EPS:
            return _abstain(f"{segment} loop has no extent")
        mean = float(np.mean(values))
        numerator = float(np.sum((values - mean) ** 2))
        return FeatureValue(float(np.sqrt(1.5 * numerator / total)))

    return calculate


def _sphericity(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        values = _eigenvalues(ctx, segment)
        if values is None:
            return _abstain(f"{segment} loop unavailable")
        if values[0] <= EPS:
            return _abstain(f"{segment} loop has no extent")
        return FeatureValue(float(values[2] / values[0]))

    return calculate


def _second_moment_trace(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        values = _eigenvalues(ctx, segment)
        if values is None:
            return _abstain(f"{segment} loop unavailable")
        # mV^2 -> uV^2 so the number is readable beside the amplitude markers.
        return FeatureValue(float(np.sum(values)) * 1.0e6)

    return calculate


# --------------------------------------------------------------------------
# bending: curvature and torsion
# --------------------------------------------------------------------------


def _frenet(ctx: FeatureContext, segment: str):
    """Curvature and torsion along the arc-length resampled loop."""

    def build():
        points = _segment_curve(ctx, segment)
        if points is None or points.shape[0] < MINIMUM_CURVE_POINTS:
            return None
        curve = _resampled(points)
        first = np.gradient(curve, axis=0)
        second = np.gradient(first, axis=0)
        third = np.gradient(second, axis=0)
        cross = np.cross(first, second)
        cross_norm = np.linalg.norm(cross, axis=1)
        speed = np.linalg.norm(first, axis=1)
        curvature = np.divide(
            cross_norm,
            np.maximum(speed**3, EPS),
            out=np.zeros_like(cross_norm),
            where=speed > EPS,
        )
        triple = np.einsum("ij,ij->i", cross, third)
        torsion = np.divide(
            triple,
            np.maximum(cross_norm**2, EPS),
            out=np.zeros_like(triple),
            where=cross_norm > EPS,
        )
        if not (np.all(np.isfinite(curvature)) and np.all(np.isfinite(torsion))):
            return None
        return curvature, torsion, speed

    return _memo(ctx, ("frenet", segment), build)


def _curvature_integral(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _frenet(ctx, segment)
        if result is None:
            return _abstain(f"{segment} loop too short for a curvature")
        curvature, _, speed = result
        return FeatureValue(float(np.trapezoid(curvature * speed)))

    return calculate


def _torsion_integral(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _frenet(ctx, segment)
        if result is None:
            return _abstain(f"{segment} loop too short for a torsion")
        _, torsion, speed = result
        return FeatureValue(float(np.trapezoid(np.abs(torsion) * speed)))

    return calculate


def _peak_curvature_position(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _frenet(ctx, segment)
        if result is None:
            return _abstain(f"{segment} loop too short for a curvature")
        curvature, _, _ = result
        if curvature.size < 2:
            return _abstain(f"{segment} loop too short for a curvature")
        return FeatureValue(
            float(int(np.argmax(curvature)) / (curvature.size - 1))
        )

    return calculate


def _torsion_sign_changes(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _frenet(ctx, segment)
        if result is None:
            return _abstain(f"{segment} loop too short for a torsion")
        _, torsion, _ = result
        # Ignore the numerical dither around zero: only sign changes that
        # cross a tenth of the segment's own torsion scale are counted.
        scale = float(np.percentile(np.abs(torsion), 90))
        if scale <= EPS:
            return FeatureValue(0.0)
        signs = np.sign(np.where(np.abs(torsion) < 0.1 * scale, 0.0, torsion))
        signs = signs[signs != 0.0]
        if signs.size < 2:
            return FeatureValue(0.0)
        return FeatureValue(float(int(np.count_nonzero(np.diff(signs) != 0))))

    return calculate


# --------------------------------------------------------------------------
# topology: linking and writhe
# --------------------------------------------------------------------------


def _closed(points: np.ndarray) -> np.ndarray:
    if np.linalg.norm(points[-1] - points[0]) <= EPS:
        return points
    return np.vstack([points, points[0]])


def _gauss_double_integral(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    """Discrete Gauss integral between two closed polygons.

    Segment midpoints and displacement vectors, which is the standard
    trapezoidal reading of the double integral and is enough for curves of a
    hundred points. Returns ``None`` when the two curves come closer than the
    discretisation can resolve, because the integrand then diverges and the
    number would be an artefact of the sampling rather than of the geometry.
    """

    a0 = first[:-1]
    da = np.diff(first, axis=0)
    b0 = second[:-1]
    db = np.diff(second, axis=0)
    midpoint_a = a0 + 0.5 * da
    midpoint_b = b0 + 0.5 * db
    difference = midpoint_a[:, None, :] - midpoint_b[None, :, :]
    distance = np.linalg.norm(difference, axis=2)
    scale = max(
        float(np.ptp(first, axis=0).max()),
        float(np.ptp(second, axis=0).max()),
    )
    if scale <= EPS:
        return None
    if float(distance.min()) < 1e-4 * scale:
        return None
    cross = np.cross(da[:, None, :], db[None, :, :])
    numerator = np.einsum("ijk,ijk->ij", difference, cross)
    return float(np.sum(numerator / distance**3) / (4.0 * np.pi))


def _qrs_t_linking_number(ctx: FeatureContext) -> FeatureValue:
    qrs = _segment_curve(ctx, "qrs")
    t_wave = _segment_curve(ctx, "t")
    if qrs is None or t_wave is None:
        return _abstain("QRS or T loop unavailable")
    value = _gauss_double_integral(
        _closed(_resampled(qrs, 61)), _closed(_resampled(t_wave, 61))
    )
    if value is None:
        return _abstain("QRS and T loops touch below the resolution")
    return FeatureValue(float(value))


def _qrst_writhe(ctx: FeatureContext) -> FeatureValue:
    curve = _segment_curve(ctx, "qrst")
    if curve is None:
        return _abstain("QRST loop unavailable")
    closed = _closed(_resampled(curve, 81))
    a0 = closed[:-1]
    da = np.diff(closed, axis=0)
    midpoint = a0 + 0.5 * da
    difference = midpoint[:, None, :] - midpoint[None, :, :]
    distance = np.linalg.norm(difference, axis=2)
    scale = float(np.ptp(closed, axis=0).max())
    if scale <= EPS:
        return _abstain("QRST loop has no extent")
    # The diagonal and its neighbours are the self-contact of a curve with
    # itself, where the integrand is singular by construction rather than by
    # geometry.
    mask = np.abs(np.subtract.outer(
        np.arange(len(midpoint)), np.arange(len(midpoint))
    )) > 1
    mask &= distance > 1e-4 * scale
    if not mask.any():
        return _abstain("QRST loop is degenerate")
    cross = np.cross(da[:, None, :], da[None, :, :])
    numerator = np.einsum("ijk,ijk->ij", difference, cross)
    contribution = np.where(mask, numerator / np.maximum(distance, EPS) ** 3, 0.0)
    return FeatureValue(float(np.sum(contribution) / (4.0 * np.pi)))


# --------------------------------------------------------------------------
# direction: near-isotropic sectors
# --------------------------------------------------------------------------


def _icosahedral_centres() -> np.ndarray:
    """The twenty face centres of a regular icosahedron, as unit vectors."""

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = []
    for sign_a in (-1.0, 1.0):
        for sign_b in (-1.0, 1.0):
            vertices.extend(
                [
                    (0.0, sign_a * 1.0, sign_b * phi),
                    (sign_a * 1.0, sign_b * phi, 0.0),
                    (sign_a * phi, 0.0, sign_b * 1.0),
                ]
            )
    points = np.unique(np.asarray(vertices, dtype=float), axis=0)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    # Face centres: every triple of mutually nearest vertices. The edge length
    # of a unit icosahedron is 2/sqrt(phi*sqrt(5)); a tolerance around it picks
    # exactly the twenty faces.
    edge = 2.0 / np.sqrt(phi * np.sqrt(5.0))
    faces = []
    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            if abs(np.linalg.norm(points[i] - points[j]) - edge) > 1e-6:
                continue
            for k in range(j + 1, count):
                if abs(np.linalg.norm(points[i] - points[k]) - edge) > 1e-6:
                    continue
                if abs(np.linalg.norm(points[j] - points[k]) - edge) > 1e-6:
                    continue
                faces.append(points[i] + points[j] + points[k])
    centres = np.asarray(faces, dtype=float)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    return centres


_ICOSAHEDRAL_SECTORS = _icosahedral_centres()


def _sector_weights(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        points = _segment_curve(ctx, segment)
        if points is None or points.shape[0] < 3:
            return None
        steps = np.diff(points, axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        keep = lengths > EPS
        if not keep.any():
            return None
        directions = steps[keep] / lengths[keep, None]
        assignment = np.argmax(directions @ _ICOSAHEDRAL_SECTORS.T, axis=1)
        weights = np.bincount(
            assignment,
            weights=lengths[keep],
            minlength=len(_ICOSAHEDRAL_SECTORS),
        )
        total = float(weights.sum())
        return weights / total if total > EPS else None

    return _memo(ctx, ("sectors", segment), build)


def _sector_entropy(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        weights = _sector_weights(ctx, segment)
        if weights is None:
            return _abstain(f"{segment} loop unavailable")
        positive = weights[weights > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        return FeatureValue(entropy / float(np.log(len(weights))))

    return calculate


def _sector_dominant_fraction(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        weights = _sector_weights(ctx, segment)
        if weights is None:
            return _abstain(f"{segment} loop unavailable")
        return FeatureValue(float(weights.max()))

    return calculate


# --------------------------------------------------------------------------
# signature: level three
# --------------------------------------------------------------------------


def _signature_level3(path: np.ndarray) -> Optional[np.ndarray]:
    """Level-3 signature tensor of a piecewise-linear path in three dimensions.

    Chen's identity, applied segment by segment. For a straight increment
    ``x`` the levels are ``x``, ``x^2/2`` and ``x^3/6``; concatenation
    multiplies the truncated tensor algebras. Twenty-seven numbers, all
    invariant under reparametrisation -- travelling the same shape faster,
    slower or with a different sampling grid gives the same tensor.
    """

    steps = np.diff(np.asarray(path, dtype=float), axis=0)
    steps = steps[np.linalg.norm(steps, axis=1) > EPS]
    if steps.shape[0] < 2:
        return None
    level1 = np.zeros(3)
    level2 = np.zeros((3, 3))
    level3 = np.zeros((3, 3, 3))
    for step in steps:
        step_2 = np.einsum("i,j->ij", step, step) / 2.0
        step_3 = np.einsum("i,j,k->ijk", step, step, step) / 6.0
        new_level3 = (
            level3
            + np.einsum("ij,k->ijk", level2, step)
            + np.einsum("i,jk->ijk", level1, step_2)
            + step_3
        )
        new_level2 = level2 + np.einsum("i,j->ij", level1, step) + step_2
        level1 = level1 + step
        level2 = new_level2
        level3 = new_level3
    return level3 if np.all(np.isfinite(level3)) else None


def _signature_norm(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        points = _segment_curve(ctx, segment)
        if points is None:
            return _abstain(f"{segment} loop unavailable")
        # Scaled to unit path length first. The level-3 tensor is homogeneous
        # of degree three, so unscaled the marker would mostly report
        # amplitude, which a dozen other markers already report.
        #
        # The scale must itself be rotation-invariant, and the obvious choice
        # is not: dividing by the bounding-box extent
        # ``norm(ptp(points, axis=0))`` made this marker move by 2,1 % under a
        # rotation, because the bounding box turns with the frame while the
        # loop does not change. Path length does not.
        extent = float(
            np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        )
        if extent <= EPS:
            return _abstain(f"{segment} loop has no extent")
        tensor = _signature_level3(points / extent)
        if tensor is None:
            return _abstain(f"{segment} loop has too few distinct steps")
        return FeatureValue(float(np.linalg.norm(tensor)))

    return calculate


# --------------------------------------------------------------------------
# respiration
# --------------------------------------------------------------------------


def _beat_axis_series(ctx: FeatureContext) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Direction of each beat's mean QRS vector, and the times it was taken."""

    def build():
        centered = ctx.origin_corrected_vcg()
        if centered is None:
            return None
        times: list[float] = []
        vectors: list[np.ndarray] = []
        for beat in ctx.delineation.beats:
            onset = getattr(beat, "r_onset", None)
            offset = getattr(beat, "r_offset", None)
            if onset is None or offset is None:
                continue
            start = int(onset.sample)
            end = int(offset.sample)
            if not (0 <= start < end < len(centered)):
                continue
            window = centered[start : end + 1]
            if not np.all(np.isfinite(window)):
                continue
            vector = window.mean(axis=0)
            if np.linalg.norm(vector) <= EPS:
                continue
            times.append(float(beat.r_peak.sample) / float(ctx.fs))
            vectors.append(vector / np.linalg.norm(vector))
        if len(vectors) < 6:
            return None
        return np.asarray(times), np.vstack(vectors)

    return _memo(ctx, ("beat_axis",), build)


def _axis_modulation_deg(ctx: FeatureContext) -> FeatureValue:
    series = _beat_axis_series(ctx)
    if series is None:
        return _abstain("fewer than six beats with a QRS window")
    _, vectors = series
    mean = vectors.mean(axis=0)
    if np.linalg.norm(mean) <= EPS:
        return _abstain("beat axes cancel")
    mean = mean / np.linalg.norm(mean)
    angles = np.degrees(np.arccos(np.clip(vectors @ mean, -1.0, 1.0)))
    return FeatureValue(float(np.ptp(angles)))


def _respiratory_rate_hz(ctx: FeatureContext) -> FeatureValue:
    series = _beat_axis_series(ctx)
    if series is None:
        return _abstain("fewer than six beats with a QRS window")
    times, vectors = series
    span = float(times[-1] - times[0])
    if span <= EPS:
        return _abstain("beats carry no time span")
    mean = vectors.mean(axis=0)
    if np.linalg.norm(mean) <= EPS:
        return _abstain("beat axes cancel")
    signal = vectors @ (mean / np.linalg.norm(mean))
    signal = signal - signal.mean()
    if float(np.std(signal)) <= EPS:
        return _abstain("beat axis does not move")
    # A Lomb-style scan rather than an FFT: the beat series is unevenly
    # spaced, and resampling it onto a grid would put the answer at whatever
    # the grid implies.
    frequencies = np.linspace(RESPIRATORY_BAND_HZ[0], RESPIRATORY_BAND_HZ[1], 96)
    angles = 2.0 * np.pi * np.outer(frequencies, times)
    power = (
        np.abs(np.cos(angles) @ signal) ** 2 + np.abs(np.sin(angles) @ signal) ** 2
    )
    return FeatureValue(float(frequencies[int(np.argmax(power))]))


def _respiratory_modulation_depth(ctx: FeatureContext) -> FeatureValue:
    series = _beat_axis_series(ctx)
    if series is None:
        return _abstain("fewer than six beats with a QRS window")
    times, vectors = series
    span = float(times[-1] - times[0])
    lower, upper = RESPIRATORY_BAND_HZ
    # At least two cycles of the lowest studied frequency. A ten-second
    # diagnostic strip cannot support this band-power claim.
    if span < 2.0 / lower or np.any(np.diff(times) <= 0):
        return _abstain("insufficient duration for respiratory band power")
    # Conservative sampling-support check, not a Nyquist theorem for
    # irregular samples. Never silently integrate a truncated target band.
    ceiling = 0.5 / float(np.max(np.diff(times)))
    if ceiling < upper:
        return _abstain("respiratory band exceeds beat sampling support")
    mean = vectors.mean(axis=0)
    if np.linalg.norm(mean) <= EPS:
        return _abstain("beat axes cancel")
    signal = vectors @ (mean / np.linalg.norm(mean))
    signal = signal - signal.mean()
    total = float(np.sum(signal**2))
    if total <= EPS:
        return _abstain("beat axis does not move")
    # One grid and one quadrature rule for numerator and denominator.
    # Include band edges exactly so the numerator is a subset integral.
    frequencies = np.unique(np.r_[np.linspace(0.02, ceiling, 192), lower, upper])
    angles = 2.0 * np.pi * np.outer(frequencies, times)
    power = (
        np.abs(np.cos(angles) @ signal) ** 2 + np.abs(np.sin(angles) @ signal) ** 2
    )
    areas = np.diff(frequencies) * (power[:-1] + power[1:]) * 0.5
    in_band = (frequencies[:-1] >= lower) & (frequencies[1:] <= upper)
    denominator = float(np.sum(areas))
    if denominator <= EPS:
        return _abstain("beat axis does not move")
    return FeatureValue(float(np.sum(areas[in_band]) / denominator))


# --------------------------------------------------------------------------
# tremor
# --------------------------------------------------------------------------


def _band_power(values: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    if values.size < 8:
        return float("nan")
    detrended = values - values.mean()
    window = np.hanning(detrended.size)
    spectrum = np.abs(np.fft.rfft(detrended * window)) ** 2
    freqs = np.fft.rfftfreq(detrended.size, d=1.0 / fs)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    if not inside.any():
        return float("nan")
    return float(np.sum(spectrum[inside]))


def _strip_scalar(ctx: FeatureContext) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        source = ctx.vcg_segment("global_vcg")
        if source is None or source.shape[0] < 64:
            return None
        if not np.all(np.isfinite(source)):
            return None
        centered = source - source.mean(axis=0)
        # Project on the strip's principal direction: a magnitude would
        # rectify the signal and double every frequency in it, which is the
        # defect the atrial spectrum already had to be repaired for.
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        return centered @ vectors[0]

    return _memo(ctx, ("strip_scalar",), build)


# The window around each R peak that the beat template covers. -100 ms to
# +450 ms spans QRS through T at any plausible rate; beats closer together
# than that simply overlap in the template, which is what a median over beats
# is for.
_TEMPLATE_BEFORE_MS = 100.0
_TEMPLATE_AFTER_MS = 450.0


def _beat_residual_strip(
    ctx: FeatureContext,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """The strip with the repeated beat subtracted, and where the beats were.

    Without this the marker measures the QRS complex. Band power between 3 and
    8 Hz inside a QRST window is dominated by the complex's own spectrum --
    measured on a synthetic record, adding a 5 Hz tone everywhere and adding it
    only in diastole gave the same ratio to three digits, because in both cases
    the systolic term was the beat rather than the tone. Subtracting the median
    beat removes the part that repeats and leaves the part that does not, which
    is what a tremor is.

    Returns the residual and a mask that is true inside the beat windows.
    """

    def build():
        signal = _strip_scalar(ctx)
        if signal is None:
            return None
        before = int(round(_TEMPLATE_BEFORE_MS * ctx.fs / 1000.0))
        after = int(round(_TEMPLATE_AFTER_MS * ctx.fs / 1000.0))
        width = before + after
        peaks = [
            int(beat.r_peak.sample)
            for beat in ctx.delineation.beats
            if beat.r_peak is not None
        ]
        windows = [
            (peak - before, peak + after)
            for peak in peaks
            if peak - before >= 0 and peak + after <= len(signal)
        ]
        if len(windows) < 3:
            return None
        stack = np.vstack([signal[start:end] for start, end in windows])
        template = np.median(stack, axis=0)
        residual = signal.copy()
        mask = np.zeros(signal.size, dtype=bool)
        for start, end in windows:
            residual[start:end] = signal[start:end] - template
            mask[start:end] = True
        if not mask.any() or mask.all():
            return None
        return residual, mask

    return _memo(ctx, ("beat_residual",), build)


def _tremor_band_fraction(ctx: FeatureContext) -> FeatureValue:
    result = _beat_residual_strip(ctx)
    if result is None:
        signal = _strip_scalar(ctx)
        if signal is None:
            return _abstain("strip unavailable or not finite")
        return _abstain("fewer than three beats for a template")
    residual, _ = result
    band = _band_power(residual, float(ctx.fs), TREMOR_BAND_HZ)
    total = _band_power(residual, float(ctx.fs), (0.5, min(40.0, 0.45 * ctx.fs)))
    if not np.isfinite(band) or not np.isfinite(total) or total <= EPS:
        return _abstain("residual carries no spectral power")
    return FeatureValue(float(band / total))


def _tremor_systole_to_diastole(ctx: FeatureContext) -> FeatureValue:
    result = _beat_residual_strip(ctx)
    if result is None:
        return _abstain("fewer than three beats for a template")
    residual, mask = result
    systole = residual[mask]
    diastole = residual[~mask]
    if systole.size < 64 or diastole.size < 64:
        return _abstain("one of the two windows is too short")
    inside = _band_power(systole, float(ctx.fs), TREMOR_BAND_HZ)
    outside = _band_power(diastole, float(ctx.fs), TREMOR_BAND_HZ)
    if not np.isfinite(inside) or not np.isfinite(outside):
        return _abstain("band power not computable in both windows")
    # Per sample, so the ratio does not report which window is longer.
    inside /= float(systole.size)
    outside /= float(diastole.size)
    if outside <= EPS or inside <= EPS:
        return _abstain("no power in the band on one side")
    return FeatureValue(float(np.log10(inside / outside)))


# --------------------------------------------------------------------------
# atrial: the interatrial block signature
# --------------------------------------------------------------------------


def _p_terminal_reversal_deg(ctx: FeatureContext) -> FeatureValue:
    points = _segment_curve(ctx, "p")
    if points is None or points.shape[0] < 6:
        return _abstain("P loop unavailable or too short")
    third = max(2, points.shape[0] // 3)
    early = np.diff(points[:third], axis=0).sum(axis=0)
    late = np.diff(points[-third:], axis=0).sum(axis=0)
    if np.linalg.norm(early) <= EPS or np.linalg.norm(late) <= EPS:
        return _abstain("P loop does not move in one of its thirds")
    cosine = float(
        np.dot(early, late) / (np.linalg.norm(early) * np.linalg.norm(late))
    )
    return FeatureValue(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))


def _p_terminal_area_fraction(ctx: FeatureContext) -> FeatureValue:
    points = _segment_curve(ctx, "p")
    if points is None or points.shape[0] < 6:
        return _abstain("P loop unavailable or too short")
    scale = float(np.max(np.abs(points)))
    if not np.isfinite(scale) or scale <= EPS:
        return _abstain("P loop has no measurable amplitude")
    normalized = points / scale
    centered = normalized - normalized.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    # Rotation within an unresolved second/third singular subspace changes
    # projected sector areas. Do not report an arbitrary SVD basis choice.
    if singular_values[0] <= EPS or (
        singular_values[1] - singular_values[2]
    ) <= 1e-12 * singular_values[0]:
        return _abstain("P principal plane numerically non-identifiable")
    scores = centered @ vectors[:2].T
    third = max(2, points.shape[0] // 3)

    # Absolute swept triangle area about the common centroid. No artificial
    # closing chord, no cancellation between oppositely oriented subloops.
    sectors = .5 * np.abs(scores[:-1, 0] * scores[1:, 1]
                          - scores[:-1, 1] * scores[1:, 0])
    total = float(sectors.sum())
    if total <= EPS:
        return _abstain("P loop encloses no area")
    return FeatureValue(float(sectors[-(third - 1):].sum() / total))


def _p_loop_direction_reversals(ctx: FeatureContext) -> FeatureValue:
    points = _segment_curve(ctx, "p")
    if points is None or points.shape[0] < 6:
        return _abstain("P loop unavailable or too short")
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    keep = lengths > EPS
    if int(keep.sum()) < 4:
        return _abstain("P loop has too few distinct steps")
    directions = steps[keep] / lengths[keep, None]
    cosines = np.einsum("ij,ij->i", directions[:-1], directions[1:])
    return FeatureValue(float(int(np.count_nonzero(cosines < 0.0))))


@dataclass(frozen=True)
class InvariantSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    evidence_note: str
    references: tuple[str, ...] = ()


def _shape_specs() -> tuple[InvariantSpec, ...]:
    specs: list[InvariantSpec] = []
    for segment in ("qrs", "t", "qrst"):
        specs.extend(
            [
                InvariantSpec(
                    f"{segment}_loop_fractional_anisotropy",
                    "Rotationsinvarianten",
                    f"Fractional anisotropy of the {segment.upper()} loop's "
                    "second-moment tensor.",
                    "sqrt(3/2)*||lambda - mean(lambda)||/||lambda||",
                    "fraction",
                    _fractional_anisotropy(segment),
                    "0 for a spherical cloud, 1 for a straight line. Replaces "
                    "the axis-wise spreads with a quantity that does not move "
                    "when the transform rotates the frame.",
                    ("Basser, Biophys J 1994",),
                ),
                InvariantSpec(
                    f"{segment}_loop_inertia_sphericity",
                    "Rotationsinvarianten",
                    f"Smallest over largest eigenvalue of the {segment.upper()} "
                    "loop's second-moment tensor.",
                    "lambda_3 / lambda_1",
                    "fraction",
                    _sphericity(segment),
                    "How far the loop leaves its own plane, on a scale that "
                    "does not depend on amplitude.",
                ),
                InvariantSpec(
                    f"{segment}_loop_second_moment_trace",
                    "Rotationsinvarianten",
                    f"Trace of the {segment.upper()} loop's second-moment "
                    "tensor.",
                    "1e6 * (lambda_1 + lambda_2 + lambda_3)",
                    "uV^2",
                    _second_moment_trace(segment),
                    "The invariant size of the loop. Amplitude attenuation by "
                    "third-space fluid should show here, against fluid "
                    "balance and paracentesis rather than a diagnosis code.",
                ),
            ]
        )
    return tuple(specs)


def _bending_specs() -> tuple[InvariantSpec, ...]:
    specs: list[InvariantSpec] = []
    for segment in ("qrs", "t", "qrst"):
        specs.extend(
            [
                InvariantSpec(
                    f"{segment}_loop_curvature_integral",
                    "Frenet-Serret",
                    f"Total absolute curvature along the {segment.upper()} loop.",
                    "integral |kappa| ds on the arc-length resampled loop",
                    "dimensionless",
                    _curvature_integral(segment),
                    "How much the loop turns in total. Fragmented conduction "
                    "adds turning without adding path length.",
                    ("do Carmo, Differential Geometry of Curves and Surfaces",),
                ),
                InvariantSpec(
                    f"{segment}_loop_torsion_integral",
                    "Frenet-Serret",
                    f"Total absolute torsion along the {segment.upper()} loop.",
                    "integral |tau| ds on the arc-length resampled loop",
                    "dimensionless",
                    _torsion_integral(segment),
                    "How far the loop leaves its own plane as it travels. "
                    "Curvature and torsion together fix a space curve up to a "
                    "rigid motion, which is the ambiguity a VCG has.",
                ),
                InvariantSpec(
                    f"{segment}_loop_peak_curvature_position",
                    "Frenet-Serret",
                    f"Where the sharpest turn of the {segment.upper()} loop "
                    "falls, as a fraction of its arc length.",
                    "argmax(kappa) / (n - 1)",
                    "fraction",
                    _peak_curvature_position(segment),
                    "A terminal sharp turn is the loop-space form of terminal "
                    "conduction delay; an early one is not the same finding.",
                ),
                InvariantSpec(
                    f"{segment}_loop_torsion_sign_changes",
                    "Frenet-Serret",
                    f"Number of torsion sign changes along the "
                    f"{segment.upper()} loop.",
                    "count of sign changes of tau above a tenth of its own p90",
                    "count",
                    _torsion_sign_changes(segment),
                    "Counts how often the loop changes the side it bends "
                    "towards -- a shape statement that survives rotation, "
                    "unlike a notch count read off one lead.",
                ),
            ]
        )
    return tuple(specs)


INVARIANT_SPECS: tuple[InvariantSpec, ...] = (
    _shape_specs()
    + _bending_specs()
    + (
        InvariantSpec(
            "qrs_t_gauss_linking_number",
            "Topologie",
            "Gauss linking integral between the closed QRS and T loops.",
            "(1/4pi) * double integral over both closed curves",
            "dimensionless",
            _qrs_t_linking_number,
            "The spatial QRS-T angle compares two mean vectors and discards "
            "the paths. This asks how far the two paths wind through each "
            "other, which is the quantity the angle approximates.",
            ("Gauss, Werke V, 1877",),
        ),
        InvariantSpec(
            "qrst_loop_writhe",
            "Topologie",
            "Writhe of the closed QRST curve.",
            "(1/4pi) * Gauss double integral of the curve with itself",
            "dimensionless",
            _qrst_writhe,
            "Self-entanglement of one beat's whole trajectory. Zero for a "
            "planar loop however complicated, so it isolates the part of the "
            "shape that planar projections cannot show.",
            ("Fuller, PNAS 1971",),
        ),
        InvariantSpec(
            "qrs_icosahedral_sector_entropy",
            "Isotrope Sektoren",
            "Direction entropy of the QRS loop over twenty icosahedral "
            "sectors, normalised.",
            "-sum p log p / log 20, p = arc length per face sector",
            "fraction",
            _sector_entropy("qrs"),
            "A draw with the octant version, measured: sd/|mean| 0,068 "
            "against 0,066 over sixty random frame rotations. Kept because it "
            "costs nothing beside the T-loop version, which does improve; the "
            "redundancy stage decides whether it says anything new.",
        ),
        InvariantSpec(
            "t_icosahedral_sector_entropy",
            "Isotrope Sektoren",
            "Direction entropy of the T loop over twenty icosahedral sectors.",
            "-sum p log p / log 20, p = arc length per face sector",
            "fraction",
            _sector_entropy("t"),
            "The one sector marker that clearly improves on its octant "
            "counterpart: sd/|mean| 0,028 against 0,100 over sixty random "
            "frame rotations, a factor of 3,5.",
        ),
        InvariantSpec(
            "qrs_signature_level3_norm",
            "Pfadsignatur",
            "Frobenius norm of the level-3 path signature of the QRS loop, "
            "scaled to unit diameter.",
            "||S3(loop / diameter)||_F, Chen recursion over linear steps",
            "dimensionless",
            _signature_norm("qrs"),
            "Signature terms are invariant under reparametrisation, so this "
            "describes the shape independently of heart rate and sampling "
            "rate -- the two confounders every other marker has to have "
            "adjusted out afterwards. Level 2 is the area vector the "
            "ventricular gradient is already built from; level 3 is unused.",
            ("Chen, Ann Math 1957", "Lyons, Rev Mat Iberoam 1998"),
        ),
        InvariantSpec(
            "t_signature_level3_norm",
            "Pfadsignatur",
            "Frobenius norm of the level-3 path signature of the T loop.",
            "||S3(loop / diameter)||_F",
            "dimensionless",
            _signature_norm("t"),
            "As for the QRS loop, on repolarisation.",
        ),
        InvariantSpec(
            "qrst_signature_level3_norm",
            "Pfadsignatur",
            "Frobenius norm of the level-3 path signature of the QRST loop.",
            "||S3(loop / diameter)||_F",
            "dimensionless",
            _signature_norm("qrst"),
            "The whole beat as one path, which is where the interaction "
            "between depolarisation and repolarisation shape would appear.",
        ),
        InvariantSpec(
            "respiratory_axis_modulation_deg",
            "Atmung",
            "Peak-to-peak swing of the beat-wise QRS mean-vector direction.",
            "max - min of angle(beat axis, record mean axis) over beats",
            "degrees",
            _axis_modulation_deg,
            "The heart axis swings with the breath. An absent swing is the "
            "interesting case: controlled ventilation and neuromuscular "
            "blockade remove it. Testable against the ventilator's own "
            "recorded rate and mode rather than against a diagnosis code.",
            ("Moody, Computers in Cardiology 1985",),
        ),
        InvariantSpec(
            "respiratory_axis_rate_hz",
            "Atmung",
            "Dominant frequency of the beat-wise axis series in the "
            "respiratory band.",
            "argmax of a Lomb scan over 0,1-0,6 Hz on unevenly spaced beats",
            "1/s",
            _respiratory_rate_hz,
            "Explicitly a surrogate: ten to thirteen beats sample the axis at "
            "about 1 Hz, which is two to three points per respiratory cycle. "
            "Its worth is decided by the agreement with the measured "
            "respiratory rate in the intensive-care tables, not by argument.",
        ),
        InvariantSpec(
            "respiratory_axis_modulation_depth",
            "Atmung",
            "Share of beat-axis variation that falls in the respiratory band.",
            "band power / total scan power of the axis projection series",
            "fraction",
            _respiratory_modulation_depth,
            "Separates a heart whose axis moves with the breath from one "
            "whose axis moves irregularly, which the swing alone cannot.",
        ),
        InvariantSpec(
            "tremor_band_power_fraction",
            "Tremor",
            "Share of the beat-subtracted strip's power in the 3-8 Hz band.",
            "P(3-8 Hz) / P(0,5-40 Hz) on the principal-axis projection after "
            "subtracting the median beat",
            "fraction",
            _tremor_band_fraction,
            "Read on its own this is confounded: fibrillatory waves occupy "
            "the same band. It is here as the denominator of the ratio below "
            "and as the quantity a reader would otherwise compute wrongly.",
        ),
        InvariantSpec(
            "tremor_band_systole_to_diastole_ratio",
            "Tremor",
            "Log ratio of 3-8 Hz power per sample inside QRS-T against the "
            "diastolic remainder.",
            "log10(P_band/sample inside the beat windows / P_band/sample "
            "outside), on the beat-subtracted residual",
            "dimensionless",
            _tremor_systole_to_diastole,
            "The discriminator of the pair. A body tremor is on the skin "
            "throughout the strip, so the ratio sits near zero; fibrillatory "
            "activity exists only between the ventricular complexes and "
            "drives it negative. Testable against Parkinson disease, "
            "delirium and alcohol withdrawal on one side and the atrial "
            "fibrillation labels on the other.",
        ),
        InvariantSpec(
            "p_loop_terminal_reversal_deg",
            "Vorhofschleife",
            "Angle between the mean direction of the first and last third of "
            "the P loop.",
            "angle(sum of steps in first third, sum of steps in last third)",
            "degrees",
            _p_terminal_reversal_deg,
            "Interatrial block activates the left atrium retrogradely, which "
            "turns the second half of the P loop back on the first. The "
            "scalar form of this is the biphasic P wave in the inferior "
            "leads; the loop shows the reversal directly. Testable against "
            "left atrial volume index and new-onset atrial fibrillation.",
            ("Bayes de Luna, J Electrocardiol 2012",),
        ),
        InvariantSpec(
            "p_loop_terminal_area_fraction",
            "Vorhofschleife",
            "Share of absolute centroid-referenced swept P-loop area in the last temporal third.",
            "sum_last_third |det(z_i,z_(i+1))| / sum_all |det(z_i,z_(i+1))|; "
            "z in the whole P loop's centroid-centered principal plane; no closing chord",
            "fraction",
            _p_terminal_area_fraction,
            "Own exploratory temporal organization measure, not equivalent to "
            "P-terminal force in V1, atrial fibrosis, Cx40 expression or filling pressure. "
            "Oppositely oriented sectors do not cancel. No clinical cutoff.",
            ("research_pathology_round_20260913/07B_ATRIAL_CONDUCTION.md: own hypothesis and formula correction",),
        ),
        InvariantSpec(
            "p_loop_direction_reversals",
            "Vorhofschleife",
            "Number of direction reversals along the P loop.",
            "count of consecutive step pairs with a negative cosine",
            "count",
            _p_loop_direction_reversals,
            "A notch count that does not depend on which lead it is read in.",
        ),
    )
)


class InvariantBiomarker(BiomarkerBase):
    def __init__(self, spec: InvariantSpec) -> None:
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
        notes = [f"invariant_version={INVARIANT_VERSION}"]
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


def rotation_invariant_marker_ids() -> tuple[str, ...]:
    """The markers whose value may not change when the frame is rotated."""

    return tuple(
        spec.name
        for spec in INVARIANT_SPECS
        if spec.family
        in {"Rotationsinvarianten", "Frenet-Serret", "Topologie", "Pfadsignatur"}
    )


def register_invariant_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in INVARIANT_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), InvariantBiomarker):
                continue
            raise RuntimeError(
                f"Invariant marker collides with an existing biomarker: "
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
                source=INVARIANT_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Prospective hypothesis. " + spec.evidence_note +
                    " No diagnostic cutoff is claimed."
                ),
                references=list(spec.references),
                implementation_status="implemented",
            )
        )
        BiomarkerRegistry.register(InvariantBiomarker(spec))


__all__ = [
    "INVARIANT_SPECS",
    "INVARIANT_SOURCE",
    "INVARIANT_VERSION",
    "InvariantBiomarker",
    "InvariantSpec",
    "RESPIRATORY_BAND_HZ",
    "TREMOR_BAND_HZ",
    "register_invariant_biomarkers",
    "rotation_invariant_marker_ids",
]
