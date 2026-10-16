"""Views of the vectorcardiogram that make time and physics visible.

The loop as usually drawn is a shape: a parametrised curve r(t) with its
parameter thrown away. The markers of the propagation, parametrisation and
decision modules recover what the shape discards -- when the vector points
where, how fast it moves, where it stalls, what the millimetre criterion
can see of it. These builders produce the data for views that show the same
things to the eye, so that a marker and a picture always read the same
recording the same way. Every builder is a pure function of arrays; drawing
is left to the caller (``render_views`` below uses matplotlib for reports
and inspection; the GUI draws with its own widgets).

The views, and where each comes from:

``time_marked_loop``
    The classical time-mark display, 2 ms dashes along the loop: where the
    marks crowd, the global vector changes slowly (not a local wavefront). The speed used for colouring is the
    smoothed spatial speed the stall, burst and landmark markers read.
``isochronal_crowding``
    The loop cut into equal-time segments, each with its arc length: a
    segment that covers little arc in its time has slow global vector motion,
    not a demonstrated local activation-mapping deceleration zone. Crowding
    is the ratio of time share to arc share.
``spatial_velocity_curve``
    ||dr/dt|| against time using the same algorithmic landmarks as the
    ``qrs_spatial_velocity_*`` markers. Equivalence to manually annotated
    Fareh B-C landmarks has not been validated.
``chronotopocardiogram``
    Direction against time -- azimuth and elevation of the heart vector as
    two curves, magnitude as a third (Titomir & Ruttkay-Nedecky 1987). The
    time-explicit view of the loop: which way the vector points, when.
``arc_length_clock``
    Arc length travelled against time, both normalised; the curve's height
    at half time is ``qrs_arc_length_fraction_at_half_time``.
``criterion_visibility_map``
    The sphere of injury-vector directions coloured by how visible each is
    to the two-contiguous-lead elevation criterion (thesis E1), with the
    recording's ST vector placed on it: the blind spots are posterior,
    rightward and superior, and the anterior sector is dimmed by the
    doubled V2-V3 threshold.
``gradient_axes``
    The QRS area vector, the T area vector and the ventricular gradient
    decomposed on the interventricular (X), apicobasal (Y) and
    anteroposterior (Z) axes -- the three repolarisation gradients that
    reproduce the normal T wave in the equivalent-dipole-layer model
    (Kloosterman et al. 2024), read from the recording's own vectors. The
    gradient is related to action-potential-area heterogeneity under source
    model assumptions (Burger; Draisma et al. 2006). Body-axis projections
    are not separately reconstructed myocardial repolarization gradients.
``discordance_split``
    The ST vector split into an antiparallel QRS-area projection and its
    geometric remainder (thesis E2). Neither component determines whether
    the underlying repolarization change is primary or secondary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .vcg_geometry import vcg_azimuth_deg, vcg_elevation_deg


EPS = 1e-12


def _points(points) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("expected an (n, 3) array of XYZ samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    return values


def smoothed_speed(points, fs: float, sigma: Optional[float] = None) -> np.ndarray:
    """||dr/dt|| in mV/s after the Gaussian smoothing the markers apply.

    One sample shorter than the trajectory. Imported lazily from the
    discovery module so this file stays free of biomarker imports at load.
    """

    from .biomarkers.discovery_v3_biomarkers import _smooth
    from .temporal_sampling import speed_smoothing_sigma_samples

    physical_sigma = speed_smoothing_sigma_samples(fs)
    # An explicit sigma remains a caller-selected sample-domain override.
    sigma = physical_sigma if sigma is None else float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("smoothing sigma must be finite and nonnegative")
    values = _points(points)
    if len(values) < 2:
        return np.zeros(0)
    smoothed = _smooth(values, sigma)
    return np.linalg.norm(np.diff(smoothed, axis=0), axis=1) * float(fs)


# ---------------------------------------------------------------- the loop


@dataclass(frozen=True)
class TimeMarkedLoop:
    points: np.ndarray          # (n, 3)
    times_ms: np.ndarray        # (n,)
    speed: np.ndarray           # (n,) speed at each sample, last one repeated
    mark_indices: np.ndarray    # samples at which a time mark is drawn


def time_marked_loop(points, fs: float, mark_ms: float = 2.0) -> TimeMarkedLoop:
    values = _points(points)
    n = len(values)
    times = np.arange(n) * 1000.0 / float(fs)
    speed = smoothed_speed(values, fs)
    if len(speed) == 0:
        speed_full = np.zeros(n)
    else:
        speed_full = np.concatenate([speed, speed[-1:]])
    step = max(1, int(round(mark_ms * float(fs) / 1000.0)))
    marks = np.arange(0, n, step)
    return TimeMarkedLoop(values, times, speed_full, marks)


@dataclass(frozen=True)
class IsochronalCrowding:
    segment_start_ms: np.ndarray   # (k,)
    segment_end_ms: np.ndarray     # (k,)
    arc_share: np.ndarray          # (k,) share of total arc length per segment
    time_share: np.ndarray         # (k,)
    crowding: np.ndarray           # (k,) time share / arc share; > 1 slow, < 1 fast


def isochronal_crowding(points, fs: float, segments: int = 10) -> IsochronalCrowding:
    values = _points(points)
    n = len(values)
    if n < 2 or segments < 1:
        empty = np.zeros(0)
        return IsochronalCrowding(empty, empty, empty, empty, empty)
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
    total = float(np.sum(steps))
    edges = np.linspace(0, n - 1, segments + 1)
    starts, ends, arcs = [], [], []
    for k in range(segments):
        lo, hi = int(np.floor(edges[k])), int(np.floor(edges[k + 1]))
        hi = max(hi, lo + 1)
        arc = float(np.sum(steps[lo:hi])) if hi <= len(steps) else float(np.sum(steps[lo:]))
        starts.append(lo * 1000.0 / float(fs))
        ends.append(min(hi, n - 1) * 1000.0 / float(fs))
        arcs.append(arc)
    arcs = np.asarray(arcs)
    arc_share = arcs / total if total > EPS else np.full(segments, np.nan)
    time_share = np.full(segments, 1.0 / segments)
    with np.errstate(divide="ignore", invalid="ignore"):
        crowding = np.where(arc_share > EPS, time_share / arc_share, np.inf)
    return IsochronalCrowding(np.asarray(starts), np.asarray(ends), arc_share, time_share, crowding)


@dataclass(frozen=True)
class SpatialVelocityCurve:
    times_ms: np.ndarray
    speed: np.ndarray
    peak_indices: tuple[int, ...]
    trough_indices: tuple[int, ...]
    initial_burst_ms: Optional[float]   # Fareh's B-C: first peak to the trough after it


def spatial_velocity_curve(points, fs: float) -> SpatialVelocityCurve:
    from .biomarkers.loop_parametrisation_biomarkers import landmarks_from_speed

    speed = smoothed_speed(points, fs)
    times = (np.arange(len(speed)) + 0.5) * 1000.0 / float(fs)
    landmarks = landmarks_from_speed(speed) if len(speed) else None
    if landmarks is None:
        return SpatialVelocityCurve(times, speed, (), (), None)
    peaks = tuple(landmarks["peaks"])
    troughs = tuple(landmarks["troughs"])
    burst = (troughs[0] - peaks[0]) * 1000.0 / float(fs) if peaks and troughs else None
    return SpatialVelocityCurve(times, speed, peaks, troughs, burst)


@dataclass(frozen=True)
class Chronotopocardiogram:
    times_ms: np.ndarray
    azimuth_deg: np.ndarray     # nan where the vector is too small for a direction
    elevation_deg: np.ndarray
    magnitude: np.ndarray


def chronotopocardiogram(points, fs: float, minimum_share: float = 0.05) -> Chronotopocardiogram:
    """Direction of the heart vector against time.

    Samples whose magnitude is below ``minimum_share`` of the maximum get no
    direction (nan): a vector that small points nowhere in particular, and
    the curve would otherwise swing wildly at onset and offset.
    """

    values = _points(points)
    n = len(values)
    times = np.arange(n) * 1000.0 / float(fs)
    magnitude = np.linalg.norm(values, axis=1)
    gate = minimum_share * float(np.max(magnitude)) if n else 0.0
    azimuth = np.full(n, np.nan)
    elevation = np.full(n, np.nan)
    for index in range(n):
        if magnitude[index] >= gate and magnitude[index] > EPS:
            az = vcg_azimuth_deg(values[index])
            el = vcg_elevation_deg(values[index])
            azimuth[index] = np.nan if az is None else az
            elevation[index] = np.nan if el is None else el
    return Chronotopocardiogram(times, azimuth, elevation, magnitude)


@dataclass(frozen=True)
class ArcLengthClock:
    time_fraction: np.ndarray
    arc_fraction: np.ndarray
    fraction_at_half_time: Optional[float]


def arc_length_clock(points) -> ArcLengthClock:
    values = _points(points)
    n = len(values)
    if n < 2:
        return ArcLengthClock(np.zeros(0), np.zeros(0), None)
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])
    time_fraction = np.linspace(0.0, 1.0, n)
    if total <= EPS:
        return ArcLengthClock(time_fraction, np.zeros(n), None)
    arc_fraction = cumulative / total
    half = float(np.interp(0.5, time_fraction, arc_fraction))
    return ArcLengthClock(time_fraction, arc_fraction, half)


# --------------------------------------------------------- the criterion


@dataclass(frozen=True)
class VisibilityMap:
    azimuth_deg: np.ndarray      # (a,) grid
    elevation_deg: np.ndarray    # (e,) grid
    visibility: np.ndarray       # (e, a) fraction of the best direction
    st_azimuth_deg: Optional[float]
    st_elevation_deg: Optional[float]
    st_visibility: Optional[float]


def _direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector from the project's angles: azimuth = atan2(Z, X),
    elevation = angle from +Y."""

    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    return np.array([np.sin(el) * np.cos(az), np.cos(el), np.sin(el) * np.sin(az)])


def criterion_visibility_map(st_vector=None, azimuth_steps: int = 72, elevation_steps: int = 37) -> VisibilityMap:
    from .biomarkers.decision_thesis_biomarkers import VISIBILITY_CEILING, criterion_visibility

    azimuths = np.linspace(-180.0, 180.0, azimuth_steps, endpoint=False)
    elevations = np.linspace(0.0, 180.0, elevation_steps)
    grid = np.zeros((elevation_steps, azimuth_steps))
    for i, el in enumerate(elevations):
        for j, az in enumerate(azimuths):
            value = criterion_visibility(_direction(az, el))
            grid[i, j] = 0.0 if value is None else min(value / VISIBILITY_CEILING, 1.0)
    st_az = st_el = st_vis = None
    if st_vector is not None:
        vector = np.asarray(st_vector, dtype=float)
        if vector.shape == (3,) and np.all(np.isfinite(vector)) and np.linalg.norm(vector) > EPS:
            st_az = vcg_azimuth_deg(vector)
            st_el = vcg_elevation_deg(vector)
            value = criterion_visibility(vector)
            st_vis = None if value is None else min(value / VISIBILITY_CEILING, 1.0)
    return VisibilityMap(azimuths, elevations, grid, st_az, st_el, st_vis)


# ------------------------------------------------- action-potential images


@dataclass(frozen=True)
class GradientAxes:
    names: tuple[str, ...]
    components: dict[str, np.ndarray]     # vector name -> (x, y, z) in its own unit
    shares: dict[str, np.ndarray]         # vector name -> squared direction cosines, sum 1
    magnitudes: dict[str, float]


def gradient_axes(**vectors) -> GradientAxes:
    """Decompose named vectors (qrs_area, t_area, svg, ...) on the three body
    axes. Shares are squared direction cosines, so they sum to one and say
    which body-axis direction a vector mostly expresses. They do not
    identify the corresponding local myocardial AP-duration gradients."""

    names = tuple(vectors.keys())
    components: dict[str, np.ndarray] = {}
    shares: dict[str, np.ndarray] = {}
    magnitudes: dict[str, float] = {}
    for name, vector in vectors.items():
        values = np.asarray(vector, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            components[name] = np.full(3, np.nan)
            shares[name] = np.full(3, np.nan)
            magnitudes[name] = float("nan")
            continue
        norm = float(np.linalg.norm(values))
        components[name] = values
        magnitudes[name] = norm
        shares[name] = (values / norm) ** 2 if norm > EPS else np.full(3, np.nan)
    return GradientAxes(names, components, shares, magnitudes)


@dataclass(frozen=True)
class DiscordanceSplit:
    st_magnitude: float
    discordant_mv: float      # the antiparallel projection on the QRS area direction
    beyond_mv: float          # geometric remainder, not proven primary ischemia
    beyond_share: float


def discordance_split(st_vector, qrs_area_vector) -> Optional[DiscordanceSplit]:
    st = np.asarray(st_vector, dtype=float)
    area = np.asarray(qrs_area_vector, dtype=float)
    if st.shape != (3,) or area.shape != (3,) or not np.all(np.isfinite(st)) or not np.all(np.isfinite(area)):
        return None
    norm = float(np.linalg.norm(area))
    if norm <= EPS:
        return None
    projection = float(np.dot(st, area / norm))
    discordant = -min(projection, 0.0)
    magnitude = float(np.linalg.norm(st))
    beyond = float(np.sqrt(max(magnitude**2 - discordant**2, 0.0)))
    share = beyond / magnitude if magnitude > EPS else float("nan")
    return DiscordanceSplit(magnitude, discordant, beyond, share)


# ---------------------------------------------------------------- drawing


def wrapped_angle_plot_data(times, angles):
    """Insert gaps at circular seams without discarding measured samples."""
    times = np.asarray(times, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if times.ndim != 1 or angles.shape != times.shape:
        raise ValueError("times and angles must be equal-length vectors")
    seams = np.flatnonzero(np.abs(np.diff(angles)) > 180.0) + 1
    return np.insert(times, seams, np.nan), np.insert(angles, seams, np.nan)


def render_views(qrs_points, t_points, fs: float, *, st_vector=None, qrs_area=None, t_area=None, svg=None, title: str = ""):
    """A six-panel matplotlib figure of the views for one recording.

    Returns the Figure; the caller saves or embeds it. Text is kept to the
    quantities themselves.
    """

    from matplotlib.figure import Figure

    loop = time_marked_loop(qrs_points, fs)
    crowd = isochronal_crowding(qrs_points, fs)
    curve = spatial_velocity_curve(qrs_points, fs)
    chrono = chronotopocardiogram(qrs_points, fs)
    clock = arc_length_clock(qrs_points)
    vis = criterion_visibility_map(st_vector)

    fig = Figure(figsize=(15, 9))
    axes = fig.subplots(2, 3)
    fig.suptitle(f"{title}\nForschungsansicht: keine lokale Aktivierungskarte oder Therapieentscheidung")

    ax = axes[0, 0]
    x, z = loop.points[:, 0], loop.points[:, 2]
    sc = ax.scatter(x, -z, c=loop.speed, cmap="viridis", s=6)
    ax.plot(x, -z, color="0.6", linewidth=0.6)
    ax.scatter(x[loop.mark_indices], -z[loop.mark_indices], s=14, facecolors="none", edgecolors="black", linewidths=0.6)
    ax.set_xlabel("X (links)")
    ax.set_ylabel("−Z (anterior)")
    ax.set_title("QRS-Loop, Horizontalebene, Zeitmarken alle 2 ms")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=ax, label="Raumgeschwindigkeit (mV/s)")

    ax = axes[0, 1]
    centres = 0.5 * (crowd.segment_start_ms + crowd.segment_end_ms)
    widths = crowd.segment_end_ms - crowd.segment_start_ms
    ax.bar(centres, crowd.crowding, width=widths * 0.9, color="0.4")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("Zeit (ms)")
    ax.set_ylabel("Zeitanteil / Bogenanteil")
    ax.set_title("Isochronen-Dichte (über 1 = langsam)")

    ax = axes[0, 2]
    ax.plot(curve.times_ms, curve.speed, color="black")
    for p in curve.peak_indices:
        ax.plot(curve.times_ms[p], curve.speed[p], "v", color="tab:red")
    for t in curve.trough_indices:
        ax.plot(curve.times_ms[t], curve.speed[t], "^", color="tab:blue")
    label = "Raumgeschwindigkeit; Spitzen und Täler nach Landmarkenregel"
    if curve.initial_burst_ms is not None:
        label += f"\nerstes algorithmisches Intervall {curve.initial_burst_ms:.0f} ms (kein validiertes B–C)"
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("Zeit (ms)")
    ax.set_ylabel("mV/s")

    ax = axes[1, 0]
    ax.plot(*wrapped_angle_plot_data(chrono.times_ms, chrono.azimuth_deg), color="tab:blue", label="Azimut (atan2(Z, X))")
    ax.plot(chrono.times_ms, chrono.elevation_deg, color="tab:orange", label="Elevation (von +Y)")
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Zeit (ms)")
    ax.set_ylabel("Grad")
    ax2 = ax.twinx()
    ax2.plot(chrono.times_ms, chrono.magnitude, color="0.5", linewidth=0.8)
    ax2.set_ylabel("Betrag (mV)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Chronotopokardiogramm: Richtung gegen Zeit")

    ax = axes[1, 1]
    ax.plot(clock.time_fraction, clock.arc_fraction, color="black")
    ax.plot([0, 1], [0, 1], color="0.7", linestyle="--")
    if clock.fraction_at_half_time is not None:
        ax.axvline(0.5, color="tab:red", linewidth=0.8)
        ax.axhline(clock.fraction_at_half_time, color="tab:red", linewidth=0.8)
        ax.set_title(f"Bogenlänge gegen Zeit; bei halber Zeit {clock.fraction_at_half_time:.2f} des Weges")
    ax.set_xlabel("Zeitanteil")
    ax.set_ylabel("Bogenanteil")

    ax = axes[1, 2]
    im = ax.imshow(vis.visibility, origin="lower", extent=(-180, 180, 0, 180), aspect="auto", cmap="magma", vmin=0, vmax=1)
    if vis.st_azimuth_deg is not None:
        ax.plot(vis.st_azimuth_deg, vis.st_elevation_deg, "o", markersize=9, markerfacecolor="white", markeredgecolor="black")
        ax.set_title(f"Fixer Dower-Geometrieproxy; ST-Anteil {vis.st_visibility:.2f}")
    else:
        ax.set_title("Fixer Dower-Geometrieproxy, kein Patienten-Kriterium")
    ax.set_xlabel("Azimut (Grad; 0 = links, ±180 = rechts, +90 = posterior)")
    ax.set_ylabel("Elevation (Grad; 0 = inferior)")
    fig.colorbar(im, ax=ax, label="Anteil der bestmöglichen Richtung")

    fig.tight_layout()
    return fig
