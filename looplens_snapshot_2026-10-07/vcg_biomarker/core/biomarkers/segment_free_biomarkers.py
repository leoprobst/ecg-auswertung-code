"""Markers that do not ask whether a P wave exists.

Every atrial marker in this project is measured inside the P segment, and the
P segment is missing more than half the time. Measured over 11.798 ECGs from
48 partitions spread across the store, the P segment is unusable in 61,0 % of
recordings against 2,6 % for QRS. That alone would only cost power. The
problem is what the loss correlates with:

===========================  ==============  ======
new_onset_af_30d             ohne P-Welle    n
===========================  ==============  ======
negativ                      52,6 %          5.570
**positiv**                  **70,7 %**      314
unbestimmt (bestehendes VHF) 69,5 %          6.116
===========================  ==============  ======

The atrial markers are missing *in the patients with diseased atria*. Every
atrial finding in the study is therefore conditioned on not having the
pathology being looked for -- informative missingness in the textbook sense,
and the likeliest reason ``p_loop_template_instability`` reaches p = 5e-27 for
a gain of ΔAUC +0,0013: the analysis only ever sees the patients whose P wave
is still measurable.

**The fix is a different window, not a weaker threshold.** The interval from
the end of the T wave to the next QRS onset is the atrial window, and it is
defined whether or not a P wave exists. It needs only the two landmarks the
delineator is most confident about. Measured over 322 delineated recordings:

===========  ============
Segment      verfuegbar
===========  ============
qrs          99,1 %
t            91,0 %
**tq**       **85,1 %**
p            37,0 %
===========  ============

**In 50,3 % of all recordings there is a TQ window where there is no P wave.**
Half the cohort gains an atrial measurement it did not have. Seven recordings
out of 322 went the other way. The window runs 390 ms at the median
(p5-p95 109-693 ms), which is a diastole.

In sinus rhythm this window contains the P wave. In fibrillation it contains
the fibrillatory waves -- which is the point: the markers below become
measurable *because* there is no P wave, rather than despite it.

**On the whole strip.** The plan's other window is the full ten seconds, for
complexity and spectral measures. One trap decides whether those markers mean
anything: a path length or a count over ten seconds is a measurement of heart
rate wearing another name. Everything strip-scoped here is normalised per
beat or per second and says so in its formula.

**Measured against fibrillation on the tracing itself.** 1.377 recordings drawn
balanced on the machine report, 683 with atrial fibrillation or flutter. The
missingness is more extreme than the cohort-wide figure suggested:

=========================  ==========
P segment measurable in    Anteil
=========================  ==========
ECGs **with** fibrillation **3,7 %**
ECGs without               48,6 %
=========================  ==========

A thirteen-fold difference. The existing atrial markers are, for practical
purposes, blind to atrial fibrillation.

Among the 1.015 recordings with **no measurable P wave** -- where every atrial
marker in this project currently returns nothing -- all ten separate
fibrillation from sinus:

==================================  =====  ==========
Marker (ohne P-Welle, 658 positiv)  AUC    Richtung
==================================  =====  ==========
tq_atrial_peak_to_peak              0,712  niedriger
tq_atrial_rms                       0,695  niedriger
tq_window_cycle_fraction            0,685  kuerzer
tq_organization_index               0,679  niedriger
tq_atrial_loop_area                 0,677  kleiner
tq_dominant_frequency_hz            0,674  hoeher
tq_spectral_concentration           0,628  niedriger
tq_atrial_direction_dispersion      0,624  niedriger
tq_atrial_to_ventricular_amplitude  0,622  niedriger
tq_spectral_entropy                 0,600  hoeher
==================================  =====  ==========

Every direction is the physiological one: fibrillatory waves are smaller, less
organised and faster than a P wave. ``tq_window_cycle_fraction`` is the
exception that proves it is not measuring morphology -- a shorter atrial window
in fibrillation is the faster ventricular rate, which is why it is documented
as the exposure rather than as an atrial marker.

An earlier draft of this table, taken on 175 recordings, put the spectral
markers at 0,55-0,60 and looked like evidence that a 390 ms window cannot carry
a spectrum. It was not: the spectrum was being taken of ``||r(t)||``, which
rectifies the signal and doubled every reported frequency. Both the sample size
and the defect had to be fixed before the numbers meant anything, and that is
the reason the sample here is eight times larger.

**Two limitations, both real.** The reference is the ECG machine's own report,
which reads the same tracing -- the circularity this project measured
elsewhere, where PTB-XL's cardiologist labels gave AUC 0,891 against 0,68 for
an independent echo reference. These numbers are therefore an upper bound on
agreement, not a diagnostic performance. And a machine report is not an
adjudicated diagnosis.

Nothing here replaces a P-segment marker. The P versions keep running exactly
as before; a reader comparing the two learns what the restriction to
measurable P waves was costing. The redundancy stage will say afterwards which
of these add nothing -- as it already did for the ``beat_``/``qrs_``/``qrst_``
kinematics, where eight groups turned out to differ only in the segment name.
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


SEGMENT_FREE_VERSION = "1.2.0"
SEGMENT_FREE_SOURCE = "segment_free_v1_1"

EPS = 1e-12

# The atrial window. Named once so no marker can quietly measure a different
# one.
ATRIAL_WINDOW = "tq"

# Fibrillatory frequency band. Atrial fibrillation runs 4-10 Hz and flutter
# 4-7 Hz (240-420 min^-1); the band is opened to 3-12 Hz so that a slow
# flutter and a fast fibrillation both fall inside it rather than being
# clipped at the edge, which would put every extreme case at the same value.
FIBRILLATORY_BAND_HZ = (3.0, 12.0)

# Half-width of the band counted as "at the dominant frequency", for the
# organisation measures. 0.75 Hz is the usual choice and is roughly the
# spectral resolution of a single 390 ms window, so a narrower band would be
# measuring the window length.
DOMINANT_BAND_HALF_WIDTH_HZ = 0.75

# A TQ window shorter than this cannot carry a fibrillatory cycle: at 3 Hz one
# cycle is 333 ms, and a spectrum from less than one cycle is not a spectrum.
# Windows below the limit abstain rather than return a number.
MINIMUM_SPECTRAL_WINDOW_MS = 120.0


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_segment_free_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_segment_free_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _finite_xyz(value: object, minimum: int = 3) -> Optional[np.ndarray]:
    if value is None:
        return None
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    points = points[np.all(np.isfinite(points), axis=1)]
    return points if len(points) >= minimum else None


def _atrial_window(ctx: FeatureContext) -> Optional[np.ndarray]:
    """The vector trajectory between the T wave and the next QRS."""

    return _memo(
        ctx,
        ("atrial",),
        lambda: _finite_xyz(ctx.vcg_segment_origin_corrected(ATRIAL_WINDOW)),
    )


def _magnitude(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points, axis=1)


def _detrended(values: np.ndarray) -> np.ndarray:
    """Remove the linear baseline before any spectral statement.

    The TQ window sits on whatever baseline wander survived filtering, and a
    drift across 390 ms is a large low-frequency component that would
    dominate every spectrum below. Removing a straight line is the least that
    makes the remaining content interpretable.
    """

    n = len(values)
    if n < 3:
        return values - float(np.mean(values))
    index = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(index, values, 1)
    return values - (slope * index + intercept)


def _atrial_scalar(ctx: FeatureContext) -> Optional[np.ndarray]:
    """A signed scalar lead through the atrial trajectory.

    Every spectral marker here reads this rather than ``||r(t)||``, and the
    difference is not cosmetic. The magnitude of a vector oscillating about the
    origin is a *rectified* signal, so its fundamental sits at twice the
    oscillation frequency. Measured on synthetic windows before this was
    fixed, the marker returned 8,06 Hz for a 4 Hz input, 10,01 for 5 and 11,96
    for 6 -- a clean factor of two that would have reported every fibrillatory
    frequency at double its value, inside a plausible-looking range.

    The projection onto the trajectory's own dominant direction keeps the sign
    and is the standard construction of a scalar lead from a vector loop.
    """

    def build() -> Optional[np.ndarray]:
        points = _atrial_window(ctx)
        if points is None:
            return None
        centred = points - points.mean(axis=0)
        try:
            _, _, basis = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        return centred @ basis[0]

    return _memo(ctx, ("atrial_scalar",), build)


def _spectrum(
    ctx: FeatureContext, values: np.ndarray
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Frequencies and power of the atrial window, or nothing.

    Zero-padded to four times the window so the dominant frequency is not
    quantised to the handful of bins a 390 ms window would otherwise provide.
    Padding does not add information; it interpolates the spectrum that is
    already there, which is what a peak location needs.
    """

    fs = float(getattr(ctx, "fs", 0) or 0)
    n = len(values)
    if fs <= 0 or n < 8:
        return None
    if (n / fs) * 1000.0 < MINIMUM_SPECTRAL_WINDOW_MS:
        return None
    centred = _detrended(np.asarray(values, dtype=float))
    window = np.hanning(n)
    padded = int(2 ** int(np.ceil(np.log2(max(n * 4, 64)))))
    spectrum = np.fft.rfft(centred * window, n=padded)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(padded, d=1.0 / fs)
    return freqs, power


def _band_mask(freqs: np.ndarray) -> np.ndarray:
    low, high = FIBRILLATORY_BAND_HZ
    return (freqs >= low) & (freqs <= high)


def _dominant(
    ctx: FeatureContext,
) -> Optional[tuple[float, np.ndarray, np.ndarray]]:
    """Dominant fibrillatory frequency, with the spectrum it came from."""

    def build():
        scalar = _atrial_scalar(ctx)
        if scalar is None:
            return None
        spectrum = _spectrum(ctx, scalar)
        if spectrum is None:
            return None
        freqs, power = spectrum
        mask = _band_mask(freqs)
        if not np.any(mask) or float(np.sum(power[mask])) <= EPS:
            return None
        peak = float(freqs[mask][int(np.argmax(power[mask]))])
        return peak, freqs, power

    return _memo(ctx, ("dominant",), build)


# --------------------------------------------------------- the atrial window


def _atrial_rms(ctx: FeatureContext) -> FeatureValue:
    points = _atrial_window(ctx)
    if points is None:
        return _abstain("no TQ window")
    magnitude = _magnitude(points)
    value = float(np.sqrt(np.mean(_detrended(magnitude) ** 2)))
    if not np.isfinite(value):
        return _abstain("non-finite amplitude")
    return FeatureValue(value)


def _atrial_peak_to_peak(ctx: FeatureContext) -> FeatureValue:
    points = _atrial_window(ctx)
    if points is None:
        return _abstain("no TQ window")
    magnitude = _detrended(_magnitude(points))
    value = float(np.max(magnitude) - np.min(magnitude))
    return FeatureValue(value) if np.isfinite(value) else _abstain("non-finite")


def _dominant_frequency(ctx: FeatureContext) -> FeatureValue:
    resolved = _dominant(ctx)
    if resolved is None:
        return _abstain("no usable atrial spectrum")
    return FeatureValue(float(resolved[0]))


def _spectral_concentration(ctx: FeatureContext) -> FeatureValue:
    """Share of band power within +-0,75 Hz of the dominant frequency.

    Organised flutter puts nearly all its power in one peak; disorganised
    fibrillation spreads it. This is the measure that separates them, and it
    needs no P wave at all.
    """

    resolved = _dominant(ctx)
    if resolved is None:
        return _abstain("no usable atrial spectrum")
    peak, freqs, power = resolved
    band = _band_mask(freqs)
    total = float(np.sum(power[band]))
    if total <= EPS:
        return _abstain("no band power")
    near = band & (np.abs(freqs - peak) <= DOMINANT_BAND_HALF_WIDTH_HZ)
    return FeatureValue(float(np.sum(power[near]) / total))


def _organization_index(ctx: FeatureContext) -> FeatureValue:
    """Power at the dominant frequency and its harmonics, over band power.

    The established organisation measure for atrial activity. A regular
    rhythm concentrates power at one frequency and its multiples; a
    disorganised one does not.
    """

    resolved = _dominant(ctx)
    if resolved is None:
        return _abstain("no usable atrial spectrum")
    peak, freqs, power = resolved
    band = _band_mask(freqs)
    total = float(np.sum(power[band]))
    if total <= EPS:
        return _abstain("no band power")
    organised = np.zeros_like(band)
    for harmonic in (1, 2, 3):
        centre = peak * harmonic
        if centre > freqs[-1]:
            break
        organised |= np.abs(freqs - centre) <= DOMINANT_BAND_HALF_WIDTH_HZ
    return FeatureValue(float(np.sum(power[organised & band]) / total))


def _spectral_entropy(ctx: FeatureContext) -> FeatureValue:
    """Normalised Shannon entropy of the atrial band, 0 to 1.

    One clean frequency gives 0; white noise across the band gives 1. Reported
    beside the concentration because the two disagree in an informative way:
    two separate peaks are concentrated at neither and orderly at both.
    """

    resolved = _dominant(ctx)
    if resolved is None:
        return _abstain("no usable atrial spectrum")
    _, freqs, power = resolved
    band = power[_band_mask(freqs)]
    total = float(np.sum(band))
    if total <= EPS or len(band) < 2:
        return _abstain("no band power")
    density = band / total
    density = density[density > EPS]
    entropy = -float(np.sum(density * np.log(density)))
    return FeatureValue(entropy / float(np.log(len(band))))


def _atrial_loop_area(ctx: FeatureContext) -> FeatureValue:
    """Area the atrial vector sweeps, by the shoelace formula in its own plane.

    Zero for a flat baseline, small and reproducible for a sinus P loop, and
    neither for fibrillatory activity.
    """

    points = _atrial_window(ctx)
    if points is None:
        return _abstain("no TQ window")
    centred = points - points.mean(axis=0)
    try:
        _, _, basis = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return _abstain("degenerate atrial loop")
    projected = centred @ basis[:2].T
    x, y = projected[:, 0], projected[:, 1]
    area = 0.5 * float(np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    return FeatureValue(area) if np.isfinite(area) else _abstain("non-finite area")


def _atrial_direction_dispersion(ctx: FeatureContext) -> FeatureValue:
    """How far the atrial vector direction wanders, in radians.

    A sinus P wave travels one way; fibrillatory activity does not settle on a
    direction. Measured as the mean angle to the resultant direction, so it is
    scale-free and does not inherit the amplitude.
    """

    points = _atrial_window(ctx)
    if points is None:
        return _abstain("no TQ window")
    norms = np.linalg.norm(points, axis=1)
    usable = norms > EPS
    if int(np.sum(usable)) < 3:
        return _abstain("atrial signal too small to have a direction")
    unit = points[usable] / norms[usable, None]
    resultant = unit.mean(axis=0)
    length = float(np.linalg.norm(resultant))
    if length <= EPS:
        return _abstain("no resultant direction")
    cosines = np.clip(unit @ (resultant / length), -1.0, 1.0)
    return FeatureValue(float(np.mean(np.arccos(cosines))))


def _atrial_to_ventricular_amplitude(ctx: FeatureContext) -> FeatureValue:
    """Atrial amplitude as a fraction of the QRS amplitude.

    The ratio rather than the amplitude, because the absolute atrial voltage
    carries the patient's build, electrode placement and the transform's gain.
    Dividing by that patient's own QRS removes all three at once.
    """

    points = _atrial_window(ctx)
    qrs = _finite_xyz(ctx.vcg_segment_origin_corrected("qrs"))
    if points is None:
        return _abstain("no TQ window")
    if qrs is None:
        return _abstain("no QRS to normalise against")
    atrial = float(np.sqrt(np.mean(_detrended(_magnitude(points)) ** 2)))
    ventricular = float(np.max(_magnitude(qrs)))
    if not np.isfinite(ventricular) or ventricular <= EPS:
        return _abstain("QRS amplitude not usable")
    return FeatureValue(atrial / ventricular)


def _atrial_window_fraction(ctx: FeatureContext) -> FeatureValue:
    """The share of the cardiac cycle that is atrial window.

    Not an atrial marker in itself -- it is the exposure the others are
    measured over, and it belongs in the table beside them. A short window
    means a fast rate or a long QT, and both change what the spectral markers
    can resolve.
    """

    bounds = ctx.bounds(ATRIAL_WINDOW)
    fs = float(getattr(ctx, "fs", 0) or 0)
    if bounds is None or fs <= 0:
        return _abstain("no TQ window")
    beats = getattr(ctx.delineation, "beats", None) or ()
    peaks = sorted(
        int(getattr(beat, "r_peak").sample)
        for beat in beats
        if getattr(beat, "r_peak", None) is not None
    )
    if len(peaks) < 2:
        return _abstain("fewer than two beats")
    start, end = bounds
    # The closing QRS onset may skip a low-confidence beat.  Dividing that
    # multi-cycle window by the median RR produced values up to 1.89 while the
    # marker was declared a fraction.  Use the RR interval that actually
    # contains this TQ window, and abstain when the delineation cannot bracket
    # it consistently.
    closing_index = next(
        (index for index, peak in enumerate(peaks) if peak >= end),
        None,
    )
    if closing_index is None or closing_index == 0:
        return _abstain("TQ window is not bracketed by consecutive R peaks")
    previous_peak = peaks[closing_index - 1]
    closing_peak = peaks[closing_index]
    if start < previous_peak or end > closing_peak:
        return _abstain("TQ bounds cross the bracketing cardiac cycle")
    cycle = float(closing_peak - previous_peak)
    if cycle <= EPS:
        return _abstain("degenerate cycle length")
    fraction = float((end - start) / cycle)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        return _abstain("TQ duration is not a fraction of its cardiac cycle")
    return FeatureValue(fraction)


# ------------------------------------------------------------- the whole strip


def _strip_path_length_per_second(ctx: FeatureContext) -> FeatureValue:
    """Vector path travelled per second over the whole recording.

    Per second, not per record. The unnormalised version of this measure is
    the trap the plan names: over a fixed ten seconds a faster heart traces a
    longer path, so the marker would be heart rate under another name.
    """

    points = _finite_xyz(ctx.vcg_segment("global_vcg"))
    fs = float(getattr(ctx, "fs", 0) or 0)
    if points is None or fs <= 0:
        return _abstain("no record trajectory")
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    seconds = len(points) / fs
    if seconds <= EPS:
        return _abstain("empty record")
    return FeatureValue(length / seconds)


def _strip_path_length_per_beat(ctx: FeatureContext) -> FeatureValue:
    """Vector path travelled per cardiac cycle.

    The same quantity divided by beats instead of seconds, which is the
    normalisation that makes it a statement about one beat's morphology rather
    than about the rate. Both are kept: they answer different questions and
    the redundancy stage can say whether the data agrees.
    """

    points = _finite_xyz(ctx.vcg_segment("global_vcg"))
    if points is None:
        return _abstain("no record trajectory")
    beats = getattr(ctx.delineation, "beats", None) or ()
    count = sum(1 for beat in beats if getattr(beat, "r_peak", None) is not None)
    if count < 1:
        return _abstain("no beats")
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    return FeatureValue(length / float(count))


def _strip_amplitude_dispersion(ctx: FeatureContext) -> FeatureValue:
    """Beat-to-beat spread of the peak vector magnitude, as a coefficient.

    Scale-free by construction, so it survives the amplitude differences the
    absolute measures inherit. High in alternans and in an irregularly
    conducted rhythm; low in a stable one.
    """

    points = _finite_xyz(ctx.vcg_segment("global_vcg"))
    if points is None:
        return _abstain("no record trajectory")
    beats = getattr(ctx.delineation, "beats", None) or ()
    peaks = [
        int(getattr(beat, "r_peak").sample)
        for beat in beats
        if getattr(beat, "r_peak", None) is not None
    ]
    peaks = sorted(p for p in peaks if 0 <= p < len(points))
    if len(peaks) < 3:
        return _abstain("fewer than three beats")
    magnitudes = np.linalg.norm(points[peaks], axis=1)
    mean = float(np.mean(magnitudes))
    if not np.isfinite(mean) or abs(mean) <= EPS:
        return _abstain("degenerate amplitude")
    return FeatureValue(float(np.std(magnitudes) / mean))


@dataclass(frozen=True)
class SegmentFreeSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    calculator: Callable[[FeatureContext], FeatureValue]
    window: str
    evidence_note: str
    references: tuple[str, ...] = ()


SEGMENT_FREE_SPECS: tuple[SegmentFreeSpec, ...] = (
    SegmentFreeSpec(
        "tq_atrial_rms",
        "Vorhoffenster",
        "Root mean square of the detrended vector magnitude in the TQ window.",
        "sqrt(mean(detrend(||r(t)||)^2)) over [T end, next QRS onset].",
        "mV",
        _atrial_rms,
        ATRIAL_WINDOW,
        "Atrial signal strength without asking whether a P wave was found. "
        "Available in 85,1 % of recordings against 37,0 % for the P segment.",
    ),
    SegmentFreeSpec(
        "tq_atrial_peak_to_peak",
        "Vorhoffenster",
        "Peak-to-peak excursion of the detrended atrial vector magnitude.",
        "max(detrend(||r||)) - min(detrend(||r||)) over the TQ window.",
        "mV",
        _atrial_peak_to_peak,
        ATRIAL_WINDOW,
        "The amplitude a reader would measure with calipers on the TP segment.",
    ),
    SegmentFreeSpec(
        "tq_dominant_frequency_hz",
        "Vorhoffenster",
        "Dominant frequency of atrial activity between 3 and 12 Hz.",
        "argmax of the Hanning-windowed periodogram of the signed principal-axis projection, within the band.",
        "Hz",
        _dominant_frequency,
        ATRIAL_WINDOW,
        "The fibrillatory frequency, classically 4-10 Hz in atrial "
        "fibrillation and 4-7 Hz in flutter. Measurable exactly where the P "
        "wave is absent.",
    ),
    SegmentFreeSpec(
        "tq_spectral_concentration",
        "Vorhoffenster",
        "Share of atrial band power within +-0,75 Hz of the dominant frequency.",
        "sum(P[|f - f_dom| <= 0.75]) / sum(P[3 <= f <= 12]).",
        "fraction",
        _spectral_concentration,
        ATRIAL_WINDOW,
        "Organised flutter concentrates its power in one peak; fibrillation "
        "spreads it.",
    ),
    SegmentFreeSpec(
        "tq_organization_index",
        "Vorhoffenster",
        "Band power at the dominant frequency and its first two harmonics.",
        "sum(P near f_dom, 2*f_dom, 3*f_dom) / sum(P in band).",
        "fraction",
        _organization_index,
        ATRIAL_WINDOW,
        "The established organisation measure for atrial activity.",
    ),
    SegmentFreeSpec(
        "tq_spectral_entropy",
        "Vorhoffenster",
        "Normalised Shannon entropy of the atrial band spectrum.",
        "-sum(p log p) / log(n) over the normalised band power.",
        "fraction",
        _spectral_entropy,
        ATRIAL_WINDOW,
        "Zero for one clean frequency, one for noise across the band. "
        "Disagrees informatively with the concentration when there are two "
        "peaks.",
    ),
    SegmentFreeSpec(
        "tq_atrial_loop_area",
        "Vorhoffenster",
        "Area swept by the atrial vector in its own best-fit plane.",
        "Shoelace area of the TQ trajectory projected on its first two "
        "principal axes.",
        "mV^2",
        _atrial_loop_area,
        ATRIAL_WINDOW,
        "Zero for a flat baseline, small and reproducible for a sinus P loop, "
        "neither for fibrillatory activity.",
    ),
    SegmentFreeSpec(
        "tq_atrial_direction_dispersion",
        "Vorhoffenster",
        "Mean angle between the atrial direction and its resultant.",
        "mean(arccos(u(t) . R / ||R||)) with u the unit vectors and R their "
        "mean.",
        "rad",
        _atrial_direction_dispersion,
        ATRIAL_WINDOW,
        "A sinus P wave travels one way; fibrillatory activity settles on no "
        "direction. Scale-free, so it does not inherit the amplitude.",
    ),
    SegmentFreeSpec(
        "tq_atrial_to_ventricular_amplitude",
        "Vorhoffenster",
        "Atrial RMS amplitude as a fraction of the peak QRS magnitude.",
        "RMS(detrend(||r||) in TQ) / max(||r|| in QRS).",
        "ratio",
        _atrial_to_ventricular_amplitude,
        ATRIAL_WINDOW,
        "Removes build, electrode placement and transform gain by dividing by "
        "the same patient's own ventricular amplitude.",
    ),
    SegmentFreeSpec(
        "tq_window_cycle_fraction",
        "Vorhoffenster",
        "Length of the TQ window as a fraction of its bracketing RR interval.",
        "(TQ end - TQ start) / (closing R peak - preceding R peak).",
        "fraction",
        _atrial_window_fraction,
        ATRIAL_WINDOW,
        "Not an atrial marker but the exposure the others are measured over: "
        "a short window limits what the spectral markers can resolve.",
    ),
    SegmentFreeSpec(
        "strip_path_length_per_second",
        "Ganzer Streifen",
        "Vector path length travelled per second over the whole recording.",
        "sum(||dr||) / (n / fs) over the full record.",
        "mV/s",
        _strip_path_length_per_second,
        "global_vcg",
        "Normalised per second. The unnormalised version over a fixed ten "
        "seconds would be a measurement of heart rate under another name.",
    ),
    SegmentFreeSpec(
        "strip_path_length_per_beat",
        "Ganzer Streifen",
        "Vector path length travelled per cardiac cycle.",
        "sum(||dr||) / number of detected beats.",
        "mV",
        _strip_path_length_per_beat,
        "global_vcg",
        "The per-beat normalisation makes this a statement about morphology "
        "rather than about rate. Kept beside the per-second version because "
        "they answer different questions.",
    ),
    SegmentFreeSpec(
        "strip_amplitude_dispersion",
        "Ganzer Streifen",
        "Coefficient of variation of the peak vector magnitude across beats.",
        "std(||r|| at R peaks) / mean(||r|| at R peaks).",
        # A coefficient of variation is scale-free but not bounded by one.
        # Values above one are possible when the beat-amplitude distribution
        # is very dispersed; calling it a fraction made valid values fail the
        # plausibility audit.
        "ratio",
        _strip_amplitude_dispersion,
        "global_vcg",
        "Scale-free, so it survives the amplitude differences the absolute "
        "measures inherit. High in alternans and irregular conduction.",
    ),
)


class SegmentFreeBiomarker(BiomarkerBase):
    def __init__(self, spec: SegmentFreeSpec) -> None:
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
        notes = [
            f"segment_free_version={SEGMENT_FREE_VERSION}",
            f"window={self._spec.window}",
        ]
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


def atrial_window_marker_ids() -> tuple[str, ...]:
    """The markers that measure the atrium without needing a P wave."""

    return tuple(
        spec.name for spec in SEGMENT_FREE_SPECS if spec.window == ATRIAL_WINDOW
    )


def register_segment_free_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in SEGMENT_FREE_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), SegmentFreeBiomarker):
                continue
            raise RuntimeError(
                f"Segment-free marker collides with an existing biomarker: "
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
                source=SEGMENT_FREE_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Prospective hypothesis measured in a window that does not "
                    "require a P wave. " + spec.evidence_note +
                    " No diagnostic cutoff is claimed."
                ),
                references=list(spec.references),
                implementation_status="implemented",
            )
        )
        BiomarkerRegistry.register(SegmentFreeBiomarker(spec))


__all__ = [
    "ATRIAL_WINDOW",
    "DOMINANT_BAND_HALF_WIDTH_HZ",
    "FIBRILLATORY_BAND_HZ",
    "MINIMUM_SPECTRAL_WINDOW_MS",
    "SEGMENT_FREE_SOURCE",
    "SEGMENT_FREE_SPECS",
    "SEGMENT_FREE_VERSION",
    "SegmentFreeBiomarker",
    "SegmentFreeSpec",
    "atrial_window_marker_ids",
    "register_segment_free_biomarkers",
]
