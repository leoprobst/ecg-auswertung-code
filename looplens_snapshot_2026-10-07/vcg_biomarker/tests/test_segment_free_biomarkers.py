"""The atrial markers are missing in the patients with diseased atria.

Measured over 11.798 ECGs, the P segment is unusable in 61,0 % of recordings.
That alone would only cost power. What makes it a defect is the correlation:
among 175 recordings drawn balanced on the machine report, the P segment was
measurable in 50,0 % of those without fibrillation and **3,4 %** of those with
it. Every atrial finding in this project is therefore conditioned on not having
the pathology it is looking for.

The TQ window -- T end to the next QRS onset -- is defined whether or not a P
wave exists. Measured over 322 delineated recordings it is available in 85,1 %
against 37,0 % for P, and in **50,3 % of all recordings there is a TQ window
where there is no P wave**.

One defect found while building this and pinned below: registering thirteen
markers displaced eight from the sentinel panel, ``p_loop_linearity`` among
them. The curated panels have fixed sizes and are filled by a family-balanced
round robin, so admitting a marker to the candidate pool pushes another one
out. Adding atrial markers had removed an atrial marker.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.segment_free_biomarkers import (
    ATRIAL_WINDOW,
    FIBRILLATORY_BAND_HZ,
    MINIMUM_SPECTRAL_WINDOW_MS,
    SEGMENT_FREE_SPECS,
    atrial_window_marker_ids,
)
from vcg_biomarker.mimic.feature_roles import feature_profile
from vcg_biomarker.mimic.panels import default_feature_panels


BY_NAME = {spec.name: spec for spec in SEGMENT_FREE_SPECS}


class _Point:
    def __init__(self, sample, confidence=0.9):
        self.sample = int(sample)
        self.confidence = float(confidence)


class _Beat:
    """Enough of a beat for the window arithmetic, and nothing more."""

    def __init__(self, r_onset, r_peak):
        self.r_onset = _Point(r_onset)
        self.r_peak = _Point(r_peak)


class _Delineation:
    def __init__(self, beats, evaluable=True):
        self.beats = beats
        self._evaluable = evaluable

    def is_segment_evaluable(self, segment, minimum_confidence=None):
        return self._evaluable


class _Context:
    """A context that answers only what the segment-free markers ask of it."""

    def __init__(self, points, bounds, fs=500, beats=(), qrs=None):
        self._points = np.asarray(points, dtype=float)
        self._bounds = bounds
        self.fs = fs
        self.delineation = _Delineation(list(beats))
        self._qrs = qrs

    def bounds(self, segment):
        return self._bounds if segment == ATRIAL_WINDOW else None

    def vcg_segment_origin_corrected(self, segment):
        if segment == ATRIAL_WINDOW:
            if self._bounds is None:
                return None
            start, end = self._bounds
            return self._points[start:end + 1]
        if segment == "qrs":
            return self._qrs
        return None

    def vcg_segment(self, segment):
        return self._points if segment == "global_vcg" else None


def _sine_window(frequency_hz, samples=200, fs=500, amplitude=0.05, seed=0):
    """A synthetic atrial window oscillating at a known frequency."""

    rng = np.random.default_rng(seed)
    t = np.arange(samples) / fs
    wave = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    noise = rng.normal(0, amplitude / 50.0, (samples, 3))
    points = np.zeros((samples, 3))
    points[:, 0] = wave
    return points + noise


def _context_for(points, fs=500, beats=None, qrs=None):
    n = len(points)
    if beats is None:
        beats = [_Beat(0, 5), _Beat(n - 1, n - 5)]
    return _Context(points, (0, n - 1), fs=fs, beats=beats, qrs=qrs)


def _value(name, ctx):
    return BY_NAME[name].calculator(ctx).value


# --- The window itself --------------------------------------------------------


def test_the_atrial_window_is_the_tq_interval():
    """Named once, so no marker can quietly measure a different window."""

    assert ATRIAL_WINDOW == "tq"
    for name in atrial_window_marker_ids():
        assert BY_NAME[name].window == "tq"


def test_the_tq_window_runs_from_the_t_end_to_the_next_qrs():
    import inspect

    from vcg_biomarker.core.biomarkers.catalog_biomarkers import FeatureContext

    source = inspect.getsource(FeatureContext.bounds)
    assert 'segment == "tq"' in source
    assert "_next_qrs_onset" in source


def test_the_next_qrs_is_taken_from_the_records_own_beats():
    """The consensus beat carries one set of fiducials; the window ends at the
    next beat, which only the record's beat list knows about."""

    from vcg_biomarker.core.biomarkers.catalog_biomarkers import FeatureContext
    import inspect

    source = inspect.getsource(FeatureContext._next_qrs_onset)
    assert "self.delineation" in source
    assert "beats" in source


def test_the_delineation_declares_what_the_tq_window_needs():
    from vcg_biomarker.core.delineation.base import BeatDelineation
    import inspect

    source = inspect.getsource(BeatDelineation.segment_confidence)
    assert '"tq": ("t_offset", "r_onset")' in source


# --- Abstention ---------------------------------------------------------------


class _EmptyContext(_Context):
    """A recording that yields no trajectory at all, for either window."""

    def vcg_segment(self, segment):
        return None


@pytest.mark.parametrize("name", [spec.name for spec in SEGMENT_FREE_SPECS])
def test_a_marker_without_its_window_abstains_rather_than_returning_zero(name):
    ctx = _EmptyContext(np.zeros((10, 3)), None, beats=[])
    result = BY_NAME[name].calculator(ctx)
    assert result.value is None
    assert result.confidence == 0.0
    assert "not_evaluable" in result.notes


def test_a_window_too_short_for_one_cycle_abstains():
    """At 3 Hz one cycle is 333 ms; a spectrum from less is not a spectrum."""

    samples = int(MINIMUM_SPECTRAL_WINDOW_MS / 1000.0 * 500) - 10
    ctx = _context_for(_sine_window(6.0, samples=samples))
    assert _value("tq_dominant_frequency_hz", ctx) is None
    # The amplitude markers do not need a spectrum and must still answer.
    assert _value("tq_atrial_rms", ctx) is not None


# --- The spectral markers against a known frequency ---------------------------


@pytest.mark.parametrize("frequency", (3.5, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0))
def test_a_known_fibrillatory_frequency_is_recovered(frequency):
    """The defect this caught: every frequency came back doubled.

    The spectrum was taken of ``||r(t)||``, and the magnitude of a vector
    oscillating about the origin is a rectified signal whose fundamental sits
    at twice the oscillation. Measured before the fix: 8,06 Hz for a 4 Hz
    input, 10,01 for 5, 11,96 for 6. Every value landed inside the
    fibrillatory band, so nothing would have looked wrong -- the marker would
    simply have reported every atrial frequency at double its value.
    """

    ctx = _context_for(_sine_window(frequency, samples=400))
    measured = _value("tq_dominant_frequency_hz", ctx)
    assert measured == pytest.approx(frequency, abs=0.3)


def test_the_spectrum_is_taken_of_a_signed_signal():
    """A magnitude cannot carry the sign, and without the sign the frequency
    doubles."""

    import inspect

    from vcg_biomarker.core.biomarkers import segment_free_biomarkers

    source = inspect.getsource(segment_free_biomarkers._dominant)
    assert "_atrial_scalar" in source
    assert "_magnitude" not in source


def test_a_frequency_outside_the_band_is_not_reported_as_dominant():
    """A 1 Hz baseline drift must not be called a fibrillatory frequency."""

    ctx = _context_for(_sine_window(1.0, samples=400, amplitude=0.2))
    measured = _value("tq_dominant_frequency_hz", ctx)
    low, high = FIBRILLATORY_BAND_HZ
    assert measured is None or low <= measured <= high


def test_a_clean_frequency_is_more_concentrated_than_noise():
    """Stated as a comparison rather than a threshold.

    The absolute values depend on the zero padding -- a padded peak spreads
    over several bins, so a pure tone scores 0,69 on the entropy rather than
    something near zero. Fixing a magic number to that would be testing the
    padding. Measured: concentration 0,752 against 0,374, entropy 0,692
    against 0,920.
    """

    rng = np.random.default_rng(3)
    clean = _context_for(_sine_window(6.0, samples=400))
    noise = _context_for(rng.normal(0, 0.05, (400, 3)))

    assert (
        _value("tq_spectral_concentration", clean)
        > 1.5 * _value("tq_spectral_concentration", noise)
    )
    assert (
        _value("tq_organization_index", clean)
        > 1.5 * _value("tq_organization_index", noise)
    )
    assert _value("tq_spectral_entropy", clean) < _value(
        "tq_spectral_entropy", noise
    )


def test_the_organisation_index_sees_harmonics_the_concentration_misses():
    """The two agree exactly on a pure tone; only a harmonic series parts them.

    Measured on the pure 6 Hz window both return 0,752 -- correct, since there
    is nothing at 12 or 18 Hz to find. A signal with harmonics must separate
    them, or the second marker is a copy of the first under another name.
    """

    fs = 500
    t = np.arange(400) / fs
    harmonic = np.zeros((400, 3))
    # 4 Hz fundamental with a strong second harmonic at 8 Hz, both in band.
    harmonic[:, 0] = 0.05 * np.sin(2 * np.pi * 4.0 * t) + 0.04 * np.sin(
        2 * np.pi * 8.0 * t
    )
    ctx = _context_for(harmonic)
    assert _value("tq_organization_index", ctx) > _value(
        "tq_spectral_concentration", ctx
    )


def test_a_linear_drift_does_not_become_atrial_amplitude():
    """The TQ window sits on whatever baseline wander survived filtering."""

    ramp = np.zeros((300, 3))
    ramp[:, 0] = np.linspace(0.0, 1.0, 300)
    ctx = _context_for(ramp)
    assert _value("tq_atrial_rms", ctx) == pytest.approx(0.0, abs=1e-9)


# --- The amplitude markers ----------------------------------------------------


def test_a_larger_atrial_signal_gives_a_larger_amplitude():
    small = _context_for(_sine_window(6.0, samples=300, amplitude=0.02))
    large = _context_for(_sine_window(6.0, samples=300, amplitude=0.20))
    assert _value("tq_atrial_rms", small) < _value("tq_atrial_rms", large)
    assert (
        _value("tq_atrial_peak_to_peak", small)
        < _value("tq_atrial_peak_to_peak", large)
    )


def test_the_atrial_to_ventricular_ratio_divides_out_the_gain():
    """Doubling every voltage must not change a ratio of two voltages."""

    points = _sine_window(6.0, samples=300, amplitude=0.05)
    qrs = np.zeros((60, 3))
    qrs[:, 0] = np.linspace(0, 1.5, 60)
    plain = _context_for(points, qrs=qrs)
    doubled = _context_for(points * 2.0, qrs=qrs * 2.0)
    assert _value("tq_atrial_to_ventricular_amplitude", plain) == pytest.approx(
        _value("tq_atrial_to_ventricular_amplitude", doubled), rel=1e-9
    )


def test_the_ratio_abstains_without_a_qrs_to_normalise_against():
    ctx = _context_for(_sine_window(6.0, samples=300), qrs=None)
    assert _value("tq_atrial_to_ventricular_amplitude", ctx) is None


def test_tq_cycle_fraction_uses_the_cycle_that_brackets_the_window():
    points = np.zeros((1_100, 3))
    beats = [_Beat(480, 500), _Beat(980, 1_000)]
    ctx = _Context(points, (700, 980), beats=beats)
    value = _value("tq_window_cycle_fraction", ctx)
    assert value == pytest.approx(280 / 500)
    assert 0.0 <= value <= 1.0


def test_tq_cycle_fraction_abstains_when_a_skipped_onset_spans_cycles():
    points = np.zeros((2_100, 3))
    beats = [
        _Beat(480, 500),
        _Beat(980, 1_000),
        _Beat(1_480, 1_500),
        _Beat(1_980, 2_000),
    ]
    # A low-confidence intervening onset can otherwise make TQ span several
    # median RR intervals, which produced the impossible A017 value 1.89.
    ctx = _Context(points, (700, 1_980), beats=beats)
    result = BY_NAME["tq_window_cycle_fraction"].calculator(ctx)
    assert result.value is None
    assert "not_evaluable" in result.notes


def test_direction_dispersion_is_small_for_one_direction():
    points = np.zeros((200, 3))
    points[:, 0] = np.linspace(0.01, 0.1, 200)
    ctx = _context_for(points)
    assert _value("tq_atrial_direction_dispersion", ctx) < 0.1


def test_direction_dispersion_is_large_for_a_wandering_vector():
    rng = np.random.default_rng(5)
    ctx = _context_for(rng.normal(0, 0.05, (200, 3)))
    assert _value("tq_atrial_direction_dispersion", ctx) > 0.8


# --- The strip markers, and the trap they exist to avoid ----------------------


def test_the_strip_path_length_is_normalised_rather_than_a_rate_measurement():
    """Over a fixed ten seconds a faster heart traces a longer path, so the
    unnormalised version would be heart rate under another name."""

    fs = 500
    slow = np.zeros((5000, 3))
    fast = np.zeros((5000, 3))
    # Same morphology per beat, twice as many beats.
    slow[:, 0] = np.sin(2 * np.pi * 1.0 * np.arange(5000) / fs)
    fast[:, 0] = np.sin(2 * np.pi * 2.0 * np.arange(5000) / fs)
    slow_beats = [_Beat(i * 500, i * 500 + 5) for i in range(10)]
    fast_beats = [_Beat(i * 250, i * 250 + 5) for i in range(20)]
    slow_ctx = _Context(slow, (0, 4999), fs=fs, beats=slow_beats)
    fast_ctx = _Context(fast, (0, 4999), fs=fs, beats=fast_beats)

    per_second_slow = _value("strip_path_length_per_second", slow_ctx)
    per_second_fast = _value("strip_path_length_per_second", fast_ctx)
    per_beat_slow = _value("strip_path_length_per_beat", slow_ctx)
    per_beat_fast = _value("strip_path_length_per_beat", fast_ctx)

    # The per-second version does track the rate; that is what it measures.
    assert per_second_fast > per_second_slow
    # The per-beat version must not: the morphology is identical.
    assert per_beat_fast == pytest.approx(per_beat_slow, rel=0.05)


def test_amplitude_dispersion_is_zero_for_identical_beats():
    points = np.zeros((1000, 3))
    points[:, 0] = 1.0
    beats = [_Beat(i * 100, i * 100 + 10) for i in range(10)]
    ctx = _Context(points, (0, 999), beats=beats)
    assert _value("strip_amplitude_dispersion", ctx) == pytest.approx(0.0, abs=1e-9)


def test_amplitude_dispersion_is_scale_free():
    rng = np.random.default_rng(6)
    points = np.zeros((1000, 3))
    points[:, 0] = rng.uniform(0.5, 1.5, 1000)
    beats = [_Beat(i * 100, i * 100 + 10) for i in range(10)]
    plain = _Context(points, (0, 999), beats=beats)
    scaled = _Context(points * 7.0, (0, 999), beats=beats)
    assert _value("strip_amplitude_dispersion", plain) == pytest.approx(
        _value("strip_amplitude_dispersion", scaled), rel=1e-9
    )


def test_amplitude_dispersion_is_a_cv_not_a_bounded_fraction():
    points = np.zeros((400, 3))
    points[[10, 110, 210], 0] = [0.01, 0.01, 10.0]
    beats = [_Beat(0, 10), _Beat(100, 110), _Beat(200, 210)]
    ctx = _Context(points, (0, 399), beats=beats)
    assert _value("strip_amplitude_dispersion", ctx) > 1.0
    assert BY_NAME["strip_amplitude_dispersion"].unit == "ratio"


# --- Registration, and the panel defect -------------------------------------


@pytest.mark.parametrize("name", [spec.name for spec in SEGMENT_FREE_SPECS])
def test_every_marker_is_registered(name):
    assert name in set(BiomarkerRegistry.available())


@pytest.mark.parametrize("name", [spec.name for spec in SEGMENT_FREE_SPECS])
def test_every_marker_is_discovery_only(name):
    """The defect: these displaced eight sentinel markers, one of them atrial.

    The curated panels are fixed-size and family-balanced, so a marker admitted
    to the candidate pool pushes another out. A hypothesis formed after seeing
    the development results belongs in Discovery Reserve, as every other such
    module already is.
    """

    profile = feature_profile(name)
    assert profile.discovery_eligible is True
    assert profile.core_sentinel_eligible is False
    assert profile.post_outcome_hypothesis is True


def test_adding_these_markers_removed_none():
    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    names = {spec.name for spec in SEGMENT_FREE_SPECS}
    assert names <= panels["discovery_reserve"]
    assert not (names & panels["primary_core"])
    assert not (names & panels["sentinel"])


def test_the_curated_panels_keep_their_declared_sizes():
    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240


def test_every_marker_states_its_window():
    for spec in SEGMENT_FREE_SPECS:
        assert spec.window in {"tq", "global_vcg"}
        assert spec.formula.strip()
        assert spec.evidence_note.strip()
