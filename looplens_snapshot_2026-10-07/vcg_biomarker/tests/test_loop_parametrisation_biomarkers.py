"""Four markers for the time the loop picture discards.

Measured on 890 recordings of the training split before the module was
finished. The two numbers the constraint turns on, per marker: R^2 against a
pure-ECG time set (QRS duration, V6 R-peak time, R-peak timing fraction,
Tpe -- the line at 0,8) and against the nearest loop neighbours (the energy
quantiles, DTW warping energy, total turning, velocity peak count -- the line
at 0,95). Both lines were drawn before the numbers were read.

  qrs_arc_length_fraction_at_half_time         ECG 0,49   loop 0,54
  qrs_spatial_velocity_first_to_last_peak_ms   ECG 0,72   loop 0,51
  qrs_spatial_velocity_initial_burst_ms        ECG 0,03   loop 0,04
  qrs_turning_fraction_at_half_time            ECG 0,10   loop 0,09

And the check Fareh 1996 makes possible: the B-C interval was 9,3 +/- 1,1 ms
under placebo in healthy volunteers, hand-placed on a 250 mm/s strip. The
automated landmark rule here gives a project median of **8,0 ms**
(interquartile 6-10).
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.loop_parametrisation_biomarkers import (
    LOOP_PARAMETRISATION_SPECS,
    MINIMUM_PEAK_SHARE,
    MINIMUM_PROMINENCE_SHARE,
    LoopParametrisationBiomarker,
    loop_parametrisation_marker_ids,
)


# --- registration and panel boundaries ------------------------------------


def test_four_markers_are_registered():
    ids = loop_parametrisation_marker_ids()
    assert len(ids) == 4
    available = set(BiomarkerRegistry.available())
    for name in ids:
        assert name in available, name
        assert isinstance(BiomarkerRegistry.get(name), LoopParametrisationBiomarker)


def test_the_curated_panels_are_unchanged():
    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240
    assert len(panels["classical_criteria"]) == 53
    new = set(loop_parametrisation_marker_ids())
    for panel_id in ("primary_core", "sentinel", "classical_criteria"):
        assert not (new & panels[panel_id]), panel_id
    assert new <= panels["discovery_reserve"]


def test_the_markers_are_post_outcome_and_out_of_the_combination_search():
    from vcg_biomarker.mimic.feature_roles import feature_profile

    for name in loop_parametrisation_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis, name
        assert profile.discovery_eligible, name
        assert not profile.core_sentinel_eligible, name
        assert not profile.combination_eligible, name


def test_every_spec_names_a_source_and_claims_no_cutoff():
    for spec in LOOP_PARAMETRISATION_SPECS:
        assert spec.references, spec.name
        definition = BiomarkerRegistry.definition(spec.name)
        assert definition.pathological_direction == "data_driven"
        assert "No diagnostic cutoff is claimed." in definition.pathological_note
        assert "not an ECG interval" in definition.pathological_note


def test_no_marker_uses_a_fiducial_beyond_the_window():
    """The constraint, pinned at the source: no marker here may read
    marker_index, so no fiducial time can enter except as the window's edge."""

    import inspect

    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers

    source = inspect.getsource(loop_parametrisation_biomarkers)
    assert "marker_index" not in source


# --- a stub context: the calculators read only these ----------------------


class _Ctx:
    def __init__(self, points, fs=500):
        self._points = np.asarray(points, dtype=float)
        self.fs = fs

    def vcg_segment_origin_corrected(self, segment):
        return self._points if segment == "qrs" else None


def _arc(points, fs=500):
    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as m

    return m._arc_length_fraction_at_half_time(_Ctx(points, fs))


def _turn(points, fs=500):
    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as m

    return m._turning_fraction_at_half_time(_Ctx(points, fs))


def _span(points, fs=500):
    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as m

    return m._first_to_last_peak(_Ctx(points, fs))


def _burst(points, fs=500):
    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as m

    return m._initial_burst_duration(_Ctx(points, fs))


# --- arc length at half time ----------------------------------------------


def test_uniform_inscription_reaches_half_the_arc_at_half_the_time():
    points = np.zeros((41, 3))
    points[:, 0] = np.linspace(0.0, 1.0, 41)
    assert _arc(points).value == pytest.approx(0.5, abs=1e-9)


def test_early_fast_late_slow_exceeds_one_half():
    """Where the time dashes crowd late, most of the loop is already drawn."""

    points = np.zeros((41, 3))
    points[:21, 0] = np.linspace(0.0, 0.9, 21)      # fast first half
    points[21:, 0] = np.linspace(0.9, 1.0, 20)      # slow second half
    value = _arc(points).value
    assert value is not None and value > 0.8


def test_the_arc_fraction_is_invariant_to_amplitude_and_rotation():
    """It reads the parametrisation only, never where the loop sits."""

    rng = np.random.default_rng(1)
    points = np.cumsum(rng.normal(size=(50, 3)), axis=0)
    base = _arc(points).value
    theta = 0.7
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    assert _arc(points * 7.5).value == pytest.approx(base, abs=1e-9)
    assert _arc(points @ rotation.T).value == pytest.approx(base, abs=1e-9)


def test_the_arc_fraction_is_not_the_energy_quantile():
    """The reservation stated before measurement, as arithmetic.

    Arc length weights speed linearly, the energy quantile weights it
    quadratically; a trajectory with one fast and one slow half separates them.
    Measured R^2 between the two on real recordings was 0,54.
    """

    points = np.zeros((41, 3))
    points[:21, 0] = np.linspace(0.0, 0.8, 21)
    points[21:, 0] = np.linspace(0.8, 1.0, 20)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_half = float(np.sum(steps[:20]) / np.sum(steps))
    energy_half = float(np.sum(steps[:20] ** 2) / np.sum(steps ** 2))
    assert _arc(points).value == pytest.approx(arc_half, abs=1e-9)
    assert energy_half != pytest.approx(arc_half, abs=0.01)


def test_a_stationary_trajectory_abstains():
    value = _arc(np.zeros((40, 3)))
    assert value.value is None


# --- turning fraction at half time ----------------------------------------


def test_a_straight_line_has_no_turning_and_abstains():
    points = np.zeros((41, 3))
    points[:, 0] = np.linspace(0.0, 1.0, 41)
    value = _turn(points)
    assert value.value is None
    assert "no measurable turning" in (value.notes or "")


def test_a_single_sharp_turn_lands_where_it_happens():
    """Turn early -> fraction near 1; turn late -> fraction near 0."""

    # The corner must not contain a stationary step: direction is undefined
    # where nothing moves, and a turn across such a step is (correctly)
    # invisible to the marker.
    early = np.zeros((41, 3))
    early[:10, 0] = np.linspace(0.0, 0.25, 10)
    early[10:, 0] = 0.25
    early[10:, 1] = np.linspace(0.03, 1.0, 31)
    late = np.zeros((41, 3))
    late[:31, 0] = np.linspace(0.0, 1.0, 31)
    late[31:, 0] = 1.0
    late[31:, 1] = np.linspace(0.03, 0.25, 10)
    assert _turn(early).value > 0.9
    assert _turn(late).value < 0.1


def test_the_turning_fraction_is_not_the_arc_fraction():
    """A loop can travel far without turning: uniform speed along a circle
    gives arc fraction one half, but a path that turns only at its end keeps
    the turning fraction low while the arc fraction stays at one half."""

    late = np.zeros((41, 3))
    late[:21, 0] = np.linspace(0.0, 0.5, 21)
    late[21:, 0] = np.linspace(0.5, 1.0, 20)
    late[:, 1] = 0.0
    # uniform speed, straight, then a right-angle turn at the very end
    late[38:, 1] = np.array([0.0, 0.025, 0.05])
    late[38:, 0] = 0.95
    assert _arc(late).value == pytest.approx(0.5, abs=0.05)
    assert _turn(late).value < 0.2


def test_the_turning_fraction_stays_within_its_bounds():
    rng = np.random.default_rng(9)
    for _ in range(40):
        value = _turn(np.cumsum(rng.normal(size=(50, 3)), axis=0)).value
        assert value is None or 0.0 <= value <= 1.0


# --- spatial velocity landmarks -------------------------------------------


def _two_burst_trajectory(fs=500):
    """Fast, pause, fast: two velocity peaks with a trough between them.

    A ramp of the speed itself, integrated into positions, so the landmark
    finder sees two clean bursts 30 samples apart.
    """

    t = np.arange(60)
    speed = 0.02 + 1.0 * np.exp(-((t - 10) / 3.0) ** 2) + 0.8 * np.exp(-((t - 40) / 3.0) ** 2)
    points = np.zeros((61, 3))
    points[1:, 0] = np.cumsum(speed)
    return points


def test_two_bursts_give_a_span_between_their_peaks():
    points = _two_burst_trajectory()
    value = _span(points)
    assert value.value is not None
    # 30 samples apart at 500 Hz is 60 ms; smoothing moves it by a sample or two.
    assert value.value == pytest.approx(60.0, abs=6.0)


def test_one_burst_gives_no_span():
    t = np.arange(60)
    speed = 0.02 + 1.0 * np.exp(-((t - 30) / 4.0) ** 2)
    points = np.zeros((61, 3))
    points[1:, 0] = np.cumsum(speed)
    value = _span(points)
    assert value.value is None
    assert "only one" in (value.notes or "")


def test_the_initial_burst_ends_at_the_first_trough():
    points = _two_burst_trajectory()
    value = _burst(points)
    assert value.value is not None
    # From the first peak (sample ~10) to the trough between the bursts (~25).
    assert 15.0 <= value.value <= 45.0


def test_a_ripple_below_the_prominence_gate_is_not_a_peak():
    """The gate that keeps a shoulder on one burst from counting as two."""

    t = np.arange(60)
    ripple = 0.5 * MINIMUM_PROMINENCE_SHARE
    speed = 0.02 + 1.0 * np.exp(-((t - 30) / 4.0) ** 2) + ripple * np.exp(-((t - 45) / 1.5) ** 2)
    points = np.zeros((61, 3))
    points[1:, 0] = np.cumsum(speed)
    assert _span(points).value is None


def test_a_peak_below_the_share_gate_is_not_a_peak():
    t = np.arange(60)
    small = 0.5 * MINIMUM_PEAK_SHARE
    speed = 0.02 + 1.0 * np.exp(-((t - 15) / 3.0) ** 2) + small * np.exp(-((t - 45) / 3.0) ** 2)
    points = np.zeros((61, 3))
    points[1:, 0] = np.cumsum(speed)
    assert _span(points).value is None


def test_the_landmarks_are_read_on_the_same_smoothed_speed_as_the_stall_markers():
    """Fareh's landmarks and the project's stall fractions must see one
    velocity curve, so the smoothing is imported, not copied."""

    import inspect

    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as m

    source = inspect.getsource(m._speed)
    assert "from .discovery_v3_biomarkers import _smooth" in source


# --- documentation carries the measurements -------------------------------


def test_the_module_records_the_fareh_replication():
    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as m

    doc = m.__doc__ or ""
    assert "9,3" in doc
    assert "18,8" in doc
    definition = BiomarkerRegistry.definition("qrs_spatial_velocity_initial_burst_ms")
    assert "9,3" in definition.pathological_note


def test_the_module_states_the_field_went_silent():
    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as m

    doc = m.__doc__ or ""
    assert "Since 2015" in doc
    assert "1969 to 1996" in doc


def test_the_module_states_what_it_refuses_to_build():
    from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as m

    doc = m.__doc__ or ""
    assert "Gini" in doc
    assert "permutation-invariant" in doc
    assert "dwell time at the origin" in doc
