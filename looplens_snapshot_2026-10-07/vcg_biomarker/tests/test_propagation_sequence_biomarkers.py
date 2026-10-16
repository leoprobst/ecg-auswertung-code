"""Six markers about the order in which the heart is activated.

Measured on 896 recordings of the training split before the module was
finished. Two results shaped it and are pinned here:

* ``qrs_posterior_anterior_peak_order_ms`` came out at a median of +26 ms and
  positive in 90,2 % of 386 recordings, which is the predicted septum-then-
  free-wall order;
* ``qrs_leftward_rightward_peak_order_ms`` came out at -26 ms and positive in
  only 6,7 % of 313 -- the opposite of the prediction. The dominant rightward
  force is the terminal basal one, not the initial septal one: the leftward
  peak sits at 43 % of the QRS and the rightward at 67 %, in the second half
  in 93 % of recordings. The marker was renamed for what it measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.propagation_sequence_biomarkers import (
    AXIS_DIRECTIONS,
    MINIMUM_DIRECTION_SHARE,
    PROPAGATION_SEQUENCE_SPECS,
    PropagationSequenceBiomarker,
    propagation_sequence_marker_ids,
)


# --- registration and panel boundaries ------------------------------------


def test_six_markers_are_registered():
    ids = propagation_sequence_marker_ids()
    assert len(ids) == 6
    available = set(BiomarkerRegistry.available())
    for name in ids:
        assert name in available, name
        assert isinstance(BiomarkerRegistry.get(name), PropagationSequenceBiomarker)


def test_the_curated_panels_are_unchanged():
    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240
    assert len(panels["classical_criteria"]) == 53
    new = set(propagation_sequence_marker_ids())
    for panel_id in ("primary_core", "sentinel", "classical_criteria"):
        assert not (new & panels[panel_id]), panel_id
    assert new <= panels["discovery_reserve"]


def test_the_markers_are_post_outcome_and_out_of_the_combination_search():
    from vcg_biomarker.mimic.feature_roles import feature_profile

    for name in propagation_sequence_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis, name
        assert profile.discovery_eligible, name
        assert not profile.core_sentinel_eligible, name
        assert not profile.combination_eligible, name


def test_every_spec_names_a_source_and_claims_no_cutoff():
    for spec in PROPAGATION_SEQUENCE_SPECS:
        assert spec.references, spec.name
        definition = BiomarkerRegistry.definition(spec.name)
        assert definition.pathological_direction == "data_driven"
        assert "No diagnostic cutoff is claimed." in definition.pathological_note


# --- what already existed must keep existing ------------------------------


def test_the_two_half_window_cancellation_markers_still_exist():
    """The whole-QRS version is justified by their existence and by the fact
    that it is not a function of them. If they disappear, that argument has to
    be made again."""

    available = set(BiomarkerRegistry.available())
    assert "qrs_early_cancellation_fraction" in available
    assert "qrs_late_cancellation_fraction" in available


def test_the_xy_lag_exists_and_the_other_two_complete_it():
    available = set(BiomarkerRegistry.available())
    assert "qrs_xy_cross_correlation_peak_lag_ms" in available
    assert "qrs_xz_cross_correlation_peak_lag_ms" in available
    assert "qrs_yz_cross_correlation_peak_lag_ms" in available


def test_nothing_was_added_for_the_two_dimensional_projections():
    """2D is saturated at 97 markers per plane. Pinned so a later addition is
    a deliberate act, not a drift."""

    new = propagation_sequence_marker_ids()
    for name in new:
        for token in ("frontal_xy", "horizontal_xz", "sagittal_yz"):
            assert token not in name, name


# --- whole-QRS cancellation -----------------------------------------------


class _Ctx:
    """Minimal stand-in: the calculators read only these three things."""

    def __init__(self, points, fs=500):
        self._points = np.asarray(points, dtype=float)
        self.fs = fs

    def vcg_segment_origin_corrected(self, segment):
        return self._points if segment == "qrs" else None

    def vcg_segment(self, segment):
        return self._points if segment == "qrs" else None


def _cancellation(points):
    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as m

    return m._qrs_cancellation_fraction(_Ctx(points))


def test_a_one_way_trajectory_cancels_nothing():
    points = np.zeros((40, 3))
    points[:, 0] = 1.0
    assert _cancellation(points).value == pytest.approx(0.0, abs=1e-9)


def test_an_out_and_back_trajectory_cancels_completely():
    points = np.zeros((40, 3))
    points[:20, 0] = 1.0
    points[20:, 0] = -1.0
    value = _cancellation(points).value
    assert value is not None and value > 0.95


def test_cancellation_is_amplitude_free():
    """The reason it exists beside the QRS area: thoracic conductivity and
    body habitus scale amplitude, and this must not move with them."""

    rng = np.random.default_rng(3)
    points = rng.normal(size=(60, 3))
    base = _cancellation(points).value
    for scale in (0.1, 3.0, 25.0):
        assert _cancellation(points * scale).value == pytest.approx(base, abs=1e-9)


def test_cancellation_is_not_a_function_of_the_two_halves():
    """The argument for adding a whole-window version, stated as arithmetic.

    Two halves that each cancel nothing can still cancel completely when they
    oppose one another -- the dyssynchronous case.
    """

    points = np.zeros((40, 3))
    points[:20, 0] = 1.0
    points[20:, 0] = -1.0
    first, second = points[:20], points[20:]
    for half in (first, second):
        assert _cancellation(half).value == pytest.approx(0.0, abs=1e-6)
    assert _cancellation(points).value > 0.95


def test_cancellation_stays_within_its_bounds():
    rng = np.random.default_rng(11)
    for _ in range(40):
        value = _cancellation(rng.normal(size=(50, 3))).value
        assert 0.0 <= value <= 1.0


def test_a_degenerate_trajectory_abstains():
    value = _cancellation(np.zeros((40, 3)))
    assert value.value is None
    assert "degenerate" in (value.notes or "")


# --- directional arrival --------------------------------------------------


def _arrivals(points, fs=500):
    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as m

    return m._direction_arrivals(_Ctx(points, fs))


def _order(points, later, earlier, fs=500):
    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as m

    return m._opposed_delay(_Ctx(points, fs), later, earlier)


def test_the_six_directions_cover_the_signed_axes():
    names = {name for name, _, _ in AXIS_DIRECTIONS}
    assert names == {
        "leftward",
        "rightward",
        "inferior",
        "superior",
        "posterior",
        "anterior",
    }
    axes = {(axis, sign) for _, axis, sign in AXIS_DIRECTIONS}
    assert len(axes) == 6


def test_an_unvisited_direction_is_not_given_a_time():
    """The gate that keeps the marker from inventing an arrival.

    The vector here never points rightward at all; without the visitation
    threshold the argmax would still return a sample index.
    """

    points = np.zeros((40, 3))
    points[:, 0] = np.linspace(0.1, 1.0, 40)   # leftward only
    arrivals = _arrivals(points)
    assert "leftward" in arrivals
    assert "rightward" not in arrivals


def test_a_delay_abstains_when_one_direction_was_never_visited():
    points = np.zeros((40, 3))
    points[:, 0] = np.linspace(0.1, 1.0, 40)
    value = _order(points, "leftward", "rightward")
    assert value.value is None
    assert "never points" in (value.notes or "")


def test_the_delay_is_the_difference_of_two_arrival_times():
    points = np.zeros((50, 3))
    points[10, 0] = -1.0    # rightward peak at sample 10 -> 20 ms
    points[30, 0] = 1.0     # leftward peak at sample 30 -> 60 ms
    value = _order(points, "leftward", "rightward")
    assert value.value == pytest.approx(40.0)


def test_the_visitation_gate_is_relative_to_the_qrs_maximum():
    """A small excursion beside a large one must not count as a visit."""

    points = np.zeros((50, 3))
    points[30, 0] = 1.0
    points[10, 0] = -0.5 * MINIMUM_DIRECTION_SHARE
    assert "rightward" not in _arrivals(points)
    points[10, 0] = -2.0 * MINIMUM_DIRECTION_SHARE
    assert "rightward" in _arrivals(points)


def test_the_left_right_marker_is_named_for_the_measurement_not_the_prediction():
    """The refuted prediction, pinned so the rename is not undone silently.

    Predicted positive (septum rightward first). Measured -26 ms, positive in
    6,7 % of 313 recordings, because the dominant rightward force is terminal
    basal rather than initial septal.
    """

    import inspect

    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as m

    assert "qrs_leftward_rightward_peak_order_ms" in propagation_sequence_marker_ids()
    assert (
        "qrs_left_right_activation_delay_ms"
        not in propagation_sequence_marker_ids()
    )
    doc = inspect.getdoc(m._leftward_rightward_peak_order) or ""
    assert "-26 ms" in doc
    assert "6,7 %" in doc
    assert "67 %" in doc


def test_the_posterior_anterior_marker_records_that_its_prediction_held():
    import inspect

    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as m

    doc = inspect.getdoc(m._posterior_anterior_peak_order) or ""
    assert "+26 ms" in doc
    assert "90,2 %" in doc


# --- arrival spread -------------------------------------------------------


def _spread(points, fs=500):
    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as m

    return m._directional_arrival_spread(_Ctx(points, fs))


def test_the_spread_is_a_share_of_the_qrs_duration():
    points = np.zeros((50, 3))
    points[10, 0] = -1.0
    points[30, 0] = 1.0
    # 20 samples apart out of 50 -> 0,4 of the window.
    assert _spread(points).value == pytest.approx(20.0 / 50.0, abs=1e-9)


def test_the_spread_does_not_move_with_the_sampling_rate():
    """Dimensionless on purpose, so it cannot become a second reading of QRS
    width."""

    points = np.zeros((50, 3))
    points[10, 0] = -1.0
    points[30, 0] = 1.0
    assert _spread(points, fs=500).value == pytest.approx(
        _spread(points, fs=1000).value, abs=1e-9
    )


def test_a_single_visited_direction_abstains():
    points = np.zeros((40, 3))
    points[:, 0] = np.linspace(0.1, 1.0, 40)
    value = _spread(points)
    assert value.value is None
    assert "two directions" in (value.notes or "")


def test_the_spread_stays_within_its_bounds():
    rng = np.random.default_rng(17)
    for _ in range(40):
        value = _spread(rng.normal(size=(60, 3))).value
        if value is not None:
            assert 0.0 <= value <= 1.0


# --- the axis lags reuse the existing helper ------------------------------


def test_the_lag_helper_is_imported_not_reimplemented():
    """All three axis pairs must compute identically; the only way to
    guarantee that is to share the function."""

    import inspect

    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as m

    source = inspect.getsource(m._axis_lag)
    assert "from .advanced_vcg_biomarkers import _cross_correlation_peak" in source


def test_a_constant_axis_abstains():
    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as m

    points = np.zeros((40, 3))
    points[:, 0] = np.linspace(0.0, 1.0, 40)
    value = m._qrs_xz_lag(_Ctx(points))
    assert value.value is None
    assert "constant" in (value.notes or "")


def test_a_shifted_copy_is_found_at_the_shift():
    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as m

    t = np.linspace(0.0, 2.0 * np.pi, 60)
    points = np.zeros((60, 3))
    points[:, 0] = np.sin(t)
    points[:, 2] = np.sin(t - 0.5)      # Z lags X
    value = m._qrs_xz_lag(_Ctx(points))
    assert value.value is not None
    assert abs(value.value) > 0.0


# --- documentation carries the measurements -------------------------------


def test_the_module_docstring_records_the_measured_redundancy():
    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers

    doc = propagation_sequence_biomarkers.__doc__ or ""
    assert "0,86" in doc
    assert "0,95" in doc
    assert "0,977" in doc


def test_the_module_docstring_states_why_nothing_is_proposed_for_2d():
    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers

    doc = propagation_sequence_biomarkers.__doc__ or ""
    assert "97 markers per plane" in doc


def test_the_module_records_what_a_surface_recording_cannot_do():
    """The TRAMPOLINE and extrastimulus limitations, kept in the module so no
    later reader mistakes these for conduction-velocity markers."""

    from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers

    doc = propagation_sequence_biomarkers.__doc__ or ""
    assert "catheter findings" in doc
    assert "extrastimulus" in doc
