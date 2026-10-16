"""Identifiability examples for DCM hypotheses, not simulated disease validation."""
import numpy as np
import pytest

from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import (
    directional_operator,
    local_qrs_cancellation,
    operator_pair,
)


def matched_signals():
    phase = np.linspace(0, 2 * np.pi, 121)
    ordered = np.column_stack((.3 + np.cos(phase), np.sin(phase), .2 * np.sin(2 * phase)))
    shuffled = ordered.copy()
    rng = np.random.default_rng(606)
    # Preserve quadrature endpoints of the full segment and both halves.
    # Every half retains exactly the same sample multiset and energy.
    for start, stop in ((1, 59), (61, 120)):
        shuffled[start:stop] = ordered[rng.permutation(np.arange(start, stop))]
    return ordered, shuffled


def test_existing_local_marker_separates_chronology_with_matched_global_geometry():
    ordered, shuffled = matched_signals()
    np.testing.assert_allclose(np.trapezoid(ordered, axis=0), np.trapezoid(shuffled, axis=0), atol=1e-13)
    np.testing.assert_allclose(directional_operator(ordered), directional_operator(shuffled), atol=1e-14)
    for start, stop in ((0, 60), (60, 121)):
        np.testing.assert_allclose(directional_operator(ordered[start:stop]),
                                   directional_operator(shuffled[start:stop]), atol=1e-14)
    assert operator_pair(ordered[:60], ordered[60:])["overlap"] == pytest.approx(
        operator_pair(shuffled[:60], shuffled[60:])["overlap"])
    # Duration, global area, and half operators alone miss this distinction.
    assert local_qrs_cancellation(shuffled, 1000) > local_qrs_cancellation(ordered, 1000) + .2


def test_local_marker_cannot_identify_direction_of_time_or_anatomical_path():
    ordered, shuffled = matched_signals()
    for signal in (ordered, shuffled):
        assert local_qrs_cancellation(signal, 1000) == pytest.approx(
            local_qrs_cancellation(signal[::-1], 1000), abs=1e-13)


def test_fixed_window_is_not_invariant_to_physical_duration():
    ordered, _ = matched_signals()
    # Same geometry and sample order, but twice the physical duration:
    # a fixed 24 ms window then spans a smaller part of the trajectory.
    assert local_qrs_cancellation(ordered, 500) < local_qrs_cancellation(ordered, 1000)
