"""Reference tests for Rehor et al.'s 2026 direct 8-lead QRS-T angles.

Primary method source: doi:10.1038/s41598-026-37361-w.  In particular, the
Integral expression is the magnitude-weighted dual integral of *angles*, not
the angle of an averaged cosine.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import advanced_vcg_biomarkers as direct
from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import (
    DIRECT_QRS_T_LEADS,
    direct_qrs_t_angles_8lead,
)
from vcg_biomarker.core.data_loader import LEAD_MAP


INDEPENDENT_INDICES = tuple(LEAD_MAP[lead] for lead in DIRECT_QRS_T_LEADS)


def _record(
    qrs: np.ndarray,
    t_wave: np.ndarray,
    *,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    qrs = np.asarray(qrs, dtype=float)
    t_wave = np.asarray(t_wave, dtype=float)
    baseline = np.zeros(8) if offset is None else np.asarray(offset, dtype=float)
    signal = np.zeros((2 + len(qrs) + len(t_wave), 12), dtype=float)
    signal[:, INDEPENDENT_INDICES] = baseline
    q0 = 2
    q1 = q0 + len(qrs) - 1
    t0 = q1 + 1
    t1 = t0 + len(t_wave) - 1
    signal[q0 : q1 + 1, INDEPENDENT_INDICES] += qrs
    signal[t0 : t1 + 1, INDEPENDENT_INDICES] += t_wave
    return signal, (q0, q1), (t0, t1), (0, 1)


def _angles(qrs: np.ndarray, t_wave: np.ndarray, *, offset=None):
    signal, qrs_bounds, t_bounds, baseline_bounds = _record(
        qrs, t_wave, offset=offset
    )
    return direct_qrs_t_angles_8lead(
        signal,
        qrs_bounds=qrs_bounds,
        t_bounds=t_bounds,
        baseline_bounds=baseline_bounds,
    )


@pytest.mark.parametrize(
    ("t_direction", "expected"),
    [
        (np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0),
        (np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 90.0),
        (np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 180.0),
    ],
)
def test_known_constant_vector_angles(t_direction: np.ndarray, expected: float):
    qrs = np.tile(np.eye(8)[0], (5, 1))
    t_wave = np.tile(t_direction, (7, 1))
    area, integral = _angles(qrs, t_wave)
    assert area == pytest.approx(expected, abs=1e-12)
    assert integral == pytest.approx(expected, abs=1e-12)


def test_integral_is_weighted_mean_of_pairwise_angles() -> None:
    qrs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    t_wave = np.tile(np.eye(8)[0], (4, 1))
    area, integral = _angles(qrs, t_wave)

    q_magnitude = np.linalg.norm(qrs, axis=1)
    t_magnitude = np.linalg.norm(t_wave, axis=1)
    q_weights = q_magnitude * np.array([0.5, 1.0, 0.5])
    t_weights = t_magnitude * np.array([0.5, 1.0, 1.0, 0.5])
    expected_integral = (
        q_weights
        @ np.degrees(
            np.arccos(
                np.clip(
                    (qrs / q_magnitude[:, None])
                    @ (t_wave / t_magnitude[:, None]).T,
                    -1.0,
                    1.0,
                )
            )
        )
        @ t_weights
        / (q_weights.sum() * t_weights.sum())
    )
    assert integral == pytest.approx(expected_integral, abs=1e-12)
    # This deliberately distinguishes the published Integral method from the
    # former nonlinear shortcut and from the Area expression.
    assert abs(float(integral) - float(area)) > 1.0


def test_uniform_scaling_and_per_lead_baseline_offsets_do_not_change_angles():
    rng = np.random.default_rng(17)
    qrs = rng.normal(size=(9, 8))
    t_wave = rng.normal(size=(13, 8))
    expected = _angles(qrs, t_wave)
    assert _angles(7.5 * qrs, 7.5 * t_wave) == pytest.approx(expected, abs=1e-12)
    offsets = rng.uniform(-20.0, 20.0, size=8)
    assert _angles(qrs, t_wave, offset=offsets) == pytest.approx(
        expected, abs=1e-12
    )


def test_all_eight_svd_coordinates_are_exactly_rotation_invariant():
    rng = np.random.default_rng(29)
    qrs = rng.normal(size=(11, 8))
    t_wave = rng.normal(size=(17, 8))
    orthogonal, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    native = _angles(qrs, t_wave)
    rotated_all_eight = _angles(qrs @ orthogonal, t_wave @ orthogonal)
    assert rotated_all_eight == pytest.approx(native, abs=1e-10)


def test_missing_lead_or_invalid_bounds_abstain() -> None:
    qrs = np.tile(np.eye(8)[0], (4, 1))
    t_wave = np.tile(np.eye(8)[1], (4, 1))
    signal, qrs_bounds, t_bounds, baseline_bounds = _record(qrs, t_wave)
    signal[qrs_bounds[0], INDEPENDENT_INDICES[-1]] = np.nan
    assert direct_qrs_t_angles_8lead(
        signal,
        qrs_bounds=qrs_bounds,
        t_bounds=t_bounds,
        baseline_bounds=baseline_bounds,
    ) == (None, None)
    assert direct_qrs_t_angles_8lead(
        signal[:, :7],
        qrs_bounds=qrs_bounds,
        t_bounds=t_bounds,
        baseline_bounds=baseline_bounds,
    ) == (None, None)
    assert direct_qrs_t_angles_8lead(
        np.nan_to_num(signal),
        qrs_bounds=(qrs_bounds[0], len(signal) + 1),
        t_bounds=t_bounds,
        baseline_bounds=baseline_bounds,
    ) == (None, None)


def test_context_method_integrates_from_j_not_from_t_onset() -> None:
    signal = np.zeros((12, 12), dtype=float)
    signal[2:4, INDEPENDENT_INDICES[0]] = 1.0
    # Repolarisation already carries a perpendicular vector in the ST segment
    # before the delineated T onset.
    signal[5:8, INDEPENDENT_INDICES[1]] = 2.0
    signal[8:11, INDEPENDENT_INDICES[0]] = 1.0

    class Context:
        ecg_12lead = signal
        _primitive_cache: dict[tuple[object, ...], object] = {}

        @staticmethod
        def bounds(segment: str):
            return {"pr": (0, 1), "qrs": (2, 4), "t": (8, 10)}[segment]

    expected = direct_qrs_t_angles_8lead(
        signal,
        qrs_bounds=(2, 4),
        t_bounds=(4, 10),
        baseline_bounds=(0, 1),
    )
    assert direct._direct_qrs_t_angles(Context()) == pytest.approx(
        expected, abs=1e-12
    )


def test_area_and_integral_markers_share_the_expensive_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def measured(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return 31.0, 37.0

    monkeypatch.setattr(direct, "direct_qrs_t_angles_8lead", measured)

    class Context:
        ecg_12lead = np.zeros((20, 12))
        _primitive_cache: dict[tuple[object, ...], object] = {}

        @staticmethod
        def bounds(segment: str):
            return {"pr": (0, 2), "qrs": (3, 8), "t": (10, 17)}[segment]

    context = Context()
    assert direct._qrs_t_area_angle_direct_8lead(context).value == 31.0
    assert direct._qrs_t_integral_angle_direct_8lead(context).value == 37.0
    assert calls == 1
