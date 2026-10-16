import numpy as np
import pytest

from vcg_biomarker.core.biomarkers.qrs_t_segment_matrix import qrs_t_segment_matrix
from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import direct_qrs_t_angles_8lead


def data():
    signal = np.random.default_rng(8).normal(size=(220, 12))
    signal[:20] = 0
    return signal


def matrix(signal):
    return qrs_t_segment_matrix(signal, qrs_bounds=(30, 80), t_bounds=(90, 180), baseline_bounds=(0, 15))


def test_cell_weighted_sum_reconstructs_original_integral():
    signal = data()
    result = matrix(signal)
    _, original = direct_qrs_t_angles_8lead(signal, qrs_bounds=(30, 80), t_bounds=(80, 180), baseline_bounds=(0, 15))
    assert result["status"] == "evaluable"
    assert len(result["cells"]) == 9
    assert sum(x["magnitude_pair_fraction"] for x in result["cells"]) == pytest.approx(1.)
    assert result["reconstructed_integral_angle_deg"] == pytest.approx(original, abs=1e-10)
    assert not result["clinical_claim_allowed"]


def test_scale_offset_invariance_and_zero_energy_is_missing():
    signal = data()
    first, second = matrix(signal), matrix(signal * 3 + 7)
    np.testing.assert_allclose([x["angle_deg"] for x in first["cells"]],
                               [x["angle_deg"] for x in second["cells"]], atol=1e-10)
    signal[30:49] = 0
    result = matrix(signal)
    assert result["status"] == "partial"
    assert result["reconstructed_integral_angle_deg"] is None
    assert all(x["angle_deg"] is None for x in result["cells"] if x["qrs_third"] == 1)


def test_quality_gate_and_invalid_order_abstain():
    result = qrs_t_segment_matrix(data(), qrs_bounds=(30, 80), t_bounds=(90, 180),
        baseline_bounds=(0, 15), segment_evaluable={"qrs": True, "t": False, "pr": True})
    assert not result["cells"]
    assert result["reason"] == "segment_quality_gate_failed"
    result = qrs_t_segment_matrix(data(), qrs_bounds=(30, 80), t_bounds=(70, 180), baseline_bounds=(0, 15))
    assert result["reason"] == "invalid_segment_order"


def test_ragged_bounds_and_nonnumeric_signal_abstain():
    result = qrs_t_segment_matrix(data(), qrs_bounds=(30,), t_bounds=(90, 180), baseline_bounds=(0, 15))
    assert result["reason"] == "invalid_segment_indices"
    result = matrix([["not_a_signal"]])
    assert result["reason"] == "invalid_twelve_lead_signal"
