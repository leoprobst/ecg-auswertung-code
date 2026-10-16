import numpy as np
import pytest

from vcg_biomarker.core.biomarkers.qrs_t_method_comparison import compare_qrs_t_methods
from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import direct_qrs_t_angles_8lead


def test_direct_branch_matches_existing_formula_and_shared_baseline_is_offset_invariant():
    signal = np.random.default_rng(191).normal(size=(100, 12))
    kwargs = dict(qrs_bounds=(20, 35), t_bounds=(40, 80), baseline_bounds=(4, 15))
    result = compare_qrs_t_methods(signal, **kwargs)
    expected = direct_qrs_t_angles_8lead(signal, **{**kwargs, "t_bounds": (35, 80)})
    np.testing.assert_allclose([result[0]["area_angle_deg"], result[0]["integral_angle_deg"]], expected, atol=1e-12)
    shifted = compare_qrs_t_methods(signal+np.arange(12)*3, **kwargs)
    for first, second in zip(result, shifted):
        assert first["status"] == "available"
        assert not first["clinical_claim_allowed"]
        np.testing.assert_allclose([first["area_angle_deg"], first["integral_angle_deg"]],
                                  [second["area_angle_deg"], second["integral_angle_deg"]], atol=1e-10)


def test_unavailable_segments_abstain_for_every_method():
    result = compare_qrs_t_methods(np.ones((100, 12)), qrs_bounds=(20, 35), t_bounds=None, baseline_bounds=(4, 15))
    assert len(result) == 4
    assert all(row["status"] == "non_estimable" for row in result)


def test_post_qrs_baseline_rejected_for_every_method():
    result = compare_qrs_t_methods(np.ones((100, 12)), qrs_bounds=(20, 35), t_bounds=(40, 80), baseline_bounds=(50, 60))
    assert all(row["reason"] == "invalid_common_segment_order_or_baseline" for row in result)


@pytest.mark.parametrize("bounds", [(20.5, 35), (20, np.nan), (True, False), (20,), (20, 35, 40)])
def test_noninteger_segment_indices_abstain(bounds):
    result = compare_qrs_t_methods(np.ones((100, 12)), qrs_bounds=bounds,
                                  t_bounds=(40, 80), baseline_bounds=(4, 15))
    assert all(row["status"] == "non_estimable" for row in result)


def test_rank_three_svd_preserves_direct_angles_and_positive_scaling():
    from vcg_biomarker.core.data_loader import LEAD_MAP
    from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import DIRECT_QRS_T_LEADS
    rng = np.random.default_rng(292)
    signal = np.zeros((100, 12))
    signal[:, [LEAD_MAP[name] for name in DIRECT_QRS_T_LEADS]] = (
        rng.normal(size=(100, 3)) @ rng.normal(size=(3, 8)))
    kwargs = dict(qrs_bounds=(20, 35), t_bounds=(40, 80), baseline_bounds=(4, 15))
    rows = compare_qrs_t_methods(signal, **kwargs)
    values = lambda row: [row["area_angle_deg"], row["integral_angle_deg"]]
    np.testing.assert_allclose(values(rows[0]), values(rows[3]), atol=1e-10)
    assert rows[3]["retained_energy_fraction"] == pytest.approx(1.0)
    scaled = compare_qrs_t_methods(signal * 2.3, **kwargs)
    for first, second in zip(rows, scaled):
        np.testing.assert_allclose(values(first), values(second), atol=1e-10)


@pytest.mark.parametrize("failed_segment", ["qrs", "t", "pr"])
def test_present_bounds_do_not_override_failed_segment_quality(failed_segment):
    gates = dict(qrs=True, t=True, pr=True)
    gates[failed_segment] = False
    rows = compare_qrs_t_methods(np.random.default_rng(191).normal(size=(100, 12)),
        qrs_bounds=(20, 35), t_bounds=(40, 80), baseline_bounds=(4, 15),
        segment_evaluable=gates)
    assert all(row["status"] == "non_estimable" and row["area_angle_deg"] is None
               and row["reason"] == "segment_quality_gate_failed:" + failed_segment for row in rows)
