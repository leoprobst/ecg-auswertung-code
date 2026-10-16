import numpy as np
import pytest

from vcg_biomarker.core.svd_projection import project_ecg_components, INDEPENDENT_LEADS
from vcg_biomarker.core.data_loader import LEAD_MAP


def test_components_reconstruct_rank_three_signal_and_have_millivolt_scale():
    rng = np.random.default_rng(77)
    data = (rng.normal(size=(200, 3))*[4, 2, 1]) @ rng.normal(size=(3, 8))
    ecg = np.zeros((200, 12))
    indices = [LEAD_MAP[x] for x in INDEPENDENT_LEADS]
    ecg[:, indices] = data
    result = project_ecg_components(ecg)
    assert result.metadata["status"] == "available"
    np.testing.assert_allclose(result.coordinates @ result.loadings, data, atol=1e-12)
    np.testing.assert_allclose(result.loadings @ result.loadings.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(project_ecg_components(ecg*3).coordinates, result.coordinates*3, atol=1e-12)
    assert result.metadata["retained_energy_fraction"] == pytest.approx(1.)
    assert not result.metadata["anatomical_axes"]


def test_redundant_limb_leads_do_not_change_projection():
    ecg = np.random.default_rng(12).normal(size=(300, 12))
    original = project_ecg_components(ecg)
    for lead in ("III", "aVR", "aVL", "aVF"):
        ecg[:, LEAD_MAP[lead]] *= 100
    np.testing.assert_array_equal(original.coordinates, project_ecg_components(ecg).coordinates)


@pytest.mark.parametrize("ecg", [np.zeros((100, 12)), np.ones((100, 12))])
def test_degenerate_signal_explicitly_abstains(ecg):
    result = project_ecg_components(ecg)
    assert result.metadata["status"] == "non_estimable"
    assert np.isnan(result.coordinates).all()


def test_missing_values_and_wrong_shapes_rejected():
    for ecg in (np.ones((100, 8)), np.full((100, 12), np.nan)):
        with pytest.raises(ValueError):
            project_ecg_components(ecg)
