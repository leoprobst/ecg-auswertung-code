"""Audit 5.3.4: effective number of tests."""
import numpy as np

from vcg_biomarker.mimic.marker_redundancy import effective_number_of_tests


def test_independent_and_identical_markers_bound_the_estimates():
    independent = effective_number_of_tests(np.eye(10))
    assert independent["m_eff_li_ji"] == 10.0 and independent["m_eff_nyholt"] == 10.0
    identical = effective_number_of_tests(np.ones((10, 10)))
    assert abs(identical["m_eff_li_ji"] - 1.0) < 1e-9   # one eigenvalue 10, the rest 0
    assert abs(identical["m_eff_nyholt"] - 1.0) < 1e-9


def test_li_ji_on_a_block_structure():
    # Two blocks of five perfectly correlated markers: two independent tests.
    block = np.kron(np.eye(2), np.ones((5, 5)))
    result = effective_number_of_tests(block)
    assert abs(result["m_eff_li_ji"] - 2.0) < 1e-9
    # A missing correlation is treated as independent (conservative).
    partial = np.eye(3)
    partial[0, 1] = partial[1, 0] = np.nan
    assert effective_number_of_tests(partial)["n_missing_pairwise_correlations"] == 1
