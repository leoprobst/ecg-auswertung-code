import numpy as np
import pytest
import statsmodels.api as sm
from scipy.stats import rankdata

from vcg_biomarker.mimic.paired_adjustment import adjusted_paired_rank_association


def fixture():
    rng = np.random.default_rng(98)
    z = rng.normal(size=(800, 3))
    x = z[:, 0]+rng.normal(size=800)
    y = .4*x+z[:, 0]+rng.normal(size=800)
    return x, y, z, np.repeat(np.arange(200)+1, 4)


def test_matches_full_statsmodels_covariance_and_ci():
    x, y, z, ids = fixture()
    result = adjusted_paired_rank_association(x, y, z, ids)
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = (rx-rx.mean())/rx.std(), (ry-ry.mean())/ry.std()
    design = np.column_stack((np.ones(len(x)), rx, (z-z.mean(0))/z.std(0)))
    fit = sm.OLS(ry, design).fit(cov_type="cluster", cov_kwds={"groups": ids,
        "use_correction": True, "df_correction": True}, use_t=True)
    assert result["adjusted_status"] == "estimable"
    np.testing.assert_allclose([result["adjusted_rank_slope"], result["adjusted_se"], result["adjusted_p_value"]],
                              [fit.params[1], fit.bse[1], fit.pvalues[1]], rtol=1e-10)
    np.testing.assert_allclose([result["adjusted_ci_low"], result["adjusted_ci_high"]], fit.conf_int()[1])


@pytest.mark.parametrize("problem", ["collinear", "few", "unknown", "constant"])
def test_abstention_without_replacement_p(problem):
    x, y, z, ids = fixture()
    if problem == "collinear": z[:, 2] = z[:, 1]
    elif problem == "few": ids[:] = 1
    elif problem == "unknown": ids = None
    else: z[:, 2] = 1
    result = adjusted_paired_rank_association(x, y, z, ids)
    assert result["adjusted_status"] == "non_estimable"
    assert np.isnan(result["adjusted_p_value"])


def test_partial_adjustment_reduces_simulated_confounding_preserves_effect():
    rng = np.random.default_rng(721)
    n = 4000
    z = rng.normal(size=(n, 1))
    x = z[:, 0]+rng.normal(size=n)
    noise = rng.normal(size=n)
    ids = np.arange(n)+1
    null = adjusted_paired_rank_association(x, z[:, 0]+noise, z, ids)
    true = adjusted_paired_rank_association(x, .6*x+z[:, 0]+noise, z, ids)
    assert abs(null["adjusted_rank_slope"]) < .1*abs(null["unadjusted_same_sample_rank_slope"])
    assert true["adjusted_rank_slope"] > .2 and true["adjusted_p_value"] < .001


def test_complete_case_counts_and_large_finite_covariates():
    x, y, z, ids = fixture()
    expected = adjusted_paired_rank_association(x[5:], y[5:], z[5:], ids[5:])
    z[:5] = np.nan
    z[:, 0] *= 1e300
    actual = adjusted_paired_rank_association(x, y, z, ids)
    assert actual["adjusted_n_input"] == 800 and actual["adjusted_n_complete"] == 795
    assert actual["adjusted_rank_slope"] == pytest.approx(expected["adjusted_rank_slope"])
    assert actual["adjusted_se"] == pytest.approx(expected["adjusted_se"])


def test_named_adapter_and_separate_fdr_never_invent_missing_adjustment():
    import pandas as pd
    from vcg_biomarker.mimic.paired_adjustment import paired_timing_quality_sensitivity, add_paired_adjustment_fdr
    x, y, _, ids = fixture()
    result = paired_timing_quality_sensitivity(pd.DataFrame(dict(subject_id=ids, delta_lab=x, delta_marker=y)),
                                              ("marker",), exposure_column="delta_lab")
    assert result.adjusted_status.eq("non_estimable").all()
    assert not result.complete_time_varying_confounder_control.any()
    frame = pd.DataFrame(dict(design=["a", "a", "b"], adjusted_status=["estimable", "estimable", "non_estimable"],
        adjusted_p_value=[.01, .04, .0001], global_q_value=[.3, .4, .5]))
    corrected = add_paired_adjustment_fdr(frame)
    np.testing.assert_allclose(corrected.adjusted_global_q_value[:2], [.02, .04])
    assert np.isnan(corrected.adjusted_global_q_value.iloc[2])
    pd.testing.assert_series_equal(frame.global_q_value, corrected.global_q_value)
