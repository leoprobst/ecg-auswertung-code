import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import rankdata

from vcg_biomarker.mimic.clustered_rank_association import clustered_rank_association


def test_matches_full_statsmodels_cluster_sandwich():
    rng = np.random.default_rng(68)
    subjects = np.repeat(np.arange(50)+1, 4)
    x = rng.normal(size=200)
    y = .4*x + np.repeat(rng.normal(size=50),4) + rng.normal(size=200)
    result = clustered_rank_association(x,y,subjects)
    rx,ry = rankdata(x),rankdata(y)
    rx=(rx-rx.mean())/rx.std()
    ry=(ry-ry.mean())/ry.std()
    fit = sm.OLS(ry,sm.add_constant(rx)).fit(cov_type="cluster",
        cov_kwds={"groups": subjects,"use_correction":True,"df_correction":True},use_t=True)
    assert result["spearman_rho"] == pytest.approx(fit.params[1])
    assert result["rank_slope_se"] == pytest.approx(fit.bse[1])
    assert result["p_value"] == pytest.approx(fit.pvalues[1])
    np.testing.assert_allclose([result["rank_slope_ci_low"],result["rank_slope_ci_high"]],fit.conf_int()[1])


def test_repeated_copies_do_not_multiply_independent_patients():
    rng=np.random.default_rng(2)
    x=rng.normal(size=80)
    y=.2*x+rng.normal(size=80)
    ids=np.arange(80)+1
    original=clustered_rank_association(x,y,ids)
    copied=clustered_rank_association(np.repeat(x,20),np.repeat(y,20),np.repeat(ids,20))
    assert copied["n_patients"] == 80
    assert copied["rank_slope_se"] == pytest.approx(original["rank_slope_se"],rel=.02)


@pytest.mark.parametrize("subjects,reason", [(None,"unknown_patient_identity"),
    ([1]*100,"too_few_independent_patients"),([None]*100,"unknown_patient_identity")])
def test_no_replacement_p_value_without_independent_patients(subjects,reason):
    x=np.arange(100,dtype=float)
    y=np.sin(x)
    result=clustered_rank_association(x,y,subjects)
    assert result["association_status"] == "non_estimable"
    assert result["abstention_reason"] == reason
    assert np.isnan(result["p_value"])


def test_public_pair_path_rejects_unknown_identity_for_inference():
    from vcg_biomarker.mimic.within_subject import within_subject_association
    result=within_subject_association(pd.DataFrame(dict(delta_lab=np.arange(100),
        delta_marker=np.sin(np.arange(100)))),("marker",)).iloc[0]
    assert result.association_status == "non_estimable"
    assert np.isnan(result.p_value)
