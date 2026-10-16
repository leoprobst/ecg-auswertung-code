import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.lead_confirmation import adjust_lead_confirmation, confirm_component
from vcg_biomarker.tests.test_lead_comparison import sample, run
from vcg_biomarker.mimic.lead_comparison import compare_leads


def test_signal_confirmation_does_not_change_frozen_predictions():
    frame = sample()
    result = run(frame)
    descriptive = compare_leads(frame, endpoint_id="synthetic", cohort_id="synthetic",
        reference_kind="simulation", family="qrs", covariates=("age", "sex_encoded"),
        iterations=0, confirm_association=False)
    pd.testing.assert_frame_equal(result.predictions, descriptive.predictions)
    assert descriptive.results.confirmation_status.eq("descriptive_only").all()
    assert descriptive.results.confirmation_p_value.isna().all()
    row = result.results.set_index("lead").loc["I"]
    assert row.confirmation_status == "estimable"
    assert row.confirmation_p_value < .001
    assert row.confirmation_direction_preserved
    assert "not_zero_auc_gain" in row.confirmation_null


def test_confirmation_fdr_counts_planned_failures_without_reporting_fake_p_values():
    frame = pd.DataFrame({"endpoint_id": ["acute_mi_high_specificity", "myocarditis_icd", "myocarditis_icd"],
        "family": ["qrs"]*3, "lead": ["I", "I", "II"],
        "confirmation_status": ["estimable", "estimable", "non_estimable"],
        "confirmation_p_value": [.001, .002, np.nan]})
    output = adjust_lead_confirmation(frame, planned_endpoints=("acute_mi_high_specificity", "myocarditis_icd"))
    np.testing.assert_allclose(output.confirmation_q_global_bh.iloc[:2], [.096, .096])
    assert output.confirmation_q_domain_bh.iloc[0] == pytest.approx(.048)
    assert output.confirmation_q_global_by.iloc[0] >= output.confirmation_q_global_bh.iloc[0]
    assert np.isnan(output.confirmation_q_global_bh.iloc[2])
    assert np.isnan(output.confirmation_p_value.iloc[2])
    assert output.confirmation_planned_hypotheses_global.eq(96).all()


def test_confirmation_declines_constant_lead_component():
    x = np.column_stack([np.linspace(0, 1, 100), np.ones(100)])
    model = {"active_columns": [0], "coefficients": [0., 1.], "mean": [0., 0.], "scale": [1., 0.]}
    result = confirm_component(model, x, x, np.arange(100)%2, 1)
    assert result["confirmation_status"] == "non_estimable"
    assert np.isnan(result["confirmation_p_value"])


def test_adjustment_removes_a_strong_shared_clinical_factor():
    rng = np.random.default_rng(834)
    train = rng.normal(size=(1200, 2))
    valid = rng.normal(size=(1200, 2))
    train[:, 1] += 2*train[:, 0]
    valid[:, 1] += 2*valid[:, 0]
    labels = (rng.random(1200) < 1/(1+np.exp(-1.5*valid[:, 0]))).astype(int)
    model = {"active_columns": [0, 1], "coefficients": [0., 0., 1.], "mean": [0., 0.], "scale": [1., 1.]}
    result = confirm_component(model, train, valid, labels, 1)
    assert result["confirmation_status"] == "estimable"
    assert result["confirmation_p_value"] > .05


def test_pilot_like_descriptive_rows_are_excluded_from_fdr():
    frame = pd.DataFrame({"endpoint_id": ["test"], "family": ["qrs"], "lead": ["I"],
        "confirmation_status": ["descriptive_only"], "confirmation_p_value": [.0001]})
    output = adjust_lead_confirmation(frame, planned_endpoints=("test",))
    assert output.confirmation_q_global_bh.isna().all()
    with pytest.raises(ValueError, match="unplanned"):
        adjust_lead_confirmation(frame, planned_endpoints=())


@pytest.mark.parametrize("column,value", [("lead", "V99"), ("family", "invented")])
def test_confirmation_rejects_unplanned_measurement_families(column, value):
    frame = pd.DataFrame({"endpoint_id": ["test"], "family": ["qrs"], "lead": ["I"]})
    frame[column] = value
    with pytest.raises(ValueError, match="prespecified"):
        adjust_lead_confirmation(frame, planned_endpoints=("test",))


def test_confirmation_rejects_duplicate_hypotheses():
    frame = pd.DataFrame({"endpoint_id": ["test"]*2, "family": ["qrs"]*2, "lead": ["I"]*2})
    with pytest.raises(ValueError, match="Duplicate"):
        adjust_lead_confirmation(frame, planned_endpoints=("test",))
