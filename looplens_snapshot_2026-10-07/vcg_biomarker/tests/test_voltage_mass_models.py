import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.voltage_mass_models import compare_voltage_mass_models


def cohort():
    rng = np.random.default_rng(78)
    n = 2500
    mass = rng.uniform(70, 350, n)
    voltage = rng.uniform(.1, 4, n)
    age = rng.normal(60, 10, n)
    outcome = rng.random(n) < 1 / (1 + np.exp(-(.3 * voltage + .004 * mass - 1.4)))
    return pd.DataFrame(dict(subject_id=np.arange(n), split=["train"] * 1500 + ["validation"] * 1000,
        label_state=np.where(outcome, "positive", "negative"), reference_eligible=True,
        extraction_status="ok", lv_mass_g=mass, voltage=voltage, age=age))


def run(frame, iterations=0):
    return compare_voltage_mass_models(frame, voltage="voltage", covariates=("age",), iterations=iterations)


def test_all_models_share_patients_and_bootstrap_comparisons():
    frame = cohort()
    frame.loc[1500, "voltage"] = np.nan
    table, models = run(frame, 100)
    assert table.status.eq("estimable").all()
    assert table.n_validation.eq(999).all()
    assert table.n_complete.eq(2499).all()
    assert len(models) == 7
    row = table.set_index("model").loc["joint_quadratic"]
    assert row.delta_auc_vs_joint_quadratic == 0
    assert row.delta_auc_vs_joint_quadratic_ci_low == 0
    assert row.delta_auc_vs_joint_quadratic_ci_high == 0
    assert not table.clinical_claim_allowed.any()


def test_validation_labels_do_not_change_fitted_models():
    frame = cohort()
    _, first = run(frame)
    mask = frame.split.eq("validation")
    frame.loc[mask, "label_state"] = frame.loc[mask, "label_state"].map({"positive": "negative", "negative": "positive"})
    _, second = run(frame)
    assert first == second


def test_locked_test_and_patient_overlap_fail_closed():
    frame = cohort()
    frame.loc[0, "split"] = "locked_test"
    with pytest.raises(PermissionError):
        run(frame)
    frame.loc[0, "split"] = "train"
    frame.loc[1500, "subject_id"] = 0
    with pytest.raises(ValueError, match="patient"):
        run(frame)


def test_unknown_references_not_negative_and_no_events_abstains():
    frame = cohort()
    frame["label_state"] = "unknown"
    table, models = run(frame)
    assert table.n_reference.eq(0).all()
    assert table.status.eq("non_estimable").all()
    assert not models


def test_separated_component_abstains():
    frame = cohort()
    frame["voltage"] = frame.label_state.eq("positive").astype(float)
    table, _ = run(frame)
    row = table.set_index("model").loc["voltage"]
    assert row.status == "non_estimable"
    assert row.abstention_reason


def test_zero_bootstrap_has_no_invented_intervals():
    table, _ = run(cohort())
    assert table.validation_auc_ci_low.isna().all()
    assert table.bootstrap_successful.eq(0).all()


def test_missingness_funnel_is_disjoint_and_preserves_complete_case_models():
    frame = cohort()
    frame.loc[0, "label_state"] = "unknown"
    frame.loc[0:1, "extraction_status"] = "failed"
    frame.loc[0:2, "age"] = np.nan
    frame.loc[0:3, "lv_mass_g"] = np.nan
    frame.loc[0:4, "voltage"] = np.nan
    frame.loc[5, "lv_mass_g"] = 0
    frame.loc[6, "lv_mass_g"] = np.inf
    frame.loc[7, "voltage"] = np.inf
    table, models = run(frame)
    counts = {"n_excluded_reference": 1, "n_excluded_quality_after_reference": 1,
        "n_excluded_covariates_after_quality": 1, "n_excluded_anatomy_after_covariates": 3,
        "n_excluded_voltage_after_anatomy": 2, "n_complete": len(frame) - 8}
    for name, expected in counts.items():
        assert table[name].eq(expected).all()
    assert table[list(counts)].sum(axis=1).eq(len(frame)).all()
    _, complete_models = run(frame.iloc[8:].copy())
    assert models == complete_models


def test_empty_reference_funnel_is_present_even_when_models_abstain():
    frame = cohort()
    frame["reference_eligible"] = False
    table, _ = run(frame)
    assert table.n_excluded_reference.eq(len(frame)).all()
    assert table.n_after_quality.eq(0).all()
    assert table.n_after_anatomy.eq(0).all()


def test_confirmation_is_one_conditional_test_not_auc_test():
    table, _ = run(cohort())
    planned = table.set_index("model").loc["joint_quadratic_plus_ratio"]
    assert planned.confirmation_status == "estimable"
    assert 0 <= planned.confirmation_p_value <= 1
    assert planned.confirmation_log_or_ci_low < planned.confirmation_log_or_ci_high
    assert "not_zero_auc_gain" in planned.confirmation_null
    assert table.loc[table.model.ne("joint_quadratic_plus_ratio"), "confirmation_status"].eq("not_planned").all()
