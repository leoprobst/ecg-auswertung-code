import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.paired_lab_context import joint_lab_context, LAB_CONTEXT_ANALYTES


def test_context_joins_same_patient_and_both_ecgs_without_cross_pair_imputation():
    rows = []
    for patient, first, second in [(1, 10, 11), (1, 11, 12), (2, 10, 11)]:
        for analyte in LAB_CONTEXT_ANALYTES:
            rows.append(dict(subject_id=patient, first_study_id=first, second_study_id=second,
                analyte_id=analyte, delta_lab=.1 if first == 10 else .2, lab_pair_status="eligible"))
    frame = pd.DataFrame(rows)
    frame.loc[(frame.first_study_id == 11) & frame.analyte_id.eq("ph"), "lab_pair_status"] = "reference_ineligible_or_missing"
    frame.loc[(frame.subject_id == 2) & frame.analyte_id.eq("calcium_ionized"), "delta_lab"] = np.inf
    original = frame.copy(deep=True)
    wide, summary = joint_lab_context(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert summary["n_pairs"] == 3 and summary["n_pairs_complete_core"] == 1
    assert summary["n_pairs_complete_all_five"] == 1
    assert summary["analytes"]["ph"] == {"n_pairs": 2, "n_patients": 2}
    assert sum(summary["coverage_patterns"].values()) == 3
    assert np.isnan(wide.loc[wide.subject_id.eq(2), "context_delta_calcium_ionized"].iloc[0])
    assert wide.loc[wide.subject_id.eq(2), "context_delta_calcium_total"].iloc[0] == .1
    assert np.isnan(wide.loc[wide.first_study_id.eq(11), "context_delta_ph"].iloc[0])
    assert not summary["clinical_claim_allowed"]


def test_duplicate_identity_rejected_and_missing_patient_not_eligible():
    row = dict(subject_id=None, first_study_id=1, second_study_id=2,
        analyte_id="potassium", delta_lab=.3, lab_pair_status="eligible")
    wide, summary = joint_lab_context(pd.DataFrame([row]))
    assert len(wide) == 1 and summary["n_pairs_any_lab"] == 0
    with pytest.raises(ValueError, match="Ambiguous"):
        joint_lab_context(pd.DataFrame([row, row]))
    with pytest.raises(ValueError, match="Missing"):
        joint_lab_context(pd.DataFrame([{"subject_id": 1}]))
    assert joint_lab_context(pd.DataFrame())[1]["status"] == "no_pairs"


def test_joint_model_uses_other_labs_and_exact_reference_fit():
    from vcg_biomarker.mimic.paired_lab_context import joint_physiology_sensitivity, PHYSIOLOGY_CORE
    from vcg_biomarker.mimic.paired_adjustment import BASELINE_CLINICAL_COVARIATES, adjusted_paired_rank_association
    rng = np.random.default_rng(914)
    n = 480
    z = rng.normal(size=(n, len(BASELINE_CLINICAL_COVARIATES)))
    z[:, 4] = rng.integers(0, 2, n)
    labs = {name: rng.normal(size=n) for name in LAB_CONTEXT_ANALYTES}
    labs["potassium"] += .8*labs["calcium_ionized"]
    y = .3*labs["potassium"]+1.2*labs["calcium_ionized"]+rng.normal(size=n)
    base = pd.DataFrame(z, columns=BASELINE_CLINICAL_COVARIATES).assign(
        subject_id=np.repeat(np.arange(120)+1, 4), first_study_id=np.arange(n)*2,
        second_study_id=np.arange(n)*2+1, lab_pair_status="eligible", delta_marker=y)
    changes = pd.concat([base.assign(analyte_id=name, delta_lab=values) for name, values in labs.items()], ignore_index=True)
    context, _ = joint_lab_context(changes)
    models = joint_physiology_sensitivity(changes, context, ["marker"])
    row = models.loc[models.design.eq("dialysis_potassium")].iloc[0]
    controls = [f"context_delta_{name}" for name in PHYSIOLOGY_CORE if name != "potassium"]
    expected = adjusted_paired_rank_association(labs["potassium"], y,
        np.column_stack((z, context[controls].to_numpy())), base.subject_id)
    simple = adjusted_paired_rank_association(labs["potassium"], y, z, base.subject_id)
    assert len(models) == 5 and row.joint_adjusted_status == "estimable"
    assert row.joint_adjusted_rank_slope == expected["adjusted_rank_slope"]
    assert row.joint_adjusted_ci_low == expected["adjusted_ci_low"]
    assert abs(row.joint_adjusted_rank_slope) < abs(simple["adjusted_rank_slope"])
    assert "context_delta_potassium" not in row.joint_adjusted_covariates
    assert "context_delta_calcium_total" not in row.joint_adjusted_covariates
    context["context_delta_ph"] = np.nan
    missing = joint_physiology_sensitivity(changes, context, ["marker"])
    assert missing.loc[missing.design.eq("dialysis_potassium"), "joint_adjusted_status"].iloc[0] == "non_estimable"
