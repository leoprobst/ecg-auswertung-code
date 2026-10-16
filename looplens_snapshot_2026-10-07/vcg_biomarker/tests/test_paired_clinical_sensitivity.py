import numpy as np
import pandas as pd

from vcg_biomarker.mimic.paired_adjustment import (
    BASELINE_CLINICAL_COVARIATES, adjusted_paired_rank_association,
    paired_sensitivities, add_all_paired_adjustment_fdr,
)


def test_clinical_fit_matches_explicit_reference_and_uses_distinct_complete_cases():
    rng = np.random.default_rng(918)
    n = 600
    z = rng.normal(size=(n, 6))
    z[:, 4] = rng.integers(0, 2, n)
    x = .8*z[:, 3]+rng.normal(size=n)
    y = .3*x+1.2*z[:, 3]+rng.normal(size=n)
    frame = pd.DataFrame(z, columns=BASELINE_CLINICAL_COVARIATES)
    frame["subject_id"] = np.repeat(np.arange(150)+1, 4)
    frame["delta_lab"], frame["delta_marker"] = x, y
    frame.loc[:39, "baseline_adjust_age"] = np.nan
    row = paired_sensitivities(frame, ["marker"], exposure_column="delta_lab").iloc[0]
    reference = adjusted_paired_rank_association(x, y, frame[list(BASELINE_CLINICAL_COVARIATES)], frame.subject_id)
    assert row.adjusted_n_complete == 600
    assert row.clinical_adjusted_n_complete == 560
    assert row.clinical_adjusted_status == "estimable"
    for name in ("adjusted_rank_slope", "adjusted_se", "adjusted_p_value", "adjusted_ci_low", "adjusted_ci_high"):
        assert row[f"clinical_{name}"] == reference[name]
    assert abs(row.clinical_adjusted_rank_slope) < abs(row.adjusted_rank_slope)
    assert not row.clinical_complete_time_varying_confounder_control
    del frame["baseline_adjust_sex_male"]
    missing = paired_sensitivities(frame, ["marker"], exposure_column="delta_lab").iloc[0]
    assert missing.clinical_adjusted_status == "non_estimable"
    assert "baseline_adjust_sex_male" in missing.clinical_adjusted_abstention_reason
    assert missing.adjusted_status == "estimable"


def test_clinical_fdr_does_not_replace_primary_or_timing_families():
    frame = pd.DataFrame(dict(design=["a", "a"], global_q_value=[.8, .9],
        adjusted_status=["estimable"]*2, adjusted_p_value=[.01, .04],
        clinical_adjusted_status=["estimable", "non_estimable"], clinical_adjusted_p_value=[.3, .001]))
    result = add_all_paired_adjustment_fdr(frame)
    assert result.adjusted_global_q_value.tolist() == [.02, .04]
    assert result.clinical_adjusted_global_q_value.iloc[0] == .3
    assert np.isnan(result.clinical_adjusted_global_q_value.iloc[1])
    assert result.global_q_value.tolist() == [.8, .9]
    frame["joint_adjusted_status"] = ["estimable", "estimable"]
    frame["joint_adjusted_p_value"] = [.02, .08]
    joint = add_all_paired_adjustment_fdr(frame)
    assert joint.joint_adjusted_global_q_value.tolist() == [.04, .08]
    assert joint.clinical_adjusted_global_q_value.iloc[0] == .3
    assert joint.adjusted_global_q_value.tolist() == [.02, .04]
