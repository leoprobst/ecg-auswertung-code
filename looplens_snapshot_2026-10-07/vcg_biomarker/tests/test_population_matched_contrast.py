import numpy as np
import pandas as pd

from vcg_biomarker.mimic.population_cohorts import _chagas_contrast


def cohort():
    def block(name, n, offset):
        return pd.DataFrame(dict(cohort=name, patient_id=np.arange(n) + offset,
            exam_id=np.arange(n) + offset, extraction_status="ok", age=50., sex="M",
            marker=np.linspace(0, 1, n)))
    return pd.concat([block("code15", 50, 1), block("sami_trop", 100, 1001)], ignore_index=True)


def test_both_cohorts_are_restricted_to_matching_support():
    frame = cohort()
    frame.loc[100:, "age"] = 80
    result = _chagas_contrast(frame, ["marker"]).iloc[0]
    assert result.n_chagas == result.n_code15_matched == 50
    assert result.n_sami_index == 100
    assert result.n_matched_pairs_before_marker == 50
    pd.testing.assert_frame_equal(_chagas_contrast(frame, ["marker"]),
        _chagas_contrast(frame.sample(frac=1, random_state=7), ["marker"]))


def test_missing_marker_removes_both_fixed_pair_members():
    frame = cohort()
    frame.loc[0, "marker"] = np.inf
    result = _chagas_contrast(frame, ["marker"]).iloc[0]
    assert result.n_chagas == result.n_code15_matched == 49
    assert result.n_pairs_excluded_marker == 1


def test_missing_sex_is_not_a_matching_stratum():
    frame = cohort()
    frame["sex"] = None
    assert _chagas_contrast(frame, ["marker"]).empty


def test_failed_index_is_not_replaced_with_a_successful_repeat():
    frame = cohort()
    repeat = frame.iloc[[0]].copy()
    repeat["exam_id"] = 9999
    frame.loc[0, "extraction_status"] = "failed"
    result = _chagas_contrast(pd.concat([frame, repeat]), ["marker"]).iloc[0]
    assert result.n_matched_pairs_before_marker == 49
    assert result.n_code15_index == 50
    assert result.n_code15_after_qc == 49
