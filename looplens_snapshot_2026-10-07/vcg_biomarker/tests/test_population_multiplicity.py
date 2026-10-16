import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.population_multiplicity import adjust_population_survival


def rows():
    return pd.DataFrame([dict(cohort="code15", marker_id="a", status="estimable", p_value=.01,
        incremental_comparison_status="estimable", incremental_p_value=.02,
        ecg_age_comparison_status="estimable", ecg_age_incremental_p_value=.03),
        dict(cohort="code15", marker_id="b", status="non_estimable", p_value=np.nan,
             incremental_comparison_status="non_estimable", ecg_age_comparison_status="non_estimable")])


def run(frame):
    return adjust_population_survival(frame,marker_ids=("a","b"),incremental_marker_ids=("a","b"))


def test_planned_unavailable_markers_stay_in_denominator_not_reported_pvalues():
    result = run(rows()).set_index("marker_id")
    assert result.loc["a","q_value_cohort"] == pytest.approx(.03)  # a,b,ECG-age
    assert result.loc["a","incremental_q_value_cohort"] == pytest.approx(.04)
    assert result.loc["a","ecg_age_incremental_q_value_cohort"] == pytest.approx(.06)
    assert np.isnan(result.loc["b","q_value_cohort"])
    missing_row = run(rows().iloc[:1]).iloc[0]
    assert missing_row.q_value_cohort == pytest.approx(.03)


def test_cohorts_not_pooled_and_invalid_pvalues_remain_missing():
    other = rows().iloc[:1].assign(cohort="sami_trop",p_value=.02)
    combined = run(pd.concat([rows(),other],ignore_index=True))
    assert combined.iloc[-1].q_value_cohort == pytest.approx(.06)
    for invalid in (-.1,1.1,np.inf,np.nan):
        assert np.isnan(run(rows().iloc[:1].assign(p_value=invalid)).iloc[0].q_value_cohort)


def test_duplicates_and_unplanned_tests_rejected():
    with pytest.raises(ValueError,match="Duplicate"):
        run(pd.concat([rows(),rows()]))
    with pytest.raises(ValueError,match="Unplanned"):
        run(rows().assign(marker_id=["x","y"]))
