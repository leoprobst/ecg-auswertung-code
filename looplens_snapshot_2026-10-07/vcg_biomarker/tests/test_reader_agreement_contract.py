import pandas as pd
import pytest

from vcg_biomarker.mimic.reference_tiers import adjudication_agreement


def reviews():
    return pd.DataFrame(dict(review_case_id=["a","a","b","b"],reviewer_slot=[1,2,1,2],
                             reviewer_verdict=["positive","positive","negative","negative"]))


def test_duplicate_verdict_cannot_be_silently_selected():
    frame = reviews()
    with pytest.raises(ValueError, match="Duplicate"):
        adjudication_agreement(pd.concat([frame,frame.iloc[:1]]))


def test_third_reviewer_is_not_silently_ignored():
    frame = reviews()
    third = frame.loc[frame.reviewer_slot.eq(1)].assign(reviewer_slot=3)
    result = adjudication_agreement(pd.concat([frame,third]))
    assert result["status"] == "not_evaluable"
    assert result["n_reviewers"] == 3
    assert "exactly_two" in result["failure_reason"]


def test_constant_agreement_does_not_claim_estimable_kappa():
    result = adjudication_agreement(reviews().assign(reviewer_verdict="positive"))
    assert result["status"] == "not_evaluable"
    assert result["raw_agreement"] == 1.
    assert result["failure_reason"] == "kappa_undefined_constant_verdict"
    assert not result["clinical_claim_allowed"]


def test_missing_verdict_is_not_a_category_and_sample_is_not_population_accuracy():
    frame = reviews()
    frame.loc[3,"reviewer_verdict"] = " "
    result = adjudication_agreement(frame)
    assert result["n_cases_submitted"] == 2
    assert result["n_cases"] == 1
    assert result["status"] == "not_evaluable"
    assert result["agreement_scope"].startswith("unweighted_selected_review_sample")
    with pytest.raises(ValueError, match="identities"):
        adjudication_agreement(reviews().assign(review_case_id=None))
