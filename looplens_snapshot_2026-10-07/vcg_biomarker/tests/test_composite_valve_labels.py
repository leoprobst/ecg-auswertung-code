"""A conjunction over nullable columns emptied two endpoints.

``significant_valvular_disease`` defined its negative arm as
``AND``-joined ``severity < 2`` over five valves.  In SQL ``NULL < 2`` is NULL,
so the arm was only true when **every** valve had been graded -- and an
echocardiogram grades the valves it was asked about.  Measured on the 206,488
studies in this project the mean number of graded valves is 2.28, and only
2,611 studies have all five.

The endpoint therefore ran with 819 negatives against 42,056 positives, a
prevalence of 0.981, and the power gate passed it as eligible because it counts
absolute class sizes rather than balance.  ``_severe`` was the same shape:
1,511 negatives at prevalence 0.928.

Asking instead what the reference standard can answer -- was any valve
assessed, and did any assessed valve reach the threshold -- gives 147,249 and
169,810 negatives, prevalences of 0.222 and 0.103.
"""

from __future__ import annotations

import duckdb
import pytest

from vcg_biomarker.mimic.labels import _echo_rules

VALVES = (
    "aortic_stenosis",
    "aortic_regurgitation",
    "mitral_stenosis",
    "mitral_regurgitation",
    "tricuspid_regurgitation",
)


def _state(label_id: str, severities: dict[str, object], *, has_tte: bool = True):
    rule = _echo_rules()[label_id]
    columns = ", ".join(
        f"{'NULL' if severities.get(valve) is None else severities[valve]}"
        f"::INTEGER AS {valve}_severity"
        for valve in VALVES
    )
    query = f"""
    WITH source AS (
        SELECT {'TRUE' if has_tte else 'FALSE'} AS has_tte_30d, {columns}
    )
    SELECT {rule.state} AS label_state FROM source
    """
    with duckdb.connect() as connection:
        return connection.execute(query).fetchone()[0]


def _all(value):
    return {valve: value for valve in VALVES}


def test_a_graded_severe_valve_is_positive():
    severities = _all(None)
    severities["aortic_stenosis"] = 3
    assert _state("significant_valvular_disease", severities) == "positive"
    assert _state("significant_valvular_disease_severe", severities) == "positive"


def test_one_graded_normal_valve_is_now_a_usable_negative():
    # This is the case the old rule threw away: 147,249 studies of 206,488.
    severities = _all(None)
    severities["mitral_regurgitation"] = 1
    assert _state("significant_valvular_disease", severities) == "negative"


def test_a_study_with_no_graded_valve_stays_unavailable():
    # Absence of grading is not evidence of a normal valve.
    assert _state("significant_valvular_disease", _all(None)) == "unavailable"


def test_a_moderate_valve_is_positive_for_significant_and_negative_for_severe():
    severities = _all(None)
    severities["tricuspid_regurgitation"] = 2
    assert _state("significant_valvular_disease", severities) == "positive"
    assert _state("significant_valvular_disease_severe", severities) == "negative"


def test_an_ungraded_valve_cannot_make_a_study_positive():
    severities = _all(None)
    severities["aortic_stenosis"] = 0
    assert _state("significant_valvular_disease", severities) == "negative"


def test_all_five_graded_and_normal_is_still_a_negative():
    assert _state("significant_valvular_disease", _all(1)) == "negative"
    assert _state("significant_valvular_disease_severe", _all(2)) == "negative"


def test_no_study_can_satisfy_both_arms():
    severities = _all(None)
    severities["aortic_stenosis"] = 3
    severities["mitral_regurgitation"] = 0
    # A severe valve alongside a normal one is positive, never both.
    assert _state("significant_valvular_disease", severities) == "positive"


def test_without_an_echocardiogram_nothing_is_assessable():
    severities = _all(1)
    assert _state(
        "significant_valvular_disease", severities, has_tte=False
    ) == "unavailable"


def test_the_summary_records_how_many_valves_were_graded():
    # A negative from two graded valves is weaker evidence than one from five,
    # and the result row has to carry that distinction.
    rule = _echo_rules()["significant_valvular_disease"]
    assert "graded valves=" in rule.evidence_summary
