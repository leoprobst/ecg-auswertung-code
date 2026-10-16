import duckdb
import pytest

from vcg_biomarker.mimic.labels import _echo_rules


@pytest.mark.parametrize("category,expected", [
    ("not severe", "indeterminate"),
    ("no moderate pulmonary hypertension", "indeterminate"),
    ("possible mild pulmonary hypertension", "indeterminate"),
    ("cannot exclude severe", "indeterminate"),
    ("suspected moderate", "indeterminate"),
    ("without severe pulmonary hypertension", "indeterminate"),
    ("mild", "positive"), ("moderate", "positive"), ("severe", "positive"),
    ("normal", "negative"), ("unlikely", "negative"),
])
def test_negated_or_uncertain_grade_cannot_confirm_ph(category, expected):
    rule = _echo_rules()["echo_pulmonary_hypertension"]
    with duckdb.connect() as connection:
        actual = connection.execute(
            f"SELECT {rule.state} FROM (SELECT TRUE AS has_tte_30d, "
            "?::VARCHAR AS pulmonary_hypertension_category)", [category],
        ).fetchone()[0]
    assert actual == expected
