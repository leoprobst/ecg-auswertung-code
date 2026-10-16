import duckdb
import pytest
from vcg_biomarker.mimic.labels import _echo_rules


@pytest.mark.parametrize("text,lavi,expected", [
    ("not enlarged", None, "indeterminate"),
    ("possible LAE", None, "indeterminate"),
    ("not enlarged", 40., "positive"),
    ("possible LAE", 25., "negative"),
    ("enlarged", 25., "indeterminate"),
    ("normal lavi", 40., "indeterminate"),
])
def test_la_uncertain_text_preserves_independent_numeric_channel(text, lavi, expected):
    rule = _echo_rules()["echo_lae"]
    with duckdb.connect() as connection:
        state = connection.execute(f"SELECT {rule.state} FROM (SELECT TRUE AS has_tte_30d, "
            "?::VARCHAR AS la_size_category, ?::DOUBLE AS la_volume_index)", [text, lavi]).fetchone()[0]
    assert state == expected


@pytest.mark.parametrize("text,expected", [
    ("not enlarged", "indeterminate"), ("no RAE", "indeterminate"),
    ("possibly enlarged", "indeterminate"), ("not normal", "indeterminate"),
    ("enlarged", "positive"), ("normal", "negative"),
])
def test_ra_negation_is_not_a_positive_or_normal_reference(text, expected):
    rule = _echo_rules()["echo_rae"]
    with duckdb.connect() as connection:
        state = connection.execute(f"SELECT {rule.state} FROM (SELECT TRUE AS has_tte_30d, "
            "?::VARCHAR AS ra_size_category)", [text]).fetchone()[0]
    assert state == expected
