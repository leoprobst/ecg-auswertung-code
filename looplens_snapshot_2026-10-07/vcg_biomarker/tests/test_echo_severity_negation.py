import duckdb
import pytest

from vcg_biomarker.mimic.evidence import _severity_sql


@pytest.mark.parametrize("raw,expected", [
    ("not severe", None), ("no severe stenosis", None),
    ("no significant moderate regurgitation", None),
    ("without severe stenosis", None), ("possible severe stenosis", None),
    ("suspected moderate", None), ("questionable mild", None),
    ("not severe; moderate", None), ("cannot exclude severe", None),
    ("severe", 3), ("moderate", 2), ("mild", 1), ("none", 0),
    ("normal", 0), ("mild-moderate", 1), (None, None),
])
def test_negation_is_not_a_positive_echo_grade(raw, expected):
    with duckdb.connect() as connection:
        actual = connection.execute(
            f"SELECT {_severity_sql('raw')} FROM (SELECT ?::VARCHAR AS raw)", [raw]
        ).fetchone()[0]
    assert actual == expected
