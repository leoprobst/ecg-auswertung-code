import duckdb
import pytest

from vcg_biomarker.mimic.labels import _echo_rules


@pytest.mark.parametrize("endpoint,column,text,numeric,value,expected", [
    ("echo_rv_dilation", "rv_size_category", "not dilated", "rv_basal_diameter_cm", None, "indeterminate"),
    ("echo_rv_dilation", "rv_size_category", "possibly dilated", "rv_basal_diameter_cm", 3.5, "negative"),
    ("echo_rv_dilation", "rv_size_category", "not dilated", "rv_basal_diameter_cm", 4.5, "positive"),
    ("echo_rv_dilation", "rv_size_category", "dilated", "rv_basal_diameter_cm", 3.5, "indeterminate"),
    ("echo_rv_systolic_dysfunction", "rv_function_category", "no dysfunction", "tapse_cm", None, "indeterminate"),
    ("echo_rv_systolic_dysfunction", "rv_function_category", "possible dysfunction", "tapse_cm", 2.0, "negative"),
    ("echo_rv_systolic_dysfunction", "rv_function_category", "not normal function", "tapse_cm", 1.2, "positive"),
    ("echo_rv_systolic_dysfunction", "rv_function_category", "normal function", "tapse_cm", 1.2, "indeterminate"),
])
def test_uncertain_text_cannot_override_independent_measurement(endpoint, column, text, numeric, value, expected):
    rule = _echo_rules()[endpoint]
    with duckdb.connect() as connection:
        actual = connection.execute(
            f"SELECT {rule.state} FROM (SELECT TRUE AS has_tte_30d, "
            f"?::VARCHAR AS {column}, ?::DOUBLE AS {numeric})", [text, value],
        ).fetchone()[0]
    assert actual == expected
