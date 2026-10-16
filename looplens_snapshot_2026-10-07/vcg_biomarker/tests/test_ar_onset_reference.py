import duckdb
import pytest
from vcg_biomarker.mimic.labels import _echo_rules


@pytest.mark.parametrize("endpoint", ["aortic_regurgitation", "aortic_regurgitation_severe"])
def test_severity_does_not_certify_onset_or_cause(endpoint):
    expression = _echo_rules()[endpoint].evidence_summary
    with duckdb.connect() as connection:
        text = connection.execute(f"SELECT {expression} FROM (SELECT 3 AS aortic_regurgitation_severity, 'severe' AS aortic_regurgitation_raw)").fetchone()[0]
    assert "onset_and_cause=unverified" in text
    assert "reported_severity_not_acute_AR_diagnosis" in text
