import duckdb
import pytest

from vcg_biomarker.mimic.labels import _echo_rules


@pytest.mark.parametrize("endpoint", ["severe_aortic_stenosis_velocity", "moderate_aortic_stenosis_velocity"])
@pytest.mark.parametrize("velocity", [2.5, 3.5, 4.5, None])
def test_velocity_reference_exposes_target_not_anatomic_exclusion(endpoint, velocity):
    rule = _echo_rules()[endpoint]
    with duckdb.connect() as connection:
        summary = connection.execute(
            f"SELECT {rule.evidence_summary} FROM (SELECT ?::DOUBLE AS aortic_peak_velocity_m_per_s)",
            [velocity],
        ).fetchone()[0]
    assert "velocity_threshold_only" in summary
    assert "below_threshold_does_not_exclude_anatomic_severe_AS" in summary
    assert "flow_and_rhythm_applicability=unverified" in summary
