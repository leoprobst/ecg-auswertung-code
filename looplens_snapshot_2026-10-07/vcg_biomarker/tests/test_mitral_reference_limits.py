"""Mitral severity remains distinct from mechanism and direct measurements."""

import duckdb
import pytest

from vcg_biomarker.mimic.labels import _echo_rules


@pytest.mark.parametrize("valve,limits", [
    ("mitral_regurgitation", ["primary_secondary_atrial_mechanism=unverified",
                             "prolapse_and_annular_disjunction=unverified",
                             "reported_severity_not_arrhythmogenic_MVP_diagnosis"]),
    ("mitral_stenosis", ["rheumatic_or_calcific_cause=unverified",
                        "reported_severity_not_measured_LA_pressure_or_electromechanical_delay"]),
])
@pytest.mark.parametrize("suffix", ["", "_severe"])
@pytest.mark.parametrize("severity", [None, 0, 1, 2, 3])
def test_mitral_summary_preserves_missingness_and_reference_limits(valve, limits, suffix, severity):
    rule = _echo_rules()[valve + suffix]
    with duckdb.connect() as connection:
        summary = connection.execute(
            f"SELECT {rule.evidence_summary} FROM "
            f"(SELECT ?::INTEGER AS {valve}_severity, ?::VARCHAR AS {valve}_raw)",
            [severity, None if severity is None else str(severity)],
        ).fetchone()[0]
    assert all(limit in summary for limit in limits)
    assert f"severity={severity if severity is not None else 'missing'};" in summary
