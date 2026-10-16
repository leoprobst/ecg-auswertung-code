from dataclasses import replace

import pytest

from vcg_biomarker.mimic.within_subject import DRUG_WINDOW, _pair_quality_clause


@pytest.mark.parametrize("minimum", [-.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_normalized_quality_threshold_is_rejected(minimum):
    with pytest.raises(ValueError, match="signal quality"):
        _pair_quality_clause(replace(DRUG_WINDOW, minimum_signal_quality=minimum))


def test_zero_threshold_still_requires_finite_observed_scores():
    import duckdb
    query = "SELECT " + _pair_quality_clause(replace(DRUG_WINDOW, minimum_signal_quality=0))
    query += " FROM (SELECT NULL::DOUBLE AS signal_quality_score) a CROSS JOIN (SELECT 1. AS signal_quality_score) b"
    with duckdb.connect() as con:
        assert con.execute(query).fetchone()[0] is None  # SQL WHERE excludes unknown, does not impute zero
