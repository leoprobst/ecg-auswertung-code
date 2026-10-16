"""Nullable DuckDB offsets must follow the established float/NaN policy."""
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.survival import (
    CENSORING_LAST_CONTACT, CENSORING_REGISTRY_WINDOW, follow_up,
)


def cohort():
    return pd.DataFrame({
        'subject_id': range(1, 8),
        'died': [0, 1, 1, 1, 1, 1, 1],
        'days_to_death': [np.nan, 30, 400, 20, -1, 10, np.nan],
        'days_to_last_seen': [20, 15, 10, np.nan, 90, -380, 30],
        'days_to_last_contact': [20, 15, 10, np.nan, 90, 10, 30],
        'ever_admitted': [1] * 7,
    })


@pytest.mark.parametrize('dtype', ['Int64', 'Float64'])
@pytest.mark.parametrize('censoring', [CENSORING_REGISTRY_WINDOW, CENSORING_LAST_CONTACT])
def test_nullable_offsets_match_float_policy_and_expected_censoring(dtype, censoring):
    plain = cohort()
    nullable = plain.copy()
    for name in ('days_to_death', 'days_to_last_seen', 'days_to_last_contact'):
        nullable[name] = nullable[name].astype(dtype)
    original = nullable.copy(deep=True)
    actual = follow_up(nullable, censoring=censoring)
    expected = follow_up(plain, censoring=censoring)
    # Normalize representations for this comparison only. The returned input
    # columns intentionally retain their original nullable/non-nullable dtype.
    pd.testing.assert_frame_equal(actual.astype(float), expected.astype(float))
    pd.testing.assert_frame_equal(nullable, original)
    if censoring == CENSORING_REGISTRY_WINDOW:
        assert actual.subject_id.tolist() == [1, 2, 3, 7]
        assert actual.event.tolist() == [0, 1, 0, 0]
        assert actual.time_days.tolist() == [365, 30, 365, 365]
    else:
        assert actual.subject_id.tolist() == [1, 2, 3, 6, 7]
        assert actual.event.tolist() == [0, 0, 0, 1, 0]
        assert actual.time_days.tolist() == [20, 15, 10, 10, 30]


def test_duckdb_nullable_days_with_missing_registry_coverage_do_not_raise():
    import duckdb
    with duckdb.connect() as connection:
        raw = connection.execute('''
            SELECT * FROM (VALUES
                (1, 1, 20, NULL::BIGINT, NULL::BIGINT, 1),
                (2, 0, NULL::BIGINT, 30, 30, 1),
                (3, 1, 10, 30, 30, 1)
            ) t(subject_id,died,days_to_death,days_to_last_seen,
                days_to_last_contact,ever_admitted)
        ''').fetchdf()
    assert raw.days_to_last_seen.isna().any()
    result = follow_up(raw)
    assert result.subject_id.tolist() == [2, 3]
    assert result.event.tolist() == [0, 1]
