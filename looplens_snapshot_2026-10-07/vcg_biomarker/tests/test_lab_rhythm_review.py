import pytest
from vcg_biomarker.gui.lab_rhythm_review import rhythm_review, RHYTHM_TEXT


def payload(kind='regular', **context):
    return {'analysis_id': 'a', 'delineation': {'rhythm_context': {'class': kind, **context}}}


@pytest.mark.parametrize('kind', list(RHYTHM_TEXT))
def test_all_saved_classes_have_explicit_non_diagnostic_statements(kind):
    result = rhythm_review(payload(kind, confidence=.9, rr_median_ms=800))
    assert result['statement_code'] == 'technical_rhythm.' + kind
    assert result['probability'] is None
    assert result['clinical_claim_allowed'] is False
    assert result['rate_from_median_rr_bpm'] == 75


def test_regular_does_not_become_sinus():
    result = rhythm_review(payload())
    assert 'Sinusursprung nicht allein' in result['text']


@pytest.mark.parametrize('value', [None, 0, -1, True, '800', float('nan'), float('inf')])
def test_invalid_rr_does_not_create_rate(value):
    assert rhythm_review(payload(rr_median_ms=value))['rate_from_median_rr_bpm'] is None


@pytest.mark.parametrize('value', [None, -.1, 1.1, True, '.8', float('nan')])
def test_invalid_confidence_is_not_probability(value):
    assert rhythm_review(payload(confidence=value))['technical_confidence'] is None


def test_missing_context_and_unknown_schema_do_not_become_normal():
    for data in ({}, {'delineation': None}, payload('future_class')):
        result = rhythm_review(data)
        assert result['statement_code'] == 'technical_rhythm.unavailable'
        assert result['rate_from_median_rr_bpm'] is None


def test_af_restrictions_and_zero_coverage_visible():
    result = rhythm_review(payload('af_compatible_pattern', p_wave_coverage=0, flags=['no_consistent_discrete_p']))
    assert result['p_wave_coverage'] == 0
    assert len(result['restrictions']) == 2
    assert 'keine bestätigte AF-Diagnose' in result['text']
    assert result['flags'] == ['no_consistent_discrete_p']
