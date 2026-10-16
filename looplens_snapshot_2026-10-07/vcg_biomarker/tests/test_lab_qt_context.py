import pytest
from vcg_biomarker.gui.lab_qt_context import qt_context


def row(name, value, **kwargs):
    return dict(interval=name, value_ms=value, status='measurable', **kwargs)


def test_wide_qrs_and_rate_are_context_not_diagnosis():
    result = qt_context([row('QRS duration', 120), row('QTc Bazett', 480, rr_seconds_median=.4)])
    assert 'Breiter QRS' in ' '.join(result['notes'])
    assert '150.0/min' in ' '.join(result['notes'])
    assert result['diagnosis'] is None and result['recalculated'] is False


@pytest.mark.parametrize('value', [None, 310, 320, 340, 360, 390, 440])
def test_sqts_reference_is_not_a_formula_agnostic_diagnosis(value):
    result = qt_context([row('QTc Hodges', value)], age=45, sex='M')
    context = result['short_qt_context']
    assert context['diagnosis'] is None
    assert context['status'] == 'clinical_context_not_assessed'
    assert context['automatic_threshold_applied'] is False
    assert context['recommendation_table'] == 46
    assert '390 ms ist keine SQTS-Diagnosegrenze' in ' '.join(result['notes'])
    assert 'ehac262' in context['source']


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf'), -1, '120'])
def test_invalid_qrs_is_unknown(value):
    assert 'QRS-Dauer unbekannt' in qt_context([row('QRS duration', value)])['notes'][0]


def test_duplicate_and_unmeasurable_not_used():
    assert 'unbekannt' in qt_context([row('QRS duration', 90), row('QRS duration', 130)])['notes'][0]
    r = row('QRS duration', 130)
    r['status'] = 'not_evaluable'
    assert 'unbekannt' in qt_context([r])['notes'][0]


def test_missing_rr_not_replaced_by_other_heart_rate():
    result = qt_context([row('QRS duration', 90), row('QTc Bazett', 470)])
    assert 'zugehöriges RR unbekannt' in ' '.join(result['notes'])


@pytest.mark.parametrize('rr', [1., .6])
def test_rate_warning_boundaries(rr):
    result = qt_context([row('QRS duration', 90), row('QTc Bazett', 470, rr_seconds_median=rr)])
    assert 'außerhalb' not in ' '.join(result['notes'])


@pytest.mark.parametrize('sex,value,expected', [
    ('M', 450, 'prolonged'), ('F', 459.9, 'within_reference_limits'),
    ('F', 460, 'prolonged'), ('M', 390, 'short'),
    ('M', 390.1, 'within_reference_limits')])
def test_adult_reference_boundaries(sex, value, expected):
    result = qt_context([row('QRS duration', 90), row('QTc Framingham', value)],
                        age=45, sex=sex, rhythm={'rr_cv': .05})
    assert result['classifications'][0]['status'] == expected
    assert result['classifications'][1]['status'] == 'not_evaluable'
    assert result['diagnosis'] is None


@pytest.mark.parametrize('override', [
    {'age': None}, {'age': 17}, {'age': True}, {'sex': None},
    {'rhythm': None}, {'rhythm': {'rr_cv': .11}}, {'rhythm': {'rr_cv': True}}])
def test_missing_or_ineligible_context_never_normal(override):
    context = dict(age=45, sex='F', rhythm={'rr_cv': .05})
    context.update(override)
    result = qt_context([row('QRS duration', 90), row('QTc Framingham', 420)], **context)
    assert result['classifications'][0]['status'] == 'not_evaluable'


def test_wide_qrs_blocks_adult_reference_and_methods_not_pooled():
    rows = [row('QRS duration', 120), row('QTc Framingham', 420), row('QTc Hodges', 480)]
    result = qt_context(rows, age=45, sex='F', rhythm={'rr_cv': .05})
    assert all(c['status'] == 'not_evaluable' for c in result['classifications'])
    rows[0]['value_ms'] = 90
    result = qt_context(rows, age=45, sex='F', rhythm={'rr_cv': .05})
    assert [c['status'] for c in result['classifications']] == ['within_reference_limits', 'prolonged']
