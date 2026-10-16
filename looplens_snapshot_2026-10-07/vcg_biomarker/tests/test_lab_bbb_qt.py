import pytest
from vcg_biomarker.mimic.lab_bbb_qt import bogossian_modified_qt


def inputs():
    common = dict(status='measurable', measurement_method='representative_beat_fiducials',
        representative_r_sample=500, sampling_rate_hz=500, start_sample=450)
    rows = [dict(common, interval='QT interval', value_ms=500, end_sample=700),
            dict(common, interval='QRS duration', value_ms=160, end_sample=530)]
    context = dict(schema_version='1.0.0', case_id='case', context='lbbb',
                   author='reviewer', reason='morphology checked', created_at='2026-09-22', revision_id='one')
    return rows, context


def test_original_formula_no_rate_correction_and_no_mutation():
    rows, context = inputs()
    result = bogossian_modified_qt(rows, context, 'case', age=45)
    assert result['value_ms'] == pytest.approx(422.4)
    assert rows[0]['value_ms'] == 500
    assert result['context_revision_id'] == 'one'
    assert 'nicht QTc' in result['interpretation_note']


@pytest.mark.parametrize('change', ['case', 'pacing', 'missing', 'duplicate', 'other_beat', 'other_onset', 'narrow', 'nan'])
def test_ineligible_inputs_abstain(change):
    rows, context = inputs()
    if change == 'case': context['case_id'] = 'other'
    if change == 'pacing': context['context'] = 'ventricular_pacing'
    if change == 'missing': context = None
    if change == 'duplicate': rows.append(dict(rows[0]))
    if change == 'other_beat': rows[1]['representative_r_sample'] = 700
    if change == 'other_onset': rows[1]['start_sample'] = 455
    if change == 'narrow': rows[1]['value_ms'] = 100
    if change == 'nan': rows[0]['value_ms'] = float('nan')
    result = bogossian_modified_qt(rows, context, 'case', age=45)
    assert result['value_ms'] is None and result['status'] == 'not_evaluable'


@pytest.mark.parametrize('age', [None, 17, True, '45', float('nan')])
def test_adult_eligibility(age):
    rows, context = inputs()
    assert bogossian_modified_qt(rows, context, 'case', age=age)['value_ms'] is None


@pytest.mark.parametrize('key,value', [('end_sample', 699), ('sampling_rate_hz', 0),
                                    ('start_sample', True), ('representative_r_sample', 800)])
def test_coordinate_consistency(key, value):
    rows, context = inputs()
    for row in rows:
        row[key] = value
    result = bogossian_modified_qt(rows, context, 'case', age=45)
    assert result['value_ms'] is None
