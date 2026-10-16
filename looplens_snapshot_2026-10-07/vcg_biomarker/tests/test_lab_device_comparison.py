import pytest

from vcg_biomarker.gui.lab_device_comparison import (
    validate_device_intervals, device_interval_comparison,
)


def document():
    return dict(schema_version='1.0.0', case_id='case', source_sha256='hash',
                device_source='Local device report', measurements=[
                    dict(interval='QTc Fridericia', value_ms=420, unit='ms')])


REFERENCE = dict(case_id='case', source_sha256='hash')


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf'), 0, -1, '420'])
def test_invalid_device_measurement(value):
    doc = document()
    doc['measurements'][0]['value_ms'] = value
    with pytest.raises(ValueError):
        validate_device_intervals(doc, REFERENCE)


@pytest.mark.parametrize('key', ['case_id', 'source_sha256'])
def test_missing_or_mismatched_identity(key):
    doc = document()
    doc[key] = 'other'
    with pytest.raises(ValueError):
        validate_device_intervals(doc, REFERENCE)
    del doc[key]
    reference = {k: v for k, v in REFERENCE.items() if k != key}
    with pytest.raises(ValueError):
        validate_device_intervals(doc, reference)


def test_exact_formula_matching_and_no_ambiguous_own_value():
    row = dict(interval='QTc Fridericia', value_ms=430, status='measurable')
    payload = {'delineation': {'interval_measurements': [row]}}
    assert device_interval_comparison(payload, document(), REFERENCE)[0]['difference_ms'] == 10
    payload['delineation']['interval_measurements'].append(row.copy())
    assert device_interval_comparison(payload, document(), REFERENCE)[0]['own_ms'] is None
    row['interval'] = 'QTc Bazett'
    payload['delineation']['interval_measurements'] = [row]
    assert device_interval_comparison(payload, document(), REFERENCE)[0]['own_ms'] is None


def test_units_duplicates_and_empty_analysis():
    doc = document()
    doc['measurements'][0]['unit'] = 's'
    with pytest.raises(ValueError):
        validate_device_intervals(doc, REFERENCE)
    doc = document()
    doc['measurements'] *= 2
    with pytest.raises(ValueError):
        validate_device_intervals(doc, REFERENCE)
    assert device_interval_comparison(None, document(), REFERENCE)[0]['own_ms'] is None
