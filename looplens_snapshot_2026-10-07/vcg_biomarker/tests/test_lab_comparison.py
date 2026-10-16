from copy import deepcopy

import pytest

from vcg_biomarker.mimic.lab_comparison import paired_intervals


def payload(value=100):
    return {"measurement_protocol": {"implementation_hash": "abc", "schema_version": "1"},
            "lab_processing": {"analysis_sampling_rate_hz": 500, "powerline_hz": 0},
            "delineation": {"interval_measurements": [{"interval": "QRS duration", "value_ms": value,
                                                       "status": "measurable"}]}}


def test_paired_difference_not_chronological_or_diagnostic():
    row = paired_intervals(payload(110), payload(90))[0]
    assert row['delta_current_minus_comparator'] == 20
    assert row['status'] == 'descriptive_only'


@pytest.mark.parametrize('protocol', ['same', True, {'schema_version': '1'}, {'schema_version':'1', 'implementation_hash':''}])
def test_matching_but_invalid_protocol_is_not_compatibility(protocol):
    a, b = payload(110), payload(90)
    a['measurement_protocol'] = b['measurement_protocol'] = protocol
    assert paired_intervals(a,b)[0]['delta_current_minus_comparator'] is None


@pytest.mark.parametrize('processing', [None, [], {'analysis_sampling_rate_hz':0, 'powerline_hz':0}, {'analysis_sampling_rate_hz':500, 'powerline_hz':True}])
def test_invalid_processing_never_generates_delta(processing):
    a,b = payload(),payload()
    a['lab_processing'] = b['lab_processing'] = processing
    assert paired_intervals(a,b)[0]['delta_current_minus_comparator'] is None


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), True, '100'])
def test_invalid_value_is_not_zero(value):
    row = paired_intervals(payload(value), payload())[0]
    assert row['current'] is None
    assert row['delta_current_minus_comparator'] is None


@pytest.mark.parametrize('change', ['protocol', 'filter', 'status', 'method', 'duplicate', 'missing'])
def test_incompatible_or_unavailable_measurement_has_no_delta(change):
    other = deepcopy(payload())
    if change == 'protocol':
        other.pop('measurement_protocol')
    elif change == 'filter':
        other['lab_processing']['powerline_hz'] = 60
    elif change == 'status':
        other['delineation']['interval_measurements'][0]['status'] = 'abstain'
    elif change == 'method':
        other['delineation']['interval_measurements'][0]['measurement_method'] = 'other'
    elif change == 'duplicate':
        other['delineation']['interval_measurements'] *= 2
    else:
        other['delineation']['interval_measurements'] = []
    assert paired_intervals(payload(), other)[0]['delta_current_minus_comparator'] is None
