from vcg_biomarker.gui.lab_interval_export import interval_export
import json
import pytest


def test_exact_values_formula_and_separate_review_status():
    payload = {'analysis_id': 'one', 'delineation': {'interval_measurements': [
        {'interval': 'QTm', 'value_ms': 422.412345, 'status': 'measurable',
         'formula': 'QT - 0.485 QRS', 'source': 'saved-source', 'measurement_method': 'derived'}]}}
    result = interval_export(payload, {'source_sha256': 'abc'})
    row = result['rows'][0]
    assert row['value_ms'] == 422.412345 and row['formula'] == 'QT - 0.485 QRS'
    assert row['review_status'] == 'automatic_not_clinically_reviewed'
    assert result['manual_overrides_applied'] is False


def test_missing_and_invalid_keep_recorded_status_without_normal_replacement():
    result = interval_export({'delineation': {'interval_measurements': [
        {'interval': 'QT', 'value_ms': float('nan'), 'status': 'measurable'}]}}, {})
    assert result['rows'][0]['value_ms'] is None
    assert result['rows'][0]['recorded_status'] == 'measurable'
    assert result['rows'][0]['value_status'] == 'unavailable_or_invalid'
    assert interval_export({}, {})['rows'] == []


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, '12', -1, None])
def test_invalid_numeric_metadata_is_json_safe(value):
    row = dict(interval='QT', value_ms=410, sampling_rate_hz=value, start_sample=value,
               end_sample=value, representative_r_sample=value, rr_seconds_median=value, n_pairs=value)
    result = interval_export({'delineation': {'interval_measurements': [row]}}, {})
    exported = result['rows'][0]
    for name in ('sampling_rate_hz', 'start_sample', 'end_sample', 'representative_r_sample', 'rr_seconds_median', 'n_pairs'):
        assert exported[name] is None
    assert exported['coordinate_status'] == 'unavailable_or_invalid'
    json.dumps(result, allow_nan=False)


def test_native_zero_sample_preserved_and_inverted_range_not_validated():
    row = dict(interval='QT', value_ms=400, sampling_rate_hz=250, start_sample=0,
               end_sample=100, representative_r_sample=10, rr_seconds_median=.9, n_pairs=3)
    payload = {'delineation': {'interval_measurements': [row]}}
    result = interval_export(payload, {})['rows'][0]
    assert result['start_sample'] == 0 and result['coordinate_status'] == 'recorded'
    row['end_sample'] = 0
    assert interval_export(payload, {})['rows'][0]['coordinate_status'] == 'unavailable_or_invalid'
