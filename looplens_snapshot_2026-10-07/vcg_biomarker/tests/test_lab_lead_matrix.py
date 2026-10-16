import pytest
from vcg_biomarker.gui.lab_lead_matrix import recorded_lead_rows, HEADERS, lead_matrix_export, st_j_review


@pytest.mark.parametrize('raw', [None, [], {}, {'lead_measurement_I_st_j_mv': float('nan')},
    {'lead_measurement_I_st_j_mv': True}, {'lead_measurement_I_st_j_mv': '0'}])
def test_absent_or_invalid_measurements_are_not_normal(raw):
    rows = recorded_lead_rows({'lead_measurements': raw})
    assert len(rows) == 12
    assert all(len(row['cells']) == len(HEADERS) for row in rows)
    assert rows[0]['cells'][5]['value'] is None


def test_signed_values_zero_and_units_preserved():
    payload = {'lead_measurements': {'lead_measurement_aVR_st_j_mv': -.15, 'lead_measurement_I_st_j_mv': 0.}}
    rows = recorded_lead_rows(payload)
    assert rows[0]['cells'][5]['value'] == 0.
    assert rows[3]['lead'] == 'aVR'
    assert rows[3]['cells'][5]['value'] == -.15
    assert 'mV·s' in HEADERS[4]
    assert payload['lead_measurements']['lead_measurement_aVR_st_j_mv'] == -.15


def test_export_full_precision_and_missing_rows_with_provenance():
    value = -.1234567890123
    payload = {'analysis_id': 'test-analysis', 'lead_measurements': {'lead_measurement_aVR_st_j_mv': value}}
    result = lead_matrix_export(payload, {'source_sha256': 'abc', 'sampling_rate_hz': 500})
    assert result['analysis_id'] == 'test-analysis'
    assert result['signal_reference']['source_sha256'] == 'abc'
    assert len(result['rows']) == 156
    cell = next(row for row in result['rows'] if row['source_key'] == 'lead_measurement_aVR_st_j_mv')
    assert cell['value'] == value
    assert cell['unit'] == 'mV'
    assert sum(row['status'] == 'unavailable' for row in result['rows']) == 155
    assert result['clinical_claim_allowed'] is False


def test_extended_j_values_and_historical_absence():
    rows = recorded_lead_rows({'lead_measurements': {'lead_measurement_I_st_st60_mv': -.123}})
    assert rows[0]['cells'][9]['value'] == -.123
    assert rows[0]['cells'][8]['value'] is None
    assert 'T-Welle' in rows[0]['cells'][9]['tooltip']


def test_st_j_missing_data_not_replaced_or_called_negative():
    result = st_j_review({})
    assert len(result['missing_leads']) == 12
    assert result['native_avr_mv'] is None
    assert '0 von 0' in result['text']
    assert result['clinical_pattern_status'] == 'not_assessed'


def test_st_j_threshold_and_native_avr_are_descriptive_only():
    result = st_j_review({'lead_measurements': {
        'lead_measurement_I_st_j_mv': -.1,
        'lead_measurement_II_st_j_mv': -.0999,
        'lead_measurement_aVR_st_j_mv': .2,
        'lead_measurement_V1_st_j_mv': 0.}})
    assert result['observed_depression_leads'] == ['I']
    assert len(result['missing_leads']) == 8
    assert result['native_avr_mv'] == .2
    assert result['v1_mv'] == 0.
    assert result['clinical_claim_allowed'] is False


def test_six_depressions_do_not_create_unsupported_clinical_pattern():
    values = {f'lead_measurement_{lead}_st_j_mv': -.15 for lead in ('I', 'II', 'III', 'aVL', 'V5', 'V6')}
    values['lead_measurement_aVR_st_j_mv'] = .12
    result = st_j_review({'lead_measurements': values})
    assert result['observed_depression_count'] == 6
    assert result['clinical_pattern_status'] == 'not_assessed'
    assert result['clinical_claim_allowed'] is False
