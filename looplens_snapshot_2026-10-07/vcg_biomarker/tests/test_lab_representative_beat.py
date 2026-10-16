import numpy as np
import pytest
from vcg_biomarker.gui.lab_representative_beat import representative_window, cluster_median, median_export, manual_template_interval


def test_manual_template_interval_sample_grid_and_hash_binding():
    data, inventory = median_inputs()
    template = cluster_median(data, inventory, 0)
    reference = {'source_sha256': 'abc'}
    result = manual_template_interval(template, reference, 'a', 1., 39., 'QRS', 'Reviewer', 'II')
    assert result['start_ms'] == 0
    assert result['end_ms'] == 40
    assert result['duration_ms'] == 40
    assert result['coordinate_system'] == 'synthetic_template_samples'
    assert result['template_sha256'] == median_export(template, reference, 'a')['template_sha256']
    assert result['template_sha256'] != median_export(template, reference, 'b')['template_sha256']
    assert result['clinical_claim_allowed'] is False


@pytest.mark.parametrize('start,end', [(0, 0), (3, 1), (-10000, 20), (0, 10000), (None, 20), (True, 20), (0, float('nan')), (.1, .2)])
def test_invalid_or_subsample_intervals_rejected(start, end):
    data, inventory = median_inputs()
    with pytest.raises(ValueError):
        manual_template_interval(cluster_median(data, inventory, 0), {'source_sha256': 'abc'}, 'a', start, end, 'QRS', 'Reviewer', 'II')


def test_template_interval_requires_author_and_known_measurement_scope():
    data, inventory = median_inputs()
    template = cluster_median(data, inventory, 0)
    for author, scope in [('', 'II'), ('Reviewer', 'unknown')]:
        with pytest.raises(ValueError):
            manual_template_interval(template, {'source_sha256': 'abc'}, 'a', 0, 20, 'QT', author, scope)


def median_inputs():
    signal = np.zeros((1100, 12))
    inventory = []
    for index, peak in enumerate((100, 300, 500, 700, 900)):
        signal[peak] = [1, 3, 2, 80, 100][index]
        inventory.append(dict(beat_index=index, r_peak_sample=peak,
                              morphology_cluster_id=0 if index < 4 else 1,
                              accepted=index != 3))
    return {'diagnostic_ecg': signal, 'sampling_rate_hz': np.array(250)}, inventory


def test_median_uses_only_accepted_same_cluster_and_records_provenance():
    data, inventory = median_inputs()
    original = data['diagnostic_ecg'].copy()
    result = cluster_median(data, inventory, 0)
    assert result['available']
    center = np.where(result['time_ms'] == 0)[0][0]
    np.testing.assert_array_equal(result['signal'][center], np.full(12, 2))
    assert [item['beat_index'] for item in result['used_beats']] == [0, 1, 2]
    assert {item['reason'] for item in result['excluded_beats']} == {'nicht_akzeptiert', 'anderes_oder_unbekanntes_cluster'}
    assert result['fiducials_ms'] == {}
    np.testing.assert_array_equal(data['diagnostic_ecg'], original)


def test_median_linear_scaling_offset_and_reordered_inventory():
    data, inventory = median_inputs()
    first = cluster_median(data, inventory, 0)
    data['diagnostic_ecg'] = data['diagnostic_ecg']*2 + 5
    second = cluster_median(data, inventory[::-1], 0)
    np.testing.assert_allclose(second['signal'], first['signal']*2+5)
    assert first['used_beats'] == second['used_beats']


def test_median_missing_quality_and_missing_cluster_not_assumed():
    data, inventory = median_inputs()
    assert not cluster_median(data, inventory, None)['available']
    assert not cluster_median(data, inventory, 1)['available']
    inventory[0].pop('accepted')
    assert not cluster_median(data, inventory, 0)['available']


def test_nonfinite_window_does_not_become_normal_or_nanmedian():
    data, inventory = median_inputs()
    data['diagnostic_ecg'][300, 0] = np.nan
    assert not cluster_median(data, inventory, 0)['available']


def test_duplicate_beat_identity_rejected():
    data, inventory = median_inputs()
    assert not cluster_median(data, inventory + [inventory[0]], 0)['available']


def test_long_record_not_computed_in_gui():
    data, inventory = median_inputs()
    data['diagnostic_ecg'] = np.zeros((15001, 12))
    assert not cluster_median(data, inventory, 0)['available']


def test_median_export_precision_provenance_and_no_clinical_fiducials():
    import json
    data, inventory = median_inputs()
    template = cluster_median(data, inventory, 0)
    document = json.loads(json.dumps(median_export(template, {'source_sha256': 'abc'}, 'analysis-id'), allow_nan=False))
    np.testing.assert_array_equal(document['values'], template['signal'])
    assert document['provenance']['used_beats'] == template['used_beats']
    assert document['clinical_claim_allowed'] is False
    assert document['fiducials'] is None
    assert document['lead_order'][3] == 'aVR'


@pytest.mark.parametrize('reference,analysis', [({}, 'a'), ({'source_sha256': 'abc'}, ''), ({'source_sha256': 'abc'}, None)])
def test_median_export_requires_traceable_inputs(reference, analysis):
    data, inventory = median_inputs()
    with pytest.raises(ValueError):
        median_export(cluster_median(data, inventory, 0), reference, analysis)


def arrays(fs=250):
    return dict(diagnostic_ecg=np.arange(24000).reshape(2000, 12), sampling_rate_hz=np.array(fs),
                fiducial_names=np.array(['p_onset', 'r_peak', 't_offset']),
                fiducial_samples=np.array([200, 300, 400]))


def test_saved_samples_not_interpolated_averaged_or_inverted():
    data = arrays()
    before = data['diagnostic_ecg'].copy()
    result = representative_window(data)
    assert result['available']
    assert result['fiducials_ms'] == dict(p_onset=-400, r_peak=0, t_offset=400)
    assert np.diff(result['time_ms']) == pytest.approx(4)
    row = np.where(result['time_ms'] == 0)[0][0]
    np.testing.assert_array_equal(result['signal'][row], before[300])
    np.testing.assert_array_equal(data['diagnostic_ecg'], before)


@pytest.mark.parametrize('fs', [0, -1, np.nan, np.inf, True, '500', [250, 500]])
def test_no_invented_sampling_rate(fs):
    assert not representative_window(arrays(fs))['available']


def test_missing_path_not_replaced_by_other_definition():
    assert not representative_window(arrays(), 'morphology_ecg')['available']


@pytest.mark.parametrize('samples', [[200, 300.5, 400], [200, -1, 400], [200, 300, 2000], [200, np.nan, 400], [200, 300]])
def test_invalid_fiducials_abstain(samples):
    data = arrays()
    data['fiducial_samples'] = np.array(samples)
    assert not representative_window(data)['available']


def test_missing_r_or_duplicate_name_abstains():
    data = arrays()
    data['fiducial_names'] = np.array(['p_onset', 'q_peak', 't_offset'])
    assert not representative_window(data)['available']
    data['fiducial_names'] = np.array(['r_peak', 'r_peak', 't_offset'])
    assert not representative_window(data)['available']
