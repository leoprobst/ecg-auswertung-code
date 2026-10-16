from vcg_biomarker.gui.lab_pwave_summary import accepted_p_wave_summary


def test_beatwise_p_summary_keeps_representative_interval_separate():
    payload = {
        'beat_inventory': [
            {'beat_index': 0, 'accepted': True, 'morphology_cluster_id': 1},
            {'beat_index': 1, 'accepted': True, 'morphology_cluster_id': 1},
            {'beat_index': 2, 'accepted': True, 'morphology_cluster_id': 1},
            {'beat_index': 3, 'accepted': False, 'morphology_cluster_id': 1},
            {'beat_index': 4, 'accepted': True, 'morphology_cluster_id': 2},
        ],
        'delineation': {'interval_measurements': [
            {'interval': 'P duration', 'value_ms': None, 'status': 'not_evaluable'}],
            'accepted_beat_wave_segments': [
                {'beat_index': 0, 'segment': 'p_wave', 'status': 'measurable', 'duration_ms': 90},
                {'beat_index': 1, 'segment': 'p_wave', 'status': 'measurable', 'duration_ms': 110},
                {'beat_index': 2, 'segment': 'p_wave', 'status': 'not_evaluable', 'duration_ms': 140},
                {'beat_index': 3, 'segment': 'p_wave', 'status': 'measurable', 'duration_ms': 200},
                {'beat_index': 4, 'segment': 'p_wave', 'status': 'measurable', 'duration_ms': 80},
            ]},
    }
    result = accepted_p_wave_summary(payload)
    assert result['status'] == 'descriptive'
    assert result['median_ms'] == 100
    assert result['n_beats'] == 2
    assert result['n_accepted_cluster_beats'] == 3
    assert payload['delineation']['interval_measurements'][0]['value_ms'] is None
