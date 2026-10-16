import numpy as np
import pytest
from vcg_biomarker.mimic.lab_qrs_timing import simple_qrs_timing


def test_known_crossings_and_units():
    result = simple_qrs_timing([0,.5,1,.5,0,-.5,-1,-.5,0], 1000)
    assert result['r_peak_time'] == 2
    assert result['s_duration'] == 4


@pytest.mark.parametrize('signal', [[0,-1,0], [0,1,1,0], [0,1,0,1,0], [0,.01,0], [0,float('nan'),0]])
def test_ambiguous_or_absent_r_abstains(signal):
    assert np.isnan(simple_qrs_timing(signal,500)['r_peak_time'])


def test_truncated_s_not_closed_at_artificial_window_end():
    result = simple_qrs_timing([0,1,0,-1,-.5],500)
    assert result['r_peak_time'] == 2
    assert np.isnan(result['s_duration'])


def test_sampling_rate_and_amplitude_scaling():
    signal = np.array([0,.5,1,.5,0,-.5,-1,-.5,0])
    a = simple_qrs_timing(signal,500)
    b = simple_qrs_timing(signal*2,1000)
    assert a['r_peak_time'] == 2*b['r_peak_time']
    assert a['s_duration'] == 2*b['s_duration']
