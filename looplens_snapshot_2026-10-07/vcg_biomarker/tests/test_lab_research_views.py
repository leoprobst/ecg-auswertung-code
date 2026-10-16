import numpy as np
import pytest

from vcg_biomarker.gui.lab_research_views import beat_amplitude_map, native_spectrogram
from vcg_biomarker.mimic.lab_score_items import lqts_qtc_item


def test_beat_map_native_time_voltage_and_edges():
    signal = np.arange(1000)/100
    result = beat_amplitude_map(signal, 100, [.1, 1., 2., 9.8])
    assert result['edge_excluded'] == 2
    assert result['values_mv'].shape == (2, 76)
    np.testing.assert_allclose(result['values_mv'][:, 25], [1, 2])
    np.testing.assert_allclose(result['relative_seconds'][[0,-1]], [-.25,.5])
    np.testing.assert_array_equal(result['beat_indices'], [1,2])


def test_display_sampling_does_not_claim_all_beats():
    result = beat_amplitude_map(np.zeros(10000), 100, np.arange(1, 99), max_beats=10)
    assert result['sampled_for_display']
    assert len(result['values_mv']) == 10
    assert result['eligible_beats'] == 98


@pytest.mark.parametrize('peaks', [[1,1], [2,1], [float('nan')]])
def test_no_fabricated_r_times(peaks):
    with pytest.raises(ValueError):
        beat_amplitude_map(np.zeros(1000), 100, peaks)


def test_spectrogram_physical_frequency_and_scaling():
    fs = 500
    x = np.sin(2*np.pi*62.5*np.arange(5000)/fs)
    a, b = native_spectrogram(x, fs), native_spectrogram(2*x, fs)
    assert a['frequencies_hz'][-1] <= fs/2
    assert a['frequencies_hz'][np.argmax(a['power_mv2_hz'].mean(axis=1))] == pytest.approx(62.5)
    np.testing.assert_allclose(b['power_mv2_hz'], 4*a['power_mv2_hz'])


def test_missing_signal_is_not_interpolated():
    x = np.zeros(1000); x[100] = np.nan
    with pytest.raises(ValueError):
        native_spectrogram(x, 500)


@pytest.mark.parametrize('qt,sex,expected', [(480,None,3.5),(479.9,None,2),(460,None,2),(459.9,'M',1),(455,'F',0),(455,None,None),(449,None,0)])
def test_lqts_single_item_is_never_a_total_or_probability(qt, sex, expected):
    result = lqts_qtc_item([{'interval':'QTc Bazett','value_ms':qt,'status':'measurable'}], sex)
    assert result['points'] == expected
    assert result['total_score'] is None
    assert result['diagnosis'] is None
    assert result['probability'] is None


def test_lqts_never_substitutes_fridericia_or_abstention():
    assert lqts_qtc_item([{'interval':'QTc Fridericia','value_ms':490,'status':'measurable'}])['points'] is None
    assert lqts_qtc_item([{'interval':'QTc Bazett','value_ms':490,'status':'not_evaluable'}])['points'] is None
