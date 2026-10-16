import numpy as np
import pytest
from vcg_biomarker.gui.lab_array_contract import validate_visualization_arrays


def test_valid_coordinates_and_missing_samples_preserved():
    signal = np.zeros((100, 12))
    signal[40, 0] = np.nan
    arrays = dict(sampling_rate_hz=np.asarray(500), r_peaks=np.array([0, 99]), morphology_ecg=signal)
    rate, peaks = validate_visualization_arrays(arrays)
    assert rate == 500 and peaks.tolist() == [0, 99]
    assert np.isnan(signal[40, 0])  # No interpolation or invented measurement.


@pytest.mark.parametrize('signal', [np.zeros((12, 100)), np.zeros((100, 8)), np.zeros(100), np.zeros((0, 12))])
def test_wrong_signal_shape_rejected(signal):
    with pytest.raises(ValueError, match='Signalform'):
        validate_visualization_arrays(dict(sampling_rate_hz=500, morphology_ecg=signal))


def test_mismatched_lengths_and_out_of_range_peak_rejected():
    arrays = dict(sampling_rate_hz=500, morphology_ecg=np.zeros((100, 12)), r_peaks=[100])
    with pytest.raises(ValueError, match='außerhalb'):
        validate_visualization_arrays(arrays)
    arrays['r_peaks'] = [99]
    arrays['record_svd_components'] = np.zeros((90, 3))
    with pytest.raises(ValueError, match='Zeitachsen'):
        validate_visualization_arrays(arrays)


@pytest.mark.parametrize('extra', [
    dict(fiducial_names=['r_peak']),
    dict(fiducial_names=['r_peak', 'r_peak'], fiducial_samples=[10, 20]),
    dict(fiducial_names=['r_peak'], fiducial_samples=[10, 20]),
    dict(segment_names=['qrs'], segment_start_samples=[20], segment_end_samples=[10]),
])
def test_invalid_landmark_contract(extra):
    with pytest.raises(ValueError):
        validate_visualization_arrays(dict(sampling_rate_hz=500, **extra))


def test_repeated_segment_names_are_valid_for_different_beats():
    validate_visualization_arrays(dict(sampling_rate_hz=500, segment_names=['qrs', 'qrs'],
                                      segment_start_samples=[10, 50], segment_end_samples=[20, 60]))
