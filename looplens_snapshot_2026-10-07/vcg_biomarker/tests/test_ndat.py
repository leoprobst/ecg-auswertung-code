import numpy as np
import pytest

from vcg_biomarker.core.biomarkers.ndat import negative_downslope_times, NDAT_LEADS
from vcg_biomarker.core.data_loader import LEAD_MAP


def record(fs=500):
    t = np.arange(fs*8) / fs
    signal = np.zeros((len(t), 12))
    bounds = []
    for start in (1., 3., 5.):
        bounds.append((int(start*fs), int((start+.12)*fs)))
        for i, lead in enumerate(NDAT_LEADS):
            signal[:, LEAD_MAP[lead]] += np.exp(-.5*((t-start-.04-i*.004)/.012)**2)
    return signal, bounds


@pytest.mark.parametrize("fs", [250, 500, 1000])
def test_known_downslope_delays(fs):
    signal, bounds = record(fs)
    result = negative_downslope_times(signal, fs, bounds)
    assert result["status"] == "evaluable"
    assert result["order"] == list(NDAT_LEADS)
    assert result["spread_ms"] == pytest.approx(20., abs=1000/fs)
    for i, lead in enumerate(NDAT_LEADS):
        assert result["leads"][lead]["time_ms"] == pytest.approx(52+i*4, abs=1000/fs)


def test_offset_and_scale_do_not_change_timing_for_good_signal():
    signal, bounds = record()
    first = negative_downslope_times(signal, 500, bounds)
    second = negative_downslope_times(signal*2+3, 500, bounds)
    assert second["status"] == "evaluable"
    assert second["spread_ms"] == first["spread_ms"]
    assert second["beat_times_ms"] == first["beat_times_ms"]


def test_missing_lead_and_pacing_abstain():
    signal, bounds = record()
    signal[:, LEAD_MAP["V6"]] = 0
    result = negative_downslope_times(signal, 500, bounds)
    assert result["status"] == "not_evaluable"
    assert result["spread_ms"] is None
    assert result["leads"]["V6"]["time_ms"] is None
    result = negative_downslope_times(signal, 500, bounds, pacing_present=True)
    assert result["reason"] == "pacing_requires_separate_validated_protocol"


def test_duplicate_beats_cannot_manufacture_repeatability():
    signal, bounds = record()
    result = negative_downslope_times(signal, 500, [bounds[0]]*3)
    assert result["reason"] == "invalid_or_overlapping_qrs_bounds"


def test_nonfinite_signal_and_insufficient_bandwidth_fail_closed():
    signal, bounds = record()
    assert negative_downslope_times(signal, 200, bounds)["status"] == "not_evaluable"
    signal[400, LEAD_MAP["V1"]] = np.nan
    assert negative_downslope_times(signal, 500, bounds)["reason"] == "nonfinite_precordial_signal"


def test_one_shifted_beat_is_not_hidden_by_zero_mad():
    signal, bounds = record()
    lead = LEAD_MAP["V1"]
    first, last = 2400, 2700
    signal[first:last, lead] = np.roll(signal[first:last, lead], 15)
    result = negative_downslope_times(signal, 500, bounds)
    assert result["leads"]["V1"]["status"] == "not_evaluable"
    assert result["spread_ms"] is None
