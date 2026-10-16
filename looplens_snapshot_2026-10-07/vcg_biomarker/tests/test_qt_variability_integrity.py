from types import SimpleNamespace as NS

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers.electrophysiology_markers import _beat_to_beat_qt_variability
from vcg_biomarker.core.biomarkers.electrophysiology_markers import _qt_variability_normalised


def test_surrogate_pairs_current_qt_with_preceding_rr_without_gap_bridging():
    peaks = [100, 600, 1150, 1650, 2250]
    durations = [150, 160, 170, 180, 190]
    beats = [NS(r_peak=NS(sample=r, confidence=1.),
                r_onset=NS(sample=r-20, confidence=1.),
                t_offset=NS(sample=r-20+d, confidence=1.))
             for r, d in zip(peaks, durations)]
    ctx = NS(fs=500., delineation=NS(beats=beats))
    expected = np.std(durations[1:], ddof=1) / np.std(np.diff(peaks), ddof=1)
    assert _qt_variability_normalised(ctx).value == pytest.approx(expected)
    beats[2].r_peak = None
    assert _qt_variability_normalised(ctx).value is None


def context():
    beats = [NS(r_onset=NS(sample=i*500, confidence=1.),
                q_peak=NS(sample=i*500+20, confidence=1.),
                t_offset=NS(sample=i*500+150+i*5, confidence=1.)) for i in range(4)]
    return NS(fs=500., delineation=NS(beats=beats))


def test_qt_sd_matches_known_intervals():
    assert _beat_to_beat_qt_variability(context()).value == pytest.approx(np.std([300, 310, 320, 330], ddof=1))


@pytest.mark.parametrize("invalid", ["missing_onset", "negative_duration", "nan_confidence", "nan_sample", "nan_fs"])
def test_invalid_intervals_do_not_produce_variability(invalid):
    ctx = context()
    beat = ctx.delineation.beats[0]
    if invalid == "missing_onset":
        beat.r_onset = None
    elif invalid == "negative_duration":
        beat.t_offset.sample = -1
    elif invalid == "nan_confidence":
        beat.r_onset.confidence = np.nan
    elif invalid == "nan_sample":
        beat.t_offset.sample = np.nan
    else:
        ctx.fs = np.nan
    assert _beat_to_beat_qt_variability(ctx).value is None
