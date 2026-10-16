from types import SimpleNamespace as NS
import numpy as np
import pytest
from vcg_biomarker.core.delineation.base import RecordDelineation


def record():
    beats = [NS(r_peak=NS(sample=i*500, confidence=1.),
                segment_confidence=lambda _: 1., qt_interval_ms=lambda _: 300.,
                qrs_duration_ms=lambda _: 90., t_offset=None, r_offset=None)
             for i in range(4)]
    return NS(beats=beats, accepted_beats=beats, fs=500.)


def test_valid_pairs_preserve_original_identity_after_sorting():
    ctx = record()
    ctx.beats = [ctx.beats[2], ctx.beats[0], ctx.beats[3], ctx.beats[1]]
    pairs = RecordDelineation.qt_rr_measurements(ctx)
    assert [p.beat_index for p in pairs] == [3, 0, 2]
    assert all(p.preceding_rr_seconds == 1. for p in pairs)


@pytest.mark.parametrize("bad", [None, np.nan, -1, 0])
def test_unknown_or_duplicate_peak_does_not_bridge(bad):
    ctx = record()
    if bad is None:
        ctx.beats[1].r_peak = None
    else:
        ctx.beats[1].r_peak.sample = bad
    assert RecordDelineation.qt_rr_measurements(ctx) == ()
    assert RecordDelineation.qt_rr_fast_beat_exclusion_count(ctx) == 0


@pytest.mark.parametrize("fs", [np.nan, np.inf, 0., -500.])
def test_invalid_sample_rate_abstains_in_both_pair_queries(fs):
    ctx = record()
    ctx.fs = fs
    assert RecordDelineation.qt_rr_measurements(ctx) == ()
    assert RecordDelineation.qt_rr_fast_beat_exclusion_count(ctx) == 0


@pytest.mark.parametrize("confidence", [np.nan, .1, 2.])
def test_bad_peak_confidence_excludes_both_touching_pairs(confidence):
    ctx = record()
    ctx.beats[1].r_peak.confidence = confidence
    assert [p.beat_index for p in RecordDelineation.qt_rr_measurements(ctx)] == [3]
