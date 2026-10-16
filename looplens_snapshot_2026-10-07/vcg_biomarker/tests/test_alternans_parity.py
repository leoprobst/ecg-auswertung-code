"""Regression for detected-beat parity after missing segment exclusion."""
from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import catalog_biomarkers as module
from vcg_biomarker.core.delineation.base import RhythmAssessment, RhythmClass


@pytest.mark.parametrize("name", ["qrs_alternans_vector", "t_wave_alternans_vector", "microvolt_t_wave_alternans_vector"])
def test_missing_middle_beat_abstains_instead_of_renumbering(monkeypatch, name):
    ctx = SimpleNamespace(fs=500, delineation=SimpleNamespace(beats=[
        SimpleNamespace(r_peak=SimpleNamespace(sample=i * 500)) for i in range(6)]))
    # Beat 2 missing; compressed odd/even groups would no longer be correct.
    curves = [np.tile([1. + i % 2, 0., 0.], (64, 1)) for i in (0, 1, 3, 4, 5)]
    monkeypatch.setattr(module, "_beat_segment_curves", lambda *args, **kwargs: curves)
    function = module._literature_feature if name.startswith("microvolt") else module._beat_variability_feature
    result = function(ctx, name)
    assert result.value is None
    assert "no parity compression" in result.notes


def test_complete_sequence_retains_microvolt_difference(monkeypatch):
    ctx = SimpleNamespace(delineation=SimpleNamespace(beats=list(range(6))))
    curves = [np.tile([1. + .002 * (i % 2), 0., 0.], (64, 1)) for i in range(6)]
    monkeypatch.setattr(module, "_beat_segment_curves", lambda *args, **kwargs: curves)

    def redundant_interpolation(*args, **kwargs):
        pytest.fail("Already time-normalized loops must not be interpolated again")
    monkeypatch.setattr(module.np, "interp", redundant_interpolation)

    result = module._literature_feature(ctx, "microvolt_t_wave_alternans_vector")
    assert result.value == pytest.approx(2.)
    assert "rhythm_pattern=unknown" in result.notes
    assert "n_beats=6" in result.notes
    assert "not_validated_clinical_TWA" in result.notes


def test_real_segment_loader_missing_t_wave_does_not_compress_parity():
    point = lambda sample: SimpleNamespace(sample=sample, confidence=1.)
    beats = [SimpleNamespace(t_onset=point(i * 20), t_offset=point(i * 20 + 10))
             for i in range(6)]
    beats[2].t_onset = None
    ctx = SimpleNamespace(delineation=SimpleNamespace(beats=beats),
                          _record_vcg_xyz=np.column_stack((np.arange(130) / 130.,
                                                         np.zeros(130), np.ones(130))))
    assert len(module._beat_segment_curves(ctx, "t")) == 5
    assert module._literature_feature(ctx, "microvolt_t_wave_alternans_vector").value is None


def test_temporal_alternans_is_not_erased_by_equal_spatial_paths():
    point = lambda sample: SimpleNamespace(sample=sample, confidence=1.)
    beats = [SimpleNamespace(t_onset=point(i * 21), t_offset=point(i * 21 + 20))
             for i in range(6)]
    phase = np.linspace(0., 1., 21)
    record = np.vstack([np.column_stack((phase ** (1 + i % 2),
                                        np.zeros(21), np.zeros(21))) for i in range(6)])
    ctx = SimpleNamespace(delineation=SimpleNamespace(beats=beats), _record_vcg_xyz=record)
    geometric = module._beat_segment_curves(ctx, "t")
    np.testing.assert_allclose(geometric[0], geometric[1], atol=1e-14)
    assert module._literature_feature(ctx, "microvolt_t_wave_alternans_vector").value > 100.


@pytest.mark.parametrize("sample,confidence", [(-1, 1.), (1000, 1.), (2.5, 1.),
                                               (0, 0.), (0, float("nan"))])
def test_alternans_rejects_invalid_fiducials_without_clamping(sample, confidence):
    point = lambda n: SimpleNamespace(sample=n, confidence=1.)
    beats = [SimpleNamespace(t_onset=point(i * 20), t_offset=point(i * 20 + 10))
             for i in range(6)]
    beats[0].t_onset = SimpleNamespace(sample=sample, confidence=confidence)
    ctx = SimpleNamespace(delineation=SimpleNamespace(beats=beats),
                          _record_vcg_xyz=np.ones((130, 3)))
    assert module._literature_feature(ctx, "microvolt_t_wave_alternans_vector").value is None


@pytest.mark.parametrize("rhythm_class", [RhythmClass.IRREGULAR, RhythmClass.AF_COMPATIBLE,
    RhythmClass.FLUTTER_COMPATIBLE, RhythmClass.MULTIPLE_ATRIAL_ACTIVITY, RhythmClass.MIXED])
def test_known_irregular_rhythm_abstains_before_odd_even_calculation(monkeypatch, rhythm_class):
    ctx = SimpleNamespace(delineation=SimpleNamespace(beats=list(range(6)),
        rhythm=RhythmAssessment(rhythm_class=rhythm_class, confidence=1.)))
    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible rhythm should abstain before loop extraction")
    monkeypatch.setattr(module, "_beat_segment_curves", forbidden)
    assert module._literature_feature(ctx, "microvolt_t_wave_alternans_vector").value is None


def test_peak_magnitude_and_vector_waveform_alternation_are_distinct(monkeypatch):
    ctx = SimpleNamespace(fs=500, delineation=SimpleNamespace(beats=[
        SimpleNamespace(r_peak=SimpleNamespace(sample=i * 500)) for i in range(6)]))
    curves = [np.tile([(-1.) ** i, 0., 0.], (64, 1)) for i in range(6)]
    monkeypatch.setattr(module, "_beat_segment_curves", lambda *args, **kwargs: curves)
    assert module._beat_variability_feature(ctx, "t_wave_alternans_vector").value == 0.
    assert module._literature_feature(ctx, "microvolt_t_wave_alternans_vector").value == pytest.approx(2000.)
