from types import SimpleNamespace as NS
import numpy as np
import pytest
from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import _early_late_overlap
from vcg_biomarker.core.delineation.base import RhythmClass, WaveStatus


def context(rhythm=RhythmClass.REGULAR):
    signal = np.vstack([np.tile([1., 0., 0.], (20, 1)), np.tile([0., 1., 0.], (20, 1))])
    return NS(fs=500., beat=NS(p_onset=NS(sample=0, confidence=1.),
        p_offset=NS(sample=39, confidence=1.), resolved_wave_status=lambda _: WaveStatus.PRESENT),
        delineation=NS(rhythm=NS(rhythm_class=rhythm)), _primitive_cache={},
        origin_corrected_vcg=lambda: signal)


def test_orthogonal_halves_and_cached_value():
    ctx = context()
    assert _early_late_overlap(ctx, "p") == pytest.approx(0.)
    assert _early_late_overlap(ctx, "p") == pytest.approx(0.)
    assert len(ctx._primitive_cache) == 1


@pytest.mark.parametrize("rhythm", [r for r in RhythmClass if r != RhythmClass.REGULAR])
def test_nonregular_or_unknown_patterns_abstain(rhythm):
    with pytest.raises(ValueError, match="P_requires"):
        _early_late_overlap(context(rhythm), "p")


def test_uncertain_boundary_abstains():
    ctx = context()
    ctx.beat.p_offset.confidence = 0.
    with pytest.raises(ValueError):
        _early_late_overlap(ctx, "p")
