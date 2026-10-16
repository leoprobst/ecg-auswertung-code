from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.gui.widgets.temporal_view_data import qrs_view_points


def inputs():
    return SimpleNamespace(xyz=np.ones((20, 3))), SimpleNamespace(
        r_onset=SimpleNamespace(sample=2), r_offset=SimpleNamespace(sample=12),
        segment_confidence=lambda _: 1.0,
    )


def test_view_uses_copy_of_valid_segment():
    vcg, beat = inputs()
    points = qrs_view_points(vcg, beat, 500)
    assert points.shape == (11, 3)
    points[:] = 0
    assert np.all(vcg.xyz == 1)


@pytest.mark.parametrize("start,end", [(-1, 12), (2, 20), (2, 3), (3.5, 12), (np.nan, 12)])
def test_invalid_boundaries_abstain(start, end):
    vcg, beat = inputs()
    beat.r_onset.sample, beat.r_offset.sample = start, end
    with pytest.raises(ValueError):
        qrs_view_points(vcg, beat, 500)


@pytest.mark.parametrize("confidence", [np.nan, np.inf, -1, 2, 0])
def test_invalid_quality_abstains(confidence):
    vcg, beat = inputs()
    beat.segment_confidence = lambda _: confidence
    with pytest.raises(ValueError):
        qrs_view_points(vcg, beat, 500)


def test_time_gap_abstains_instead_of_compressing():
    vcg, beat = inputs()
    vcg.xyz[5, 1] = np.nan
    with pytest.raises(ValueError, match="Zeitlücken"):
        qrs_view_points(vcg, beat, 500)
