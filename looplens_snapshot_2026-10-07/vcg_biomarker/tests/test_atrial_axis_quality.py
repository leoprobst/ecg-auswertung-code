from types import SimpleNamespace as NS
import numpy as np
import pytest
from vcg_biomarker.core.biomarkers import advanced_vcg_biomarkers as advanced
from vcg_biomarker.core.delineation.base import WaveStatus


@pytest.mark.parametrize("calculator,other", [
    (advanced._p_loop_qrs_loop_axis_angle_deg, "qrs"),
    (advanced._p_loop_t_loop_axis_angle_deg, "t"),
])
@pytest.mark.parametrize("bad", [0., np.nan, np.inf, 1.1])
def test_each_segment_must_be_qualified(monkeypatch, calculator, other, bad):
    samples = np.linspace(-1, 1, 10)[:, None]
    signal = np.vstack([samples * [1., 0., 0.], samples * [1., 1., 0.]])
    quality = {"p": 1., other: 1.}
    beat = NS(p_onset=NS(sample=0), p_offset=NS(sample=9),
              r_onset=NS(sample=10), r_offset=NS(sample=19),
              t_onset=NS(sample=10), t_offset=NS(sample=19),
              resolved_wave_status=lambda _: WaveStatus.PRESENT,
              segment_confidence=lambda name: quality[name])
    monkeypatch.setattr(advanced, "_cluster_beats_or_accepted", lambda *args: [beat]*3)
    ctx = NS(_record_vcg_xyz=signal)
    assert calculator(ctx).value == pytest.approx(45.)
    for segment in ("p", other):
        quality[segment] = bad
        assert calculator(ctx).value is None
        quality[segment] = 1.
