from types import SimpleNamespace as NS
import pytest
from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import _qt_variability_for_cluster


def context(clusters, indices):
    return NS(delineation=NS(beats=[NS(morphology_cluster_id=c, r_peak=NS(sample=i*500)) for i, c in enumerate(clusters)],
        qt_rr_measurements=lambda: [NS(beat_index=i, qt_ms=300.+10*i, confidence=1.) for i in indices]))


def test_adjacent_cluster_differences_match_reference():
    assert _qt_variability_for_cluster(context([0]*4, [1, 2, 3]), 0).value == pytest.approx(10.)


@pytest.mark.parametrize("clusters,indices", [([0, 1, 1, 1], [1, 2, 3]),
    ([0]*5, [0, 2, 4]), ([0, 1, 0, 1, 0], [0, 1, 2, 3, 4]),
    ([0]*4, [1, 1, 2, 3])])
def test_no_pooling_gap_bridging_or_duplicate_identity(clusters, indices):
    assert _qt_variability_for_cluster(context(clusters, indices), 0).value is None


def test_original_index_order_must_match_chronological_order():
    ctx = context([0]*4, [1, 2, 3])
    ctx.delineation.beats[2].r_peak.sample = 100
    assert _qt_variability_for_cluster(ctx, 0).value is None
