from vcg_biomarker.core.delineation.base import BeatDelineation, MarkerPoint
from vcg_biomarker.core.delineation.fusion_delineator import FusionDelineator


def test_candidate_capture_preserves_sources_without_changing_fusion():
    a = BeatDelineation(r_peak=MarkerPoint(500, .9, 'a'), r_onset=MarkerPoint(480,.9,'a'))
    b = BeatDelineation(r_peak=MarkerPoint(501, .9, 'b'), r_onset=MarkerPoint(483,.9,'b'))
    sources = [('neurokit',a,.4), ('multilead',b,.35)]
    ordinary = FusionDelineator()
    recorded = FusionDelineator(capture_candidates=True)
    left = ordinary._fuse_beat(sources,500)
    right = recorded._fuse_beat(sources,500)
    assert left == right
    assert ordinary.candidate_snapshots == {}
    rows = recorded.candidate_snapshots[right.r_peak.sample]
    assert [(r['method'],r['sample']) for r in rows if r['landmark']=='r_onset'] == [('neurokit',480),('multilead',483)]
    assert next(r for r in rows if r['sample']==483)['time_seconds'] == .966
