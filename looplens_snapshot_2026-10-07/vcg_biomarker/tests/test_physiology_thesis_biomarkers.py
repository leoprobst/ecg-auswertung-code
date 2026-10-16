"""Markers for the theses of 23.09.2026 (T6, T10, T15)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers import physiology_thesis_biomarkers as pt

FS = 500


def _point(sample):
    return SimpleNamespace(sample=sample, confidence=0.9)


def _ctx(clusters, qrs_vectors, t_vectors, *, t_length=150, spacing=1000):
    """Beats every ``spacing`` samples: QRS 50 samples, JT ``t_length`` samples,
    each segment a constant vector, so every area is exact."""
    n = spacing * (len(clusters) + 1)
    record = np.zeros((n, 3))
    beats = []
    for index, (cluster, q, t) in enumerate(zip(clusters, qrs_vectors, t_vectors)):
        start = 100 + index * spacing
        record[start:start + 50] = q
        record[start + 50:start + 50 + t_length] = t
        beats.append(SimpleNamespace(
            morphology_cluster_id=cluster, r_onset=_point(start), r_offset=_point(start + 49),
            t_offset=_point(start + 49 + t_length - 1), p_onset=None,
            r_peak=_point(start + 20),
            resolved_wave_status=lambda wave: SimpleNamespace(value="present")))
    origin = SimpleNamespace(primary=SimpleNamespace(vector_millivolts=np.zeros(3)))
    return SimpleNamespace(_record_vcg_xyz=record, vcg_origin_assessment=origin,
                           delineation=SimpleNamespace(beats=beats), fs=FS)


def test_full_compensation_gives_zero_gradient_change_and_opposed_areas():
    q0, t0 = np.array([1.0, 0.2, 0.0]), np.array([0.3, 0.1, 0.0])
    d = np.array([-0.8, 0.5, 0.3])
    clusters = [0, 0, 0, 1]
    # Trapezoid over one beat: the QRS window holds 50 samples q, the JT
    # window the last QRS sample and 149 samples t. QRST area =
    # (49.5 q + 148.5 t) / fs, QRS area = 49 q / fs. A T change of
    # -49.5/148.5 of the QRS change leaves the whole integral unchanged.
    ctx = _ctx(clusters, [q0, q0, q0, q0 + d], [t0, t0, t0, t0 - d * 49.5 / 148.5])
    assert pt._svg_to_qrs_ratio(ctx).value < 1e-9
    assert abs(pt._t_qrs_cosine(ctx).value + 1.0) < 1e-9


def test_no_compensation_gives_ratio_one():
    q0, t0 = np.array([1.0, 0.2, 0.0]), np.array([0.3, 0.1, 0.0])
    d = np.array([-0.8, 0.5, 0.3])
    ctx = _ctx([0, 0, 0, 1], [q0, q0, q0, q0 + d], [t0] * 4)
    # Unchanged T: the JT area still carries the half trapezoid of the last
    # QRS sample, so R = 49.5 / 49 rather than exactly 1.
    assert abs(pt._svg_to_qrs_ratio(ctx).value - 49.5 / 49) < 1e-9


def test_truncated_t_wave_and_missing_clusters_abstain():
    q0, t0 = np.array([1.0, 0.2, 0.0]), np.array([0.3, 0.1, 0.0])
    # T of 1100 samples runs into the next beat: every beat except the last dropped.
    ctx = _ctx([0, 0, 0, 1], [q0] * 4, [t0] * 4, t_length=1100, spacing=1000)
    assert pt._svg_to_qrs_ratio(ctx).value is None
    only_primary = _ctx([0, 0, 0, 0], [q0] * 4, [t0] * 4)
    assert pt._svg_delta(only_primary).value is None


def test_markers_are_registered_as_post_outcome_discovery_markers():
    from vcg_biomarker.mimic.panels import default_feature_panels

    ids = set(pt.physiology_thesis_marker_ids())
    assert len(ids) == 8 and ids <= set(BiomarkerRegistry.available())
    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert not ids & panels["primary_core"] and not ids & panels["sentinel"]
    assert ids <= panels["discovery_reserve"]


def test_j_tpeak_correction_uses_the_published_exponent():
    beat = SimpleNamespace(j_point=_point(100), r_offset=_point(100), t_peak=_point(250),
                           resolved_wave_status=lambda wave: SimpleNamespace(value="present"))
    beats = [SimpleNamespace(r_peak=_point(s)) for s in (0, 400, 800, 1200)]
    ctx = SimpleNamespace(delineation=SimpleNamespace(representative_beat=beat, beats=beats), fs=FS)
    assert pt._j_tpeak(ctx).value == 300.0
    assert abs(pt._j_tpeak_c(ctx).value - 300.0 / 0.8 ** 0.58) < 1e-9
