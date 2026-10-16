from __future__ import annotations

import os

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import (
    ADVANCED_SPECS,
    AdvancedVCGBiomarker,
)
from vcg_biomarker.core.biomarkers.base import BiomarkerDefinition
from vcg_biomarker.core.biomarkers.catalog_biomarkers import (
    CatalogFeatureBiomarker,
    FeatureContext,
    _angle_between,
    _feature_segment_dependencies,
    _fit_superellipse_exponent,
)
from vcg_biomarker.core.biomarkers.ecg_biomarkers import (
    QTCFridericiaBiomarker,
    QRSDurationBiomarker,
)
from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
    RhythmAssessment,
    RhythmClass,
    WaveStatus,
    measurement_confidence_threshold,
)
from vcg_biomarker.core.delineation.benchmark import (
    AnnotatedWave,
    PredictedWave,
    _cluster_global_references,
    _filter_prediction_to_reference_coverage,
    _parse_ludb_annotation,
)
from vcg_biomarker.core.delineation.fusion_delineator import FusionDelineator
from vcg_biomarker.core.delineation.multilead_delineator import (
    MultileadDelineator,
    _WaveCandidate,
)
from vcg_biomarker.core.preprocessing import (
    construct_robust_morphology_record,
    detect_r_peaks_multilead,
)
from vcg_biomarker.core.vcg_transform import KorsTransform
from vcg_biomarker.mimic.classical import ClassicalECGCriteria
from vcg_biomarker.mimic.panels import default_feature_panels


FS = 500


def _add_piecewise_wave(
    signal: np.ndarray,
    center: int,
    offsets_seconds: tuple[float, ...],
    amplitudes: tuple[float, ...],
    scale: float,
) -> None:
    samples = center + np.rint(np.asarray(offsets_seconds) * FS).astype(int)
    start = max(0, int(samples[0]))
    end = min(len(signal), int(samples[-1]) + 1)
    if end <= start:
        return
    support = np.arange(start, end)
    signal[start:end] += scale * np.interp(support, samples, amplitudes)


def _synthetic_ecg(
    *,
    rr_seconds: tuple[float, ...] = (0.82,) * 10,
    p_wave: bool = True,
    qrs_width: str = "normal",
    invert_t: bool = False,
    noise_sd: float = 0.004,
    baseline_wander: float = 0.0,
    atrial_frequency_hz: float | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    r_times = [0.75]
    for interval in rr_seconds:
        r_times.append(r_times[-1] + interval)
    duration = r_times[-1] + 0.8
    n_samples = int(np.ceil(duration * FS))
    r_peaks = np.rint(np.asarray(r_times) * FS).astype(int)
    ecg = np.zeros((n_samples, 12), dtype=float)
    gains = np.asarray(
        [0.75, 1.00, 0.65, -0.70, 0.55, 0.80, -0.45, 0.65, 0.85, 1.05, 1.10, 0.95]
    )
    p_gains = np.asarray(
        [0.70, 1.00, 0.70, -0.55, 0.55, 0.85, -0.35, 0.50, 0.55, 0.60, 0.55, 0.50]
    )
    t_gains = gains.copy()
    if invert_t:
        t_gains = -np.abs(t_gains)

    if qrs_width == "wide":
        qrs_offsets = (-0.105, -0.070, 0.0, 0.075, 0.135)
    else:
        qrs_offsets = (-0.050, -0.027, 0.0, 0.028, 0.055)
    for r_peak in r_peaks:
        for lead in range(12):
            if p_wave:
                _add_piecewise_wave(
                    ecg[:, lead],
                    int(r_peak),
                    (-0.225, -0.165, -0.105),
                    (0.0, 0.13, 0.0),
                    float(p_gains[lead]),
                )
            _add_piecewise_wave(
                ecg[:, lead],
                int(r_peak),
                qrs_offsets,
                (0.0, -0.18, 1.0, -0.30, 0.0),
                float(gains[lead]),
            )
            _add_piecewise_wave(
                ecg[:, lead],
                int(r_peak),
                (0.135, 0.285, 0.485),
                (0.0, 0.32, 0.0),
                float(t_gains[lead]),
            )

    time = np.arange(n_samples) / FS
    if baseline_wander:
        ecg += baseline_wander * np.sin(2.0 * np.pi * 0.23 * time)[:, None]
    if atrial_frequency_hz is not None:
        atrial = 0.075 * np.sin(2.0 * np.pi * atrial_frequency_hz * time)
        ecg[:, 1] += atrial
        ecg[:, 6] += 0.8 * atrial
        ecg[:, 7] += 0.5 * atrial
    rng = np.random.default_rng(seed)
    ecg += rng.normal(0.0, noise_sd, size=ecg.shape)
    return ecg, r_peaks


def _marker(
    sample: int,
    confidence: float = 0.9,
    source: str = "test",
) -> MarkerPoint:
    return MarkerPoint(sample=sample, confidence=confidence, source=source)


def _interval_record(
    rr_seconds: tuple[float, ...],
    qt_ms: tuple[float, ...],
    *,
    rhythm: RhythmClass,
) -> RecordDelineation:
    beats: list[BeatDelineation] = []
    r_sample = 300
    for index, qt in enumerate(qt_ms):
        if index:
            r_sample += int(round(rr_seconds[index - 1] * FS))
        onset = r_sample - 20
        offset = r_sample + 30
        t_offset = onset + int(round(qt / 1000.0 * FS))
        beats.append(
            BeatDelineation(
                r_peak=_marker(r_sample),
                r_onset=_marker(onset),
                r_offset=_marker(offset),
                j_point=_marker(offset),
                t_onset=_marker(r_sample + 75),
                t_peak=_marker(r_sample + 130),
                t_offset=_marker(t_offset),
                qrs_status=WaveStatus.PRESENT,
                t_status=WaveStatus.PRESENT,
            )
        )
    return RecordDelineation(
        beats=beats,
        fs=FS,
        n_samples=beats[-1].t_offset.sample + 100,
        ecg_id=7,
        quality_score=0.9,
        rhythm=RhythmAssessment(rhythm_class=rhythm, confidence=0.9),
    )


def _complete_test_beat(
    r_sample: int,
    *,
    morphology_outlier: bool = False,
    coupling_ratio: float | None = None,
) -> BeatDelineation:
    return BeatDelineation(
        r_peak=_marker(r_sample),
        p_onset=_marker(r_sample - 110),
        p_peak=_marker(r_sample - 82),
        p_offset=_marker(r_sample - 55),
        r_onset=_marker(r_sample - 22),
        r_offset=_marker(r_sample + 32),
        j_point=_marker(r_sample + 32),
        t_onset=_marker(r_sample + 70),
        t_peak=_marker(r_sample + 140),
        t_offset=_marker(r_sample + 225),
        p_status=WaveStatus.PRESENT,
        qrs_status=WaveStatus.PRESENT,
        t_status=WaveStatus.PRESENT,
        morphology_outlier=morphology_outlier,
        coupling_ratio=coupling_ratio,
    )


def test_morphology_consensus_marks_only_non_dominant_beats() -> None:
    rng = np.random.default_rng(407)
    r_samples = tuple(range(500, 5_500, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = np.zeros((6_000, 12), dtype=float)
    offsets = np.arange(-60, 90)
    template = np.exp(-0.5 * (offsets / 14.0) ** 2)
    for index, r_sample in enumerate(r_samples):
        if index < 8:
            signal[r_sample + offsets] = template[:, None] * np.linspace(
                0.8,
                1.2,
                12,
            )[None, :]
        else:
            signal[r_sample + offsets] = rng.normal(0.0, 1.0, (len(offsets), 12))

    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)

    assert diagnostics["morphology_consensus_assessable"] is True
    assert diagnostics["severe_signal_incoherence"] is False
    assert diagnostics["morphology_dominant_fraction"] == pytest.approx(0.8)
    assert [beat.morphology_outlier for beat in beats] == [False] * 8 + [True] * 2


def test_incoherent_beat_morphology_forces_segment_abstention() -> None:
    rng = np.random.default_rng(408)
    r_samples = tuple(range(500, 5_500, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = rng.normal(0.0, 1.0, (6_000, 12))

    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)
    record = RecordDelineation(
        beats=beats,
        fs=FS,
        n_samples=len(signal),
        ecg_id=410,
        diagnostics=diagnostics,
    )

    assert diagnostics["severe_signal_incoherence"] is True
    assert record.segment_confidence("qrs") >= 0.35
    assert record.is_segment_evaluable("qrs") is False
    assert record.is_segment_evaluable("st") is False
    assert record.is_segment_evaluable("t") is False


def test_robust_morphology_template_excludes_corrupted_medoid_beat() -> None:
    r_samples = (350, 850, 1350, 1850)
    beats = [
        _complete_test_beat(
            r_sample,
            morphology_outlier=index == 0,
        )
        for index, r_sample in enumerate(r_samples)
    ]
    record = RecordDelineation(
        beats=beats,
        fs=FS,
        n_samples=2300,
        ecg_id=99,
        quality_score=0.9,
    )
    signal = np.zeros((2300, 12), dtype=float)
    amplitudes = (10.0, 0.95, 1.00, 1.05)
    offsets = np.arange(-22, 33)
    qrs_shape = np.exp(-0.5 * (offsets / 7.0) ** 2)
    for r_sample, amplitude in zip(r_samples, amplitudes):
        signal[r_sample + offsets] = amplitude * qrs_shape[:, None]

    robust, diagnostics = construct_robust_morphology_record(
        signal,
        record,
    )

    representative_r = record.representative_beat.r_peak.sample
    assert representative_r == r_samples[0]
    assert signal[representative_r, 0] == 10.0
    np.testing.assert_allclose(
        robust[representative_r, 0],
        1.0,
        atol=0.06,
    )
    assert diagnostics["status"] == "used"
    assert diagnostics["candidate_beat_count"] == 4
    assert diagnostics["used_beat_count"] == 3
    assert diagnostics["excluded_morphology_outlier_count"] == 1
    assert signal[representative_r, 0] == 10.0


def _paint_qrs_shape(
    signal: np.ndarray,
    r_samples: tuple[int, ...],
    shape: np.ndarray,
    offsets: np.ndarray,
    gains: np.ndarray,
) -> None:
    for r_sample in r_samples:
        signal[r_sample + offsets] = shape[:, None] * gains[None, :]


_CLUSTER_OFFSETS = np.arange(-60, 90)
_CLUSTER_GAINS = np.linspace(0.8, 1.2, 12)
_CLUSTER_SIGMA = 14.0
_CLUSTER_SHAPE_SINUS = np.exp(-0.5 * (_CLUSTER_OFFSETS / _CLUSTER_SIGMA) ** 2)
_CLUSTER_SHAPE_ECTOPIC_A = -1.0 * np.exp(
    -0.5 * ((_CLUSTER_OFFSETS - 25) / 22.0) ** 2
)
# First and second Gaussian-derivative shapes: near-orthogonal to the sinus
# template and to each other, so a three-way medoid search is not swayed by
# incidental cross-correlation between the two non-dominant shapes.
_CLUSTER_SHAPE_ECTOPIC_HERMITE_1 = -(
    _CLUSTER_OFFSETS / _CLUSTER_SIGMA**2
) * _CLUSTER_SHAPE_SINUS
_CLUSTER_SHAPE_ECTOPIC_HERMITE_1 /= np.max(np.abs(_CLUSTER_SHAPE_ECTOPIC_HERMITE_1))
_CLUSTER_SHAPE_ECTOPIC_HERMITE_2 = (
    (_CLUSTER_OFFSETS**2 - _CLUSTER_SIGMA**2) / _CLUSTER_SIGMA**4
) * _CLUSTER_SHAPE_SINUS
_CLUSTER_SHAPE_ECTOPIC_HERMITE_2 /= np.max(np.abs(_CLUSTER_SHAPE_ECTOPIC_HERMITE_2))


def test_bigeminy_pattern_upgrades_from_severe_incoherence_to_dual_template() -> None:
    r_samples = tuple(range(500, 5_500, 500))
    sinus_samples = r_samples[0::2]
    ectopic_samples = r_samples[1::2]
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = np.zeros((6_000, 12), dtype=float)
    _paint_qrs_shape(
        signal, sinus_samples, _CLUSTER_SHAPE_SINUS, _CLUSTER_OFFSETS, _CLUSTER_GAINS
    )
    _paint_qrs_shape(
        signal,
        ectopic_samples,
        _CLUSTER_SHAPE_ECTOPIC_A,
        _CLUSTER_OFFSETS,
        _CLUSTER_GAINS,
    )

    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)

    assert diagnostics["morphology_consensus_assessable"] is True
    assert diagnostics["morphology_cluster_count"] == 2
    assert diagnostics["morphology_multi_morphology_compatible"] is True
    assert diagnostics["severe_signal_incoherence"] is False
    assert diagnostics["morphology_secondary_beat_count"] == 5
    assert diagnostics["morphology_secondary_fraction"] == pytest.approx(0.5)
    assert diagnostics["morphology_secondary_centroid_distance_mv"] > 0.0

    # A perfectly balanced 5/5 alternation has no ground-truth "dominant"
    # shape, so either physical group may end up labelled cluster 0 versus
    # 1 -- what matters is that the two clusters exactly partition the two
    # true shape groups rather than mixing beats from both.
    sinus_index = {r_samples.index(sample) for sample in sinus_samples}
    ectopic_index = {r_samples.index(sample) for sample in ectopic_samples}
    cluster0 = {
        index for index, beat in enumerate(beats) if beat.morphology_cluster_id == 0
    }
    cluster1 = {
        index for index, beat in enumerate(beats) if beat.morphology_cluster_id == 1
    }
    assert {frozenset(cluster0), frozenset(cluster1)} == {
        frozenset(sinus_index),
        frozenset(ectopic_index),
    }
    assert all(
        not beat.morphology_outlier
        for beat in beats
        if beat.morphology_cluster_id == 0
    )
    assert all(
        beat.morphology_outlier
        for beat in beats
        if beat.morphology_cluster_id == 1
    )


def test_trigeminy_pattern_characterizes_secondary_cluster_without_severity() -> None:
    r_samples = tuple(range(500, 6_500, 500))
    ectopic_samples = r_samples[2::3]
    sinus_samples = tuple(s for s in r_samples if s not in ectopic_samples)
    assert len(ectopic_samples) == 4
    assert len(sinus_samples) == 8
    beats = [
        _complete_test_beat(
            sample,
            coupling_ratio=0.62 if sample in ectopic_samples else 1.0,
        )
        for sample in r_samples
    ]
    signal = np.zeros((7_000, 12), dtype=float)
    _paint_qrs_shape(
        signal, sinus_samples, _CLUSTER_SHAPE_SINUS, _CLUSTER_OFFSETS, _CLUSTER_GAINS
    )
    _paint_qrs_shape(
        signal,
        ectopic_samples,
        _CLUSTER_SHAPE_ECTOPIC_A,
        _CLUSTER_OFFSETS,
        _CLUSTER_GAINS,
    )

    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)

    assert diagnostics["severe_signal_incoherence"] is False
    assert diagnostics["morphology_dominant_beat_count"] == 8
    assert diagnostics["morphology_dominant_fraction"] == pytest.approx(2 / 3)
    assert diagnostics["morphology_cluster_count"] == 2
    assert diagnostics["morphology_secondary_beat_count"] == 4
    assert diagnostics["morphology_secondary_mean_coupling_ratio"] == pytest.approx(
        0.62
    )


def test_two_multifocal_shapes_form_secondary_and_tertiary_clusters() -> None:
    r_samples = tuple(range(500, 7_000, 500))
    assert len(r_samples) == 13
    sinus_samples = r_samples[:7]
    ectopic_a_samples = r_samples[7:10]
    ectopic_b_samples = r_samples[10:13]
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = np.zeros((7_500, 12), dtype=float)
    _paint_qrs_shape(
        signal, sinus_samples, _CLUSTER_SHAPE_SINUS, _CLUSTER_OFFSETS, _CLUSTER_GAINS
    )
    _paint_qrs_shape(
        signal,
        ectopic_a_samples,
        _CLUSTER_SHAPE_ECTOPIC_HERMITE_1,
        _CLUSTER_OFFSETS,
        _CLUSTER_GAINS,
    )
    _paint_qrs_shape(
        signal,
        ectopic_b_samples,
        _CLUSTER_SHAPE_ECTOPIC_HERMITE_2,
        _CLUSTER_OFFSETS,
        _CLUSTER_GAINS,
    )

    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)

    assert diagnostics["morphology_cluster_count"] == 3
    assert diagnostics["morphology_dominant_beat_count"] == 7
    cluster_ids = [beat.morphology_cluster_id for beat in beats]
    assert cluster_ids[:7] == [0] * 7
    assert set(cluster_ids[7:10]) in ({1}, {2})
    assert set(cluster_ids[10:13]) in ({1}, {2})
    assert set(cluster_ids[7:10]) != set(cluster_ids[10:13])


def test_secondary_cluster_yields_own_robust_morphology_template() -> None:
    r_samples = tuple(range(500, 5_500, 500))
    sinus_samples = r_samples[0::2]
    ectopic_samples = r_samples[1::2]
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = np.zeros((6_000, 12), dtype=float)
    _paint_qrs_shape(
        signal, sinus_samples, _CLUSTER_SHAPE_SINUS, _CLUSTER_OFFSETS, _CLUSTER_GAINS
    )
    _paint_qrs_shape(
        signal,
        ectopic_samples,
        _CLUSTER_SHAPE_ECTOPIC_A,
        _CLUSTER_OFFSETS,
        _CLUSTER_GAINS,
    )
    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)
    assert diagnostics["morphology_cluster_count"] == 2
    record = RecordDelineation(beats=beats, fs=FS, n_samples=len(signal), ecg_id=97)

    primary, primary_diag = construct_robust_morphology_record(
        signal, record, morphology_cluster_id=0
    )
    secondary, secondary_diag = construct_robust_morphology_record(
        signal, record, morphology_cluster_id=1
    )

    assert primary_diag["status"] == "used"
    assert primary_diag["morphology_cluster_id"] == 0
    assert secondary_diag["status"] == "used"
    assert secondary_diag["morphology_cluster_id"] == 1
    assert secondary_diag["candidate_beat_count"] == 5
    sinus_peak = int(np.argmax(_CLUSTER_SHAPE_SINUS))
    ectopic_trough = int(np.argmin(_CLUSTER_SHAPE_ECTOPIC_A))
    assert primary[sinus_samples[0] + _CLUSTER_OFFSETS[sinus_peak], 1] > 0.5
    assert (
        secondary[ectopic_samples[0] + _CLUSTER_OFFSETS[ectopic_trough], 1] < -0.5
    )


def _advanced_marker_value(
    name: str,
    record: RecordDelineation,
    signal: np.ndarray,
    vcg_xyz: np.ndarray,
) -> float | None:
    spec = next(item for item in ADVANCED_SPECS if item.name == name)
    result = AdvancedVCGBiomarker(spec).compute(
        record, ecg_12lead=signal, vcg_xyz=vcg_xyz, fs=FS
    )
    return result.value


def test_f1_secondary_cluster_markers_on_well_separated_trigeminy() -> None:
    baseline_rr = 400
    short_coupling = 250
    compensatory_pause = 800 - short_coupling
    sample = 500
    r_samples: list[int] = [sample]
    ectopic_indices: set[int] = set()
    for block in range(3):
        for _ in range(4):
            sample += baseline_rr
            r_samples.append(sample)
        sample += short_coupling
        r_samples.append(sample)
        ectopic_indices.add(len(r_samples) - 1)
        sample += compensatory_pause
        r_samples.append(sample)
    for _ in range(4):
        sample += baseline_rr
        r_samples.append(sample)

    beats = [_complete_test_beat(value) for value in r_samples]
    signal = np.zeros((sample + 500, 12), dtype=float)
    sinus_samples = tuple(
        value for index, value in enumerate(r_samples) if index not in ectopic_indices
    )
    ectopic_samples = tuple(r_samples[index] for index in sorted(ectopic_indices))
    _paint_qrs_shape(
        signal, sinus_samples, _CLUSTER_SHAPE_SINUS, _CLUSTER_OFFSETS, _CLUSTER_GAINS
    )
    _paint_qrs_shape(
        signal,
        ectopic_samples,
        _CLUSTER_SHAPE_ECTOPIC_A,
        _CLUSTER_OFFSETS,
        _CLUSTER_GAINS,
    )

    FusionDelineator._assign_beat_coupling_context(beats, FS)
    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)
    assert diagnostics["morphology_cluster_count"] == 2
    assert diagnostics["morphology_secondary_beat_count"] == 3

    for index in ectopic_indices:
        assert beats[index].coupling_ratio == pytest.approx(0.625, abs=0.02)

    vcg_xyz = KorsTransform().transform(signal).xyz
    record = RecordDelineation(beats=beats, fs=FS, n_samples=len(signal), ecg_id=201)

    pause_ratio = _advanced_marker_value(
        "secondary_morphology_compensatory_pause_ratio", record, signal, vcg_xyz
    )
    assert pause_ratio == pytest.approx(1.0, abs=0.05)

    axis = _advanced_marker_value(
        "secondary_morphology_frontal_axis_deg", record, signal, vcg_xyz
    )
    assert axis is not None

    for name in (
        "secondary_morphology_t_amplitude_beat_to_beat_variability",
        "secondary_morphology_st_level_beat_to_beat_variability",
        "secondary_morphology_qt_delta_ms",
        "secondary_morphology_qrs_duration_delta_ms",
        "secondary_morphology_qrs_area_vector_magnitude_delta",
    ):
        value = _advanced_marker_value(name, record, signal, vcg_xyz)
        assert value is not None and np.isfinite(value), name

    # Isolated ectopic beats have no adjacent same-cluster QT pairs.
    # Bridging intervening sinus beats would violate this marker's definition.
    assert _advanced_marker_value(
        "secondary_morphology_qt_beat_to_beat_variability_ms", record, signal, vcg_xyz
    ) is None


def test_f1_secondary_cluster_markers_abstain_without_secondary_cluster() -> None:
    r_samples = tuple(range(500, 5_500, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = np.zeros((6_000, 12), dtype=float)
    _paint_qrs_shape(
        signal, r_samples, _CLUSTER_SHAPE_SINUS, _CLUSTER_OFFSETS, _CLUSTER_GAINS
    )
    FusionDelineator._assign_beat_coupling_context(beats, FS)
    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)
    assert "morphology_secondary_beat_count" not in diagnostics

    vcg_xyz = KorsTransform().transform(signal).xyz
    record = RecordDelineation(beats=beats, fs=FS, n_samples=len(signal), ecg_id=202)

    for name in (
        "secondary_morphology_compensatory_pause_ratio",
        "secondary_morphology_frontal_axis_deg",
        "secondary_morphology_qt_delta_ms",
        "post_ectopic_t_amplitude_delta_mv",
        "post_ectopic_t_axis_delta_deg",
    ):
        assert _advanced_marker_value(name, record, signal, vcg_xyz) is None


def test_f6_f7_post_ectopic_and_parasystole_markers() -> None:
    baseline_rr = 400
    short_coupling = 250
    compensatory_pause = 800 - short_coupling
    sample = 500
    r_samples: list[int] = [sample]
    ectopic_indices: set[int] = set()
    for block in range(3):
        for _ in range(4):
            sample += baseline_rr
            r_samples.append(sample)
        sample += short_coupling
        r_samples.append(sample)
        ectopic_indices.add(len(r_samples) - 1)
        sample += compensatory_pause
        r_samples.append(sample)
    for _ in range(4):
        sample += baseline_rr
        r_samples.append(sample)

    beats = [_complete_test_beat(value) for value in r_samples]
    signal = np.zeros((sample + 500, 12), dtype=float)
    sinus_samples = tuple(
        value for index, value in enumerate(r_samples) if index not in ectopic_indices
    )
    ectopic_samples = tuple(r_samples[index] for index in sorted(ectopic_indices))
    _paint_qrs_shape(
        signal, sinus_samples, _CLUSTER_SHAPE_SINUS, _CLUSTER_OFFSETS, _CLUSTER_GAINS
    )
    _paint_qrs_shape(
        signal,
        ectopic_samples,
        _CLUSTER_SHAPE_ECTOPIC_A,
        _CLUSTER_OFFSETS,
        _CLUSTER_GAINS,
    )
    t_offsets = np.arange(70, 226)
    t_shape = 0.3 * np.exp(-0.5 * ((t_offsets - 140) / 25.0) ** 2)
    for beat in beats:
        signal[beat.r_peak.sample + t_offsets] = (
            t_shape[:, None] * _CLUSTER_GAINS[None, :]
        )

    FusionDelineator._assign_beat_coupling_context(beats, FS)
    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)
    assert diagnostics["morphology_secondary_beat_count"] == 3
    post_indices = {index + 1 for index in ectopic_indices}
    assert all(beats[index].morphology_cluster_id == 0 for index in post_indices)

    vcg_xyz = KorsTransform().transform(signal).xyz
    record = RecordDelineation(beats=beats, fs=FS, n_samples=len(signal), ecg_id=203)

    for name in (
        "post_ectopic_t_amplitude_delta_mv",
        "post_ectopic_t_axis_delta_deg",
    ):
        value = _advanced_marker_value(name, record, signal, vcg_xyz)
        assert value is not None and np.isfinite(value), name

    coupling_cv = _advanced_marker_value(
        "secondary_morphology_coupling_variability_cv", record, signal, vcg_xyz
    )
    assert coupling_cv == pytest.approx(0.0, abs=0.05)

    denominator_fit = _advanced_marker_value(
        "secondary_morphology_common_denominator_fit", record, signal, vcg_xyz
    )
    assert denominator_fit == pytest.approx(1.0, abs=0.02)

    primary_mdi = _advanced_marker_value(
        "primary_morphology_maximum_deflection_index", record, signal, vcg_xyz
    )
    secondary_mdi = _advanced_marker_value(
        "secondary_morphology_maximum_deflection_index", record, signal, vcg_xyz
    )
    mdi_delta = _advanced_marker_value(
        "secondary_morphology_maximum_deflection_index_delta", record, signal, vcg_xyz
    )
    assert primary_mdi is not None and 0.0 <= primary_mdi <= 1.0
    assert secondary_mdi is not None and 0.0 <= secondary_mdi <= 1.0
    assert mdi_delta == pytest.approx(secondary_mdi - primary_mdi)

    for name in (
        "sum_absolute_qrst_integral_mv_ms",
        "t_loop_biphasic_index",
        "primary_morphology_qrs_octant_id",
        "secondary_morphology_qrs_octant_id",
        "qrs_axis_kors_vs_dower_delta_deg",
        "t_axis_kors_vs_dower_delta_deg",
        "rr_poincare_sd1_ms",
        "rr_poincare_sd2_ms",
        "qrs_magnitude_poincare_sd1_mv",
        "qrs_magnitude_poincare_sd2_mv",
        "t_magnitude_poincare_sd1_mv",
        "t_magnitude_poincare_sd2_mv",
        "within_record_qrs_axis_drift_deg",
        "within_record_qrs_amplitude_drift_mv",
        "within_record_qtc_drift_ms",
    ):
        value = _advanced_marker_value(name, record, signal, vcg_xyz)
        assert value is not None and np.isfinite(value), name

    octant = _advanced_marker_value(
        "primary_morphology_qrs_octant_id", record, signal, vcg_xyz
    )
    assert octant in {float(v) for v in range(8)}

    for name in (
        "qrs_multiscale_entropy",
        "t_multiscale_entropy",
        "qrs_higuchi_fractal_dimension",
        "t_higuchi_fractal_dimension",
        "qrs_xy_cross_correlation_peak_lag_ms",
        "qrs_xy_cross_correlation_peak_value",
        "qrs_time_reversal_asymmetry",
        "t_time_reversal_asymmetry",
    ):
        value = _advanced_marker_value(name, record, signal, vcg_xyz)
        assert value is not None and np.isfinite(value), name


def test_f12_f15_tier2_markers_abstain_on_degenerate_flat_signal() -> None:
    r_samples = tuple(range(500, 5_500, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = np.zeros((6_000, 12), dtype=float)
    for beat in beats:
        beat.morphology_cluster_id = 0

    vcg_xyz = KorsTransform().transform(signal).xyz
    record = RecordDelineation(beats=beats, fs=FS, n_samples=len(signal), ecg_id=213)

    for name in (
        "qrs_multiscale_entropy",
        "t_multiscale_entropy",
        "qrs_time_reversal_asymmetry",
        "t_time_reversal_asymmetry",
    ):
        assert _advanced_marker_value(name, record, signal, vcg_xyz) is None

    for name in ("qrs_higuchi_fractal_dimension", "t_higuchi_fractal_dimension"):
        value = _advanced_marker_value(name, record, signal, vcg_xyz)
        assert value is None or np.isfinite(value)


_QRS_NOTCH_OFFSETS = np.arange(-22, 33)
_QRS_SHAPE_CLEAN = np.exp(-0.5 * (_QRS_NOTCH_OFFSETS / 8.0) ** 2)
_QRS_SHAPE_FRAGMENTED = np.exp(
    -0.5 * ((_QRS_NOTCH_OFFSETS + 8) / 4.0) ** 2
) + 0.6 * np.exp(-0.5 * ((_QRS_NOTCH_OFFSETS - 8) / 4.0) ** 2)


def _paint_lead_shape(
    signal: np.ndarray,
    r_samples: tuple[int, ...],
    lead_index: int,
    shape: np.ndarray,
    offsets: np.ndarray,
) -> None:
    for r_sample in r_samples:
        signal[r_sample + offsets, lead_index] = shape


def test_f2_fragmented_qrs_present_in_contiguous_leads() -> None:
    from vcg_biomarker.core.data_loader import LEAD_MAP

    r_samples = tuple(range(500, 4_000, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = np.zeros((4_500, 12), dtype=float)
    qrs_offsets = np.arange(-22, 33)
    for lead in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V3", "V4", "V5", "V6"):
        _paint_lead_shape(
            signal, r_samples, LEAD_MAP[lead], _QRS_SHAPE_CLEAN, qrs_offsets
        )
    for lead in ("II", "III"):
        _paint_lead_shape(
            signal, r_samples, LEAD_MAP[lead], _QRS_SHAPE_FRAGMENTED, np.arange(-22, 33)
        )
    for beat in beats:
        beat.morphology_cluster_id = 0

    vcg_xyz = KorsTransform().transform(signal).xyz
    record = RecordDelineation(beats=beats, fs=FS, n_samples=len(signal), ecg_id=210)

    present = _advanced_marker_value(
        "fragmented_qrs_present_flag", record, signal, vcg_xyz
    )
    lead_count = _advanced_marker_value(
        "fragmented_qrs_lead_count", record, signal, vcg_xyz
    )
    assert present == 1.0
    assert lead_count == pytest.approx(2.0)


def test_f2_fragmented_qrs_absent_when_all_leads_clean() -> None:
    r_samples = tuple(range(500, 4_000, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = np.zeros((4_500, 12), dtype=float)
    for lead_index in range(12):
        _paint_lead_shape(
            signal, r_samples, lead_index, _QRS_SHAPE_CLEAN, np.arange(-22, 33)
        )
    for beat in beats:
        beat.morphology_cluster_id = 0

    vcg_xyz = KorsTransform().transform(signal).xyz
    record = RecordDelineation(beats=beats, fs=FS, n_samples=len(signal), ecg_id=211)

    present = _advanced_marker_value(
        "fragmented_qrs_present_flag", record, signal, vcg_xyz
    )
    assert present == 0.0


def test_f2_fragmented_qrs_excluded_for_wide_qrs() -> None:
    r_samples = tuple(range(500, 4_000, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    for beat in beats:
        beat.morphology_cluster_id = 0
        beat.wide_qrs = True
        beat.bundle_branch_block = True
    signal = np.zeros((4_500, 12), dtype=float)
    for lead_index in range(12):
        _paint_lead_shape(
            signal, r_samples, lead_index, _QRS_SHAPE_FRAGMENTED, np.arange(-22, 33)
        )

    vcg_xyz = KorsTransform().transform(signal).xyz
    record = RecordDelineation(beats=beats, fs=FS, n_samples=len(signal), ecg_id=212)

    assert record.representative_beat.bundle_branch_block is True
    present = _advanced_marker_value(
        "fragmented_qrs_present_flag", record, signal, vcg_xyz
    )
    assert present is None


def test_pr_segment_and_pr_interval_use_distinct_boundaries() -> None:
    beat = _complete_test_beat(500)
    record = RecordDelineation(
        beats=[beat],
        fs=FS,
        n_samples=1000,
        ecg_id=100,
        quality_score=0.9,
    )
    signal = np.zeros((1000, 12), dtype=float)
    context = FeatureContext(record, signal, signal[:, :3], FS)

    assert context.bounds("pr") == (
        beat.p_offset.sample,
        beat.r_onset.sample,
    )
    assert beat.pr_interval_samples == (
        beat.r_onset.sample - beat.p_onset.sample
    )
    assert beat.segment_confidence("pr") == min(
        beat.p_offset.confidence,
        beat.r_onset.confidence,
    )


def test_angle_and_terminal_t_features_abstain_at_record_boundary() -> None:
    assert _angle_between(None, np.ones(3)) is None
    assert _angle_between(np.ones(3), np.array([1.0, np.nan, 0.0])) is None

    beat = _complete_test_beat(900)
    record = RecordDelineation(
        beats=[beat],
        fs=FS,
        n_samples=1000,
        ecg_id=102,
        quality_score=0.9,
    )
    ecg = np.zeros((1000, 12), dtype=float)
    vcg = np.zeros((1000, 3), dtype=float)
    context = FeatureContext(record, ecg, vcg, FS)

    assert context.marker_index("t_peak") is None
    assert context.marker_index("t_end") is None
    results = BiomarkerRegistry.compute_all(
        record,
        ecg,
        vcg,
        FS,
        names=[
            "st_vector_to_tpeak_angle_deg",
            "tpeak_to_tend_vector",
            "tpeak_tend_vector_dispersion",
        ],
    )
    assert all(
        not result.computation_notes.startswith("Error:")
        for result in results
    )
    assert all(result.value is None for result in results)


@pytest.mark.parametrize("expected_exponent", [1.0, 2.0, 4.0, 6.0])
def test_superellipse_exponent_recovers_known_loop_shape(
    expected_exponent: float,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    points = np.column_stack(
        [
            2.0
            * np.sign(cosine)
            * np.abs(cosine) ** (2.0 / expected_exponent),
            0.8
            * np.sign(sine)
            * np.abs(sine) ** (2.0 / expected_exponent),
            np.zeros_like(theta),
        ]
    )

    fitted, fit_error = _fit_superellipse_exponent(points)

    assert fitted == pytest.approx(expected_exponent, abs=0.02)
    assert fit_error < 1e-4


def test_beat_variability_uses_untouched_record_signal() -> None:
    beats = [
        _complete_test_beat(r_sample)
        for r_sample in (350, 850, 1350, 1850)
    ]
    record = RecordDelineation(
        beats=beats,
        fs=FS,
        n_samples=2300,
        ecg_id=101,
        quality_score=0.9,
    )
    record_vcg = np.zeros((2300, 3), dtype=float)
    for beat in beats:
        record_vcg[
            beat.r_onset.sample : beat.r_offset.sample + 1,
            0,
        ] = 1.0
    representative_vcg = record_vcg.copy()
    representative = record.representative_beat
    representative_vcg[
        representative.r_onset.sample : representative.r_offset.sample + 1
    ] = np.asarray([0.0, 1.0, 0.0])
    ecg = np.zeros((2300, 12), dtype=float)

    with_record_source = BiomarkerRegistry.compute_all(
        record,
        ecg,
        representative_vcg,
        FS,
        names=["qrs_beat_area_axis_coherence"],
        record_ecg_12lead=ecg,
        record_vcg_xyz=record_vcg,
    )[0]
    without_record_source = BiomarkerRegistry.compute_all(
        record,
        ecg,
        representative_vcg,
        FS,
        names=["qrs_beat_area_axis_coherence"],
    )[0]

    assert with_record_source.value == pytest.approx(1.0)
    assert without_record_source.value < 0.9


def test_clean_multilead_detection_recovers_r_peaks_and_all_segments() -> None:
    ecg, expected_r = _synthetic_ecg()
    detected_r = detect_r_peaks_multilead(ecg, FS)
    assert len(detected_r) == len(expected_r)
    assert np.max(np.abs(detected_r - expected_r)) <= 3

    vcg = KorsTransform().transform(ecg).xyz
    record = FusionDelineator().delineate_record(
        ecg,
        expected_r,
        FS,
        ecg_id=1,
        vcg_xyz=vcg,
    )
    representative = record.representative_beat
    assert representative is not None
    assert representative.segment_confidence("p") >= 0.35
    assert representative.segment_confidence("qrs") >= 0.35
    assert representative.segment_confidence("t") >= 0.35
    assert 70.0 <= representative.qrs_duration_ms(FS) <= 130.0
    assert record.rhythm.rhythm_class == RhythmClass.REGULAR
    assert record.rhythm.p_candidate_coverage is not None
    assert record.rhythm.p_candidate_coverage >= 0.60


def test_multilead_detector_is_polarity_invariant_for_inverted_t_waves() -> None:
    ecg, r_peaks = _synthetic_ecg(invert_t=True)
    beats = MultileadDelineator().delineate(ecg, r_peaks, FS)
    evaluable = [beat for beat in beats if beat.segment_confidence("t") >= 0.35]
    assert len(evaluable) >= int(0.8 * len(beats))
    assert all(beat.resolved_wave_status("t") == WaveStatus.PRESENT for beat in evaluable)


def test_wide_qrs_is_measured_instead_of_clipped_to_normal_window() -> None:
    ecg, r_peaks = _synthetic_ecg(qrs_width="wide")
    beats = MultileadDelineator().delineate(ecg, r_peaks, FS)
    durations = [
        beat.qrs_duration_ms(FS)
        for beat in beats
        if beat.segment_confidence("qrs") >= 0.35
    ]
    assert len(durations) >= int(0.8 * len(beats))
    assert float(np.median(durations)) > 160.0
    assert sum(beat.wide_qrs for beat in beats) >= int(0.8 * len(beats))


def test_qrs_threshold_requires_uncertainty_separated_crossing() -> None:
    uncertain = BeatDelineation(
        r_peak=_marker(300),
        r_onset=MarkerPoint(
            sample=270,
            confidence=0.9,
            source="test",
            uncertainty_ms=7.0,
        ),
        r_offset=MarkerPoint(
            sample=331,
            confidence=0.9,
            source="test",
            uncertainty_ms=6.0,
        ),
        qrs_status=WaveStatus.PRESENT,
    )
    assert uncertain.qrs_duration_ms(FS) == 122.0
    assert uncertain.qrs_duration_bounds_ms(FS) == (109.0, 135.0)
    assert not uncertain.qrs_definitely_exceeds_ms(FS, 120.0)

    clearly_wide = BeatDelineation(
        r_peak=_marker(300),
        r_onset=MarkerPoint(
            sample=255,
            confidence=0.9,
            source="test",
            uncertainty_ms=5.0,
        ),
        r_offset=MarkerPoint(
            sample=335,
            confidence=0.9,
            source="test",
            uncertainty_ms=5.0,
        ),
        qrs_status=WaveStatus.PRESENT,
    )
    assert clearly_wide.qrs_duration_ms(FS) == 160.0
    assert clearly_wide.qrs_definitely_exceeds_ms(FS, 120.0)


def test_global_wave_edge_requires_two_independent_leads() -> None:
    candidates = [
        _WaveCandidate(100, 120, 155, 8.0, 0),
        _WaveCandidate(102, 121, 156, 7.0, 1),
        _WaveCandidate(101, 122, 190, 9.0, 2),
    ]
    onset, onset_support = MultileadDelineator._supported_wave_edge(
        candidates,
        FS,
        boundary="onset",
        tolerance_ms=22.0,
    )
    offset, offset_support = MultileadDelineator._supported_wave_edge(
        candidates,
        FS,
        boundary="offset",
        tolerance_ms=22.0,
    )
    assert onset == 100
    assert {item.lead_index for item in onset_support} == {0, 1, 2}
    assert offset == 156
    assert {item.lead_index for item in offset_support} == {0, 1}


def test_noise_and_baseline_wander_reduce_certainty_without_invalid_order() -> None:
    clean, r_peaks = _synthetic_ecg()
    noisy, _ = _synthetic_ecg(
        noise_sd=0.055,
        baseline_wander=0.18,
        seed=99,
    )
    delineator = FusionDelineator()
    clean_record = delineator.delineate_record(
        clean,
        r_peaks,
        FS,
        ecg_id=2,
        vcg_xyz=KorsTransform().transform(clean).xyz,
    )
    noisy_record = delineator.delineate_record(
        noisy,
        r_peaks,
        FS,
        ecg_id=3,
        vcg_xyz=KorsTransform().transform(noisy).xyz,
    )
    assert noisy_record.quality_score <= clean_record.quality_score
    assert noisy_record.review_required
    for beat in noisy_record.accepted_beats:
        if beat.r_onset is not None and beat.r_offset is not None:
            assert beat.r_onset.sample < beat.r_peak.sample < beat.r_offset.sample
        if beat.t_onset is not None and beat.t_offset is not None:
            assert beat.t_onset.sample < beat.t_offset.sample


def test_irregular_no_p_pattern_suppresses_discrete_p_measurements() -> None:
    ecg, r_peaks = _synthetic_ecg(
        rr_seconds=(0.62, 0.91, 0.70, 1.21, 0.83, 0.58, 0.98, 0.74, 1.14),
        p_wave=False,
        noise_sd=0.006,
    )
    record = FusionDelineator().delineate_record(
        ecg,
        r_peaks,
        FS,
        ecg_id=4,
        vcg_xyz=KorsTransform().transform(ecg).xyz,
    )
    assert record.rhythm.rhythm_class == RhythmClass.AF_COMPATIBLE
    assert record.rhythm.confidence >= 0.60
    assert all(beat.p_onset is None for beat in record.beats)
    assert all(
        beat.resolved_wave_status("p")
        == WaveStatus.REPLACED_BY_ATRIAL_ACTIVITY
        for beat in record.beats
    )
    assert not record.is_segment_evaluable("p")
    assert not record.is_segment_evaluable("pr")


def test_alternating_ectopic_pattern_is_not_mislabeled_as_af() -> None:
    ecg, r_peaks = _synthetic_ecg(
        rr_seconds=(0.62, 1.08, 0.71, 1.21, 0.66, 0.94, 0.58, 1.13, 0.74),
        p_wave=False,
        noise_sd=0.006,
    )
    record = FusionDelineator().delineate_record(
        ecg,
        r_peaks,
        FS,
        ecg_id=41,
        vcg_xyz=KorsTransform().transform(ecg).xyz,
    )
    assert record.rhythm.rhythm_class == RhythmClass.MIXED
    assert "alternating_rr_ectopic_pattern" in record.rhythm.flags
    assert not record.is_segment_evaluable("p")
    assert all(
        beat.resolved_wave_status("p") == WaveStatus.NOT_ASSESSABLE
        for beat in record.beats
    )


def test_sinus_arrhythmia_with_stable_p_coupling_is_not_called_af() -> None:
    ecg, r_peaks = _synthetic_ecg(
        rr_seconds=(0.74, 0.79, 0.86, 0.95, 1.02, 0.95, 0.86, 0.79, 0.74),
        p_wave=True,
        noise_sd=0.004,
    )
    record = FusionDelineator().delineate_record(
        ecg,
        r_peaks,
        FS,
        ecg_id=42,
        vcg_xyz=KorsTransform().transform(ecg).xyz,
    )
    assert record.rhythm.rhythm_class not in {
        RhythmClass.AF_COMPATIBLE,
        RhythmClass.FLUTTER_COMPATIBLE,
    }
    assert "stable_discrete_p_coupling" in record.rhythm.flags
    assert not record.rhythm.low_discrete_p_coverage
    assert record.is_segment_evaluable("p")


def test_low_p_coverage_guard_yields_to_stable_discrete_coupling() -> None:
    guarded = RhythmAssessment(
        rhythm_class=RhythmClass.REGULAR,
        confidence=0.8,
        p_wave_coverage=0.30,
        flags=("no_consistent_discrete_p",),
    )
    stable = RhythmAssessment(
        rhythm_class=RhythmClass.REGULAR,
        confidence=0.8,
        p_wave_coverage=0.30,
        flags=(
            "no_consistent_discrete_p",
            "stable_discrete_p_coupling",
        ),
    )

    assert guarded.low_discrete_p_coverage
    assert not stable.low_discrete_p_coverage


def test_organized_atrial_activity_never_forces_a_single_p_wave() -> None:
    ecg, r_peaks = _synthetic_ecg(
        p_wave=False,
        atrial_frequency_hz=5.3,
        noise_sd=0.002,
    )
    record = FusionDelineator().delineate_record(
        ecg,
        r_peaks,
        FS,
        ecg_id=5,
        vcg_xyz=KorsTransform().transform(ecg).xyz,
    )
    assert record.rhythm.rhythm_class in {
        RhythmClass.FLUTTER_COMPATIBLE,
        RhythmClass.MULTIPLE_ATRIAL_ACTIVITY,
    }
    assert record.rhythm.discrete_p_not_evaluable
    assert not record.is_segment_evaluable("p")


def test_conflicting_detector_boundaries_choose_consensus_not_average() -> None:
    fused = FusionDelineator()._fuse_marker(
        [
            ("neurokit", _marker(100, source="nk"), 0.4),
            ("multilead", _marker(102, source="ml"), 0.35),
            ("vcg", _marker(160, source="vcg"), 0.25),
        ],
        FS,
        marker_name="t_offset",
    )
    assert fused is not None
    assert 100 <= fused.sample <= 102
    assert fused.conflict_ms >= 100.0
    assert "vcg" not in fused.source


def test_supported_multilead_t_boundary_wins_without_hiding_conflict() -> None:
    multilead = MarkerPoint(
        sample=132,
        confidence=0.82,
        source="multilead_t_consensus",
        uncertainty_ms=12.0,
        support_count=4,
        lead_names=("II", "V4", "V5", "V6"),
        evidence_sources=("latest_supported_lead_boundary",),
    )
    fused = FusionDelineator()._fuse_marker(
        [
            ("neurokit", _marker(100, source="nk"), 0.4),
            ("multilead", multilead, 0.35),
            ("vcg", _marker(102, source="vcg"), 0.25),
        ],
        FS,
        marker_name="t_offset",
    )
    assert fused is not None
    assert fused.sample == 132
    assert fused.conflict_ms == 64.0
    assert fused.confidence < multilead.confidence
    assert "latest_supported_lead_boundary" in fused.evidence_sources


def test_low_confidence_segment_is_not_sent_to_biomarker_search() -> None:
    beat = BeatDelineation(
        r_peak=_marker(300),
        r_onset=_marker(275, 0.25),
        r_offset=_marker(330, 0.25),
        t_onset=_marker(370, 0.25),
        t_peak=_marker(420, 0.25),
        t_offset=_marker(500, 0.25),
        qrs_status=WaveStatus.PRESENT,
        t_status=WaveStatus.PRESENT,
    )
    record = RecordDelineation(
        beats=[beat],
        fs=FS,
        n_samples=700,
        ecg_id=8,
        quality_score=0.3,
    )
    definition = BiomarkerDefinition(
        name="qrs_t_hausdorff_distance",
        family="QRS-T distance",
        description="test",
        formula="test",
        unit="mV",
    )
    result = CatalogFeatureBiomarker(definition).compute(
        record,
        ecg_12lead=np.zeros((700, 12)),
        vcg_xyz=np.zeros((700, 3)),
        fs=FS,
    )
    assert result.value is None
    assert result.confidence == 0.0
    assert "not_evaluable" in result.computation_notes


def test_high_boundary_uncertainty_is_not_sent_to_biomarker_search() -> None:
    beat = BeatDelineation(
        r_peak=_marker(300),
        r_onset=MarkerPoint(
            sample=275,
            confidence=0.90,
            source="test",
            uncertainty_ms=34.0,
        ),
        r_offset=MarkerPoint(
            sample=330,
            confidence=0.90,
            source="test",
            uncertainty_ms=8.0,
        ),
        t_onset=_marker(370),
        t_peak=_marker(420),
        t_offset=_marker(500),
        qrs_status=WaveStatus.PRESENT,
        t_status=WaveStatus.PRESENT,
    )
    record = RecordDelineation(
        beats=[beat],
        fs=FS,
        n_samples=700,
        ecg_id=82,
        quality_score=0.9,
    )
    assert beat.has_excessive_segment_uncertainty("qrs")
    assert beat.segment_confidence("qrs") == 0.0
    assert record.segment_confidence("qrs") == 0.0
    result = QRSDurationBiomarker().compute(
        record,
        ecg_12lead=np.zeros((700, 12)),
        vcg_xyz=np.zeros((700, 3)),
        fs=FS,
    )
    assert result.value is None
    assert "not reliably evaluable" in result.computation_notes


def test_present_wave_with_missing_boundary_abstains_without_error() -> None:
    beat = BeatDelineation(
        r_peak=_marker(300),
        r_onset=_marker(275),
        r_offset=None,
        qrs_status=WaveStatus.PRESENT,
    )
    assert beat.resolved_wave_status("qrs") == WaveStatus.PRESENT
    assert beat.segment_confidence("qrs") == 0.0
    assert beat.segment_boundary_uncertainty_ms("qrs") is None


def test_all_segment_dependent_panel_features_abstain_on_low_confidence() -> None:
    beat = BeatDelineation(
        r_peak=_marker(300, 0.20),
        p_onset=_marker(190, 0.20),
        p_peak=_marker(220, 0.20),
        p_offset=_marker(250, 0.20),
        r_onset=_marker(275, 0.20),
        r_offset=_marker(330, 0.20),
        j_point=_marker(330, 0.20),
        t_onset=_marker(370, 0.20),
        t_peak=_marker(420, 0.20),
        t_offset=_marker(500, 0.20),
        p_status=WaveStatus.PRESENT,
        qrs_status=WaveStatus.PRESENT,
        t_status=WaveStatus.PRESENT,
    )
    record = RecordDelineation(
        beats=[beat],
        fs=FS,
        n_samples=700,
        ecg_id=81,
        quality_score=0.20,
    )
    rng = np.random.default_rng(18)
    ecg = rng.normal(0.0, 0.01, size=(700, 12))
    vcg = rng.normal(0.0, 0.01, size=(700, 3))
    feature_ids = tuple(
        dict.fromkeys(
            feature_id
            for panel in default_feature_panels()
            for feature_id in panel.feature_ids
            if not feature_id.startswith("classical_")
        )
    )
    results = {
        result.name: result
        for result in BiomarkerRegistry.compute_all(
            record,
            ecg_12lead=ecg,
            vcg_xyz=vcg,
            fs=FS,
            names=feature_ids,
        )
    }
    # Every marker here rests on fiducials measured at confidence 0.20, which
    # is below the floor of every wave (P/T 0.15 is cleared, but these markers
    # all need QRS as well, whose floor is 0.35). See
    # DELINEATION_ABSTENTION_2026-08-18.md for why the floor is per wave: the
    # confidence scale is a minimum over boundary markers, and a P boundary
    # cannot reach the values a QRS boundary reaches.
    leaked = []
    for feature_id in feature_ids:
        dependencies = _feature_segment_dependencies(
            feature_id,
            BiomarkerRegistry.definition(feature_id).family,
        )
        result = results[feature_id]
        floor = max(
            (measurement_confidence_threshold(item) for item in dependencies),
            default=0.0,
        )
        if (
            dependencies
            and floor > 0.20
            and result.value is not None
            and np.isfinite(result.value)
        ):
            leaked.append(feature_id)
    assert leaked == [], (
        "markers whose weakest dependency sits below its wave floor must "
        f"abstain; these did not: {leaked[:10]}"
    )
    # Spatial magnitudes are fiducial-global but not origin-free.  With no
    # qualified TP/PR origin they must now abstain instead of silently using
    # the acquisition zero vector.
    assert results["spatial_magnitude_max"].value is None
    assert "physiological VCG origin unavailable" in (
        results["spatial_magnitude_max"].computation_notes
    )


def test_irregular_qtc_uses_each_preceding_rr_and_requires_five_pairs() -> None:
    rr = (0.62, 1.08, 0.74, 1.20, 0.68, 0.96)
    qt = (390.0, 350.0, 430.0, 370.0, 445.0, 360.0, 410.0)
    record = _interval_record(rr, qt, rhythm=RhythmClass.IRREGULAR)
    result = QTCFridericiaBiomarker().compute(record, fs=FS)
    expected = np.median(
        np.asarray(qt[1:]) / np.cbrt(np.asarray(rr))
    )
    assert result.value is not None
    np.testing.assert_allclose(
        result.value,
        expected,
        atol=1e-9,
    )

    short_record = _interval_record(
        rr[:3],
        qt[:4],
        rhythm=RhythmClass.IRREGULAR,
    )
    short_result = QTCFridericiaBiomarker().compute(short_record, fs=FS)
    assert short_result.value is None
    assert "requires 5" in short_result.computation_notes


def test_manual_revision_remains_the_representative_reference() -> None:
    from vcg_biomarker.mimic.ecg_lab import _apply_annotation_revision
    from vcg_biomarker.mimic.models import AnnotationRevision

    ecg, r_peaks = _synthetic_ecg()
    delineator = FusionDelineator()
    record = delineator.delineate_record(
        ecg,
        r_peaks,
        FS,
        ecg_id=9,
        vcg_xyz=KorsTransform().transform(ecg).xyz,
    )
    target = record.median_beat
    assert target is not None
    r_peak = target.r_peak.sample
    revision = AnnotationRevision(
        revision_id="manual-1",
        case_id="case-1",
        parent_revision_id="",
        fiducials={
            "p_onset": r_peak - 112,
            "p_peak": r_peak - 82,
            "p_offset": r_peak - 52,
            "qrs_onset": r_peak - 25,
            "r_peak": r_peak,
            "qrs_offset": r_peak + 30,
            "t_onset": r_peak + 72,
            "t_peak": r_peak + 142,
            "t_offset": r_peak + 238,
        },
    )
    _apply_annotation_revision(record, revision)
    delineator.refresh_record_quality(record)
    representative = record.representative_beat
    assert representative is not None
    assert representative.r_onset.sample == r_peak - 25
    assert representative.t_offset.sample == r_peak + 238
    assert representative.r_onset.manually_reviewed
    assert representative.t_offset.manually_reviewed


def test_multilead_u_wave_requires_a_distinct_supported_post_t_wave() -> None:
    signal = np.zeros((800, 12), dtype=float)
    for lead, scale in ((1, 1.0), (7, 0.8), (8, 0.65)):
        signal[370:501, lead] = np.interp(
            np.arange(370, 501),
            (370, 430, 500),
            (0.0, 0.30 * scale, 0.0),
        )
        signal[535:626, lead] = np.interp(
            np.arange(535, 626),
            (535, 575, 625),
            (0.0, 0.08 * scale, 0.0),
        )
    beat = BeatDelineation(
        r_peak=_marker(300),
        r_onset=_marker(275),
        r_offset=_marker(330),
        t_onset=_marker(370),
        t_peak=_marker(430),
        t_offset=_marker(500),
        qrs_status=WaveStatus.PRESENT,
        t_status=WaveStatus.PRESENT,
    )

    FusionDelineator._detect_u_waves(signal, [beat], FS)

    assert beat.u_status == WaveStatus.PRESENT
    assert beat.u_onset is not None
    assert beat.u_peak is not None
    assert beat.u_offset is not None
    assert 550 <= beat.u_peak.sample <= 600
    assert beat.u_peak.support_count >= 2


def test_missing_u_consensus_is_indeterminate_not_absent() -> None:
    beat = BeatDelineation(
        r_peak=_marker(300),
        r_onset=_marker(275),
        r_offset=_marker(330),
        t_onset=_marker(370),
        t_peak=_marker(430),
        t_offset=_marker(500),
        qrs_status=WaveStatus.PRESENT,
        t_status=WaveStatus.PRESENT,
    )

    FusionDelineator._detect_u_waves(np.zeros((800, 12)), [beat], FS)

    assert beat.u_status == WaveStatus.INDETERMINATE
    assert beat.u_wave_absent is False
    assert beat.u_onset is beat.u_peak is beat.u_offset is None


def test_short_hf_qrs_segment_abstains_instead_of_raising() -> None:
    beat = BeatDelineation(
        r_peak=_marker(305),
        r_onset=_marker(300),
        r_offset=_marker(312),
        t_onset=_marker(360),
        t_peak=_marker(420),
        t_offset=_marker(500),
        qrs_status=WaveStatus.PRESENT,
        t_status=WaveStatus.PRESENT,
    )
    record = RecordDelineation(
        beats=[beat],
        fs=FS,
        n_samples=700,
        ecg_id=91,
        quality_score=0.9,
    )
    definition = BiomarkerDefinition(
        name="qrs_fragmentation_high_frequency_energy",
        family="QRS morphology",
        description="test",
        formula="test",
        unit="mV2",
    )
    result = CatalogFeatureBiomarker(definition).compute(
        record,
        ecg_12lead=np.zeros((700, 12)),
        vcg_xyz=np.zeros((700, 3)),
        fs=FS,
    )

    assert result.value is None
    assert result.computation_notes.startswith("not_evaluable:")
    assert "Error:" not in result.computation_notes


def test_annotation_widget_colors_evaluable_wave_regions_and_correct_st() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.gui.widgets.annotation_widget import AnnotationWidget

    app = QApplication.instance() or QApplication([])
    beat = BeatDelineation(
        r_peak=_marker(300),
        p_onset=_marker(190),
        p_peak=_marker(220),
        p_offset=_marker(250),
        r_onset=_marker(275),
        r_offset=_marker(330),
        j_point=_marker(330),
        t_onset=_marker(370),
        t_peak=_marker(420),
        t_offset=_marker(500),
        p_status=WaveStatus.PRESENT,
        qrs_status=WaveStatus.PRESENT,
        t_status=WaveStatus.PRESENT,
    )
    record = RecordDelineation(
        beats=[beat],
        fs=FS,
        n_samples=700,
        ecg_id=10,
        quality_score=0.9,
    )
    widget = AnnotationWidget(AppConfig())
    widget.display_ecg(np.zeros((700, 12)), FS, record)
    app.processEvents()
    assert len(widget._wave_region_items) == 5 * 12
    rows = {
        widget.interval_table.item(row, 0).text(): float(
            widget.interval_table.item(row, 1).text()
        )
        for row in range(widget.interval_table.rowCount())
    }
    assert rows["ST Segment"] == 80.0
    widget.show_regions_cb.setChecked(False)
    app.processEvents()
    assert not widget._wave_region_items
    widget.close()


def test_explicit_fused_status_blocks_markers_that_are_still_present() -> None:
    beat = BeatDelineation(
        r_peak=_marker(300),
        p_onset=_marker(190),
        p_peak=_marker(220),
        p_offset=_marker(250),
        r_onset=_marker(275),
        r_offset=_marker(330),
        t_onset=_marker(370),
        t_peak=_marker(420),
        t_offset=_marker(500),
        p_status=WaveStatus.FUSED,
        qrs_status=WaveStatus.PRESENT,
        t_status=WaveStatus.FUSED,
    )
    assert beat.resolved_wave_status("p") == WaveStatus.FUSED
    assert beat.resolved_wave_status("t") == WaveStatus.FUSED
    assert beat.segment_confidence("p") == 0.0
    assert beat.segment_confidence("t") == 0.0
    assert beat.pr_interval_ms(FS) is None
    assert beat.qt_interval_ms(FS) is None


def test_low_confidence_qrs_blocks_classical_qrs_rules() -> None:
    beat = BeatDelineation(
        r_peak=_marker(300),
        r_onset=_marker(275, 0.20),
        r_offset=_marker(330, 0.20),
        j_point=_marker(330, 0.20),
        qrs_status=WaveStatus.PRESENT,
    )
    record = RecordDelineation(
        beats=[beat],
        fs=FS,
        n_samples=700,
        ecg_id=11,
        quality_score=0.2,
    )
    results = ClassicalECGCriteria().compute(
        np.zeros((700, 12)),
        record,
        age=55,
        sex="F",
    )
    for criterion_id in (
        "sokolow_lyon",
        "cornell_voltage",
        "pathological_q_waves",
        "st_elevation_contiguous",
    ):
        assert not results[criterion_id].evaluable
        assert results[criterion_id].value is None
        assert results[criterion_id].positive is None


def test_ludb_parser_builds_global_multilead_envelope() -> None:
    samples = [100, 120, 140, 200, 225, 250]
    symbols = ["(", "p", ")", "(", "N", ")"]
    lead_i, malformed = _parse_ludb_annotation(
        samples,
        symbols,
        "i",
    )
    assert malformed == 0
    assert [wave.wave for wave in lead_i] == ["p", "qrs"]

    waves = [
        AnnotatedWave("qrs", 196, 224, 252, "i"),
        AnnotatedWave("qrs", 190, 226, 260, "ii"),
        AnnotatedWave("qrs", 194, 225, 256, "v1"),
    ]
    references = _cluster_global_references(
        waves,
        FS,
        tolerance_ms=50.0,
    )
    assert len(references) == 1
    reference = references[0]
    assert reference.onset == 190
    assert reference.peak == 225
    assert reference.offset == 260
    assert reference.lead_count == 3


def test_ludb_coverage_does_not_count_unannotated_edge_cycles() -> None:
    references = {
        wave: [
            _cluster_global_references(
                [
                    AnnotatedWave(wave, 480, 500, 530, "i"),
                    AnnotatedWave(wave, 478, 502, 532, "ii"),
                ],
                FS,
                tolerance_ms=50.0,
            )[0],
            _cluster_global_references(
                [
                    AnnotatedWave(wave, 980, 1000, 1030, "i"),
                    AnnotatedWave(wave, 978, 1002, 1032, "ii"),
                ],
                FS,
                tolerance_ms=50.0,
            )[0],
        ]
        for wave in ("p", "qrs", "t")
    }
    predictions = {
        wave: [
            PredictedWave(wave, 180, 200, 230, 0.9),
            PredictedWave(wave, 480, 500, 530, 0.9),
            PredictedWave(wave, 980, 1000, 1030, 0.9),
            PredictedWave(wave, 1280, 1300, 1330, 0.9),
        ]
        for wave in ("p", "qrs", "t")
    }
    filtered = _filter_prediction_to_reference_coverage(
        predictions,
        references,
        FS,
    )
    assert all(
        [prediction.peak for prediction in filtered[wave]] == [500, 1000]
        for wave in ("p", "qrs", "t")
    )


def test_g1_t_window_margin_covers_combined_p_window_margin() -> None:
    """G1 regression guard: t_end's own protective margin before the next
    QRS onset must stay >= the next beat's combined P-window margin, or a
    T-offset landing at its own window boundary can invert the next
    beat's p_start/p_end window -- systematic at fast heart rates."""
    from vcg_biomarker.core.delineation.multilead_delineator import (
        P_WINDOW_GAP_AFTER_T_FRACTION,
        P_WINDOW_PRE_QRS_MARGIN_FRACTION,
        T_WINDOW_PRE_NEXT_QRS_MARGIN_FRACTION,
    )

    assert T_WINDOW_PRE_NEXT_QRS_MARGIN_FRACTION >= (
        P_WINDOW_GAP_AFTER_T_FRACTION + P_WINDOW_PRE_QRS_MARGIN_FRACTION
    )


def test_g2_previous_qrs_end_fallback_uses_previous_beats_own_r_peak() -> None:
    """G2 regression guard: at short RR, if the previous beat's QRS bounds
    were not established, the p_start fallback anchor must not land before
    the previous beat's own R-peak (physiologically impossible)."""
    ecg, r_peaks = _synthetic_ecg(rr_seconds=(0.30,) * 8, p_wave=True, noise_sd=0.15)
    delineator = MultileadDelineator()
    beats = delineator.delineate(ecg, r_peaks, FS, vcg_xyz=None)
    assert len(beats) >= 6
    for index in range(1, len(beats)):
        if beats[index].p_onset is None:
            continue
        assert beats[index].p_onset.sample > int(r_peaks[index - 1])


def test_g3_morphology_clusters_stay_separated_at_fast_wide_qrs_rate() -> None:
    """G3 regression guard: at RR=240ms (250bpm), two alternating wide-QRS
    populations must still cluster cleanly instead of the fixed-width
    snippet window reading into the neighboring beat's deflection."""
    rr = 120
    r_samples = tuple(500 + index * rr for index in range(16))
    sinus_samples = r_samples[0::2]
    ectopic_samples = r_samples[1::2]
    beats = [_complete_test_beat(sample) for sample in r_samples]
    signal = np.zeros((500 + 16 * rr + 200, 12), dtype=float)
    _paint_qrs_shape(
        signal, sinus_samples, _CLUSTER_SHAPE_SINUS, _CLUSTER_OFFSETS, _CLUSTER_GAINS
    )
    _paint_qrs_shape(
        signal,
        ectopic_samples,
        _CLUSTER_SHAPE_ECTOPIC_A,
        _CLUSTER_OFFSETS,
        _CLUSTER_GAINS,
    )

    diagnostics = FusionDelineator._mark_morphology_outliers(signal, beats, FS)

    assert diagnostics["morphology_cluster_count"] == 2
    assert diagnostics["morphology_dominant_fraction"] == pytest.approx(0.5)
    assert diagnostics["morphology_secondary_beat_count"] == 8
    cluster_a = {beats[i].morphology_cluster_id for i in range(0, 16, 2)}
    cluster_b = {beats[i].morphology_cluster_id for i in range(1, 16, 2)}
    assert len(cluster_a) == 1 and len(cluster_b) == 1
    assert cluster_a != cluster_b


def test_g4_duplicate_r_peak_detection_rejected_as_implausible() -> None:
    r_samples = tuple(range(500, 5_500, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    duplicate = _complete_test_beat(r_samples[3] + 40)
    all_beats = list(beats) + [duplicate]
    record = RecordDelineation(
        beats=all_beats, fs=FS, n_samples=6_000, ecg_id=220
    )
    plausible_ids = {id(beat) for beat in record.plausible_beats}
    # The duplicate AND its too-close neighbor are both excluded -- from
    # local RR alone there is no way to tell which of a suspiciously close
    # pair is the spurious detection, so both are rejected rather than
    # arbitrarily keeping one.
    assert id(duplicate) not in plausible_ids
    assert id(beats[3]) not in plausible_ids
    assert all(
        id(beat) in plausible_ids for index, beat in enumerate(beats) if index != 3
    )


def test_g4_normal_and_short_but_real_rr_remain_plausible() -> None:
    r_samples = tuple(range(500, 5_500, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    record = RecordDelineation(beats=beats, fs=FS, n_samples=6_000, ecg_id=221)
    assert {id(beat) for beat in record.plausible_beats} == {
        id(beat) for beat in beats
    }

    fast_samples = tuple(500 + index * 220 for index in range(10))
    fast_beats = [_complete_test_beat(sample) for sample in fast_samples]
    fast_record = RecordDelineation(
        beats=fast_beats, fs=FS, n_samples=4_000, ecg_id=222
    )
    assert {id(beat) for beat in fast_record.plausible_beats} == {
        id(beat) for beat in fast_beats
    }


def test_g4_manual_revision_candidate_at_same_r_peak_not_rejected() -> None:
    """A revision candidate replacing a beat at the identical R-peak
    sample must not be rejected as a duplicate-detection artifact just
    because the pre-revision beat is still present in record.beats."""
    r_samples = tuple(range(500, 5_500, 500))
    beats = [_complete_test_beat(sample) for sample in r_samples]
    record = RecordDelineation(beats=beats, fs=FS, n_samples=6_000, ecg_id=223)
    candidate = _complete_test_beat(r_samples[2])
    assert record._is_physiologically_plausible(candidate)


def test_g1_g3_end_to_end_narrow_and_wide_fast_rhythms_do_not_crash() -> None:
    """Full FusionDelineator pipeline at genuinely tachycardic rates
    (regular narrow-complex ~158bpm, regular wide-complex ~171bpm) must
    complete without crashing and must not misclassify a regular fast
    rhythm as AF/irregular."""
    narrow_ecg, narrow_peaks = _synthetic_ecg(
        rr_seconds=(0.38,) * 12, p_wave=True, noise_sd=0.004
    )
    narrow_record = FusionDelineator().delineate_record(
        narrow_ecg,
        narrow_peaks,
        FS,
        ecg_id=230,
        vcg_xyz=KorsTransform().transform(narrow_ecg).xyz,
    )
    assert narrow_record.rhythm.rhythm_class not in {
        RhythmClass.AF_COMPATIBLE,
        RhythmClass.FLUTTER_COMPATIBLE,
    }

    wide_ecg, wide_peaks = _synthetic_ecg(
        rr_seconds=(0.35,) * 12,
        p_wave=False,
        qrs_width="wide",
        noise_sd=0.004,
    )
    wide_record = FusionDelineator().delineate_record(
        wide_ecg,
        wide_peaks,
        FS,
        ecg_id=231,
        vcg_xyz=KorsTransform().transform(wide_ecg).xyz,
    )
    assert wide_record.rhythm.rhythm_class == RhythmClass.WIDE_QRS


def test_g6_rate_aware_pattern_flags() -> None:
    narrow_ecg, narrow_peaks = _synthetic_ecg(
        rr_seconds=(0.38,) * 12, p_wave=True, noise_sd=0.004
    )
    narrow_record = FusionDelineator().delineate_record(
        narrow_ecg,
        narrow_peaks,
        FS,
        ecg_id=240,
        vcg_xyz=KorsTransform().transform(narrow_ecg).xyz,
    )
    assert "narrow_complex_tachycardia_pattern" in narrow_record.rhythm.flags
    assert "wide_complex_tachycardia_pattern" not in narrow_record.rhythm.flags

    wide_fast_ecg, wide_fast_peaks = _synthetic_ecg(
        rr_seconds=(0.35,) * 12, p_wave=False, qrs_width="wide", noise_sd=0.004
    )
    wide_fast_record = FusionDelineator().delineate_record(
        wide_fast_ecg,
        wide_fast_peaks,
        FS,
        ecg_id=241,
        vcg_xyz=KorsTransform().transform(wide_fast_ecg).xyz,
    )
    assert "wide_complex_tachycardia_pattern" in wide_fast_record.rhythm.flags
    assert (
        "accelerated_idioventricular_pattern" not in wide_fast_record.rhythm.flags
    )

    aivr_ecg, aivr_peaks = _synthetic_ecg(
        rr_seconds=(0.70,) * 12, p_wave=False, qrs_width="wide", noise_sd=0.004
    )
    aivr_record = FusionDelineator().delineate_record(
        aivr_ecg,
        aivr_peaks,
        FS,
        ecg_id=242,
        vcg_xyz=KorsTransform().transform(aivr_ecg).xyz,
    )
    assert "accelerated_idioventricular_pattern" in aivr_record.rhythm.flags
    assert "wide_complex_tachycardia_pattern" not in aivr_record.rhythm.flags


def test_g7_wct_differentiation_markers() -> None:
    wide_ecg, wide_peaks = _synthetic_ecg(
        rr_seconds=(0.35,) * 12,
        p_wave=False,
        qrs_width="wide",
        noise_sd=0.001,
    )
    wide_vcg_xyz = KorsTransform().transform(wide_ecg).xyz
    wide_record = FusionDelineator().delineate_record(
        wide_ecg,
        wide_peaks,
        FS,
        ecg_id=250,
        vcg_xyz=wide_vcg_xyz,
    )
    assert "wide_complex_tachycardia_pattern" in wide_record.rhythm.flags
    for name in (
        "precordial_rs_absent_flag",
        "precordial_concordance_flag",
        "avr_vi_vt_ratio",
    ):
        value = _advanced_marker_value(name, wide_record, wide_ecg, wide_vcg_xyz)
        assert value is not None and np.isfinite(value), name

    narrow_ecg, narrow_peaks = _synthetic_ecg(
        rr_seconds=(0.38,) * 12, p_wave=True, noise_sd=0.004
    )
    narrow_record = FusionDelineator().delineate_record(
        narrow_ecg,
        narrow_peaks,
        FS,
        ecg_id=251,
        vcg_xyz=KorsTransform().transform(narrow_ecg).xyz,
    )
    assert "wide_complex_tachycardia_pattern" not in narrow_record.rhythm.flags
    for name in (
        "precordial_rs_absent_flag",
        "precordial_concordance_flag",
        "avr_vi_vt_ratio",
    ):
        value = _advanced_marker_value(
            name,
            narrow_record,
            narrow_ecg,
            KorsTransform().transform(narrow_ecg).xyz,
        )
        assert value is None, name


def test_g5_qt_rr_fast_beat_exclusion_count() -> None:
    record = _interval_record(
        (0.82, 0.20, 0.82),
        (380.0, 380.0, 380.0, 380.0),
        rhythm=RhythmClass.REGULAR,
    )
    assert record.qt_rr_fast_beat_exclusion_count() == 1
    assert "qt_rr_fast_beat_exclusion_count" not in record.diagnostics


def _g12_test_record(
    offsets: tuple[float, ...],
    amplitudes: tuple[float, ...],
    onset_offset_samples: int,
    offset_offset_samples: int,
    *,
    pr_interval_samples: int | None = None,
) -> tuple[np.ndarray, RecordDelineation, np.ndarray]:
    r_samples = tuple(range(500, 500 + 8 * 600, 600))
    n_samples = r_samples[-1] + 500
    gains = np.asarray(
        [0.75, 1.00, 0.65, -0.70, 0.55, 0.80, -0.45, 0.65, 0.85, 1.05, 1.10, 0.95]
    )
    signal = np.zeros((n_samples, 12), dtype=float)
    for r_sample in r_samples:
        for lead in range(12):
            _add_piecewise_wave(
                signal[:, lead], r_sample, offsets, amplitudes, float(gains[lead])
            )
    beats = [
        BeatDelineation(
            r_peak=_marker(r_sample),
            p_onset=(
                _marker(r_sample + onset_offset_samples - pr_interval_samples)
                if pr_interval_samples is not None
                else None
            ),
            p_status=(
                WaveStatus.PRESENT if pr_interval_samples is not None else WaveStatus.ABSENT
            ),
            r_onset=_marker(r_sample + onset_offset_samples),
            r_offset=_marker(r_sample + offset_offset_samples),
            j_point=_marker(r_sample + offset_offset_samples),
            qrs_status=WaveStatus.PRESENT,
        )
        for r_sample in r_samples
    ]
    record = RecordDelineation(beats=beats, fs=FS, n_samples=n_samples, ecg_id=260)
    vcg_xyz = KorsTransform().transform(signal).xyz
    return signal, record, vcg_xyz


def test_g12_delta_wave_and_accessory_pathway_markers() -> None:
    # WPW-like slurred QRS onset: slow initial 40ms rise (-80ms to -40ms
    # before R) followed by a sharp rise to the R peak.
    delta_signal, delta_record, delta_vcg_xyz = _g12_test_record(
        (-0.080, -0.040, 0.0, 0.030, 0.060),
        (0.0, 0.15, 1.0, -0.30, 0.0),
        -40,
        30,
    )
    present = _advanced_marker_value(
        "delta_wave_present_flag", delta_record, delta_signal, delta_vcg_xyz
    )
    assert present == 1.0

    axis = _advanced_marker_value(
        "delta_wave_frontal_axis_deg", delta_record, delta_signal, delta_vcg_xyz
    )
    assert axis is not None and np.isfinite(axis)

    quadrant = _advanced_marker_value(
        "accessory_pathway_quadrant_estimate", delta_record, delta_signal, delta_vcg_xyz
    )
    assert quadrant in (0.0, 1.0, 2.0, 3.0)

    # Normal, sharp QRS onset -- no delta wave, so the axis/quadrant
    # markers must abstain rather than report a spurious localization.
    normal_signal, normal_record, normal_vcg_xyz = _g12_test_record(
        (-0.050, -0.027, 0.0, 0.028, 0.055),
        (0.0, -0.18, 1.0, -0.30, 0.0),
        -25,
        28,
    )
    assert (
        _advanced_marker_value(
            "delta_wave_present_flag", normal_record, normal_signal, normal_vcg_xyz
        )
        == 0.0
    )
    assert (
        _advanced_marker_value(
            "delta_wave_frontal_axis_deg", normal_record, normal_signal, normal_vcg_xyz
        )
        is None
    )
    assert (
        _advanced_marker_value(
            "accessory_pathway_quadrant_estimate",
            normal_record,
            normal_signal,
            normal_vcg_xyz,
        )
        is None
    )


def test_g12_delta_wave_requires_short_pr_when_p_available() -> None:
    # Same slurred-onset QRS shape as the positive delta-wave case, but
    # now with a NORMAL (long) PR interval of 200ms -- a real LUDB
    # spot-check found the slope heuristic alone false-positives on
    # unrelated causes of a slurred QRS onset (LVH/scar, post-MI Q
    # waves), so a short PR (<120ms) must corroborate it.
    signal, record, vcg_xyz = _g12_test_record(
        (-0.080, -0.040, 0.0, 0.030, 0.060),
        (0.0, 0.15, 1.0, -0.30, 0.0),
        -40,
        30,
        pr_interval_samples=100,
    )
    present = _advanced_marker_value(
        "delta_wave_present_flag", record, signal, vcg_xyz
    )
    assert present == 0.0

    # Same slurred onset with a genuinely SHORT PR interval (100ms) must
    # still flag positive.
    short_pr_signal, short_pr_record, short_pr_vcg_xyz = _g12_test_record(
        (-0.080, -0.040, 0.0, 0.030, 0.060),
        (0.0, 0.15, 1.0, -0.30, 0.0),
        -40,
        30,
        pr_interval_samples=50,
    )
    short_pr_present = _advanced_marker_value(
        "delta_wave_present_flag", short_pr_record, short_pr_signal, short_pr_vcg_xyz
    )
    assert short_pr_present == 1.0


def test_g13_flutter_av_block_ratio_estimate() -> None:
    r_samples = tuple(range(500, 500 + 6 * 400, 400))
    beats = [_complete_test_beat(r_sample) for r_sample in r_samples]
    n_samples = r_samples[-1] + 500
    signal = np.zeros((n_samples, 12), dtype=float)
    vcg_xyz = KorsTransform().transform(signal).xyz

    flutter_rhythm = RhythmAssessment(
        rhythm_class=RhythmClass.FLUTTER_COMPATIBLE,
        confidence=0.9,
        rr_median_ms=400.0,
        atrial_frequency_hz=5.0,
    )
    flutter_record = RecordDelineation(
        beats=beats, fs=FS, n_samples=n_samples, ecg_id=270, rhythm=flutter_rhythm
    )
    ratio = _advanced_marker_value(
        "flutter_av_block_ratio_estimate", flutter_record, signal, vcg_xyz
    )
    assert ratio == pytest.approx(2.0)

    regular_rhythm = RhythmAssessment(
        rhythm_class=RhythmClass.REGULAR,
        confidence=0.9,
        rr_median_ms=400.0,
        atrial_frequency_hz=5.0,
    )
    regular_record = RecordDelineation(
        beats=beats, fs=FS, n_samples=n_samples, ecg_id=271, rhythm=regular_rhythm
    )
    assert (
        _advanced_marker_value(
            "flutter_av_block_ratio_estimate", regular_record, signal, vcg_xyz
        )
        is None
    )
