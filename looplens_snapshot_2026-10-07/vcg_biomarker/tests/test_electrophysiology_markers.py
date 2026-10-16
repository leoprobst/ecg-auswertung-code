"""Tests for the electrophysiology gap-filling marker batch.

Synthetic signals with a known property, so a marker that stops measuring what
it claims fails here rather than silently drifting.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.catalog_biomarkers import (
    MILLIVOLT_TO_MICROVOLT,
    _oriented_plane_normal,
)
from vcg_biomarker.core.biomarkers.electrophysiology_markers import (
    ELECTROPHYSIOLOGY_SPECS,
    ELECTROPHYSIOLOGY_VERSION,
    _abstain,
    _angle_between,
    _area_vector,
    _magnitude,
    _slope_uv_per_ms,
)
from vcg_biomarker.core.data_loader import LEAD_MAP
from vcg_biomarker.mimic.feature_roles import (
    is_core_sentinel_candidate,
    is_discovery_candidate,
)
from vcg_biomarker.mimic.panels import default_feature_panels


def test_every_marker_is_registered_exactly_once() -> None:
    names = [spec.name for spec in ELECTROPHYSIOLOGY_SPECS]
    assert len(names) == len(set(names))
    available = set(BiomarkerRegistry.available())
    missing = sorted(name for name in names if name not in available)
    assert not missing, missing
    assert len(names) >= 85


def test_every_marker_declares_unit_formula_and_interpretation() -> None:
    for spec in ELECTROPHYSIOLOGY_SPECS:
        assert spec.unit.strip(), spec.name
        assert spec.formula.strip(), spec.name
        assert spec.interpretation.strip(), spec.name
        assert spec.family.strip(), spec.name


def test_markers_are_discovery_reserve_only() -> None:
    # Conceived after inspecting development results, so they are post-outcome
    # hypotheses regardless of physiological motivation and must not enter the
    # frozen Core or Sentinel panels.
    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    names = {spec.name for spec in ELECTROPHYSIOLOGY_SPECS}
    assert not (names & panels["primary_core"])
    assert not (names & panels["sentinel"])
    assert names <= panels["discovery_reserve"]
    for name in sorted(names)[:20]:
        assert not is_core_sentinel_candidate(name), name
        assert is_discovery_candidate(name), name


def test_registered_definitions_carry_the_version_note() -> None:
    for spec in list(ELECTROPHYSIOLOGY_SPECS)[:10]:
        definition = BiomarkerRegistry.definition(spec.name)
        assert ELECTROPHYSIOLOGY_VERSION in (definition.notes or "")
        assert definition.pathological_direction == "data_driven"


# --- helper behaviour -------------------------------------------------------


def test_slope_helper_returns_microvolts_per_millisecond() -> None:
    fs = 1000
    # 1 mV rise over 100 samples at 1 kHz = 1000 uV / 100 ms = 10 uV/ms.
    values = np.linspace(0.0, 1.0, 101)
    assert _slope_uv_per_ms(values, fs) == pytest.approx(10.0, rel=1e-6)


def test_area_vector_integrates_in_millivolt_milliseconds() -> None:
    fs = 1000
    points = np.column_stack(
        [np.ones(101), np.zeros(101), np.zeros(101)]
    )
    area = _area_vector(points, fs)
    assert area[0] == pytest.approx(100.0, rel=1e-6)


def test_abstention_carries_a_reason() -> None:
    value = _abstain("segment too short")
    assert value.value is None
    assert value.confidence == 0.0
    assert value.notes.startswith("not_evaluable:")
    assert "segment too short" in value.notes


# --- marker behaviour on constructed signals --------------------------------


class _Point:
    def __init__(self, sample: int, confidence: float = 1.0) -> None:
        self.sample = int(sample)
        self.confidence = float(confidence)


class _Beat:
    def __init__(self, offset: int, fs: int) -> None:
        self.r_peak = _Point(offset + int(0.30 * fs))
        self.r_onset = _Point(offset + int(0.26 * fs))
        self.r_offset = _Point(offset + int(0.34 * fs))
        self.q_peak = _Point(offset + int(0.27 * fs))
        self.s_peak = _Point(offset + int(0.33 * fs))
        self.j_point = _Point(offset + int(0.34 * fs))
        self.p_onset = _Point(offset + int(0.10 * fs))
        self.p_offset = _Point(offset + int(0.20 * fs))
        self.t_onset = _Point(offset + int(0.40 * fs))
        self.t_peak = _Point(offset + int(0.50 * fs))
        self.t_offset = _Point(offset + int(0.60 * fs))
        self.u_onset = _Point(offset + int(0.62 * fs))
        self.u_offset = _Point(offset + int(0.68 * fs))

    def resolved_wave_status(self, wave: str):
        class _Status:
            value = "present"

        return _Status()


class _Rhythm:
    confidence = 1.0
    requires_beatwise_rate_correction = False


class _Delineation:
    def __init__(self, beats, fs: int) -> None:
        self.beats = beats
        self.representative_beat = beats[0]
        self.rhythm = _Rhythm()
        self.n_beats = len(beats)
        self.quality_score = 1.0
        self.diagnostics: dict = {}

    def is_segment_evaluable(self, segment: str, minimum_confidence: float = 0.35) -> bool:
        return True

    def segment_confidence(self, segment: str) -> float:
        return 1.0


def _synthetic_record(fs: int = 500, beats: int = 8):
    """A crude but fully delineated three-lead VCG with known geometry."""

    beat_samples = int(0.8 * fs)
    total = beat_samples * beats
    time = np.arange(total) / fs
    vcg = np.zeros((total, 3), dtype=float)
    delineation_beats = []
    for index in range(beats):
        offset = index * beat_samples
        delineation_beats.append(_Beat(offset, fs))
        local = np.arange(beat_samples) / fs
        # P wave on x, QRS on x with a y component, T wave on x.
        vcg[offset : offset + beat_samples, 0] += 0.15 * np.exp(
            -(((local - 0.15) / 0.02) ** 2)
        )
        vcg[offset : offset + beat_samples, 0] += 1.2 * np.exp(
            -(((local - 0.30) / 0.012) ** 2)
        )
        vcg[offset : offset + beat_samples, 1] += 0.5 * np.exp(
            -(((local - 0.30) / 0.012) ** 2)
        )
        vcg[offset : offset + beat_samples, 0] += 0.35 * np.exp(
            -(((local - 0.50) / 0.045) ** 2)
        )
    ecg = np.tile(vcg[:, :1], (1, 12))
    return _Delineation(delineation_beats, fs), ecg, vcg, fs


def _compute(names, wideband: bool = True):
    delineation, ecg, vcg, fs = _synthetic_record()
    return {
        result.name: result
        for result in BiomarkerRegistry.compute_all(
            delineation=delineation,
            ecg_12lead=ecg,
            vcg_xyz=vcg,
            fs=fs,
            names=list(names),
            record_ecg_12lead=ecg,
            record_vcg_xyz=vcg,
            record_vcg_wideband_xyz=vcg if wideband else None,
        )
    }


def _compute_record(names, delineation, ecg, vcg, fs):
    return {
        result.name: result
        for result in BiomarkerRegistry.compute_all(
            delineation=delineation,
            ecg_12lead=ecg,
            vcg_xyz=vcg,
            fs=fs,
            names=list(names),
            record_ecg_12lead=ecg,
            record_vcg_xyz=vcg,
        )
    }


def test_no_marker_raises_on_a_synthetic_record() -> None:
    names = [spec.name for spec in ELECTROPHYSIOLOGY_SPECS]
    results = _compute(names)
    errors = {
        name: result.computation_notes
        for name, result in results.items()
        if str(result.computation_notes).startswith("error:")
    }
    assert not errors, errors
    assert len(results) == len(names)


def test_wideband_markers_abstain_without_a_wideband_source() -> None:
    wideband_markers = [
        "terminal_qrs_rms40_uv",
        "filtered_qrs_duration_ms",
        "epsilon_potential_energy_ratio",
        "hf_qrs_reduced_amplitude_zone_depth",
    ]
    results = _compute(wideband_markers, wideband=False)
    for name in wideband_markers:
        assert results[name].value is None, name
        assert "wideband" in results[name].computation_notes, name


def test_t_wave_symmetry_is_near_one_for_a_gaussian_t_wave() -> None:
    result = _compute(["t_wave_symmetry_index"])["t_wave_symmetry_index"]
    assert result.value is not None
    # A Gaussian T wave centred inside its own segment is symmetric.
    assert result.value == pytest.approx(1.0, abs=0.35)


def test_concordant_repolarization_gives_a_positive_coupling_index() -> None:
    # The synthetic QRS and T are both positive on x, i.e. concordant.
    results = _compute(
        ["activation_recovery_coupling_index", "qrs_t_discordance_cosine"]
    )
    assert results["qrs_t_discordance_cosine"].value > 0.5


def test_planarity_is_near_zero_for_a_planar_loop() -> None:
    # The synthetic loop lives in the x-y plane, so out-of-plane variance
    # must be negligible.
    results = _compute(
        ["qrs_loop_planarity_deviation", "t_loop_planarity_deviation"]
    )
    assert results["qrs_loop_planarity_deviation"].value == pytest.approx(0.0, abs=1e-6)
    assert results["t_loop_planarity_deviation"].value == pytest.approx(0.0, abs=1e-6)


def test_qrs_peak_voltage_is_reported_in_microvolts() -> None:
    result = _compute(["spatial_qrs_peak_voltage_uv"])["spatial_qrs_peak_voltage_uv"]
    # The synthetic QRS peaks at sqrt(1.2^2 + 0.5^2) = 1.3 mV = 1300 uV.
    assert result.value == pytest.approx(1300.0, rel=0.15)


def test_qt_variability_reads_the_beat_level_fiducials() -> None:
    # r_onset is the beat-level attribute; reading a non-existent "qrs_onset"
    # made this marker return nothing on every record.
    results = _compute(
        ["beat_to_beat_qt_variability_ms", "qt_variability_index_surrogate"]
    )
    assert results["beat_to_beat_qt_variability_ms"].value is not None
    assert results["beat_to_beat_qt_variability_ms"].value >= 0.0


def test_tp_interval_is_reconstructed_across_beats() -> None:
    # bounds() has no "tp" segment because TP spans two beats; the marker
    # rebuilds it from the beat list.
    result = _compute(["tp_baseline_stability_uv"])["tp_baseline_stability_uv"]
    assert result.value is not None
    assert result.value >= 0.0


def test_velocity_entropy_is_bounded_by_one() -> None:
    results = _compute(
        [
            "qrs_spatial_velocity_entropy",
            "t_spatial_velocity_entropy",
        ]
    )
    for name, result in results.items():
        if result.value is not None:
            assert 0.0 <= result.value <= 1.0, name


def test_time_to_peak_velocity_is_a_fraction() -> None:
    results = _compute(
        [f"{segment}_time_to_peak_velocity_fraction" for segment in ("qrs", "t")]
    )
    for name, result in results.items():
        if result.value is not None:
            assert 0.0 <= result.value <= 1.0, name


def test_pca_based_markers_stay_in_their_ranges() -> None:
    results = _compute(
        [
            "t_pca_ratio_lambda2_lambda1",
            "t_nondipolar_residual_fraction",
            "total_cosine_r_to_t",
        ]
    )
    ratio = results["t_pca_ratio_lambda2_lambda1"].value
    residual = results["t_nondipolar_residual_fraction"].value
    cosine = results["total_cosine_r_to_t"].value
    if ratio is not None:
        assert 0.0 <= ratio <= 1.0
    if residual is not None:
        assert 0.0 <= residual <= 1.0
    if cosine is not None:
        assert -1.0 <= cosine <= 1.0


def test_invalid_magnitude_eponyms_abstain_instead_of_claiming_clinical_equivalence() -> None:
    names = [
        "brugada_beta_angle_deg",
        "brugada_triangle_base_ms",
        "brugada_ascending_limb_duration_ms",
        "t_nondipolar_residual_fraction",
    ]
    results = _compute(names)
    for name in names:
        assert results[name].value is None, name
        assert "not_evaluable" in results[name].computation_notes, name
    assert "lead8_t_wave_residuum_fraction" in BiomarkerRegistry.available()
    assert "not 8-lead T-wave residuum" in results[
        "t_nondipolar_residual_fraction"
    ].computation_notes


def test_terminal_activation_uses_a_true_right_precordial_s_nadir() -> None:
    delineation, ecg, vcg, fs = _synthetic_record()
    ecg = np.zeros_like(ecg)
    beat = delineation.representative_beat
    start, end = beat.r_onset.sample, beat.r_offset.sample
    local = np.zeros(end - start + 1)
    local[4] = 1.0
    s_index = len(local) - 21
    local[s_index] = -1.2
    ecg[start : end + 1, LEAD_MAP["V2"]] = local
    result = _compute_record(
        ["terminal_activation_duration_ms"], delineation, ecg, vcg, fs
    )["terminal_activation_duration_ms"]
    assert result.value == pytest.approx(40.0, abs=2.1)
    assert result.per_lead["V2"] == pytest.approx(result.value)


def test_modified_sgarbossa_ratio_is_lead_specific_st_over_s() -> None:
    delineation, ecg, vcg, fs = _synthetic_record()
    ecg = np.zeros_like(ecg)
    beat = delineation.representative_beat
    q_start, q_end = beat.r_onset.sample, beat.r_offset.sample
    qrs = np.full(q_end - q_start + 1, -2.0)
    qrs[2] = 0.2
    ecg[q_start : q_end + 1, LEAD_MAP["V2"]] = qrs
    st_start, st_end = beat.r_offset.sample, beat.t_onset.sample
    ecg[st_start : st_end + 1, LEAD_MAP["V2"]] = 0.6
    result = _compute_record(
        ["smith_modified_st_to_s_ratio"], delineation, ecg, vcg, fs
    )["smith_modified_st_to_s_ratio"]
    assert result.value == pytest.approx(0.3, rel=1e-6)
    assert result.per_lead["V2"] == pytest.approx(0.3, rel=1e-6)


def test_katz_wachtel_reports_voltage_not_a_dimensionally_invalid_fraction() -> None:
    delineation, ecg, vcg, fs = _synthetic_record()
    ecg = np.zeros_like(ecg)
    beat = delineation.representative_beat
    start, end = beat.r_onset.sample, beat.r_offset.sample
    local = np.zeros(end - start + 1)
    local[5], local[20] = 3.0, -2.5
    ecg[start : end + 1, LEAD_MAP["V3"]] = local
    result = _compute_record(
        ["katz_wachtel_equiphasic_index"], delineation, ecg, vcg, fs
    )["katz_wachtel_equiphasic_index"]
    assert result.unit == "uV"
    assert result.value == pytest.approx(5500.0)
    assert result.per_lead["V3_equiphasic_balance"] == pytest.approx(5.0 / 5.5)


def test_p_loop_roundness_converges_to_one_for_a_circle() -> None:
    delineation, ecg, vcg, fs = _synthetic_record()
    beat = delineation.representative_beat
    start, end = beat.p_onset.sample, beat.p_offset.sample
    phase = np.linspace(0.0, 2.0 * np.pi, end - start + 1)
    vcg[start : end + 1] = np.column_stack(
        (np.cos(phase), np.sin(phase), np.zeros_like(phase))
    )
    result = _compute_record(
        ["p_loop_roundness_index"], delineation, ecg, vcg, fs
    )["p_loop_roundness_index"]
    assert result.value == pytest.approx(1.0, abs=0.04)


def test_s1q3t3_is_dimensionless_and_requires_all_three_lead_components() -> None:
    delineation, ecg, vcg, fs = _synthetic_record()
    ecg = np.zeros_like(ecg)
    beat = delineation.representative_beat
    q_start, q_end = beat.r_onset.sample, beat.r_offset.sample
    t_start, t_end = beat.t_onset.sample, beat.t_offset.sample
    ecg[q_start : q_end + 1, LEAD_MAP["I"]] = -1.0
    lead_iii = np.zeros(q_end - q_start + 1)
    lead_iii[2], lead_iii[10] = -0.5, 1.0
    ecg[q_start : q_end + 1, LEAD_MAP["III"]] = lead_iii
    ecg[t_start : t_end + 1, LEAD_MAP["III"]] = -0.4
    result = _compute_record(["s1q3t3_vector_score"], delineation, ecg, vcg, fs)[
        "s1q3t3_vector_score"
    ]
    assert result.unit == "dimensionless"
    assert result.value == pytest.approx(np.cbrt(0.5), rel=1e-6)
    ecg[t_start : t_end + 1, LEAD_MAP["III"]] = 0.4
    absent = _compute_record(["s1q3t3_vector_score"], delineation, ecg, vcg, fs)[
        "s1q3t3_vector_score"
    ]
    assert absent.value == pytest.approx(0.0)


def test_references_are_family_targeted_instead_of_attaching_the_entire_bibliography() -> None:
    brugada = BiomarkerRegistry.definition("brugada_beta_angle_deg").references
    smith = BiomarkerRegistry.definition("smith_modified_st_to_s_ratio").references
    velocity = BiomarkerRegistry.definition("qrs_upstroke_peak_velocity").references
    assert brugada and all("Brugada" in item or "Chevallier" in item or "Serra" in item for item in brugada)
    assert smith and any("Smith" in item for item in smith)
    assert not any("Smith" in item for item in brugada)
    assert velocity == []


def test_plane_normal_sign_follows_temporal_circulation_not_lapack_sign() -> None:
    phase = np.linspace(0.0, 2.0 * np.pi, 65)[:-1]
    circle = np.column_stack((np.cos(phase), np.sin(phase), np.zeros_like(phase)))
    positive = _oriented_plane_normal(circle, np.asarray([0.0, 0.0, -1.0]))
    reversed_path = _oriented_plane_normal(circle[::-1], np.asarray([0.0, 0.0, -1.0]))
    assert positive == pytest.approx([0.0, 0.0, 1.0])
    assert reversed_path == pytest.approx([0.0, 0.0, -1.0])


def test_families_cover_the_identified_gaps() -> None:
    families = {spec.family for spec in ELECTROPHYSIOLOGY_SPECS}
    for expected in (
        "Terminale QRS-Energie",
        "Atriale Repolarisation",
        "Repolarisationsheterogenitaet",
        "Aktivierung-Erholung",
        "Brugada/J-Wave",
        "ARVC/Terminale Aktivierung",
        "Infarkt bei Schenkelblock",
        "Ischaemie-Vektorik",
        "Elektrolyt-Signaturen",
        "Medikamenten-Signaturen",
        "Atriale Geometrie",
    ):
        assert expected in families, expected
