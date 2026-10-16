"""Tests for the third wave: echo diastology, absolute risk, circadian,
dose and external cohorts.

Where a quantity has a known analytic value the test checks against it rather
than against a stored expectation, so a regression shows up as a wrong number
and not merely as a changed one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.core.delineation.benchmark_suite import (
    benchmark_root_diagnostics,
    resolve_benchmark_root,
)
from vcg_biomarker.mimic.absolute_risk import (
    EVENT_CENSORED,
    EVENT_COMPETING,
    EVENT_OF_INTEREST,
    aalen_johansen,
    absolute_risk_report,
    calibration_by_risk_group,
    calibration_metrics,
    kaplan_meier_complement,
    net_benefit_from_absolute_risk,
    stratified_absolute_risk,
)
from vcg_biomarker.mimic.circadian import (
    CIRCADIAN_COVARIATES,
    acquisition_intensity_profile,
    circadian_features,
    circadian_variation_table,
    exceeds_measurement_noise,
    marker_circadian_variation,
    summarize_circadian,
)
from vcg_biomarker.mimic.evidence import echo_derived_diastolic_columns
from vcg_biomarker.mimic.external_cohorts import (
    EXTERNAL_SOURCES,
    ROLE_DELINEATION_BENCHMARK,
    ROLE_DEVELOPMENT_REUSED,
    ROLE_INDEPENDENT_EXTERNAL,
    SNOMED_LABELS,
    eligible_for_diagnostic_validation,
    external_claim_scope,
    independent_source_ids,
    map_snomed_codes,
    source_by_directory,
    source_powerline_hz,
    source_registry_table,
)
from vcg_biomarker.mimic.external_validation import (
    MAXIMUM_TRANSFERABLE_SHIFT,
    distribution_shift,
    extraction_coverage,
    load_frozen_external_contract,
    site_consistency,
)
from vcg_biomarker.mimic.statistics import MarkerModel
from vcg_biomarker.mimic.storage import content_hash, file_sha256
from vcg_biomarker.mimic.label_registry import default_label_definitions
from vcg_biomarker.mimic.medication_dose import (
    DOSE_STATUS_RANGE_MIDPOINT,
    DOSE_STATUS_REDACTED,
    dose_response_levels,
    dose_response_trend,
    exposure_summary,
    parse_dose_frame,
    parse_dose_value,
    to_milligrams,
)


# --- 1: echo diastology and geometry --------------------------------------


def test_diastolic_and_geometry_columns_are_registered() -> None:
    columns = set(echo_derived_diastolic_columns())
    for expected in (
        "mitral_e_velocity_m_per_s",
        "mitral_a_velocity_m_per_s",
        "mitral_e_deceleration_time_ms",
        "mitral_e_a_ratio",
        "aortic_peak_velocity_m_per_s",
        "la_anteroposterior_dimension_cm",
        "ra_length_cm",
        "aortic_sinus_diameter_cm",
        "relative_wall_thickness_echo",
    ):
        assert expected in columns, expected


def test_new_quantitative_echo_endpoints_exist_with_rules() -> None:
    from vcg_biomarker.mimic.labels import _label_rules

    definitions = {item.label_id for item in default_label_definitions()}
    rules = _label_rules()
    for endpoint in (
        "diastolic_relaxation_abnormality",
        "restrictive_filling_pattern",
        "elevated_filling_pressure_proxy",
        "severe_aortic_stenosis_velocity",
        "moderate_aortic_stenosis_velocity",
        "la_dilatation_linear",
        "ra_dilatation_linear",
        "concentric_remodeling_rwt",
        "aortic_root_dilatation",
    ):
        assert endpoint in definitions, endpoint
        assert endpoint in rules, endpoint
    # Registry and SQL implementation must stay in lockstep.
    assert definitions == set(rules)


def test_atrial_endpoints_are_paired_with_atrial_markers() -> None:
    from vcg_biomarker.mimic.validation import PRIMARY_ADJUSTED_MARKERS

    # The point of the diastolic extension: a P-loop marker now has an
    # atrial-size reference standard instead of only LVEF and valve grades.
    assert PRIMARY_ADJUSTED_MARKERS["la_dilatation_linear"] == "p_loop_width_major"
    assert (
        PRIMARY_ADJUSTED_MARKERS["diastolic_relaxation_abnormality"]
        == "p_loop_width_major"
    )
    assert (
        PRIMARY_ADJUSTED_MARKERS["severe_aortic_stenosis_velocity"]
        == "svg_magnitude"
    )


def test_new_echo_endpoint_declarations_do_not_attest_actual_reference() -> None:
    from vcg_biomarker.mimic.reference_tiers import TIER_UNKNOWN, reference_tier_table

    table = reference_tier_table(default_label_definitions()).set_index("label_id")
    for endpoint in (
        "diastolic_relaxation_abnormality",
        "severe_aortic_stenosis_velocity",
        "la_dilatation_linear",
    ):
        assert table.loc[endpoint, "reference_tier"] == TIER_UNKNOWN
        assert not bool(table.loc[endpoint, "accuracy_claim_supported"])


# --- 2: absolute risk under competing events ------------------------------


def _competing_risk_sample(seed: int = 5, n: int = 20000):
    rng = np.random.default_rng(seed)
    rate_event, rate_death = 0.10, 0.15
    t_event = rng.exponential(1 / rate_event, n)
    t_death = rng.exponential(1 / rate_death, n)
    t_censor = rng.uniform(0, 6, n)
    duration = np.minimum(np.minimum(t_event, t_death), t_censor)
    event = np.where(
        (t_event <= t_death) & (t_event <= t_censor),
        EVENT_OF_INTEREST,
        np.where(
            (t_death < t_event) & (t_death <= t_censor),
            EVENT_COMPETING,
            EVENT_CENSORED,
        ),
    )
    return duration, event, rate_event, rate_death


def test_aalen_johansen_matches_the_analytic_cumulative_incidence() -> None:
    duration, event, rate_event, rate_death = _competing_risk_sample()
    curve = aalen_johansen(duration, event)
    total = rate_event + rate_death
    for horizon in (1.0, 3.0, 5.0):
        truth = rate_event / total * (1.0 - np.exp(-total * horizon))
        assert curve.risk_at(horizon) == pytest.approx(truth, abs=0.01)


def test_naive_one_minus_km_overestimates_with_competing_events() -> None:
    duration, event, rate_event, rate_death = _competing_risk_sample()
    curve = aalen_johansen(duration, event)
    naive = kaplan_meier_complement(duration, event)
    # The distortion grows with the horizon, which is exactly why treating a
    # competing death as censoring is not a harmless simplification.
    bias_short = naive.risk_at(1.0) - curve.risk_at(1.0)
    bias_long = naive.risk_at(5.0) - curve.risk_at(5.0)
    assert bias_short > 0
    assert bias_long > bias_short
    assert bias_long > 0.05


def test_cumulative_incidence_interval_stays_inside_the_unit_interval() -> None:
    duration, event, _, _ = _competing_risk_sample(seed=7, n=2000)
    curve = aalen_johansen(duration, event)
    low, high = curve.confidence_interval_at(3.0)
    assert 0.0 <= low <= curve.risk_at(3.0) <= high <= 1.0


def test_no_competing_events_reduces_to_one_minus_km() -> None:
    rng = np.random.default_rng(11)
    duration = rng.exponential(5.0, 3000)
    event = np.where(duration < 4.0, EVENT_OF_INTEREST, EVENT_CENSORED)
    aj = aalen_johansen(duration, event)
    km = kaplan_meier_complement(duration, event)
    assert aj.risk_at(3.0) == pytest.approx(km.risk_at(3.0), abs=1e-9)


def test_aalen_johansen_scales_to_many_unique_event_times() -> None:
    """Guard against the former O(n event times * n rows) implementation."""

    n = 100_000
    duration = np.arange(1, n + 1, dtype=float)
    event = np.where(
        np.arange(n) % 3 == 0,
        EVENT_COMPETING,
        EVENT_OF_INTEREST,
    )
    started = time.perf_counter()
    curve = aalen_johansen(duration, event)
    elapsed = time.perf_counter() - started
    assert len(curve.times) == n
    assert curve.n_at_risk[0] == n
    assert curve.n_at_risk[-1] == 1
    assert curve.cause_incidence[-1] + curve.competing_incidence[-1] == (
        pytest.approx(1.0)
    )
    # The vectorized implementation normally takes far below 0.1 seconds on
    # the supported workstation.  A generous bound keeps this stable on CI
    # while the former quadratic path cannot approach it.
    assert elapsed < 3.0


def test_calibration_rejects_a_random_prediction() -> None:
    duration, event, _, _ = _competing_risk_sample(seed=9, n=8000)
    rng = np.random.default_rng(4)
    predicted = np.clip(rng.normal(0.25, 0.08, len(duration)), 0.01, 0.9)
    table = calibration_by_risk_group(predicted, duration, event, horizon=3.0)
    metrics = calibration_metrics(table)
    assert metrics["status"] == "evaluable"
    assert not metrics["calibration_adequate_for_decision_curve"]
    assert metrics["gate_reason"]


def test_decision_curve_gate_blocks_on_miscalibration() -> None:
    duration, event, _, _ = _competing_risk_sample(seed=13, n=6000)
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {
            "risk": np.clip(rng.normal(0.30, 0.05, len(duration)), 0.01, 0.9),
            "duration": duration,
            "event": event,
        }
    )
    report = absolute_risk_report(
        frame,
        predicted_risk_column="risk",
        duration_column="duration",
        event_column="event",
        horizon=3.0,
    )
    assert report["decision_curve_claim_permitted"] is False
    assert report["clinical_use_authorized"] is False
    assert np.isfinite(report["cohort_absolute_risk"])
    assert not report["net_benefit"].empty


def test_net_benefit_treat_all_matches_the_cohort_risk() -> None:
    duration, event, _, _ = _competing_risk_sample(seed=17, n=5000)
    risk = aalen_johansen(duration, event).risk_at(3.0)
    table = net_benefit_from_absolute_risk(
        np.full(len(duration), 0.5),
        duration,
        event,
        horizon=3.0,
        thresholds=(0.2,),
    )
    expected = risk - (1 - risk) * 0.2 / 0.8
    assert table.iloc[0]["net_benefit_treat_all"] == pytest.approx(expected, abs=1e-6)


def test_stratified_absolute_risk_reports_the_competing_bias() -> None:
    duration, event, _, _ = _competing_risk_sample(seed=19, n=6000)
    frame = pd.DataFrame(
        {
            "duration": duration,
            "event": event,
            "stratum": np.where(np.arange(len(duration)) % 2 == 0, "a", "b"),
        }
    )
    table = stratified_absolute_risk(
        frame,
        duration_column="duration",
        event_column="event",
        stratum_column="stratum",
        horizons=(3.0,),
    )
    assert set(table["stratum"]) == {"a", "b"}
    assert (table["competing_risk_bias"] > 0).all()
    assert (table["estimator"] == "aalen_johansen").all()


# --- 3: circadian ---------------------------------------------------------


def test_cosinor_recovers_a_known_daily_rhythm() -> None:
    rng = np.random.default_rng(3)
    n = 6000
    hours = rng.uniform(0, 24, n)
    times = pd.Timestamp("2150-03-01") + pd.to_timedelta(hours, unit="h")
    amplitude, acrophase = 8.0, 5.0
    values = (
        420
        + amplitude * np.cos(2 * np.pi * (hours - acrophase) / 24)
        + rng.normal(0, 6, n)
    )
    result = marker_circadian_variation(values, times, marker_id="qtc")
    assert result["status"] == "evaluable"
    assert result["amplitude"] == pytest.approx(amplitude, abs=0.6)
    assert result["acrophase_hour"] == pytest.approx(acrophase, abs=0.5)
    assert result["mesor"] == pytest.approx(420.0, abs=0.5)


def test_a_flat_marker_shows_no_rhythm() -> None:
    rng = np.random.default_rng(8)
    n = 4000
    hours = rng.uniform(0, 24, n)
    times = pd.Timestamp("2150-03-01") + pd.to_timedelta(hours, unit="h")
    result = marker_circadian_variation(420 + rng.normal(0, 6, n), times)
    assert result["amplitude"] < 1.0
    assert result["variance_explained"] < 0.01


def test_rhythm_is_judged_against_measurement_noise() -> None:
    variation = {"peak_to_trough": 16.0}
    assert exceeds_measurement_noise(variation, 5.0) is True
    assert exceeds_measurement_noise(variation, 25.0) is False
    assert exceeds_measurement_noise(variation, float("nan")) is False


def test_hour_is_encoded_circularly_not_linearly() -> None:
    times = pd.to_datetime(
        ["2150-01-01 23:30", "2150-01-02 00:30", "2150-01-01 11:30"]
    )
    features = circadian_features(times)
    # 23:30 and 00:30 are one hour apart and must be close in the encoding,
    # while midday must be far from both.
    late = features.iloc[0][["acquisition_hour_sin", "acquisition_hour_cos"]]
    early = features.iloc[1][["acquisition_hour_sin", "acquisition_hour_cos"]]
    noon = features.iloc[2][["acquisition_hour_sin", "acquisition_hour_cos"]]
    assert np.hypot(*(late - early).to_numpy()) < 0.3
    assert np.hypot(*(late - noon).to_numpy()) > 1.5
    assert set(CIRCADIAN_COVARIATES) <= set(features.columns)


def test_acquisition_intensity_profile_flags_quiet_hours() -> None:
    # Realistic spread: a daytime peak against a low but non-empty night, so
    # the relative intensity is measured against a full day rather than two
    # isolated hours.
    stamps: list[str] = []
    for hour in range(24):
        volume = 400 if hour == 9 else (20 if hour < 6 else 120)
        stamps.extend([f"2150-01-01 {hour:02d}:00"] * volume)
    times = pd.to_datetime(stamps)
    profile = acquisition_intensity_profile(times)
    by_hour = profile.set_index("hour")
    assert by_hour.loc[9, "relative_intensity"] > 1.0
    assert bool(by_hour.loc[9, "routine_hour"])
    assert not bool(by_hour.loc[4, "routine_hour"])


def test_circadian_table_uses_repeatability_to_judge_rhythms() -> None:
    rng = np.random.default_rng(21)
    n = 4000
    hours = rng.uniform(0, 24, n)
    frame = pd.DataFrame(
        {
            "ecg_time": pd.Timestamp("2150-05-01")
            + pd.to_timedelta(hours, unit="h"),
            "big": 400 + 10 * np.cos(2 * np.pi * hours / 24) + rng.normal(0, 3, n),
            "small": 400 + 0.4 * np.cos(2 * np.pi * hours / 24)
            + rng.normal(0, 3, n),
        }
    )
    repeatability = pd.DataFrame(
        {
            "marker_id": ["big", "small"],
            "smallest_detectable_change": [5.0, 5.0],
        }
    )
    table = circadian_variation_table(
        frame,
        ("big", "small"),
        repeatability=repeatability,
    )
    by_marker = table.set_index("marker_id")
    assert bool(by_marker.loc["big", "rhythm_exceeds_measurement_noise"])
    assert not bool(by_marker.loc["small", "rhythm_exceeds_measurement_noise"])
    summary = summarize_circadian(table)
    assert summary["markers_with_rhythm"] == 1


def test_adjustment_sets_carry_the_circadian_pair() -> None:
    from vcg_biomarker.mimic import adjusted_models

    source = Path(adjusted_models.__file__).read_text(encoding="utf-8")
    assert '"acquisition_hour_sin"' in source
    assert '"acquisition_hour_cos"' in source


# --- 4: dose, route, rate -------------------------------------------------


def test_dose_parsing_separates_redaction_from_zero() -> None:
    assert parse_dose_value("25").value == pytest.approx(25.0)
    redacted = parse_dose_value("___")
    assert redacted.status == DOSE_STATUS_REDACTED
    assert np.isnan(redacted.value)
    ranged = parse_dose_value("0.5-5")
    assert ranged.status == DOSE_STATUS_RANGE_MIDPOINT
    assert ranged.value == pytest.approx(2.75)


def test_unit_normalisation_only_converts_masses() -> None:
    assert to_milligrams(2, "g") == (2000.0, "mg")
    value, scale = to_milligrams(200, "mcg")
    assert value == pytest.approx(0.2)
    assert scale == "mg"
    # Volume and activity are not masses and must not be silently converted.
    assert to_milligrams(10, "mL") == (10.0, "ml")
    assert to_milligrams(5, "units") == (5.0, "units")


def test_dose_frame_marks_redaction_as_missing() -> None:
    frame = pd.DataFrame(
        {
            "dose_given": ["25", "___", "40"],
            "dose_given_unit": ["mg", "mg", "mcg"],
        }
    )
    parsed = parse_dose_frame(
        frame,
        value_column="dose_given",
        unit_column="dose_given_unit",
    )
    assert bool(parsed.loc[1, "dose_redacted"])
    assert np.isnan(parsed.loc[1, "dose_normalised"])
    assert parsed.loc[2, "dose_normalised"] == pytest.approx(0.04)


def test_unknown_dose_is_its_own_level_not_the_low_group() -> None:
    doses = [float(value) for value in range(10, 210, 10)] + [np.nan, np.nan]
    result = dose_response_levels(doses)
    assert result["status"] == "evaluable"
    levels = list(result["levels"])
    assert levels.count("unknown") == 2
    assert result["n_unknown"] == 2
    assert set(levels) == {"low", "high", "unknown"}
    assert result["unknown_kept_separate"] is True


def test_dose_response_trend_recovers_a_known_slope() -> None:
    rng = np.random.default_rng(2)
    n = 800
    dose = rng.uniform(0, 400, n)
    subjects = rng.integers(0, 200, n)
    response = 420 + 0.05 * dose + rng.normal(0, 8, n)
    result = dose_response_trend(dose, response, subject_ids=subjects)
    assert result["status"] == "evaluable"
    assert result["slope_per_unit_dose"] == pytest.approx(0.05, abs=0.01)
    assert result["ci_low"] < 0.05 < result["ci_high"]
    assert "cluster" in result["inference_method"]


def test_exposure_summary_separates_route_and_rate() -> None:
    summary = exposure_summary(
        [10.0, 20.0, np.nan],
        scales=["mg", "mg", "unknown"],
        rates=[1.0, 5.0, np.nan],
        routes=["IV", "PO", "PO"],
    )
    assert summary["cumulative_dose_mg"] == pytest.approx(30.0)
    assert summary["peak_infusion_rate"] == pytest.approx(5.0)
    assert summary["any_intravenous"] is True
    assert summary["any_oral"] is True
    assert summary["dose_known_fraction"] == pytest.approx(2 / 3)


# --- 5: external cohorts --------------------------------------------------


def test_source_roles_separate_independent_from_reused() -> None:
    roles = {source.source_id: source.role for source in EXTERNAL_SOURCES}
    # PTB-XL is this project's development transfer set; reporting it as
    # external validation would present development data as independent.
    assert roles["ptb_xl"] == ROLE_DEVELOPMENT_REUSED
    # INCART is 257 Hz half-hour recordings, not a diagnostic 10-second ECG.
    assert roles["st_petersburg_incart"] == ROLE_DELINEATION_BENCHMARK
    assert roles["georgia"] == ROLE_INDEPENDENT_EXTERNAL
    assert "ptb_xl" not in independent_source_ids()
    assert "st_petersburg_incart" not in independent_source_ids()
    assert len(independent_source_ids()) >= 5


def test_directory_aliases_resolve_to_sources() -> None:
    assert source_by_directory("ptb-xl").source_id == "ptb_xl"
    assert source_by_directory("ningbo").source_id == "ningbo"
    assert source_by_directory("unknown_source") is None


def test_snomed_mapping_reports_unmapped_codes() -> None:
    mapped, unmapped = map_snomed_codes(["164889003", "999999999", "55827005"])
    assert "atrial_fibrillation" in mapped
    assert "left_ventricular_hypertrophy" in mapped
    assert unmapped == ("999999999",)
    assert len(SNOMED_LABELS) >= 30


def test_eligibility_excludes_reused_and_incomparable_records() -> None:
    inventory = pd.DataFrame(
        {
            "source_id": ["georgia", "ptb_xl", "st_petersburg_incart", "ningbo"],
            "independent_external": [True, False, False, True],
            "n_leads": [12, 12, 12, 12],
            "duration_seconds": [10.0, 10.0, 1800.0, 10.0],
            "n_labels": [2, 3, 1, 0],
        }
    )
    scoped = eligible_for_diagnostic_validation(inventory)
    assert scoped["eligible_for_diagnostic_validation"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert scoped["ineligibility_reason"].tolist()[1] == "source_outside_external_validation_role"
    assert scoped["ineligibility_reason"].tolist()[2] == "source_outside_external_validation_role"
    assert scoped["ineligibility_reason"].tolist()[3] == "no_mapped_label"


def test_claim_scope_refuses_accuracy_against_an_independent_standard() -> None:
    inventory = pd.DataFrame(
        {
            "source_id": ["georgia"],
            "source_country": ["USA"],
            "independent_external": [True],
            "n_leads": [12],
            "duration_seconds": [10.0],
            "n_labels": [1],
        }
    )
    scope = external_claim_scope(inventory)
    assert scope["supports_external_analytical_validation"] is True
    assert scope["supports_external_agreement_with_expert_reading"] is True
    # The labels are another reader's interpretation of the same recording.
    assert scope["supports_external_accuracy_against_echo_or_laboratory"] is False
    assert "ptb_xl" in scope["development_reused_sources_excluded"]
    assert "st_petersburg_incart" in scope["delineation_only_sources"]


def test_source_registry_is_reviewable() -> None:
    table = source_registry_table()
    assert len(table) == len(EXTERNAL_SOURCES)
    assert table["country"].nunique() >= 4
    assert table["independent"].sum() >= 5


def test_distribution_shift_flags_untransferable_operating_points() -> None:
    rng = np.random.default_rng(6)
    features = pd.DataFrame(
        {
            "source_id": ["a"] * 200 + ["b"] * 200,
            "extraction_status": ["ok"] * 400,
            "marker": np.concatenate(
                [rng.normal(65, 40, 200), rng.normal(140, 40, 200)]
            ),
        }
    )
    shift = distribution_shift(features, {"marker": (65.0, 40.0)}, ("marker",))
    by_source = shift.set_index("source_id")
    assert abs(by_source.loc["a", "standardised_shift"]) < MAXIMUM_TRANSFERABLE_SHIFT
    assert bool(by_source.loc["a", "distribution_shift_within_screening_tolerance"])
    assert not bool(by_source.loc["a", "operating_point_transferable"])
    assert by_source.loc["b", "standardised_shift"] > MAXIMUM_TRANSFERABLE_SHIFT
    assert not bool(by_source.loc["b", "operating_point_transferable"])


def test_site_consistency_detects_a_pooled_mixture() -> None:
    agreement = pd.DataFrame(
        {
            "source_id": ["a", "b", "c", "__pooled__"],
            "status": ["evaluable"] * 4,
            "aligned_auc": [0.50, 0.78, 0.56, 0.58],
        }
    )
    result = site_consistency(agreement)
    assert result["sites_evaluable"] == 3
    assert result["auc_range"] == pytest.approx(0.28)
    assert result["consistent_across_sites"] is False


def test_extraction_coverage_reports_abstention() -> None:
    features = pd.DataFrame(
        {
            "source_id": ["a"] * 10,
            "extraction_status": ["ok"] * 8 + ["failed"] * 2,
            "marker": [1.0] * 4 + [np.nan] * 4 + [np.nan] * 2,
        }
    )
    coverage = extraction_coverage(features, ("marker",)).iloc[0]
    assert coverage["extraction_rate"] == pytest.approx(0.8)
    assert coverage["coverage"] == pytest.approx(0.4)
    assert coverage["abstention_rate"] == pytest.approx(0.6)
    assert coverage["post_qc_marker_missingness"] == pytest.approx(0.5)


def test_external_mains_frequency_is_prespecified_per_source() -> None:
    assert source_powerline_hz("georgia") == 60.0
    assert source_powerline_hz("cpsc_2018") == 50.0
    assert source_powerline_hz("ptb") == 50.0
    with pytest.raises(ValueError, match="No prespecified"):
        source_powerline_hz("unknown_site")


def test_external_contract_loads_only_hash_bound_shortlisted_models(tmp_path) -> None:
    model = MarkerModel(
        feature_id="marker",
        direction=-1,
        mean=2.0,
        standard_deviation=0.5,
        thresholds={"specificity_95": -1.5},
        n_train=100,
        n_positive_train=40,
        n_negative_train=60,
    )
    model_row = {
        "endpoint_id": "qtc_gt_500",
        "panel_id": "primary_core",
        **model.to_dict(),
    }
    models_path = tmp_path / "trained_marker_models.json"
    models_path.write_text(json.dumps({"models": [model_row]}), encoding="utf-8")
    entry = {
        "endpoint_id": "qtc_gt_500",
        "panel_id": "primary_core",
        "feature_id": "marker",
        "criterion_id": "criterion-1",
        "marker_model_hash": content_hash(model_row),
    }
    shortlist_core = {"shortlist_version": "1.1.0", "entries": [entry]}
    shortlist = {
        **shortlist_core,
        "development_run_id": "dev-1",
        "locked_test_read": False,
        "shortlist_hash": content_hash(shortlist_core),
    }
    shortlist_path = tmp_path / "locked_test_marker_shortlist.json"
    shortlist_path.write_text(json.dumps(shortlist), encoding="utf-8")
    payload = {
        "tracked_file_sha256": {
            "trained_marker_models": file_sha256(models_path),
            "locked_test_marker_shortlist": file_sha256(shortlist_path),
        }
    }
    freeze_hash = content_hash(payload)
    (tmp_path / "freeze_manifest.json").write_text(
        json.dumps(
            {"run_id": "dev-1", "freeze_hash": freeze_hash, "payload": payload}
        ),
        encoding="utf-8",
    )
    (tmp_path / "test_lock.json").write_text(
        json.dumps(
            {"run_id": "dev-1", "freeze_hash": freeze_hash, "invalidated_at": None}
        ),
        encoding="utf-8",
    )
    contract = load_frozen_external_contract(tmp_path)
    assert contract.freeze_hash == freeze_hash
    assert contract.marker_ids == ("marker",)
    assert contract.markers[0].model.direction == -1
    assert contract.markers[0].model.thresholds["specificity_95"] == -1.5

    # A changed direction cannot be smuggled into the external run even if
    # the JSON is otherwise well formed.
    model_row["direction"] = 1
    models_path.write_text(json.dumps({"models": [model_row]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        load_frozen_external_contract(tmp_path)


def test_incart_signal_only_copy_is_reported_as_incomplete(tmp_path) -> None:
    # The Challenge collection ships INCART without the beat annotations this
    # benchmark scores against. "Present but unusable" and "absent" need
    # different fixes, so they must not read the same.
    collection = tmp_path / "challenge"
    incart = collection / "st_petersburg_incart" / "g1"
    incart.mkdir(parents=True)
    (incart / "I0001.hea").write_text("I0001 12 257 462600\n", encoding="utf-8")
    diagnostics = benchmark_root_diagnostics(
        tmp_path / "empty",
        "incart",
        additional_roots=[collection],
    )
    assert diagnostics["status"] == "present_but_incomplete"
    assert diagnostics["candidates"][0]["signal_files_present"] is True
    assert "I01.atr" in diagnostics["candidates"][0]["missing_markers"]
    assert resolve_benchmark_root(
        tmp_path / "empty",
        "incart",
        additional_roots=[collection],
    ) is None


def test_external_stage_requires_a_configured_collection(tmp_path) -> None:
    from vcg_biomarker.mimic.config import MimicProjectConfig
    from vcg_biomarker.mimic.services import run_service_stage

    config = MimicProjectConfig(
        mimic_iv_root=tmp_path,
        mimic_ed_root=tmp_path,
        mimic_ecg_root=tmp_path,
        echo_structured_path=tmp_path / "a.csv",
        echo_study_list_path=tmp_path / "b.csv",
        echo_record_list_path=tmp_path / "c.csv",
        work_root=tmp_path / "work",
        external_ecg_root=None,
    )
    with pytest.raises(RuntimeError, match="external_ecg_root"):
        run_service_stage(config, "external-validation", {})


def test_chartevents_ingest_keeps_every_numeric_icu_item(tmp_path) -> None:
    """The ICU chart filter used to discard 4,078 of 4,095 item types.

    Seventeen hand-picked itemids were kept -- vitals, ventilator mode, GCS --
    and everything else the ICU charts was dropped at ingest. Because the
    filter sits in the ingest, widening it later means reading the source
    again and rebuilding every downstream artifact, so the choice belongs
    before a long run rather than after one.
    """

    import gzip

    from vcg_biomarker.mimic.event_ingest import (
        CHARTEVENTS_CURATED_ITEMIDS,
        chartevents_itemid_filter_sql,
        numeric_chartevent_itemids,
    )

    icu = tmp_path / "mimic" / "icu"
    icu.mkdir(parents=True)
    rows = [
        "itemid,label,linksto,param_type,category",
        "220045,Heart Rate,chartevents,Numeric,Routine Vital Signs",
        "220074,Central Venous Pressure,chartevents,Numeric,Hemodynamics",
        "224690,Respiratory Rate (Total),chartevents,Numeric,Respiratory",
        "223849,Ventilator Mode,chartevents,Text,Respiratory",
        "225183,Dialysate Rate,chartevents,Numeric,Dialysis",
        "226260,Not a chart item,inputevents,Numeric,Medications",
    ]
    with gzip.open(icu / "d_items.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")

    class _Config:
        mimic_iv_root = tmp_path / "mimic"

    resolved = numeric_chartevent_itemids(_Config())
    # Every numeric chart item, not a hand-maintained subset.
    assert {220045, 220074, 224690, 225183} <= set(resolved)
    # Items belonging to another table are not chart items.
    assert 226260 not in resolved
    # Categorical items the numeric rule would drop are kept explicitly.
    assert set(CHARTEVENTS_CURATED_ITEMIDS) <= set(resolved)

    filter_sql = chartevents_itemid_filter_sql(_Config())
    assert "220074" in filter_sql
    assert filter_sql.count(",") == len(resolved) - 1


def test_an_unreadable_item_catalog_degrades_to_the_curated_list(tmp_path) -> None:
    from vcg_biomarker.mimic.event_ingest import (
        CHARTEVENTS_CURATED_ITEMIDS,
        numeric_chartevent_itemids,
    )

    class _Missing:
        mimic_iv_root = tmp_path / "absent"

    # An incomplete download must degrade to the previous behaviour rather
    # than to an empty filter, which would silently ingest nothing.
    assert numeric_chartevent_itemids(_Missing()) == tuple(
        CHARTEVENTS_CURATED_ITEMIDS
    )


def test_config_payload_accepts_anything_config_shaped() -> None:
    """Regression: the evidence attestation could never succeed.

    ``scientific_config_payload`` dispatched on ``isinstance(MimicProjectConfig)``
    and fell through to ``dict(config)`` otherwise. The attestation runs the
    whole evidence build against a shadow root through a wrapper that carries
    ``to_dict`` but is not that class, so it raised ``TypeError`` before
    comparing a single row -- and the caller reported that as "exact
    event-layer equivalence could not be proven", which reads as a finding
    about the data instead of a type error.
    """

    from vcg_biomarker.mimic.config import (
        operational_config_payload,
        scientific_config_payload,
    )

    class _Wrapper:
        """Duck-typed like the shadow and attestation config wrappers."""

        def to_dict(self) -> dict:
            return {"random_seed": 7, "workers": 3, "mimic_iv_root": "/data"}

    scientific = scientific_config_payload(_Wrapper())
    assert scientific["random_seed"] == 7
    # Resource knobs must stay out of the scientific hash.
    assert "workers" not in scientific
    assert operational_config_payload(_Wrapper())["workers"] == 3

    # Plain mappings keep working.
    assert scientific_config_payload({"random_seed": 2})["random_seed"] == 2


def test_the_equivalence_rebuild_gets_the_projects_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proof that cannot finish is not a proof.

    The shadow rebuild executes the same queries as the real one, so capping
    it below the project budget does not make it cheaper -- it makes it run
    out of memory inside the laboratory window query and report a data
    finding instead of a resource limit.
    """

    from vcg_biomarker.mimic.evidence import _AttestationResourceConfig

    class _Base:
        duckdb_memory_limit_gb = 16
        workers = 6

        def to_dict(self) -> dict:
            return {}

    monkeypatch.setenv("VCG_JOB_MEMORY_GB", "10")
    scoped = _AttestationResourceConfig(_Base())
    assert scoped.duckdb_memory_limit_gb == 10
    # Still single-threaded and low-memory: a correctness check must not
    # crowd out the run it is checking.
    assert scoped.workers == 1
    assert scoped.evidence_low_memory_mode is True

    class _Tiny:
        duckdb_memory_limit_gb = 1
        workers = 1

        def to_dict(self) -> dict:
            return {}

    # A tiny project budget still wins over the machine probe: the floor of
    # two gigabytes is there so the check can start at all.
    assert _AttestationResourceConfig(_Tiny()).duckdb_memory_limit_gb == 2


def test_icu_vitals_only_scan_the_items_its_aggregates_use() -> None:
    """The widened chart ingest made an old inefficiency load-bearing.

    Every aggregate and the HAVING filter to one of three item groups, so
    rows outside them cannot influence any output value -- they were only
    carried through the join. With seventeen ingested items that was
    invisible; with 779 the intermediate result grew 1.9x and ran the shadow
    build out of memory while producing identical numbers.
    """

    import re
    from pathlib import Path

    source = Path("vcg_biomarker/mimic/evidence.py").read_text(encoding="utf-8")
    for match in re.finditer(r"\{chartevents_relation\}", source):
        block = source[max(0, match.start() - 1400) : match.start() + 1600]
        assert "c.itemid IN (" in block, (
            "every chartevents scan must restrict to the items it uses"
        )
