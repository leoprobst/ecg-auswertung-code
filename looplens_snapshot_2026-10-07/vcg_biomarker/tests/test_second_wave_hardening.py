"""Regression tests for the second hardening wave (external review findings).

Covers the seven blockers plus the new analysis layers. Each test pins one
behaviour that was either wrong or missing, so a later refactor cannot quietly
undo it.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.core.delineation.benchmark_suite import (
    DELINEATION_RELEASE_DATASETS,
    benchmark_staleness,
    delineation_implementation_hash,
)
from vcg_biomarker.core.preprocessing import prepare_signal_paths
from vcg_biomarker.mimic.note_context import (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_END_OF_STAY,
    EXPERIENCER_OTHER,
    EXPERIENCER_PATIENT,
    TEMPORALITY_CURRENT,
    TEMPORALITY_HISTORICAL,
    TEMPORALITY_HYPOTHETICAL,
    annotate_note_usage,
    assert_no_predictor_leakage,
    draw_note_validation_sample,
    note_availability_state,
    note_pattern_validation_metrics,
    resolve_mention_context,
)
from vcg_biomarker.mimic.mappings import LAB_ITEM_GROUPS
from vcg_biomarker.mimic.reference_tiers import (
    TIER_A,
    TIER_C,
    TIER_UNKNOWN,
    adjudication_agreement,
    classify_reference_tier,
    draw_adjudication_sample,
    reference_tier_table,
    source_concordance,
)
from vcg_biomarker.mimic.repeatability import (
    marker_repeatability,
    change_exceeds_measurement_error,
    intraclass_correlation_absolute_agreement,
    repeatability_table,
    summarize_repeatability,
    within_subject_variance,
)
from vcg_biomarker.mimic.selection_bias import (
    DEFAULT_CONTROL_ASSOCIATIONS,
    coverage_census,
    evaluate_control_associations,
    holdout_strata,
    observation_propensity_weights,
    summarize_control_associations,
    verification_bias_bounds,
)
from vcg_biomarker.mimic.signal_sources import (
    SOURCE_INDIVIDUAL_BEATS,
    SOURCE_MEDIAN_BEAT,
    SOURCE_WIDEBAND,
    WIDEBAND_REQUIRED_FEATURE_IDS,
    assert_signal_source_policy_consistent,
    signal_source_for_feature,
    signal_source_table,
)
from vcg_biomarker.mimic.target_trial import (
    CLAIM_ASSOCIATION,
    CLAIM_EMULATED_EFFECT,
    DEFAULT_PROTOCOLS,
    TargetTrialProtocol,
    assess_protocol_compliance,
    dose_exposure_levels,
    protocol_registry_table,
)
from vcg_biomarker.mimic.trajectory import (
    fit_trajectory_association,
    observation_intensity_diagnostics,
    observation_intensity_weights,
    summarize_trajectories,
)


# --- ICU event path -------------------------------------------------------


def test_icu_relations_resolve_through_the_event_registry() -> None:
    from vcg_biomarker.mimic import evidence

    source = Path(evidence.__file__).read_text(encoding="utf-8")
    # The flat-file lookup is what silently emptied the ICU layer while 62.9 M
    # chartevents rows sat in the partitioned registry.
    assert 'standardized_root / "chartevents.parquet"' not in source
    assert "_icu_event_relations" in source
    assert "icu_evidence_coverage" in source


def test_icu_coverage_reports_absence_rather_than_implying_none(
    tmp_path,
) -> None:
    from vcg_biomarker.mimic.evidence import icu_evidence_coverage

    class _Config:
        standardized_root = tmp_path
        mimic_iv_root = tmp_path / "mimic"

    coverage = icu_evidence_coverage(_Config())
    assert coverage["status"] in ("no_icu_stays", "not_available")
    assert coverage["assessed"] is False
    assert coverage["zero_rows_are_not_evidence_of_assessed_absence"] is True


# --- signal sources and units --------------------------------------------


def test_wideband_path_is_the_supported_150_hz_diagnostic_source() -> None:
    fs = 500
    samples = fs * 10
    time = np.arange(samples) / fs
    # The ADC Nyquist is not the analogue recorder bandwidth.  MIMIC's
    # documented per-record front-end corner is 150 Hz, so the HF source must
    # not manufacture a nominal 150--225-Hz band from the same samples.
    signal = np.column_stack(
        [np.sin(2 * np.pi * 190.0 * time) for _ in range(12)]
    )
    paths = prepare_signal_paths(signal, fs, powerline=60.0)
    assert "wideband" in paths
    assert paths["wideband"] is paths["diagnostic"]
    assert paths["wideband"] is paths["morphology"]


def test_high_frequency_marker_requires_the_wideband_source() -> None:
    assert (
        "qrs_fragmentation_high_frequency_energy"
        in WIDEBAND_REQUIRED_FEATURE_IDS
    )
    assert (
        signal_source_for_feature("qrs_fragmentation_high_frequency_energy")
        == SOURCE_WIDEBAND
    )


def test_high_frequency_marker_abstains_without_a_wideband_source() -> None:
    from vcg_biomarker.core.biomarkers import catalog_biomarkers

    source = Path(catalog_biomarkers.__file__).read_text(encoding="utf-8")
    assert "vcg_segment_wideband" in source
    assert "no wideband signal source" in source


def test_microvolt_marker_is_scaled_to_microvolts() -> None:
    from vcg_biomarker.core.biomarkers import catalog_biomarkers

    assert catalog_biomarkers.MILLIVOLT_TO_MICROVOLT == 1000.0
    source = Path(catalog_biomarkers.__file__).read_text(encoding="utf-8")
    marker = source.index('if name == "microvolt_t_wave_alternans_vector"')
    body = source[marker : marker + 3000]
    assert "MILLIVOLT_TO_MICROVOLT" in body
    # Squared amplitude needs the factor squared, not the same factor.
    hf = source.index('if name == "qrs_fragmentation_high_frequency_energy"')
    hf_body = source[hf : hf + 4000]
    assert "MILLIVOLT_TO_MICROVOLT ** 2" in hf_body


def test_beat_variability_markers_declare_individual_beats() -> None:
    for feature_id in (
        "t_wave_alternans_vector",
        "qrs_axis_beat_to_beat_variability_deg",
        "rr_poincare_sd1_ms",
        "p_loop_template_instability",
    ):
        assert signal_source_for_feature(feature_id) == SOURCE_INDIVIDUAL_BEATS


def test_signal_source_defaults_to_the_median_beat() -> None:
    assert signal_source_for_feature("qrs_loop_area") == SOURCE_MEDIAN_BEAT
    table = signal_source_table(["qrs_loop_area"])
    assert table[0]["signal_source"] == SOURCE_MEDIAN_BEAT
    assert "Medianschlag" in table[0]["signal_source_caveat"]


def test_signal_source_policy_matches_the_implementation() -> None:
    assert_signal_source_policy_consistent(
        {
            "qrs_fragmentation_high_frequency_energy": SOURCE_WIDEBAND,
            "t_wave_alternans_vector": SOURCE_INDIVIDUAL_BEATS,
            "qrs_loop_area": SOURCE_MEDIAN_BEAT,
        }
    )
    with pytest.raises(AssertionError):
        assert_signal_source_policy_consistent(
            {"qrs_fragmentation_high_frequency_energy": SOURCE_MEDIAN_BEAT}
        )


# --- feature metadata -----------------------------------------------------


def test_feature_metadata_columns_are_persisted() -> None:
    from vcg_biomarker.mimic import feature_record

    source = Path(feature_record.__file__).read_text(encoding="utf-8")
    assert "_store_feature_metadata" in source
    for column in (
        "feature_confidence_json",
        "feature_per_lead_json",
        "feature_unevaluable_reason_json",
        "feature_unevaluable_count",
    ):
        assert column in source


def test_unevaluable_reasons_are_captured_not_discarded() -> None:
    from vcg_biomarker.mimic.feature_record import _store_biomarker_results

    class _Result:
        def __init__(self, name, value, notes="", confidence=1.0, per_lead=None):
            self.name = name
            self.value = value
            self.computation_notes = notes
            self.confidence = confidence
            self.per_lead = per_lead

    destination: dict = {}
    failures: list = []
    metadata: dict = {}
    _store_biomarker_results(
        destination,
        [
            _Result("good", 1.0, notes="measured successfully", confidence=0.9, per_lead={"I": 1.0}),
            _Result("abstained", None, notes="not_evaluable: QRS too short"),
            _Result("broken", None, notes="error: boom"),
        ],
        failures,
        metadata=metadata,
    )
    assert destination["good"] == 1.0
    assert metadata["confidence"]["good"] == 0.9
    assert metadata["per_lead"]["good"] == {"I": 1.0}
    assert "QRS too short" in metadata["unevaluable_reason"]["abstained"]
    assert failures and failures[0]["feature_id"] == "broken"
    # An error is a failure, not an abstention reason.
    assert "broken" not in metadata["unevaluable_reason"]
    assert "good" not in metadata["unevaluable_reason"]
    assert metadata["computation_notes"]["good"] == "measured successfully"


# --- delineation benchmark ------------------------------------------------


def test_release_core_is_the_only_blocking_dataset_set() -> None:
    assert set(DELINEATION_RELEASE_DATASETS) == {"ludb", "qtdb", "nstdb"}


def test_benchmark_staleness_detects_a_changed_implementation() -> None:
    current = delineation_implementation_hash()
    assert benchmark_staleness({"implementation_hash": current})[
        "benchmark_stale"
    ] is False
    stale = benchmark_staleness({"implementation_hash": "0" * 64})
    assert stale["benchmark_stale"] is True
    assert stale["status"] == "stale_recompute_required"
    stale_policy = benchmark_staleness(
        {
            "implementation_hash": current,
            "release_policy_version": "obsolete-policy",
        }
    )
    assert stale_policy["benchmark_stale"] is True
    assert stale_policy["release_policy_version_changed"] is True
    unknown = benchmark_staleness({})
    assert unknown["status"] == "unknown_no_recorded_hash"


def test_guided_run_only_requires_the_release_core() -> None:
    from vcg_biomarker.mimic.config import MimicProjectConfig
    from vcg_biomarker.mimic.workflow import _stage_arguments

    config = MimicProjectConfig(
        mimic_iv_root="x",
        mimic_ed_root="x",
        mimic_ecg_root="x",
        echo_structured_path="x",
        echo_study_list_path="x",
        echo_record_list_path="x",
        work_root="x",
    )
    args = _stage_arguments(
        config,
        "validate-delineation",
        preset_id="exhaustive_research",
        run_id="run",
    )
    assert args["strict"] is True
    assert args["max_records"] is None
    assert set(args["required_dataset_ids"]) == set(DELINEATION_RELEASE_DATASETS)
    assert "incart" in args["dataset_ids"]
    assert "incart" not in args["required_dataset_ids"]


# --- repeatability --------------------------------------------------------


def test_repeatability_recovers_a_known_measurement_error() -> None:
    rng = np.random.default_rng(3)
    truth = rng.normal(50.0, 10.0, 800)
    first = truth + rng.normal(0.0, 2.0, 800)
    second = truth + rng.normal(0.0, 2.0, 800)
    result = marker_repeatability(first, second, marker_id="demo")
    assert result["status"] == "evaluable"
    assert result["within_subject_sd"] == pytest.approx(2.0, abs=0.25)
    assert result["icc"] == pytest.approx(0.96, abs=0.05)
    assert result["repeatability_coefficient"] == pytest.approx(
        2.77 * result["within_subject_sd"],
        rel=1e-3,
    )
    assert result["reliable_for_individual_change"] is True


def test_a_noisy_marker_is_flagged_as_unusable_for_individual_change() -> None:
    rng = np.random.default_rng(4)
    truth = rng.normal(50.0, 10.0, 400)
    first = truth + rng.normal(0.0, 12.0, 400)
    second = truth + rng.normal(0.0, 12.0, 400)
    result = marker_repeatability(first, second, marker_id="noisy")
    assert result["icc"] < 0.75
    assert result["reliable_for_individual_change"] is False
    assert result["reliability_gate_reason"] == "icc_below_threshold"


def test_within_subject_variance_halves_the_squared_difference() -> None:
    first = np.array([10.0, 12.0, 14.0])
    second = np.array([12.0, 10.0, 16.0])
    assert within_subject_variance(first, second) == pytest.approx(2.0)


def test_perfect_agreement_gives_unit_icc() -> None:
    values = np.arange(50, dtype=float)
    result = intraclass_correlation_absolute_agreement(values, values.copy())
    assert result["icc"] == pytest.approx(1.0)


def test_smallest_detectable_change_gates_individual_deltas() -> None:
    flags = change_exceeds_measurement_error([1.0, 5.0, -9.0], 5.5)
    assert flags.tolist() == [False, False, True]
    assert change_exceeds_measurement_error([100.0], float("nan")).tolist() == [
        False
    ]


def test_repeatability_table_and_summary_agree() -> None:
    rng = np.random.default_rng(5)
    truth = rng.normal(0.0, 5.0, 300)
    pairs = pd.DataFrame(
        {
            "a_first": truth + rng.normal(0.0, 0.5, 300),
            "a_second": truth + rng.normal(0.0, 0.5, 300),
            "b_first": truth + rng.normal(0.0, 9.0, 300),
            "b_second": truth + rng.normal(0.0, 9.0, 300),
        }
    )
    table = repeatability_table(pairs, ("a", "b"))
    assert len(table) == 2
    summary = summarize_repeatability(table)
    assert summary["markers_assessed"] == 2
    assert summary["markers_reliable"] == 1
    assert summary["markers_unreliable"] == ["b"]


# --- trajectories ---------------------------------------------------------


def _trajectory_series(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subject_id in range(1, 161):
        slope = 2.0 if subject_id % 2 == 0 else -1.0
        start = pd.Timestamp("2150-01-01")
        for step in range(int(rng.integers(3, 9))):
            rows.append(
                {
                    "subject_id": subject_id,
                    "split": "train",
                    "ecg_time": start + pd.Timedelta(days=2 * step),
                    "marker": 50.0 + slope * 2 * step + rng.normal(0.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def test_trajectory_slopes_recover_the_true_rate() -> None:
    table = summarize_trajectories(_trajectory_series(), marker_ids=("marker",))
    assert (table["slope_status"] == "fitted").all()
    even = table.loc[table["subject_id"] % 2 == 0, "slope_per_day"].mean()
    odd = table.loc[table["subject_id"] % 2 == 1, "slope_per_day"].mean()
    assert even == pytest.approx(2.0, abs=0.2)
    assert odd == pytest.approx(-1.0, abs=0.2)


def test_trajectory_gee_recovers_a_known_odds_ratio() -> None:
    rng = np.random.default_rng(21)
    table = summarize_trajectories(_trajectory_series(), marker_ids=("marker",))
    probability = 1.0 / (
        1.0 + np.exp(-0.6 * table["slope_per_day"].to_numpy())
    )
    table["outcome"] = (rng.random(len(table)) < probability).astype(int)
    fit = fit_trajectory_association(
        table,
        marker_column="slope_per_day",
        outcome_column="outcome",
        baseline_column="baseline_value",
    )
    assert fit["status"] == "evaluable"
    assert fit["odds_ratio"] == pytest.approx(np.exp(0.6), rel=0.35)
    assert fit["baseline_adjusted"] is True
    assert "subject_clusters" in fit["inference_method"]


def test_trajectory_gee_refuses_a_separated_fit() -> None:
    table = summarize_trajectories(_trajectory_series(), marker_ids=("marker",))
    table["outcome"] = (table["subject_id"] % 2 == 0).astype(int)
    fit = fit_trajectory_association(
        table,
        marker_column="slope_per_day",
        outcome_column="outcome",
    )
    assert fit["status"] == "not_evaluable"
    assert fit["separation_suspected"] is True


def test_observation_intensity_weights_downweight_frequent_recording() -> None:
    counts = np.array([1, 1, 2, 4, 20], dtype=float)
    result = observation_intensity_weights(counts)
    assert result["status"] == "evaluable"
    weights = result["weights"]
    # More recordings must mean less weight per subject.
    assert weights[0] > weights[3] > weights[4]
    assert result["effective_sample_size"] < len(counts)


def test_informative_observation_is_detected() -> None:
    counts = np.concatenate([np.full(100, 2.0), np.full(100, 12.0)])
    outcome = np.concatenate([np.zeros(100), np.ones(100)])
    diagnostics = observation_intensity_diagnostics(counts, outcome)
    assert diagnostics["status"] == "evaluable"
    assert diagnostics["count_outcome_auc"] > 0.9
    assert diagnostics["informative_observation_suspected"] is True


# --- reference tiers ------------------------------------------------------


def test_echo_and_lab_anchored_endpoints_are_tier_a() -> None:
    assert classify_reference_tier(("structured-measurement",), evidence_available=True, temporally_eligible=True) == TIER_A
    assert classify_reference_tier(("labevents", "d_labitems"), evidence_available=True, temporally_eligible=True) == TIER_A
    assert classify_reference_tier(("diagnoses_icd",)) == TIER_C


def test_same_recording_device_source_is_never_a_reference() -> None:
    assert classify_reference_tier(("machine_measurements",)) == TIER_UNKNOWN


def test_registered_endpoints_are_tiered_and_qtc_is_excluded() -> None:
    from vcg_biomarker.mimic.label_registry import default_label_definitions

    table = reference_tier_table(default_label_definitions())
    by_label = table.set_index("label_id")
    assert by_label.loc["qtc_gt_480", "reference_tier"] == TIER_UNKNOWN
    assert not bool(by_label.loc["qtc_gt_480", "accuracy_claim_supported"])
    # A registry declares sources, not that a patient was actually measured.
    assert by_label.loc["echo_lvh_strict", "reference_tier"] == TIER_UNKNOWN
    assert not bool(by_label.loc["echo_lvh_strict", "accuracy_claim_supported"])
    assert table["reference_tier"].isin(
        [TIER_A, "B_multi_source_concordant", TIER_C, TIER_UNKNOWN]
    ).all()


def test_source_concordance_locates_disagreement() -> None:
    frame = pd.DataFrame(
        {
            "icd": [1, 1, 0, np.nan],
            "lab": [1, 0, 0, np.nan],
        }
    )
    result = source_concordance(frame, ["icd", "lab"])
    assert result["concordance_state"].tolist() == [
        "concordant_positive",
        "discordant",
        "concordant_negative",
        "unobserved",
    ]


def test_adjudication_sample_is_stratified_blinded_and_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": range(1, 401),
            "study_id": range(1001, 1401),
            "label_state": ["positive"] * 120
            + ["negative"] * 200
            + ["indeterminate"] * 80,
            "conc": ["concordant_positive"] * 100
            + ["discordant"] * 40
            + ["concordant_negative"] * 180
            + ["unobserved"] * 80,
        }
    )
    sample = draw_adjudication_sample(
        frame,
        endpoint_id="demo",
        concordance_column="conc",
    )
    assert set(sample["review_stratum"]) == {
        "positive",
        "negative",
        "uncertain",
        "discordant",
    }
    # The reviewer must not see what the pipeline decided.
    assert "label_state" not in sample.columns
    assert not bool(sample["blinded"].any())
    assert sample["reviewer_slot"].nunique() == 2
    again = draw_adjudication_sample(
        frame,
        endpoint_id="demo",
        concordance_column="conc",
    )
    assert sample["review_case_id"].tolist() == again["review_case_id"].tolist()


def test_adjudication_agreement_computes_kappa() -> None:
    reviews = pd.DataFrame(
        {
            "review_case_id": ["a", "a", "b", "b", "c", "c"],
            "reviewer_slot": [1, 2, 1, 2, 1, 2],
            "reviewer_verdict": [
                "positive",
                "positive",
                "negative",
                "negative",
                "positive",
                "negative",
            ],
        }
    )
    result = adjudication_agreement(reviews)
    assert result["status"] == "evaluable"
    assert result["n_cases"] == 3
    assert result["raw_agreement"] == pytest.approx(2 / 3)


# --- notes ----------------------------------------------------------------


def test_family_history_is_not_attributed_to_the_patient() -> None:
    text = "Family history is notable for atrial fibrillation in the father."
    context = resolve_mention_context(text, text.index("atrial"))
    assert context.experiencer == EXPERIENCER_OTHER
    assert context.counts_for_patient is False


def test_historical_and_hypothetical_mentions_are_not_incident() -> None:
    historical = "History of atrial fibrillation, on anticoagulation."
    context = resolve_mention_context(historical, historical.index("atrial"))
    assert context.temporality == TEMPORALITY_HISTORICAL
    assert context.is_incident is False

    hypothetical = "Return to the ED if atrial fibrillation recurs."
    context = resolve_mention_context(hypothetical, hypothetical.index("atrial"))
    assert context.temporality == TEMPORALITY_HYPOTHETICAL
    assert context.is_incident is False

    current = "Patient developed atrial fibrillation on hospital day 4."
    context = resolve_mention_context(current, current.index("atrial"))
    assert context.experiencer == EXPERIENCER_PATIENT
    assert context.temporality == TEMPORALITY_CURRENT
    assert context.is_incident is True


def test_section_header_overrides_local_phrasing() -> None:
    context = resolve_mention_context(
        "Coronary artery disease.",
        0,
        section_name="Family History",
    )
    assert context.experiencer == EXPERIENCER_OTHER
    assert context.experiencer_cue == "section:family_history"


def test_discharge_summaries_are_never_ecg_time_predictors() -> None:
    states = note_availability_state(
        ["discharge", "radiology", "radiology", "radiology"],
        pd.to_datetime(["2150-01-01", "2150-01-01", "2150-01-03", None]),
        pd.to_datetime(["2150-01-02"] * 4),
    )
    # Even with a timestamp before the ECG, a discharge summary summarises the
    # whole stay and cannot be a baseline predictor.
    assert states.tolist() == [
        AVAILABILITY_END_OF_STAY,
        AVAILABILITY_AVAILABLE,
        "written_after_ecg",
        "timestamp_unknown",
    ]


def test_predictor_leakage_is_refused() -> None:
    frame = pd.DataFrame(
        {
            "note_kind": ["discharge"],
            "note_time": pd.to_datetime(["2150-01-01"]),
            "ecg_time": pd.to_datetime(["2150-01-02"]),
        }
    )
    annotated = annotate_note_usage(frame, role="predictor")
    with pytest.raises(AssertionError, match="leakage|available"):
        assert_no_predictor_leakage(annotated)
    # The same note as an outcome source is legitimate.
    as_outcome = annotate_note_usage(frame, role="outcome_or_context")
    assert_no_predictor_leakage(as_outcome)


def test_note_validation_sample_and_metrics() -> None:
    mentions = pd.DataFrame(
        {
            "label_id": ["af"] * 40 + ["hf"] * 40,
            "state": (["positive"] * 20 + ["negative"] * 20) * 2,
            "matched_clause": ["clause"] * 80,
        }
    )
    sample = draw_note_validation_sample(mentions, per_label=5)
    assert len(sample) == 20
    assert set(sample["annotation_state"]) == {""}
    sample.loc[sample["state"] == "positive", "annotation_state"] = "positive"
    sample.loc[sample["state"] == "negative", "annotation_state"] = "negative"
    metrics = note_pattern_validation_metrics(sample)
    assert metrics["status"] == "evaluable"
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


# --- laboratory coverage --------------------------------------------------


def test_second_wave_lab_groups_are_registered_and_screened() -> None:
    from vcg_biomarker.mimic.association_domains import (
        default_context_association_definitions,
    )

    for group in (
        "sodium",
        "bicarbonate",
        "phosphate",
        "glucose",
        "hba1c",
        "urea_nitrogen",
        "c_reactive_protein",
        "white_blood_cells",
        "tsh",
        "free_t4",
        "inr",
        "d_dimer",
        "albumin",
    ):
        assert group in LAB_ITEM_GROUPS, group
    association_ids = {
        item.association_id for item in default_context_association_definitions()
    }
    assert "lab_sodium" in association_ids
    assert "lab_hba1c" in association_ids
    assert "lab_d_dimer" in association_ids


def test_lab_pivot_is_generated_for_every_group() -> None:
    from vcg_biomarker.mimic.evidence import _lab_window_pivot_sql

    excluded = ("potassium", "troponin_i", "troponin_t")
    sql = _lab_window_pivot_sql(LAB_ITEM_GROUPS, exclude=excluded)
    for group in LAB_ITEM_GROUPS:
        if group in excluded:
            assert f"AS {group}_nearest," not in sql
        else:
            assert f"AS {group}_nearest," in sql
            assert f"AS {group}_count_6h," in sql


# --- selection and verification bias --------------------------------------


def test_coverage_census_reports_standardised_differences() -> None:
    rng = np.random.default_rng(6)
    population = pd.DataFrame(
        {
            "has_ecg": [True] * 500 + [False] * 500,
            "age": np.concatenate(
                [rng.normal(70, 8, 500), rng.normal(50, 8, 500)]
            ),
            "noise": rng.normal(0, 1, 1000),
        }
    )
    census = coverage_census(
        population,
        covariate_columns=("age", "noise"),
    )
    by_covariate = census.set_index("covariate")
    assert by_covariate.loc["age", "materially_different"]
    assert abs(by_covariate.loc["age", "standardised_difference"]) > 1.0
    assert not by_covariate.loc["noise", "materially_different"]


def test_observation_propensity_weights_report_overlap() -> None:
    rng = np.random.default_rng(8)
    age = rng.normal(60, 12, 2000)
    probability = 1.0 / (1.0 + np.exp(-(age - 60) / 12.0))
    observed = rng.random(2000) < probability
    result = observation_propensity_weights(
        observed,
        pd.DataFrame({"age": age}),
    )
    assert result["status"] == "evaluable"
    assert 0.0 < result["propensity_min"] < result["propensity_max"] < 1.0
    assert result["effective_sample_fraction"] > 0.5
    assert "untestable" in result["interpretation"]


def test_verification_bias_bounds_bracket_the_naive_estimate() -> None:
    bounds = verification_bias_bounds(
        verified_positive=100,
        verified_negative=300,
        unverified=200,
        marker_positive_verified=80,
        marker_positive_unverified=20,
        marker_positive_verified_negative=30,
    )
    assert bounds["status"] == "evaluable"
    assert bounds["sensitivity_verified_only"] == pytest.approx(0.8)
    assert bounds["sensitivity_lower_bound"] < bounds["sensitivity_verified_only"]
    assert bounds["sensitivity_upper_bound"] >= bounds["sensitivity_verified_only"]
    assert bounds["specificity_verified_only"] == pytest.approx(0.9)
    assert bounds["specificity_lower_bound"] < bounds["specificity_verified_only"]
    assert bounds["specificity_upper_bound"] >= bounds["specificity_verified_only"]
    assert 0.0 <= bounds["ppv_lower_bound"] <= bounds["ppv_upper_bound"] <= 1.0
    assert bounds["verified_only_estimate_trustworthy"] is False


def test_verification_bounds_include_marker_missingness_extremes() -> None:
    observed_only = verification_bias_bounds(
        verified_positive=100,
        verified_negative=300,
        unverified=200,
        marker_positive_verified=80,
        marker_positive_unverified=20,
        marker_positive_verified_negative=30,
    )
    joint = verification_bias_bounds(
        verified_positive=110,
        verified_negative=315,
        unverified=225,
        marker_positive_verified=80,
        marker_positive_unverified=20,
        marker_positive_verified_negative=30,
        marker_missing_verified_positive=10,
        marker_missing_verified_negative=15,
        marker_missing_unverified=25,
    )

    # The same observed confusion table plus unresolved marker values can
    # only widen (never improve) the identified interval.
    assert joint["sensitivity_lower_bound"] <= observed_only[
        "sensitivity_lower_bound"
    ]
    assert joint["sensitivity_upper_bound"] >= observed_only[
        "sensitivity_upper_bound"
    ]
    assert joint["specificity_lower_bound"] <= observed_only[
        "specificity_lower_bound"
    ]
    assert joint["specificity_upper_bound"] >= observed_only[
        "specificity_upper_bound"
    ]
    assert joint["n_marker_missing_unverified"] == 25
    assert "joint_verification_and_marker_missingness" in joint["assumption"]


def test_holdout_strata_flag_small_groups() -> None:
    frame = pd.DataFrame(
        {
            "machine_cart_id": ["a"] * 300 + ["b"] * 50,
            "care_setting": ["ed"] * 350,
        }
    )
    strata = holdout_strata(frame, minimum_stratum_size=200)
    carts = strata.loc[strata["stratum_kind"] == "cart"].set_index(
        "stratum_value"
    )
    assert bool(carts.loc["a", "eligible_for_holdout"])
    assert not bool(carts.loc["b", "eligible_for_holdout"])
    assert carts.loc["b", "reason"] == "stratum_too_small"
    missing = strata.loc[strata["stratum_kind"] == "acquisition_era"]
    assert missing.iloc[0]["reason"] == "column_unavailable"


def test_negative_control_failure_indicates_residual_bias() -> None:
    results = pd.DataFrame(
        {
            "target_id": [
                "echo_lvh_strict",
                "hyperkalemia_near_ecg",
                "hypokalemia_near_ecg",
                "context_marital_status",
                "context_admission_weekday",
                "context_insurance",
                "context_ecg_sequence_number",
            ],
            "effect_size": [0.4, 0.3, 0.25, 0.18, 0.02, 0.01, 0.02],
            "fdr_significant": [True, True, True, True, False, False, False],
        }
    )
    table = evaluate_control_associations(results)
    summary = summarize_control_associations(table)
    assert summary["positive_controls_recovered"] == 3
    assert summary["residual_bias_indicated"] is True
    assert summary["failed_negative_controls"] == ["negative_marital_status"]


def test_control_associations_are_prespecified_and_balanced() -> None:
    kinds = [control.kind for control in DEFAULT_CONTROL_ASSOCIATIONS]
    assert kinds.count("positive") >= 3
    assert kinds.count("negative") >= 3
    for control in DEFAULT_CONTROL_ASSOCIATIONS:
        assert control.rationale.strip()


# --- target trial ---------------------------------------------------------


def _compliant_cohort(size: int = 200) -> pd.DataFrame:
    time_zero = pd.Timestamp("2150-06-01")
    return pd.DataFrame(
        {
            "treatment_arm": ["beta_blocker"] * (size // 2)
            + ["calcium_channel_blocker"] * (size // 2),
            "time_zero": [time_zero] * size,
            "prior_exposure_days": [0] * size,
            "baseline_ecg_time": [time_zero - pd.Timedelta(hours=3)] * size,
            "follow_up_ecg_time": [time_zero + pd.Timedelta(hours=8)] * size,
        }
    )


def test_a_compliant_cohort_permits_an_emulated_effect() -> None:
    result = assess_protocol_compliance(
        _compliant_cohort(),
        DEFAULT_PROTOCOLS[0],
    )
    assert result.claim_level == CLAIM_EMULATED_EFFECT
    assert not result.violated
    assert result.to_dict()["causal_claim_permitted"] is True


def test_a_single_arm_cohort_is_downgraded_to_association() -> None:
    cohort = _compliant_cohort()
    cohort["treatment_arm"] = "beta_blocker"
    result = assess_protocol_compliance(cohort, DEFAULT_PROTOCOLS[0])
    assert result.claim_level == CLAIM_ASSOCIATION
    assert "single_arm_no_active_comparator" in result.violated
    assert result.to_dict()["causal_claim_permitted"] is False


def test_prevalent_users_and_missing_follow_up_are_violations() -> None:
    cohort = _compliant_cohort()
    cohort.loc[: len(cohort) // 2, "prior_exposure_days"] = 5
    cohort.loc[: int(len(cohort) * 0.5), "follow_up_ecg_time"] = pd.NaT
    result = assess_protocol_compliance(cohort, DEFAULT_PROTOCOLS[0])
    assert "prevalent_users_included" in result.violated
    assert "informative_follow_up_ecg_selection" in result.violated
    assert result.diagnostics["follow_up_ecg_missing_share"] > 0.0


def test_baseline_after_time_zero_is_a_violation() -> None:
    cohort = _compliant_cohort()
    cohort["baseline_ecg_time"] = cohort["time_zero"] + pd.Timedelta(hours=1)
    result = assess_protocol_compliance(cohort, DEFAULT_PROTOCOLS[0])
    assert "baseline_ecg_after_time_zero" in result.violated


def test_a_protocol_needs_at_least_two_strategies() -> None:
    with pytest.raises(ValueError, match="at least two"):
        TargetTrialProtocol(
            protocol_id="broken",
            eligibility="x",
            treatment_strategies=("only one",),
            assignment_procedure="x",
            follow_up_start="x",
            outcome="x",
            causal_contrast="x",
            analysis_plan="x",
        )


def test_dose_levels_use_prespecified_cutpoints() -> None:
    result = dose_exposure_levels(np.arange(1, 101, dtype=float))
    assert result["status"] == "evaluable"
    assert result["cutpoints_prespecified"] is True
    assert set(result["levels"]) == {"low", "medium", "high"}
    assert len(result["cut_points"]) == 2


def test_protocol_registry_is_reviewable() -> None:
    table = protocol_registry_table()
    assert len(table) == len(DEFAULT_PROTOCOLS)
    for column in (
        "eligibility",
        "treatment_strategies",
        "assignment_procedure",
        "follow_up_start",
        "outcome",
        "causal_contrast",
        "analysis_plan",
    ):
        assert column in table.columns
        assert table[column].astype(str).str.len().gt(0).all()
