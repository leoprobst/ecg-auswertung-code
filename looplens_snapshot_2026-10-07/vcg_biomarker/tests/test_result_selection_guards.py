"""The endpoint summary, the criterion catalog and the operating points may only
be carried by rows that were actually tested.

Found in the A117 10k pilot: a marker measurable in fourteen recordings, one
of them a case, showed an AUC of 1.0 and became the pilot's "best marker" for
pulmonary embolism; a rotation-direction flag with an AUC of 0.52 became the
best marker for hypomagnesaemia because its "sensitivity at 95 % specificity"
read 0.96 -- at a specificity of 0.08, the most specific threshold the flag
has. Both selections were mechanical, and both are wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vcg_biomarker.mimic.criteria import selected_single_marker_ids
from vcg_biomarker.mimic.result_explorer import (
    NO_FORMAL_MARKER_STATUS,
    build_result_explorer_tables,
)
from vcg_biomarker.mimic.statistics import (
    evaluate_marker,
    fit_marker_model,
    learn_operating_thresholds,
)


def _screening_row(**overrides: object) -> dict:
    row = {
        "endpoint_id": "pulmonary_embolism_icd",
        "panel_id": "discovery_reserve",
        "evaluation_split": "validation",
        "analysis_type": "single_marker",
        "feature_id": "qrs_t_hausdorff_distance",
        "inference_status": "formal_inference",
        "n_positive": 90,
        "n_negative": 900,
        "missingness": 0.05,
        "roc_auc": 0.62,
        "q_value": 0.2,
        "assessability_gate_passed": True,
        "specificity_95_sensitivity": 0.15,
        "specificity_95_specificity": 0.95,
    }
    row.update(overrides)
    return row


def test_descriptive_rows_never_become_an_endpoints_best_marker() -> None:
    screening = pd.DataFrame(
        [
            _screening_row(
                feature_id="secondary_morphology_qt_beat_to_beat_variability_ms",
                inference_status="descriptive_only",
                n_positive=1,
                n_negative=13,
                missingness=0.99,
                roc_auc=1.0,
                q_value=np.nan,
                specificity_95_sensitivity=1.0,
            ),
            _screening_row(),
        ]
    )
    tables = build_result_explorer_tables(screening)
    summary = tables.endpoint_summary.iloc[0]
    assert summary.best_vcg_marker == "qrs_t_hausdorff_distance"
    assert summary.best_vcg_auc == 0.62
    ordered = tables.pathology_marker
    assert ordered.feature_id.tolist() == [
        "qrs_t_hausdorff_distance",
        "secondary_morphology_qt_beat_to_beat_variability_ms",
    ]
    assert ordered.selection_eligible.tolist() == [True, False]


def test_missing_or_ambiguous_selection_evidence_fails_closed() -> None:
    from vcg_biomarker.mimic.selection_contract import formal_selection_eligible

    base = pd.DataFrame([_screening_row()])
    for column in ("inference_status", "assessability_gate_passed"):
        assert not formal_selection_eligible(base.drop(columns=column)).any()
    for value in (None, pd.NA, np.nan, False, "false", "unknown", 0):
        assert not formal_selection_eligible(
            base.assign(assessability_gate_passed=value)
        ).any()
    assert formal_selection_eligible(base).all()


def test_specificity_is_not_rounded_up_and_missing_is_unknown() -> None:
    from vcg_biomarker.mimic.result_explorer import _annotate_operating_point_attainability

    result = _annotate_operating_point_attainability(pd.DataFrame({
        "operating_specificity": [0.90, np.nan, 0.95, 1.5],
        "sensitivity_at_95_specificity": [0.8] * 4,
    }))
    assert result.operating_point_attainable.tolist() == [False, False, True, False]
    assert result.operating_point_validation_status.tolist() == [
        "observed_target_not_met", "unknown", "observed_target_met", "unknown"
    ]
    assert result.sensitivity_at_95_specificity_reported.tolist() == [0.8] * 4


def test_validation_specificity_does_not_reselect_training_threshold():
    rows = [
        _screening_row(evaluation_split="train", specificity_95_specificity=0.96,
                       specificity_95_threshold_aligned=1.0, specificity_95_threshold_direction="above"),
        _screening_row(evaluation_split="validation", specificity_95_specificity=0.90,
                       specificity_95_threshold_aligned=1.0, specificity_95_threshold_direction="above"),
    ]
    result = build_result_explorer_tables(pd.DataFrame(rows)).marker_pathology.iloc[0]
    assert result.operating_point_training_status == "training_target_met"
    assert result.operating_point_validation_status == "observed_target_not_met"
    assert result.operating_point_ranking_eligible
    assert result.sensitivity_at_95_specificity == 0.15
    rows[1]["specificity_95_threshold_aligned"] = 0.8
    result = build_result_explorer_tables(pd.DataFrame(rows)).marker_pathology.iloc[0]
    assert result.operating_point_training_status == "threshold_mismatch"
    assert not result.operating_point_ranking_eligible
    assert pd.isna(result.sensitivity_at_95_specificity)


def test_duplicate_training_rows_do_not_prove_threshold_provenance():
    train = _screening_row(evaluation_split="train", specificity_95_specificity=0.96,
                          specificity_95_threshold_aligned=1.0,
                          specificity_95_threshold_direction="above")
    validation = {**train, "evaluation_split": "validation"}
    result = build_result_explorer_tables(pd.DataFrame([train, train, validation])).marker_pathology.iloc[0]
    assert result.operating_point_training_status == "unknown_training_evidence"
    assert not result.operating_point_ranking_eligible


def test_an_endpoint_without_a_formal_row_has_no_best_marker() -> None:
    screening = pd.DataFrame(
        [
            _screening_row(
                inference_status="descriptive_only",
                n_positive=2,
                n_negative=5,
                roc_auc=1.0,
                q_value=np.nan,
            )
        ]
    )
    tables = build_result_explorer_tables(screening)
    summary = tables.endpoint_summary.iloc[0]
    assert summary.best_vcg_marker == ""
    assert pd.isna(summary.best_vcg_auc)
    assert summary.status == NO_FORMAL_MARKER_STATUS
    assert summary.pathology_group != ""
    # The row itself is still in the marker table -- as what it is.
    assert len(tables.pathology_marker) == 1
    assert not bool(tables.pathology_marker.selection_eligible.iloc[0])


def test_an_unattained_specificity_target_cannot_rank_a_marker() -> None:
    screening = pd.DataFrame(
        [
            _screening_row(
                endpoint_id="hypomagnesemia_near_ecg",
                feature_id="qrs_loop_horizontal_xz_rotation_direction",
                roc_auc=0.52,
                specificity_95_sensitivity=0.96,
                specificity_95_specificity=0.08,
            ),
            _screening_row(
                endpoint_id="hypomagnesemia_near_ecg",
                feature_id="t_loop_width_major",
                roc_auc=0.61,
                specificity_95_sensitivity=0.12,
                specificity_95_specificity=0.95,
            ),
        ]
    )
    tables = build_result_explorer_tables(screening)
    assert tables.endpoint_summary.iloc[0].best_vcg_marker == "t_loop_width_major"
    markers = tables.pathology_marker.set_index("feature_id")
    flag = markers.loc["qrs_loop_horizontal_xz_rotation_direction"]
    assert not bool(flag.operating_point_attainable)
    assert pd.isna(flag.sensitivity_at_95_specificity)
    assert flag.sensitivity_at_95_specificity_reported == 0.96
    assert bool(markers.loc["t_loop_width_major"].operating_point_attainable)


def test_criterion_candidates_are_drawn_only_from_formal_rows() -> None:
    labels = np.asarray([0] * 40 + [1] * 40, dtype=np.int8)
    values = labels + np.linspace(-0.3, 0.3, len(labels))
    model = fit_marker_model(values, labels, "qrs_t_hausdorff_distance")
    screening = pd.DataFrame(
        [
            _screening_row(inference_status="descriptive_only", roc_auc=1.0, n_positive=2),
        ]
    )
    assert selected_single_marker_ids(
        endpoint_id="pulmonary_embolism_icd",
        screening=screening,
        stability=pd.DataFrame(),
        marker_models={"qrs_t_hausdorff_distance": model},
        maximum=3,
    ) == ()
    formal = pd.DataFrame([_screening_row()])
    assert selected_single_marker_ids(
        endpoint_id="pulmonary_embolism_icd",
        screening=formal,
        stability=pd.DataFrame(),
        marker_models={"qrs_t_hausdorff_distance": model},
        maximum=3,
    ) == ("qrs_t_hausdorff_distance",)


def test_a_flag_without_a_specific_threshold_gets_the_predict_nobody_point() -> None:
    # 90 % of controls and 96 % of cases carry the top value of a ternary
    # marker: no threshold reaches 95 % specificity except the one above all
    # observed values.
    rng = np.random.default_rng(3)
    controls = rng.choice([-1.0, 0.0, 1.0], size=1000, p=[0.05, 0.05, 0.90])
    cases = rng.choice([-1.0, 0.0, 1.0], size=100, p=[0.02, 0.02, 0.96])
    values = np.concatenate([controls, cases])
    labels = np.concatenate([np.zeros(1000, dtype=np.int8), np.ones(100, dtype=np.int8)])
    thresholds = learn_operating_thresholds(values, labels)
    assert thresholds["specificity_95"] > 1.0
    metrics = evaluate_marker(values, labels, direction=1, thresholds=thresholds)
    assert metrics["specificity_95_specificity"] == 1.0
    assert metrics["specificity_95_sensitivity"] == 0.0
    assert metrics["specificity_95_target_attainable"] is True
    # The same marker read the other way round: still nobody positive.
    flipped = evaluate_marker(-values, labels, direction=-1, thresholds=thresholds)
    assert flipped["specificity_95_specificity"] == 1.0
    assert flipped["specificity_95_target_attainable"] is True


def test_a_continuous_marker_keeps_its_attainable_operating_points() -> None:
    rng = np.random.default_rng(11)
    labels = np.asarray([0] * 500 + [1] * 100, dtype=np.int8)
    values = labels * 1.2 + rng.normal(0.0, 1.0, len(labels))
    thresholds = learn_operating_thresholds(values, labels)
    metrics = evaluate_marker(values, labels, direction=1, thresholds=thresholds)
    assert metrics["specificity_95_specificity"] >= 0.95
    assert 0.0 < metrics["specificity_95_sensitivity"] < 1.0
    assert metrics["specificity_95_target_attainable"] is True
    assert metrics["sensitivity_90_target_attainable"] is True
    assert "youden_exploratory_target_attainable" not in metrics


def test_regular_rhythm_auc_is_reported_beside_the_pooled_auc() -> None:
    from vcg_biomarker.mimic.validation import _regular_rhythm_metrics

    rng = np.random.default_rng(5)
    n = 400
    rhythm = np.where(np.arange(n) % 4 == 0, "af_compatible_pattern", "regular")
    labels = (rng.random(n) < 0.3).astype(np.int8)
    values = labels * 1.5 + rng.normal(0.0, 1.0, n)
    # The marker is only measurable in regular rhythm, as the Direct-8 family is.
    values[rhythm != "regular"] = np.nan
    frame = pd.DataFrame({"rhythm_context": rhythm})
    metrics = _regular_rhythm_metrics(frame, values, labels, 1)
    assert metrics["n_regular_positive"] + metrics["n_regular_negative"] == 300
    assert metrics["regular_rhythm_fraction_of_evaluable"] == 1.0
    assert 0.7 < metrics["roc_auc_regular_rhythm"] < 1.0
    # Without the column the stratum is unknown, not "regular".
    blank = _regular_rhythm_metrics(pd.DataFrame(index=range(n)), values, labels, 1)
    assert np.isnan(blank["roc_auc_regular_rhythm"])
    assert blank["n_regular_positive"] == 0
    # Too few cases in the stratum: no number.
    few = _regular_rhythm_metrics(frame.iloc[:20], values[:20], labels[:20], 1)
    assert np.isnan(few["roc_auc_regular_rhythm"])


def test_paired_availability_comparison_has_its_own_verdict_labels() -> None:
    from vcg_biomarker.mimic.validation import (
        PAIRED_AVAILABILITY_COMPARISON_TYPE,
        PAIRED_AVAILABILITY_COMPARISONS,
        _comparison_verdict,
    )

    # Every prespecified pair names the Fréchet coupling as the candidate and
    # the Direct-8 beat-median angle as the comparator; nothing is chosen at
    # run time.
    for pairs in PAIRED_AVAILABILITY_COMPARISONS.values():
        assert pairs == (
            (
                "qrs_t_normalized_discrete_frechet",
                "qrs_t_integral_angle_direct_8lead_beat_median_deg",
            ),
        )
    base = {
        "comparison_type": PAIRED_AVAILABILITY_COMPARISON_TYPE,
        "inference_status": "formal_inference",
    }
    assert _comparison_verdict({**base, "delta_auc_ci_low": 0.01, "delta_auc_ci_high": 0.05})[0] == (
        "new_marker_significantly_better"
    )
    key, label = _comparison_verdict({**base, "delta_auc_ci_low": -0.06, "delta_auc_ci_high": -0.01})
    assert key == "comparator_marker_significantly_better"
    assert "klassische Regel" not in label
    assert _comparison_verdict({**base, "delta_auc_ci_low": -0.02, "delta_auc_ci_high": 0.03})[0] == "equivalent"
    assert _comparison_verdict({**base, "inference_status": "underpowered"})[0] == "insufficient_evidence"
