"""Regression tests for the 2026-08-12 methodology audit fixes.

Each test pins one defect the audit found, so a later refactor cannot quietly
restore it. See AUDIT_2026-08-12_VORGEHEN.md for the findings these cover.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.adjusted_models import anchor_year_group_start
from vcg_biomarker.mimic.association_domains import (
    _scope_definitions_to_available_sources,
    default_context_association_definitions,
)
from vcg_biomarker.mimic.criteria import (
    is_analytical_concordance_endpoint,
    is_time_to_event_primary_endpoint,
)
from vcg_biomarker.mimic.delineation_sensitivity import (
    evaluate_interval_criterion_sensitivity,
    machine_interval_values,
    summarize_comparator_sensitivity,
)
from vcg_biomarker.mimic.jobs import project_config_drift
from vcg_biomarker.mimic.label_registry import default_label_definitions
from vcg_biomarker.mimic.models import TargetKind
from vcg_biomarker.mimic.note_labels import (
    NOTE_COVERAGE_FILENAME,
    read_note_coverage,
)
from vcg_biomarker.mimic.phenome_adjustment import (
    PHENOME_ADJUSTMENT_COVARIATES,
)
from vcg_biomarker.mimic.storage import StageManifest


# --- B1: absent note sources must not read as assessed absence -------------


def test_stage_manifest_non_complete_status_is_not_reused(tmp_path) -> None:
    manifest = StageManifest.complete(
        stage="parse-notes",
        stage_version="2.2.0",
        input_hash="abc",
        config_hash="def",
        output_paths=(),
        row_counts={"note_encounters": 0},
        status="incomplete_note_sources_not_configured",
    )
    assert manifest.status != "complete"
    assert not manifest.matches("abc", "def", "2.2.0")


def test_missing_note_coverage_reports_not_run(tmp_path) -> None:
    class _Config:
        evidence_root = tmp_path

    coverage = read_note_coverage(_Config())
    assert coverage["status"] == "not_run"
    assert coverage["assessed"] is False


def test_note_coverage_only_counts_as_assessed_when_complete(tmp_path) -> None:
    class _Config:
        evidence_root = tmp_path

    for status, expected in (
        ("complete", True),
        ("partial", False),
        ("not_configured", False),
        ("configured_but_not_ingested", False),
    ):
        (tmp_path / NOTE_COVERAGE_FILENAME).write_text(
            json.dumps({"status": status}),
            encoding="utf-8",
        )
        assert read_note_coverage(_Config())["assessed"] is expected


def test_note_domains_are_dropped_and_recorded_when_notes_absent(
    tmp_path,
) -> None:
    class _Config:
        evidence_root = tmp_path

    definitions = default_context_association_definitions()
    note_definitions = [
        item for item in definitions if item.domain.startswith("note_")
    ]
    assert note_definitions, "expected note-dependent association domains"

    retained, coverage = _scope_definitions_to_available_sources(
        _Config(),
        definitions,
    )
    retained_ids = {item.association_id for item in retained}
    for definition in note_definitions:
        assert definition.association_id not in retained_ids
    dropped = coverage.loc[~coverage["source_available"].astype(bool)]
    assert set(dropped["association_id"]) == {
        item.association_id for item in note_definitions
    }
    assert (dropped["assessment_state"] == "not_assessed").all()
    assert (dropped["unavailable_reason"] != "").all()


def test_note_domains_are_kept_when_notes_were_parsed(tmp_path) -> None:
    class _Config:
        evidence_root = tmp_path

    (tmp_path / NOTE_COVERAGE_FILENAME).write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    definitions = default_context_association_definitions()
    retained, coverage = _scope_definitions_to_available_sources(
        _Config(),
        definitions,
    )
    assert len(retained) == len(definitions)
    assert coverage["source_available"].astype(bool).all()


# --- B2: configuration drift against a pinned job snapshot -----------------


def test_resource_changes_are_operational_drift_only() -> None:
    snapshot = {"workers": 6, "duckdb_memory_limit_gb": 20, "split_salt": "a"}
    live = {"workers": 14, "duckdb_memory_limit_gb": 24, "split_salt": "a"}
    drift = project_config_drift(snapshot, live)
    assert drift["operational"] == ["duckdb_memory_limit_gb", "workers"]
    assert drift["scientific"] == []
    assert drift["source_added"] == []


def test_newly_configured_note_root_is_flagged_as_source_added() -> None:
    drift = project_config_drift(
        {"mimic_note_root": None},
        {"mimic_note_root": "E:/Database/mimic-iv-note"},
    )
    assert drift["source_added"] == ["mimic_note_root"]


def test_split_salt_change_is_scientific_drift() -> None:
    drift = project_config_drift({"split_salt": "a"}, {"split_salt": "b"})
    assert drift["scientific"] == ["split_salt"]


# --- W1: de-identification-shifted years must not encode calendar time -----


def test_anchor_year_group_start_parses_the_real_era() -> None:
    parsed = anchor_year_group_start(
        ["2008 - 2010", "2014 - 2016", "2020 - 2022", None, "", "junk"]
    )
    assert list(parsed[:3]) == [2008.0, 2014.0, 2020.0]
    assert parsed[3:].isna().all()


def test_shifted_anchor_years_do_not_separate_eras() -> None:
    # Mirrors the real distribution measured on this project's patients table:
    # every true three-year group spans essentially the same shifted years, so
    # a split on the shifted year is not a split on calendar time.
    frame = pd.DataFrame(
        {
            "anchor_year_group": ["2008 - 2010"] * 3 + ["2020 - 2022"] * 3,
            "shifted_year": [2110, 2160, 2203, 2115, 2158, 2189],
        }
    )
    by_group = frame.groupby("anchor_year_group")["shifted_year"]
    early, late = by_group.min().to_list(), by_group.max().to_list()
    assert early[0] < late[1] and early[1] < late[0], "ranges must overlap"
    era = anchor_year_group_start(frame["anchor_year_group"])
    assert era.nunique() == 2, "the group field does separate the eras"


def test_adjusted_models_use_era_not_shifted_year() -> None:
    from vcg_biomarker.mimic import adjusted_models

    source = (adjusted_models.__file__ or "")
    assert source
    text = Path(source).read_text(encoding="utf-8")
    assert "anchor_era_start_year" in text
    assert '"ecg_year"' not in text


# --- W2: same-waveform reference standards are not diagnostic --------------


def test_qtc_endpoints_are_registered_as_analytical_concordance() -> None:
    definitions = {item.label_id: item for item in default_label_definitions()}
    for label_id in ("qtc_gt_480", "qtc_gt_500"):
        assert (
            definitions[label_id].target_kind
            is TargetKind.ANALYTICAL_CONCORDANCE
        )
        assert is_analytical_concordance_endpoint(label_id)
    assert definitions["echo_lvh_strict"].target_kind is TargetKind.DIAGNOSTIC
    assert not is_analytical_concordance_endpoint("echo_lvh_strict")


def test_concordance_endpoints_cannot_reach_a_replication_grade() -> None:
    from vcg_biomarker.mimic.criteria import _evidence_stage
    from vcg_biomarker.mimic.models import CriterionStage

    evidence = {"core_replication_evidence_passed": True}
    stage, _ = _evidence_stage(
        {"inference_status": "formal_inference"},
        evidence=evidence,
        endpoint_id="qtc_gt_480",
    )
    assert stage is CriterionStage.HYPOTHESIS
    stage, _ = _evidence_stage(
        {"inference_status": "formal_inference"},
        evidence=evidence,
        endpoint_id="echo_lvh_strict",
    )
    assert stage is CriterionStage.TRAIN_STABLE


# --- W3: comparator sensitivity to the delineation boundary bias -----------


def _sensitivity_frame(own_qt_offset_ms: float) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    size = 500
    true_qt = rng.normal(400.0, 30.0, size)
    rate = rng.normal(72.0, 8.0, size)
    machine_qt = true_qt + rng.normal(0.0, 3.0, size)
    own_qt = true_qt + own_qt_offset_ms + rng.normal(0.0, 3.0, size)
    own_qtc = own_qt / np.cbrt(60.0 / rate)
    return pd.DataFrame(
        {
            "label": rng.integers(0, 2, size),
            "classical_qtc_gt_480": own_qtc,
            "machine_qt_interval_ms": machine_qt,
            "machine_heart_rate_bpm": rate,
        }
    )


def test_machine_qtc_is_fridericia_corrected() -> None:
    frame = pd.DataFrame(
        {
            "machine_qt_interval_ms": [400.0],
            "machine_heart_rate_bpm": [60.0],
        }
    )
    values = machine_interval_values(frame, "qtc_fridericia")
    assert values[0] == pytest.approx(400.0)


def test_out_of_range_rate_makes_machine_qtc_unmeasurable() -> None:
    frame = pd.DataFrame(
        {
            "machine_qt_interval_ms": [400.0, 400.0],
            "machine_heart_rate_bpm": [0.0, 500.0],
        }
    )
    assert np.isnan(machine_interval_values(frame, "qtc_fridericia")).all()


def test_early_t_offset_makes_the_own_rule_fire_less_often() -> None:
    frame = _sensitivity_frame(own_qt_offset_ms=-20.0)
    result = evaluate_interval_criterion_sensitivity(
        frame,
        endpoint_id="demo",
        evaluation_split="validation",
    )
    row = result.loc[result["criterion_id"] == "qtc_gt_480"].iloc[0]
    assert row["interval_bias_ms"] < -10.0
    assert row["own_positive_rate"] < row["machine_positive_rate"]
    assert row["comparator_bias_direction"] == "own_rule_fires_less_often"


def test_unbiased_delineation_shows_no_material_comparator_shift() -> None:
    frame = _sensitivity_frame(own_qt_offset_ms=0.0)
    result = evaluate_interval_criterion_sensitivity(
        frame,
        endpoint_id="demo",
        evaluation_split="validation",
    )
    row = result.loc[result["criterion_id"] == "qtc_gt_480"].iloc[0]
    assert abs(row["interval_bias_ms"]) < 5.0
    assert bool(row["comparator_conclusion_robust"])


def test_sensitivity_reports_unmeasurable_criteria_instead_of_dropping() -> None:
    result = evaluate_interval_criterion_sensitivity(
        pd.DataFrame({"label": [0, 1]}),
        endpoint_id="demo",
        evaluation_split="validation",
    )
    assert (result["sensitivity_status"] == "not_evaluable").all()
    assert not result["comparator_conclusion_robust"].any()
    summary = summarize_comparator_sensitivity(result)
    assert summary["criteria_evaluated"] == 0


# --- W4/W5: heart rate and acuity are adjusted for ------------------------


def test_phewas_adjustment_includes_heart_rate_and_acuity() -> None:
    assert "heart_rate" in PHENOME_ADJUSTMENT_COVARIATES
    assert "index_icu_stay" in PHENOME_ADJUSTMENT_COVARIATES


def test_endpoint_models_adjust_for_heart_rate() -> None:
    from vcg_biomarker.mimic import adjusted_models

    text = Path(adjusted_models.__file__).read_text(encoding="utf-8")
    assert '"heart_rate",' in text


def test_unavailable_covariates_are_named_not_silently_dropped() -> None:
    from vcg_biomarker.mimic.phenome_adjustment import (
        PhenomeAdjustmentPolicy,
        confirm_replicated_phenome_hits,
    )

    rng = np.random.default_rng(4242)
    rows, current, history = [], [], []
    for index in range(400):
        positive = index < 200
        subject_id = index + 1
        rows.append(
            {
                "subject_id": subject_id,
                "study_id": subject_id + 10_000,
                "split": "validation",
                "is_control_index": not positive,
                "is_case_index": positive,
                "age": rng.normal(70.0 if positive else 58.0, 8.0),
                "sex": "M" if index % 2 else "F",
                "signal_quality_score": rng.uniform(0.7, 1.0),
                "marker": 0.9 * positive + rng.normal(0.0, 0.6),
            }
        )
        if positive:
            current.append(
                {
                    "subject_id": subject_id,
                    "study_id": subject_id + 10_000,
                    "split": "validation",
                    "phenotype_id": "P1",
                }
            )
            history.append(
                {"subject_id": subject_id, "body_system": "circulatory"}
            )
    replicated = pd.DataFrame(
        [
            {
                "phenotype_id": "P1",
                "phenotype_name": "Synthetic",
                "body_system": "circulatory",
                "panel_id": "primary_core",
                "feature_id": "marker",
                "train_direction": 1,
                "replicated": True,
            }
        ]
    )
    adjusted = confirm_replicated_phenome_hits(
        replicated,
        pd.DataFrame(rows),
        pd.DataFrame(current),
        pd.DataFrame(history),
        policy=PhenomeAdjustmentPolicy(minimum_events_per_parameter=5),
    ).iloc[0]
    # Neither heart rate nor ICU state exists in this frame, so both must be
    # named as unavailable rather than quietly treated as adjusted for.
    unavailable = set(str(adjusted["covariates_unavailable"]).split("|"))
    assert {"heart_rate", "index_icu_stay"} <= unavailable
    assert not bool(adjusted["covariate_set_complete"])
    assert "age" in str(adjusted["covariates_applied"])


# --- W6: the binary future arm is a sensitivity analysis ------------------


def test_new_onset_af_declares_its_selection_and_primary_arm() -> None:
    definitions = {item.label_id: item for item in default_label_definitions()}
    for label_id in ("new_onset_af_30d", "new_onset_af_90d", "new_onset_af_1y"):
        definition = definitions[label_id]
        assert definition.primary_analysis_arm == "time_to_event"
        assert "complete_followup" in definition.estimand
        assert "condition" in definition.selection_mechanism
        assert is_time_to_event_primary_endpoint(label_id)


def test_binary_future_endpoints_cannot_carry_a_replication_grade() -> None:
    from vcg_biomarker.mimic.criteria import _evidence_stage, _limitations
    from vcg_biomarker.mimic.models import CriterionStage

    stage, _ = _evidence_stage(
        {"inference_status": "formal_inference"},
        evidence={"core_replication_evidence_passed": True},
        endpoint_id="new_onset_af_30d",
    )
    assert stage is CriterionStage.HYPOTHESIS
    limitations = _limitations({}, None, {}, endpoint_id="new_onset_af_30d")
    assert any("Zensierung" in item for item in limitations)


# --- M1: exit-time logic survived the dead-code removal -------------------


def test_future_duration_uses_event_death_and_administrative_exit() -> None:
    from vcg_biomarker.mimic.future_phenome import _future_state_masks

    index_end = pd.Timestamp("2150-01-01")
    frame = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "index_end_time": [index_end] * 4,
            # 1: event at day 10, 2: death at day 20, 3: censored at day 45,
            # 4: fully observed through the 90-day horizon.
            "last_observed_time": [
                index_end + pd.Timedelta(days=200),
                index_end + pd.Timedelta(days=200),
                index_end + pd.Timedelta(days=45),
                index_end + pd.Timedelta(days=200),
            ],
            "date_of_death": [
                pd.NaT,
                index_end + pd.Timedelta(days=20),
                pd.NaT,
                pd.NaT,
            ],
            "sex": ["M"] * 4,
        }
    )
    events = pd.Series(
        {1: index_end + pd.Timedelta(days=10)},
        dtype="datetime64[ns]",
    )
    masks = _future_state_masks(
        frame,
        events,
        pd.Series({"male_only": False, "female_only": False}),
        horizon_days=90,
    )
    duration = masks["duration_days"]
    assert masks["positive"].tolist() == [True, False, False, False]
    assert masks["competing_death"].tolist() == [False, True, False, False]
    assert duration[0] == pytest.approx(10.0)
    assert duration[1] == pytest.approx(20.0)
    assert duration[2] == pytest.approx(45.0)
    assert duration[3] == pytest.approx(90.0), "administrative horizon caps"
    assert masks["negative"].tolist() == [False, False, False, True]
