"""Derived markers and the status-code phenotype arm.

Two things are pinned here beyond the arithmetic: that a derived marker cannot
reach a report without being declared as one, and that a status code whose
documentation follows the ECG finding is never offered as a predictor of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.derived_markers import (
    DERIVED_MARKER_KINDS,
    KIND_CIRCADIAN_RESIDUAL,
    KIND_PROVOKED_RESERVE,
    KIND_RECOVERY_KINETICS,
    KIND_RELIABILITY_COMPOSITE,
    annotate_results_with_derived_kind,
    assert_derived_markers_are_declared,
    circadian_reference,
    circadian_residual,
    circadian_residual_table,
    declare,
    derived_marker_declarations,
    provoked_reserve,
    recovery_kinetics,
    reliability_weighted_composite,
    reliability_weights,
)
from vcg_biomarker.mimic.status_phenotypes import (
    STATUS_PHENOTYPES,
    TEMPORAL_FOLLOWS,
    coverage_summary,
    map_status_codes,
    status_association_definitions,
    status_flags_sql,
    status_phenotype_registry,
    unmapped_code_census,
)


# --- derived markers --------------------------------------------------------


def _diurnal_frame(seed: int = 4, n: int = 3000):
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 24, n)
    times = pd.Timestamp("2150-03-01") + pd.to_timedelta(hours, unit="h")
    values = 420 + 8 * np.cos(2 * np.pi * (hours - 5) / 24) + rng.normal(0, 6, n)
    return pd.DataFrame({"ecg_time": times, "qtc": values}), 6.0


def test_circadian_residual_removes_the_diurnal_component() -> None:
    frame, noise_sd = _diurnal_frame()
    reference = circadian_reference(frame["qtc"], frame["ecg_time"], marker_id="qtc")
    residual = circadian_residual(frame["qtc"], frame["ecg_time"], reference)
    # The residual should retain the noise and lose the 8 ms rhythm.
    assert np.nanstd(residual) == pytest.approx(noise_sd, abs=0.6)
    assert np.nanstd(residual) < np.std(frame["qtc"]) - 1.0


def test_circadian_residual_is_nan_when_the_reference_fails() -> None:
    residual = circadian_residual(
        [1.0, 2.0, 3.0],
        pd.to_datetime(["2150-01-01"] * 3),
        {"status": "not_evaluable"},
    )
    # Never silently relabel the raw value as a residual.
    assert np.isnan(residual).all()


def test_circadian_residual_table_declares_every_column() -> None:
    frame, _ = _diurnal_frame()
    table, declarations = circadian_residual_table(frame, ["qtc"])
    assert list(table.columns) == ["circadian_residual__qtc"]
    assert len(declarations) == 1
    declaration = declarations[0]
    assert declaration.kind == KIND_CIRCADIAN_RESIDUAL
    assert declaration.base_marker_id == "qtc"
    assert declaration.per_record_marker is False
    assert declaration.core_sentinel_eligible is False
    assert declaration.post_outcome_hypothesis is True


def test_provoked_reserve_recovers_per_patient_sensitivity() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for subject in range(300):
        sensitivity = 0.09 if subject % 3 == 0 else 0.02
        for dose in (0, 100, 200, 400):
            rows.append(
                {
                    "subject_id": subject,
                    "dose": dose,
                    "qtc": 420 + sensitivity * dose + rng.normal(0, 4),
                }
            )
    table = provoked_reserve(
        pd.DataFrame(rows),
        marker_column="qtc",
        dose_column="dose",
    )
    evaluable = table.loc[table["status"].eq("evaluable")]
    assert len(evaluable) == 300
    low_reserve = evaluable.loc[evaluable["subject_id"] % 3 == 0]
    high_reserve = evaluable.loc[evaluable["subject_id"] % 3 != 0]
    assert low_reserve["reserve_slope_per_unit_dose"].median() == pytest.approx(
        0.09,
        abs=0.015,
    )
    assert high_reserve["reserve_slope_per_unit_dose"].median() == pytest.approx(
        0.02,
        abs=0.015,
    )
    assert (evaluable["derived_marker_kind"] == KIND_PROVOKED_RESERVE).all()


def test_provoked_reserve_abstains_without_dose_variation() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "dose": [100.0, 100.0, 100.0],
            "qtc": [420.0, 430.0, 425.0],
        }
    )
    table = provoked_reserve(frame, marker_column="qtc", dose_column="dose")
    assert table.iloc[0]["status"] == "not_evaluable"
    assert "dose_variation" in table.iloc[0]["failure_reason"]


def test_recovery_kinetics_measures_return_towards_baseline() -> None:
    rng = np.random.default_rng(11)
    landmark = pd.Timestamp("2150-06-01 12:00")
    rows = []
    for subject in range(80):
        rows.append(
            {
                "subject_id": subject,
                "ecg_time": landmark - pd.Timedelta(hours=6),
                "landmark_time": landmark,
                "marker": 100.0,
            }
        )
        for hours in (2, 6, 12, 24, 48):
            rows.append(
                {
                    "subject_id": subject,
                    "ecg_time": landmark + pd.Timedelta(hours=hours),
                    "landmark_time": landmark,
                    "marker": 100.0
                    + 40 * np.exp(-hours / 12.0)
                    + rng.normal(0, 1),
                }
            )
    table = recovery_kinetics(pd.DataFrame(rows), marker_column="marker")
    evaluable = table.loc[table["status"].eq("evaluable")]
    assert len(evaluable) == 80
    assert evaluable["baseline_value"].median() == pytest.approx(100.0, abs=1.0)
    assert evaluable["peak_deviation"].median() > 20.0
    assert evaluable["recovery_half_time_hours"].median() > 0.0
    # Observation density is reported because it is itself informative.
    assert (evaluable["observation_density_per_day"] > 0).all()


def test_recovery_kinetics_abstains_without_a_baseline() -> None:
    landmark = pd.Timestamp("2150-06-01 12:00")
    rows = [
        {
            "subject_id": 1,
            "ecg_time": landmark + pd.Timedelta(hours=hours),
            "landmark_time": landmark,
            "marker": 100.0,
        }
        for hours in (1, 2, 3, 4)
    ]
    table = recovery_kinetics(pd.DataFrame(rows), marker_column="marker")
    assert table.iloc[0]["status"] == "not_evaluable"


def test_reliability_weights_are_inverse_variance() -> None:
    repeatability = pd.DataFrame(
        {"marker_id": ["a", "b"], "within_subject_sd": [1.0, 4.0]}
    )
    weights = reliability_weights(repeatability, ["a", "b"])
    # sd 1 vs 4 means variance 1 vs 16, so the weight ratio must be 16.
    assert weights["a"] / weights["b"] == pytest.approx(16.0)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_components_without_a_repeatability_estimate_are_excluded() -> None:
    repeatability = pd.DataFrame(
        {"marker_id": ["a"], "within_subject_sd": [1.0]}
    )
    weights = reliability_weights(repeatability, ["a", "unknown"])
    assert set(weights) == {"a"}


def test_composite_requires_a_majority_of_its_components() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan],
            "b": [1.0, 2.0, np.nan, np.nan],
        }
    )
    composite = reliability_weighted_composite(frame, {"a": 0.5, "b": 0.5})
    assert np.isfinite(composite[0])
    # The last row has neither component; a composite from nothing is NaN.
    assert np.isnan(composite[3])


# --- presentation contract --------------------------------------------------


def test_results_are_annotated_with_the_derived_kind() -> None:
    declaration = declare(
        "circadian_residual__qtc",
        KIND_CIRCADIAN_RESIDUAL,
        "qtc",
        "ms",
    )
    results = pd.DataFrame(
        {"feature_id": ["qrs_loop_area", "circadian_residual__qtc"]}
    )
    annotated = annotate_results_with_derived_kind(results, [declaration])
    by_feature = annotated.set_index("feature_id")
    assert by_feature.loc["qrs_loop_area", "derived_marker_kind"] == "per_record_marker"
    assert by_feature.loc["qrs_loop_area", "per_record_marker"]
    assert (
        by_feature.loc["circadian_residual__qtc", "derived_marker_kind"]
        == KIND_CIRCADIAN_RESIDUAL
    )
    assert not by_feature.loc["circadian_residual__qtc", "per_record_marker"]
    assert by_feature.loc["circadian_residual__qtc", "presentation_caveat_de"]


def test_undeclared_derived_marker_is_refused() -> None:
    results = pd.DataFrame({"feature_id": ["circadian_residual__qtc"]})
    with pytest.raises(AssertionError, match="without a declaration"):
        assert_derived_markers_are_declared(results, [])
    declaration = declare(
        "circadian_residual__qtc",
        KIND_CIRCADIAN_RESIDUAL,
        "qtc",
        "ms",
    )
    assert_derived_markers_are_declared(results, [declaration])


def test_every_kind_carries_a_caveat_and_an_observation_unit() -> None:
    for kind in DERIVED_MARKER_KINDS:
        declaration = declare(f"{kind}__demo", kind, "demo", "unit")
        payload = declaration.to_dict()
        assert payload["presentation_caveat_de"].strip()
        assert payload["observation_unit"] in {"ecg", "patient", "patient_episode"}
        assert payload["per_record_marker"] is False
        assert payload["core_sentinel_eligible"] is False


def test_provoked_reserve_is_declared_a_trait_not_a_state() -> None:
    declaration = declare("x", KIND_PROVOKED_RESERVE, "qtc", "ms/mg")
    assert declaration.state_or_trait == "trait"
    assert declaration.observation_unit == "patient"
    assert "Confounding by indication" in declaration.caveat_de
    recovery = declare("y", KIND_RECOVERY_KINETICS, "qtc", "h")
    assert recovery.observation_unit == "patient_episode"
    composite = declare("z", KIND_RELIABILITY_COMPOSITE, "qtc", "z")
    assert "eingefroren" in composite.caveat_de


def test_declaration_table_is_reviewable() -> None:
    declarations = [
        declare(f"{kind}__demo", kind, "demo", "unit")
        for kind in DERIVED_MARKER_KINDS
    ]
    table = derived_marker_declarations(declarations)
    assert len(table) == len(DERIVED_MARKER_KINDS)
    assert not table["presentation_caveat_de"].eq("").any()


# --- status-code phenotypes -------------------------------------------------


def test_status_registry_covers_the_dropped_code_families() -> None:
    ids = {item.status_id for item in STATUS_PHENOTYPES}
    for expected in (
        "long_term_anticoagulant_use",
        "long_term_antiplatelet_use",
        "long_term_insulin_use",
        "do_not_resuscitate_status",
        "cardiac_device_present",
        "dialysis_dependence",
        "tobacco_use_status",
        "alcohol_use_status",
    ):
        assert expected in ids, expected
    assert len(STATUS_PHENOTYPES) >= 15


def test_status_codes_match_the_real_mimic_codes() -> None:
    mapped = map_status_codes(
        [
            ("Z7901", 10),
            ("V5861", 9),
            ("Z66", 10),
            ("V4986", 9),
            ("Z794", 10),
            ("I214", 10),
        ]
    )
    assert "Z7901" in mapped["long_term_anticoagulant_use"]
    assert "V5861" in mapped["long_term_anticoagulant_use"]
    assert "Z66" in mapped["do_not_resuscitate_status"]
    assert "V4986" in mapped["do_not_resuscitate_status"]
    assert "Z794" in mapped["long_term_insulin_use"]
    # A genuine disease code belongs to the disease taxonomy, not here.
    assert not any("I214" in codes for codes in mapped.values())


def test_status_flags_sql_produces_one_column_per_phenotype() -> None:
    sql = status_flags_sql("dx")
    for phenotype in STATUS_PHENOTYPES:
        assert f"AS status_{phenotype.status_id}" in sql
    assert sql.count("BOOL_OR") == len(STATUS_PHENOTYPES)


def test_status_that_follows_the_ecg_is_not_offered_as_a_predictor() -> None:
    definitions = {
        item["association_id"]: item for item in status_association_definitions()
    }
    anticoagulation = definitions["status_long_term_anticoagulant_use"]
    # Anticoagulation is started because atrial fibrillation was seen, so it
    # must never be presented as a predictor of that finding.
    assert anticoagulation["temporal_role"] == TEMPORAL_FOLLOWS
    assert anticoagulation["usable_as_predictor"] is False
    insulin = definitions["status_long_term_insulin_use"]
    assert insulin["usable_as_predictor"] is True


def test_status_definitions_carry_a_claim_ceiling() -> None:
    for definition in status_association_definitions():
        assert definition["reference_tier"].startswith("C_")
        assert definition["claim_ceiling"] == (
            "association_with_care_context_not_pathology"
        )
        assert definition["notes"].strip()


def test_unmapped_census_separates_disease_and_status_coverage() -> None:
    diagnoses = pd.DataFrame(
        {
            "icd_version": [10, 10, 10, 10],
            "icd_code": ["I214", "Z7901", "Z66", "Y929"],
            "n": [100, 80, 60, 40],
        }
    )
    census = unmapped_code_census(
        diagnoses.loc[diagnoses.index.repeat(diagnoses["n"])],
        [("I214", 10)],
    )
    by_code = census.set_index("icd_code")
    assert bool(by_code.loc["I214", "covered_by_disease_taxonomy"])
    assert not bool(by_code.loc["Z7901", "covered_by_disease_taxonomy"])
    assert bool(by_code.loc["Z7901", "covered_by_status_arm"])
    assert bool(by_code.loc["Z66", "covered_by_status_arm"])
    # Place-of-occurrence codes remain genuinely uncovered and are reported.
    assert not bool(by_code.loc["Y929", "covered"])
    summary = coverage_summary(census)
    assert summary["row_coverage_added_by_status_arm"] > 0.0
    assert summary["codes_uncovered"] >= 1
    assert summary["largest_uncovered_codes"]


def test_registry_table_is_reviewable() -> None:
    table = status_phenotype_registry()
    assert len(table) == len(STATUS_PHENOTYPES)
    assert not table["rationale_de"].eq("").any()
    assert table["temporal_role"].nunique() >= 2
