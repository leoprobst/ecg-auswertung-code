"""Data-driven breadth across the event axes.

Pins the two properties that make a broad screen defensible: the frequency
floor is prespecified rather than tuned after seeing results, and an entity
below the floor is reported as underpowered rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.broad_event_screen import (
    AXIS_CCSR,
    AXIS_LABORATORY,
    AXIS_MICROBIOLOGY,
    AXIS_PROCEDURE,
    BROAD_AXES,
    DEFAULT_POLICIES,
    BroadAxisPolicy,
    assert_policies_prespecified,
    axis_coverage,
    ccsr_target_counts_sql,
    combined_coverage_report,
    enumerate_targets,
    laboratory_target_counts_sql,
    microbiology_target_counts_sql,
    policy_for,
    procedure_target_counts_sql,
    screen_family_size,
)


def _counts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["a", "b", "c", "d"],
            "entity_label": ["Alpha", "Beta", "Gamma", "Delta"],
            "n_subjects": [5000, 400, 120, 10],
            "n_observations": [50000, 4000, 1200, 40],
        }
    )


def test_every_axis_has_a_policy() -> None:
    for axis in BROAD_AXES:
        policy = policy_for(axis)
        assert policy.axis == axis
        assert policy.notes_de.strip()
        assert policy.claim_ceiling.strip()
    with pytest.raises(KeyError):
        policy_for("not_an_axis")


def test_frequency_floor_cannot_be_set_below_the_inference_minimum() -> None:
    with pytest.raises(ValueError, match="formal inference"):
        BroadAxisPolicy(AXIS_CCSR, minimum_subjects=10)
    with pytest.raises(ValueError, match="Unknown broad axis"):
        BroadAxisPolicy("invented", minimum_subjects=100)


def test_entities_below_the_floor_are_reported_not_dropped() -> None:
    policy = BroadAxisPolicy(AXIS_CCSR, minimum_subjects=200)
    targets = enumerate_targets(
        _counts(),
        policy,
        entity_column="entity_id",
        label_column="entity_label",
    )
    # All four entities survive into the table; only two are screened.
    assert len(targets) == 4
    assert int(targets["eligible_for_screen"].sum()) == 2
    excluded = targets.loc[~targets["eligible_for_screen"].astype(bool)]
    assert (
        excluded["ineligibility_reason"]
        .str.contains("underpowered_here_not_uninformative")
        .all()
    )


def test_observation_floor_is_applied_independently() -> None:
    policy = BroadAxisPolicy(
        AXIS_LABORATORY,
        minimum_subjects=100,
        minimum_observations=5000,
        target_kind="continuous",
    )
    targets = enumerate_targets(
        _counts(),
        policy,
        entity_column="entity_id",
    ).set_index("entity_id")
    # "c" clears the subject floor but not the observation floor.
    assert bool(targets.loc["a", "eligible_for_screen"])
    assert not bool(targets.loc["c", "eligible_for_screen"])
    assert "observation_floor" in targets.loc["c", "ineligibility_reason"]


def test_coverage_reports_what_the_broad_screen_adds() -> None:
    targets = enumerate_targets(
        _counts(),
        BroadAxisPolicy(AXIS_LABORATORY, minimum_subjects=100, target_kind="continuous"),
        entity_column="entity_id",
    )
    coverage = axis_coverage(targets, curated_entity_ids=["a"])
    assert coverage["entities_total"] == 4
    assert coverage["entities_screened"] == 3
    assert coverage["entities_curated_before"] == 1
    assert coverage["entities_added_by_broad_screen"] == 2
    assert 0.0 < coverage["observation_coverage"] <= 1.0
    assert "unterpowert" in coverage["interpretation_de"]


def test_family_size_is_reported_rather_than_hidden() -> None:
    targets = enumerate_targets(
        _counts(),
        BroadAxisPolicy(AXIS_CCSR, minimum_subjects=100),
        entity_column="entity_id",
    )
    family = screen_family_size(targets, n_markers=1000)
    assert family["eligible_targets"] == 3
    assert family["tests_in_family"] == 3000
    assert "FDR" in family["multiplicity_note_de"]


def test_prespecified_floors_are_enforced() -> None:
    expected = {
        policy.axis: policy.minimum_subjects for policy in DEFAULT_POLICIES
    }
    assert_policies_prespecified(DEFAULT_POLICIES, expected=expected)
    moved = dict(expected)
    moved[AXIS_CCSR] = expected[AXIS_CCSR] + 50
    with pytest.raises(AssertionError, match="preregistered"):
        assert_policies_prespecified(DEFAULT_POLICIES, expected=moved)


def test_laboratory_axis_is_quantitative_and_carries_a_higher_ceiling() -> None:
    policy = policy_for(AXIS_LABORATORY)
    # A laboratory value is a measurement, not an administrative code, so it
    # is not capped at the same claim level as a diagnosis code.
    assert policy.target_kind == "continuous"
    assert policy.reference_tier.startswith("A_")
    assert policy.claim_ceiling != "association_only_not_accuracy"
    for axis in (AXIS_CCSR, AXIS_PROCEDURE, AXIS_MICROBIOLOGY):
        assert policy_for(axis).reference_tier.startswith("C_")


def test_every_count_query_compiles_and_returns_the_expected_shape() -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE dx AS SELECT * FROM (VALUES "
        "(1, 'I214'), (2, 'I214'), (3, 'E119')) AS t(subject_id, icd_code)"
    )
    connection.execute(
        "CREATE TABLE ccsr AS SELECT * FROM (VALUES "
        "('I214', 'CIR009', 'Acute myocardial infarction'), "
        "('E119', 'END002', 'Diabetes')) "
        "AS t(icd_code, ccsr_category, ccsr_description)"
    )
    frame = connection.execute(ccsr_target_counts_sql("ccsr", "dx")).fetchdf()
    assert set(frame.columns) == {
        "entity_id",
        "entity_label",
        "n_subjects",
        "n_observations",
    }
    assert int(frame.loc[frame["entity_id"] == "CIR009", "n_subjects"].iloc[0]) == 2

    connection.execute(
        "CREATE TABLE labs AS SELECT * FROM (VALUES "
        "(1, 50811, 12.0), (2, 50811, 13.0), (3, 50811, NULL)) "
        "AS t(subject_id, itemid, valuenum)"
    )
    connection.execute(
        "CREATE TABLE d_lab AS SELECT * FROM (VALUES "
        "(50811, 'Hemoglobin')) AS t(itemid, label)"
    )
    frame = connection.execute(
        laboratory_target_counts_sql("labs", "d_lab")
    ).fetchdf()
    assert frame.iloc[0]["entity_label"] == "Hemoglobin"
    # Only numeric values count towards the observation floor.
    assert int(frame.iloc[0]["n_observations"]) == 2

    connection.execute(
        "CREATE TABLE proc AS SELECT * FROM (VALUES "
        "(1, '0270'), (2, '0270')) AS t(subject_id, icd_code)"
    )
    assert not connection.execute(
        procedure_target_counts_sql("proc")
    ).fetchdf().empty

    connection.execute(
        "CREATE TABLE micro AS SELECT * FROM (VALUES "
        "(1, 'STAPH AUREUS'), (2, 'STAPH AUREUS'), (3, '')) "
        "AS t(subject_id, organism)"
    )
    frame = connection.execute(
        microbiology_target_counts_sql("micro")
    ).fetchdf()
    # Blank organism strings are not an organism.
    assert len(frame) == 1
    assert int(frame.iloc[0]["n_subjects"]) == 2


def test_combined_report_covers_every_axis() -> None:
    coverages = []
    for axis in BROAD_AXES:
        targets = enumerate_targets(
            _counts(),
            policy_for(axis),
            entity_column="entity_id",
        )
        coverages.append(axis_coverage(targets))
    report = combined_coverage_report(coverages)
    assert set(report["axis"]) == set(BROAD_AXES)
    assert (report["entities_total"] == 4).all()


def test_empty_input_produces_an_empty_but_typed_table() -> None:
    targets = enumerate_targets(
        pd.DataFrame(),
        policy_for(AXIS_CCSR),
        entity_column="entity_id",
    )
    assert targets.empty
    assert "eligible_for_screen" in targets.columns
    assert axis_coverage(targets)["entities_total"] == 0
