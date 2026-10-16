"""The release gate must be able to fail, and must not pass by relabelling.

Policy v1.0.0 was replaced because it could not be satisfied at any operating
point, not because it was inconvenient.  That is a dangerous kind of change to
make: the same edit that fixes an unsatisfiable test also describes what
someone would do to slip a weak result through.  These tests pin the
difference.

The invariants:

* every criterion names where its threshold comes from -- an unattributed
  threshold cannot support a release claim;
* a declared limitation never reports as passed, or it is just a disabled
  test with a nicer name;
* a declared limitation still gets measured and carried into the output;
* the criteria that carry the production claim still gate, and still fail
  when the numbers are bad.
"""

from __future__ import annotations

import copy

import pytest

from vcg_biomarker.core.delineation.benchmark_suite import (
    DELINEATION_DECLARED_LIMITATIONS,
    DELINEATION_RELEASE_POLICY_VERSION,
    DELINEATION_RELEASE_THRESHOLDS,
    PROVENANCE_COMPARATOR,
    PROVENANCE_EXTERNAL,
    PROVENANCE_LIMITATION,
    PROVENANCE_REGRESSION,
    PROVENANCE_STRUCTURAL,
    criterion_provenance,
    evaluate_delineation_release_gate,
)


ADMISSIBLE = {
    PROVENANCE_COMPARATOR,
    PROVENANCE_EXTERNAL,
    PROVENANCE_REGRESSION,
    PROVENANCE_STRUCTURAL,
    PROVENANCE_LIMITATION,
}


def test_the_policy_version_records_the_rewrite():
    assert DELINEATION_RELEASE_POLICY_VERSION == "delineation-release-v2.0.0"


def test_an_unlisted_criterion_is_reported_as_unattributed():
    """Silence must not read as approval."""

    provenance = criterion_provenance("ludb", "invented_metric")
    assert provenance["kind"] == "unattributed"
    assert "cannot support a release claim" in provenance["note"]


@pytest.mark.parametrize(
    "dataset, name",
    [
        ("ludb", "p_sensitivity"),
        ("ludb", "qrs_sensitivity"),
        ("ludb", "t_sensitivity"),
        ("ludb", "p_record_evaluable_rate"),
        ("ludb", "p_onset_p95_ms"),
        ("qtdb", "qrs_sensitivity"),
        ("nstdb", "noise_qrs_sensitivity"),
    ],
)
def test_every_live_criterion_names_its_origin(dataset, name):
    provenance = criterion_provenance(dataset, name)
    assert provenance["kind"] in ADMISSIBLE
    assert provenance["source"]


def test_a_declared_limitation_is_reported_as_such_not_as_comparator():
    provenance = criterion_provenance("ludb", "p_onset_p95_ms")
    assert provenance["kind"] == PROVENANCE_LIMITATION


def test_the_production_claim_still_rests_on_external_or_published_sources():
    """If every gating criterion were a regression floor, nothing is claimed."""

    kinds = {
        criterion_provenance("ludb", name)["kind"]
        for name in ("qrs_sensitivity", "t_sensitivity", "p_sensitivity")
    }
    assert kinds <= {PROVENANCE_COMPARATOR, PROVENANCE_EXTERNAL}


def _passing_results() -> dict:
    """A synthetic result set that meets every gating criterion."""

    def waves(sensitivity, ppv, mae, p95):
        return {
            wave: {
                "sensitivity": sensitivity,
                "positive_predictive_value": ppv,
                "boundary_errors_ms": {
                    location: {
                        "mean_absolute": mae,
                        "p95_absolute": p95,
                        "mae_ms": mae,
                        "p95_abs_ms": p95,
                    }
                    for location in ("onset", "offset")
                },
            }
            for wave in ("p", "qrs", "t")
        }

    return {
        "ludb": {
            "status": "complete",
            "records_requested": 200,
            "records_completed": 200,
            "records_quarantined": 0,
            "dataset_spec_version": "1.0.1",
            "run_is_record_capped": False,
            "groups": {
                "all": {
                    "record_evaluable_rate": {"p": 0.66, "qrs": 0.99, "t": 0.97},
                    "evaluable_waves": waves(0.99, 0.99, 5.0, 15.0),
                    "waves": waves(0.99, 0.99, 5.0, 15.0),
                }
            },
        },
        "qtdb": {
            "status": "complete",
            "records_requested": 105,
            "records_completed": 105,
            "records_quarantined": 0,
            "dataset_spec_version": "1.0.0",
            "run_is_record_capped": False,
            "waves": waves(0.90, 0.90, 5.0, 15.0),
        },
        "nstdb": {
            "status": "complete",
            "records_requested": 12,
            "records_quarantined": 0,
            "windows_completed": 24,
            "dataset_spec_version": "1.0.0",
            "run_is_record_capped": False,
            "clean": {"qrs_sensitivity_median": 0.99, "qrs_ppv_median": 0.99},
            "noise": {"qrs_sensitivity_median": 0.95, "qrs_ppv_median": 0.95},
        },
    }


def _gate(results):
    return evaluate_delineation_release_gate(
        results, max_records=None, max_seconds_per_record=120
    )


def test_a_declared_limitation_never_reports_as_passed():
    """Even when the measurement happens to meet the retired threshold."""

    gate = _gate(_passing_results())
    for assessment in gate["datasets"].values():
        for check in assessment["checks"]:
            if check.get("gating") is False:
                assert check["status"] == "declared_limitation"


def test_a_declared_limitation_is_carried_into_the_output():
    gate = _gate(_passing_results())
    criteria = {entry["criterion"] for entry in gate["declared_limitations"]}
    assert criteria == set(DELINEATION_DECLARED_LIMITATIONS)
    assert gate["affected_landmarks"] == ["p_onset"]
    for entry in gate["declared_limitations"]:
        assert entry["landmark"]
        assert entry["external_standard"]
        assert entry["reason"]
        assert entry["consequence"]


def test_a_declared_limitation_does_not_appear_among_failures():
    gate = _gate(_passing_results())
    for criterion in DELINEATION_DECLARED_LIMITATIONS:
        assert criterion not in gate["failures"]


def test_the_gate_still_fails_on_a_bad_production_number():
    """The point of the rewrite is a test that can still fail."""

    results = copy.deepcopy(_passing_results())
    groups = results["ludb"]["groups"]["all"]
    groups["evaluable_waves"]["qrs"]["sensitivity"] = 0.40
    gate = _gate(results)
    assert not gate["passed"]
    assert "ludb:qrs_sensitivity" in gate["failures"]


def test_the_gate_still_fails_when_a_cohort_collapses():
    results = copy.deepcopy(_passing_results())
    results["qtdb"]["waves"]["qrs"]["sensitivity"] = 0.10
    gate = _gate(results)
    assert not gate["passed"]
    assert "qtdb:qrs_sensitivity" in gate["failures"]


def test_a_missing_value_still_fails_closed():
    results = copy.deepcopy(_passing_results())
    results["ludb"]["groups"]["all"]["evaluable_waves"]["t"]["sensitivity"] = None
    gate = _gate(results)
    assert "ludb:t_sensitivity" in gate["failures"]


def test_the_reduced_lead_arm_uses_per_boundary_bounds():
    """A single per-wave bound would stop testing the easier boundary."""

    t_mae = DELINEATION_RELEASE_THRESHOLDS["qtdb"]["maximum_boundary_mae_ms"]["t"]
    assert isinstance(t_mae, dict)
    assert t_mae["offset"] < t_mae["onset"]


def test_the_ludb_production_criteria_were_not_loosened():
    """Everything carrying the production claim keeps its v1.0.0 value."""

    ludb = DELINEATION_RELEASE_THRESHOLDS["ludb"]
    assert ludb["minimum_sensitivity"] == {"p": 0.75, "qrs": 0.95, "t": 0.85}
    assert ludb["minimum_ppv"] == {"p": 0.75, "qrs": 0.95, "t": 0.85}
    assert ludb["maximum_boundary_mae_ms"] == {"p": 35.0, "qrs": 20.0, "t": 40.0}
    assert ludb["minimum_records_completed"] == 190
    assert ludb["minimum_record_evaluable_rate"]["qrs"] == 0.95
    assert ludb["minimum_record_evaluable_rate"]["t"] == 0.85
