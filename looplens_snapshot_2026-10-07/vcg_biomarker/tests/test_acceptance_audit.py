from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vcg_biomarker.mimic.acceptance import (
    AcceptanceExpectations,
    _check_group_vcg,
    _is_complete_anonymized_print_build,
    _check_mimic_lock,
    _check_phewas,
    refresh_pre_freeze_acceptance_audit,
    run_development_acceptance_audit,
)
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.group_vcg import group_vcg_implementation_hash
from vcg_biomarker.mimic.phenome_promotions import (
    refresh_phenome_promotions,
)
from vcg_biomarker.mimic.storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    content_hash,
)


def _config(tmp_path: Path) -> MimicProjectConfig:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "structured.csv",
        echo_study_list_path=tmp_path / "study.csv",
        echo_record_list_path=tmp_path / "record.csv",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    return config


def test_mimic_acceptance_lock_has_explicit_pre_freeze_state(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()

    pending = _check_mimic_lock(run_root, require_freeze=False)
    required = _check_mimic_lock(run_root, require_freeze=True)

    assert pending.status == "pending"
    assert required.status == "fail"
    assert "freeze" in required.detail.lower()


def test_mimic_acceptance_lock_rejects_consumed_test(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    freeze_payload = {"run_id": "development", "code_hash": "abc"}
    freeze_hash = content_hash(freeze_payload)
    atomic_write_json(
        run_root / "freeze_manifest.json",
        {
            "run_id": "development",
            "freeze_hash": freeze_hash,
            "payload": freeze_payload,
            "locked_test_status": "sealed_unread",
        },
    )
    lock = {
        "run_id": "development",
        "freeze_hash": freeze_hash,
        "evaluation_started_at": None,
        "consumed_at": None,
        "consumed_artifact_hash": None,
        "invalidated_at": None,
        "invalidation_reason": None,
    }
    atomic_write_json(run_root / "test_lock.json", lock)

    pristine = _check_mimic_lock(run_root, require_freeze=True)
    assert pristine.status == "pass"
    assert pristine.observed["locked_test_outcomes_read"] is False

    lock["evaluation_started_at"] = "2026-07-28T00:00:00+00:00"
    lock["consumed_at"] = "2026-07-28T00:01:00+00:00"
    atomic_write_json(run_root / "test_lock.json", lock)
    consumed = _check_mimic_lock(run_root, require_freeze=True)
    assert consumed.status == "fail"


def test_acceptance_audit_persists_failures_without_opening_test_sets(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = run_development_acceptance_audit(
        config,
        run_id="incomplete-development",
        ptb_run_id="missing-ptb",
        expectations=AcceptanceExpectations(
            ecg_count=1,
            subject_count=1,
            echo_study_count=1,
            development_feature_rows=1,
            label_definition_count=1,
            minimum_bootstrap_iterations=1,
            group_vcg_endpoints=(),
        ),
        require_freeze=False,
        strict=False,
        deep_feature_verification=False,
    )

    assert payload["status"] == "failed"
    assert payload["locked_test_outcomes_read"] is False
    assert payload["ptb_fold_10_outcomes_read"] is False
    # Sixteen: the laboratory-admission check (a group the
    # unit rule does not recognise disappears from every window aggregate)
    # and the echo-column check (a declared column whose source name does not
    # exist is indistinguishable from an unmeasured quantity).
    assert len(payload["checks"]) == 16
    assert any(item["status"] == "pending" for item in payload["checks"])

    audit_path = (
        config.runs_root
        / "incomplete-development"
        / "acceptance_pre_freeze_audit.json"
    )
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    assert persisted["audit_hash"] == payload["audit_hash"]
    assert (
        config.runs_root
        / "incomplete-development"
        / "acceptance_pre_freeze_audit.parquet"
    ).is_file()


def test_pre_freeze_refresh_rejects_tampered_audit(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_root = config.runs_root / "development"
    run_root.mkdir(parents=True)
    atomic_write_json(
        run_root / "acceptance_pre_freeze_audit.json",
        {
            "run_id": "development",
            "status": "ready_to_freeze",
            "locked_test_outcomes_read": False,
            "ptb_fold_10_outcomes_read": False,
            "audit_hash": "not-a-valid-hash",
        },
    )

    with pytest.raises(RuntimeError, match="hash verification"):
        refresh_pre_freeze_acceptance_audit(
            config,
            run_id="development",
        )


def _print_build(
    root: Path,
    *,
    include_case_ids: bool = False,
    include_case_appendix: bool = False,
    analysis_status: str = "complete_development",
    artifact_completeness: str = "complete",
    locked_test_status: str = "sealed_not_evaluated",
) -> Path:
    root.mkdir(parents=True)
    outputs = tuple(
        root / f"report{suffix}" for suffix in (".pdf", ".html", ".xlsx")
    )
    for path in outputs:
        atomic_write_text(path, "test")
    manifest = StageManifest.complete(
        stage="report-build:development:test",
        stage_version="test",
        input_hash="input",
        config_hash="config",
        output_paths=tuple(str(path.resolve()) for path in outputs),
    )
    manifest_path = root / "report_build.manifest.json"
    manifest.save(manifest_path)
    atomic_write_json(
        root / "report_spec.json",
        {
            "report_spec": {
                "include_case_ids": include_case_ids,
                "include_case_appendix": include_case_appendix,
            },
            "provenance": {
                "analysis_status": analysis_status,
                "artifact_completeness": artifact_completeness,
                "locked_test_status": locked_test_status,
            },
        },
    )
    return manifest_path


def test_print_report_acceptance_requires_anonymous_complete_development(
    tmp_path: Path,
) -> None:
    assert _is_complete_anonymized_print_build(
        _print_build(tmp_path / "valid")
    )
    assert not _is_complete_anonymized_print_build(
        _print_build(tmp_path / "ids", include_case_ids=True)
    )
    assert not _is_complete_anonymized_print_build(
        _print_build(tmp_path / "appendix", include_case_appendix=True)
    )
    assert not _is_complete_anonymized_print_build(
        _print_build(tmp_path / "preview", analysis_status="partial_preview")
    )
    assert not _is_complete_anonymized_print_build(
        _print_build(
            tmp_path / "consumed",
            locked_test_status="locked_test_consumed",
        )
    )
    corrupt = tmp_path / "corrupt" / "report_build.manifest.json"
    corrupt.parent.mkdir()
    atomic_write_text(corrupt, "{")
    assert not _is_complete_anonymized_print_build(corrupt)


def test_phewas_acceptance_requires_refreshed_candidate_roles(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_root = config.runs_root / "development"
    phenome_root = run_root / "phenome"
    phenome_root.mkdir(parents=True)
    replicated_path = phenome_root / "phenome_replicated_hits.parquet"
    atomic_write_parquet(
        pd.DataFrame(
            {
                "phenotype_id": ["P1", "P1"],
                "feature_id": [
                    "fiducial_confidence",
                    "spatial_qrs_t_angle_deg",
                ],
                "q_value": [0.0001, 0.01],
                "roc_auc": [0.91, 0.72],
            }
        ),
        replicated_path,
    )
    source_manifest = StageManifest.complete(
        stage="discover-phenome",
        stage_version="test",
        input_hash="source-input",
        config_hash="source-config",
        output_paths=(str(replicated_path.resolve()),),
        row_counts={"eligible_phenotypes": 1, "features": 2},
        notes=(
            "locked_test_scanned=false",
            f"feature_snapshot_hash={'b' * 64}",
        ),
    )
    source_manifest.save(phenome_root / "phenome.manifest.json")
    refresh_phenome_promotions(config, "development")

    observed, expected, _ = _check_phewas(run_root)
    assert observed["promotion_refresh_status"] == "verified"
    assert observed["invalid_promotion_feature_role_counts"] == {}
    assert (
        observed["excluded_methodological_top_candidates"] == 1
    )
    assert expected["invalid_promotion_feature_role_counts"] == {}

    candidates_path = (
        phenome_root / "phenome_promotion_candidates.parquet"
    )
    candidates = pd.read_parquet(candidates_path)
    candidates.at[0, "feature_ids"] = ["fiducial_confidence"]
    atomic_write_parquet(candidates, candidates_path)
    with pytest.raises(RuntimeError, match="PheWAS"):
        _check_phewas(run_root)


def test_group_vcg_acceptance_requires_complete_descriptive_artifacts(
    tmp_path: Path,
) -> None:
    endpoint_id = "endpoint"
    root = (
        tmp_path
        / "group_vcg"
        / f"endpoint={endpoint_id}"
        / "split=validation"
    )
    root.mkdir(parents=True)
    outputs = tuple(
        root / name
        for name in (
            "group_vcg_summary.parquet",
            "group_vcg_difference.parquet",
            "group_vcg_case_audit.parquet",
            "group_vcg_provenance.json",
        )
    )
    time_points = 2
    shared = {
        "endpoint_id": endpoint_id,
        "split": "validation",
        "feature_snapshot_hash": "snapshot",
        "artifact_completeness": "complete",
        "exploration_grade": "descriptive_development",
    }
    summary_rows = []
    for label_state in ("negative", "positive"):
        for sample_index in range(time_points):
            summary_rows.append(
                {
                    **shared,
                    "label_state": label_state,
                    "sample_index": sample_index,
                    "time_ms": float(sample_index),
                    "n_subjects": 10,
                    "median_x_mV": 0.1,
                    "median_y_mV": 0.2,
                    "median_z_mV": 0.3,
                }
            )
    difference_rows = [
        {
            **shared,
            "sample_index": sample_index,
            "time_ms": float(sample_index),
            "delta_x_mV": 0.1,
            "delta_y_mV": 0.2,
            "delta_z_mV": 0.3,
            "delta_magnitude_mV": 0.4,
        }
        for sample_index in range(time_points)
    ]
    audit_rows = [
        {
            "subject_id": index,
            "study_id": 100 + index,
            "endpoint_id": endpoint_id,
            "label_state": (
                "positive" if index <= 10 else "negative"
            ),
            "split": "validation",
            "status": "included",
        }
        for index in range(1, 21)
    ]
    atomic_write_parquet(pd.DataFrame(summary_rows), outputs[0])
    atomic_write_parquet(pd.DataFrame(difference_rows), outputs[1])
    atomic_write_parquet(pd.DataFrame(audit_rows), outputs[2])
    atomic_write_json(
        outputs[3],
        {
            "stage_version": "test",
            "implementation_hash": group_vcg_implementation_hash(),
            "input_hash": "input",
            "endpoint_id": endpoint_id,
            "split": "validation",
            "locked_test_read": False,
            "feature_snapshot_verification": {"valid": True},
            "clinical_event_evidence_provenance": {
                "status": "verified_exact_equivalence"
            },
            "class_counts": {"positive": 10, "negative": 10},
        },
    )
    manifest = StageManifest.complete(
        stage="group-vcg",
        stage_version="test",
        input_hash="input",
        config_hash="config",
        output_paths=tuple(str(path.resolve()) for path in outputs),
        row_counts={
            "included_positive": 10,
            "included_negative": 10,
            "time_points": time_points,
            "bootstrap_iterations": 500,
        },
        notes=("scope=development_only", "locked_test_read=False"),
    )
    manifest.save(root / "group_vcg.manifest.json")
    expectations = AcceptanceExpectations(
        group_vcg_endpoints=(endpoint_id,)
    )

    observed, _, _ = _check_group_vcg(tmp_path, expectations)
    assert observed["completed_endpoint_count"] == 1

    atomic_write_text(outputs[0], "not a parquet file")
    with pytest.raises(RuntimeError, match="incomplete"):
        _check_group_vcg(tmp_path, expectations)
    atomic_write_parquet(pd.DataFrame(summary_rows), outputs[0])

    outputs[0].unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        _check_group_vcg(tmp_path, expectations)


@pytest.mark.parametrize(
    "mode, comparisons, passes",
    [
        (
            "generated_by_current_implementation",
            {"status": "generated_current", "all_exact": True},
            True,
        ),
        (
            "legacy_artifact_exact_rebuild",
            {
                "all_exact": True,
                "locked_test_rows_materialized": False,
                "long_rule_results": [{"label_id": "a", "exact": True}],
            },
            True,
        ),
        (
            "legacy_artifact_exact_rebuild",
            {"all_exact": True, "locked_test_rows_materialized": False, "long_rule_results": []},
            False,
        ),
    ],
)
def test_label_attestation_check_accepts_both_attestation_modes(
    tmp_path: Path, monkeypatch, mode, comparisons, passes
) -> None:
    """Freshly built labels attest themselves without a rule-by-rule rebuild.

    The check expected the legacy shape (63 comparisons) for both modes, so a
    run with freshly built labels failed its acceptance audit at the very end.
    """
    from vcg_biomarker.mimic import acceptance
    from vcg_biomarker.mimic.acceptance import _check_label_attestation

    config = _config(tmp_path)
    payload = {
        "scope": "train_validation_only",
        "mode": mode,
        "comparisons": comparisons,
        "locked_test_read_for_attestation": False,
    }
    monkeypatch.setattr(
        acceptance, "verify_development_label_attestation", lambda config: payload
    )
    expectations = AcceptanceExpectations(
        ecg_count=1, subject_count=1, echo_study_count=1, development_feature_rows=1,
        label_definition_count=1, minimum_bootstrap_iterations=1, group_vcg_endpoints=(),
    )
    if passes:
        observed, expected, _ = _check_label_attestation(config, expectations)
        assert observed == expected
    else:
        with pytest.raises(RuntimeError, match="differs"):
            _check_label_attestation(config, expectations)


def test_artifacts_of_later_actions_are_pending_in_the_study_audit(tmp_path: Path) -> None:
    """Group VCG, print builds and the PTB revalidation come after the study.

    The study's own ``audit`` stage demanded them and therefore failed on
    every run; they are now pending, which still blocks the final audit with
    ``require_freeze``.
    """
    from vcg_biomarker.mimic.acceptance import (
        PendingArtifact,
        _check_group_vcg,
        _check_ptb_lock,
    )

    run_root = tmp_path / "run"
    run_root.mkdir()
    expectations = AcceptanceExpectations(
        ecg_count=1, subject_count=1, echo_study_count=1, development_feature_rows=1,
        label_definition_count=1, minimum_bootstrap_iterations=1,
        group_vcg_endpoints=("reduced_lvef", "echo_lae"),
    )
    with pytest.raises(PendingArtifact):
        _check_group_vcg(run_root, expectations)
    # One endpoint written, one missing: that is a failure, not a pending state.
    (run_root / "group_vcg" / "endpoint=reduced_lvef" / "split=validation").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="missing"):
        _check_group_vcg(run_root, expectations)

    config = _config(tmp_path)
    root = config.work_root / "ptb_revalidation" / "ptb-x"
    root.mkdir(parents=True)
    payload = {"code_hash": "stale"}
    freeze_hash = content_hash(payload)
    atomic_write_json(root / "ptb_freeze_manifest.json", {
        "payload": payload, "freeze_hash": freeze_hash, "ptb_fold_10_status": "sealed_unread",
    })
    atomic_write_json(root / "ptb_test_lock.json", {
        "freeze_hash": freeze_hash, "evaluation_started_at": None, "consumed_at": None,
        "invalidated_at": None,
    })
    atomic_write_json(root / "ptb_development_provenance.json", {
        "folds_read": list(range(1, 10)), "locked_fold_10_read": False,
    })
    with pytest.raises(PendingArtifact, match="earlier marker implementation"):
        _check_ptb_lock(config, "ptb-x")
    # A read fold 10 is a broken seal, never pending.
    atomic_write_json(root / "ptb_development_provenance.json", {
        "folds_read": list(range(1, 11)), "locked_fold_10_read": True,
    })
    with pytest.raises(RuntimeError, match="seal failed"):
        _check_ptb_lock(config, "ptb-x")


def test_expected_feature_rows_come_from_the_extraction_manifest(tmp_path: Path) -> None:
    """O1 (audit 2026-09-23): no fixed row constant; the manifest decides."""
    from vcg_biomarker.mimic.acceptance import _expected_development_feature_rows

    root = tmp_path / "features"
    root.mkdir()
    atomic_write_json(
        root / "features.manifest.json",
        {"row_counts": {"selected_ecgs": 640_110}},
    )
    derived, source = _expected_development_feature_rows(
        root, AcceptanceExpectations()
    )
    assert derived == 640_110
    assert source.startswith("extraction_manifest:")

    explicit, source = _expected_development_feature_rows(
        root, AcceptanceExpectations(development_feature_rows=12)
    )
    assert (explicit, source) == (12, "explicit_expectation")

    atomic_write_json(root / "features.manifest.json", {"row_counts": {}})
    with pytest.raises(RuntimeError, match="no selected ECG count"):
        _expected_development_feature_rows(root, AcceptanceExpectations())


def test_default_expectation_has_no_fixed_feature_row_constant() -> None:
    assert AcceptanceExpectations().development_feature_rows is None
