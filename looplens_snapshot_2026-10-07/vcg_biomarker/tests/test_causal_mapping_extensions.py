from dataclasses import replace
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.causal_treatment import (
    CausalProtocol,
    estimate_treatment_effects,
)
from vcg_biomarker.mimic.decision_research import TreatmentProfile, PROFILES
from vcg_biomarker.mimic.mapping_comparison import (
    beat_transform_features,
    compare_mapping_transforms,
    METHODS,
)
from vcg_biomarker.mimic.research_extensions import run_manifest, import_mapping_beats
from vcg_biomarker.tests.test_focus_localization import mapping


def causal_fixture():
    rng = np.random.default_rng(4)
    n = 1800
    x = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-0.6 * x)))
    frame = pd.DataFrame(
        dict(
            cohort_id="synthetic",
            patient_id=np.arange(n),
            split="train",
            treatment=a,
            treatment_time=0,
            covariate_available_time=-1,
            followup_complete=True,
            outcome_horizon_seconds=30 * 86400,
            x=x,
            benefit=2 * a + 3 * x + rng.normal(size=n),
            harm=-a + x + rng.normal(size=n),
        )
    )
    profile = TreatmentProfile(
        "pci",
        "treatment_benefit_research",
        "strategy_a",
        "strategy_b",
        "benefit",
        "harm",
        "eligibility",
        "30_days",
        "frozen-test-protocol",
    )
    return frame, CausalProtocol(profile, ("x",), outcome_kind="continuous")


def estimate(frame, protocol):
    return estimate_treatment_effects(
        frame,
        protocol,
        verified_fields=PROFILES["pci"],
        record_level_verified=True,
        confounding_plan_verified=True,
        identity_audit_verified=True,
    )


def test_aipw_recovers_effects_and_is_deterministic():
    frame, protocol = causal_fixture()
    result = estimate(frame, protocol)
    assert result["status"] == "research_estimate"
    assert abs(result["effects"]["benefit"]["ate"] - 2) < 0.15
    assert abs(result["effects"]["harm"]["ate"] + 1) < 0.15
    assert result == estimate(frame, protocol)
    assert not result["clinical_claim_allowed"]


@pytest.mark.parametrize(
    "column,value,reason",
    [
        ("split", "locked_test", "development_only"),
        ("covariate_available_time", 1, "future_covariate"),
        ("treatment_time", 3600, "time_zero"),
        ("followup_complete", False, "censoring"),
        ("treatment", 0, "insufficient_treatment"),
    ],
)
def test_causal_gates(column, value, reason):
    frame, protocol = causal_fixture()
    frame[column] = value
    result = estimate(frame, protocol)
    assert result["status"] == "non_estimable"
    assert reason in str(result["reasons"])


def test_causal_rejects_endpoint_as_confounder_and_patient_duplicates():
    frame, protocol = causal_fixture()
    assert (
        estimate(frame, replace(protocol, confounders=("benefit",)))["status"]
        == "non_estimable"
    )
    assert "one_index" in str(estimate(pd.concat([frame, frame]), protocol)["reasons"])


def comparison_fixture():
    frames = {}
    for method in METHODS:
        frame = mapping()
        frame["transform"] = method
        if method == "svd":
            frame["axis_convention"] = "svd_invariant_no_anatomical_axes"
        frames[method] = frame
    return frames, {method: ["initial_force"] for method in METHODS}


def test_paired_transform_comparison():
    frames, columns = comparison_fixture()
    frames["kors"].loc[0, "initial_force"] = np.nan
    result = compare_mapping_transforms(
        frames,
        columns,
        task="ventricular_focus",
        folds=2,
        bootstraps=20,
        identity_audit_verified=True,
    )
    assert result["paired_signal_count"] == 39
    assert result["availability_before_intersection"]["kors"]["import_rows"] == 40
    assert len(result["paired_differences"]) == 6
    assert all(
        d["patient_equal_accuracy_difference"] == 0
        for d in result["paired_differences"].values()
    )


def test_mapping_comparison_rejects_discordant_reference():
    frames, columns = comparison_fixture()
    frames["kors"].loc[0, "region"] = "rv_other"
    with pytest.raises(ValueError, match="reference_mismatch"):
        compare_mapping_transforms(
            frames,
            columns,
            task="ventricular_focus",
            bootstraps=20,
            identity_audit_verified=True,
        )


def test_transform_features_baseline_invariant():
    rng = np.random.default_rng(1)
    ecg = rng.normal(size=(150, 12))
    first = beat_transform_features(ecg, np.zeros(12), 1000, 20, 120)
    second = beat_transform_features(ecg + 7, np.full(12, 7), 1000, 20, 120)
    for method in METHODS:
        assert np.allclose(list(first[method].values()), list(second[method].values()))


def test_center_and_frozen_validation_comparisons():
    frames, columns = comparison_fixture()
    for frame in frames.values():
        frame["center_id"] = ["a"] * 20 + ["b"] * 20
    for mode in ("leave_center_out", "frozen_validation"):
        if mode == "frozen_validation":
            for frame in frames.values():
                frame.loc[20:, "split"] = "validation"
        result = compare_mapping_transforms(
            frames,
            columns,
            task="ventricular_focus",
            folds=2,
            evaluation=mode,
            bootstraps=20,
            identity_audit_verified=True,
        )
        assert result["results"]["kors"]["macro_f1"] == 1


def test_binary_causal_and_overlap_failure():
    frame, protocol = causal_fixture()
    rng = np.random.default_rng(42)
    frame["benefit"] = rng.binomial(1, 0.3 + 0.1 * frame.treatment)
    frame["harm"] = rng.binomial(1, 0.4 - 0.1 * frame.treatment)
    result = estimate(frame, replace(protocol, outcome_kind="binary"))
    assert result["status"] == "research_estimate"
    assert abs(result["effects"]["benefit"]["ate"] - 0.1) < 0.08
    frame["treatment"] = (frame.x > 0).astype(int)
    assert "overlap" in str(estimate(frame, protocol)["reasons"])


def test_causal_manifest_isolated_output(tmp_path):
    frame, protocol = causal_fixture()
    frame.to_parquet(tmp_path / "data.parquet")
    job = {
        "schema_version": "1.0.0",
        "kind": "causal",
        "data": "data.parquet",
        "protocol": asdict(protocol),
        "verification": {
            "verified_fields": PROFILES["pci"],
            "record_level_verified": True,
            "confounding_plan_verified": True,
            "identity_audit_verified": True,
        },
    }
    manifest = tmp_path / "job.json"
    manifest.write_text(json.dumps(job), encoding="utf-8")
    result = run_manifest(manifest, tmp_path / "new_output")
    assert result["status"] == "research_estimate"
    assert "sources_sha256" in result["provenance"]
    with pytest.raises(FileExistsError):
        run_manifest(manifest, tmp_path / "new_output")


def test_raw_mapping_import_and_content_identity(tmp_path):
    frame = mapping().iloc[:2].copy()
    frame["signal_file"] = ["first.npz", "second.npz"]
    frame["segment"] = "qrs"
    rng = np.random.default_rng(34)
    signal = dict(
        ecg_mv=rng.normal(size=(100, 12)),
        baseline_mv=np.zeros(12),
        sampling_rate=np.array(1000.0),
        onset=np.array(0),
        offset=np.array(80),
    )
    np.savez(tmp_path / "first.npz", **signal)
    np.savez_compressed(tmp_path / "second.npz", **signal)
    with pytest.raises(ValueError, match="duplicate_signal"):
        import_mapping_beats(frame, tmp_path, task="ventricular_focus")
    signal["ecg_mv"] = rng.normal(size=(100, 12))
    np.savez(tmp_path / "second.npz", **signal)
    frames, columns, provenance = import_mapping_beats(
        frame, tmp_path, task="ventricular_focus"
    )
    assert set(frames) == set(METHODS)
    assert len(provenance) == 2
    assert "early_10ms_integral_0_mv_s" in columns["kors"]
    frame["split"] = "locked_test"
    with pytest.raises(ValueError, match="locked"):
        import_mapping_beats(frame, tmp_path, task="ventricular_focus")


def test_deferred_branches_do_not_claim_readiness(tmp_path):
    from vcg_biomarker.mimic.research_extensions import research_readiness

    report = research_readiness(tmp_path)
    assert not report["blocks_primary_development_run"]
    assert (
        report["branches"]["mapping_localization"]["status"]
        == "deferred_missing_mapping_reference"
    )
    assert not any(item["automatic_execution"] for item in report["branches"].values())
    (tmp_path / "mapping_imports").mkdir()
    assert (
        research_readiness(tmp_path)["branches"]["mapping_localization"]["status"]
        == "requires_record_level_validation"
    )


def test_corrupt_parquet_records_failure_without_success(tmp_path):
    data = tmp_path / "bad.parquet"
    data.write_bytes(b"invalid parquet")
    manifest = tmp_path / "job.json"
    manifest.write_text(
        json.dumps({"schema_version": "1.0.0", "kind": "causal", "data": "bad.parquet"})
    )
    with pytest.raises(ValueError):
        run_manifest(manifest, tmp_path / "output")
    assert (tmp_path / "output" / "failure.json").is_file()
    assert not (tmp_path / "output" / "result.json").exists()


def test_locked_path_rejected_before_parquet_read(tmp_path, monkeypatch):
    locked = tmp_path / "locked_test"
    locked.mkdir()
    (locked / "data.parquet").write_bytes(b"not read")
    manifest = tmp_path / "job.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "kind": "causal",
                "data": "locked_test/data.parquet",
            }
        )
    )
    monkeypatch.setattr(
        pd, "read_parquet", lambda *a, **k: pytest.fail("locked test read")
    )
    with pytest.raises(ValueError, match="locked_test_input_forbidden"):
        run_manifest(manifest, tmp_path / "output")
