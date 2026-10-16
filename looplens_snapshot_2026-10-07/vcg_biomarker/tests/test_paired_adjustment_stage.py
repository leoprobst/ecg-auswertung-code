"""Real adjustment and persisted stage; raw pair loaders explicitly synthetic."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic import within_subject as stage


@pytest.mark.parametrize("clinical_context", [False, True])
@pytest.mark.parametrize("include_dose", [False, True])
@pytest.mark.parametrize("joint_references", [False, True])
def test_stage_persists_named_sensitivity_and_does_not_claim_full_adjustment(tmp_path, monkeypatch, clinical_context, include_dose, joint_references):
    config = SimpleNamespace(catalog_path=tmp_path/"catalog.duckdb",
        runs_root=tmp_path/"runs", master_root=tmp_path/"master",
        standardized_root=tmp_path/"standardized", evidence_root=tmp_path/"evidence",
        workers=1, duckdb_memory_limit_gb=2, to_dict=lambda: {"test_workspace": str(tmp_path)})
    config.master_root.mkdir()
    config.evidence_root.mkdir()
    n = 400
    rng = np.random.default_rng(147)
    z = rng.normal(size=(n, 3))
    x = rng.normal(size=n)
    y = .4*x+.5*z[:, 0]+rng.normal(size=n)
    ids = np.repeat(np.arange(100)+1, 4)
    pairs = pd.DataFrame(dict(subject_id=ids, first_study_id=2*np.arange(n), second_study_id=2*np.arange(n)+1,
        marker_first=1., marker_second=1.+y, hours_between=3.+.2*z[:, 0],
        signal_quality_first=.8+.02*z[:, 1], signal_quality_second=.8+.02*z[:, 2]))
    labs = pd.DataFrame(dict(subject_id=np.repeat(ids, 2), study_id=np.arange(2*n),
        value=np.column_stack((np.full(n, 4.), 4.+.3*x)).ravel(), itemid=50971, unit="mmol/l",
        analyte_id="potassium", lab_reference_status="eligible"))
    master = labs[["subject_id", "study_id"]].assign(split="train")
    if clinical_context:
        master["age"] = 30.+master.subject_id/2
        master["sex"] = np.where(master.subject_id.mod(2).eq(0), "M", "F")
        master["machine_heart_rate_bpm"] = rng.uniform(60., 100., len(master))
    master.to_parquet(config.master_root/"ecg_master_index.parquet", index=False)
    feature_path = tmp_path/"features.parquet"
    master.assign(marker=1., extraction_status="ok").to_parquet(feature_path, index=False)
    if joint_references:
        extras = []
        for name, item, unit, baseline, spread in (
            ("calcium_total", 50893, "mg/dl", 8.5, .3),
            ("calcium_ionized", 50808, "mmol/l", 1.2, .05),
            ("magnesium", 50960, "mg/dl", 2., .1),
            ("ph", 50820, "units", 7.4, .02),
        ):
            values = np.column_stack((np.full(n, baseline), baseline+spread*rng.normal(size=n))).ravel()
            extras.append(labs.assign(analyte_id=name, itemid=item, unit=unit, value=values))
        labs = pd.concat([labs, *extras], ignore_index=True)
    monkeypatch.setattr(stage, "build_dialysis_lab_pairs", lambda *a, **k: (pairs, labs))
    monkeypatch.setattr(stage, "_events_glob", lambda *_: "unused_by_stub")
    monkeypatch.setattr(stage, "dose_pairs_all_classes_query", lambda *a, **k: "SELECT ''::VARCHAR AS drug_class WHERE FALSE")
    if include_dose:
        dose_path = tmp_path / "synthetic_dose_pairs.parquet"
        doses = pairs.assign(drug_class="beta_blocker", delta_marker=y,
            dose_amount=np.exp(x), dose_itemid=1, dose_unit="mg", dose_aggregation_status="eligible")
        doses.loc[0, "dose_aggregation_status"] = "conflicting_order_identity"
        doses.to_parquet(dose_path, index=False)
        monkeypatch.setattr(stage, "dose_pairs_all_classes_query", lambda *a, **k: f"SELECT * FROM read_parquet('{dose_path.as_posix()}')")
    stage.run_within_subject_change(config, run_id="adjustment", feature_glob=feature_path.as_posix(),
                                   markers=("marker",), drug_classes=("beta_blocker",) if include_dose else ())
    root = config.runs_root/"adjustment"/"within_subject"
    results = pd.read_parquet(root/"within_subject_change.parquet")
    n_lab_models = 5 if joint_references else 1
    assert len(results) == n_lab_models + int(include_dose)
    row = results.loc[results.design.eq("dialysis_potassium")].iloc[0]
    assert row.adjusted_status == "estimable"
    assert row.adjusted_model == "timing_quality_sensitivity_only"
    assert row.adjusted_n_patients == 100
    assert np.isfinite(row.adjusted_global_q_value)
    assert row.adjusted_global_q_value >= row.adjusted_p_value
    summary = json.loads((root/"within_subject_summary.json").read_text())
    assert summary["adjusted_sensitivity"]["n_estimable"] == n_lab_models + int(include_dose)
    assert summary["adjusted_sensitivity"]["complete_time_varying_confounder_control"] is False
    assert summary["inference"]["clinical_claim_allowed"] is False
    joint = pd.read_parquet(root/"dialysis_joint_lab_context.parquet")
    assert len(joint) == n
    assert joint.context_delta_potassium.notna().all()
    assert bool(joint.context_delta_ph.isna().all()) == (not joint_references)
    assert summary["joint_lab_coverage"]["n_pairs_complete_core"] == (n if joint_references else 0)
    assert summary["joint_lab_coverage"]["analytes"]["potassium"]["n_pairs"] == n
    joint_estimable = joint_references and clinical_context
    assert row.joint_adjusted_status == ("estimable" if joint_estimable else "non_estimable")
    assert summary["joint_physiology_sensitivity"]["n_estimable"] == (5 if joint_estimable else 0)
    if joint_estimable:
        from statsmodels.stats.multitest import multipletests
        lab_results = results.loc[results.design.str.startswith("dialysis_")]
        assert lab_results.joint_adjusted_n_patients.eq(100).all()
        np.testing.assert_allclose(lab_results.joint_adjusted_global_q_value,
            multipletests(lab_results.joint_adjusted_p_value, method="fdr_bh")[1])
        assert lab_results.joint_adjusted_ci_low.le(lab_results.joint_adjusted_rank_slope).all()
        assert lab_results.joint_adjusted_ci_high.ge(lab_results.joint_adjusted_rank_slope).all()
    persisted_pairs = pd.read_parquet(root/"dialysis_paired_changes.parquet")
    assert len(persisted_pairs) >= len(pairs)
    assert persisted_pairs.baseline_context_status.eq("matched_development_index").all()
    assert bool(persisted_pairs.baseline_adjust_age.isna().all()) == (not clinical_context)
    assert "partial_clinical_sensitivity" in summary["baseline_context_policy"]
    assert row.clinical_adjusted_status == ("estimable" if clinical_context else "non_estimable")
    assert summary["baseline_clinical_sensitivity"]["n_estimable"] == int(clinical_context) * (n_lab_models + int(include_dose))
    if clinical_context:
        assert np.isfinite(row.clinical_adjusted_global_q_value)
        assert row.clinical_adjusted_n_patients == 100
    persisted_doses = pd.read_parquet(root/"dose_paired_changes.parquet")
    assert len(persisted_doses) == (n if include_dose else 0)
    assert summary["dose_pair_artifact"]["n_pairs"] == len(persisted_doses)
    manifest = json.loads((root/"within_subject.manifest.json").read_text())
    assert str((root/"dose_paired_changes.parquet").resolve()) in manifest["output_paths"]
    assert manifest["row_counts"]["dose_pairs"] == len(persisted_doses)
    assert any(record["path"] == str((root/"dose_paired_changes.parquet").resolve()) for record in manifest["output_integrity"])
    if include_dose:
        comparator = summary["dose_response"]["beta_blocker"]["heart_rate_comparator"]
        assert comparator["null_or_opposite_sign_proves_pairing_error"] is False
        assert "not_guaranteed" in comparator["expected_direction"]
        assert persisted_doses.dose_rank.isna().sum() == 1
        assert persisted_doses.dose_rank_status.iloc[0] == "invalid_or_mixed_dose_identity_unit_amount"
        assert persisted_doses.baseline_context_status.eq("matched_development_index").all()
        assert persisted_doses.first_study_id.tolist() == pairs.first_study_id.tolist()
        expected = stage.normalise_dose_within_drug(doses)
        pd.testing.assert_series_equal(persisted_doses.dose_rank, expected.dose_rank)
